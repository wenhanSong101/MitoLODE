"""
MGMI external validation -- independent evidence from three public databases.

For each candidate perturbation target produced by mgmi_inference.py, this
module queries three external sources that do NOT overlap with any of
MitoKG's construction sources (STRING / Reactome / DrugBank / CORUM /
MitoCarta), avoiding circular validation:

    1. OpenTargets Platform      - PD genetic association score
    2. NCBI PubMed eutils        - gene + "Parkinson" co-occurrence count
    3. Ensembl REST API          - gene biotype confirmation

Composite score (weights):
    external_score = 0.50 * OT_score + 0.30 * PubMed_score + 0.20 * Ensembl_score

Confidence tiers:
    HIGH    : external_score >= 0.45 AND OT_score > 0
    MEDIUM  : external_score >= 0.28
    LOW     : otherwise

Input:
    mitokg_llm_raw_v2.json (produced by mgmi_inference.py)

Output:
    mgmi_external_validation.csv          full record (all API fields)
    mgmi_external_validation_summary.csv  compact comparison table
    fig_external_validation.png            4-panel visualization

Usage:
    python mgmi_external_validation.py \\
        --mgmi-json /path/to/mitokg_llm_raw_v2.json \\
        --out-dir ./output_mgmi_validation
"""

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests


PD_EFO_ID = "MONDO_0005180"
PD_EFO_ALIASES = {"MONDO_0005180", "EFO_0002508", "Orphanet_411602",
                    "MONDO_0008199", "EFO_0022609"}
TIMEOUT_SEC = 15
MAX_RETRIES = 2
RETRY_DELAY = 3

W_OPENTARGETS = 0.50
W_PUBMED = 0.30
W_ENSEMBL = 0.20

SCORE_HIGH = 0.45
SCORE_MEDIUM = 0.28

PUBMED_MAX_COUNT = 50
PUBMED_PD_TERMS = "Parkinson"
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_API_KEY = ""

DEFAULT_TOP_N = 15

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OT_REST_URL = "https://api.platform.opentargets.org/api/v4/association/filter"

ENSEMBL_URL = "https://rest.ensembl.org/lookup/symbol/homo_sapiens"

ENSEMBL_ID_CACHE = {
    "UQCC3": "ENSG00000204922",
    "SDHAF3": "ENSG00000196636",
    "SDHAF4": "ENSG00000154079",
    "SDHD": "ENSG00000204370",
    "SDHA": "ENSG00000073578",
    "SDHB": "ENSG00000117118",
    "NDUFA7": "ENSG00000267855",
    "NDUFB7": "ENSG00000099795",
    "NDUFB8": "ENSG00000166136",
    "MRPS10": "ENSG00000048544",
    "PRDX2": "ENSG00000167815",
    "CYCS": "ENSG00000172115",
    "UQCC2": "ENSG00000100116",
    "COX6C": "ENSG00000164919",
    "NDUFA9": "ENSG00000103061",
}


def safe_request(method, url, **kwargs):
    """HTTP request with retries. Returns {ok, status, data, error}."""
    kwargs.setdefault("timeout", TIMEOUT_SEC)
    for attempt in range(MAX_RETRIES + 1):
        try:
            if method.upper() == "POST":
                resp = requests.post(url, **kwargs)
            else:
                resp = requests.get(url, **kwargs)
            if resp.status_code == 200:
                try:
                    return {"ok": True, "status": 200, "data": resp.json(), "error": ""}
                except Exception:
                    return {"ok": True, "status": 200, "data": resp.text, "error": ""}
            try:
                err_body = resp.json()
                err_detail = str(err_body)[:400]
            except Exception:
                err_body = None
                err_detail = resp.text[:400]
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            return {"ok": False, "status": resp.status_code,
                     "data": err_body,
                     "error": f"HTTP {resp.status_code}: {err_detail}"}
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            return {"ok": False, "status": 0, "data": None, "error": "Timeout"}
        except Exception as e:
            return {"ok": False, "status": 0, "data": None, "error": str(e)}
    return {"ok": False, "status": 0, "data": None, "error": "Max retries exceeded"}


def _symbol_to_ensembl(gene_symbol):
    cached = ENSEMBL_ID_CACHE.get(gene_symbol.upper()) or ENSEMBL_ID_CACHE.get(gene_symbol)
    if cached:
        return cached, {"ok": True, "status": 200,
                         "data": {"source": "local_cache", "ensembl_id": cached}, "error": ""}

    query = """
    query SearchTarget($symbol: String!) {
      search(queryString: $symbol, entityNames: ["target"], page: {index: 0, size: 5}) {
        hits {
          id
          name
          entity
          object {
            ... on Target {
              id
              approvedSymbol
            }
          }
        }
      }
    }
    """
    result = safe_request(
        "POST", OT_GRAPHQL_URL,
        json={"query": query, "variables": {"symbol": gene_symbol}},
        headers={"Content-Type": "application/json",
                   "Accept": "application/json"}
    )
    if not result["ok"]:
        return "", result
    try:
        data = result["data"]
        if "data" in data:
            data = data["data"]
        hits = data.get("search", {}).get("hits", [])
        for h in hits:
            obj = h.get("object") or {}
            if obj.get("approvedSymbol", "").upper() == gene_symbol.upper():
                return obj["id"], result
        for h in hits:
            if h.get("entity") == "target":
                obj = h.get("object") or {}
                if obj.get("id"):
                    return obj["id"], result
    except Exception:
        pass
    return "", result


def query_opentargets(gene_symbol, ensembl_id_override=""):
    """query PD association score for gene in OpenTargets."""
    out = {
        "ot_ensembl_id": "", "ot_overall_score": 0.0,
        "ot_genetic_score": 0.0, "ot_literature_score": 0.0,
        "ot_known_drug_score": 0.0, "ot_n_evidence": 0,
        "ot_raw_search": "", "ot_raw_assoc": "", "ot_error": ""
    }

    if ensembl_id_override and ensembl_id_override.startswith("ENSG"):
        ensembl_id = ensembl_id_override
        search_resp = {"ok": True, "status": 200,
                         "data": {"source": "ensembl_api_override",
                                    "ensembl_id": ensembl_id}, "error": ""}
        out["ot_raw_search"] = json.dumps(search_resp["data"])
    else:
        ensembl_id, search_resp = _symbol_to_ensembl(gene_symbol)
        out["ot_raw_search"] = json.dumps(
            search_resp["data"] if search_resp["ok"] else {"error": search_resp["error"]},
            ensure_ascii=False
        )[:800]

    if not ensembl_id:
        out["ot_error"] = f"Symbol '{gene_symbol}' not found in OpenTargets"
        return out

    out["ot_ensembl_id"] = ensembl_id

    query = """
    query GeneAllDiseases($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        associatedDiseases(
          page: { index: 0, size: 50 }
        ) {
          count
          rows {
            disease { id name }
            score
            datatypeScores { id score }
          }
        }
      }
    }
    """
    result = safe_request(
        "POST", OT_GRAPHQL_URL,
        json={"query": query, "variables": {"ensemblId": ensembl_id}},
        headers={"Content-Type": "application/json",
                   "Accept": "application/json"}
    )
    out["ot_raw_assoc"] = json.dumps(
        result["data"] if result["ok"] else {"error": result["error"]},
        ensure_ascii=False
    )[:1200]

    if not result["ok"]:
        rest_result = {"ok": False, "data": None, "error": "not tried"}
        for rest_url_candidate in [
            f"https://api.platform.opentargets.org/api/v4/target/{ensembl_id}/associations",
            OT_REST_URL,
        ]:
            _params = ({"disease": PD_EFO_ID, "size": 5}
                         if "target/" in rest_url_candidate
                         else {"targetId": ensembl_id, "diseaseId": PD_EFO_ID, "size": 5})
            rest_result = safe_request(
                "GET", rest_url_candidate,
                params=_params,
                headers={"Accept": "application/json"}
            )
            if rest_result["ok"]:
                break
        out["ot_raw_assoc"] += " | REST_fallback: " + json.dumps(
            rest_result["data"] if rest_result["ok"] else {"error": rest_result["error"]},
            ensure_ascii=False
        )[:400]
        if rest_result["ok"]:
            try:
                rest_data = rest_result["data"]
                hits = rest_data.get("data") or []
                if hits:
                    h = hits[0]
                    sc = (h.get("score") or
                            (h.get("association_score") or {}).get("overall") or
                            h.get("overall_score") or 0)
                    out["ot_overall_score"] = float(sc or 0)
                    out["ot_n_evidence"] = int(rest_data.get("total", 0) or 0)
                    out["ot_error"] = ""
                    return out
            except Exception:
                pass
        out["ot_error"] = result["error"]
        return out

    try:
        data = result["data"]
        if "data" in data:
            data = data["data"]
        target_data = data.get("target") or {}
        assoc = target_data.get("associatedDiseases") or {}
        rows = assoc.get("rows") or []
        out["ot_n_evidence"] = int(assoc.get("count", 0) or 0)

        pd_row = None
        for r in rows:
            dis_id = (r.get("disease") or {}).get("id", "")
            dis_name = (r.get("disease") or {}).get("name", "").lower()
            if (dis_id in PD_EFO_ALIASES or
                    any(alias.lower() in dis_id.lower() for alias in PD_EFO_ALIASES) or
                    "parkinson" in dis_name):
                pd_row = r
                break

        if pd_row is None:
            out["ot_error"] = f"PD not in top-{assoc.get('count', 0)} associated diseases (score near 0)"
            out["ot_overall_score"] = 0.0
            return out

        out["ot_overall_score"] = float(pd_row.get("score", 0) or 0)
        for ds in (pd_row.get("datatypeScores") or []):
            cid = (ds.get("id") or ds.get("componentId") or "").lower()
            sc = float(ds.get("score", 0) or 0)
            if "genetic" in cid:
                out["ot_genetic_score"] = sc
            elif "literature" in cid or "text" in cid:
                out["ot_literature_score"] = sc
            elif "drug" in cid or "known" in cid:
                out["ot_known_drug_score"] = sc
    except Exception as e:
        out["ot_error"] = f"Parse error: {e}"

    return out


def query_pubmed_cooccurrence(gene_symbol):
    """gene + Parkinson PubMed literature co-occurrence count, log-normalized."""
    out = {"pm_count": 0, "pm_score": 0.0,
             "pm_query": "", "pm_raw": "", "pm_error": ""}

    query_term = f'{gene_symbol}[Gene Name] AND {PUBMED_PD_TERMS}[Title/Abstract]'
    out["pm_query"] = query_term

    params = {"db": "pubmed", "term": query_term,
                "retmode": "json", "retmax": "0"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    result = safe_request("GET", NCBI_EUTILS_BASE, params=params)
    out["pm_raw"] = json.dumps(
        result["data"] if result["ok"] else {"error": result["error"]},
        ensure_ascii=False
    )[:400]

    if not result["ok"]:
        params2 = {"db": "pubmed",
                     "term": f'{gene_symbol} AND {PUBMED_PD_TERMS}[Title/Abstract]',
                     "retmode": "json", "retmax": "0"}
        if NCBI_API_KEY:
            params2["api_key"] = NCBI_API_KEY
        result2 = safe_request("GET", NCBI_EUTILS_BASE, params=params2)
        out["pm_raw"] += " | fallback:" + json.dumps(
            result2["data"] if result2["ok"] else {"error": result2["error"]},
            ensure_ascii=False
        )[:200]
        result = result2 if result2["ok"] else result
        if not result["ok"]:
            out["pm_error"] = result["error"]
            return out

    try:
        count = int(result["data"].get("esearchresult", {}).get("count", 0) or 0)
        out["pm_count"] = count
        out["pm_score"] = round(
            min(1.0, math.log(count + 1) / math.log(PUBMED_MAX_COUNT + 1)), 4
        )
    except Exception as e:
        out["pm_error"] = f"Parse error: {e}"

    return out


def query_ensembl(gene_symbol):
    """Ensembl REST: gene existence and biotype."""
    out = {
        "ens_id": "", "ens_biotype": "", "ens_description": "",
        "ens_chromosome": "", "ens_is_protein_coding": False,
        "ens_score": 0.0, "ens_raw": "", "ens_error": ""
    }
    url = f"{ENSEMBL_URL}/{gene_symbol}"
    result = safe_request(
        "GET", url,
        headers={"Content-Type": "application/json"},
        params={"expand": 0}
    )
    out["ens_raw"] = json.dumps(
        result["data"] if result["ok"] else {"error": result["error"]},
        ensure_ascii=False
    )[:600]

    if not result["ok"]:
        out["ens_error"] = result["error"]
        return out

    try:
        d = result["data"]
        out["ens_id"] = d.get("id", "")
        out["ens_biotype"] = d.get("biotype", "")
        out["ens_description"] = (d.get("description") or "")[:200]
        out["ens_chromosome"] = str(d.get("seq_region_name", ""))
        out["ens_is_protein_coding"] = (out["ens_biotype"] == "protein_coding")
        out["ens_score"] = 1.0 if out["ens_is_protein_coding"] else 0.3
    except Exception as e:
        out["ens_error"] = f"Parse error: {e}"

    return out


def compute_external_score(ot, up, ens):
    """weighted composite score and confidence tier."""
    ot_score = float(ot.get("ot_overall_score") or 0)
    up_score = float(up.get("pm_score") or 0)
    ens_score = float(ens.get("ens_score") or 0)
    ot_failed = bool(ot.get("ot_error"))

    ot_contrib = ot_score * W_OPENTARGETS
    up_contrib = up_score * W_PUBMED
    ens_contrib = ens_score * W_ENSEMBL

    external_score = round(ot_contrib + up_contrib + ens_contrib, 4)

    if external_score >= SCORE_HIGH and ot_score > 0:
        ext_confidence = "HIGH"
    elif external_score >= SCORE_MEDIUM:
        ext_confidence = "MEDIUM"
    else:
        ext_confidence = "LOW"

    ot_status_note = (
        f"OT_FAILED({ot.get('ot_error', '')})" if ot_failed
        else f"OT_OK(score={ot_score:.4f})"
    )
    scoring_log = (
        f"[OT] {ot_status_note}"
        f" | overall_score={ot_score:.4f} x weight={W_OPENTARGETS}"
        f" -> contrib={ot_contrib:.4f}"
        f" | n_evidence={ot.get('ot_n_evidence', 0)}"
        f" | genetic={ot.get('ot_genetic_score', 0):.4f}"
        f" | literature={ot.get('ot_literature_score', 0):.4f}"
        " || "
        f"[PubMed] pm_count={up.get('pm_count', 0)}"
        f" | pm_score={up_score:.4f}"
        f" | query='{up.get('pm_query', '')[:60]}'"
        f" | score={up_score:.4f} x weight={W_PUBMED}"
        f" -> contrib={up_contrib:.4f}"
        f" | error='{up.get('pm_error', '')}'"
        " || "
        f"[Ensembl] biotype={ens.get('ens_biotype', '')}"
        f" | protein_coding={ens.get('ens_is_protein_coding', False)}"
        f" | score={ens_score:.1f} x weight={W_ENSEMBL}"
        f" -> contrib={ens_contrib:.4f}"
        f" | error='{ens.get('ens_error', '')}'"
        " || "
        f"[TOTAL] {ot_contrib:.4f}+{up_contrib:.4f}+{ens_contrib:.4f}"
        f"={external_score:.4f}"
        f" | HIGH_cond: score>={SCORE_HIGH} AND ot>0 -> {external_score >= SCORE_HIGH and ot_score > 0}"
        f" | MEDIUM_cond: score>={SCORE_MEDIUM} -> {external_score >= SCORE_MEDIUM}"
        f" -> ext_confidence={ext_confidence}"
    )

    return {
        "external_score": external_score,
        "ext_confidence": ext_confidence,
        "ot_failed": ot_failed,
        "ot_contrib": round(ot_contrib, 4),
        "up_contrib": round(up_contrib, 4),
        "ens_contrib": round(ens_contrib, 4),
        "scoring_log": scoring_log,
    }


def run_external_validation(candidates, out_dir, top_n):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total = len(candidates[:top_n])

    print(f"\n{'=' * 65}")
    print(f"MGMI external validation  |  TOP_N={top_n}  |  "
            f"time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sources: OpenTargets({PD_EFO_ID}) + PubMed + Ensembl")
    print(f"Weights: OT={W_OPENTARGETS} / PubMed={W_PUBMED} / Ensembl={W_ENSEMBL}")
    print(f"Thresholds: HIGH>={SCORE_HIGH} / MEDIUM>={SCORE_MEDIUM} / else LOW")
    print(f"{'=' * 65}")

    for i, cand in enumerate(candidates[:top_n], 1):
        gene = cand["gene"]
        print(f"\n[{i:02d}/{total}] {gene} "
                f"({cand['timepoint']}, {cand['direction']}, {cand['pathway']})")

        ts_start = datetime.now().isoformat()

        print(f"  -> Ensembl...    ", end="", flush=True)
        ens = query_ensembl(gene)
        ens_status = (f"id={ens.get('ens_id', '')} biotype={ens['ens_biotype']}"
                        if not ens["ens_error"] else f"ERR:{ens['ens_error']}")
        print(f" {ens_status}")

        print(f"  -> OpenTargets...", end="", flush=True)
        ot = query_opentargets(gene, ensembl_id_override=ens.get("ens_id", ""))
        ot_status = (f"score={ot['ot_overall_score']:.4f}"
                      if not ot["ot_error"] else f"ERR:{ot['ot_error']}")
        print(f" {ot_status}")

        print(f"  -> PubMed...     ", end="", flush=True)
        up = query_pubmed_cooccurrence(gene)
        up_status = (f"count={up['pm_count']} score={up['pm_score']:.3f}"
                      if not up["pm_error"] else f"ERR:{up['pm_error']}")
        print(f" {up_status}")

        score_result = compute_external_score(ot, up, ens)
        ts_end = datetime.now().isoformat()

        old_conf = cand.get("confidence", "HIGH")
        new_conf = score_result["ext_confidence"]
        conf_order = ["LOW", "MEDIUM", "HIGH"]
        if old_conf not in conf_order:
            old_conf = "HIGH"
        conf_change = (
            "UPGRADED" if conf_order.index(new_conf) > conf_order.index(old_conf)
            else "DOWNGRADED" if conf_order.index(new_conf) < conf_order.index(old_conf)
            else "UNCHANGED"
        )

        print(f"  external_score={score_result['external_score']:.4f}  "
                f"ext_confidence={new_conf}  "
                f"(MGMI_original:{old_conf} -> {conf_change})")

        record = {
            "input_gene": gene,
            "input_timepoint": cand["timepoint"],
            "input_direction": cand["direction"],
            "input_pathway": cand["pathway"],
            "input_delta_r": cand.get("delta_r", ""),
            "input_r_after": cand.get("r_after", ""),
            "input_kg_weight": cand.get("kg_weight", ""),
            "input_mgmi_confidence": old_conf,
            "input_reasoning_truncated": str(cand.get("reasoning", ""))[:300],

            "ot_ensembl_id": ot["ot_ensembl_id"],
            "ot_overall_score": ot["ot_overall_score"],
            "ot_genetic_score": ot["ot_genetic_score"],
            "ot_literature_score": ot["ot_literature_score"],
            "ot_known_drug_score": ot["ot_known_drug_score"],
            "ot_n_evidence": ot["ot_n_evidence"],
            "ot_error": ot["ot_error"],
            "ot_raw_search": ot["ot_raw_search"],
            "ot_raw_assoc": ot["ot_raw_assoc"],

            "pm_query": up["pm_query"],
            "pm_count": up["pm_count"],
            "pm_score": up["pm_score"],
            "up_score": up["pm_score"],
            "pm_error": up["pm_error"],
            "pm_raw": up["pm_raw"],

            "ens_id": ens["ens_id"],
            "ens_biotype": ens["ens_biotype"],
            "ens_description": ens["ens_description"],
            "ens_chromosome": ens["ens_chromosome"],
            "ens_is_protein_coding": ens["ens_is_protein_coding"],
            "ens_score": ens["ens_score"],
            "ens_error": ens["ens_error"],
            "ens_raw": ens["ens_raw"],

            "ot_contrib": score_result["ot_contrib"],
            "up_contrib": score_result["up_contrib"],
            "ens_contrib": score_result["ens_contrib"],
            "ot_failed": score_result["ot_failed"],
            "external_score": score_result["external_score"],
            "scoring_log": score_result["scoring_log"],

            "ext_confidence": new_conf,
            "conf_change": conf_change,

            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "top_n_config": top_n,
            "pd_efo_id": PD_EFO_ID,
            "weight_ot": W_OPENTARGETS,
            "weight_pubmed": W_PUBMED,
            "weight_ensembl": W_ENSEMBL,
            "threshold_high": SCORE_HIGH,
            "threshold_medium": SCORE_MEDIUM,
        }
        records.append(record)
        time.sleep(1.0)

    df = pd.DataFrame(records)

    csv_path = out_dir / "mgmi_external_validation.csv"
    df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
    print(f"\nMain table saved: {csv_path}  ({len(df)} rows x {len(df.columns)} cols)")

    summary_cols = [
        "input_gene", "input_timepoint", "input_direction", "input_pathway",
        "input_delta_r", "input_kg_weight", "input_mgmi_confidence",
        "ot_overall_score", "ot_n_evidence", "ot_ensembl_id", "ot_failed",
        "pm_count", "pm_score", "pm_query",
        "ens_biotype", "ens_id",
        "ot_contrib", "up_contrib", "ens_contrib",
        "external_score", "ext_confidence", "conf_change",
    ]
    df_summary = df[[c for c in summary_cols if c in df.columns]].copy()
    summary_path = out_dir / "mgmi_external_validation_summary.csv"
    df_summary.to_csv(str(summary_path), index=False, encoding="utf-8-sig")
    print(f"Summary table saved: {summary_path}")

    return df


def plot_validation_results(df, out_dir):
    if df.empty:
        print("no data to plot")
        return

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"MGMI External Validation - Independent Evidence Integration\n"
        f"Disease: Parkinson's ({PD_EFO_ID})  |  "
        f"n={len(df)} targets  |  "
        f"Sources: OpenTargets + PubMed + Ensembl",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    CONF_COLORS = {"HIGH": "#2ca02c", "MEDIUM": "#ff7f0e", "LOW": "#d62728"}
    CHANGE_COLORS = {"UNCHANGED": "#1f77b4", "UPGRADED": "#2ca02c", "DOWNGRADED": "#d62728"}

    genes = df["input_gene"].tolist()
    ext_scores = df["external_score"].tolist()
    ext_confs = df["ext_confidence"].tolist()
    old_confs = df["input_mgmi_confidence"].tolist()
    changes = df["conf_change"].tolist()

    sort_idx = sorted(range(len(ext_scores)), key=lambda i: ext_scores[i], reverse=True)
    s_genes = [genes[i] for i in sort_idx]
    s_scores = [ext_scores[i] for i in sort_idx]
    s_confs = [ext_confs[i] for i in sort_idx]
    bar_colors = [CONF_COLORS.get(c, "#aaa") for c in s_confs]

    bars = ax1.barh(range(len(s_genes)), s_scores, color=bar_colors,
                      edgecolor="#333", height=0.7)
    ax1.set_yticks(range(len(s_genes)))
    ax1.set_yticklabels(s_genes, fontsize=9)
    ax1.invert_yaxis()
    ax1.axvline(SCORE_HIGH, color="#2ca02c", lw=1.2, ls="--", alpha=0.7)
    ax1.axvline(SCORE_MEDIUM, color="#ff7f0e", lw=1.2, ls="--", alpha=0.7)
    for bar, sc in zip(bars, s_scores):
        ax1.text(sc + 0.003, bar.get_y() + bar.get_height() / 2,
                  f"{sc:.3f}", va="center", fontsize=7.5)
    ax1.set_xlabel("External Validation Score", fontsize=10)
    ax1.set_title("External Score per Target\n(color = ext_confidence)",
                    fontsize=10, fontweight="bold")
    ax1.grid(axis="x", alpha=0.25)
    patches = [mpatches.Patch(color=v, label=k) for k, v in CONF_COLORS.items()]
    ax1.legend(handles=patches + [
        plt.Line2D([0], [0], color="#2ca02c", ls="--", lw=1.2, label=f"HIGH>={SCORE_HIGH}"),
        plt.Line2D([0], [0], color="#ff7f0e", ls="--", lw=1.2, label=f"MEDIUM>={SCORE_MEDIUM}"),
    ], fontsize=7, loc="lower right")

    x = np.arange(len(genes))
    ot_c = df["ot_contrib"].tolist()
    up_c = df["up_contrib"].tolist()
    ens_c = df["ens_contrib"].tolist()
    ax2.bar(x, ot_c, label=f"OpenTargets (w={W_OPENTARGETS})", color="#4A7FC1", edgecolor="#333")
    ax2.bar(x, up_c, bottom=ot_c,
             label=f"PubMed Cooccurrence (w={W_PUBMED})", color="#D85A30", edgecolor="#333")
    ax2.bar(x, ens_c, bottom=[a + b for a, b in zip(ot_c, up_c)],
             label=f"Ensembl biotype (w={W_ENSEMBL})", color="#1D9E75", edgecolor="#333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Score Contribution", fontsize=10)
    ax2.set_title("Score Decomposition by Data Source", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    conf_order = ["HIGH", "MEDIUM", "LOW"]
    old_pos = [conf_order.index(c) if c in conf_order else 0 for c in old_confs]
    new_pos = [conf_order.index(c) if c in conf_order else 0 for c in ext_confs]
    ch_colors = [CHANGE_COLORS.get(c, "#aaa") for c in changes]

    for i, (g, op, np_, ch) in enumerate(zip(genes, old_pos, new_pos, changes)):
        ax3.annotate("", xy=(np_, i), xytext=(op, i),
                       arrowprops=dict(arrowstyle="->", color=ch_colors[i],
                                         lw=1.5, connectionstyle="arc3,rad=0.1"))
        ax3.scatter([op], [i], color="#aaaaaa", s=60, zorder=3)
        ax3.scatter([np_], [i], color=CHANGE_COLORS.get(ch, "#aaa"), s=80, zorder=4, marker="D")
        ax3.text(np_ + 0.08, i, g, va="center", fontsize=8)

    ax3.set_yticks(range(len(genes)))
    ax3.set_yticklabels(genes, fontsize=8)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(conf_order, fontsize=10)
    ax3.set_xlim(-0.5, 3.0)
    ax3.set_title("Confidence Level Change\n(circle=MGMI original  diamond=External validated)",
                    fontsize=10, fontweight="bold")
    ax3.grid(axis="x", alpha=0.2)
    change_patches = [mpatches.Patch(color=v, label=k) for k, v in CHANGE_COLORS.items()]
    ax3.legend(handles=change_patches, fontsize=8, loc="lower right")

    gen_sc = df["ot_genetic_score"].fillna(0).tolist()
    lit_sc = df["ot_literature_score"].fillna(0).tolist()
    ov_sc = df["ot_overall_score"].fillna(0).tolist()
    sc_colors = [CONF_COLORS.get(c, "#aaa") for c in ext_confs]

    ax4.scatter(gen_sc, lit_sc, c=sc_colors, s=[max(30, v * 500) for v in ov_sc],
                 edgecolors="#333", linewidths=0.6, alpha=0.85, zorder=3)
    for i, g in enumerate(genes):
        ax4.annotate(g, (gen_sc[i], lit_sc[i]),
                       textcoords="offset points", xytext=(5, 3), fontsize=7.5)
    ax4.set_xlabel("OpenTargets Genetic Association Score", fontsize=10)
    ax4.set_ylabel("OpenTargets Literature Mining Score", fontsize=10)
    ax4.set_title("OpenTargets Evidence Breakdown\n(bubble size proportional to overall_score)",
                    fontsize=10, fontweight="bold")
    ax4.grid(alpha=0.25)
    patches4 = [mpatches.Patch(color=v, label=f"ext_{k}") for k, v in CONF_COLORS.items()]
    ax4.legend(handles=patches4, fontsize=8, loc="upper right")

    fig.savefig(str(out_dir / "fig_external_validation.png"),
                  dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_dir / 'fig_external_validation.png'}")


def print_summary(df):
    if df.empty:
        return
    print(f"\n{'=' * 65}")
    print("External validation summary")
    print(f"{'=' * 65}")
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        n = (df["ext_confidence"] == conf).sum()
        print(f"  external {conf:6s}: {n}/{len(df)}")
    print()
    for ch in ["UNCHANGED", "UPGRADED", "DOWNGRADED"]:
        n = (df["conf_change"] == ch).sum()
        genes = df[df["conf_change"] == ch]["input_gene"].tolist()
        print(f"  {ch:12s}: {n}  {genes}")
    print()
    print(f"  {'Gene':<12} {'Timepoint':<8} {'OT_score':<10} {'PubMed':<10} "
            f"{'Ens':<15} {'Ext_score':<10} {'Old->New conf'}")
    print(f"  {'-' * 85}")
    for _, row in df.iterrows():
        print(f"  {row['input_gene']:<12} {row['input_timepoint']:<8} "
                f"{row['ot_overall_score']:<10.4f} "
                f"{str(row.get('pm_count', 0)):<10} "
                f"{row['ens_biotype']:<15} "
                f"{row['external_score']:<10.4f} "
                f"{row['input_mgmi_confidence']}->{row['ext_confidence']}")
    print(f"\n  average external_score: {df['external_score'].mean():.4f}")
    print(f"{'=' * 65}")


def parse_args():
    p = argparse.ArgumentParser(description='MGMI external validation via independent databases')
    p.add_argument('--mgmi-json', type=Path, required=True,
                    help='mitokg_llm_raw_v2.json produced by mgmi_inference.py')
    p.add_argument('--out-dir', type=Path, default=Path('./output_mgmi_validation'))
    p.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                    help=f'how many top candidates to validate (default {DEFAULT_TOP_N})')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading candidates from: {args.mgmi_json}")
    if not args.mgmi_json.exists():
        print(f"ERROR: input file not found: {args.mgmi_json}")
        sys.exit(1)
    with open(args.mgmi_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    if not isinstance(candidates, list):
        print("ERROR: JSON should be a list of candidate dicts")
        sys.exit(1)
    print(f"Loaded {len(candidates)} candidates, validating top {args.top_n}")

    df = run_external_validation(candidates, args.out_dir, args.top_n)
    plot_validation_results(df, args.out_dir)
    print_summary(df)


if __name__ == "__main__":
    main()
