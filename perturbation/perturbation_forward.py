"""
LRRK2 forward virtual perturbation across time windows.

Goal: starting from Disease cells at Day0 / Day10 / Day14, apply perturbations
and predict Day42. Evaluate how closely the perturbed Disease trajectory can
approach the Control Day42 ground truth. Identifies "which genes to modify
at which developmental window" for maximal disease correction.

Pipeline:
    Layer 1A : single-gene scan (replace Disease gene with Control mean)
    Layer 1B : pathway-level cascade perturbation
    Layer 1C : two-pathway combined perturbation
    Layer 2  : gradient-guided optimal perturbation
    Figures  : 8 diagnostic plots + Excel workbook with 7 sheets

Usage:
    python perturbation_forward.py \\
        --data-dir /path/to/data/lrrk2 \\
        --gnn-dir  /path/to/MitoKG/gnn \\
        --weight   /path/to/best_by_r_lrrk2.pt \\
        --out-dir  ./output_perturbation_forward
"""

import argparse
import random
import sys
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torchdiffeq import odeint

warnings.filterwarnings('ignore')

from model import JTLatentODE_v6

try:
    import openpyxl
    XLSX_OK = True
except ImportError:
    print("[warning] openpyxl not installed, Excel export skipped")
    XLSX_OK = False


MAX_DAYS = 42.0
TRAIN_RATIO = 0.70
SEED = 99

TOP_SCAN_GENES = 984
L2_STEPS = 3000
L2_LR = 0.005
L2_LAMBDA_SPARSE = 0.01
L2_LAMBDA_KG = 0.1
L2_TOP_N = 20
RUN_DUAL_PATHWAY = True

COND_DIS = 1
COND_CTL = 0

DIS_KEY = 'LRRK2'
CTL_KEY = 'Control'
TP_ORDER = ['Day0', 'Day10', 'Day14', 'Day42']
TP_DAYS = {'Day0': 0.5, 'Day10': 10., 'Day14': 14., 'Day42': 42.}
START_TPS = ['Day0', 'Day10', 'Day14']


PATHWAY_GROUPS = {
    'Complex_I': ['NDUFA1', 'NDUFA2', 'NDUFA3', 'NDUFA4', 'NDUFA5', 'NDUFA6', 'NDUFA7', 'NDUFA8', 'NDUFA9',
                   'NDUFA10', 'NDUFA11', 'NDUFA12', 'NDUFA13', 'NDUFAB1', 'NDUFB1', 'NDUFB2', 'NDUFB3',
                   'NDUFB4', 'NDUFB5', 'NDUFB6', 'NDUFB7', 'NDUFB8', 'NDUFB9', 'NDUFB10', 'NDUFB11',
                   'NDUFC1', 'NDUFC2', 'NDUFS1', 'NDUFS2', 'NDUFS3', 'NDUFS4', 'NDUFS5', 'NDUFS6',
                   'NDUFS7', 'NDUFS8', 'NDUFV1', 'NDUFV2', 'NDUFV3'],
    'Complex_II': ['SDHA', 'SDHB', 'SDHC', 'SDHD', 'SDHAF1', 'SDHAF2', 'SDHAF3', 'SDHAF4'],
    'Complex_III': ['CYC1', 'UQCRC1', 'UQCRC2', 'UQCRFS1', 'UQCRB', 'UQCRQ', 'UQCRH', 'UQCR10', 'UQCR11',
                     'UQCC1', 'UQCC2', 'UQCC3'],
    'Complex_IV': ['COX4I1', 'COX5A', 'COX5B', 'COX6A1', 'COX6B1', 'COX6C', 'COX7A2', 'COX7A2L',
                    'COX7B', 'COX7C', 'COX8A', 'COX11', 'COX14', 'COX15', 'COX16', 'COX17', 'COX19', 'COX20'],
    'ATP_Synthase': ['ATP5F1A', 'ATP5F1B', 'ATP5F1C', 'ATP5F1D', 'ATP5F1E', 'ATP5PB', 'ATP5PD', 'ATP5PF',
                      'ATP5PO', 'ATP5IF1', 'ATPAF1', 'ATPAF2'],
    'TCA_Cycle': ['CS', 'ACO2', 'IDH2', 'IDH3A', 'IDH3B', 'IDH3G', 'OGDH', 'DLST', 'DLD', 'SUCLA2',
                   'SUCLG1', 'SUCLG2', 'SDHA', 'FH', 'MDH2'],
    'Mitophagy_PINK1_LRRK2': ['PINK1', 'PRKN', 'LRRK2', 'PARK7', 'HTRA2', 'PARL', 'TOMM20', 'TOMM40',
                                'TOMM7', 'TOMM22', 'TIMM23', 'TIMM44'],
    'Mito_Dynamics': ['DNM1L', 'FIS1', 'MFF', 'MFN1', 'MFN2', 'OPA1', 'BNIP3', 'BNIP3L', 'FUNDC1',
                       'FUNDC2', 'RHOT1', 'RHOT2'],
    'Mito_Ribosome': ['MRPL1', 'MRPL2', 'MRPL3', 'MRPL4', 'MRPL10', 'MRPL11', 'MRPL12', 'MRPL13',
                       'MRPS5', 'MRPS6', 'MRPS7', 'MRPS9', 'MRPS10', 'MRPS11', 'MRPS12'],
    'Oxidative_Stress': ['SOD1', 'SOD2', 'GPX1', 'GPX4', 'PRDX2', 'PRDX3', 'PRDX5', 'TXN2', 'TXNRD2',
                           'GSR', 'CYCS'],
}

COLORS = {'Complex_I': '#4A7FC1', 'Complex_II': '#D85A30', 'Complex_III': '#1D9E75',
          'Complex_IV': '#7F77DD', 'ATP_Synthase': '#EF9F27', 'TCA_Cycle': '#D4537E',
          'Mitophagy_PINK1_LRRK2': '#E24B4A', 'Mito_Dynamics': '#639922',
          'Mito_Ribosome': '#888780', 'Oxidative_Stress': '#BA7517', 'Other': '#B4B2A9'}

TP_COLORS = {'Day0': '#AEC6E8', 'Day10': '#6A9FCC', 'Day14': '#2E6FAA', 'Day42': '#0D2D5E'}


class VAEDecoder_Opt(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, t_feat_dim=16, kg_ctx_dim=32):
        super().__init__()
        in_dim = latent_dim + t_feat_dim + 2 * kg_ctx_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, z, t_feat, kg_ctx, cond_emb):
        return self.net(torch.cat([z, t_feat, kg_ctx, cond_emb], dim=-1))


def parse_args():
    p = argparse.ArgumentParser(description='LRRK2 forward virtual perturbation')
    p.add_argument('--data-dir', type=Path, required=True,
                   help='directory containing expr_scaled.npy and cell_meta.csv')
    p.add_argument('--gnn-dir', type=Path, required=True,
                   help='directory containing gene_embeddings.npy and gene_list.txt')
    p.add_argument('--weight', type=Path, required=True,
                   help='path to trained MitoLODE checkpoint (best_by_r_lrrk2.pt)')
    p.add_argument('--out-dir', type=Path, default=Path('./output_perturbation_forward'))
    p.add_argument('--seed', type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}  |  Seed: {args.seed}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({'font.family': 'DejaVu Sans',
                          'axes.spines.top': False, 'axes.spines.right': False,
                          'figure.dpi': 150})

    print("\n" + "=" * 60)
    print("Step 1: load LRRK2 data")
    print("=" * 60)

    all_expr = np.load(str(args.data_dir / "expr_scaled.npy")).astype(np.float32)
    all_meta = pd.read_csv(str(args.data_dir / "cell_meta.csv"))

    for old, new in [('Condition', 'condition'), ('Timepoint', 'timepoint')]:
        if old in all_meta.columns and new not in all_meta.columns:
            all_meta.rename(columns={old: new}, inplace=True)

    tr_idx, te_idx = [], []
    for tp in TP_ORDER:
        for cond in [DIS_KEY, CTL_KEY]:
            idx = np.where(((all_meta['timepoint'] == tp) & (all_meta['condition'] == cond)).values)[0]
            if len(idx) < 4:
                tr_idx.extend(idx.tolist())
                continue
            np.random.seed(args.seed)
            np.random.shuffle(idx)
            cut = max(1, int(len(idx) * TRAIN_RATIO))
            tr_idx.extend(idx[:cut].tolist())
            te_idx.extend(idx[cut:].tolist())

    train_expr = all_expr[tr_idx]
    test_expr = all_expr[te_idx]
    train_meta = all_meta.iloc[tr_idx].reset_index(drop=True)
    test_meta = all_meta.iloc[te_idx].reset_index(drop=True)

    N_GENES = train_expr.shape[1]
    print(f"  train={train_expr.shape}  test={test_expr.shape}  N_GENES={N_GENES}")

    gene_emb_np = np.load(str(args.gnn_dir / "gene_embeddings.npy")).astype(np.float32)[:N_GENES]
    gene_emb_t = torch.tensor(gene_emb_np, dtype=torch.float32).to(DEVICE)

    gene_list_path = args.gnn_dir / "gene_list.txt"
    if not gene_list_path.exists():
        gene_list_path = args.gnn_dir / "gene_list"
    with open(str(gene_list_path), encoding='utf-8') as f:
        gene_names = [l.strip() for l in f if l.strip()][:N_GENES]
    gene_name_lower = [g.lower() for g in gene_names]
    print(f"  gene_list loaded: {len(gene_names)} genes")

    expressed_idx = list(range(N_GENES))

    def get_mean(expr, meta, tp, cond):
        m = ((meta['condition'] == cond) & (meta['timepoint'] == tp)).values
        return expr[m].mean(0) if m.sum() >= 2 else None

    test_gt = {}
    for tp in TP_ORDER:
        for cond, lbl in [(DIS_KEY, 'dis'), (CTL_KEY, 'ctl')]:
            v = get_mean(test_expr, test_meta, tp, cond)
            if v is not None:
                test_gt[(tp, lbl)] = v

    true_ctl_day42 = test_gt.get(('Day42', 'ctl'))
    true_dis_day42 = test_gt.get(('Day42', 'dis'))
    assert true_ctl_day42 is not None, "Control Day42 GT is missing"
    assert true_dis_day42 is not None, "Disease Day42 GT is missing"
    print(f"  endpoint target: Control Day42 GT (shape={true_ctl_day42.shape})")

    print("\n" + "=" * 60)
    print("Step 2: load MitoLODE model")
    print("=" * 60)

    ckpt = torch.load(str(args.weight), map_location=DEVICE, weights_only=False)
    hp = ckpt.get('hp', {})
    hp.setdefault('latent_dim', 32)
    hp.setdefault('hidden_dim', 256)
    hp.setdefault('gene_emb_dim', 64)
    hp.setdefault('mask_ratio', 0.25)
    hp.setdefault('ode_hidden', 128)
    hp.setdefault('kg_ctx_dim', 32)
    hp.setdefault('t_feat_dim', 16)
    hp.setdefault('dz_scale', 0.15)
    hp.setdefault('delta_scale', 0.20)

    model = JTLatentODE_v6(
        input_dim=N_GENES,
        latent_dim=hp['latent_dim'],
        hidden_dim=hp['hidden_dim'],
        gene_emb_dim=hp['gene_emb_dim'],
        mask_ratio=hp['mask_ratio'],
        ode_hidden=hp['ode_hidden'],
        kg_ctx_dim=hp['kg_ctx_dim'],
        t_feat_dim=hp['t_feat_dim'],
        max_days=MAX_DAYS,
        dz_scale=hp['dz_scale'],
        delta_scale=hp['delta_scale'],
        use_temporal_attention=True,
    ).to(DEVICE)

    model.decoder = VAEDecoder_Opt(
        hp['latent_dim'], N_GENES, hp['hidden_dim'],
        hp['t_feat_dim'], hp['kg_ctx_dim']
    ).to(DEVICE)

    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  model loaded. parameters={sum(p.numel() for p in model.parameters()):,}")

    def _t_feat(day_scalar, B):
        d = torch.full((B,), day_scalar, dtype=torch.float32, device=DEVICE)
        return model.ode_func.t_encoder((d / MAX_DAYS).unsqueeze(-1).clamp(0, 1))

    def _decode(z, day_scalar, cond_idx):
        B = z.size(0)
        t_feat = _t_feat(day_scalar, B)
        kg_ctx = model.ode_func.kg_proj(gene_emb_t.mean(0, keepdim=True)).expand(B, -1)
        cond_emb = model.ode_func.cond_emb(
            torch.full((B,), cond_idx, dtype=torch.long, device=DEVICE))
        return model.decoder(z, t_feat, kg_ctx, cond_emb)

    @torch.no_grad()
    def encode_cells(expr_np, day_scalar):
        x = torch.tensor(expr_np, dtype=torch.float32, device=DEVICE)
        days = torch.full((len(x),), day_scalar, dtype=torch.float32, device=DEVICE)
        mu, *_ = model.encode(x, gene_emb_t, time_days=days)
        return mu.mean(0, keepdim=True)

    @torch.no_grad()
    def predict_from_expr(expr_np, start_day, target_day, cond_idx):
        z0 = encode_cells(expr_np, start_day)
        t_q = torch.tensor([start_day, target_day], dtype=torch.float32, device=DEVICE)
        model.ode_func.set_context(gene_emb_t)
        model.ode_func.set_condition(cond_idx)
        z_traj = odeint(model.ode_func, z0, t_q, method='rk4', options={'step_size': 0.5})
        return _decode(z_traj[1], target_day, cond_idx).cpu().numpy()[0]

    def eval_metrics(pred, true_ctl, true_dis=None):
        r_ctl = float(pearsonr(pred, true_ctl)[0])
        rmse = float(np.sqrt(np.mean((pred - true_ctl) ** 2)))
        r2 = float(r2_score(true_ctl, pred))
        cosine = float(1 - cosine_dist(pred, true_ctl))
        latent_dist = float(np.linalg.norm(pred - true_ctl))
        out = {'r_vs_ctl': r_ctl, 'rmse': rmse, 'r2': r2,
               'cosine': cosine, 'l2_dist': latent_dist}
        if true_dis is not None:
            out['r_vs_dis'] = float(pearsonr(pred, true_dis)[0])
        return out

    pathway_idx = {}
    for pathway, genes in PATHWAY_GROUPS.items():
        idx = []
        for g in genes:
            try:
                i = gene_name_lower.index(g.lower())
                idx.append(i)
            except ValueError:
                pass
        if idx:
            pathway_idx[pathway] = idx

    print(f"\n  pathway mapping ({len(pathway_idx)} pathways matched):")
    for pw, idx in pathway_idx.items():
        print(f"    {pw:<30} {len(idx)} genes")

    def get_gene_pathway(gene_name):
        gn_lower = gene_name.lower()
        for pw, idx_list in pathway_idx.items():
            for i in idx_list:
                if i < len(gene_names) and gene_names[i].lower() == gn_lower:
                    return pw
        return 'Other'

    all_kg_gene_idx = set()
    for idx_list in pathway_idx.values():
        all_kg_gene_idx.update(idx_list)
    kg_mask = torch.zeros(N_GENES, device=DEVICE)
    for i in all_kg_gene_idx:
        kg_mask[i] = 1.0

    print("\n" + "=" * 60)
    print("Step 3: baseline prediction (no perturbation)")
    print("=" * 60)

    start_data = {}
    for tp in START_TPS:
        day = TP_DAYS[tp]
        dis_mask = ((test_meta['condition'] == DIS_KEY) & (test_meta['timepoint'] == tp)).values
        ctl_mask = ((test_meta['condition'] == CTL_KEY) & (test_meta['timepoint'] == tp)).values
        dis_expr = test_expr[dis_mask]
        ctl_expr = test_expr[ctl_mask]
        dis_mean = dis_expr.mean(0) if len(dis_expr) >= 2 else None
        ctl_mean = ctl_expr.mean(0) if len(ctl_expr) >= 2 else None
        start_data[tp] = {
            'day': day, 'dis_expr': dis_expr, 'ctl_expr': ctl_expr,
            'dis_mean': dis_mean, 'ctl_mean': ctl_mean,
            'n_dis': len(dis_expr), 'n_ctl': len(ctl_expr)
        }
        print(f"  {tp} (day={day}): Disease n={len(dis_expr)}, Control n={len(ctl_expr)}")

    baseline = {}
    for tp in START_TPS:
        sd = start_data[tp]
        if sd['dis_mean'] is None:
            continue
        pred = predict_from_expr(sd['dis_expr'], sd['day'], 42.0, COND_DIS)
        m = eval_metrics(pred, true_ctl_day42, true_dis_day42)
        baseline[tp] = {'pred': pred, **m}
        print(f"  baseline {tp}->Day42: r_vs_Ctl={m['r_vs_ctl']:.4f}  "
              f"r_vs_Dis={m['r_vs_dis']:.4f}  RMSE={m['rmse']:.4f}  "
              f"cosine={m['cosine']:.4f}  L2={m['l2_dist']:.4f}")

    ideal_preds = {}
    for tp in START_TPS:
        sd = start_data[tp]
        if sd['ctl_mean'] is None:
            continue
        pred = predict_from_expr(sd['ctl_expr'], sd['day'], 42.0, COND_CTL)
        m = eval_metrics(pred, true_ctl_day42, true_dis_day42)
        ideal_preds[tp] = {'pred': pred, **m}
        print(f"  ideal {tp}->Day42 (Ctl->Ctl): r_vs_Ctl={m['r_vs_ctl']:.4f}")

    print("\n" + "=" * 60)
    print("Step 4: Layer 1A - single-gene scan (replace with Control mean)")
    print("=" * 60)

    layer1a_results = {}
    for tp in START_TPS:
        sd = start_data[tp]
        if sd['dis_mean'] is None or sd['ctl_mean'] is None:
            continue

        dis_mean = sd['dis_mean']
        ctl_mean = sd['ctl_mean']
        day = sd['day']
        base_r = baseline[tp]['r_vs_ctl']

        diff = np.abs(dis_mean - ctl_mean)
        scan_pool = [i for i in expressed_idx if diff[i] > 1e-6]
        scan_pool_sorted = sorted(scan_pool, key=lambda i: diff[i], reverse=True)
        scan_genes = scan_pool_sorted[:TOP_SCAN_GENES]

        print(f"\n  {tp}: scanning {len(scan_genes)} genes...")
        t0 = time.time()

        results_tp = []
        for gi in scan_genes:
            x_ko = dis_mean.copy()
            x_ko[gi] = ctl_mean[gi]
            pred_ko = predict_from_expr(x_ko[np.newaxis, :], day, 42.0, COND_DIS)
            m = eval_metrics(pred_ko, true_ctl_day42, true_dis_day42)
            delta_r = m['r_vs_ctl'] - base_r
            gname = gene_names[gi] if gi < len(gene_names) else f"Gene_{gi}"
            pw = get_gene_pathway(gname)
            results_tp.append({
                'gene_idx': gi, 'gene': gname, 'pathway': pw,
                'timepoint': tp, 'start_day': day,
                'perturb_type': 'replace_ctl_mean',
                'delta_r_vs_ctl': round(delta_r, 5),
                'r_after_vs_ctl': round(m['r_vs_ctl'], 5),
                'r_baseline_vs_ctl': round(base_r, 5),
                'r_vs_dis': round(m['r_vs_dis'], 5),
                'rmse': round(m['rmse'], 5),
                'r2': round(m['r2'], 5),
                'cosine_sim': round(m['cosine'], 5),
                'l2_dist': round(m['l2_dist'], 5),
                'ctl_dis_diff_at_start': round(float(ctl_mean[gi] - dis_mean[gi]), 5),
            })

        results_tp.sort(key=lambda x: x['delta_r_vs_ctl'], reverse=True)
        layer1a_results[tp] = results_tp
        elapsed = time.time() - t0
        print(f"  {tp} done ({elapsed:.1f}s). Top-5:")
        for r in results_tp[:5]:
            print(f"    {r['gene']:<12} dr={r['delta_r_vs_ctl']:+.4f}  "
                  f"r_after={r['r_after_vs_ctl']:.4f}  pathway={r['pathway']}")

    print("\n" + "=" * 60)
    print("Step 5: Layer 1B - pathway cascade perturbation")
    print("=" * 60)

    layer1b_results = {}
    for tp in START_TPS:
        sd = start_data[tp]
        if sd['dis_mean'] is None or sd['ctl_mean'] is None:
            continue
        dis_mean = sd['dis_mean']
        ctl_mean = sd['ctl_mean']
        day = sd['day']
        base_r = baseline[tp]['r_vs_ctl']

        results_tp = []
        for pathway, idx in pathway_idx.items():
            x_pw = dis_mean.copy()
            for i in idx:
                x_pw[i] = ctl_mean[i]
            pred_pw = predict_from_expr(x_pw[np.newaxis, :], day, 42.0, COND_DIS)
            m = eval_metrics(pred_pw, true_ctl_day42, true_dis_day42)
            delta_r = m['r_vs_ctl'] - base_r
            results_tp.append({
                'timepoint': tp, 'start_day': day, 'pathway': pathway,
                'n_genes_perturbed': len(idx),
                'delta_r_vs_ctl': round(delta_r, 5),
                'r_after_vs_ctl': round(m['r_vs_ctl'], 5),
                'r_baseline_vs_ctl': round(base_r, 5),
                'r_vs_dis': round(m['r_vs_dis'], 5),
                'rmse': round(m['rmse'], 5),
                'r2': round(m['r2'], 5),
                'cosine_sim': round(m['cosine'], 5),
                'l2_dist': round(m['l2_dist'], 5),
            })

        results_tp.sort(key=lambda x: x['delta_r_vs_ctl'], reverse=True)
        layer1b_results[tp] = results_tp
        print(f"\n  {tp} pathway perturbation results:")
        for r in results_tp:
            print(f"    {r['pathway']:<30} dr={r['delta_r_vs_ctl']:+.4f}  "
                  f"r_after={r['r_after_vs_ctl']:.4f}")

    print("\n" + "=" * 60)
    print("Step 6: Layer 1C - dual-pathway joint perturbation")
    print("=" * 60)

    layer1c_results = {}
    if RUN_DUAL_PATHWAY and len(pathway_idx) >= 2:
        pw_names = list(pathway_idx.keys())
        combos = list(combinations(pw_names, 2))
        print(f"  {len(combos)} combinations x {len(START_TPS)} timepoints")

        for tp in START_TPS:
            sd = start_data[tp]
            if sd['dis_mean'] is None or sd['ctl_mean'] is None:
                continue
            dis_mean = sd['dis_mean']
            ctl_mean = sd['ctl_mean']
            day = sd['day']
            base_r = baseline[tp]['r_vs_ctl']

            single_dr = {r['pathway']: r['delta_r_vs_ctl']
                          for r in layer1b_results.get(tp, [])}

            results_tp = []
            for pw_a, pw_b in combos:
                x_dual = dis_mean.copy()
                for i in pathway_idx[pw_a]:
                    x_dual[i] = ctl_mean[i]
                for i in pathway_idx[pw_b]:
                    x_dual[i] = ctl_mean[i]
                pred_d = predict_from_expr(x_dual[np.newaxis, :], day, 42.0, COND_DIS)
                m = eval_metrics(pred_d, true_ctl_day42, true_dis_day42)
                delta_r = m['r_vs_ctl'] - base_r
                dr_a = single_dr.get(pw_a, 0.0)
                dr_b = single_dr.get(pw_b, 0.0)
                synergy = delta_r - (dr_a + dr_b)
                n_genes = len(pathway_idx[pw_a]) + len(pathway_idx[pw_b])
                results_tp.append({
                    'timepoint': tp, 'start_day': day,
                    'pathway_A': pw_a, 'pathway_B': pw_b,
                    'n_genes_perturbed': n_genes,
                    'delta_r_vs_ctl': round(delta_r, 5),
                    'r_after_vs_ctl': round(m['r_vs_ctl'], 5),
                    'r_baseline_vs_ctl': round(base_r, 5),
                    'delta_r_A_alone': round(dr_a, 5),
                    'delta_r_B_alone': round(dr_b, 5),
                    'synergy_score': round(synergy, 5),
                    'r_vs_dis': round(m['r_vs_dis'], 5),
                    'rmse': round(m['rmse'], 5),
                    'r2': round(m['r2'], 5),
                    'cosine_sim': round(m['cosine'], 5),
                    'l2_dist': round(m['l2_dist'], 5),
                })

            results_tp.sort(key=lambda x: x['delta_r_vs_ctl'], reverse=True)
            layer1c_results[tp] = results_tp
            print(f"\n  {tp} Top-3 dual-pathway combinations:")
            for r in results_tp[:3]:
                print(f"    {r['pathway_A']} + {r['pathway_B']}")
                print(f"      dr={r['delta_r_vs_ctl']:+.4f}  synergy={r['synergy_score']:+.4f}")

    print("\n" + "=" * 60)
    print("Step 7: Layer 2 - gradient-guided optimal perturbation")
    print("=" * 60)

    layer2_results = {}
    for tp in START_TPS:
        sd = start_data[tp]
        if sd['dis_mean'] is None:
            continue
        dis_mean = sd['dis_mean']
        day = sd['day']
        base_r = baseline[tp]['r_vs_ctl']

        print(f"\n  -- {tp} (day={day}) --")
        target_t = torch.tensor(true_ctl_day42, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x_dis_t = torch.tensor(dis_mean, dtype=torch.float32, device=DEVICE)
        day_t = torch.tensor([day], dtype=torch.float32, device=DEVICE)

        delta = torch.zeros(N_GENES, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=L2_LR)

        best_loss = float('inf')
        best_delta = None
        loss_hist = []
        r_hist = []

        print(f"  optimization {L2_STEPS} steps  lr={L2_LR}  lambda_sparse={L2_LAMBDA_SPARSE}  lambda_kg={L2_LAMBDA_KG}")
        t0 = time.time()

        for step in range(L2_STEPS):
            optimizer.zero_grad()
            x_perturbed = (x_dis_t + delta).clamp(-5.0, 5.0).unsqueeze(0)
            mu, logvar, z, mask, dz, _ = model.encode(x_perturbed, gene_emb_t, time_days=day_t)
            t_q = torch.tensor([day, 42.0], dtype=torch.float32, device=DEVICE)
            model.ode_func.set_context(gene_emb_t)
            model.ode_func.set_condition(COND_DIS)
            z_traj = odeint(model.ode_func, mu, t_q, method='rk4', options={'step_size': 0.5})
            pred = _decode(z_traj[1], 42.0, COND_DIS)

            recon_loss = F.mse_loss(pred, target_t)
            sparse_loss = delta.abs().mean()
            nonkg_loss = (delta.abs() * (1.0 - kg_mask)).mean()
            loss = recon_loss + L2_LAMBDA_SPARSE * sparse_loss + L2_LAMBDA_KG * nonkg_loss

            loss.backward()
            optimizer.step()
            loss_hist.append(float(loss.item()))

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()

            if step % 500 == 0 or step == L2_STEPS - 1:
                with torch.no_grad():
                    pred_np = pred.cpu().numpy()[0]
                    r_now = float(pearsonr(pred_np, true_ctl_day42)[0])
                r_hist.append((step, r_now))
                print(f"    step {step:4d} | loss={loss.item():.4f}  r_vs_Ctl={r_now:.4f}")

        delta_np = best_delta.cpu().numpy()
        x_opt_np = np.clip(dis_mean + delta_np, -5.0, 5.0)
        pred_opt = predict_from_expr(x_opt_np[np.newaxis, :], day, 42.0, COND_DIS)
        m_opt = eval_metrics(pred_opt, true_ctl_day42, true_dis_day42)

        elapsed = time.time() - t0
        print(f"  {tp} done ({elapsed:.1f}s)")
        print(f"    baseline r_vs_Ctl={base_r:.4f} -> optimal {m_opt['r_vs_ctl']:.4f}  "
              f"dr={m_opt['r_vs_ctl'] - base_r:+.4f}")

        top_idx = np.argsort(np.abs(delta_np))[::-1][:L2_TOP_N]
        ctl_mean = sd['ctl_mean'] if sd['ctl_mean'] is not None else np.zeros(N_GENES)
        top_genes = []
        for i in top_idx:
            gname = gene_names[i] if i < len(gene_names) else f"Gene_{i}"
            top_genes.append({
                'rank': len(top_genes) + 1,
                'gene': gname,
                'pathway': get_gene_pathway(gname),
                'timepoint': tp, 'start_day': day,
                'delta_value': round(float(delta_np[i]), 5),
                'direction': 'up' if delta_np[i] > 0 else 'down',
                'ctl_dis_diff_at_start': round(float(ctl_mean[i] - dis_mean[i]), 5),
                'delta_r_vs_ctl': round(float(m_opt['r_vs_ctl'] - base_r), 5),
                'r_after_vs_ctl': round(m_opt['r_vs_ctl'], 5),
                'r_baseline_vs_ctl': round(base_r, 5),
                'r_vs_dis': round(m_opt['r_vs_dis'], 5),
                'rmse': round(m_opt['rmse'], 5),
                'r2': round(m_opt['r2'], 5),
                'cosine_sim': round(m_opt['cosine'], 5),
                'l2_dist': round(m_opt['l2_dist'], 5),
            })

        layer2_results[tp] = {
            'delta_np': delta_np, 'x_opt_np': x_opt_np,
            'pred_opt': pred_opt, 'metrics': m_opt,
            'top_genes': top_genes,
            'loss_hist': loss_hist, 'r_hist': r_hist,
            'base_r': base_r,
        }

        print(f"  Top-5 targets:")
        for tg in top_genes[:5]:
            print(f"    {tg['gene']:<12} delta={tg['delta_value']:+.4f} "
                  f"{'up' if tg['direction'] == 'up' else 'down'}  pathway={tg['pathway']}")

    print("\n" + "=" * 60)
    print("Step 8: visualization")
    print("=" * 60)

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics_names = ['r_vs_ctl', 'rmse', 'cosine']
        metric_labels = ['Pearson r vs Control GT', 'RMSE', 'Cosine Similarity']
        for ax, metric, mlabel in zip(axes, metrics_names, metric_labels):
            tps_valid = [tp for tp in START_TPS if tp in baseline]
            vals_base = [baseline[tp][metric] for tp in tps_valid]
            vals_ideal = [ideal_preds[tp][metric] for tp in tps_valid if tp in ideal_preds]
            colors = [TP_COLORS[tp] for tp in tps_valid]
            bars = ax.bar(tps_valid, vals_base, color=colors, edgecolor='#333', width=0.5, alpha=0.85)
            if vals_ideal:
                ax.bar([t + '_ideal' for t in tps_valid[:len(vals_ideal)]],
                        vals_ideal, color=colors[:len(vals_ideal)],
                        edgecolor='#333', width=0.5, alpha=0.35, hatch='//')
            for bar, v in zip(bars, vals_base):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                         f'{v:.4f}', ha='center', va='bottom', fontsize=8.5)
            ax.set_title(mlabel, fontsize=11, fontweight='bold')
            ax.set_xlabel('Start timepoint', fontsize=10)
            ax.tick_params(axis='x', rotation=15)
            ax.grid(axis='y', alpha=0.3)
        fig.suptitle('LRRK2 - Baseline Prediction (Disease -> Day42)\nvs Control Day42 GT',
                      fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(str(args.out_dir / "fig01_baseline_summary.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("  Fig01: baseline summary")
    except Exception as e:
        print(f"  Fig01 failed: {e}")

    for tp in START_TPS:
        if tp not in layer1a_results:
            continue
        try:
            top30 = layer1a_results[tp][:30]
            fig, ax = plt.subplots(figsize=(14, 8))
            deltas = [r['delta_r_vs_ctl'] for r in top30]
            colors = [COLORS.get(r['pathway'], COLORS['Other']) for r in top30]
            ax.barh(range(len(top30)), deltas, color=colors, edgecolor='#333', height=0.72)
            ax.set_yticks(range(len(top30)))
            ax.set_yticklabels([f"{r['gene']} ({'up' if r['ctl_dis_diff_at_start'] > 0 else 'down'})"
                                for r in top30], fontsize=9)
            ax.axvline(0, color='#333', lw=0.8, ls='--')
            ax.set_xlabel('dr vs Control Day42 GT', fontsize=11)
            base_r = baseline[tp]['r_vs_ctl']
            ax.set_title(f'Layer 1A - Single-gene Correction: Top-30  [{tp} -> Day42]\n'
                          f'(Baseline r={base_r:.4f}, replace disease gene with Control mean)',
                          fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.25)
            legend_patches = [Patch(color=c, label=p.replace('_', ' '))
                                for p, c in COLORS.items() if p != 'Other']
            ax.legend(handles=legend_patches, fontsize=7.5, loc='lower right', ncol=2, framealpha=0.8)
            fig.tight_layout()
            fig.savefig(str(args.out_dir / f"fig02_{tp}_layer1a_topgenes.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Fig02_{tp}")
        except Exception as e:
            print(f"  Fig02_{tp} failed: {e}")

    for tp in START_TPS:
        if tp not in layer1b_results:
            continue
        try:
            res = layer1b_results[tp]
            fig, ax = plt.subplots(figsize=(10, 6))
            pw_names = [r['pathway'].replace('_', ' ') for r in res]
            deltas = [r['delta_r_vs_ctl'] for r in res]
            colors = [COLORS.get(r['pathway'], COLORS['Other']) for r in res]
            ax.barh(range(len(res)), deltas, color=colors, edgecolor='#333', height=0.65)
            ax.set_yticks(range(len(res)))
            ax.set_yticklabels(pw_names, fontsize=10)
            ax.axvline(0, color='#333', lw=0.8, ls='--')
            ax.set_xlabel('dr vs Control Day42 GT', fontsize=11)
            ax.set_title(f'Layer 1B - Pathway Perturbation  [{tp} -> Day42]\n'
                          f'(Baseline r={baseline[tp]["r_vs_ctl"]:.4f})',
                          fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.25)
            for i, r in enumerate(res):
                x_val = r['delta_r_vs_ctl']
                ax.text(x_val + 0.0005 if x_val >= 0 else x_val - 0.0005, i,
                         f"r={r['r_after_vs_ctl']:.4f}", va='center',
                         ha='left' if x_val >= 0 else 'right', fontsize=8)
            fig.tight_layout()
            fig.savefig(str(args.out_dir / f"fig03_{tp}_layer1b_pathway.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Fig03_{tp}")
        except Exception as e:
            print(f"  Fig03_{tp} failed: {e}")

    try:
        all_top_genes = set()
        for tp in START_TPS:
            if tp in layer1a_results:
                for r in layer1a_results[tp][:20]:
                    all_top_genes.add(r['gene'])
        all_top_genes = sorted(all_top_genes)
        hm_data = np.full((len(all_top_genes), len(START_TPS)), np.nan)
        for j, tp in enumerate(START_TPS):
            if tp not in layer1a_results:
                continue
            gene_dr = {r['gene']: r['delta_r_vs_ctl'] for r in layer1a_results[tp]}
            for i, g in enumerate(all_top_genes):
                if g in gene_dr:
                    hm_data[i, j] = gene_dr[g]
        fig, ax = plt.subplots(figsize=(8, max(6, len(all_top_genes) * 0.35)))
        vmax = np.nanmax(np.abs(hm_data))
        sns.heatmap(hm_data, ax=ax, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                     xticklabels=START_TPS, yticklabels=all_top_genes,
                     linewidths=0.4, cbar_kws={'label': 'dr vs Control Day42 GT'})
        ax.set_title('Time Window x Gene - Perturbation Effect Heatmap\n'
                      '(Union of Top-20 genes from each timepoint)',
                      fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=10)
        fig.tight_layout()
        fig.savefig(str(args.out_dir / "fig04_timewindow_gene_heatmap.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("  Fig04")
    except Exception as e:
        print(f"  Fig04 failed: {e}")

    try:
        fig, axes = plt.subplots(1, len(START_TPS), figsize=(15, 4), sharey=False)
        for ax, tp in zip(axes, START_TPS):
            if tp not in layer2_results:
                ax.set_visible(False)
                continue
            res = layer2_results[tp]
            lh = res['loss_hist']
            base_r = res['base_r']
            opt_r = res['metrics']['r_vs_ctl']
            ax.plot(lh, color='#D85A30', lw=1.2, alpha=0.9)
            ax.set_title(f'{tp} -> Day42\nr: {base_r:.4f} -> {opt_r:.4f} ({opt_r - base_r:+.4f})',
                          fontsize=10, fontweight='bold')
            ax.set_xlabel('Step', fontsize=9)
            ax.set_ylabel('Loss', fontsize=9)
            ax.grid(alpha=0.25)
        fig.suptitle('Layer 2 - Gradient Optimization Convergence',
                      fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(str(args.out_dir / "fig05_gradient_convergence.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("  Fig05")
    except Exception as e:
        print(f"  Fig05 failed: {e}")

    for tp in START_TPS:
        if tp not in layer2_results:
            continue
        try:
            res = layer2_results[tp]
            top_genes = res['top_genes']
            fig, ax = plt.subplots(figsize=(12, 7))
            deltas = [tg['delta_value'] for tg in top_genes]
            colors = [COLORS.get(tg['pathway'], COLORS['Other']) for tg in top_genes]
            ax.barh(range(len(top_genes)), [abs(d) for d in deltas],
                     color=colors, edgecolor='#333', height=0.7)
            ax.set_yticks(range(len(top_genes)))
            ax.set_yticklabels(
                [f"{tg['gene']} ({'up' if tg['direction'] == 'up' else 'down'})" for tg in top_genes],
                fontsize=9.5)
            ax.set_xlabel('|delta| Perturbation Magnitude', fontsize=11)
            base_r = res['base_r']
            opt_r = res['metrics']['r_vs_ctl']
            ax.set_title(f'Layer 2 - Gradient-Guided Optimal Perturbation: Top-{L2_TOP_N}  [{tp} -> Day42]\n'
                          f'r_vs_Ctl: {base_r:.4f} -> {opt_r:.4f}  (dr={opt_r - base_r:+.4f})',
                          fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.25)
            legend_patches = [Patch(color=c, label=p.replace('_', ' '))
                                for p, c in COLORS.items() if p != 'Other']
            ax.legend(handles=legend_patches, fontsize=7.5, loc='lower right', ncol=2, framealpha=0.8)
            fig.tight_layout()
            fig.savefig(str(args.out_dir / f"fig06_{tp}_layer2_targets.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Fig06_{tp}")
        except Exception as e:
            print(f"  Fig06_{tp} failed: {e}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        tps_valid = [tp for tp in START_TPS if tp in layer2_results]
        base_rs = [layer2_results[tp]['base_r'] for tp in tps_valid]
        opt_rs = [layer2_results[tp]['metrics']['r_vs_ctl'] for tp in tps_valid]
        ideal_rs = [ideal_preds[tp]['r_vs_ctl'] for tp in tps_valid if tp in ideal_preds]
        x = np.arange(len(tps_valid))
        w = 0.28
        ax.bar(x - w, base_rs, w, label='Baseline (no perturb)', color='#AEC6E8', edgecolor='#333')
        ax.bar(x, opt_rs, w, label='After L2 perturbation', color='#2E6FAA', edgecolor='#333')
        if ideal_rs:
            ax.bar(x + w, ideal_rs[:len(tps_valid)], w,
                     label='Ideal (Ctl->Ctl)', color='#88C888', edgecolor='#333')
        ax.set_xticks(x)
        ax.set_xticklabels(tps_valid)
        ax.set_ylabel('Pearson r vs Control Day42 GT', fontsize=10)
        ax.set_title('Pearson r Comparison Across Time Windows', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        for xi, (b, o) in enumerate(zip(base_rs, opt_rs)):
            ax.text(xi, o + 0.003, f'+{o - b:.4f}', ha='center', va='bottom', fontsize=8,
                     color='#2E6FAA', fontweight='bold')

        ax2 = axes[1]
        delta_rs = [layer2_results[tp]['metrics']['r_vs_ctl'] - layer2_results[tp]['base_r']
                     for tp in tps_valid]
        bar_colors = [TP_COLORS[tp] for tp in tps_valid]
        bars = ax2.bar(tps_valid, delta_rs, color=bar_colors, edgecolor='#333', width=0.5)
        for bar, v in zip(bars, delta_rs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                      f'{v:+.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.set_ylabel('dr (improvement over baseline)', fontsize=10)
        ax2.set_title('dr by Time Window', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(0, color='#333', lw=0.8)

        fig.suptitle('LRRK2 - Time Window Perturbation Comparison (-> Day42)',
                      fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(str(args.out_dir / "fig07_timewindow_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("  Fig07")
    except Exception as e:
        print(f"  Fig07 failed: {e}")

    for tp in START_TPS:
        if tp not in layer2_results:
            continue
        try:
            res = layer2_results[tp]
            delta_np = res['delta_np']
            sd = start_data[tp]
            dis_mean = sd['dis_mean']
            x_opt_np = res['x_opt_np']
            pred_opt = res['pred_opt']
            pred_base = baseline[tp]['pred']
            N_SHOW = 40
            top_idx = np.argsort(np.abs(delta_np))[::-1][:N_SHOW]
            glabels = [gene_names[i] if i < len(gene_names) else f"G{i}" for i in top_idx]

            hm_data = np.stack([
                dis_mean[top_idx],
                x_opt_np[top_idx],
                pred_base[top_idx],
                pred_opt[top_idx],
                true_dis_day42[top_idx],
                true_ctl_day42[top_idx],
            ], axis=0)
            row_labels = [
                f'Dis {tp} (input)',
                f'Perturbed {tp}',
                'Pred Day42 (baseline)',
                'Pred Day42 (perturbed)',
                'GT Disease Day42',
                'GT Control Day42',
            ]
            fig, ax = plt.subplots(figsize=(16, 9))
            sns.heatmap(hm_data.T, ax=ax, cmap='RdBu_r', center=0,
                         xticklabels=row_labels, yticklabels=glabels,
                         linewidths=0.1, cbar_kws={'label': 'Expression (z-score)'})
            base_r = res['base_r']
            opt_r = res['metrics']['r_vs_ctl']
            ax.set_title(f'Layer 2 - Perturbation Heatmap  [{tp} -> Day42]  Top-{N_SHOW} Target Genes\n'
                          f'r_vs_Ctl: {base_r:.4f} -> {opt_r:.4f}',
                          fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelsize=7)
            ax.tick_params(axis='x', labelsize=9, rotation=20)
            fig.tight_layout()
            fig.savefig(str(args.out_dir / f"fig08_{tp}_layer2_heatmap.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Fig08_{tp}")
        except Exception as e:
            print(f"  Fig08_{tp} failed: {e}")

    if XLSX_OK:
        try:
            xl_path = args.out_dir / "perturbation_lrrk2_temporal.xlsx"
            with pd.ExcelWriter(str(xl_path), engine='openpyxl') as writer:
                summary_rows = []
                for tp in START_TPS:
                    if tp not in baseline:
                        continue
                    b = baseline[tp]
                    l2 = layer2_results.get(tp, {})
                    l2m = l2.get('metrics', {})
                    summary_rows.append({
                        'timepoint': tp, 'start_day': TP_DAYS[tp],
                        'n_dis_cells': start_data[tp]['n_dis'],
                        'n_ctl_cells': start_data[tp]['n_ctl'],
                        'baseline_r_vs_ctl': b['r_vs_ctl'],
                        'baseline_r_vs_dis': b.get('r_vs_dis', ''),
                        'baseline_rmse': b['rmse'], 'baseline_r2': b['r2'],
                        'baseline_cosine': b['cosine'], 'baseline_l2_dist': b['l2_dist'],
                        'ideal_r_vs_ctl': ideal_preds.get(tp, {}).get('r_vs_ctl', ''),
                        'L1A_best_gene': layer1a_results.get(tp, [{}])[0].get('gene', '') if layer1a_results.get(tp) else '',
                        'L1A_best_pathway': layer1a_results.get(tp, [{}])[0].get('pathway', '') if layer1a_results.get(tp) else '',
                        'L1A_best_delta_r': layer1a_results.get(tp, [{}])[0].get('delta_r_vs_ctl', '') if layer1a_results.get(tp) else '',
                        'L1A_best_r_after': layer1a_results.get(tp, [{}])[0].get('r_after_vs_ctl', '') if layer1a_results.get(tp) else '',
                        'L1B_best_pathway': layer1b_results.get(tp, [{}])[0].get('pathway', '') if layer1b_results.get(tp) else '',
                        'L1B_best_delta_r': layer1b_results.get(tp, [{}])[0].get('delta_r_vs_ctl', '') if layer1b_results.get(tp) else '',
                        'L2_opt_r_vs_ctl': l2m.get('r_vs_ctl', ''),
                        'L2_delta_r': round(l2m.get('r_vs_ctl', 0) - b['r_vs_ctl'], 5) if l2m else '',
                        'L2_opt_rmse': l2m.get('rmse', ''),
                        'L2_opt_r2': l2m.get('r2', ''),
                        'L2_opt_cosine': l2m.get('cosine', ''),
                        'L2_opt_l2_dist': l2m.get('l2_dist', ''),
                        'L2_top1_gene': l2.get('top_genes', [{}])[0].get('gene', '') if l2.get('top_genes') else '',
                        'L2_top1_pathway': l2.get('top_genes', [{}])[0].get('pathway', '') if l2.get('top_genes') else '',
                        'L2_top1_direction': l2.get('top_genes', [{}])[0].get('direction', '') if l2.get('top_genes') else '',
                    })
                pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

                all_l1a = []
                for tp in START_TPS:
                    all_l1a.extend(layer1a_results.get(tp, []))
                if all_l1a:
                    col_order = ['timepoint', 'start_day', 'gene', 'pathway', 'perturb_type',
                                   'delta_r_vs_ctl', 'r_after_vs_ctl', 'r_baseline_vs_ctl',
                                   'r_vs_dis', 'rmse', 'r2', 'cosine_sim', 'l2_dist',
                                   'ctl_dis_diff_at_start']
                    df_l1a = pd.DataFrame(all_l1a)
                    df_l1a = df_l1a[[c for c in col_order if c in df_l1a.columns]]
                    df_l1a.to_excel(writer, sheet_name='L1A_SingleGene', index=False)

                cross_rows = []
                for tp in START_TPS:
                    for rank, r in enumerate(layer1a_results.get(tp, [])[:20], 1):
                        row = {'rank': rank}
                        row.update(r)
                        cross_rows.append(row)
                if cross_rows:
                    pd.DataFrame(cross_rows).to_excel(
                        writer, sheet_name='L1A_Top20_ByTimepoint', index=False)

                all_l1b = []
                for tp in START_TPS:
                    all_l1b.extend(layer1b_results.get(tp, []))
                if all_l1b:
                    pd.DataFrame(all_l1b).to_excel(
                        writer, sheet_name='L1B_PathwayPerturb', index=False)

                all_l1c = []
                for tp in START_TPS:
                    all_l1c.extend(layer1c_results.get(tp, []))
                if all_l1c:
                    col_order_c = ['timepoint', 'start_day', 'pathway_A', 'pathway_B',
                                     'n_genes_perturbed', 'delta_r_vs_ctl', 'r_after_vs_ctl',
                                     'r_baseline_vs_ctl', 'delta_r_A_alone', 'delta_r_B_alone',
                                     'synergy_score', 'r_vs_dis', 'rmse', 'r2', 'cosine_sim', 'l2_dist']
                    df_l1c = pd.DataFrame(all_l1c)
                    df_l1c = df_l1c[[c for c in col_order_c if c in df_l1c.columns]]
                    df_l1c.to_excel(writer, sheet_name='L1C_DualPathway', index=False)

                all_l2 = []
                for tp in START_TPS:
                    all_l2.extend(layer2_results.get(tp, {}).get('top_genes', []))
                if all_l2:
                    col_order_l2 = ['timepoint', 'start_day', 'rank', 'gene', 'pathway', 'direction',
                                      'delta_value', 'ctl_dis_diff_at_start',
                                      'delta_r_vs_ctl', 'r_after_vs_ctl', 'r_baseline_vs_ctl',
                                      'r_vs_dis', 'rmse', 'r2', 'cosine_sim', 'l2_dist']
                    df_l2 = pd.DataFrame(all_l2)
                    df_l2 = df_l2[[c for c in col_order_l2 if c in df_l2.columns]]
                    df_l2.to_excel(writer, sheet_name='L2_GradientTargets', index=False)

                try:
                    gene_set = set()
                    for tp in START_TPS:
                        for r in layer1a_results.get(tp, []):
                            gene_set.add(r['gene'])
                    cross_data = {'gene': sorted(gene_set)}
                    for tp in START_TPS:
                        dr_map = {r['gene']: r['delta_r_vs_ctl']
                                    for r in layer1a_results.get(tp, [])}
                        cross_data[f'delta_r_{tp}'] = [
                            dr_map.get(g, np.nan) for g in cross_data['gene']]
                    pw_map = {}
                    for tp in START_TPS:
                        for r in layer1a_results.get(tp, []):
                            pw_map[r['gene']] = r['pathway']
                    cross_data['pathway'] = [pw_map.get(g, 'Other') for g in cross_data['gene']]
                    df_cross = pd.DataFrame(cross_data)
                    dr_cols = [c for c in df_cross.columns if c.startswith('delta_r_Day')]
                    df_cross['max_delta_r'] = df_cross[dr_cols].max(axis=1)
                    df_cross['best_timepoint'] = df_cross[dr_cols].idxmax(axis=1).str.replace('delta_r_', '')
                    df_cross.sort_values('max_delta_r', ascending=False, inplace=True)
                    df_cross.to_excel(writer, sheet_name='CrossTimepoint_GeneMatrix', index=False)
                except Exception as e:
                    print(f"  sheet CrossTimepoint failed: {e}")

            print(f"  Excel saved: {xl_path}")
        except Exception as e:
            print(f"  Excel save failed: {e}")
            traceback.print_exc()

    print("\n" + "=" * 65)
    print("LRRK2 Forward Perturbation Summary")
    print("=" * 65)
    print(f"  endpoint: Disease Day42 -> approach Control Day42 GT")
    print()
    for tp in START_TPS:
        if tp not in baseline:
            continue
        b = baseline[tp]
        l2 = layer2_results.get(tp, {})
        l2m = l2.get('metrics', {})
        print(f"  -- {tp} (day={TP_DAYS[tp]}) --")
        print(f"     baseline: r_vs_Ctl={b['r_vs_ctl']:.4f}  RMSE={b['rmse']:.4f}  cosine={b['cosine']:.4f}")
        if tp in ideal_preds:
            print(f"     ideal: r_vs_Ctl={ideal_preds[tp]['r_vs_ctl']:.4f}")
        if layer1a_results.get(tp):
            top1 = layer1a_results[tp][0]
            print(f"     L1A best: {top1['gene']:<12} dr={top1['delta_r_vs_ctl']:+.4f}  pathway={top1['pathway']}")
        if layer1b_results.get(tp):
            top1b = layer1b_results[tp][0]
            print(f"     L1B best pathway: {top1b['pathway']:<30} dr={top1b['delta_r_vs_ctl']:+.4f}")
        if l2m:
            delta_r = l2m['r_vs_ctl'] - b['r_vs_ctl']
            print(f"     L2 gradient opt: r {b['r_vs_ctl']:.4f} -> {l2m['r_vs_ctl']:.4f}  "
                  f"dr={delta_r:+.4f}  RMSE={l2m['rmse']:.4f}  cosine={l2m['cosine']:.4f}")
            if l2.get('top_genes'):
                tg = l2['top_genes'][0]
                print(f"     L2 Top-1 target: {tg['gene']:<12} delta={tg['delta_value']:+.4f} "
                      f"{'up' if tg['direction'] == 'up' else 'down'}  pathway={tg['pathway']}")
        print()

    print(f"  Output: {args.out_dir}")


if __name__ == '__main__':
    main()
