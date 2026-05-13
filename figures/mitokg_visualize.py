"""
MitoKG heterogeneous knowledge graph visualization.

Renders the full MitoKG graph (or a Top-N degree subset) as a publication-
quality figure with four node types (gene / pathway / drug / complex) and
five edge types (PPI / gene-pathway / drug-target / protein-complex /
pathway-hierarchy). Produces PNG, SVG, and PDF.

This script reads directly from the released MitoKG artifacts:
    hetero_graph.pt        (PyG HeteroData with all nodes and edges)
    node_id_maps.pkl       (node-index to biological identifier maps)
    gene_list.txt          (gene symbols)

Usage:
    # quick preview (about 600 nodes, 3-8 minutes)
    python mitokg_visualize.py --gnn-dir /path/to/MitoKG/gnn --out-dir ./out

    # full graph (2549 nodes, 23019 edges, 20-35 minutes)
    python mitokg_visualize.py --gnn-dir /path/to/MitoKG/gnn --out-dir ./out --full

Note: full-graph rendering is memory- and time-intensive; start with the
default preview mode to confirm the pipeline works.
"""

import argparse
import pickle
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


PREVIEW_CONFIG = dict(
    max_gene_nodes=380, max_pathway_nodes=95, max_drug_nodes=65, max_complex_nodes=65,
    max_ppi_edges=3500, max_gp_edges=1600, max_dt_edges=500,
    max_pc_edges=550, max_pp_edges=450,
    fig_width=24, fig_height=22,
    node_size_min=12, node_size_max=1000, size_power=0.68,
    edge_color_ppi="#3A7CB8", edge_width_ppi=0.35, edge_alpha_ppi=0.13,
    edge_color_gp="#D96B2D", edge_width_gp=0.60, edge_alpha_gp=0.28,
    edge_color_dt="#B03060", edge_width_dt=0.85, edge_alpha_dt=0.48,
    edge_color_pc="#2E7D52", edge_width_pc=0.80, edge_alpha_pc=0.42,
    edge_color_pp="#6A3FA0", edge_width_pp=0.65, edge_alpha_pp=0.35,
    arc_rad_ppi=0.07, arc_rad_gp=0.18, arc_rad_dt=0.25, arc_rad_pc=0.22, arc_rad_pp=0.15,
    spring_k_coarse=0.06, spring_iter_coarse=80,
    spring_k_fine=0.04, spring_iter_fine=100,
    label_font_size=5.8, label_cand_gene=120, label_cand_pathway=80,
    label_cand_drug=50, label_cand_complex=40, label_max_total=120,
    label_margin_x=0.055, label_margin_y=0.028,
    legend_font_size=12, title_font_size=16,
)

FULL_CONFIG = dict(
    max_gene_nodes=1132, max_pathway_nodes=815, max_drug_nodes=305, max_complex_nodes=297,
    max_ppi_edges=14763, max_gp_edges=5990, max_dt_edges=656,
    max_pc_edges=810, max_pp_edges=804,
    fig_width=36, fig_height=32,
    node_size_min=4, node_size_max=400, size_power=0.68,
    edge_color_ppi="#3A7CB8", edge_width_ppi=0.15, edge_alpha_ppi=0.04,
    edge_color_gp="#D96B2D", edge_width_gp=0.30, edge_alpha_gp=0.10,
    edge_color_dt="#B03060", edge_width_dt=0.50, edge_alpha_dt=0.25,
    edge_color_pc="#2E7D52", edge_width_pc=0.45, edge_alpha_pc=0.20,
    edge_color_pp="#6A3FA0", edge_width_pp=0.35, edge_alpha_pp=0.18,
    arc_rad_ppi=0.05, arc_rad_gp=0.12, arc_rad_dt=0.18, arc_rad_pc=0.15, arc_rad_pp=0.10,
    spring_k_coarse=0.025, spring_iter_coarse=40,
    spring_k_fine=0.018, spring_iter_fine=60,
    label_font_size=5.0, label_cand_gene=200, label_cand_pathway=120,
    label_cand_drug=80, label_cand_complex=60, label_max_total=150,
    label_margin_x=0.04, label_margin_y=0.02,
    legend_font_size=14, title_font_size=18,
)

CMAP_GENE = "Blues";    CMAP_GENE_LO = 0.42;    CMAP_GENE_HI = 0.92
CMAP_COMPLEX = "GnBu";  CMAP_COMPLEX_LO = 0.42; CMAP_COMPLEX_HI = 0.88
CMAP_PATHWAY = "YlOrRd"; CMAP_PATHWAY_LO = 0.28; CMAP_PATHWAY_HI = 0.78
CMAP_DRUG = "RdPu";     CMAP_DRUG_LO = 0.35;    CMAP_DRUG_HI = 0.82
ALPHA_GENE = 0.85
ALPHA_PATHWAY = 0.88
ALPHA_DRUG = 0.90
ALPHA_COMPLEX = 0.88
PERIPHERY_R_THRESH = 0.55
ELLIPSE_ASPECT = 1.1
CLIP_R_PERCENT = 90
EDGE_PULL_BACK = 0.70
EDGE_PULL_DEG = 3
PATHWAY_PERIPH_THRESH = 0.45
PATHWAY_R_MIN_SCALE = 0.20
PATHWAY_R_MAX_SCALE = 0.85
PATHWAY_ANGLE_JITTER = 0.60
LABEL_FONT_FAMILY = "Times New Roman"
LABEL_COLOR = "#1a1a1a"
LAYOUT_SEED = 42
PNG_DPI = 300
PDF_DPI = 300


def parse_args():
    p = argparse.ArgumentParser(description='MitoKG graph visualization')
    p.add_argument('--gnn-dir', type=Path, required=True,
                    help='MitoKG gnn/ directory (hetero_graph.pt, node_id_maps.pkl, gene_list.txt)')
    p.add_argument('--out-dir', type=Path, default=Path('./output_visualization'))
    p.add_argument('--full', action='store_true',
                    help='render the full graph (2549 nodes, 20-35 minutes). Default is preview mode.')
    p.add_argument('--seed', type=int, default=LAYOUT_SEED)
    return p.parse_args()


def load_graph_from_hetero_pt(gnn_dir: Path):
    """Load nodes and edges from MitoKG released hetero_graph.pt."""
    print("  loading hetero_graph.pt...")
    hetero = torch.load(str(gnn_dir / "hetero_graph.pt"),
                          map_location='cpu', weights_only=False)
    with open(str(gnn_dir / "node_id_maps.pkl"), "rb") as f:
        node_maps = pickle.load(f)

    gene_list_path = gnn_dir / "gene_list.txt"
    if not gene_list_path.exists():
        gene_list_path = gnn_dir / "gene_list"
    with open(str(gene_list_path), encoding='utf-8') as f:
        gene_symbols = [l.strip() for l in f if l.strip()]

    id2uniprot = {v: k for k, v in node_maps['gene'].items()}
    id2pathway = {v: k for k, v in node_maps['pathway'].items()}
    id2drug = {v: k for k, v in node_maps['drug'].items()}
    id2complex = {v: k for k, v in node_maps['complex'].items()}

    uniprot2symbol = {}
    for sym, idx in zip(gene_symbols, range(len(gene_symbols))):
        if idx in id2uniprot:
            uniprot2symbol[id2uniprot[idx]] = sym

    G = nx.Graph()

    for idx, uniprot in id2uniprot.items():
        node_id = f"gene_{uniprot}"
        G.add_node(node_id, node_type="gene",
                     label=uniprot2symbol.get(uniprot, uniprot))
    for idx, pw_id in id2pathway.items():
        node_id = f"pathway_{pw_id}"
        G.add_node(node_id, node_type="pathway", label=pw_id)
    for idx, drug_id in id2drug.items():
        node_id = f"drug_{drug_id}"
        G.add_node(node_id, node_type="drug", label=drug_id)
    for idx, corum_id in id2complex.items():
        node_id = f"complex_{corum_id}"
        G.add_node(node_id, node_type="complex", label=f"CORUM_{corum_id}")

    edge_type_map = {
        ('gene', 'interacts', 'gene'): ("gene_", "gene_", "ppi"),
        ('gene', 'involved_in', 'pathway'): ("gene_", "pathway_", "gene_pathway"),
        ('drug', 'targets', 'gene'): ("drug_", "gene_", "drug_target"),
        ('gene', 'member_of', 'complex'): ("gene_", "complex_", "protein_complex"),
        ('pathway', 'has_child', 'pathway'): ("pathway_", "pathway_", "pathway_hier"),
    }

    id_lookups = {
        "gene_": id2uniprot,
        "pathway_": id2pathway,
        "drug_": id2drug,
        "complex_": id2complex,
    }

    for et, (src_prefix, dst_prefix, etype_label) in edge_type_map.items():
        if et not in hetero.edge_types:
            continue
        ei = hetero[et].edge_index.cpu().numpy()
        src_lookup = id_lookups[src_prefix]
        dst_lookup = id_lookups[dst_prefix]
        count = 0
        for s_idx, d_idx in zip(ei[0], ei[1]):
            s_idx = int(s_idx)
            d_idx = int(d_idx)
            if s_idx not in src_lookup or d_idx not in dst_lookup:
                continue
            s = f"{src_prefix}{src_lookup[s_idx]}"
            d = f"{dst_prefix}{dst_lookup[d_idx]}"
            if s != d and G.has_node(s) and G.has_node(d):
                G.add_edge(s, d, edge_type=etype_label)
                count += 1
        print(f"  {etype_label}: {count} edges")

    return G


def top_by_type(G, deg, ntype, max_n):
    ns = sorted([n for n, d in G.nodes(data=True) if d.get("node_type") == ntype],
                  key=lambda x: deg.get(x, 0), reverse=True)
    return set(ns[:max_n])


def filter_top_n(G, cfg):
    """Keep top-N nodes by degree of each type."""
    deg_full = dict(G.degree())
    keep = (top_by_type(G, deg_full, "gene", cfg['max_gene_nodes']) |
              top_by_type(G, deg_full, "pathway", cfg['max_pathway_nodes']) |
              top_by_type(G, deg_full, "drug", cfg['max_drug_nodes']) |
              top_by_type(G, deg_full, "complex", cfg['max_complex_nodes']))
    G.remove_nodes_from([n for n in list(G.nodes()) if n not in keep])
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"  after filter: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_layout(G, cfg, seed):
    print("  computing layout...")
    np.random.seed(seed)
    random.seed(seed)

    deg_sub = dict(G.degree())
    nodes_list = list(G.nodes())
    N = len(nodes_list)

    degs_arr = np.array([deg_sub.get(n, 1) for n in nodes_list], dtype=float)
    deg_norm = degs_arr / degs_arr.max()
    r_init = (1.0 - deg_norm ** 0.5) + np.random.uniform(0, 0.15, N)
    type_angle_offset = {"gene": 0.0, "complex": 1.57, "pathway": 3.14, "drug": 4.71}
    angles = np.array([
        type_angle_offset.get(G.nodes[n].get("node_type", "gene"), 0.0)
        + np.random.uniform(0, 1.55)
        for n in nodes_list
    ])
    pos_init = {n: (float(r_init[i] * np.cos(angles[i])),
                      float(r_init[i] * np.sin(angles[i])))
                  for i, n in enumerate(nodes_list)}

    print("  spring coarse pass...")
    pos = nx.spring_layout(G, k=cfg['spring_k_coarse'],
                              iterations=cfg['spring_iter_coarse'],
                              pos=pos_init, seed=seed, weight=None)
    print("  spring fine pass...")
    pos = nx.spring_layout(G, k=cfg['spring_k_fine'],
                              iterations=cfg['spring_iter_fine'],
                              pos=pos, seed=seed + 1, weight=None)

    coords = np.array([[pos[n][0], pos[n][1]] for n in nodes_list])
    centroid = coords.mean(axis=0)
    coords -= centroid
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    coords_rot = coords @ eigvecs
    x_scale = np.percentile(np.abs(coords_rot[:, 0]), CLIP_R_PERCENT) or 1.0
    y_scale = np.percentile(np.abs(coords_rot[:, 1]), CLIP_R_PERCENT) or 1.0
    coords_norm = np.stack([coords_rot[:, 0] / x_scale, coords_rot[:, 1] / y_scale], axis=1)
    r_norm_arr = np.sqrt(coords_norm[:, 0] ** 2 + coords_norm[:, 1] ** 2)
    th_arr = np.arctan2(coords_norm[:, 1], coords_norm[:, 0])
    r_soft = np.minimum(r_norm_arr, 1.0) ** 0.88
    x_final = r_soft * np.cos(th_arr) * x_scale * ELLIPSE_ASPECT
    y_final = r_soft * np.sin(th_arr) * y_scale
    coords_final = np.stack([x_final, y_final], axis=1) @ eigvecs.T
    for i, n in enumerate(nodes_list):
        pos[n] = (float(coords_final[i, 0]), float(coords_final[i, 1]))

    for n, (x, y) in list(pos.items()):
        if deg_sub.get(n, 0) <= EDGE_PULL_DEG:
            pos[n] = (x * (1 - EDGE_PULL_BACK), y * (1 - EDGE_PULL_BACK))

    all_pos_arr = np.array([[pos[n][0], pos[n][1]] for n in nodes_list])
    r_all = np.sqrt(all_pos_arr[:, 0] ** 2 + all_pos_arr[:, 1] ** 2)
    r_max = np.percentile(r_all, 95) or 1.0
    node_r_norm = {n: float(r_all[i] / r_max) for i, n in enumerate(nodes_list)}

    print("  scattering peripheral pathway nodes...")
    rng = np.random.RandomState(seed + 777)
    for n in nodes_list:
        if G.nodes[n].get("node_type") != "pathway":
            continue
        rn = node_r_norm[n]
        if rn <= PATHWAY_PERIPH_THRESH:
            continue
        x, y = pos[n]
        r_cur = np.sqrt(x ** 2 + y ** 2)
        th_cur = np.arctan2(y, x)
        r_new = rng.uniform(r_cur * PATHWAY_R_MIN_SCALE,
                              r_cur * PATHWAY_R_MAX_SCALE)
        th_new = th_cur + rng.uniform(-PATHWAY_ANGLE_JITTER, PATHWAY_ANGLE_JITTER)
        pos[n] = (float(r_new * np.cos(th_new)),
                    float(r_new * np.sin(th_new)))

    all_pos_arr2 = np.array([[pos[n][0], pos[n][1]] for n in nodes_list])
    r_all2 = np.sqrt(all_pos_arr2[:, 0] ** 2 + all_pos_arr2[:, 1] ** 2)
    r_max2 = np.percentile(r_all2, 95) or 1.0
    node_r_norm = {n: float(r_all2[i] / r_max2) for i, n in enumerate(nodes_list)}

    return pos, node_r_norm, deg_sub


def render(G, pos, node_r_norm, deg_sub, cfg, out_dir: Path):
    print("  rendering...")
    fig, ax = plt.subplots(figsize=(cfg['fig_width'], cfg['fig_height']),
                               facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    all_x = [pos[n][0] for n in pos]
    all_y = [pos[n][1] for n in pos]
    pad = 0.10
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")
    x_range = (max(all_x) + pad) - (min(all_x) - pad)
    y_range = (max(all_y) + pad) - (min(all_y) - pad)

    def draw_edges_smart(edgelist, color, width, alpha, arc_rad):
        for u, v in edgelist:
            if u not in pos or v not in pos:
                continue
            ru = node_r_norm.get(u, 1.0)
            rv = node_r_norm.get(v, 1.0)
            if ru < PERIPHERY_R_THRESH and rv < PERIPHERY_R_THRESH:
                ax.add_patch(FancyArrowPatch(
                    pos[u], pos[v],
                    connectionstyle=f"arc3,rad={arc_rad}",
                    color=color, linewidth=width, alpha=alpha,
                    arrowstyle="-", zorder=1))
            else:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                          color=color, linewidth=width, alpha=alpha, zorder=1)

    for etype, ec, ew, ea, rad in [
        ("pathway_hier", cfg['edge_color_pp'], cfg['edge_width_pp'], cfg['edge_alpha_pp'], cfg['arc_rad_pp']),
        ("drug_target", cfg['edge_color_dt'], cfg['edge_width_dt'], cfg['edge_alpha_dt'], cfg['arc_rad_dt']),
        ("protein_complex", cfg['edge_color_pc'], cfg['edge_width_pc'], cfg['edge_alpha_pc'], cfg['arc_rad_pc']),
        ("gene_pathway", cfg['edge_color_gp'], cfg['edge_width_gp'], cfg['edge_alpha_gp'], cfg['arc_rad_gp']),
        ("ppi", cfg['edge_color_ppi'], cfg['edge_width_ppi'], cfg['edge_alpha_ppi'], cfg['arc_rad_ppi']),
    ]:
        elist = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == etype]
        print(f"    drawing {etype}: {len(elist)} edges...")
        draw_edges_smart(elist, ec, ew, ea, rad)

    max_deg = max(deg_sub.values()) if deg_sub else 1

    def get_colors(nodes_t, cmap_name, lo, hi):
        cmap = cm.get_cmap(cmap_name)
        degs = np.array([deg_sub.get(n, 1) for n in nodes_t], dtype=float)
        norm = (degs - degs.min()) / (degs.max() - degs.min() + 1e-8)
        return [cmap(lo + v * (hi - lo)) for v in norm]

    def deg2size(n):
        r = (max(deg_sub.get(n, 1), 1) / max_deg) ** cfg['size_power']
        return cfg['node_size_min'] + r * (cfg['node_size_max'] - cfg['node_size_min'])

    legend_colors = {}
    for ntype, cmap_name, lo, hi, alpha in [
        ("gene", CMAP_GENE, CMAP_GENE_LO, CMAP_GENE_HI, ALPHA_GENE),
        ("complex", CMAP_COMPLEX, CMAP_COMPLEX_LO, CMAP_COMPLEX_HI, ALPHA_COMPLEX),
        ("pathway", CMAP_PATHWAY, CMAP_PATHWAY_LO, CMAP_PATHWAY_HI, ALPHA_PATHWAY),
        ("drug", CMAP_DRUG, CMAP_DRUG_LO, CMAP_DRUG_HI, ALPHA_DRUG),
    ]:
        ns = [n for n, d in G.nodes(data=True) if d.get("node_type") == ntype]
        if not ns:
            continue
        legend_colors[ntype] = cm.get_cmap(cmap_name)((lo + hi) / 2)
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns],
                     s=[deg2size(n) for n in ns],
                     c=get_colors(ns, cmap_name, lo, hi),
                     alpha=alpha, linewidths=0, zorder=3)

    print("  placing labels...")
    char_w_data = cfg['label_font_size'] / 72.0 / cfg['fig_width'] * x_range * 0.62
    char_h_data = cfg['label_font_size'] / 72.0 / cfg['fig_height'] * y_range * 1.20

    def label_bbox(x, y, text):
        w = len(text) * char_w_data + cfg['label_margin_x'] * char_w_data
        h = char_h_data + cfg['label_margin_y'] * char_h_data
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def boxes_overlap(b1, b2):
        return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])

    candidates = []
    for ntype, cand_n in [("gene", cfg['label_cand_gene']),
                              ("pathway", cfg['label_cand_pathway']),
                              ("drug", cfg['label_cand_drug']),
                              ("complex", cfg['label_cand_complex'])]:
        ns = sorted([n for n, d in G.nodes(data=True) if d.get("node_type") == ntype],
                      key=lambda x: deg_sub.get(x, 0), reverse=True)
        for n in ns[:cand_n]:
            candidates.append((deg_sub.get(n, 0), n))
    candidates.sort(key=lambda x: x[0], reverse=True)

    placed_boxes = []
    placed_count = 0
    for _, n in candidates:
        if placed_count >= cfg['label_max_total']:
            break
        x, y = pos[n]
        lbl = G.nodes[n].get("label", n)
        lbl = (lbl[:15] + "...") if len(lbl) > 16 else lbl
        bbox = label_bbox(x, y, lbl)
        if any(boxes_overlap(bbox, pb) for pb in placed_boxes):
            continue
        is_hub = deg_sub.get(n, 0) > max_deg * 0.25
        ax.text(x, y, lbl,
                  fontsize=cfg['label_font_size'] + (0.8 if is_hub else 0),
                  fontfamily=LABEL_FONT_FAMILY, color=LABEL_COLOR,
                  ha="center", va="center",
                  fontweight="bold" if is_hub else "normal", zorder=5)
        placed_boxes.append(bbox)
        placed_count += 1
    print(f"  labels placed: {placed_count}/{len(candidates)} candidates")

    n_cnt = {t: len([n for n, d in G.nodes(data=True) if d.get("node_type") == t])
               for t in ("gene", "pathway", "drug", "complex")}
    node_handles = [
        mpatches.Patch(color=legend_colors.get("gene", "#3A86FF"),
                         label=f"Gene / Protein  (n={n_cnt['gene']})"),
        mpatches.Patch(color=legend_colors.get("complex", "#70C14C"),
                         label=f"Complex  (n={n_cnt['complex']})"),
        mpatches.Patch(color=legend_colors.get("pathway", "#FB8C00"),
                         label=f"Pathway  (n={n_cnt['pathway']})"),
        mpatches.Patch(color=legend_colors.get("drug", "#C9184A"),
                         label=f"Drug  (n={n_cnt['drug']})"),
    ]
    edge_handles = [
        Line2D([0], [0], color=cfg['edge_color_ppi'], lw=2.0, alpha=0.85, label="PPI Interaction"),
        Line2D([0], [0], color=cfg['edge_color_gp'], lw=2.0, alpha=0.85, label="Gene-Pathway"),
        Line2D([0], [0], color=cfg['edge_color_dt'], lw=2.0, alpha=0.85, label="Drug-Target"),
        Line2D([0], [0], color=cfg['edge_color_pc'], lw=2.0, alpha=0.85, label="Protein-Complex"),
        Line2D([0], [0], color=cfg['edge_color_pp'], lw=2.0, alpha=0.85, label="Pathway Hierarchy"),
    ]
    ax.legend(handles=node_handles + edge_handles, loc="lower left",
                fontsize=cfg['legend_font_size'], framealpha=0.93,
                edgecolor="#cccccc", facecolor="white",
                prop={"family": LABEL_FONT_FAMILY, "size": cfg['legend_font_size']})

    ax.set_title(
        "MitoKG: Mitochondria-Centered Heterogeneous Knowledge Graph\n"
        f"({G.number_of_nodes()} nodes  .  {G.number_of_edges()} edges  .  "
        "4 node types  .  5 edge types)",
        fontsize=cfg['title_font_size'], fontfamily=LABEL_FONT_FAMILY,
        color="#111111", pad=12, fontweight="bold")

    print("  saving files...")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "mitokg_visualization"
    fig.savefig(str(base) + ".svg", format="svg", bbox_inches="tight", facecolor="white")
    print(f"    saved: {base}.svg")
    fig.savefig(str(base) + ".png", format="png", bbox_inches="tight",
                  facecolor="white", dpi=PNG_DPI)
    print(f"    saved: {base}.png")
    fig.savefig(str(base) + ".pdf", format="pdf", bbox_inches="tight",
                  facecolor="white", dpi=PDF_DPI)
    print(f"    saved: {base}.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    cfg = FULL_CONFIG if args.full else PREVIEW_CONFIG

    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("  MitoKG Knowledge Graph Visualization")
    print("  Mode:", "FULL (20-35 min)" if args.full else "PREVIEW (3-8 min)")
    print("=" * 60)

    G = load_graph_from_hetero_pt(args.gnn_dir)
    print(f"  raw graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    filter_top_n(G, cfg)
    if G.number_of_nodes() == 0:
        raise SystemExit("ERROR: no nodes remain after filtering.")

    pos, node_r_norm, deg_sub = compute_layout(G, cfg, args.seed)
    render(G, pos, node_r_norm, deg_sub, cfg, args.out_dir)

    print(f"\n  done -> {args.out_dir}")


if __name__ == "__main__":
    main()
