"""
LRRK2 reverse virtual perturbation across time windows.

Goal: starting from Control cells at Day0 / Day10 / Day14, apply perturbations
and predict under Disease condition at Day42. Evaluate how closely the
perturbed Control trajectory approaches the Disease Day42 ground truth.
Identifies "which genes, when perturbed in healthy state, drive the trajectory
toward disease" - complementary to forward perturbation.

The prediction condition is set to Disease (cond_dis=1) for symmetry with
the forward experiment:
    forward:  Disease start + Disease cond + replace genes with Control mean -> approach Control GT
    reverse:  Control start + Disease cond + replace genes with Disease mean -> approach Disease GT

Pipeline:
    Layer 1A : single-gene scan, both replace (set to Disease mean) and knockout (set to zero)
    Layer 1B : pathway-level cascade perturbation
    Layer 1C : two-pathway combined perturbation
    Layer 2  : gradient-guided optimal perturbation
    Known PD genes (PINK1/LRRK2/PRKN/PARK7/SNCA/HTRA2/TOMM20/TOMM40) are always
    reported regardless of rank.

Usage:
    python perturbation_reverse.py \\
        --data-dir /path/to/data/lrrk2 \\
        --gnn-dir  /path/to/MitoKG/gnn \\
        --weight   /path/to/best_by_r_lrrk2.pt \\
        --out-dir  ./output_perturbation_reverse
"""

import argparse
import random
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torchdiffeq import odeint

warnings.filterwarnings('ignore')

from model import JTLatentODE_v6

try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment
    XLSX_OK = True
except ImportError:
    print("[warning] openpyxl not installed, Excel export skipped")
    XLSX_OK = False


SEED = 99
TRAIN_RATIO = 0.70
MAX_DAYS = 42.0

TOP_SCAN_GENES = 984
TOP_SHOW = 30

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
TARGET_DAY = 42.0

KNOWN_PD_GENES = [
    'PINK1', 'LRRK2', 'PRKN', 'PARK7', 'SNCA',
    'HTRA2', 'TOMM20', 'TOMM40',
]

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

PATHWAY_COLORS = {
    'Complex_I': '#6B4F8E',
    'Complex_II': '#D4717A',
    'Complex_III': '#7DADA0',
    'Complex_IV': '#8A89B0',
    'ATP_Synthase': '#2D2D42',
    'TCA_Cycle': '#8E6BAA',
    'Mitophagy_PINK1_LRRK2': '#7D5C7A',
    'Mito_Dynamics': '#5E6B8A',
    'Mito_Ribosome': '#3B3D6B',
    'Oxidative_Stress': '#B8909A',
    'Other': '#C8C4CC',
}


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


def save_fig(fig, base_path):
    """save PNG / SVG / PDF triple format"""
    base = str(base_path).replace('.png', '').replace('.svg', '').replace('.pdf', '')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(base + '.svg', bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    print(f"  [saved] {Path(base).name}  (.png / .svg / .pdf)")


def load_model(weight_path, n_genes, max_days, device):
    ckpt = torch.load(str(weight_path), map_location=device, weights_only=False)
    hp = ckpt.get('hp', {})
    for k, v in [('latent_dim', 32), ('hidden_dim', 256), ('gene_emb_dim', 64),
                   ('mask_ratio', 0.25), ('ode_hidden', 128), ('kg_ctx_dim', 32),
                   ('t_feat_dim', 16), ('dz_scale', 0.15), ('delta_scale', 0.20)]:
        hp.setdefault(k, v)
    m = JTLatentODE_v6(
        input_dim=n_genes, latent_dim=hp['latent_dim'], hidden_dim=hp['hidden_dim'],
        gene_emb_dim=hp['gene_emb_dim'], mask_ratio=hp['mask_ratio'],
        ode_hidden=hp['ode_hidden'], kg_ctx_dim=hp['kg_ctx_dim'],
        t_feat_dim=hp['t_feat_dim'], max_days=max_days,
        dz_scale=hp['dz_scale'], delta_scale=hp['delta_scale'],
        use_temporal_attention=True).to(device)
    m.decoder = VAEDecoder_Opt(
        hp['latent_dim'], n_genes, hp['hidden_dim'],
        hp['t_feat_dim'], hp['kg_ctx_dim']).to(device)
    m.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
    m.eval()
    print(f"  model loaded  parameters={sum(p.numel() for p in m.parameters()):,}")
    return m, hp


def build_pathway_idx(gene_names):
    gnl = [g.lower() for g in gene_names]
    pidx = {}
    for pw, genes in PATHWAY_GROUPS.items():
        idx = [gnl.index(g.lower()) for g in genes if g.lower() in gnl]
        if idx:
            pidx[pw] = idx
    return pidx


def get_gene_pathway(gene_name, gene_names, pathway_idx):
    gnl = gene_name.lower()
    for pw, idx_list in pathway_idx.items():
        for i in idx_list:
            if i < len(gene_names) and gene_names[i].lower() == gnl:
                return pw
    return 'Other'


def eval_r(pred, true):
    return float(pearsonr(pred.ravel(), true.ravel())[0])


def make_predict_fn(model, gene_emb_t, max_days, device):
    def _t_feat(day_scalar, B):
        d = torch.full((B,), day_scalar, dtype=torch.float32, device=device)
        return model.ode_func.t_encoder((d / max_days).unsqueeze(-1).clamp(0, 1))

    def _decode(z, day_scalar, cond_idx):
        B = z.size(0)
        t_feat = _t_feat(day_scalar, B)
        kg_ctx = model.ode_func.kg_proj(gene_emb_t.mean(0, keepdim=True)).expand(B, -1)
        cond_emb = model.ode_func.cond_emb(
            torch.full((B,), cond_idx, dtype=torch.long, device=device))
        return model.decoder(z, t_feat, kg_ctx, cond_emb)

    @torch.no_grad()
    def predict_from_expr(expr_np, start_day, target_day, cond_idx):
        x = torch.tensor(expr_np, dtype=torch.float32, device=device)
        days = torch.full((len(x),), start_day, dtype=torch.float32, device=device)
        mu, *_ = model.encode(x, gene_emb_t, time_days=days)
        z0 = mu.mean(0, keepdim=True)
        t_q = torch.tensor([start_day, target_day], dtype=torch.float32, device=device)
        model.ode_func.set_context(gene_emb_t)
        model.ode_func.set_condition(cond_idx)
        zt = odeint(model.ode_func, z0, t_q, method='rk4', options={'step_size': 0.5})
        return _decode(zt[1], target_day, cond_idx).cpu().numpy()[0]

    return predict_from_expr, _decode, _t_feat


def report_known_genes(layer1a_replace, layer1a_ko, gene_names, tag):
    """always print KNOWN_PD_GENES rankings regardless of position"""
    print(f"\n  [known PD genes forced report] {tag}")
    print(f"  {'gene':<10} {'Replace rank':>18} {'KO rank':>16}")
    print(f"  {'-' * 55}")

    rep_rank = {gn.lower(): (i + 1, dr)
                 for i, (gi, gn, dr, ra) in enumerate(layer1a_replace)}
    ko_rank = {gn.lower(): (i + 1, dr)
                for i, (gi, gn, dr, ra) in enumerate(layer1a_ko)}

    gnl = [g.lower() for g in gene_names]
    for gname in KNOWN_PD_GENES:
        gl = gname.lower()
        in_list = gl in gnl
        rep_info = rep_rank.get(gl, (None, None))
        ko_info = ko_rank.get(gl, (None, None))
        rep_str = (f"#{rep_info[0]:4d} ({rep_info[1]:+.4f})"
                    if rep_info[0] is not None else "  not in dataset")
        ko_str = (f"#{ko_info[0]:4d} ({ko_info[1]:+.4f})"
                   if ko_info[0] is not None else "  not in dataset")
        flag = "" if in_list else " [not in gene list]"
        print(f"  {gname:<10} {rep_str:>20} {ko_str:>18}{flag}")


def run_one_timewindow(
    predict_fn, decode_fn, model, gene_emb_t, gene_names, pathway_idx, kg_mask,
    ctl_start_expr, dis_start_expr,
    ctl_start_mean, dis_start_mean,
    start_day, target_day,
    true_dis_gt, true_ctl_gt,
    device, tag='',
):
    N_GENES = len(gene_names)
    print(f"\n  {'-' * 55}")
    print(f"  time window: {tag}")
    print(f"  Control start: {len(ctl_start_expr)} cells  "
          f"Disease start: {len(dis_start_expr)} cells")
    print(f"  prediction condition: Disease (cond_dis={COND_DIS})")

    baseline_pred = predict_fn(ctl_start_expr, start_day, target_day, COND_DIS)
    baseline_r_dis = eval_r(baseline_pred, true_dis_gt)
    baseline_r_ctl = eval_r(baseline_pred, true_ctl_gt)
    ideal_pred = predict_fn(dis_start_expr, start_day, target_day, COND_DIS)
    ideal_r_dis = eval_r(ideal_pred, true_dis_gt)
    ref_pred = predict_fn(ctl_start_expr, start_day, target_day, COND_CTL)
    ref_r_dis = eval_r(ref_pred, true_dis_gt)

    print(f"  baseline A (Ctl start + Dis cond) vs Dis GT: r={baseline_r_dis:.4f}")
    print(f"  baseline B (Ctl start + Ctl cond) vs Dis GT: r={ref_r_dis:.4f}")
    print(f"  ideal (Dis start + Dis cond) vs Dis GT: r={ideal_r_dis:.4f}")
    print(f"  baseline A vs Ctl GT: r={baseline_r_ctl:.4f}")

    res = {
        'baseline_r_dis': baseline_r_dis,
        'baseline_r_ctl': baseline_r_ctl,
        'ref_r_dis': ref_r_dis,
        'ideal_r_dis': ideal_r_dis,
    }

    diff = np.abs(dis_start_mean - ctl_start_mean)
    scan_idx = np.argsort(diff)[::-1][:TOP_SCAN_GENES]
    print(f"  [Layer1A] scanning {len(scan_idx)} differential genes (Replace + KO)...")

    layer1a_replace = []
    layer1a_ko = []
    for gi in scan_idx:
        gname = gene_names[gi] if gi < N_GENES else f"Gene_{gi}"
        x_rep = ctl_start_mean.copy()
        x_rep[gi] = dis_start_mean[gi]
        pred_rep = predict_fn(x_rep[np.newaxis, :], start_day, target_day, COND_DIS)
        r_rep = eval_r(pred_rep, true_dis_gt)
        dr_rep = r_rep - baseline_r_dis
        layer1a_replace.append((gi, gname, dr_rep, r_rep))

        x_ko = ctl_start_mean.copy()
        x_ko[gi] = 0.0
        pred_ko = predict_fn(x_ko[np.newaxis, :], start_day, target_day, COND_DIS)
        r_ko = eval_r(pred_ko, true_dis_gt)
        dr_ko = r_ko - baseline_r_dis
        layer1a_ko.append((gi, gname, dr_ko, r_ko))

    layer1a_replace.sort(key=lambda x: x[2], reverse=True)
    layer1a_ko.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top-5 disease-inducing genes (Replace): " +
          "  ".join(f"{gn}(dr={dr:+.4f})" for _, gn, dr, _ in layer1a_replace[:5]))
    print(f"  Top-5 disease-inducing genes (KO):      " +
          "  ".join(f"{gn}(dr={dr:+.4f})" for _, gn, dr, _ in layer1a_ko[:5]))

    res['layer1a_replace'] = layer1a_replace
    res['layer1a_ko'] = layer1a_ko
    report_known_genes(layer1a_replace, layer1a_ko, gene_names, tag)

    print(f"\n  [Layer1B] pathway cascade perturbation (Replace, Disease condition)...")
    layer1b = []
    for pw, idx in pathway_idx.items():
        x_pw = ctl_start_mean.copy()
        for i in idx:
            x_pw[i] = dis_start_mean[i]
        pred = predict_fn(x_pw[np.newaxis, :], start_day, target_day, COND_DIS)
        r_after = eval_r(pred, true_dis_gt)
        delta_r = r_after - baseline_r_dis
        layer1b.append((pw, len(idx), delta_r, r_after))
        print(f"    {pw:<30} n={len(idx):3d}  dr={delta_r:+.4f}  r_after={r_after:.4f}")
    layer1b.sort(key=lambda x: x[2], reverse=True)
    res['layer1b'] = layer1b

    if RUN_DUAL_PATHWAY and len(pathway_idx) >= 2:
        n_combos = len(list(combinations(pathway_idx.keys(), 2)))
        print(f"\n  [Layer1C] dual-pathway joint perturbation ({n_combos} combinations)...")
        layer1c = []
        for pw_a, pw_b in combinations(pathway_idx.keys(), 2):
            x_dual = ctl_start_mean.copy()
            for i in pathway_idx[pw_a]:
                x_dual[i] = dis_start_mean[i]
            for i in pathway_idx[pw_b]:
                x_dual[i] = dis_start_mean[i]
            pred = predict_fn(x_dual[np.newaxis, :], start_day, target_day, COND_DIS)
            r_after = eval_r(pred, true_dis_gt)
            delta_r = r_after - baseline_r_dis
            dr_a = next((x[2] for x in layer1b if x[0] == pw_a), 0.0)
            dr_b = next((x[2] for x in layer1b if x[0] == pw_b), 0.0)
            synergy = delta_r - (dr_a + dr_b)
            layer1c.append((pw_a, pw_b,
                             len(pathway_idx[pw_a]) + len(pathway_idx[pw_b]),
                             delta_r, r_after, synergy))
        layer1c.sort(key=lambda x: x[3], reverse=True)
        print(f"  Top-3 combinations:")
        for pa, pb, ng, dr, ra, syn in layer1c[:3]:
            print(f"    {pa.replace('_', ' ')} + {pb.replace('_', ' ')}  "
                  f"dr={dr:+.4f}  synergy={syn:+.4f}")
        res['layer1c'] = layer1c

    print(f"\n  [Layer2] gradient-guided optimal perturbation ({L2_STEPS} steps)...")
    target_t = torch.tensor(true_dis_gt, dtype=torch.float32, device=device).unsqueeze(0)
    x_ctl_t = torch.tensor(ctl_start_mean, dtype=torch.float32, device=device)
    delta = torch.zeros(N_GENES, device=device, requires_grad=True)
    opt_l2 = torch.optim.Adam([delta], lr=L2_LR)
    best_loss = float('inf')
    best_delta = None
    loss_hist = []

    for step in range(L2_STEPS):
        opt_l2.zero_grad()
        x_pert = (x_ctl_t + delta).clamp(-5.0, 5.0).unsqueeze(0)
        days_in = torch.tensor([start_day], dtype=torch.float32, device=device)
        mu, logvar, z, mask, dz, _ = model.encode(x_pert, gene_emb_t, time_days=days_in)
        t_q = torch.tensor([start_day, target_day], dtype=torch.float32, device=device)
        model.ode_func.set_context(gene_emb_t)
        model.ode_func.set_condition(COND_DIS)
        zt = odeint(model.ode_func, mu, t_q, method='rk4', options={'step_size': 0.5})
        pred = decode_fn(zt[1], target_day, COND_DIS)
        recon = F.mse_loss(pred, target_t)
        sparse = delta.abs().mean()
        nonkg = (delta.abs() * (1.0 - kg_mask)).mean()
        loss = recon + L2_LAMBDA_SPARSE * sparse + L2_LAMBDA_KG * nonkg
        loss.backward()
        opt_l2.step()
        loss_hist.append(float(loss.item()))
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().clone()
        if step % 1000 == 0 or step == L2_STEPS - 1:
            with torch.no_grad():
                r_now = eval_r(pred.cpu().numpy()[0], true_dis_gt)
            print(f"    step {step:4d}  loss={loss.item():.4f}  r_vs_Dis={r_now:.4f}")

    delta_np = best_delta.cpu().numpy()
    x_opt = np.clip(ctl_start_mean + delta_np, -5.0, 5.0)
    pred_opt = predict_fn(x_opt[np.newaxis, :], start_day, target_day, COND_DIS)
    r_opt_dis = eval_r(pred_opt, true_dis_gt)
    print(f"  optimal perturbation: r_vs_Dis={r_opt_dis:.4f} "
          f"(dr={r_opt_dis - baseline_r_dis:+.4f})")

    top_idx = np.argsort(np.abs(delta_np))[::-1][:L2_TOP_N]
    top_genes_l2 = [
        (gene_names[i] if i < N_GENES else f"Gene_{i}",
         float(delta_np[i]),
         float(dis_start_mean[i] - ctl_start_mean[i]),
         get_gene_pathway(gene_names[i] if i < N_GENES else f"Gene_{i}",
                          gene_names, pathway_idx))
        for i in top_idx
    ]
    print(f"\n  Layer2 Top-{L2_TOP_N} targets (|delta| sorted):")
    for gn, d, diff_val, pw in top_genes_l2[:10]:
        print(f"    {gn:<14} delta={d:+.4f}{'up' if d > 0 else 'down'}  "
              f"Dis-Ctl diff={diff_val:+.4f}  [{pw}]")

    res['layer2'] = {
        'delta_np': delta_np, 'r_before': baseline_r_dis,
        'r_after': r_opt_dis, 'delta_r': r_opt_dis - baseline_r_dis,
        'top_genes': top_genes_l2, 'loss_hist': loss_hist,
    }
    return res


def plot_layer1a_compare(layer1a_replace, layer1a_ko, gene_names, pathway_idx,
                          baseline_r_dis, title_en, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    for ax, layer1a, label in [
        (axes[0], layer1a_replace, 'Replace (set to Disease mean)'),
        (axes[1], layer1a_ko, 'Knockout (set to zero)'),
    ]:
        top = layer1a[:TOP_SHOW]
        names = [x[1] for x in top]
        deltas = [x[2] for x in top]
        colors = [PATHWAY_COLORS.get(
                    get_gene_pathway(n, gene_names, pathway_idx),
                    PATHWAY_COLORS['Other']) for n in names]
        ylabels = [f"* {n}" if n in KNOWN_PD_GENES else n for n in names]
        ax.barh(range(len(top)), deltas, color=colors, edgecolor='#444', height=0.7)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.axvline(0, color='#444', lw=0.8, ls='--')
        ax.set_xlabel('Delta r vs Disease GT', fontsize=11)
        ax.set_title(f'{label}\nBaseline r = {baseline_r_dis:.4f}',
                      fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.25)
        ax.tick_params(labelsize=9)

    patches = [Patch(color=c, label=p.replace('_', ' '))
                for p, c in PATHWAY_COLORS.items() if p != 'Other']
    axes[0].legend(handles=patches, fontsize=7.5, loc='lower right',
                    ncol=2, framealpha=0.8)
    fig.suptitle(title_en, fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)


def plot_layer1b(layer1b, title_en, save_path):
    if not layer1b:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    pw_names = [x[0].replace('_', ' ') for x in layer1b]
    pw_deltas = [x[2] for x in layer1b]
    pw_colors = [PATHWAY_COLORS.get(x[0], PATHWAY_COLORS['Other']) for x in layer1b]
    ax.barh(range(len(layer1b)), pw_deltas, color=pw_colors, edgecolor='#444', height=0.65)
    ax.set_yticks(range(len(layer1b)))
    ax.set_yticklabels(pw_names, fontsize=10)
    ax.axvline(0, color='#444', lw=0.8, ls='--')
    ax.set_xlabel('Delta r vs Disease GT', fontsize=11)
    ax.set_title(title_en, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25)
    ax.tick_params(labelsize=9)
    for i, (x, r) in enumerate(zip(pw_deltas, [x[3] for x in layer1b])):
        ax.text(x + 0.0005 if x >= 0 else x - 0.0005, i,
                 f'r={r:.4f}', va='center',
                 ha='left' if x >= 0 else 'right', fontsize=8)
    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)


def plot_timewindow_heatmap(tp_results_dict, gene_names, pathway_idx, save_path):
    all_top_genes = []
    for tp, res in tp_results_dict.items():
        if 'layer1a_replace' not in res:
            continue
        top_g = [gn for (gi, gn, dr, ra) in res['layer1a_replace'][:20] if dr > 0]
        for pg in KNOWN_PD_GENES:
            if pg not in top_g:
                top_g.append(pg)
        all_top_genes.extend(top_g)
    all_top_genes = list(dict.fromkeys(all_top_genes))[:45]
    if not all_top_genes:
        return

    tps = list(tp_results_dict.keys())
    data = np.full((len(all_top_genes), len(tps)), float('nan'))
    for j, tp in enumerate(tps):
        res = tp_results_dict[tp]
        if 'layer1a_replace' not in res:
            continue
        dr_map = {gn: dr for (gi, gn, dr, ra) in res['layer1a_replace']}
        for i, gn in enumerate(all_top_genes):
            data[i, j] = dr_map.get(gn, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(tps) * 2 + 2),
                                       max(8, len(all_top_genes) * 0.38 + 2)))
    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 0.1
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(tps)))
    ax.set_xticklabels(tps, fontsize=10)
    ax.set_yticks(range(len(all_top_genes)))
    yticklabels = []
    for gn in all_top_genes:
        pw = get_gene_pathway(gn, gene_names, pathway_idx)
        mark = "* " if gn in KNOWN_PD_GENES else "  "
        yticklabels.append(f"{mark}{gn} [{pw.replace('_', ' ')[:10]}]")
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_title('LRRK2 - Reverse Perturbation Time Window Comparison\n'
                  '(* = known PD causal gene)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Delta r vs Disease GT (Replace)')
    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)


def plot_known_genes_summary(tp_results, gene_names, pathway_idx, save_path):
    """heatmap summary of known PD causal genes across all time windows"""
    all_tps = list(tp_results.keys())
    n_genes = len(KNOWN_PD_GENES)
    n_tps = len(all_tps)
    if n_tps == 0:
        return

    data_rep = np.full((n_genes, n_tps), float('nan'))
    data_ko = np.full((n_genes, n_tps), float('nan'))

    for j, tp in enumerate(all_tps):
        res = tp_results[tp]
        dr_rep = {gn: dr for (gi, gn, dr, ra) in res.get('layer1a_replace', [])}
        dr_ko = {gn: dr for (gi, gn, dr, ra) in res.get('layer1a_ko', [])}
        for i, gname in enumerate(KNOWN_PD_GENES):
            data_rep[i, j] = dr_rep.get(gname, float('nan'))
            data_ko[i, j] = dr_ko.get(gname, float('nan'))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_tps * 1.5 + 4), max(5, n_genes * 0.5)))
    for ax, data, title in [(axes[0], data_rep, 'Replace (set to Disease mean)'),
                              (axes[1], data_ko, 'Knockout (set to zero)')]:
        if np.all(np.isnan(data)):
            vmax = 0.1
        else:
            vmax = np.nanmax(np.abs(data))
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_tps))
        ax.set_xticklabels(all_tps, fontsize=9, rotation=30)
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(KNOWN_PD_GENES, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Delta r')
    fig.suptitle('Known PD Causal Genes - Reverse Perturbation Summary',
                  fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)


def save_excel(tp_results, gene_names, pathway_idx, save_path):
    if not XLSX_OK:
        return

    rows_summary = []
    rows_replace = []
    rows_ko = []
    rows_pathway = []
    rows_l2 = []

    for tp, res in tp_results.items():
        rows_summary.append({
            'timepoint': tp,
            'baseline_r_dis': round(res['baseline_r_dis'], 5),
            'baseline_r_ctl': round(res['baseline_r_ctl'], 5),
            'ref_r_dis_ctl_cond': round(res['ref_r_dis'], 5),
            'ideal_r_dis': round(res['ideal_r_dis'], 5),
            'L2_r_after': round(res.get('layer2', {}).get('r_after', 0), 5),
            'L2_delta_r': round(res.get('layer2', {}).get('delta_r', 0), 5),
        })
        base = res['baseline_r_dis']
        for rank, (gi, gn, dr, ra) in enumerate(res.get('layer1a_replace', []), 1):
            rows_replace.append({
                'timepoint': tp, 'rank': rank, 'gene': gn,
                'pathway': get_gene_pathway(gn, gene_names, pathway_idx),
                'delta_r_vs_dis': round(dr, 5),
                'r_after_vs_dis': round(ra, 5),
                'baseline_r_vs_dis': round(base, 5),
                'is_known_pd_gene': gn in KNOWN_PD_GENES,
            })
        for rank, (gi, gn, dr, ra) in enumerate(res.get('layer1a_ko', []), 1):
            rows_ko.append({
                'timepoint': tp, 'rank': rank, 'gene': gn,
                'pathway': get_gene_pathway(gn, gene_names, pathway_idx),
                'delta_r_vs_dis': round(dr, 5),
                'r_after_vs_dis': round(ra, 5),
                'baseline_r_vs_dis': round(base, 5),
                'is_known_pd_gene': gn in KNOWN_PD_GENES,
            })
        for (pw, n_g, dr, ra) in res.get('layer1b', []):
            rows_pathway.append({
                'timepoint': tp, 'pathway': pw, 'n_genes': n_g,
                'delta_r_vs_dis': round(dr, 5),
                'r_after_vs_dis': round(ra, 5),
            })
        if 'layer2' in res:
            for rank, (gn, d, diff_val, pw) in enumerate(res['layer2']['top_genes'], 1):
                rows_l2.append({
                    'timepoint': tp, 'rank': rank, 'gene': gn,
                    'pathway': pw, 'delta_value': round(d, 5),
                    'direction': 'up' if d > 0 else 'down',
                    'dis_ctl_diff': round(diff_val, 5),
                    'is_known_pd_gene': gn in KNOWN_PD_GENES,
                })

    with pd.ExcelWriter(str(save_path), engine='openpyxl') as writer:
        pd.DataFrame(rows_summary).to_excel(writer, sheet_name='Summary', index=False)
        if rows_replace:
            pd.DataFrame(rows_replace).to_excel(writer, sheet_name='L1A_Replace', index=False)
        if rows_ko:
            pd.DataFrame(rows_ko).to_excel(writer, sheet_name='L1A_Knockout', index=False)
        if rows_pathway:
            pd.DataFrame(rows_pathway).to_excel(writer, sheet_name='L1B_Pathway', index=False)
        if rows_l2:
            pd.DataFrame(rows_l2).to_excel(writer, sheet_name='L2_GradientTargets', index=False)

    print(f"  Excel saved: {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description='LRRK2 reverse virtual perturbation')
    p.add_argument('--data-dir', type=Path, required=True,
                    help='directory containing expr_scaled.npy and cell_meta.csv')
    p.add_argument('--gnn-dir', type=Path, required=True,
                    help='directory containing gene_embeddings.npy and gene_list.txt')
    p.add_argument('--weight', type=Path, required=True,
                    help='path to trained MitoLODE checkpoint (best_by_r_lrrk2.pt)')
    p.add_argument('--out-dir', type=Path, default=Path('./output_perturbation_reverse'))
    p.add_argument('--seed', type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Seed: {args.seed}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.unicode_minus': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })

    print("\n" + "=" * 60)
    print("LRRK2 reverse perturbation (Control -> Disease trajectory)")
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
    te_expr = all_expr[te_idx]
    te_meta = all_meta.iloc[te_idx].reset_index(drop=True)

    N_GENES = all_expr.shape[1]

    gene_emb_t = torch.tensor(
        np.load(str(args.gnn_dir / "gene_embeddings.npy")).astype(np.float32)[:N_GENES],
        dtype=torch.float32).to(device)

    gene_list_path = args.gnn_dir / "gene_list.txt"
    if not gene_list_path.exists():
        gene_list_path = args.gnn_dir / "gene_list"
    with open(str(gene_list_path), encoding='utf-8') as f:
        gene_names = [l.strip() for l in f if l.strip()][:N_GENES]
    print(f"  gene_list: {len(gene_names)} genes")

    pathway_idx = build_pathway_idx(gene_names)
    kg_mask = torch.zeros(N_GENES, device=device)
    for idx_l in pathway_idx.values():
        for i in idx_l:
            kg_mask[i] = 1.0

    model, _ = load_model(args.weight, N_GENES, MAX_DAYS, device)
    predict_fn, decode_fn, _ = make_predict_fn(model, gene_emb_t, MAX_DAYS, device)

    def gm(expr, meta, tp, cond):
        m = ((meta['condition'] == cond) & (meta['timepoint'] == tp)).values
        return expr[m].mean(0) if m.sum() >= 2 else None

    true_dis_day42 = gm(te_expr, te_meta, 'Day42', DIS_KEY)
    true_ctl_day42 = gm(te_expr, te_meta, 'Day42', CTL_KEY)
    assert true_dis_day42 is not None and true_ctl_day42 is not None, \
        "Day42 GT missing"

    tp_results = {}
    for stp in START_TPS:
        sday = TP_DAYS[stp]
        dm = ((te_meta['condition'] == DIS_KEY) & (te_meta['timepoint'] == stp)).values
        cm = ((te_meta['condition'] == CTL_KEY) & (te_meta['timepoint'] == stp)).values
        de = te_expr[dm]
        ce = te_expr[cm]
        if len(de) < 2 or len(ce) < 2:
            print(f"  [skip] LRRK2 {stp}: not enough cells")
            continue

        res = run_one_timewindow(
            predict_fn=predict_fn, decode_fn=decode_fn,
            model=model, gene_emb_t=gene_emb_t,
            gene_names=gene_names, pathway_idx=pathway_idx, kg_mask=kg_mask,
            ctl_start_expr=ce, dis_start_expr=de,
            ctl_start_mean=ce.mean(0), dis_start_mean=de.mean(0),
            start_day=sday, target_day=TARGET_DAY,
            true_dis_gt=true_dis_day42, true_ctl_gt=true_ctl_day42,
            device=device,
            tag=f'LRRK2 Control {stp} -> Day42 (Disease condition)')
        tp_results[stp] = res

        if 'layer1a_replace' in res:
            plot_layer1a_compare(
                res['layer1a_replace'], res['layer1a_ko'],
                gene_names, pathway_idx, res['baseline_r_dis'],
                f'LRRK2 Reverse Perturbation Layer1A: Control {stp} -> Day42 (Disease condition)',
                args.out_dir / f"lrrk2_fig1_layer1a_{stp}")
        if 'layer1b' in res:
            plot_layer1b(
                res['layer1b'],
                f'LRRK2 Reverse Perturbation Layer1B: Pathway Cascade, Control {stp} -> Day42',
                args.out_dir / f"lrrk2_fig2_layer1b_{stp}")

    plot_timewindow_heatmap(tp_results, gene_names, pathway_idx,
                              args.out_dir / "lrrk2_fig3_timewindow")
    plot_known_genes_summary(tp_results, gene_names, pathway_idx,
                                args.out_dir / "lrrk2_fig4_known_pd_genes_summary")
    save_excel(tp_results, gene_names, pathway_idx,
                 args.out_dir / "reverse_perturbation_results.xlsx")

    print("\n" + "=" * 70)
    print("Reverse perturbation experiment complete.")
    print(f"  Output: {args.out_dir}")
    print("  Key outputs (each in .png / .svg / .pdf):")
    print("    lrrk2_fig1_layer1a_<tp>          Replace + KO side-by-side")
    print("    lrrk2_fig2_layer1b_<tp>          Pathway cascade")
    print("    lrrk2_fig3_timewindow            Time window heatmap")
    print("    lrrk2_fig4_known_pd_genes_summary  Known PD genes summary")
    print("    reverse_perturbation_results.xlsx  Full results")


if __name__ == '__main__':
    main()
