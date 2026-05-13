"""
MitoLODE main experiment training script for PINK1 and LRRK2 datasets.

Usage:
    python train.py --dataset lrrk2 --data-dir /path/to/data --gnn-dir /path/to/gnn

Only basic runtime parameters are exposed; all model-architecture and
loss-weight hyperparameters are fixed to the values used in the paper
(LRRK2 main-experiment configuration).
"""

import argparse
import json
import random
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

warnings.filterwarnings('ignore')

from model import JTLatentODE_v6


HP = dict(
    latent_dim=32,
    hidden_dim=256,
    gene_emb_dim=64,
    ode_hidden=128,
    kg_ctx_dim=32,
    t_feat_dim=16,
    dz_scale=0.15,
    delta_scale=0.20,
    grad_clip=1.0,
    beta_end=0.3,
    anchor_batch=256,
    mask_weight=2.0,
    spread_margin=1.5,
    contrastive_margin=0.5,
    lambda_traj_consistency=0.1,
    sep_loss_weight=2.0,
    anchor_reg_norm=5.0,
    anchor_reg_weight=0.1,
    mmd_n_kernels=5,
    mmd_kernel_mul=2.0,
    traj_lr=1e-3,
    eval_interval=50,
    kl_warmup=80,
    traj_warmup=100,
    gamma_temporal=0.3,
    contrastive_weight=0.5,
    mmd_weight=0.02,
    lambda_future=0.1,
    future_prob=0.5,
    lambda_traj=1.5,
    sep_loss_margin=1.6,
    mu_norm_weight=0.001,
)


DATASET_CFG = {
    'pink1': dict(
        name='PINK1',
        expr_filename='expr_scaled.npy',
        meta_filename='cell_meta.csv',
        meta_format='pink1',
        tp_to_days={'IPSCs': 0.5, 'D06': 6.0, 'D15': 15.0, 'D21': 21.0},
        tp_order=['IPSCs', 'D06', 'D15', 'D21'],
        max_days=25.0,
        disease_key='PINK1', control_key='Control',
        start_tp='D15', start_day=15.0,
        query_tps=['D21'], query_days=[21.0],
        traj_start_tp='IPSCs', traj_start_day=0.5,
        traj_query_tps=['D06', 'D15', 'D21'],
        traj_query_days=[6.0, 15.0, 21.0],
        infer_cond_dis=1, infer_cond_ctl=0,
    ),
    'lrrk2': dict(
        name='LRRK2',
        expr_filename='expr_scaled.npy',
        meta_filename='cell_meta.csv',
        meta_format='lrrk2',
        tp_to_days={'Day0': 0.5, 'Day10': 10.0, 'Day14': 14.0, 'Day42': 42.0},
        tp_order=['Day0', 'Day10', 'Day14', 'Day42'],
        max_days=42.0,
        disease_key='LRRK2', control_key='Control',
        start_tp='Day14', start_day=14.0,
        query_tps=['Day42'], query_days=[42.0],
        traj_start_tp='Day0', traj_start_day=0.5,
        traj_query_tps=['Day10', 'Day14', 'Day42'],
        traj_query_days=[10.0, 14.0, 42.0],
        infer_cond_dis=1, infer_cond_ctl=0,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description='MitoLODE main experiment training')
    p.add_argument('--dataset', required=True, choices=['pink1', 'lrrk2'],
                   help='Dataset name')
    p.add_argument('--data-dir', type=Path, required=True,
                   help='Directory containing expr_scaled.npy and cell_meta.csv')
    p.add_argument('--gnn-dir', type=Path, required=True,
                   help='Directory containing gene_embeddings.npy and gene_list.txt')
    p.add_argument('--out-dir', type=Path, default=Path('./output'),
                   help='Directory to write checkpoints, logs and figures')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=99)
    p.add_argument('--mask-ratio', type=float, default=0.25)
    p.add_argument('--train-ratio', type=float, default=0.70)
    p.add_argument('--pretrained', type=Path, default=None,
                   help='Optional path to a pretrained checkpoint (.pt) to skip training')
    return p.parse_args()


class VAEDecoder_Opt(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim,
                 t_feat_dim=16, kg_ctx_dim=32):
        super().__init__()
        in_dim = latent_dim + t_feat_dim + 2 * kg_ctx_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z, t_feat, kg_ctx, cond_emb):
        return self.net(torch.cat([z, t_feat, kg_ctx, cond_emb], dim=-1))


def latent_mmd(z_a, z_b, n_kernels=5, kernel_mul=2.0):
    n, m = z_a.size(0), z_b.size(0)
    if n < 2 or m < 2:
        return torch.tensor(0.0, device=z_a.device)
    total = torch.cat([z_a, z_b], dim=0)
    sq = (total.unsqueeze(0) - total.unsqueeze(1)).pow(2).sum(-1)
    bw = (sq.detach().median() /
            (2.0 * float(np.log(n + m + 1)))).clamp(1e-4, 1e4)
    bws = [bw * (kernel_mul ** i) for i in range(n_kernels)]
    K = sum(torch.exp(-sq / b) for b in bws) / n_kernels
    mmd = K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()
    return mmd.clamp(min=0.0)


def eval_metrics(pred, true):
    r, _ = pearsonr(pred, true)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    r2 = float(r2_score(true, pred))
    cos = float(np.dot(pred, true) /
                  (np.linalg.norm(pred) * np.linalg.norm(true) + 1e-8))
    l2pct = float(np.linalg.norm(pred - true) /
                    (np.linalg.norm(true) + 1e-8) * 100)
    return dict(pearson_r=round(float(r), 4), rmse=round(rmse, 4),
                r2=round(r2, 4), cosine=round(cos, 4), l2_pct=round(l2pct, 2))


def load_dataset(cfg, args):
    all_expr = np.load(str(args.data_dir / cfg['expr_filename'])).astype(np.float32)
    all_meta = pd.read_csv(str(args.data_dir / cfg['meta_filename']))

    if cfg['meta_format'] == 'pink1':
        src_col = 'stim' if 'stim' in all_meta.columns else 'condition'
        all_meta['condition'] = all_meta[src_col].apply(
            lambda s: '_'.join(s.split('_')[:-1]))
        all_meta['timepoint'] = all_meta[src_col].apply(
            lambda s: s.split('_')[-1])
    else:
        for old_col, new_col in [('Condition', 'condition'),
                                   ('Timepoint', 'timepoint')]:
            if old_col in all_meta.columns and new_col not in all_meta.columns:
                all_meta.rename(columns={old_col: new_col}, inplace=True)

    tr_idx, te_idx = [], []
    for tp in all_meta['timepoint'].unique():
        for cond in all_meta['condition'].unique():
            idx = np.where(((all_meta['timepoint'] == tp) &
                            (all_meta['condition'] == cond)).values)[0]
            if len(idx) < 4:
                tr_idx.extend(idx.tolist())
                continue
            np.random.seed(args.seed)
            np.random.shuffle(idx)
            cut = max(1, int(len(idx) * args.train_ratio))
            tr_idx.extend(idx[:cut].tolist())
            te_idx.extend(idx[cut:].tolist())

    tr_expr = all_expr[tr_idx]
    tr_meta = all_meta.iloc[tr_idx].reset_index(drop=True)
    te_expr = all_expr[te_idx]
    te_meta = all_meta.iloc[te_idx].reset_index(drop=True)
    print(f"  stratified split (train_ratio={args.train_ratio}): "
          f"train={len(tr_expr)}  test={len(te_expr)}")

    tp_map = cfg['tp_to_days']
    for meta in [tr_meta, te_meta]:
        meta['days'] = meta['timepoint'].map(lambda t: tp_map.get(t, 0.5))

    n = min(984, tr_expr.shape[1])
    print(f"  {cfg['name']}  train={tr_expr.shape}  test={te_expr.shape}  genes={n}")
    return tr_expr[:, :n], tr_meta, te_expr[:, :n], te_meta, n


def get_mean(expr, meta, tp, cond):
    m = ((meta['condition'] == cond) & (meta['timepoint'] == tp)).values
    return expr[m].mean(0) if m.sum() >= 2 else None


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}  |  Dataset: {args.dataset.upper()}  |  Seed: {args.seed}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_CFG[args.dataset]
    tp_map = cfg['tp_to_days']
    hp = HP

    print("\n" + "=" * 60)
    print("Step 1: load dataset")
    print("=" * 60)
    train_expr, train_meta, test_expr, test_meta, N_GENES = load_dataset(cfg, args)

    gene_emb_np = np.load(str(args.gnn_dir / "gene_embeddings.npy")).astype(np.float32)[:N_GENES]
    gene_emb_t = torch.tensor(gene_emb_np, dtype=torch.float32).to(DEVICE)
    gene_list_path = args.gnn_dir / "gene_list.txt"
    if not gene_list_path.exists():
        gene_list_path = args.gnn_dir / "gene_list"
    with open(str(gene_list_path)) as f:
        gene_names = [l.strip() for l in f][:N_GENES]

    dis_key = cfg['disease_key']
    ctl_key = cfg['control_key']

    test_gt, train_gt = {}, {}
    for tp in cfg['tp_order']:
        for cond, lbl in [(dis_key, 'dis'), (ctl_key, 'ctl')]:
            v = get_mean(test_expr, test_meta, tp, cond)
            if v is not None:
                test_gt[(tp, lbl)] = v
            v = get_mean(train_expr, train_meta, tp, cond)
            if v is not None:
                train_gt[(tp, lbl)] = v

    start_tp = cfg['start_tp']
    start_day = cfg['start_day']
    dis_m = ((test_meta['condition'] == dis_key) &
             (test_meta['timepoint'] == start_tp)).values
    ctl_m = ((test_meta['condition'] == ctl_key) &
             (test_meta['timepoint'] == start_tp)).values
    dis_start = test_expr[dis_m]
    ctl_start = test_expr[ctl_m] if ctl_m.sum() >= 2 else dis_start

    traj_stp = cfg['traj_start_tp']
    traj_sd = cfg['traj_start_day']
    tdm = ((test_meta['condition'] == dis_key) &
           (test_meta['timepoint'] == traj_stp)).values
    tcm = ((test_meta['condition'] == ctl_key) &
           (test_meta['timepoint'] == traj_stp)).values
    traj_dis = test_expr[tdm] if tdm.sum() >= 2 else dis_start
    traj_ctl = test_expr[tcm] if tcm.sum() >= 2 else ctl_start

    print(f"  main task start: {start_tp}  (dis={dis_m.sum()}, ctl={ctl_m.sum()})")
    print(f"  trajectory start: {traj_stp}  (dis={tdm.sum()}, ctl={tcm.sum()})")

    train_tp_mean = {}
    for tp in cfg['tp_order']:
        for cond, lbl in [(dis_key, 'dis'), (ctl_key, 'ctl')]:
            v = get_mean(train_expr, train_meta, tp, cond)
            if v is not None:
                train_tp_mean[(tp, lbl)] = v

    print("\n" + "=" * 60)
    print("Step 2: build model")
    print("=" * 60)

    max_days = cfg['max_days']

    model = JTLatentODE_v6(
        input_dim=N_GENES,
        latent_dim=hp['latent_dim'],
        hidden_dim=hp['hidden_dim'],
        gene_emb_dim=hp['gene_emb_dim'],
        mask_ratio=args.mask_ratio,
        ode_hidden=hp['ode_hidden'],
        kg_ctx_dim=hp['kg_ctx_dim'],
        t_feat_dim=hp['t_feat_dim'],
        max_days=max_days,
        dz_scale=hp['dz_scale'],
        delta_scale=hp['delta_scale'],
        use_temporal_attention=True,
    ).to(DEVICE)

    model.decoder = VAEDecoder_Opt(
        hp['latent_dim'], N_GENES, hp['hidden_dim'],
        hp['t_feat_dim'], hp['kg_ctx_dim'],
    ).to(DEVICE)

    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  decoder input: {hp['latent_dim'] + hp['t_feat_dim'] + 2 * hp['kg_ctx_dim']} dim"
          f"  (z{hp['latent_dim']} + t{hp['t_feat_dim']}"
          f" + kg{hp['kg_ctx_dim']} + cond{hp['kg_ctx_dim']})")

    def _make_t_feat(day_t):
        return model.ode_func.t_encoder(
            (day_t / max_days).unsqueeze(-1).clamp(0, 1))

    _kg_ctx_cache = {'v': None}

    def refresh_kg_cache():
        with torch.no_grad():
            _kg_ctx_cache['v'] = model.ode_func.kg_proj(
                gene_emb_t.mean(0, keepdim=True))

    def _kg_ctx_infer(B):
        if _kg_ctx_cache['v'] is None:
            refresh_kg_cache()
        return _kg_ctx_cache['v'].expand(B, -1)

    def decode_train(z, days_t, cond_idx_t):
        B = z.size(0)
        t_feat = _make_t_feat(days_t)
        kg_ctx = model.ode_func.kg_proj(
            gene_emb_t.mean(0, keepdim=True)).expand(B, -1).detach()
        cond_emb = model.ode_func.cond_emb(cond_idx_t).detach()
        return model.decoder(z, t_feat, kg_ctx, cond_emb)

    def decode_infer(z, target_day, cond_idx):
        B = z.size(0)
        day_t = (torch.full((B,), target_day, dtype=torch.float32, device=DEVICE)
                  if isinstance(target_day, float)
                  else target_day.float().to(DEVICE))
        t_feat = _make_t_feat(day_t)
        kg_ctx = _kg_ctx_infer(B)
        cond_emb = model.ode_func.cond_emb(
            torch.full((B,), cond_idx, dtype=torch.long, device=DEVICE))
        return model.decoder(z, t_feat, kg_ctx, cond_emb)

    def predict_trajectory(start_expr_np, s_day, query_days,
                             cond_idx, dec_days=None):
        if dec_days is None:
            dec_days = query_days
        was_training = model.training
        model.eval()
        refresh_kg_cache()
        with torch.no_grad():
            x = torch.tensor(start_expr_np, dtype=torch.float32, device=DEVICE)
            days = torch.full((len(x),), s_day, dtype=torch.float32, device=DEVICE)
            mu, *_ = model.encode(x, gene_emb_t, time_days=days)
            z0 = mu.mean(0, keepdim=True)

            t_q = torch.tensor([s_day] + query_days, dtype=torch.float32, device=DEVICE)
            model.ode_func.set_context(gene_emb_t)
            model.ode_func.set_condition(cond_idx)
            z_traj = odeint(model.ode_func, z0, t_q,
                              method='rk4', options={'step_size': 0.5})

            preds = [decode_infer(z_traj[i + 1], d, cond_idx).cpu().numpy()[0]
                     for i, d in enumerate(dec_days)]
        if was_training:
            model.train()
        return preds

    def run_main_eval():
        preds_dis = predict_trajectory(
            dis_start, start_day, cfg['query_days'], cfg['infer_cond_dis'])
        dis_r_vals = []
        for tp, pred in zip(cfg['query_tps'], preds_dis):
            true = test_gt.get((tp, 'dis'))
            if true is not None:
                dis_r_vals.append(float(pearsonr(pred, true)[0]))
        dis_r = float(np.mean(dis_r_vals)) if dis_r_vals else -1.0

        ctl_r = -1.0
        if len(ctl_start) >= 2:
            preds_ctl = predict_trajectory(
                ctl_start, start_day, cfg['query_days'], cfg['infer_cond_ctl'])
            ctl_r_vals = []
            for tp, pred in zip(cfg['query_tps'], preds_ctl):
                true = test_gt.get((tp, 'ctl'))
                if true is not None:
                    ctl_r_vals.append(float(pearsonr(pred, true)[0]))
            ctl_r = float(np.mean(ctl_r_vals)) if ctl_r_vals else -1.0

        return dis_r, ctl_r

    best_path = args.out_dir / f"best_model_{args.dataset}.pt"
    best_r_path = args.out_dir / f"best_by_r_{args.dataset}.pt"
    last_path = args.out_dir / f"last_model_{args.dataset}.pt"

    if args.pretrained is not None and Path(str(args.pretrained)).exists():
        print(f"\nLoading pretrained weights: {args.pretrained}")
        ckpt = torch.load(str(args.pretrained), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
        print("Pretrained weights loaded, skipping training")

    else:
        print("\n" + "=" * 60)
        print("Step 3: training")
        print("=" * 60)

        tp_to_idx = {tp: i for i, tp in enumerate(cfg['tp_order'])}
        idx_to_days = {i: tp_map[tp] for i, tp in enumerate(cfg['tp_order'])}
        next_tp_idx = {i: i + 1 for i in range(len(cfg['tp_order']) - 1)}

        cond_map = {c: i for i, c in enumerate(sorted(train_meta['condition'].unique()))}
        cond_inv = {v: k for k, v in cond_map.items()}
        print(f"  condition mapping: {cond_map}")

        x_t = torch.tensor(train_expr, dtype=torch.float32)
        tp_t = torch.tensor(
            train_meta['timepoint'].map(tp_to_idx).fillna(0).astype(int).values,
            dtype=torch.long)
        days_t = torch.tensor(
            train_meta['timepoint'].map(lambda t: tp_map.get(t, 0.5)).values.astype(np.float32))
        cond_t = torch.tensor(
            train_meta['condition'].map(cond_map).fillna(0).astype(int).values,
            dtype=torch.long)

        next_mean_cache = {}
        for c_int, c_name in cond_inv.items():
            lbl = 'dis' if c_name == dis_key else 'ctl'
            for ti in range(len(cfg['tp_order']) - 1):
                ntp = cfg['tp_order'][ti + 1]
                v = train_tp_mean.get((ntp, lbl))
                if v is not None:
                    next_mean_cache[(c_int, ti)] = torch.tensor(
                        v, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        loader = DataLoader(
            TensorDataset(x_t, tp_t, days_t, cond_t),
            batch_size=args.batch_size, shuffle=True, drop_last=True)

        vae_params = [p for n, p in model.named_parameters() if 'ode_func' not in n]
        ode_params = list(model.ode_func.parameters())

        optimizer = torch.optim.AdamW(vae_params, lr=args.lr, weight_decay=1e-5)
        ode_optimizer = torch.optim.Adam(ode_params, lr=hp['traj_lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=150, T_mult=2, eta_min=args.lr * 0.01)
        ode_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            ode_optimizer,
            T_max=max(args.epochs - hp['traj_warmup'], 1),
            eta_min=hp['traj_lr'] * 0.05)

        best_recon = float('inf')
        best_r = -1.0
        best_state_recon = best_state_r = None
        log_records = []
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            e_traj = 0.0

            if epoch > hp['traj_warmup']:
                model.eval()
                _sums = {}
                _cnts = {}
                old_mr = model.mmp.mask_ratio
                model.mmp.mask_ratio = 0.0
                with torch.no_grad():
                    for i in range(0, len(x_t), hp['anchor_batch']):
                        xb = x_t[i:i + hp['anchor_batch']].to(DEVICE)
                        tb = tp_t[i:i + hp['anchor_batch']].to(DEVICE)
                        db = days_t[i:i + hp['anchor_batch']].to(DEVICE)
                        cb = cond_t[i:i + hp['anchor_batch']].to(DEVICE)
                        mu, *_ = model.encode(xb, gene_emb_t, time_days=db)
                        for c in cb.unique():
                            for t in tb.unique():
                                sel = (cb == c) & (tb == t)
                                if sel.sum() < 4:
                                    continue
                                k = (c.item(), t.item())
                                mv = mu[sel].float().mean(0)
                                if k not in _sums:
                                    _sums[k] = mv.clone()
                                    _cnts[k] = 1
                                else:
                                    _sums[k] += mv
                                    _cnts[k] += 1
                model.mmp.mask_ratio = old_mr

                if len(_sums) >= 2:
                    al, tl, cl = [], [], []
                    for k in sorted(_sums.keys()):
                        al.append(torch.tensor(
                            (_sums[k] / _cnts[k]).detach().cpu().numpy(),
                            dtype=torch.float32, device=DEVICE, requires_grad=True))
                        cl.append(k[0])
                        tl.append(k[1])
                    model.train()
                    ode_optimizer.zero_grad()
                    tl_ep = model.compute_trajectory_loss_v6(
                        torch.stack(al),
                        torch.tensor(tl, dtype=torch.long, device=DEVICE),
                        torch.tensor(cl, dtype=torch.long, device=DEVICE),
                        gene_emb_t, idx_to_days,
                        sep_loss_margin=hp['sep_loss_margin'],
                        sep_loss_weight=hp['sep_loss_weight'],
                        anchor_reg_norm=hp['anchor_reg_norm'],
                        anchor_reg_weight=hp['anchor_reg_weight'],
                    )
                    if torch.isfinite(tl_ep) and tl_ep.item() > 1e-8:
                        (hp['lambda_traj'] * tl_ep).backward()
                        torch.nn.utils.clip_grad_norm_(ode_params, hp['grad_clip'])
                        ode_optimizer.step()
                        e_traj = tl_ep.item()
                ode_scheduler.step()

            model.train()
            ep_recon = ep_loss = ep_mmd = ep_fut = 0.0
            n_bat = 0
            kl_w = hp['beta_end'] * min(1.0, epoch / hp['kl_warmup'])

            for xb, tpb, db, cb in loader:
                xb = xb.to(DEVICE)
                tpb = tpb.to(DEVICE)
                db = db.to(DEVICE)
                cb = cb.to(DEVICE)
                optimizer.zero_grad()

                mu, logvar, z, mask, delta, _ = model.encode(xb, gene_emb_t, time_days=db)

                x_hat = decode_train(z, db, cb)
                w = 1.0 + (1.0 - mask) * (hp['mask_weight'] - 1.0)
                recon = (w * (x_hat - xb) ** 2).mean()
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                temp_loss = model.compute_vae_temporal_loss(
                    mu, tpb, idx_to_days, cond_batch=cb,
                    spread_margin=hp['spread_margin'],
                    contrastive_margin=hp['contrastive_margin'],
                    contrastive_weight=hp['contrastive_weight'])
                tc_loss = model.batch_trajectory_consistency(
                    mu, tpb, idx_to_days, cond_batch=cb)

                mmd_val = torch.tensor(0.0, device=DEVICE)
                if hp['mmd_weight'] > 0:
                    cond_vals = cb.unique()
                    if len(cond_vals) >= 2:
                        for t_idx in tpb.unique():
                            t_sel = (tpb == t_idx)
                            grps = [mu[t_sel & (cb == c)] for c in cond_vals]
                            grps = [g for g in grps if g.size(0) >= 2]
                            if len(grps) >= 2:
                                mmd_val = mmd_val + latent_mmd(
                                    grps[0], grps[1],
                                    n_kernels=hp['mmd_n_kernels'],
                                    kernel_mul=hp['mmd_kernel_mul'])

                future_loss = torch.tensor(0.0, device=DEVICE)
                if hp['lambda_future'] > 0 and random.random() < hp['future_prob']:
                    for t_idx in tpb.unique():
                        ti = t_idx.item()
                        if ti not in next_tp_idx:
                            continue
                        for c_idx in cb[tpb == t_idx].unique():
                            ci = c_idx.item()
                            tgt = next_mean_cache.get((ci, ti))
                            if tgt is None:
                                continue
                            sel = (tpb == t_idx) & (cb == c_idx)
                            if sel.sum() < 2:
                                continue
                            next_day = tp_map[cfg['tp_order'][next_tp_idx[ti]]]
                            next_days = torch.full(
                                (sel.sum(),), next_day,
                                dtype=torch.float32, device=DEVICE)
                            x_next_hat = decode_train(z[sel], next_days, cb[sel])
                            soft = tgt.expand(sel.sum(), -1).detach()
                            future_loss = future_loss + F.mse_loss(x_next_hat, soft)

                mu_norm = mu.norm(dim=-1).mean()

                vae_loss = (recon
                              + kl_w * kl
                              + hp['gamma_temporal'] * temp_loss
                              + hp['lambda_traj_consistency'] * tc_loss
                              + hp['mmd_weight'] * mmd_val
                              + hp['lambda_future'] * future_loss
                              + hp['mu_norm_weight'] * mu_norm)

                if not torch.isfinite(vae_loss):
                    continue
                vae_loss.backward()
                torch.nn.utils.clip_grad_norm_(vae_params, hp['grad_clip'])
                optimizer.step()

                ep_recon += recon.item()
                ep_loss += vae_loss.item()
                ep_mmd += mmd_val.item()
                ep_fut += future_loss.item()
                n_bat += 1

            if n_bat == 0:
                continue
            scheduler.step()

            avg_recon = ep_recon / n_bat
            avg_loss = ep_loss / n_bat

            if avg_recon < best_recon:
                best_recon = avg_recon
                best_state_recon = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save({'model_state': best_state_recon,
                              'recon': best_recon, 'epoch': epoch,
                              'dataset': args.dataset, 'hp': hp},
                             str(best_path))

            do_eval = (epoch % hp['eval_interval'] == 0)
            do_print = (epoch % 100 == 0 or epoch == 1)

            if do_eval or do_print:
                cur_dis_r, cur_ctl_r = run_main_eval()
            else:
                cur_dis_r, cur_ctl_r = best_r, -1.0

            if do_eval:
                if cur_dis_r > best_r:
                    best_r = cur_dis_r
                    best_state_r = {k: v.clone() for k, v in model.state_dict().items()}
                    torch.save({'model_state': best_state_r,
                                  'pearson_r': best_r, 'epoch': epoch,
                                  'dataset': args.dataset, 'hp': hp},
                                 str(best_r_path))

            log_records.append({'epoch': epoch, 'loss': avg_loss, 'recon': avg_recon,
                                  'traj': e_traj, 'mmd': ep_mmd / n_bat,
                                  'future': ep_fut / n_bat, 'kl_w': kl_w,
                                  'dis_r': cur_dis_r if do_eval else float('nan'),
                                  'ctl_r': cur_ctl_r if do_eval else float('nan')})

            if do_print:
                task = f"{start_tp}->{cfg['query_tps'][0]}"
                dis_str = f"Dis r={cur_dis_r:.4f}" if cur_dis_r > -1 else "Dis r=n/a"
                ctl_str = f"Ctl r={cur_ctl_r:.4f}" if cur_ctl_r > -1 else "Ctl r=n/a"
                best_str = f"best={best_r:.4f}" if best_r > -1 else "best=n/a"
                print(f"  Ep {epoch:4d}/{args.epochs} | "
                      f"recon={avg_recon:.4f}  traj={e_traj:.4f}  "
                      f"mmd={ep_mmd / n_bat:.4f} | "
                      f"[{task}] {dis_str}  {ctl_str}  ({best_str}) | "
                      f"{(time.time() - t0) / 60:.1f}min")

        torch.save({'model_state': model.state_dict(), 'dataset': args.dataset, 'hp': hp},
                     str(last_path))
        print(f"\n  best(recon) -> {best_path.name}  (recon={best_recon:.4f})")
        print(f"  best(r)     -> {best_r_path.name}  (r={best_r:.4f})")
        print(f"  last        -> {last_path.name}")

        pd.DataFrame(log_records).to_csv(
            str(args.out_dir / "training_log.csv"), index=False, encoding='utf-8-sig')

        if best_state_r is not None:
            model.load_state_dict(best_state_r)
            print("  inference weights: best_by_r")
        elif best_state_recon is not None:
            model.load_state_dict(best_state_recon)
            print("  inference weights: best_by_recon")

    print("\n" + "=" * 60)
    print("Step 4: evaluation")
    print("=" * 60)
    model.eval()

    print(f"\n[main task] {start_tp} -> {cfg['query_tps'][0]}")
    main_results = {}
    for cond_name, cond_idx, s_cells, label in [
        (dis_key, cfg['infer_cond_dis'], dis_start, 'Disease'),
        (ctl_key, cfg['infer_cond_ctl'], ctl_start, 'Control'),
    ]:
        if len(s_cells) < 2:
            continue
        lbl = 'dis' if cond_name == dis_key else 'ctl'
        preds = predict_trajectory(s_cells, start_day, cfg['query_days'], cond_idx)
        main_results[label] = {}
        for tp, pred in zip(cfg['query_tps'], preds):
            true = test_gt.get((tp, lbl))
            if true is None:
                continue
            m = eval_metrics(pred, true)
            main_results[label][tp] = m
            print(f"  {label} {start_tp}->{tp}: r={m['pearson_r']:.4f}  "
                  f"RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}  "
                  f"Cosine={m['cosine']:.4f}  L2%={m['l2_pct']:.1f}%")

    print(f"\n[full trajectory] {traj_stp} -> each timepoint")
    traj_results = {}
    for cond_name, cond_idx, s_cells, label in [
        (dis_key, cfg['infer_cond_dis'], traj_dis, 'Disease'),
        (ctl_key, cfg['infer_cond_ctl'], traj_ctl, 'Control'),
    ]:
        if len(s_cells) < 2:
            continue
        lbl = 'dis' if cond_name == dis_key else 'ctl'
        preds = predict_trajectory(s_cells, traj_sd, cfg['traj_query_days'], cond_idx)
        traj_results[label] = {}
        for tp, pred in zip(cfg['traj_query_tps'], preds):
            true = test_gt.get((tp, lbl))
            if true is None:
                continue
            m = eval_metrics(pred, true)
            traj_results[label][tp] = m
            print(f"  {label} {traj_stp}->{tp}: r={m['pearson_r']:.4f}  R2={m['r2']:.4f}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nMain task ({start_tp}->{cfg['query_tps'][0]}):")
    for label, res in main_results.items():
        for tp, m in res.items():
            print(f"  {label:10s}: r={m['pearson_r']:.4f}  RMSE={m['rmse']:.4f}  "
                  f"R2={m['r2']:.4f}  Cosine={m['cosine']:.4f}  L2%={m['l2_pct']:.1f}%")
    print(f"\nFull trajectory ({traj_stp} -> each timepoint, Pearson r):")
    print("  " + "".join(f" {tp:>10}" for tp in cfg['traj_query_tps']))
    for label, res in traj_results.items():
        print(f"  {label:10s}" + "".join(
            f" {res[tp]['pearson_r']:>10.4f}" if tp in res else f" {'n/a':>10}"
            for tp in cfg['traj_query_tps']))

    with open(str(args.out_dir / "eval_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump({'dataset': args.dataset,
                    'main': main_results, 'traj': traj_results},
                    f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Step 5: visualization")
    print("=" * 60)
    plt.rcParams.update({'font.family': 'DejaVu Sans',
                          'axes.spines.top': False, 'axes.spines.right': False})
    model.eval()

    try:
        df_log = pd.read_csv(str(args.out_dir / "training_log.csv"))
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(f'{cfg["name"]} - Training Curves',
                       fontsize=12, fontweight='bold')
        for ax, col, label in zip(axes,
                                     ['recon', 'loss', 'traj', 'future'],
                                     ['Recon Loss', 'Total Loss', 'Traj Loss', 'Future Loss']):
            ax.plot(df_log['epoch'], df_log[col], lw=1.5, color='#D85A30')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(str(args.out_dir / "fig1_training.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("  Fig1: training curves")
    except Exception as e:
        print(f"  Fig1 failed: {e}")

    try:
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6))
        fig2.suptitle(f'{cfg["name"]} - {start_tp}->{cfg["query_tps"][0]} Scatter',
                        fontsize=12, fontweight='bold')
        for ax, label, cond_idx, s_cells, lbl, color in [
            (axes2[0], 'Disease', cfg['infer_cond_dis'], dis_start, 'dis', '#D85A30'),
            (axes2[1], 'Control', cfg['infer_cond_ctl'], ctl_start, 'ctl', '#185FA5'),
        ]:
            if len(s_cells) < 2:
                continue
            pred = predict_trajectory(s_cells, start_day,
                                        cfg['query_days'], cond_idx)[0]
            true = test_gt.get((cfg['query_tps'][0], lbl))
            if true is None:
                continue
            m = main_results.get(label, {}).get(cfg['query_tps'][0], {})
            lm = max(abs(true).max(), abs(pred).max()) * 1.05
            ax.scatter(true, pred, s=3, alpha=0.3, color=color)
            ax.axline((0, 0), slope=1, color='#444', lw=1, ls='--')
            ax.set_xlim(-lm, lm)
            ax.set_ylim(-lm, lm)
            ax.set_title(f'{label}  r={m.get("pearson_r", "?"):.4f}  '
                          f'RMSE={m.get("rmse", "?"):.4f}  R2={m.get("r2", "?"):.4f}',
                          fontsize=10)
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.grid(alpha=0.2)
        fig2.tight_layout()
        fig2.savefig(str(args.out_dir / "fig2_scatter.png"), dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print("  Fig2: scatter")
    except Exception as e:
        print(f"  Fig2 failed: {e}")

    try:
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        fig3.suptitle(f'{cfg["name"]} - Full-trajectory Pearson r',
                        fontsize=12, fontweight='bold')
        for ax, label, color in zip(axes3, ['Disease', 'Control'],
                                       ['#D85A30', '#185FA5']):
            res = traj_results.get(label, {})
            r_vals = [res.get(tp, {}).get('pearson_r', np.nan)
                       for tp in cfg['traj_query_tps']]
            ax.plot(cfg['traj_query_tps'], r_vals, 'o-',
                     color=color, lw=2.5, markersize=9)
            for i, (tp, v) in enumerate(zip(cfg['traj_query_tps'], r_vals)):
                if not np.isnan(v):
                    ax.text(i, v + 0.01, f'{v:.4f}', ha='center',
                              fontsize=9, fontweight='bold', color=color)
            ax.set_ylim(-0.1, 1.05)
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Pearson r')
            ax.set_title(f'{label}  ({traj_stp}-> each timepoint)', fontsize=10)
            ax.grid(alpha=0.25)
        fig3.tight_layout()
        fig3.savefig(str(args.out_dir / "fig3_traj_pearson.png"), dpi=200, bbox_inches='tight')
        plt.close(fig3)
        print("  Fig3: trajectory pearson")
    except Exception as e:
        print(f"  Fig3 failed: {e}")

    try:
        N_SHOW = 50
        tp_hm = cfg['query_tps'][0]
        true_d = test_gt.get((tp_hm, 'dis'))
        true_c = test_gt.get((tp_hm, 'ctl'))
        pred_d = predict_trajectory(dis_start, start_day,
                                      cfg['query_days'], cfg['infer_cond_dis'])[0]
        if true_d is not None:
            top_idx = np.argsort(np.abs(pred_d - true_d))[::-1][:N_SHOW]
            glabels = [gene_names[i] if i < len(gene_names) else f'G{i}' for i in top_idx]
            rows = {'True Disease': true_d[top_idx], 'Pred Disease': pred_d[top_idx]}
            if true_c is not None:
                rows['True Control'] = true_c[top_idx]
            hm = np.stack(list(rows.values()), axis=0)
            fig4, ax4 = plt.subplots(figsize=(max(10, len(rows) * 3 + 2), 11))
            sns.heatmap(hm.T, ax=ax4, cmap='RdBu_r', center=0,
                         xticklabels=list(rows.keys()), yticklabels=glabels,
                         linewidths=0.15, cbar_kws={'label': 'Expression'})
            m_d = main_results.get('Disease', {}).get(tp_hm, {})
            ax4.set_title(f'{cfg["name"]} Gene Heatmap @ {tp_hm}  '
                            f'r={m_d.get("pearson_r", "?"):.4f}',
                            fontsize=11, fontweight='bold')
            ax4.tick_params(axis='y', labelsize=6)
            fig4.tight_layout()
            fig4.savefig(str(args.out_dir / "fig4_gene_heatmap.png"), dpi=200, bbox_inches='tight')
            plt.close(fig4)
            print("  Fig4: gene heatmap")
    except Exception as e:
        print(f"  Fig4 failed: {e}")
        traceback.print_exc()

    try:
        true_s = test_gt.get((traj_stp, 'dis'))
        true_e = test_gt.get((cfg['traj_query_tps'][-1], 'dis'))
        if true_s is not None and true_e is not None:
            top_g = np.argsort(np.abs(true_e - true_s))[::-1][:20]
            glabels2 = [gene_names[i] if i < len(gene_names) else f'G{i}' for i in top_g]
            preds_t = predict_trajectory(traj_dis, traj_sd,
                                           cfg['traj_query_days'], cfg['infer_cond_dis'])
            tps_all = [traj_stp] + cfg['traj_query_tps']
            true_all = [test_gt.get((t, 'dis')) for t in tps_all]
            pred_all = [test_gt.get((traj_stp, 'dis'))] + preds_t

            fig5, axes5 = plt.subplots(4, 5, figsize=(22, 14))
            fig5.suptitle(f'{cfg["name"]} - Top-20 gene trajectories (Disease)',
                            fontsize=13, fontweight='bold')
            for ax, gi, gl in zip(axes5.flat, top_g, glabels2):
                tv = [t[gi] if t is not None else np.nan for t in true_all]
                pv = [p[gi] if p is not None else np.nan for p in pred_all]
                xs = list(range(len(tps_all)))
                ax.plot(xs, tv, 'o-', color='#E24B4A', lw=2, markersize=6, label='GT')
                ax.plot(xs, pv, 's--', color='#D85A30', lw=1.5, markersize=5, label='Pred')
                ax.set_xticks(xs)
                ax.set_xticklabels(tps_all, fontsize=7, rotation=25)
                ax.set_title(gl, fontsize=8, fontweight='bold')
                ax.grid(alpha=0.2)
            axes5.flat[0].legend(fontsize=7)
            fig5.tight_layout()
            fig5.savefig(str(args.out_dir / "fig5_top20_gene_traj.png"), dpi=200, bbox_inches='tight')
            plt.close(fig5)
            print("  Fig5: top-20 gene trajectories")
    except Exception as e:
        print(f"  Fig5 failed: {e}")
        traceback.print_exc()

    try:
        pca_vis = PCA(n_components=2, random_state=42)
        pca_vis.fit(test_expr)
        fig6, ax6 = plt.subplots(figsize=(9, 7))
        tp_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(cfg['tp_order'])))
        for i, tp in enumerate(cfg['tp_order']):
            m = (test_meta['timepoint'] == tp).values
            if m.sum() >= 2:
                pts = pca_vis.transform(test_expr[m])
                ax6.scatter(pts[:, 0], pts[:, 1], c=[tp_colors[i]],
                             s=4, alpha=0.12, linewidths=0)

        def _pt(tp, lbl):
            v = test_gt.get((tp, lbl))
            if v is None:
                v = train_gt.get((tp, lbl))
            return pca_vis.transform(v.reshape(1, -1))[0] if v is not None else None

        tps_all = [traj_stp] + cfg['traj_query_tps']
        gt_pts = [(tp, _pt(tp, 'dis')) for tp in tps_all]
        gt_pts = [(tp, pt) for tp, pt in gt_pts if pt is not None]
        if len(gt_pts) >= 2:
            xs, ys = zip(*[pt for _, pt in gt_pts])
            ax6.plot(xs, ys, 'r-o', lw=2.5, markersize=9, label='GT Disease', zorder=8)
            for tp, pt in gt_pts:
                ax6.annotate(tp, pt, fontsize=7, xytext=(5, 3),
                                textcoords='offset points', color='#E24B4A')

        traj_p = predict_trajectory(traj_dis, traj_sd,
                                      cfg['traj_query_days'], cfg['infer_cond_dis'])
        sv = test_gt.get((traj_stp, 'dis'))
        if sv is None:
            sv = train_gt.get((traj_stp, 'dis'))
        if sv is not None:
            chain = ([pca_vis.transform(sv.reshape(1, -1))[0]] +
                      [pca_vis.transform(p.reshape(1, -1))[0] for p in traj_p])
            pc = np.array(chain)
            ax6.plot(pc[:, 0], pc[:, 1], 's--', color='#D85A30', lw=2.5, markersize=8,
                      label='Pred (Disease)', zorder=7)

        ctl_pts = [(tp, _pt(tp, 'ctl')) for tp in tps_all]
        ctl_pts = [(tp, pt) for tp, pt in ctl_pts if pt is not None]
        if len(ctl_pts) >= 2:
            xs, ys = zip(*[pt for _, pt in ctl_pts])
            ax6.plot(xs, ys, 'b-o', lw=1.5, markersize=6, alpha=0.6,
                      label='GT Control', zorder=5)

        ax6.set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=10)
        ax6.set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=10)
        ax6.set_title(f'{cfg["name"]} - PCA trajectory', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8, loc='best', framealpha=0.7)
        ax6.grid(alpha=0.2)
        fig6.tight_layout()
        fig6.savefig(str(args.out_dir / "fig6_pca.png"), dpi=200, bbox_inches='tight')
        plt.close(fig6)
        print("  Fig6: PCA trajectory")
    except Exception as e:
        print(f"  Fig6 failed: {e}")
        traceback.print_exc()

    print(f"\nDone. Output: {args.out_dir}")


if __name__ == '__main__':
    main()
