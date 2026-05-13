"""
MitoLODE core model.

Components:
    TimeConditionedEmbedding : per-gene time-dependent offset on HGT embeddings.
    KGGateAttention          : KGSFM gate modulating input expression by KG prior.
    VAEEncoder / VAEDecoder  : base VAE encoder and decoder.
    MaskedMitoPrior          : stochastic input masking (mitochondrial mask prior).
    KGGuidedODEFunc          : KG- and condition-aware ODE velocity field.
    JTLatentODE_v6           : top-level model that assembles everything.
    jt_lode_loss             : joint loss for reconstruction + KL + trajectory + temporal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeConditionedEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 64, max_days: float = 42.0,
                 n_genes: int = 984, delta_scale: float = 0.20):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_days = max_days
        self.delta_scale = delta_scale

        self.time_encoder = nn.Sequential(
            nn.Linear(1, emb_dim), nn.Tanh(),
            nn.Linear(emb_dim, emb_dim), nn.Tanh(),
        )

        self.sensitivity_net = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 2), nn.GELU(),
            nn.Linear(emb_dim // 2, 1),
        )
        nn.init.normal_(self.sensitivity_net[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.sensitivity_net[-1].bias)

        self.gene_sensitivity = nn.Parameter(
            torch.ones(n_genes) + torch.randn(n_genes) * 0.01
        )

    def forward(self, gene_emb: torch.Tensor, time_days: torch.Tensor):
        G, d = gene_emb.shape
        B = time_days.shape[0]
        gene_emb = gene_emb.to(time_days.device)

        t_norm = (time_days / self.max_days).unsqueeze(-1).clamp(0, 1)
        t_feat = self.time_encoder(t_norm)

        E_exp = gene_emb.unsqueeze(0).expand(B, -1, -1)
        t_exp = t_feat.unsqueeze(1).expand(B, G, -1)

        raw_score = self.sensitivity_net(torch.cat([E_exp, t_exp], dim=-1))
        gene_sens = F.softplus(self.gene_sensitivity[:G]).unsqueeze(0).unsqueeze(-1)

        E_norm = F.normalize(E_exp, dim=-1)
        scale = torch.tanh(raw_score) * gene_sens * self.delta_scale
        scale = scale * t_norm.unsqueeze(1)

        delta = scale * E_norm
        E_t = E_exp + delta
        return E_t, delta


class KGGateAttention(nn.Module):
    def __init__(self, input_dim: int, gene_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(gene_emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor, gene_emb: torch.Tensor) -> torch.Tensor:
        gene_emb = gene_emb.to(x.device)
        if gene_emb.dim() == 2:
            gate = torch.sigmoid(self.gate_net(gene_emb).squeeze(-1))
            return x * gate.unsqueeze(0)
        elif gene_emb.dim() == 3:
            B, G, d = gene_emb.shape
            gate = torch.sigmoid(
                self.gate_net(gene_emb.reshape(B * G, d)).reshape(B, G)
            )
            return x * gate
        else:
            raise ValueError(
                f"KGGateAttention: unexpected gene_emb dim {gene_emb.dim()}"
            )


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-6.0, 2.0)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int,
                 t_feat_dim: int = 16):
        super().__init__()
        self.t_feat_dim = t_feat_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + t_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor, t_feat: torch.Tensor = None) -> torch.Tensor:
        if t_feat is None:
            t_feat = torch.zeros(z.shape[0], self.t_feat_dim,
                                  device=z.device, dtype=z.dtype)
        return self.net(torch.cat([z, t_feat], dim=-1))


class MaskedMitoPrior(nn.Module):
    def __init__(self, input_dim: int, mask_ratio: float = 0.25):
        super().__init__()
        self.input_dim = input_dim
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor):
        if self.training and self.mask_ratio > 0:
            mask = (torch.rand(x.shape, device=x.device) > self.mask_ratio).float()
            return x * mask, mask
        return x, torch.ones_like(x)


class KGGuidedODEFunc(nn.Module):
    def __init__(self,
                 latent_dim: int = 32,
                 ode_hidden: int = 128,
                 kg_ctx_dim: int = 32,
                 t_feat_dim: int = 16,
                 max_days: float = 42.0,
                 dz_scale: float = 0.06):
        super().__init__()
        self.latent_dim = latent_dim
        self.kg_ctx_dim = kg_ctx_dim
        self.t_feat_dim = t_feat_dim
        self.max_days = max_days
        self.dz_scale = dz_scale

        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_feat_dim), nn.Tanh(),
            nn.Linear(t_feat_dim, t_feat_dim), nn.Tanh(),
        )
        self.kg_proj = nn.Sequential(
            nn.Linear(64, kg_ctx_dim), nn.Tanh(),
        )
        self.cond_emb = nn.Embedding(2, kg_ctx_dim)
        nn.init.normal_(self.cond_emb.weight, 0.0, 0.01)

        in_dim = latent_dim + t_feat_dim + kg_ctx_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, ode_hidden), nn.Tanh(),
            nn.Linear(ode_hidden, ode_hidden), nn.Tanh(),
            nn.Linear(ode_hidden, latent_dim), nn.Tanh(),
        )
        nn.init.uniform_(self.net[-2].weight, -0.005, 0.005)
        nn.init.zeros_(self.net[-2].bias)

        self.register_buffer('_kg_ctx', torch.zeros(1, kg_ctx_dim))
        self.register_buffer('_cond_ctx', torch.zeros(1, kg_ctx_dim))
        self._ctx_ready = False

    def set_context(self, gene_emb: torch.Tensor):
        ctx = self.kg_proj(gene_emb.mean(dim=0, keepdim=True))
        self._kg_ctx = ctx.detach()
        self._ctx_ready = True

    def set_condition(self, cond_idx: int):
        idx = torch.tensor([cond_idx], dtype=torch.long,
                            device=self.cond_emb.weight.device)
        self._cond_ctx = self.cond_emb(idx)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        t_val = t.item() if t.numel() == 1 else t[0].item()
        t_norm = torch.tensor(
            [[t_val / self.max_days]], device=z.device, dtype=z.dtype
        ).clamp(0, 1)
        t_feat = self.t_encoder(t_norm).expand(B, -1)
        kg_ctx = (self._kg_ctx + self._cond_ctx).expand(B, -1) \
                  if self._ctx_ready \
                  else torch.zeros(B, self.kg_ctx_dim, device=z.device)
        inp = torch.cat([z, t_feat, kg_ctx], dim=-1)
        return self.net(inp) * self.dz_scale


class JTLatentODE_v6(nn.Module):
    def __init__(self,
                 input_dim: int = 984,
                 latent_dim: int = 32,
                 hidden_dim: int = 256,
                 gene_emb_dim: int = 64,
                 mask_ratio: float = 0.25,
                 ode_hidden: int = 128,
                 kg_ctx_dim: int = 32,
                 t_feat_dim: int = 16,
                 max_days: float = 42.0,
                 dz_scale: float = 0.06,
                 ode_rtol: float = 1e-2,
                 ode_atol: float = 1e-3,
                 use_temporal_attention: bool = True,
                 n_attn_heads: int = 4,
                 delta_scale: float = 0.20):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_days = max_days
        self.t_feat_dim = t_feat_dim
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.use_temporal_attention = use_temporal_attention

        self.kg_attn = KGGateAttention(input_dim, gene_emb_dim, hidden_dim)
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim, hidden_dim,
                                    t_feat_dim=t_feat_dim)
        self.mmp = MaskedMitoPrior(input_dim, mask_ratio)

        if self.use_temporal_attention:
            self.temporal_kg_attn = TimeConditionedEmbedding(
                emb_dim=gene_emb_dim,
                max_days=max_days,
                n_genes=input_dim,
                delta_scale=delta_scale,
            )

        self.ode_func = KGGuidedODEFunc(
            latent_dim=latent_dim, ode_hidden=ode_hidden, kg_ctx_dim=kg_ctx_dim,
            t_feat_dim=t_feat_dim, max_days=max_days, dz_scale=dz_scale,
        )

    def _make_t_feat(self, time_days: torch.Tensor) -> torch.Tensor:
        t_norm = (time_days / self.max_days).unsqueeze(-1).clamp(0, 1)
        return self.ode_func.t_encoder(t_norm)

    def encode(self, x: torch.Tensor, gene_emb: torch.Tensor,
               time_days: torch.Tensor = None):
        delta = None
        t_feat = None

        if self.use_temporal_attention and time_days is not None:
            E_t, delta = self.temporal_kg_attn(gene_emb, time_days)
            effective_emb = E_t
            t_feat = self._make_t_feat(time_days)
        else:
            effective_emb = gene_emb
            if time_days is not None:
                t_feat = self._make_t_feat(time_days)

        x_w = self.kg_attn(x, effective_emb)
        x_m, mask = self.mmp(x_w)
        mu, logvar = self.encoder(x_m)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu, logvar, z, mask, delta, t_feat

    def decode(self, z: torch.Tensor, t_feat: torch.Tensor = None) -> torch.Tensor:
        return self.decoder(z, t_feat)

    def compute_anchors(self, x_batch, t_batch, t_days_batch, gene_emb):
        was_training = self.mmp.training
        self.mmp.eval()
        with torch.no_grad():
            mu, _, _, _, _, _ = self.encode(x_batch, gene_emb, time_days=t_days_batch)
        if was_training:
            self.mmp.train()
        anchors = {}
        for t_idx in t_batch.unique():
            sel = (t_batch == t_idx)
            if sel.sum() < 2:
                continue
            anchors[t_idx.item()] = mu[sel].mean(dim=0)
        return anchors

    def compute_trajectory_loss_v6(self,
                                    mu_stack: torch.Tensor,
                                    tp_stack: torch.Tensor,
                                    cond_stack: torch.Tensor,
                                    gene_emb: torch.Tensor,
                                    t_days_map: dict,
                                    sep_loss_margin: float = 2.0,
                                    sep_loss_weight: float = 2.0,
                                    anchor_reg_norm: float = 5.0,
                                    anchor_reg_weight: float = 0.1) -> torch.Tensor:
        device = mu_stack.device
        self.ode_func.set_context(gene_emb)

        batch_anchors = {}
        for c in cond_stack.unique():
            for t in tp_stack.unique():
                sel = (cond_stack == c) & (tp_stack == t)
                if sel.sum() >= 1:
                    batch_anchors[(c.item(), t.item())] = mu_stack[sel].mean(0).float()

        if len(batch_anchors) < 2:
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for cond in list(set(c for c, _ in batch_anchors.keys())):
            self.ode_func.set_condition(int(cond))
            avail = sorted(
                [(t, t_days_map[t]) for (c, t) in batch_anchors.keys()
                 if c == cond and t in t_days_map],
                key=lambda x: x[1]
            )
            deduped, last_d = [], -999.0
            for t_idx, t_day in avail:
                d = max(float(t_day), 0.1)
                if d - last_d > 0.05:
                    deduped.append((t_idx, d))
                    last_d = d
            if len(deduped) < 2:
                continue

            t0_idx = deduped[0][0]
            z0 = batch_anchors[(cond, t0_idx)].unsqueeze(0)
            if not torch.isfinite(z0).all() or z0.norm() > 50.0:
                continue

            t_tensor = torch.tensor([d for _, d in deduped],
                                      dtype=torch.float32, device=device)
            try:
                from torchdiffeq import odeint
                z_traj = odeint(self.ode_func, z0, t_tensor,
                                 method='rk4', options={'step_size': 0.5})
            except Exception as e:
                if not hasattr(self, '_traj_err_printed'):
                    print(f"  [ODE error] {e}")
                    self._traj_err_printed = True
                continue

            if not torch.isfinite(z_traj).all():
                continue

            for i, (t_idx, _) in enumerate(deduped[1:], 1):
                if (cond, t_idx) not in batch_anchors:
                    continue
                z_pred = z_traj[i, 0, :].float()
                z_target = batch_anchors[(cond, t_idx)]
                step_loss = F.mse_loss(z_pred, z_target)
                if torch.isfinite(step_loss) and step_loss < 1e4:
                    total_loss = total_loss + step_loss
                    n_pairs += 1

        traj_mse = total_loss / max(n_pairs, 1)

        sep_loss = torch.tensor(0.0, device=device)
        n_sep = 0
        all_keys = list(batch_anchors.keys())
        for i in range(len(all_keys)):
            ci, ti = all_keys[i]
            for j in range(i + 1, len(all_keys)):
                cj, tj = all_keys[j]
                if ci == cj and ti != tj:
                    dist = (batch_anchors[all_keys[i]] - batch_anchors[all_keys[j]]).norm()
                    sep_loss = sep_loss + F.relu(sep_loss_margin - dist)
                    n_sep += 1
        if n_sep > 0:
            sep_loss = sep_loss / n_sep

        anchor_reg = torch.tensor(0.0, device=device)
        for v in batch_anchors.values():
            anchor_reg = anchor_reg + F.relu(v.norm() - anchor_reg_norm)
        anchor_reg = anchor_reg / max(len(batch_anchors), 1)

        return traj_mse + sep_loss_weight * sep_loss + anchor_reg_weight * anchor_reg

    @staticmethod
    def compute_vae_temporal_loss(mu_batch: torch.Tensor,
                                    tp_batch: torch.Tensor,
                                    t_days_map: dict,
                                    cond_batch: torch.Tensor = None,
                                    spread_margin: float = 1.5,
                                    contrastive_margin: float = 0.5,
                                    contrastive_weight: float = 0.5) -> torch.Tensor:
        device = mu_batch.device
        loss_dir = torch.tensor(0.0, device=device)
        loss_spread = torch.tensor(0.0, device=device)
        n_dir = n_spread = 0

        conditions = cond_batch.unique() if cond_batch is not None \
                     else torch.tensor([0], device=device)

        cond_tp_means = {}

        for cond in conditions:
            tp_means = {}
            for t_idx in tp_batch.unique():
                if cond_batch is not None:
                    sel = (cond_batch == cond) & (tp_batch == t_idx)
                else:
                    sel = (tp_batch == t_idx)
                if sel.sum() < 2:
                    continue
                tp_means[t_idx.item()] = mu_batch[sel].mean(dim=0)
                cond_tp_means[(cond.item(), t_idx.item())] = tp_means[t_idx.item()]

            sorted_tps = sorted(
                [(t_idx, t_days_map[t_idx]) for t_idx in tp_means if t_idx in t_days_map],
                key=lambda x: x[1]
            )
            if len(sorted_tps) < 2:
                continue

            zs = [tp_means[t_idx] for t_idx, _ in sorted_tps]

            for i in range(len(zs) - 1):
                dist = (zs[i + 1] - zs[i]).norm()
                loss_spread = loss_spread + F.relu(spread_margin - dist)
                n_spread += 1

            if len(zs) >= 3:
                for i in range(len(zs) - 2):
                    dz1 = zs[i + 1] - zs[i].detach()
                    dz2 = zs[i + 2] - zs[i + 1]
                    if dz1.norm() < 1e-6 or dz2.norm() < 1e-6:
                        continue
                    cos = F.cosine_similarity(dz1.unsqueeze(0), dz2.unsqueeze(0))
                    loss_dir = loss_dir + (1.0 - cos)
                    n_dir += 1

        contrastive_loss = torch.tensor(0.0, device=device)
        n_contrast = 0
        if cond_batch is not None and len(conditions) > 1:
            cond_list = [c.item() for c in conditions]
            for t_idx in tp_batch.unique().tolist():
                t_means = [(c, cond_tp_means[(c, t_idx)])
                           for c in cond_list if (c, t_idx) in cond_tp_means]
                for ii in range(len(t_means)):
                    for jj in range(ii + 1, len(t_means)):
                        dist = (t_means[ii][1] - t_means[jj][1]).norm()
                        contrastive_loss = contrastive_loss + F.relu(contrastive_margin - dist)
                        n_contrast += 1
        if n_contrast > 0:
            contrastive_loss = contrastive_loss / n_contrast

        return (loss_dir / max(n_dir, 1)
                + loss_spread / max(n_spread, 1)
                + contrastive_weight * contrastive_loss)

    @staticmethod
    def batch_trajectory_consistency(mu_batch: torch.Tensor,
                                       tp_batch: torch.Tensor,
                                       t_days_map: dict,
                                       cond_batch: torch.Tensor = None) -> torch.Tensor:
        device = mu_batch.device
        loss = torch.tensor(0.0, device=device)
        n = 0

        conditions = cond_batch.unique() if cond_batch is not None \
                     else torch.tensor([0], device=device)

        for cond in conditions:
            tp_means = {}
            for t_idx in tp_batch.unique():
                sel = ((cond_batch == cond) & (tp_batch == t_idx)) \
                       if cond_batch is not None else (tp_batch == t_idx)
                if sel.sum() < 2:
                    continue
                tp_means[t_idx.item()] = mu_batch[sel].mean(dim=0)

            sorted_tps = sorted(
                [(t, t_days_map[t]) for t in tp_means if t in t_days_map],
                key=lambda x: x[1]
            )
            if len(sorted_tps) < 3:
                continue
            zs = [tp_means[t] for t, _ in sorted_tps]

            for i in range(len(zs) - 2):
                dz1 = zs[i + 1] - zs[i].detach()
                dz2 = zs[i + 2] - zs[i + 1]
                if dz1.norm() < 1e-6 or dz2.norm() < 1e-6:
                    continue
                cos = F.cosine_similarity(dz1.unsqueeze(0), dz2.unsqueeze(0))
                loss = loss + F.relu(-cos)
                n += 1

        return loss / max(n, 1)

    def forward(self, x: torch.Tensor, gene_emb: torch.Tensor,
                time_days: torch.Tensor = None):
        mu, logvar, z, mask, delta, t_feat = self.encode(x, gene_emb, time_days)
        x_hat = self.decoder(z, t_feat)
        return mu, logvar, z, x_hat, mask, delta

    @torch.no_grad()
    def predict_continuous(self,
                             z0: torch.Tensor,
                             t_query: torch.Tensor,
                             gene_emb: torch.Tensor,
                             cond_idx: int = 0):
        from torchdiffeq import odeint
        self.ode_func.set_context(gene_emb)
        self.ode_func.set_condition(cond_idx)
        z_traj = odeint(self.ode_func, z0, t_query,
                         method='rk4', options={'step_size': 0.5})
        return z_traj[:, 0, :]


def jt_lode_loss(mu, logvar, x_hat, x_orig, mask,
                  traj_loss, temporal_loss,
                  beta=0.5, mask_weight=2.0,
                  lambda_traj=1.0, gamma_temporal=0.1):
    w = 1.0 + (1.0 - mask) * (mask_weight - 1.0)
    recon = (w * (x_hat - x_orig) ** 2).mean()
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl + lambda_traj * traj_loss + gamma_temporal * temporal_loss
    return total, recon, kl
