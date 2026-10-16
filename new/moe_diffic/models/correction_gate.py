"""实验 11：干扰强度 / 恢复置信度驱动的自适应修正   ŝ = s_base + λ · r̂ ，λ∈[0,1]。

输入全部是接收端可得量：y（观测）、s_base（基础估计）、r̂（扩散采样出的残差估计，已乘回尺度）。**不使用真实 SIR/干扰类别**，
唯一例外是 variant='oracle'（仅作上界分析，不可部署）。
variant：
  none   λ≡1（无门控，对照）
  sir    只用强度估计 (ρ̂, u)：SIR 估计器由带内 SIR 标签用高斯 NLL 监督（标签来自合成数据的训练/验证集）
  conf   只用置信度特征：log‖r̂‖², log‖s_base−y‖², log‖y‖² 及分段能量
  both   二者共同
  oracle 用真实 SIR（上界）
local=True 时输出 n_seg 段的 λ(t)（线性插值到全长），适合突发干扰；否则每条样本一个标量 λ。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .prior_router import PriorGuidedRouter

VARIANTS = ('none', 'sir', 'conf', 'both', 'oracle')


def seg_log_energy(x, n_seg):
    """x: (B,2,L) -> (B,n_seg) 每段的对数能量。"""
    e = x.pow(2).sum(1)
    L = e.shape[-1] // n_seg * n_seg
    return torch.log(e[:, :L].reshape(e.shape[0], n_seg, -1).mean(-1) + 1e-8)


class CorrectionGate(nn.Module):
    def __init__(self, variant='both', local=False, n_seg=16):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('variant must be one of %s' % (VARIANTS,))
        self.variant, self.local, self.n_seg = variant, bool(local), n_seg
        self.embed = PriorGuidedRouter(num_experts=2, top_k=1)          # 复用其特征提取（RMS 归一化的时/频/时频特征）
        feat_dim = 64 * 2 + 64 * 2 + 32 * 2
        del self.embed.head
        self.sir_head = nn.Sequential(nn.Linear(feat_dim, 64), nn.SiLU(), nn.Linear(64, 2))
        n_conf = 3 + 2 * n_seg
        n_in = {'none': 1, 'sir': 2, 'conf': n_conf, 'both': 2 + n_conf, 'oracle': 1 + n_conf}[variant]
        self.mlp = nn.Sequential(nn.Linear(n_in, 64), nn.SiLU(), nn.Linear(64, n_seg if local else 1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.constant_(self.mlp[-1].bias, 3.0)      # 初始 λ≈0.95，接近“不门控”

    def sir_estimate(self, y):
        o = self.sir_head(self.embed.embed(y)).float()
        return o[:, 0], o[:, 1].clamp(-6.0, 2.0)

    def conf_feats(self, y, s_base, r):
        return torch.cat([torch.stack([torch.log(r.pow(2).mean((1, 2)) + 1e-8), torch.log((s_base - y).pow(2).mean((1, 2)) + 1e-8),
                                       torch.log(y.pow(2).mean((1, 2)) + 1e-8)], 1),
                          seg_log_energy(r, self.n_seg), seg_log_energy(s_base - y, self.n_seg)], 1)

    def forward(self, y, s_base, r, sir=None):
        """返回 λ：(B,1,1) 或 (B,1,L)。"""
        B, L = y.shape[0], y.shape[-1]
        if self.variant == 'none':
            return torch.ones(B, 1, 1, device=y.device, dtype=y.dtype)
        parts = []
        if self.variant in ('sir', 'both'):
            mu, lv = self.sir_estimate(y)
            parts.append(torch.stack([mu.detach(), lv.detach()], 1))
        if self.variant == 'oracle':
            parts.append((sir / 10.0).unsqueeze(1))
        if self.variant in ('conf', 'both', 'oracle'):
            parts.append(self.conf_feats(y, s_base, r))
        z = self.mlp(torch.cat(parts, 1).float())
        lam = torch.sigmoid(z)
        if self.local:
            lam = F.interpolate(lam.unsqueeze(1), size=L, mode='linear', align_corners=False)      # (B,1,L)
        else:
            lam = lam.unsqueeze(-1)                                                                 # (B,1,1)
        return lam.to(y.dtype)

    def sir_nll(self, y, sir_db):
        mu, lv = self.sir_estimate(y)
        t = sir_db.float() / 10.0
        return (0.5 * (lv + (t - mu) ** 2 / lv.exp())).mean()
