"""实验 4：形态—强度双因素路由器（Dual-Factor Router）。

两个分支（输入都是按样本 RMS 归一化后的 I/Q，只看形状，不看绝对幅度）：
  形态分支 M：时域/频谱/时频三路特征 -> h_m；可选地预测物理属性（辅助监督，标签来自训练集的干扰信号）
  强度分支 S：同样的特征 -> 带内 SIR 的均值 ρ̂ 与对数方差 u（高斯 NLL 监督，标签是合成数据的 SIR；**推理只用 y**）
门控  g = softmax( [ a(h_m) + b(ρ̂, u) + c(h_m, ρ̂) ] / τ )，ρ̂ 与 u 在送入门控前 detach（强度估计器只由 SIR 监督训练）。
变体 variant：
  dual         a + b + c（形态与强度共同决定）
  dual_noint   a + b（无交互项）
  morph        只有 a（形态只决定专家）
  strength     只有 b（强度决定专家）
  oracle_sir   与 dual 相同，但用真实 SIR 代替 ρ̂（**仅作上界分析，不可部署**；需要 aux['sir']）
注意：RMS 归一化只移除整体幅度，不能移除 SIR（公共缩放下信号/干扰功率比不变）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .prior_router import PriorGuidedRouter

VARIANTS = ('dual', 'dual_noint', 'morph', 'strength', 'oracle_sir')


class DualFactorRouter(PriorGuidedRouter):
    def __init__(self, num_experts=4, top_k=2, noise_std=0.2, tau=0.5, variant='dual', n_attr=12, hidden=128):
        super().__init__(num_experts=num_experts, top_k=top_k, noise_std=noise_std, hidden=hidden)
        if variant not in VARIANTS:
            raise ValueError('variant must be one of %s' % (VARIANTS,))
        del self.head
        self.variant, self.tau, self.hidden = variant, float(tau), hidden
        feat_dim = 64 * 2 + 64 * 2 + 32 * 2
        self.morph = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU())
        self.a = nn.Linear(hidden, num_experts)
        self.attr_head = nn.Linear(hidden, n_attr)
        self.strength = nn.Sequential(nn.Linear(feat_dim, 64), nn.SiLU(), nn.Linear(64, 2))
        self.b = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, num_experts))
        self.c_gate = nn.Linear(1, hidden)
        self.c = nn.Linear(hidden, num_experts)

    def forward(self, x, training=True, aux=None):
        feat = self.embed(x)
        h = self.morph(feat)
        sm = self.strength(feat).float()
        mu, logvar = sm[:, 0], sm[:, 1].clamp(-6.0, 2.0)                       # SIR/10 的均值与对数方差
        if self.variant == 'oracle_sir':
            if aux is None or aux.get('sir') is None:
                raise ValueError('oracle_sir 需要真实 SIR（aux["sir"]）；它只用于上界分析')
            rho = (aux['sir'].to(mu.dtype) / 10.0)
        else:
            rho = mu.detach()
        u = logvar.detach()
        logits = 0.0
        if self.variant in ('dual', 'dual_noint', 'morph', 'oracle_sir'):
            logits = logits + self.a(h)
        if self.variant in ('dual', 'dual_noint', 'strength', 'oracle_sir'):
            logits = logits + self.b(torch.stack([rho, u], dim=-1))
        if self.variant in ('dual', 'oracle_sir'):
            logits = logits + self.c(h * torch.sigmoid(self.c_gate(rho.unsqueeze(-1))))
        logits = (logits / self.tau).float()           # bf16 混合精度下路由逻辑一律用 fp32
        probs = F.softmax(logits, dim=-1)
        noisy = logits + self.noise_std * torch.randn_like(logits) if (training and self.noise_std > 0) else logits
        if self.top_k < self.num_experts:
            topv, topi = noisy.topk(self.top_k, dim=-1)
            gates = torch.zeros_like(logits).scatter(1, topi, F.softmax(topv, dim=-1))
            mask = torch.zeros_like(logits).scatter(1, topi, 1.0)
        else:
            gates = F.softmax(noisy, dim=-1)
            mask = torch.ones_like(logits)
        f = mask.mean(dim=0) / self.top_k
        balance = self.num_experts * (f * probs.mean(dim=0)).sum()
        return dict(logits=logits, probs=probs, gates=gates, balance_loss=balance, sir_mu=mu, sir_logvar=logvar, attr_pred=self.attr_head(h))

    @staticmethod
    def sir_nll(out, sir_db):
        t = (sir_db.float() / 10.0)
        return (0.5 * (out['sir_logvar'] + (t - out['sir_mu']) ** 2 / out['sir_logvar'].exp())).mean()
