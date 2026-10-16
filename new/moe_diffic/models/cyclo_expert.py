"""循环平稳（谱相关）专家：实验 M2（TF + Cyclo）。
特征来自 M2-0 分析（reports/m2_0_cyclo_analysis.md 的 Bmin 方案）：接收 I/Q 的频带分辨谱相关，在符号率循环频率 α0=1/SPS=0.25 处
取非共轭与共轭（BPSK 的非圆性，沿帧做 ±0.004 的 CFO 搜索）两组，每组 8 维：s_coh(相干系数相位对齐和)、s_inc(相干系数幅度均值)、
s_pow(功率归一化)、5 个子带相干系数幅度，取 log(x+1e-3)，共 16 维（无可学习参数，不使用任何标签）。
再经 LayerNorm + 两层 MLP 映射成与其它专家同维的条件向量 (B, feature_dim)。
CYX = 对照专家：完全相同的流程，但循环频率换成"离循环"的 α=0.28125（信号没有该循环特征），用来检验增益是否来自循环性本身而非"多了一个特征分支"。
估计器：Nw=128 汉宁窗（周期）、hop=32 的时间平滑谱相关；|f|≤0.0625 共 17 个 bin（f=j/128, j=-8..8）。
参考：Gardner, "The Spectral Correlation Theory of Cyclostationary Time-Series", Signal Processing 1986。
"""
import math
import torch
import torch.nn as nn


def _stats(S, Pa, Pb, Pbar, J, groups):
    g = S / torch.sqrt(Pa * Pb + 1e-12)                                  # 局部相干系数 (..., J)
    s_coh = g.sum(-1).abs() / J
    s_inc = g.abs().mean(-1)
    s_pow = S.sum(-1).abs() / (J * Pbar.unsqueeze(-1).squeeze(-1) + 1e-12)
    sub = torch.stack([g.abs()[..., a:b].mean(-1) for a, b in groups], dim=-1)
    return torch.cat([torch.stack([s_coh, s_inc, s_pow], dim=-1), sub], dim=-1)   # (..., 8)


def cyclo_features(time_feat, shift=16, nw=128, hop=32, j_half=8, cfo_max=0.004, cfo_step=0.0004):
    """time_feat: (B, 2, L) 实数 I/Q -> (B, 16)。shift = α/2 对应的 FFT bin 数（α=0.25, nw=128 时为 16）。"""
    with torch.autocast(device_type='cuda', enabled=False):
        x = time_feat.float()
        z = torch.complex(x[:, 0], x[:, 1])                              # (B, L)
        win = torch.hann_window(nw, periodic=True, device=x.device)
        X = torch.fft.fft(z.unfold(-1, nw, hop) * win, dim=-1)          # (B, K, Nw)
        B, K, _ = X.shape
        j = torch.arange(-j_half, j_half + 1, device=x.device)
        J = j.numel()
        edges = [0, 4, 7, 10, 13, J]
        groups = list(zip(edges[:-1], edges[1:]))
        Pbar = (X.abs() ** 2).mean(dim=(1, 2))                           # (B,)
        # 非共轭：S(f) = <X(f+α/2) X*(f-α/2)>
        a, b = X[..., (j + shift) % nw], X[..., (j - shift) % nw]
        Pa, Pb = (a.abs() ** 2).mean(1), (b.abs() ** 2).mean(1)
        f_nc = _stats((a * b.conj()).mean(1), Pa, Pb, Pbar, J, groups)
        # 共轭：S(f) = <X(α/2+f) X(α/2-f)>，循环频率 = α+2·CFO，沿帧做 CFO 搜索（取相干系数最大者）
        a, b = X[..., (shift + j) % nw], X[..., (shift - j) % nw]
        Pa, Pb = (a.abs() ** 2).mean(1), (b.abs() ** 2).mean(1)
        prod = a * b                                                     # (B, K, J)
        deltas = torch.arange(-cfo_max, cfo_max + 1e-9, cfo_step, device=x.device)
        ramp = torch.exp(-2j * math.pi * deltas[:, None] * (torch.arange(K, device=x.device) * hop)[None, :])   # (D, K)
        Sd = torch.einsum('dk,bkj->bdj', ramp.to(prod.dtype), prod) / K  # (B, D, J)
        fd = _stats(Sd, Pa.unsqueeze(1), Pb.unsqueeze(1), Pbar.unsqueeze(1), J, groups)    # (B, D, 8)
        best = fd[..., 0].argmax(dim=1)
        f_cj = fd[torch.arange(B, device=x.device), best]                # (B, 8)
        return torch.log(torch.cat([f_nc, f_cj], dim=-1) + 1e-3)


class CycloExpert(nn.Module):
    def __init__(self, in_channels=2, signal_length=2560, feature_dim=256, hidden=128, shift=16):
        super().__init__()
        self.shift = int(shift)
        self.norm = nn.LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, hidden), nn.SiLU(), nn.Linear(hidden, feature_dim))

    def forward(self, time_feat):
        return self.mlp(self.norm(cyclo_features(time_feat, shift=self.shift)))


class CycloControlExpert(CycloExpert):
    """离循环对照：α=0.28125（shift=18 个 bin）。"""

    def __init__(self, in_channels=2, signal_length=2560, feature_dim=256, hidden=128):
        super().__init__(in_channels, signal_length, feature_dim, hidden, shift=18)
