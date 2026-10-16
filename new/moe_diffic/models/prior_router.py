"""先验引导的稀疏路由器 (Prior-Guided Sparse Router)。

相对原 BiasedRouter 的改动：
  1. 输入先做按样本功率(RMS)归一化 —— 路由只看信号"形状"，不受干扰强度(SIR)左右
     （实测原路由器的权重主要随 SIR 变化，而不是随干扰类型变化）。
  2. 同时提取 时域 / 频谱 / 时频 三路特征（原路由器只有一路粗糙的时域卷积）。
  3. 训练期用「干扰类型 → 擅长它的专家」的物理先验分布做弱监督（交叉熵）；推理期只看信号，不需要标签。
  4. 稀疏 top-k 门控（k=config.moe.k），未选中的专家权重严格为 0；训练期给 logits 加小噪声以保留探索。
  5. Switch 式负载均衡损失（可微），替代原先恒为常数的"负载"项。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 专家顺序须与 config.moe.active_experts 一致：频域 / 多尺度 / 时频 / 时域
EXPERT_ORDER = ['FD', 'MS', 'TF', 'TD']

# 干扰类型 -> 各专家的先验权重（行和为 1）。依据干扰最显著的特征域：
#   ST/MT 频谱尖峰 -> FD；NB 窄带谱隆起 -> FD(+MS)；PL 时域突发 -> TD；
#   LFM/NFM/RFM 频率随时间变化 -> TF；WB 宽带平坦 -> MS(+FD)
# 这是一个可调的设计假设，不是真值；可在论文中作为消融（先验强度/是否使用）。
TYPE_PRIOR = {
    'ST':  {'FD': 0.60, 'TF': 0.20, 'MS': 0.10, 'TD': 0.10},
    'MT':  {'FD': 0.60, 'TF': 0.20, 'MS': 0.10, 'TD': 0.10},
    'NB':  {'FD': 0.50, 'MS': 0.30, 'TF': 0.10, 'TD': 0.10},
    'PL':  {'TD': 0.60, 'TF': 0.20, 'MS': 0.10, 'FD': 0.10},
    'LFM': {'TF': 0.60, 'FD': 0.20, 'TD': 0.10, 'MS': 0.10},
    'NFM': {'TF': 0.50, 'FD': 0.30, 'TD': 0.10, 'MS': 0.10},
    'RFM': {'TF': 0.50, 'MS': 0.20, 'FD': 0.20, 'TD': 0.10},
    'WB':  {'MS': 0.50, 'FD': 0.30, 'TF': 0.10, 'TD': 0.10},
}


def build_prior_table(type_names, expert_names=EXPERT_ORDER):
    """返回 (num_types, num_experts) 的先验矩阵；未登记的类型使用均匀分布。"""
    rows = []
    for t in type_names:
        d = TYPE_PRIOR.get(t)
        if d is None:
            rows.append([1.0 / len(expert_names)] * len(expert_names))
        else:
            rows.append([d.get(e, 0.0) for e in expert_names])
    tab = torch.tensor(rows, dtype=torch.float32)
    return tab / tab.sum(dim=1, keepdim=True)


class PriorGuidedRouter(nn.Module):
    def __init__(self, num_experts=4, top_k=2, noise_std=0.2, n_fft=64, hop_length=32, hidden=128):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(int(top_k), num_experts))
        self.noise_std = float(noise_std)
        self.n_fft, self.hop = int(n_fft), int(hop_length)
        self.register_buffer('window', torch.hann_window(self.n_fft), persistent=False)

        # 时域分支：输入 [I, Q, |c|]
        self.time_enc = nn.Sequential(
            nn.Conv1d(3, 32, 7, stride=2, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.SiLU(),
        )
        # 频谱分支：输入 log 幅度谱（fftshift 后池化到 256 点）
        self.spec_enc = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.SiLU(),
        )
        # 时频分支：输入 log |STFT|
        self.stft_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
        )
        feat_dim = 64 * 2 + 64 * 2 + 32 * 2
        self.head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Linear(hidden, num_experts))

    @staticmethod
    def _pool1d(h):
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)

    def embed(self, x):
        """x: (B, 2, L) -> (B, feat_dim)。先按样本 RMS 归一化，去除强度信息。"""
        rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        x = x / rms
        c = torch.complex(x[:, 0], x[:, 1])                                  # (B, L)

        t = self._pool1d(self.time_enc(torch.stack([x[:, 0], x[:, 1], c.abs()], dim=1)))

        spec = torch.fft.fftshift(torch.fft.fft(c, dim=-1), dim=-1) / (c.shape[-1] ** 0.5)
        spec = torch.log1p(spec.abs()).unsqueeze(1)                          # (B, 1, L)
        spec = F.adaptive_avg_pool1d(spec, 256)
        s = self._pool1d(self.spec_enc(spec))

        st = torch.stft(c, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
                        return_complex=True, onesided=False)                 # (B, n_fft, T)
        st = torch.log1p(torch.fft.fftshift(st.abs(), dim=1)).unsqueeze(1)   # (B, 1, F, T)
        h = self.stft_enc(st)
        f = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return torch.cat([t, s, f], dim=1)

    def forward(self, x, training=True):
        logits = self.head(self.embed(x)).float()                            # (B, K)，供路由监督使用（无噪声）；bf16 下转 fp32
        probs = F.softmax(logits, dim=-1)
        noisy = logits + self.noise_std * torch.randn_like(logits) if (training and self.noise_std > 0) else logits
        topv, topi = noisy.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(logits).scatter(1, topi, F.softmax(topv, dim=-1))
        mask = torch.zeros_like(logits).scatter(1, topi, 1.0)
        # Switch 式均衡损失：K * Σ_i f_i P_i（均匀时为 1）；f=被选中的比例，P=平均路由概率（可微）
        f = mask.mean(dim=0) / self.top_k
        balance = self.num_experts * (f * probs.mean(dim=0)).sum()
        return dict(logits=logits, probs=probs, gates=gates, balance_loss=balance)
