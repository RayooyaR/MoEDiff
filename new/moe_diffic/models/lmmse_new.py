"""加窗 LMMSE（维纳）滤波器，推理部分。系数由 lmmse_fit.py 从训练集二阶统计量闭式求得。

与原始 models/baseline/lmmse_baseline.py 的对应：同样是"滑窗 + 窗内 I/Q 联合线性估计 + 重叠平均"，窗长 256、重叠 128；
区别在于协方差用整个训练集的精确样本估计（原实现用 0.01 的指数滑动平均，每个窗口更新一次，有效记忆仅约 100 个窗口≈几条样本）。
"""
import torch
import torch.nn as nn


class LMMSEFilter(nn.Module):
    def __init__(self, W, window=256, overlap=128, length=2560):
        super().__init__()
        self.window, self.overlap, self.length = int(window), int(overlap), int(length)
        self.register_buffer('W', W.float())
        self.starts = list(range(0, self.length - self.window + 1, self.window - self.overlap))

    @torch.no_grad()
    def sample(self, y, num_steps=None, **kw):
        B, C, L = y.shape
        wins = torch.stack([y[:, :, s:s + self.window] for s in self.starts], dim=1)      # (B, nW, 2, W)
        est = (wins.reshape(B, len(self.starts), -1) @ self.W.T).reshape(B, len(self.starts), C, self.window)
        out = torch.zeros_like(y)
        cnt = torch.zeros(1, 1, L, device=y.device, dtype=y.dtype)
        for i, s in enumerate(self.starts):
            out[:, :, s:s + self.window] += est[:, i]
            cnt[:, :, s:s + self.window] += 1
        return out / cnt.clamp_min(1)


def extract_windows(x, window, overlap):
    """x: (N, 2, L) -> (N*nW, 2*window)，窗内按 [I 全部, Q 全部] 展平。"""
    L = x.shape[-1]
    starts = list(range(0, L - window + 1, window - overlap))
    return torch.stack([x[:, :, s:s + window] for s in starts], dim=1).reshape(-1, 2 * window)
