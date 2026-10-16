"""局部（带时间位置）的干扰定位条件：实验 E1（干扰定位专家）及其 oracle 上界。

特征 = 干扰信号的时频功率分布：STFT(n_fft=128, hop=32, 汉宁窗, 双边谱) 取中心 |f|<0.25（覆盖信号带宽 (1+β)/8≤0.19）的 64 个 bin，每 4 个 bin 合成 1 个子带 -> 16 个子带，
逐样本相对功率 P/mean(P) 取 log10(·+1e-3)，截到 [-3,3] 后除以 2。时间轴 81 帧，线性插值到信号长度后作为去噪网络的 16 个额外输入通道。
  oracle_mask：用真实干扰（训练/测试时都取真值；仅作信息上界诊断，不可部署）
  e1：由接收信号 y 预测同样的特征（E1PredictorCNN），训练时用真实干扰特征作稠密辅助监督（干扰 = y − s − 噪声，来自配对样本本身，不需要额外标签）；推理只用 y。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BANDS, N_FFT, HOP = 16, 128, 32


def interf_features(interf):
    """interf: (B, 2, L) -> (B, 16, T=L/HOP+1)"""
    with torch.autocast(device_type='cuda', enabled=False):
        x = interf.float()
        z = torch.complex(x[:, 0], x[:, 1])
        S = torch.stft(z, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=torch.hann_window(N_FFT, device=x.device), center=True, return_complex=True, onesided=False)   # (B, 128, T)
        P = S.real ** 2 + S.imag ** 2
        idx = torch.cat([torch.arange(96, 128), torch.arange(0, 32)]).to(x.device)           # f = -32..31 个 bin（|f| < 0.25）
        P = P[:, idx].reshape(P.shape[0], N_BANDS, 4, P.shape[-1]).mean(2)                   # (B, 16, T)
        rel = P / (P.mean(dim=(1, 2), keepdim=True) + 1e-12)
        return torch.log10(rel + 1e-3).clamp(-3, 3) / 2


def upsample_time(feat, length):
    return F.interpolate(feat, size=length, mode='linear', align_corners=False)


class E1PredictorCNN(nn.Module):
    """y(2, L) -> 干扰时频功率特征 (16, L/32)。5 次步长 2 的卷积。"""

    def __init__(self, in_ch=2, base=32):
        super().__init__()
        chs = [base, 2 * base, 2 * base, 2 * base, 2 * base]
        layers, c = [], in_ch
        for i, co in enumerate(chs):
            layers += [nn.Conv1d(c, co, 7 if i == 0 else 5, stride=2, padding=3 if i == 0 else 2), nn.GroupNorm(8, co), nn.SiLU()]
            c = co
        layers += [nn.Conv1d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, N_BANDS, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y)
