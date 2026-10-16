"""局部干扰定位特征：单音落在正确子带、脉冲落在正确时段、形状与可微分；E1 预测器输出形状。"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.local_cond import interf_features, upsample_time, E1PredictorCNN, N_BANDS
L = 2560; n = torch.arange(L).float()
f0 = 0.1                                                    # 归一化频率 0.1 -> bin 12.8 -> 落在从 -32 起第 (32+12.8)/4=11 个子带
z = torch.exp(2j * np.pi * f0 * n)
x = torch.stack([z.real, z.imag])[None]
F_ = interf_features(x)
assert F_.shape == (1, N_BANDS, L // 32 + 1)
band = int(F_.mean(-1).argmax()); assert band == 11, band
env = torch.zeros(L); env[1000:1300] = 1.0                   # 脉冲在 1000..1300：对应帧 31..40
zz = env * torch.exp(2j * np.pi * 0.0 * n); xp = torch.stack([zz.real, zz.imag])[None]
Fp = interf_features(xp).mean(1)[0]; t_peak = int(Fp.argmax()) * 32
assert 900 <= t_peak <= 1400, t_peak
up = upsample_time(F_, L); assert up.shape == (1, N_BANDS, L)
m = E1PredictorCNN(); y = torch.randn(3, 2, L); out = m(y); assert out.shape == (3, N_BANDS, L // 32), out.shape
out.sum().backward(); assert all(p.grad is not None for p in m.parameters())
print('OK 局部干扰定位特征：单音子带、脉冲时段、预测器形状均正确')
