"""循环平稳专家：与 M2-0 分析数值同量级（干净信号 α=0.25 的 s_coh≈0.44~0.85；纯干扰+噪声显著更低）、可微分接入、对照专家用离循环 α。"""
import os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..', 'loader'))
from models.cyclo_expert import CycloExpert, CycloControlExpert, cyclo_features
from moe_dataset import MoEDiffDataset
ds = MoEDiffDataset(os.path.join(HERE, '..', '..', 'data', 'v2'), 'test', noise_mode='with')
S = ds.samples[::40][:150]
clean = torch.tensor(np.stack([s['clean'] for s in S]), dtype=torch.float32)
mix = torch.tensor(np.stack([s['noisy'] for s in S]), dtype=torch.float32)
bad = torch.tensor(np.stack([s['noisy'] - s['clean'] for s in S]), dtype=torch.float32)       # 纯干扰+噪声
fc, fm, fb = (cyclo_features(t) for t in (clean, mix, bad))
assert fc.shape == (len(S), 16) and torch.isfinite(fc).all() and torch.isfinite(fm).all()
coh = lambda f: (f[:, 0].exp() - 1e-3)                                      # 非共轭 s_coh
assert 0.3 < coh(fc).mean() < 1.0, float(coh(fc).mean())                    # 与 M2-0 表一致量级(0.44~0.85)
assert coh(fc).mean() > 3 * coh(fb).mean(), (float(coh(fc).mean()), float(coh(fb).mean()))
sir = np.array([s['sir_db'] for s in S]); lo = torch.tensor(sir <= -10)
assert coh(fm)[~lo].mean() > coh(fm)[lo].mean()                              # 混合信号的特征随 SIR 升高而增强
fx = cyclo_features(clean, shift=18)
assert coh(fx).mean() < 0.2 * coh(fc).mean()                                 # 离循环频率上信号没有特征
e = CycloExpert(feature_dim=64); out = e(mix); assert out.shape == (len(S), 64)
out.sum().backward(); assert all(p.grad is not None for p in e.parameters())
assert CycloControlExpert(feature_dim=64).shift == 18
assert torch.allclose(cyclo_features(mix[:7]), cyclo_features(mix)[:7], atol=1e-5)    # 批大小无关
print('OK 循环平稳专家：干净 s_coh=%.2f、纯干扰 %.3f、离循环 %.3f；参数 %d' % (coh(fc).mean(), coh(fb).mean(), coh(fx).mean(), sum(p.numel() for p in e.parameters())))
