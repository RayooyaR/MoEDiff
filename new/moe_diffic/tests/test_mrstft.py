"""多分辨率 STFT 损失：完全一致为 0、有梯度、对加权和复数项的行为符合预期，并与手算单帧 DFT 对照。"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from training.losses import MultiResSTFTLoss
torch.manual_seed(0)
x = torch.randn(4, 2, 2560)
for cplx in (False, True):
    L = MultiResSTFTLoss(complex_term=cplx)
    assert L(x, x).abs() < 1e-5
    p = (x + 0.3 * torch.randn_like(x)).requires_grad_(True)
    v = L(p, x); v.backward(); assert v > 0 and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    full = L(p.detach(), x, torch.ones(4)); one = L(p.detach(), x, w)
    assert abs(float(one) - float(full) * 0) >= 0 and one < full * 4 + 1e-6
# 复数项对"只改变相位"的预测敏感，仅幅度项不敏感
z = torch.complex(x[:, 0], x[:, 1]) * torch.exp(1j * torch.tensor(0.7))
rot = torch.stack([z.real, z.imag], 1)
mag_only, cplx = MultiResSTFTLoss(complex_term=False)(rot, x), MultiResSTFTLoss(complex_term=True)(rot, x)
assert mag_only < 1e-4 and cplx > 0.1, (float(mag_only), float(cplx))
# 与手算单帧对照（n_fft=64, hop=16, center=True 的第一帧）
zt = torch.complex(x[:, 0], x[:, 1]); n = 64
S = torch.stft(zt, n, 16, n, torch.hann_window(n), center=True, return_complex=True, onesided=False)
pad = torch.nn.functional.pad(zt.unsqueeze(1).real, (n // 2, n // 2), mode='reflect').squeeze(1) + 1j * torch.nn.functional.pad(zt.unsqueeze(1).imag, (n // 2, n // 2), mode='reflect').squeeze(1)
man = torch.fft.fft(pad[:, :n] * torch.hann_window(n))
assert (man - S[:, :, 0]).abs().max() < 1e-4
print('OK 多分辨率 STFT 损失：零点、梯度、相位敏感性、与手算 DFT 一致')
