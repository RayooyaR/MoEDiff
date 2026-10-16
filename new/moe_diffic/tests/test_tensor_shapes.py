"""所有恢复网络 / 去噪主干输出 (B,2,2560)，支持不同 batch size，条件注入各模式可用。"""
from _common import *
import torch
from models import PhysicsMoEModel
from models.residual_diffusion import ResidualDiffusionModel

variants = {
    'default': [], 'tf_only': ['--experts', 'TF'], 'unet': ['--experts', 'TF', '--backbone', 'unet1d', '--unet_base', '32'],
    'diffwave': ['--experts', 'TF', '--backbone', 'diffwave', '--diffwave_res', '64', '--diffwave_layers', '6'],
    'film': ['--cond_mode', 'film'], 'adaln': ['--cond_mode', 'adaln'], 'tok16': ['--experts', 'TF', '--tf_tokens', '16'],
    'shared': ['--shared_expert'], 'dual': ['--router_type', 'dual'], 'topk': ['--fusion', 'topk_mask', '--router_type', 'dual'], 'sum': ['--fusion', 'sum_single'],
    'uniform': ['--fusion', 'uniform'], 'attr': ['--attr_mode', 'joint'], 'uncond': ['--disable_moe', '--true_uncond'],
}
for name, extra in variants.items():
    args, cfg = make_cfg(extra)
    m = PhysicsMoEModel(cfg).to(DEV).eval()
    for B in (1, 3, 8):
        y = torch.randn(B, 2, 2560, device=DEV); x = torch.randn(B, 2, 2560, device=DEV)
        r = m(x, y)
        assert r['noise_pred'].shape == (B, 2, 2560), (name, B, r['noise_pred'].shape)
        s = m.sample(y, num_steps=2)
        assert s.shape == (B, 2, 2560) and torch.isfinite(s).all(), (name, B)
args, cfg = make_cfg(['--experts', 'TF'])
m = ResidualDiffusionModel(cfg).to(DEV).eval()
y = torch.randn(4, 2, 2560, device=DEV); out = m.sample(y, num_steps=2, return_parts=True)
assert out[0].shape == out[1].shape == (4, 2, 2560)
print('PASS test_tensor_shapes (%d 种变体 × 3 种 batch + 残差扩散)' % len(variants))
