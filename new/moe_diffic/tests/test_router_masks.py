"""Top-K 稀疏的真实性：
 (1) fusion=topk_mask 时，被屏蔽专家 token 的内容不应影响去噪输出（真正屏蔽）；
 (2) fusion=xattn 时“门控为 0”只是把 token 乘 0，token 仍以 0 向量参与注意力（Key/Value 的偏置投影不为 0），因此与“真屏蔽”不等价——本测试量化这个差别。"""
from _common import *
import torch
from models import PhysicsMoEModel

args, cfg = make_cfg(['--fusion', 'topk_mask', '--router_type', 'dual', '--moe_k', '2'])
m = randomize(PhysicsMoEModel(cfg).to(DEV)).eval()
torch.manual_seed(0)
y = torch.randn(6, 2, 2560, device=DEV); x = torch.randn(6, 2, 2560, device=DEV); t = torch.randint(0, 1000, (6,), device=DEV)
cond, info = m.compute_moe_condition(y, training=False)
mask = info['cond_mask']
assert mask is not None and mask.dtype == torch.bool and (~mask).sum(1).min() >= 1, '至少保留一个专家'
assert ((~mask).sum(1) == 2).all(), 'top-2 应恰好保留 2 个 token'
out1 = m.denoise(x, y, t, cond, mask)
cond2 = cond.clone(); cond2[mask] = torch.randn_like(cond2[mask]) * 5.0           # 篡改被屏蔽 token 的内容
out2 = m.denoise(x, y, t, cond2, mask)
assert torch.allclose(out1, out2, atol=1e-5), '真屏蔽下，被屏蔽 token 的内容不应改变输出'
# 对照：不传 mask，仅把 token 置零 vs 真正屏蔽
cond_zero = cond.clone(); cond_zero[mask] = 0
d_zero_vs_mask = (m.denoise(x, y, t, cond_zero, None) - out1).abs().max().item()
print('PASS test_router_masks：真屏蔽有效；“乘零不屏蔽”与“真屏蔽”的最大输出差 = %.3e（>0 说明乘零并非真稀疏）' % d_zero_vs_mask)
