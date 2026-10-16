"""采样可复现：同种子同输入 -> 逐位相同；不同种子不同；确定性路径(init_scale=0)与种子无关；true_uncond 的输出不依赖条件。"""
from _common import *
import torch
from models import PhysicsMoEModel

args, cfg = make_cfg(['--experts', 'TF'])
m = randomize(PhysicsMoEModel(cfg).to(DEV)).eval()
y = torch.randn(4, 2, 2560, device=DEV)
torch.manual_seed(7); a = m.sample(y, num_steps=3)
torch.manual_seed(7); b = m.sample(y, num_steps=3)
torch.manual_seed(8); c = m.sample(y, num_steps=3)
assert torch.equal(a, b), '同种子应逐位相同'
assert not torch.allclose(a, c), '不同种子应不同'
torch.manual_seed(1); d1 = m.sample(y, num_steps=3, init_scale=0.0)
torch.manual_seed(2); d2 = m.sample(y, num_steps=3, init_scale=0.0)
assert torch.equal(d1, d2), '确定性路径与种子无关'
# true_uncond：去噪网络收到 cond=None，输出与任意条件无关
args, cfg = make_cfg(['--disable_moe', '--true_uncond'])
u = randomize(PhysicsMoEModel(cfg).to(DEV)).eval()
cond, info = u.compute_moe_condition(y, training=False)
assert cond is None
x = torch.randn(4, 2, 2560, device=DEV); t = torch.randint(0, 1000, (4,), device=DEV)
o1 = u.denoise(x, y, t, None)
args2, cfg2 = make_cfg(['--disable_moe'])
z = PhysicsMoEModel(cfg2).to(DEV).eval()                                  # 伪零 token 版本
z.load_state_dict(u.state_dict(), strict=False)
cz, _ = z.compute_moe_condition(y, training=False)
assert cz.abs().sum() == 0 and cz.shape[1] == 1
o2 = z.denoise(x, y, t, cz)
print('PASS test_sampler_reproducibility；伪零 token 与真无条件的输出差 = %.3e（>0 说明旧“无 MoE”基线并非真正无条件）' % (o1 - o2).abs().max().item())
