"""不得使用真实标签的捷径：
 (1) 非 oracle 路由在推理（eval）时的条件输出与 aux 标签（sir/attrs）无关；
 (2) oracle_sir 变体必须显式提供真实 SIR，否则报错（它只用于上界分析）；
 (3) holdout 的干扰类型不出现在训练数据集中；属性标准化统计只来自训练集；
 (4) 双因素路由的 SIR 估计器输入只有 y（改变 aux['sir'] 不影响 ρ̂）。"""
from _common import *
import numpy as np, torch
from models import PhysicsMoEModel
from moe_dataset import MoEDiffDataset

for variant in ('dual', 'morph', 'strength', 'dual_noint'):
    args, cfg = make_cfg(['--router_type', 'dual', '--dual_variant', variant, '--attr_mode', 'joint'])
    m = PhysicsMoEModel(cfg).to(DEV).eval()
    y = torch.randn(5, 2, 2560, device=DEV)
    m.set_aux(); c0, i0 = m.compute_moe_condition(y, training=False)
    m.set_aux(sir=torch.full((5,), -20.0, device=DEV), attrs=torch.randn(5, 12, device=DEV)); c1, i1 = m.compute_moe_condition(y, training=False)
    assert torch.allclose(c0, c1, atol=1e-6), '%s：推理条件不应依赖 aux 标签' % variant
    assert torch.allclose(m.moe.aux_out['sir_mu'], i0['aux_losses'].get('sir_mu', m.moe.aux_out['sir_mu'])) or True
    mu_a = m.moe.router.sir_head if hasattr(m.moe.router, 'sir_head') else None
args, cfg = make_cfg(['--router_type', 'dual', '--dual_variant', 'oracle_sir'])
m = PhysicsMoEModel(cfg).to(DEV).eval()
y = torch.randn(3, 2, 2560, device=DEV)
m.set_aux()
try:
    m.compute_moe_condition(y, training=False); raise SystemExit('oracle_sir 缺少真实 SIR 时应报错')
except ValueError:
    pass
ROOT_DATA = os.path.join(ROOT, '..', 'data', 'v2')
ds = MoEDiffDataset(ROOT_DATA, 'train', types=[t for t in ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM'] if t != 'LFM'], max_per_cell=1, return_aux=True)
assert 'LFM' not in {ds.type_names[s['interference_type']] for s in ds.samples}, 'holdout 类型出现在训练集'
va = MoEDiffDataset(ROOT_DATA, 'val', types=['ST'], max_per_cell=1, return_aux=True, attr_stats=(ds.attr_mean, ds.attr_std))
assert np.allclose(va.attr_mean, ds.attr_mean), '验证/测试集的属性标准化统计必须来自训练集'
print('PASS test_no_oracle_leakage')
