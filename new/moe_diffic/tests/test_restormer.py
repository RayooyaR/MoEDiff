"""Restormer 1D 基线：形状/梯度/可逆性，以及与官方 2D 实现在 H=1 时的逐模块数值一致性（有官方源码时）。"""
import os, sys, types
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.baseline.restormer_baseline import Restormer1DBaseline, MDTA, GDFN, LayerNorm1d, shuffle1d, unshuffle1d

m = Restormer1DBaseline(dim=16)
x, c = torch.randn(3, 2, 128), torch.randn(3, 2, 128)
r = m(x, c); assert r['pred'].shape == x.shape and torch.isfinite(r['loss'])
r['loss'].backward(); assert all(p.grad is not None for p in m.parameters())
a = torch.randn(2, 6, 32); assert torch.equal(shuffle1d(unshuffle1d(a)), a) and unshuffle1d(a).shape == (2, 12, 16)
assert torch.allclose(m.sample(x), m.net(x))
ref_f = '/root/autodl-tmp/ref_repos/Restormer/basicsr/models/archs/restormer_arch.py'
msg = 'shapes ok'
if os.path.exists(ref_f):
    e = types.ModuleType('einops')
    def rearrange(x, pat, **k):
        if pat == 'b c h w -> b (h w) c': b, c_, h, w = x.shape; return x.permute(0, 2, 3, 1).reshape(b, h * w, c_)
        if pat == 'b (h w) c -> b c h w': b, l, c_ = x.shape; return x.reshape(b, k['h'], k['w'], c_).permute(0, 3, 1, 2)
        if pat == 'b (head c) h w -> b head c (h w)': b, C, h, w = x.shape; return x.reshape(b, k['head'], C // k['head'], h * w)
        if pat == 'b head c (h w) -> b (head c) h w': b, H, c_, l = x.shape; return x.reshape(b, H * c_, k['h'], k['w'])
        raise NotImplementedError(pat)
    e.rearrange = rearrange; sys.modules.setdefault('einops', e)
    ref = types.ModuleType('ref'); exec(compile(open(ref_f).read().replace('from pdb import set_trace as stx', ''), 'ref', 'exec'), ref.__dict__)
    dim, xx = 16, torch.randn(2, 16, 32)
    def conv(sd): return {k: ((v[:, :, 1, :] if v.shape[2] == 3 else v.squeeze(-2)) if v.ndim == 4 else v) for k, v in sd.items()}
    with torch.no_grad():
        for o, n in ((ref.Attention(dim, 2, False), MDTA(dim, 2, False)), (ref.FeedForward(dim, 2.66, False), GDFN(dim, 2.66, False))):
            n.load_state_dict(conv(o.state_dict())); assert (o(xx.unsqueeze(2)).squeeze(2) - n(xx)).abs().max() < 1e-5
        lo, ln = ref.LayerNorm(dim, 'WithBias'), LayerNorm1d(dim, True); ln.load_state_dict({'weight': lo.body.weight, 'bias': lo.body.bias})
        assert (lo(xx.unsqueeze(2)).squeeze(2) - ln(xx)).abs().max() < 1e-5
    msg += '; 与官方 MDTA/GDFN/LayerNorm 数值一致'
print('OK', msg)
