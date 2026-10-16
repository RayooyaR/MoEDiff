"""实验 7：一维卷积类去噪主干，与 DiTDenoiser1D 接口一致：forward(x, t, moe_cond_seq=None, cond_mask=None)。

UNet1DDenoiser   多尺度 1D U-Net（Wave-U-Net 式下采样/上采样 + 跳连；2560→1280→640→320）。
DiffWave1DDenoiser  扩张卷积残差网络（DiffWave，Kong et al., ICLR 2021）；按复基带 I/Q 两通道使用。
条件注入：cond_mode = xattn（交叉注意力，在低分辨率层）| film | adaln | none，见 condition_adapter.py。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_adapter import ConditionAdapter
from .dit_blocks import TimestepEmbedder


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class CrossAttn1D(nn.Module):
    """对 (B,C,L) 特征做交叉注意力（query=特征，key/value=条件 token），输出零初始化的残差。"""

    def __init__(self, ch, cond_dim, heads=4):
        super().__init__()
        self.norm = _gn(ch)
        self.q = nn.Linear(ch, ch)
        self.attn = nn.MultiheadAttention(ch, heads, kdim=cond_dim, vdim=cond_dim, batch_first=True)
        self.out = nn.Linear(ch, ch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, cond, mask=None):
        h = self.norm(x).transpose(1, 2)
        a, _ = self.attn(self.q(h), cond, cond, key_padding_mask=mask, need_weights=False)
        return x + self.out(a).transpose(1, 2)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, tdim, k=3):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv1d(cin, cout, k, padding=k // 2)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = _gn(cout), nn.Conv1d(cout, cout, k, padding=k // 2)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, temb, film=None):
        h = self.c1(F.silu(self.n1(x))) + self.t(temb).unsqueeze(-1)
        if film is not None:
            h = film(h)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet1DDenoiser(nn.Module):
    def __init__(self, in_channels, out_channels, signal_length, hidden_size=256, base=64, mults=(1, 2, 4, 4), n_res=2,
                 heads=4, cond_mode='xattn', attn_levels=(2, 3)):
        super().__init__()
        self.in_channels, self.out_channels, self.hidden_size = in_channels, out_channels, hidden_size
        self.cond_mode = cond_mode
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.inp = nn.Conv1d(in_channels, base, 7, padding=3)
        chs = [base * m for m in mults]
        self.down, self.down_attn, self.ds = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = base; skips = [base]
        n_blocks = 0
        for li, co in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(n_res):
                blocks.append(ResBlock1D(c, co, hidden_size)); c = co; skips.append(c); n_blocks += 1
            self.down.append(blocks)
            self.down_attn.append(CrossAttn1D(c, hidden_size, heads) if (cond_mode == 'xattn' and li in attn_levels) else nn.Identity())
            if li < len(chs) - 1:
                self.ds.append(nn.Conv1d(c, c, 3, stride=2, padding=1)); skips.append(c)
        self.mid1, self.mid2 = ResBlock1D(c, c, hidden_size), ResBlock1D(c, c, hidden_size); n_blocks += 2
        self.mid_attn = CrossAttn1D(c, hidden_size, heads) if cond_mode == 'xattn' else nn.Identity()
        self.up, self.up_attn, self.us = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for li in reversed(range(len(chs))):
            co = chs[li]
            blocks = nn.ModuleList()
            for _ in range(n_res + 1):
                blocks.append(ResBlock1D(c + skips.pop(), co, hidden_size)); c = co; n_blocks += 1
            self.up.append(blocks)
            self.up_attn.append(CrossAttn1D(c, hidden_size, heads) if (cond_mode == 'xattn' and li in attn_levels) else nn.Identity())
            if li > 0:
                self.us.append(nn.Conv1d(c, c, 3, padding=1))
        self.out_norm, self.out = _gn(c), nn.Conv1d(c, out_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.adapter = ConditionAdapter(cond_mode, hidden_size, hidden_size, 1) if cond_mode != 'xattn' else None     # film 的逐块投影见 film_proj
        # film 作用在各 ResBlock 的输出通道上，通道数不同，这里为每个块建独立的投影
        self.film_proj = None
        if cond_mode == 'film':
            outs = [b.c1.out_channels for lv in self.down for b in lv] + [self.mid1.c1.out_channels, self.mid2.c1.out_channels] + \
                   [b.c1.out_channels for lv in self.up for b in lv]
            self.film_proj = nn.ModuleList([nn.Linear(hidden_size, 2 * o) for o in outs])
            for m in self.film_proj:
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def _film(self, idx, g):
        if self.film_proj is None or g is None:
            return None
        scale, shift = self.film_proj[idx](g).chunk(2, dim=-1)
        return lambda h: h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

    def forward(self, x, t, moe_cond_seq=None, cond_mask=None):
        temb = self.t_embedder(t)
        g = None
        if self.adapter is not None:
            g = self.adapter.prepare(moe_cond_seq, cond_mask)
            tb = self.adapter.t_bias(g)
            if tb is not None:
                temb = temb + tb
        cond = moe_cond_seq if self.cond_mode == 'xattn' else None
        h = self.inp(x); hs = [h]; bi = 0
        for li, blocks in enumerate(self.down):
            for b in blocks:
                h = b(h, temb, self._film(bi, g)); bi += 1; hs.append(h)
            if cond is not None and not isinstance(self.down_attn[li], nn.Identity):
                h = self.down_attn[li](h, cond, cond_mask)
            if li < len(self.ds):
                h = self.ds[li](h); hs.append(h)
        h = self.mid1(h, temb, self._film(bi, g)); bi += 1
        if cond is not None:
            h = self.mid_attn(h, cond, cond_mask)
        h = self.mid2(h, temb, self._film(bi, g)); bi += 1
        for ui, blocks in enumerate(self.up):
            li = len(self.down) - 1 - ui
            for b in blocks:
                h = b(torch.cat([h, hs.pop()], 1), temb, self._film(bi, g)); bi += 1
            if cond is not None and not isinstance(self.up_attn[ui], nn.Identity):
                h = self.up_attn[ui](h, cond, cond_mask)
            if li > 0:
                h = self.us[ui](F.interpolate(h, scale_factor=2, mode='nearest'))
        return self.out(F.silu(self.out_norm(h)))


class DiffWaveLayer(nn.Module):
    def __init__(self, res, tdim, dilation):
        super().__init__()
        self.dil = nn.Conv1d(res, 2 * res, 3, padding=dilation, dilation=dilation)
        self.t = nn.Linear(tdim, res)
        self.out = nn.Conv1d(res, 2 * res, 1)

    def forward(self, x, temb, bias=None):
        h = self.dil(x + self.t(temb).unsqueeze(-1))
        if bias is not None:
            h = h + bias
        a, b = h.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)
        res, skip = self.out(h).chunk(2, dim=1)
        return (x + res) / math.sqrt(2.0), skip


class DiffWave1DDenoiser(nn.Module):
    def __init__(self, in_channels, out_channels, signal_length, hidden_size=256, res_channels=128, layers=20, cycle=10,
                 heads=4, cond_mode='xattn', attn_layers=(6, 13)):
        super().__init__()
        self.in_channels, self.out_channels, self.hidden_size = in_channels, out_channels, hidden_size
        self.cond_mode = cond_mode
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.inp = nn.Conv1d(in_channels, res_channels, 1)
        self.layers = nn.ModuleList([DiffWaveLayer(res_channels, hidden_size, 2 ** (i % cycle)) for i in range(layers)])
        self.attn = nn.ModuleDict({str(i): CrossAttn1D(res_channels, hidden_size, heads) for i in attn_layers}) if cond_mode == 'xattn' else nn.ModuleDict()
        self.skip_proj = nn.Conv1d(res_channels, res_channels, 1)
        self.out = nn.Conv1d(res_channels, out_channels, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.adapter = None
        if cond_mode in ('film', 'adaln'):
            self.adapter = ConditionAdapter(cond_mode, hidden_size, hidden_size, 1)
            if cond_mode == 'film':
                self.bias_proj = nn.ModuleList([nn.Linear(hidden_size, 2 * res_channels) for _ in range(layers)])
                for m in self.bias_proj:
                    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        elif cond_mode == 'none':
            self.adapter = ConditionAdapter('none', hidden_size, hidden_size, 1)

    def forward(self, x, t, moe_cond_seq=None, cond_mask=None):
        temb = self.t_embedder(t)
        g = None
        if self.adapter is not None:
            g = self.adapter.prepare(moe_cond_seq, cond_mask)
            tb = self.adapter.t_bias(g)
            if tb is not None:
                temb = temb + tb
        cond = moe_cond_seq if self.cond_mode == 'xattn' else None
        h = F.relu(self.inp(x)); skip = 0
        for i, layer in enumerate(self.layers):
            bias = self.bias_proj[i](g).unsqueeze(-1) if (self.cond_mode == 'film' and g is not None) else None
            h, s = layer(h, temb, bias)
            skip = skip + s
            if cond is not None and str(i) in self.attn:
                h = self.attn[str(i)](h, cond, cond_mask)
        return self.out(F.relu(self.skip_proj(skip / math.sqrt(len(self.layers)))))
