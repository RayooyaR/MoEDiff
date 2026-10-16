"""实验 9：条件注入方式。把 MoE 的条件 token 序列 (B, K, D) 注入去噪网络的三种方式：

  xattn  交叉注意力（现有做法；token 逐个参与，可按位置择优）
  film   FiLM：条件池化成全局向量，逐层产生 (scale, shift) 对隐状态做仿射调制（Perez et al., AAAI 2018）
  adaln  AdaLN：条件池化成全局向量，加到时间步嵌入上，通过各层 adaLN 参数间接调制（DiT 的做法）
  none   不注入（无条件对照）

池化时使用 mask（True=被屏蔽的 token 不参与）。FiLM/AdaLN 的输出层零初始化，保证训练开始时等价于无条件。
"""
import torch
import torch.nn as nn

MODES = ('xattn', 'film', 'adaln', 'none')


def masked_mean(cond, mask=None):
    """cond: (B,K,D)；mask: (B,K) bool，True=屏蔽。返回 (B,D)。"""
    if mask is None:
        return cond.mean(1)
    keep = (~mask).unsqueeze(-1).to(cond.dtype)
    return (cond * keep).sum(1) / keep.sum(1).clamp_min(1.0)


class ConditionAdapter(nn.Module):
    def __init__(self, mode, cond_dim, hidden, n_layers):
        super().__init__()
        if mode not in MODES:
            raise ValueError('cond_mode must be one of %s' % (MODES,))
        self.mode, self.n_layers = mode, n_layers
        if mode in ('film', 'adaln'):
            self.pool_proj = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU())
        if mode == 'adaln':
            self.to_t = nn.Linear(hidden, hidden)
            nn.init.zeros_(self.to_t.weight); nn.init.zeros_(self.to_t.bias)
        if mode == 'film':
            self.to_film = nn.ModuleList([nn.Linear(hidden, 2 * hidden) for _ in range(n_layers)])
            for m in self.to_film:
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    @property
    def use_xattn(self):
        return self.mode == 'xattn'

    def prepare(self, cond, mask=None):
        """返回全局向量 g (B,hidden) 或 None。"""
        if self.mode in ('film', 'adaln') and cond is not None:
            return self.pool_proj(masked_mean(cond, mask))
        return None

    def t_bias(self, g):
        return self.to_t(g) if (self.mode == 'adaln' and g is not None) else None

    def film(self, layer, g, x):
        """x: (B, N, C) 或 (B, C, L)（按最后一维是否等于通道数判断由调用方负责，这里只接受 (B,N,C)）。"""
        if self.mode != 'film' or g is None:
            return x
        scale, shift = self.to_film[layer](g).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def film_ch(self, layer, g, x):
        """x: (B, C, L)。"""
        if self.mode != 'film' or g is None:
            return x
        scale, shift = self.to_film[layer](g).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
