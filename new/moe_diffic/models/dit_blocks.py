import math
import numpy as np
import torch
import torch.nn as nn

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_size = freq_size

    def forward(self, t):
        half = self.freq_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # 新增：Cross-Attention，用于主动查询 MoE 条件序列
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # AdaLN 扩大为 9 块参数 (支持 Self-Attn, Cross-Attn, MLP 的平移/缩放/门控)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        )

    def forward(self, x, t_cond, moe_cond_seq=None, cond_mask=None):
        # t_cond 引导 adaLN
        adaln_params = self.adaLN(t_cond).chunk(9, dim=1)
        shift_a, scale_a, gate_a = adaln_params[0], adaln_params[1], adaln_params[2]
        shift_c, scale_c, gate_c = adaln_params[3], adaln_params[4], adaln_params[5]
        shift_m, scale_m, gate_m = adaln_params[6], adaln_params[7], adaln_params[8]

        # 1. 自注意力
        h = self.norm1(x) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        h, _ = self.self_attn(h.contiguous(), h.contiguous(), h.contiguous(), need_weights=False)
        x = x + gate_a.unsqueeze(1) * h

        # 2. 交叉注意力注入 MoE 序列知识
        if moe_cond_seq is not None:
            h_c = self.norm_cross(x) * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)
            # Query 是隐状态 h_c，Key 和 Value 是专家序列 moe_cond_seq
            h_c, _ = self.cross_attn(query=h_c, key=moe_cond_seq, value=moe_cond_seq, key_padding_mask=cond_mask, need_weights=False)   # cond_mask: True=屏蔽（真正的 Top-K 稀疏）
            x = x + gate_c.unsqueeze(1) * h_c

        # 3. MLP
        h = self.norm2(x) * (1 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_m.unsqueeze(1) * h

        return x

class DiTDenoiser1D(nn.Module):
    def __init__(self, in_channels, out_channels, signal_length, hidden_size, depth, num_heads, mlp_ratio=4.0, patch_size=4, cond_mode='xattn'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.x_embedder = nn.Conv1d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        num_patches = signal_length // patch_size
        self.num_patches = num_patches

        self.t_embedder = TimestepEmbedder(hidden_size)

        # 注销原有的 cond_fusion (不再将时间步和MoE特征直接拼接)
        # self.cond_fusion = ...

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, patch_size * out_channels)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

        # [exp] 条件注入方式：xattn 为原行为（不新增参数）；film/adaln/none 见 condition_adapter.py
        self.cond_mode = cond_mode
        self.adapter = None
        if cond_mode != 'xattn':
            from .condition_adapter import ConditionAdapter
            self.adapter = ConditionAdapter(cond_mode, hidden_size, hidden_size, depth)

        self._init_weights(num_patches, hidden_size)

    def _init_weights(self, num_patches, hidden_size):
        pos = np.arange(num_patches, dtype=np.float32)
        omega = np.arange(hidden_size // 2, dtype=np.float64)
        omega /= hidden_size / 2.0
        omega = 1.0 / 10000 ** omega
        out = np.einsum('m,d->md', pos, omega)
        emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        self.pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(self, x, t, moe_cond_seq=None, cond_mask=None):
        x = self.x_embedder(x).transpose(1, 2) + self.pos_embed  # (B, num_patches, hidden)
        
        t_emb = self.t_embedder(t) # (B, hidden_size)

        g = None
        if self.adapter is not None:          # [exp] film/adaln/none：条件不走交叉注意力
            g = self.adapter.prepare(moe_cond_seq, cond_mask)
            tb = self.adapter.t_bias(g)
            if tb is not None:
                t_emb = t_emb + tb
            moe_cond_seq, cond_mask = None, None

        for i, block in enumerate(self.blocks):
            if self.adapter is not None:
                x = self.adapter.film(i, g, x)
            # 分开传递时间步向量 t_emb 与专家序列 moe_cond_seq
            x = block(x, t_cond=t_emb, moe_cond_seq=moe_cond_seq, cond_mask=cond_mask)

        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=1) # 最终层依然只依赖时间步
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)  

        p = self.patch_size
        c_out = self.out_channels
        x = x.reshape(x.shape[0], -1, p, c_out)  
        x = x.permute(0, 3, 1, 2)                 
        x = x.reshape(x.shape[0], c_out, -1)      
        return x