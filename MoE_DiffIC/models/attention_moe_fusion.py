"""
attention_moe_fusion.py
=======================

Step 2 融合层：AttentionMoEFusion

目标：
- 输入: 4 个专家特征堆叠后的 X_experts, 形状 (B, 4, D)
- 输出: 融合条件向量 C_fusion, 形状 (B, D)

机制：
1. 可学习位置编码: pos_embed (1, 4, D)
2. 可学习全局查询: global_query (1, 1, D)
3. Cross-Attention:
   Query = global_query
   Key/Value = X_experts + pos_embed
"""

import torch
import torch.nn as nn


class AttentionMoEFusion(nn.Module):
    """基于可学习全局 Query 的跨专家 Cross-Attention 融合模块。"""

    def __init__(self, feature_dim=256, num_experts=4, num_heads=8, attn_dropout=0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts

        # 四个专家位置信息：FD/MS/TF/TD 的绝对语义编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_experts, feature_dim))

        # 单个全局 Query，表示“当前样本要提取的全局干扰条件”
        self.global_query = nn.Parameter(torch.randn(1, 1, feature_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(feature_dim)

    def forward(self, x_experts, return_attn=False):
        """
        Args:
            x_experts: (B, 4, D)
            return_attn: 是否返回注意力权重

        Returns:
            c_fusion: (B, D)
            attn_weights(optional): (B, 1, 4)
        """
        bsz, num_tokens, dim = x_experts.shape
        if num_tokens != self.num_experts:
            raise ValueError(
                f"Expected num_experts={self.num_experts}, got {num_tokens}."
            )
        if dim != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {dim}."
            )

        # 加位置编码，保留“第几个专家”这一绝对语义
        kv = x_experts + self.pos_embed  # (B, 4, D)

        # 扩展全局 Query 到 batch 维
        q = self.global_query.expand(bsz, -1, -1)  # (B, 1, D)

        # Cross-Attention: Query 从 4 个专家中读取全局融合表示
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=True,
        )

        # 取唯一 query token 输出并做归一化
        c_fusion = self.out_norm(attn_out.squeeze(1))  # (B, D)

        if return_attn:
            return c_fusion, attn_weights  # (B, D), (B, 1, 4)
        return c_fusion
