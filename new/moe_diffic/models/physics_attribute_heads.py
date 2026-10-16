"""实验 3：物理属性辅助头。每个专家的（门控前）特征经一个小 MLP 预测它所对应的 3 个物理属性（见 loader/attributes.py）。
标签来自训练集的干扰信号（连续物理量，不含类别信息）。用于：联合训练 / 预训练后冻结 / 预训练后微调；
也可对冻结特征训练线性探针做跨类型迁移检验（probe_features.py）。"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'loader'))
from attributes import ATTR_GROUPS  # noqa: E402


class PhysicsAttributeHeads(nn.Module):
    def __init__(self, feature_dim, expert_names, hidden=128):
        super().__init__()
        self.names = [n for n in expert_names if n in ATTR_GROUPS]
        self.heads = nn.ModuleDict({n: nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, len(ATTR_GROUPS[n])))
                                    for n in self.names})

    def forward(self, feats):
        """feats: {expert_name: (B, D)} -> {expert_name: (B, 3)}"""
        return {n: self.heads[n](feats[n]) for n in self.names if n in feats}

    def loss(self, preds, target):
        """target: (B, 12) 标准化后的属性；返回 (总损失, {专家: 损失})"""
        per = {n: torch.nn.functional.mse_loss(p.float(), target[:, ATTR_GROUPS[n]].float()) for n, p in preds.items()}
        tot = sum(per.values()) / max(len(per), 1)
        return tot, per
