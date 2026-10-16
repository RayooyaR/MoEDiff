"""实验 2：共享专家。始终激活（不参与路由），从原始 I/Q（外加包络）提取各干扰类型都要用到的“目标通信信号共性”特征。
输出 (B, 1, D) 单个 token。轻量 1D CNN：3 通道(I, Q, |c|) → 两层步长 2 卷积 → 均值+最大池化 → 线性。"""
import torch
import torch.nn as nn


class SharedSignalExpert(nn.Module):
    def __init__(self, in_channels=2, feature_dim=256, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels + 1, hidden, 7, stride=2, padding=3), nn.GroupNorm(4, hidden), nn.SiLU(),
            nn.Conv1d(hidden, 2 * hidden, 5, stride=2, padding=2), nn.GroupNorm(4, 2 * hidden), nn.SiLU(),
            nn.Conv1d(2 * hidden, 2 * hidden, 3, stride=2, padding=1), nn.GroupNorm(4, 2 * hidden), nn.SiLU())
        self.proj = nn.Linear(4 * hidden, feature_dim)

    def forward(self, time_feat):
        env = torch.sqrt(time_feat.pow(2).sum(1, keepdim=True) + 1e-12)
        h = self.net(torch.cat([time_feat, env], dim=1))
        return self.proj(torch.cat([h.mean(-1), h.amax(-1)], dim=1)).unsqueeze(1)
