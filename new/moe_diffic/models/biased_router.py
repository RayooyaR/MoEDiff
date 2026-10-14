import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasedRouter(nn.Module):
    """
    无监督倾向性路由器 (Unsupervised Biased Router)
    基于输入信号动态分配专家权重，训练时使用 Gumbel-Softmax 促进探索，推理时使用带温度的 Softmax 强化偏好。
    """
    def __init__(self, in_channels=2, signal_length=128, num_experts=4, temperature=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        
        # 轻量级特征提取器：从时域信号粗略感知干扰类型
        self.condition_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 路由分类器
        self.router_linear = nn.Linear(64, num_experts)

    def forward(self, x_signal, training=True):
        """
        Args:
            x_signal: (B, 2, L) 时域输入信号
            training: 训练/推理标志
        Returns:
            weights: (B, num_experts) 倾向性权重
        """
        h = self.condition_extractor(x_signal)      # (B, 64)
        logits = self.router_linear(h)              # (B, num_experts)
        
        if training:
            # 训练阶段：引入 Gumbel 噪声，保证探索能力，避免路由坍塌 (hard=False 保证梯度回传)
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            # 推理阶段：温度缩放，强化选择倾向性
            weights = F.softmax(logits / self.temperature, dim=-1)
            
        return weights