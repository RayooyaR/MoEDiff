import torch
import torch.nn as nn
import torch.nn.functional as F

from .domain_experts import (
    DOMAIN_EXPERT_SPECS,
    ALL_DOMAIN_EXPERT_NAMES,
    ExpertType,
)
from .biased_router import BiasedRouter


class PhysicsMoE(nn.Module):
    """
    PhysicsMoE（动态路由序列表达版）
    使用 BiasedRouter 进行倾向性门控，输出 (B, num_experts, D) 序列。
    """

    def __init__(self, config):
        super().__init__()

        in_channels = config.data.num_channels
        signal_length = config.data.signal_length
        n_fft = getattr(config.modal, "n_fft", 32)
        hop_length = getattr(config.modal, "hop_length", 8)
        n_scales = getattr(config.modal, "n_scales", 3)

        self.feature_dim = getattr(config.moe, "expert_feature_dim", 256)
        
        active_names = getattr(config.moe, "active_experts", None)
        if active_names is None:
            active_names = ALL_DOMAIN_EXPERT_NAMES

        self.active_expert_names = list(active_names)
        self.num_experts = len(self.active_expert_names)

        self.experts = nn.ModuleDict()
        for name in self.active_expert_names:
            cls = DOMAIN_EXPERT_SPECS[name]["class"]
            if name == ExpertType.FREQUENCY_DOMAIN:
                self.experts[name] = cls(in_channels=in_channels, signal_length=signal_length, feature_dim=self.feature_dim)
            elif name == ExpertType.MULTI_SCALE:
                self.experts[name] = cls(in_channels=in_channels, signal_length=signal_length, n_scales=n_scales, feature_dim=self.feature_dim)
            elif name == ExpertType.TIME_FREQUENCY:
                self.experts[name] = cls(in_channels=in_channels, signal_length=signal_length, n_fft=n_fft, hop_length=hop_length, feature_dim=self.feature_dim)
            elif name == ExpertType.TIME_DOMAIN:
                self.experts[name] = cls(in_channels=in_channels, signal_length=signal_length, feature_dim=self.feature_dim)

        # 引入无监督倾向性路由
        self.router = BiasedRouter(
            in_channels=in_channels,
            signal_length=signal_length,
            num_experts=self.num_experts,
            temperature=0.1
        )

        self.diversity_coef = getattr(config.moe, "diversity_coef", 0.05)

    @staticmethod
    def _compute_diversity_loss(expert_tokens):
        """计算专家特征正交/多样性损失"""
        bsz, k, _ = expert_tokens.shape
        if k <= 1:
            return expert_tokens.new_tensor(0.0)

        x = F.normalize(expert_tokens, p=2, dim=-1)  
        sim = torch.bmm(x, x.transpose(1, 2))        

        mask = torch.triu(torch.ones(k, k, device=expert_tokens.device, dtype=torch.bool), diagonal=1)
        return sim[:, mask].mean()

    def forward(self, modal_dict, training=True):
        time_feat = modal_dict["time_feat"]
        freq_feat = modal_dict["freq_feat"]
        stft_feat = modal_dict["stft_feat"]
        ms_feat = modal_dict["ms_feat"]

        # 1. 路由感知获取权重
        gates = self.router(time_feat, training=training)  # (B, num_experts)

        # 2. 专家特征提取
        expert_features = []
        for name in self.active_expert_names:
            expert = self.experts[name]
            if name == ExpertType.FREQUENCY_DOMAIN:
                feat = expert(freq_feat, time_feat)      
            elif name == ExpertType.MULTI_SCALE:
                feat = expert(ms_feat, time_feat)        
            elif name == ExpertType.TIME_FREQUENCY:
                feat = expert(stft_feat, time_feat)      
            elif name == ExpertType.TIME_DOMAIN:
                feat = expert(time_feat)                 
            expert_features.append(feat)

        x_experts = torch.stack(expert_features, dim=1)  # (B, K, D)

        # 3. 加权构建条件序列 (Weighted Token Sequence)
        # 注意：此处是乘法，不进行求和压缩！
        moe_cond_seq = x_experts * gates.unsqueeze(-1)   # (B, K, D)

        # 4. 计算多样性损失（正交损失）
        diversity_loss = self.diversity_coef * self._compute_diversity_loss(x_experts)

        balance_loss = moe_cond_seq.new_tensor(0.0)

        # 返回的是序列 moe_cond_seq
        return moe_cond_seq, balance_loss, gates, diversity_loss