import torch
import torch.nn as nn
import torch.nn.functional as F

from .domain_experts import (
    DOMAIN_EXPERT_SPECS,
    ALL_DOMAIN_EXPERT_NAMES,
    ExpertType,
)
from .biased_router import BiasedRouter
from .prior_router import PriorGuidedRouter
from .dual_factor_router import DualFactorRouter
from .shared_signal_expert import SharedSignalExpert
from .physics_attribute_heads import PhysicsAttributeHeads


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
                self.experts[name] = cls(in_channels=in_channels, signal_length=signal_length, n_fft=n_fft, hop_length=hop_length, feature_dim=self.feature_dim,
                                         n_tokens=int(getattr(config.moe, 'tf_tokens', 1)))
            elif name == ExpertType.TIME_DOMAIN:
                self.experts[name] = cls(in_channels=in_channels, signal_length=signal_length, feature_dim=self.feature_dim)
            elif name in ExpertType.GENERIC or name in (ExpertType.CYCLO, ExpertType.CYCLO_CONTROL):
                self.experts[name] = cls(in_channels=in_channels, signal_length=signal_length, feature_dim=self.feature_dim)

        # [new-data] router_type: legacy=原无监督 BiasedRouter；prior=先验引导的稀疏路由 PriorGuidedRouter
        self.router_type = getattr(config.moe, "router_type", "legacy")
        self.last_route_logits = None
        if self.router_type == "prior":
            self.router = PriorGuidedRouter(
                num_experts=self.num_experts,
                top_k=getattr(config.moe, "k", 2),
                noise_std=getattr(config.moe, "router_noise", 0.2),
            )
        elif self.router_type == "dual":      # [exp] 实验 4：形态—强度双因素路由
            self.router = DualFactorRouter(
                num_experts=self.num_experts,
                top_k=getattr(config.moe, "k", 2),
                noise_std=getattr(config.moe, "router_noise", 0.2),
                tau=getattr(config.moe, "dual_tau", 0.5),
                variant=getattr(config.moe, "dual_variant", "dual"),
            )
        else:
            self.router = BiasedRouter(
                in_channels=in_channels,
                signal_length=signal_length,
                num_experts=self.num_experts,
                temperature=0.1
            )

        # ---- [exp] 可选扩展（默认全部关闭，结构与原版逐位一致）----
        self.fusion = getattr(config.moe, "fusion", "xattn")          # xattn | uniform | sum_single | topk_mask
        if self.fusion not in ("xattn", "uniform", "sum_single", "topk_mask", "anchor"):
            raise ValueError("fusion must be one of xattn|uniform|sum_single|topk_mask|anchor")
        if self.fusion == "anchor":                      # 锚定残差融合：TF 专家为锚(权重恒为 1)，其余专家的贡献 α_k 由零初始化的静态标量 + 动态门控决定 -> 训练起点等价于 TF 单专家
            assert ExpertType.TIME_FREQUENCY in self.active_expert_names and self.num_experts >= 2, "anchor 融合需要 TF 专家和至少一个其它专家"
            self.anchor_idx = self.active_expert_names.index(ExpertType.TIME_FREQUENCY)
            n_other = self.num_experts - 1
            self.anchor_a = nn.Parameter(torch.zeros(n_other))
            self.anchor_gate = nn.Linear(self.num_experts * self.feature_dim, n_other)
            nn.init.zeros_(self.anchor_gate.weight); nn.init.zeros_(self.anchor_gate.bias)
        self.shared_expert = SharedSignalExpert(in_channels=in_channels, feature_dim=self.feature_dim) if getattr(config.moe, "shared_expert", False) else None
        self.attr_heads = PhysicsAttributeHeads(self.feature_dim, self.active_expert_names) if getattr(config.moe, "attr_heads", False) else None
        tf_tok = int(getattr(config.moe, "tf_tokens", 1)) if ExpertType.TIME_FREQUENCY in self.active_expert_names else 1
        group = []
        for ei, name in enumerate(self.active_expert_names):
            group += [ei] * (tf_tok if name == ExpertType.TIME_FREQUENCY else 1)
        self.register_buffer("group_idx", torch.tensor(group, dtype=torch.long), persistent=False)
        self.aux_in = {}      # 由模型在前向前写入：sir (B,), attrs (B,12)（训练期标签，仅用于辅助监督/oracle 分析）
        self.aux_out = {}     # 前向后读取：辅助损失与诊断量
        self.last_mask = None
        self.last_expert_feats = {}

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
        route_out = None
        self.aux_out = {}
        if self.router_type == "dual":
            route_out = self.router(time_feat, training=training, aux=self.aux_in)
            gates = route_out["gates"]
            self.last_route_logits = route_out["logits"]
            self.aux_out["sir_mu"], self.aux_out["sir_logvar"] = route_out["sir_mu"], route_out["sir_logvar"]
            if self.aux_in.get("sir") is not None:
                self.aux_out["sir_nll"] = DualFactorRouter.sir_nll(route_out, self.aux_in["sir"])
            if self.aux_in.get("attrs") is not None:
                self.aux_out["morph_attr"] = torch.nn.functional.mse_loss(route_out["attr_pred"].float(), self.aux_in["attrs"].float())
        elif self.router_type == "prior":
            route_out = self.router(time_feat, training=training)
            gates = route_out["gates"]                       # 稀疏 top-k，未选中专家权重为 0
            self.last_route_logits = route_out["logits"]     # 供训练期的路由监督损失使用
        else:
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
            elif name == ExpertType.TIME_DOMAIN or name in ExpertType.GENERIC or name in (ExpertType.CYCLO, ExpertType.CYCLO_CONTROL):
                feat = expert(time_feat)                 
            expert_features.append(feat)

        # 每个专家一个池化特征 (B, K, D)（TF 为多 token 时取均值），用于多样性损失与属性头
        pooled = [f if f.ndim == 2 else f.mean(1) for f in expert_features]
        x_pooled = torch.stack(pooled, dim=1)
        x_experts = torch.cat([f.unsqueeze(1) if f.ndim == 2 else f for f in expert_features], dim=1)   # (B, Ktok, D)
        self.last_expert_feats = {n: p for n, p in zip(self.active_expert_names, pooled)}

        # 3. 加权构建条件序列 (Weighted Token Sequence)
        # 注意：此处是乘法，不进行求和压缩！
        gates_tok = gates[:, self.group_idx]
        mask = None
        if self.fusion == "anchor":
            others = [i for i in range(self.num_experts) if i != self.anchor_idx]
            alpha = torch.tanh(self.anchor_a + self.anchor_gate(x_pooled.flatten(1)))                    # (B, K-1)，起点为 0
            comb = x_pooled[:, self.anchor_idx] + (alpha.unsqueeze(-1) * x_pooled[:, others]).sum(1)      # 求和而非平均：不稀释 TF 信号
            moe_cond_seq = comb.unsqueeze(1)
            full = torch.ones(x_pooled.shape[0], self.num_experts, device=alpha.device, dtype=alpha.dtype)
            full[:, others] = alpha
            gates = full                                                                                  # 记录用：锚=1，其余=α
            mask = None
            self.aux_out["anchor_alpha"] = alpha.detach()
        elif self.fusion == "uniform":                      # 静态融合：不使用门控
            moe_cond_seq = x_experts
        elif self.fusion == "sum_single":                 # 门控加权求和成单个 token
            moe_cond_seq = (x_pooled * gates.unsqueeze(-1)).sum(1, keepdim=True)
        elif self.fusion == "topk_mask":                  # 真正的稀疏：门控为 0 的专家 token 被交叉注意力屏蔽
            moe_cond_seq = x_experts * gates_tok.unsqueeze(-1)
            mask = gates_tok <= 0
        else:
            moe_cond_seq = x_experts * gates_tok.unsqueeze(-1)   # (B, K, D)
        if self.shared_expert is not None:                # 共享专家：始终激活，排在序列最前
            sh = self.shared_expert(time_feat)
            moe_cond_seq = torch.cat([sh, moe_cond_seq], dim=1)
            if mask is not None:
                mask = torch.cat([torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        self.last_mask = mask

        # [exp] 物理属性辅助头（实验 3）
        if self.attr_heads is not None and self.aux_in.get("attrs") is not None:
            preds = self.attr_heads(self.last_expert_feats)
            tot, per = self.attr_heads.loss(preds, self.aux_in["attrs"])
            self.aux_out["attr_loss"], self.aux_out["attr_per"] = tot, per

        # 4. 计算多样性损失（正交损失）
        diversity_loss = self.diversity_coef * self._compute_diversity_loss(x_pooled)

        balance_loss = route_out["balance_loss"] if route_out is not None else moe_cond_seq.new_tensor(0.0)

        # 返回的是序列 moe_cond_seq
        return moe_cond_seq, balance_loss, gates, diversity_loss