"""
MoE_DiffIC 模型模块

核心组件 (V5.4 完整基线支持):
  - PhysicsMoEModel      : 主模型架构（ModalProjector + PhysicsMoE + DiT）
  - PhysicsMoESampler    : 采样器（封装 DDIM 采样逻辑）
  - PhysicsMoEScheduler  : 调度器（beta schedule 管理）
  - DiT 模块分离         : dit_blocks.py（TimestepEmbedder, DiTBlock, DiTDenoiser1D）
  - domain_experts       : 4 个领域专家 (FD/MS/TF/TD)
  - physics_moe          : 物理感知 MoE 路由器
  - modal_projector      : 多模态特征投影

基线模型 (对比实验):
  - ICDMBaseline         : ICDM扩散基线（纯DiT，无MoE）
  - DiffCom1DBaseline    : DiffCom 迁移基线（DPS后验引导）
  - ICDMDual1DBaseline   : ICDM 迁移基线（双模型分数解耦）
  - PhysicsCNNBaseline   : 纯监督CNN基线（无扩散过程）
  - MatchedFilteringBaseline : 匹配滤波传统基线（无可训练参数）
  - LMMSEBaseline        : 线性最小均方误差自适应滤波基线
"""

from .physics_moe_model import PhysicsMoEModel
from .physics_moe_pipeline import (
    PhysicsMoEScheduler,
    PhysicsMoESampler,
    create_scheduler,
    create_sampler,
)
from .dit_blocks import DiTBlock, DiTDenoiser1D
from .attention_moe_fusion import AttentionMoEFusion

# from .baseline.icdm_baseline import ICDMBaseline, DiffCom1DBaseline, ICDMDual1DBaseline
from .baseline.cnn_baseline import PhysicsCNNBaseline
from .baseline.matched_filtering_baseline import MatchedFilteringBaseline, create_matched_filtering_baseline
from .baseline.lmmse_baseline import LMMSEBaseline, create_lmmse_baseline

__all__ = [
    'PhysicsMoEModel',
    'PhysicsMoEScheduler',
    'PhysicsMoESampler',
    'create_scheduler',
    'create_sampler',
    'DiTBlock',
    'DiTDenoiser1D',
    'AttentionMoEFusion',
    'ICDMBaseline',
    'DiffCom1DBaseline',
    'ICDMDual1DBaseline',
    'PhysicsCNNBaseline',
    'MatchedFilteringBaseline',
    'LMMSEBaseline',
    'create_matched_filtering_baseline',
    'create_lmmse_baseline',
]
