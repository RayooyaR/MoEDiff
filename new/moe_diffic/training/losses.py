"""
损失函数模块  —  扩散模型训练的组合损失体系

=========================================================================
模块定位 (在整体架构中的角色):
=========================================================================
  本模块在 train.py 的 train() 函数中被实例化和调用:
    criterion = CombinedLoss(config)
    total_loss, loss_dict = criterion(noise_pred, noise_target, gates, balance_loss)

  损失组成 (CombinedLoss 统一编排):
    L_total = w_diff × L_diffusion + L_moe_balance

  其中:
    L_diffusion     (必须): MSE(ε̂, ε) — 噪声预测误差, 扩散模型核心
    L_moe_balance   (可选): CV² 平衡损失 — 避免 MoE 专家坍缩

=========================================================================
各损失类的调用关系:
=========================================================================
  CombinedLoss  (统一入口)
    ├── DiffusionLoss       → MSE / L1 / Huber
    ├── MoEBalanceLoss      → CV²(importance) + CV²(load)
"""

import torch
import numpy as np
import torch.nn as nn


# ============================================================
#  DiffusionLoss  —  扩散模型核心损失
# ============================================================

class DiffusionLoss(nn.Module):
    """
    扩散模型的噪声预测损失

    【物理意义】
      扩散模型训练的核心目标: 学习预测添加的噪声 ε
      L = ||ε̂(x_t, t, c) - ε||²

    支持三种损失函数:
      mse:   ||ε̂ - ε||², 对大误差敏感 (默认)
      l1:    |ε̂ - ε|, 对异常值鲁棒
      huber: SmoothL1, 兼顾两者的优点
    """
    
    def __init__(self, loss_type='mse', reduction='mean'):
        """
        Args:
            loss_type (str):   损失类型 — 'mse' / 'l1' / 'huber'
            reduction (str):   归约方式 — 'mean' / 'sum' / 'none'
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred, target, weights=None):
        """
        Args:
            pred (Tensor):    (B, 2, L) — 预测的噪声 ε̂
            target (Tensor):  (B, 2, L) — 真实噪声 ε
            weights (Tensor): (B,) 可选的样本权重 (如 SNR 加权)

        Returns:
            loss (Tensor): 标量损失
        """
        if weights is None:
            return self.loss_fn(pred, target)
        else:
            # 带权重: element-wise loss × per-sample weight
            loss = (pred - target) ** 2 if self.loss_type == 'mse' else torch.abs(pred - target)
            # weights: (B,) → (B, 1, 1) 广播到 (B, 2, L)
            loss = loss * weights.view(-1, *([1] * (loss.dim() - 1)))
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss


# ============================================================
#  MoEBalanceLoss  —  MoE 负载均衡损失
# ============================================================

class MoEBalanceLoss(nn.Module):
    """
    MoE 负载平衡损失: 防止专家坍缩 (expert collapse)

    【问题背景】
      MoE 训练中常见的退化: 路由器学会只使用 1-2 个专家,
      其余专家得不到梯度, 永远无法被选中 → 富者愈富 (Matthew effect)

    【解决方案: CV² 正则化】
      L_balance = λ × (CV²(importance) + CV²(load))

      importance_i = Σ_x gate_i(x)     每个专家的累积权重
      load_i       = Σ_x 1[gate_i(x)>0] 每个专家被激活的次数

      CV² = Var(x) / Mean(x)² (变异系数的平方)
      - 当所有专家均匀使用时, CV²≈0
      - 当严重不均时, CV² >> 0

    【参数建议】
      balance_coef=0.01: 轻微正则, 不干扰主损失
      过大的 balance_coef 会阻止专家特化
    """
    
    def __init__(self, num_experts, balance_coef=0.01):
        """
        Args:
            num_experts (int):    专家数量 (如 4)
            balance_coef (float): 均衡系数 λ, 默认 0.01
        """
        super().__init__()
        self.num_experts = num_experts
        self.balance_coef = balance_coef
    
    def forward(self, gates):
        """
        Args:
            gates (Tensor):         (B, 4) — MoE 门控权重 (可选, 无 MoE 时为 None)
            balance_loss (Tensor):  标量 — MoE 模块预计算的平衡损失 (可选)
        """
        # 重要性 (importance): 每个专家在整个 batch 上的累积权重
        importance = gates.sum(0)  # (num_experts,)
        
        # 负载 (load): 每个专家被激活 (gate > 0) 的样本数
        load = (gates > 0).float().sum(0)  # (num_experts,)
        
        # 变异系数: 越大表示越不均衡
        cv_importance = self._cv_squared(importance)
        cv_load = self._cv_squared(load)
        
        return self.balance_coef * (cv_importance + cv_load)
    
    def _cv_squared(self, x):
        """
        计算变异系数的平方: CV² = Var(x) / Mean(x)²
        
        CV²=0: 完全均匀  (理想状态)
        CV²↑:  越不均衡
        """
        eps = 1e-10
        if x.numel() <= 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)


# ============================================================
#  CombinedLoss  —  统一组合损失 (train.py 使用)
# ============================================================

class CombinedLoss(nn.Module):
    """
    组合损失: 统一编排所有子损失

    【总损失公式】
      L_total = w_diff × L_diffusion
              + w_recon × L_reconstruction
              + w_spec × L_spectral
              + w_const × L_constellation
              + L_moe_balance

    【默认权重】
      w_diff  = 1.0  (扩散噪声监督, 主导)
      w_recon = 0.5  (显式 clean 波形重建)
      w_spec  = 0.05 (频域一致性, 抑制底噪和谱泄漏)
      w_const = 0.05 (QPSK 星座约束, 主要压低 BER)
      balance_coef = 0.01  (MoE 均衡, 内含在 MoEBalanceLoss 中)

    【与两种模型的兼容性】
    PhysicsMoEDiffusion: 噪声监督 + clean 重建 + 频域 + MoE 均衡
    ICDMBaseline: 噪声监督 + clean 重建 + 频域
    """

    def __init__(self, config):
        """
        Args:
            config (SimpleNamespace): 需包含:
                config.training.loss_type = 'mse'
                config.training.diffusion_loss_weight = 1.0
                config.training.reconstruction_loss_weight = 0.5
                config.training.spectral_loss_weight = 0.05
                config.training.constellation_loss_weight = 0.05
                config.moe.num_experts = 4
                config.moe.balance_loss_coef = 0.01
        """
        super().__init__()

        loss_type = getattr(config.training, 'loss_type', 'mse')
        self.diffusion_loss = DiffusionLoss(loss_type=loss_type)
        self.reconstruction_loss = nn.SmoothL1Loss(reduction='mean')
        self.spectral_loss = nn.SmoothL1Loss(reduction='mean')
        self.moe_balance_loss = MoEBalanceLoss(
            num_experts=config.moe.num_experts,
            balance_coef=config.moe.balance_loss_coef
        )

        self.diffusion_weight = getattr(config.training, 'diffusion_loss_weight', 1.0)
        self.reconstruction_weight = getattr(config.training, 'reconstruction_loss_weight', 0.5)
        self.spectral_weight = getattr(config.training, 'spectral_loss_weight', 0.05)
        self.constellation_weight = getattr(config.training, 'constellation_loss_weight', 0.05)
        self.route_weight = float(getattr(config.training, 'route_loss_weight', 0.0))
        self.constellation_temperature = float(getattr(config.training, 'constellation_temperature', 0.10))
        self.samples_per_symbol = int(getattr(config.data, 'samples_per_symbol', getattr(config.data, 'sps', 16)))
        self.symbol_offset = int(getattr(config.data, 'tau_s', self.samples_per_symbol // 2))

    @staticmethod
    def _weighted_smooth_l1(pred, target, weight):
        # [new-data] 逐样本加权的 SmoothL1：weight (B,)；等价于 reduction='mean' 但每个样本乘以各自权重
        per = torch.nn.functional.smooth_l1_loss(pred, target, reduction='none')
        per = per.flatten(1).mean(dim=1)
        return (weight.to(per.dtype) * per).mean()

    def _spectral_magnitude_loss(self, pred, target, weight=None):
        pred_spec = torch.log1p(torch.abs(torch.fft.rfft(pred, dim=-1)))
        target_spec = torch.log1p(torch.abs(torch.fft.rfft(target, dim=-1)))
        if weight is not None:
            return self._weighted_smooth_l1(pred_spec, target_spec, weight)
        return self.spectral_loss(pred_spec, target_spec)

    def _qpsk_constellation_loss(self, x0_pred: torch.Tensor) -> torch.Tensor:
        start = int(self.symbol_offset % self.samples_per_symbol)
        symbols = x0_pred[:, :, start::self.samples_per_symbol]
        if symbols.numel() == 0:
            return torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)

        sym = symbols.permute(0, 2, 1).reshape(-1, 2)
        if sym.shape[-1] != 2:
            return torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)

        scale = float(np.sqrt(2.0))
        centers = torch.tensor([
            [1.0 / scale, 1.0 / scale],
            [1.0 / scale, -1.0 / scale],
            [-1.0 / scale, 1.0 / scale],
            [-1.0 / scale, -1.0 / scale],
        ], device=x0_pred.device, dtype=x0_pred.dtype)

        dist2 = ((sym.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=-1)
        probs = torch.softmax(-dist2 / max(self.constellation_temperature, 1e-4), dim=-1)
        return (probs * dist2).sum(dim=-1).mean()

    def forward(self, noise_pred, noise_target, x0_pred=None, x0_target=None, gates=None, balance_loss=None,
                x0_weight=None, route_logits=None, route_target=None, diversity_loss=None):
        """
        计算总损失及各子损失

        Args:
            noise_pred (Tensor):    (B, 2, L) — 模型预测的噪声 ε̂
            noise_target (Tensor):  (B, 2, L) — 真实噪声 ε
            x0_pred (Tensor):       (B, 2, L) — 由噪声预测还原的 clean 估计
            x0_target (Tensor):     (B, 2, L) — 真实 clean 信号
            gates (Tensor):         (B, 4) — MoE 门控权重 (可选, 无 MoE 时为 None)
            balance_loss (Tensor):  标量 — MoE 模块预计算的平衡损失 (可选)

        Returns:
            total (Tensor):     标量总损失
            losses (dict):      各子损失字典 — {'diffusion', 'reconstruction', 'spectral', 'moe_balance', 'total'}
        """
        losses = {}
        losses['diffusion'] = self.diffusion_loss(noise_pred, noise_target)

        if x0_pred is not None and x0_target is not None:
            # [new-data] x0_weight (B,)：对 x0 空间损失按样本加权。x0_pred=(x_t-√(1-ᾱ)ε̂)/√ᾱ 在 t→T 时被 1/√ᾱ 放大，
            # 乘以 ᾱ_t 后重建项变为 (1-ᾱ_t)·‖ε̂-ε‖² 量级，不再爆炸。None 表示保持原行为。
            if x0_weight is not None:
                losses['reconstruction'] = self._weighted_smooth_l1(x0_pred, x0_target, x0_weight)
            else:
                losses['reconstruction'] = self.reconstruction_loss(x0_pred, x0_target)
            losses['spectral'] = self._spectral_magnitude_loss(x0_pred, x0_target, weight=x0_weight)
        else:
            zero = torch.tensor(0.0, device=noise_pred.device)
            losses['reconstruction'] = zero
            losses['spectral'] = zero

        if x0_pred is not None and self.constellation_weight > 0:   # [new-data] 权重为 0 时不计算
            losses['constellation'] = self._qpsk_constellation_loss(x0_pred)
        else:
            losses['constellation'] = torch.tensor(0.0, device=noise_pred.device)

        if balance_loss is not None and torch.is_tensor(balance_loss):
            bl = balance_loss.mean() if balance_loss.numel() > 1 else balance_loss
            if bl.item() > 0:
                losses['moe_balance'] = self.moe_balance_loss.balance_coef * bl
            elif gates is not None:
                losses['moe_balance'] = self.moe_balance_loss(gates)
            else:
                losses['moe_balance'] = torch.tensor(0.0, device=noise_pred.device)
        elif gates is not None:
            losses['moe_balance'] = self.moe_balance_loss(gates)
        else:
            losses['moe_balance'] = torch.tensor(0.0, device=noise_pred.device)

        # [new-data] 多样性损失：仅在显式传入时计入（原代码算出后被丢弃）
        losses['diversity'] = diversity_loss if diversity_loss is not None else torch.tensor(0.0, device=noise_pred.device)
        # [new-data] 路由监督：softmax(logits) 与先验分布的交叉熵（soft label）
        if route_logits is not None and route_target is not None:
            losses['route'] = -(route_target * torch.log_softmax(route_logits, dim=-1)).sum(dim=-1).mean()
        else:
            losses['route'] = torch.tensor(0.0, device=noise_pred.device)

        total = (
            self.diffusion_weight * losses['diffusion'] +
            self.reconstruction_weight * losses['reconstruction'] +
            self.spectral_weight * losses['spectral'] +
            self.constellation_weight * losses['constellation'] +
            losses['moe_balance'] +
            self.route_weight * losses['route'] +
            losses['diversity']
        )
        losses['total'] = total

        return total, losses
