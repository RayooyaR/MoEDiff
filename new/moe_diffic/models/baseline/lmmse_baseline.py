"""
Linear Minimum Mean Square Error (LMMSE) Baseline — 自适应滤波基线
===================================================================

本实现尽量贴近论文中的定义：
- 将观测信号 y 建模为目标信号 s 与干扰项叠加
- 使用二阶统计量估计线性最优滤波器
- 使用 kappa^2 作为干扰强度的物理量，并允许设置 omega = omega_scale * kappa^2
- 对于合成数据，kappa^2 可由 clean / noisy 训练批次自动估计

接口保持与 train.py / evaluate_*.py 兼容：
- forward(clean, noisy) -> 更新统计量，返回空字典
- sample(noisy) -> 输出 LMMSE 估计信号
"""

from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LMMSEBaseline(nn.Module):
    """Windowed LMMSE baseline with kappa^2-aware regularization."""

    def __init__(self, config):
        super().__init__()

        self.signal_length = int(config.data.signal_length)
        self.num_channels = int(config.data.num_channels)

        default_window = min(self.signal_length, 256)
        self.window_length = int(getattr(config, 'lmmse_window_length', default_window))
        self.overlap = int(getattr(config, 'lmmse_overlap', max(1, self.window_length // 2)))
        self.regularization = getattr(config, 'lmmse_reg', None)
        self.omega_scale = float(getattr(config, 'lmmse_omega_scale', 1.0))
        self._update_rate = float(getattr(config, 'lmmse_update_rate', 0.01))
        self.use_kappa2_regularization = bool(getattr(config, 'lmmse_use_kappa2_reg', True))
        self.kappa2_momentum = float(getattr(config, 'lmmse_kappa2_momentum', 0.9))
        self.fixed_kappa2 = getattr(config, 'lmmse_kappa2', None)

        if self.window_length <= 0:
            raise ValueError('lmmse_window_length must be positive')
        if self.overlap < 0 or self.overlap >= self.window_length:
            raise ValueError('lmmse_overlap must satisfy 0 <= overlap < window_length')

        dim = self.window_length * self.num_channels
        self.register_buffer('R_xx', torch.eye(dim) * 0.1)
        self.register_buffer('R_yy', torch.eye(dim) * 1.0)
        self.register_buffer('R_xy', torch.zeros(dim, dim))
        self.register_buffer('samples_seen', torch.tensor(0, dtype=torch.long))
        self.register_buffer('kappa2_est', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('kappa2_updates', torch.tensor(0, dtype=torch.long))

        step = max(1, self.window_length - self.overlap)
        self.window_starts = list(range(0, max(1, self.signal_length - self.window_length + 1), step))
        if not self.window_starts:
            self.window_starts = [0]

    def _extract_windows(self, signal: torch.Tensor) -> torch.Tensor:
        """Extract flattened IQ windows of shape (B, num_windows, 2*W)."""
        if signal.ndim != 3:
            raise ValueError(f'Expected input shape (B, 2, L), got {tuple(signal.shape)}')

        B, C, L = signal.shape
        windows = []

        for start in self.window_starts:
            end = start + self.window_length
            if end <= L:
                window = signal[:, :, start:end]
            else:
                padded = F.pad(signal, (0, end - L))
                window = padded[:, :, start:end]
            window_flat = window.transpose(1, 2).contiguous().view(B, -1)
            windows.append(window_flat)

        if not windows:
            padded_signal = F.pad(signal, (0, max(0, self.window_length - L)))
            window = padded_signal[:, :, :self.window_length]
            return window.transpose(1, 2).contiguous().view(B, 1, -1)

        return torch.stack(windows, dim=1)

    def _estimate_batch_kappa2(self, clean_signals: torch.Tensor, noisy_signals: torch.Tensor) -> float:
        """Estimate kappa^2 from a clean/noisy training batch."""
        interference = noisy_signals - clean_signals
        signal_power = torch.mean(clean_signals.float() ** 2)
        int_power = torch.mean(interference.float() ** 2)
        return float((int_power / (signal_power + 1e-12)).item())

    def _update_kappa2_estimate(self, clean_signals: torch.Tensor, noisy_signals: torch.Tensor) -> None:
        if self.fixed_kappa2 is not None:
            self.kappa2_est.fill_(float(self.fixed_kappa2))
            self.kappa2_updates += 1
            return

        if not self.use_kappa2_regularization:
            return

        batch_kappa2 = self._estimate_batch_kappa2(clean_signals, noisy_signals)
        if int(self.kappa2_updates.item()) == 0:
            new_value = batch_kappa2
        else:
            old_value = float(self.kappa2_est.item())
            new_value = self.kappa2_momentum * old_value + (1.0 - self.kappa2_momentum) * batch_kappa2
        self.kappa2_est.fill_(float(new_value))
        self.kappa2_updates += 1

    def update_covariance_matrices(self, clean_signals: torch.Tensor, noisy_signals: torch.Tensor):
        """Update covariance estimates from a batch of clean/noisy windows."""
        x_windows = self._extract_windows(clean_signals)
        y_windows = self._extract_windows(noisy_signals)

        B, num_windows, dim = x_windows.shape
        alpha = float(self._resolve_update_rate())

        for b in range(B):
            for w in range(num_windows):
                x = x_windows[b, w:w + 1, :]
                y = y_windows[b, w:w + 1, :]

                R_xx_batch = torch.mm(x.T, x)
                R_yy_batch = torch.mm(y.T, y)
                R_xy_batch = torch.mm(x.T, y)

                self.R_xx.data = (1.0 - alpha) * self.R_xx + alpha * R_xx_batch
                self.R_yy.data = (1.0 - alpha) * self.R_yy + alpha * R_yy_batch
                self.R_xy.data = (1.0 - alpha) * self.R_xy + alpha * R_xy_batch

                self.samples_seen += 1

        self._update_kappa2_estimate(clean_signals, noisy_signals)

    def _resolve_update_rate(self) -> float:
        return self._update_rate if hasattr(self, '_update_rate') else 1.0

    def _resolve_kappa2(self) -> float:
        if self.fixed_kappa2 is not None:
            return float(self.fixed_kappa2)
        if int(self.kappa2_updates.item()) > 0:
            return float(self.kappa2_est.item())
        return 1.0

    def _resolve_regularization(self) -> float:
        if self.regularization is not None:
            return float(self.regularization)
        if self.use_kappa2_regularization:
            return float(self.omega_scale * self._resolve_kappa2())
        return 1e-6

    def _compute_wiener_filter(self) -> torch.Tensor:
        """Compute the regularized Wiener/LMMSE weight matrix."""
        r_yy = torch.nan_to_num(self.R_yy, nan=0.0, posinf=1e6, neginf=-1e6)
        r_xy = torch.nan_to_num(self.R_xy, nan=0.0, posinf=1e6, neginf=-1e6)

        omega = self._resolve_regularization()
        r_yy_reg = r_yy + omega * torch.eye(r_yy.size(0), device=r_yy.device, dtype=r_yy.dtype)

        try:
            W = torch.linalg.solve(r_yy_reg, r_xy.T).T
        except Exception:
            W = torch.pinverse(r_yy_reg) @ r_xy

        return torch.nan_to_num(W, nan=0.0, posinf=1e6, neginf=-1e6)

    def _reconstruct_signal(self, windows: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Reconstruct the full waveform by overlap-averaging windows."""
        B, _, L = target_shape
        B_w, num_windows, dim = windows.shape

        output = torch.zeros(B, self.num_channels, L, device=windows.device, dtype=windows.dtype)
        weight_counts = torch.zeros(B, self.num_channels, L, device=windows.device, dtype=windows.dtype)

        for w, start in enumerate(self.window_starts):
            if w >= num_windows:
                break

            end = min(start + self.window_length, L)
            window_data = windows[:, w, :].view(B, self.num_channels, self.window_length)
            valid_len = end - start

            output[:, :, start:end] += window_data[:, :, :valid_len]
            weight_counts[:, :, start:end] += 1.0

        return output / (weight_counts + 1e-8)

    def _apply_lmmse_filter(self, noisy_signal: torch.Tensor) -> torch.Tensor:
        """Apply the current LMMSE estimator to a noisy batch."""
        if noisy_signal.ndim != 3:
            raise ValueError(f'Expected input shape (B, 2, L), got {tuple(noisy_signal.shape)}')

        W = self._compute_wiener_filter()
        y_windows = self._extract_windows(noisy_signal)
        B, num_windows, dim = y_windows.shape

        filtered_windows = []
        for b in range(B):
            batch_windows = []
            for w in range(num_windows):
                y_window = y_windows[b, w, :]
                y_window = torch.nan_to_num(y_window, nan=0.0, posinf=1e6, neginf=-1e6)
                x_hat_window = torch.mv(W, y_window)
                batch_windows.append(torch.nan_to_num(x_hat_window, nan=0.0, posinf=1e6, neginf=-1e6))
            filtered_windows.append(torch.stack(batch_windows))

        filtered_windows = torch.stack(filtered_windows)
        return self._reconstruct_signal(filtered_windows, noisy_signal.shape)

    def forward(self, clean: torch.Tensor, noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Training-mode entry point that only updates second-order statistics."""
        self.update_covariance_matrices(clean, noisy)
        return {}

    @torch.no_grad()
    def sample(self, x_noisy: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Inference entry point kept compatible with diffusion-style models."""
        return self._apply_lmmse_filter(x_noisy)

    def set_kappa2(self, kappa2: float) -> None:
        """Manually override the effective kappa^2 used for omega resolution."""
        self.fixed_kappa2 = float(kappa2)
        self.kappa2_est.fill_(float(kappa2))
        self.kappa2_updates += 1

    def get_covariance_info(self) -> Dict[str, torch.Tensor]:
        return {
            'R_xx': self.R_xx.clone(),
            'R_yy': self.R_yy.clone(),
            'R_xy': self.R_xy.clone(),
            'samples_seen': self.samples_seen.clone(),
            'kappa2_est': self.kappa2_est.clone(),
            'kappa2_updates': self.kappa2_updates.clone(),
            'regularization': torch.tensor(self._resolve_regularization()),
            'omega_scale': torch.tensor(self.omega_scale),
            'condition_number_Ryy': torch.linalg.cond(self.R_yy),
            'window_length': torch.tensor(self.window_length),
            'overlap': torch.tensor(self.overlap),
        }


class RLSLMMSEBaseline(LMMSEBaseline):
    """Optional RLS-style variant kept for compatibility."""

    def __init__(self, config):
        super().__init__(config)
        self.forgetting_factor = float(getattr(config, 'rls_lambda', 0.99))
        self.rls_delta = float(getattr(config, 'rls_delta', 1.0))
        dim = self.window_length * self.num_channels
        self.register_buffer('P', torch.eye(dim) * (1.0 / self.rls_delta))
        self.register_buffer('w', torch.zeros(dim, dim))

    def update_rls(self, x_window: torch.Tensor, y_window: torch.Tensor):
        k = torch.mv(self.P, y_window) / (self.forgetting_factor + torch.dot(y_window, torch.mv(self.P, y_window)))
        e = x_window - torch.mv(self.w, y_window)
        self.w += torch.outer(k, e)
        self.P = (self.P - torch.outer(k, torch.mv(self.P, y_window))) / self.forgetting_factor

    def _compute_wiener_filter(self) -> torch.Tensor:
        return self.w


def create_lmmse_baseline(config, rls=False):
    if rls:
        return RLSLMMSEBaseline(config)
    return LMMSEBaseline(config)
