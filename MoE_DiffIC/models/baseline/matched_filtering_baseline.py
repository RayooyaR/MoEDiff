"""
Matched Filtering (MF) Baseline  —  严格匹配滤波基线
=====================================================

该实现严格遵循通信理论中的匹配滤波器定义：

  h[n] = s^*[L - 1 - n]

其中 s[n] 是发射端已知的成形脉冲。对于本仓库的 QPSK 数据，
clean_signal 由根升余弦 (RRC) 脉冲成形生成，因此接收端只需对
已知的 RRC 脉冲做共轭时间反转即可得到匹配滤波器。

本实现保留 waveform 级 sample() 接口用于兼容现有评估流程，同时
额外提供符号率抽样与 QPSK 硬判决，便于按论文描述完成 demod 流程。
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rrc_taps(beta: float = 0.5, sps: int = 16, span: int = 4) -> np.ndarray:
    """Generate a unit-energy root-raised-cosine pulse."""
    n = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64)
    t = n / sps
    taps = np.zeros_like(t)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-12:
            taps[i] = (
                beta
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            num = (
                np.sin(np.pi * ti * (1.0 - beta))
                + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / (den + 1e-12)

    taps = taps / np.sqrt(np.sum(taps ** 2) + 1e-12)
    return taps.astype(np.float32)


class MatchedFilteringBaseline(nn.Module):
    """Strict matched filtering baseline with optional symbol-rate demodulation."""

    def __init__(self, config):
        super().__init__()

        self.signal_length = config.data.signal_length
        self.num_channels = config.data.num_channels
        self.samples_per_symbol = int(getattr(config.data, 'samples_per_symbol', getattr(config.data, 'sps', 16)))
        self.rrc_beta = float(getattr(config.data, 'rrc_beta', 0.5))
        self.rrc_span = int(getattr(config.data, 'rrc_span', 4))
        self.symbol_offset = int(getattr(config.data, 'tau_s', self.samples_per_symbol // 2))

        pulse = _rrc_taps(beta=self.rrc_beta, sps=self.samples_per_symbol, span=self.rrc_span)
        self.kernel_size = int(pulse.shape[0])
        self.group_delay = (self.kernel_size - 1) // 2

        taps = torch.from_numpy(np.conj(pulse).astype(np.float32))
        self.register_buffer('matched_filter_taps', taps.view(1, 1, -1).repeat(self.num_channels, 1, 1))

    def update_statistics(self, clean_signals: torch.Tensor, noisy_signals: torch.Tensor):
        """Compat hook: strict MF does not learn statistics."""
        return None

    def forward(self, clean: torch.Tensor, noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Training-compatible entry point with no trainable loss."""
        return {}

    def _apply_matched_filter(self, noisy_signal: torch.Tensor) -> torch.Tensor:
        """Apply the strict RRC matched filter to each IQ channel."""
        if noisy_signal.ndim != 3:
            raise ValueError(f'Expected input shape (B, 2, L), got {tuple(noisy_signal.shape)}')
        if noisy_signal.shape[1] != self.num_channels:
            raise ValueError(f'Expected {self.num_channels} channels, got {noisy_signal.shape[1]}')

        taps = self.matched_filter_taps.to(device=noisy_signal.device, dtype=noisy_signal.dtype)
        return F.conv1d(
            noisy_signal,
            weight=taps,
            bias=None,
            stride=1,
            padding=self.group_delay,
            groups=self.num_channels,
        )

    def _sample_symbol_rate(self, filtered_signal: torch.Tensor) -> torch.Tensor:
        """Take one sample per symbol using the configured timing offset."""
        if filtered_signal.ndim != 3:
            raise ValueError(f'Expected input shape (B, 2, L), got {tuple(filtered_signal.shape)}')

        start = int(self.symbol_offset % self.samples_per_symbol)
        return filtered_signal[:, :, start::self.samples_per_symbol]

    def _qpsk_hard_decision(self, symbol_iq: torch.Tensor) -> torch.Tensor:
        """Map sampled symbols to the nearest QPSK constellation point."""
        decision = torch.sign(symbol_iq)
        decision = torch.where(decision == 0, torch.ones_like(decision), decision)
        scale = torch.sqrt(torch.tensor(2.0, device=symbol_iq.device, dtype=symbol_iq.dtype))
        return (decision / scale).to(symbol_iq.dtype)

    @torch.no_grad()
    def demodulate(self, x_noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Matched-filter, sample, and perform hard QPSK detection."""
        mf = self._apply_matched_filter(x_noisy)
        sym = self._sample_symbol_rate(mf)
        hard = self._qpsk_hard_decision(sym)
        bits = torch.empty((hard.shape[0], hard.shape[2] * 2), dtype=torch.uint8, device=hard.device)
        bits[:, 0::2] = (hard[:, 0, :] < 0).to(torch.uint8)
        bits[:, 1::2] = (hard[:, 1, :] < 0).to(torch.uint8)
        return {
            'matched_filtered': mf,
            'sampled_symbols': sym,
            'hard_symbols': hard,
            'bits': bits,
        }

    @torch.no_grad()
    def sample(self, x_noisy: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Inference entry point kept for compatibility with diffusion models."""
        return self._apply_matched_filter(x_noisy)

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        return {
            'matched_filter_taps': self.matched_filter_taps.clone(),
            'samples_per_symbol': torch.tensor(self.samples_per_symbol),
            'rrc_beta': torch.tensor(self.rrc_beta),
            'rrc_span': torch.tensor(self.rrc_span),
            'symbol_offset': torch.tensor(self.symbol_offset),
            'group_delay': torch.tensor(self.group_delay),
        }


class AdaptiveMatchedFilter(MatchedFilteringBaseline):
    """Compatibility wrapper that still uses the strict matched filter."""

    def __init__(self, config):
        super().__init__(config)

    def update_statistics_adaptive(self, clean_signals: torch.Tensor,
                                   noisy_signals: torch.Tensor,
                                   interference_types: Optional[torch.Tensor] = None):
        return None

    def _apply_adaptive_filter(self, noisy_signal: torch.Tensor,
                               interference_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._apply_matched_filter(noisy_signal)


def create_matched_filtering_baseline(config, adaptive=False):
    if adaptive:
        return AdaptiveMatchedFilter(config)
    return MatchedFilteringBaseline(config)
