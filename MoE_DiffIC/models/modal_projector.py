"""
ModalProjector & GeometricAugmentor  —  显式模态变换前置模块

=========================================================================
模块定位 (在整体架构中的角色):
=========================================================================
  PhysicsMoEDiffusion 主模型中, 该模块是最先被调用的组件:
    x_noisy (B,2,128) → DataAugmentation
                          ├── ModalProjector  → modal_dict  → PhysicsMoE → 各域专家
                          └── GeometricAugmentor → geom_out  → (当前 bypass)

=========================================================================
设计哲学:
=========================================================================
  将原先隐藏在各 Expert 内部的信号变换抽离为一个独立的 *前置* 模块,
  使后续的域专家只需专注于各自物理域内的特征学习，而不必重复实现
  FFT/STFT/下采样等变换。

  好处:
    1. 变换只做一次，避免多个专家重复计算 FFT/STFT
    2. 所有变换均为可微操作，梯度可无损回传
    3. 统一管理信号→各物理域的映射，方便扩展新域

=========================================================================
输入输出:
=========================================================================
  输入:  x_signal  (B, 2, L)   I/Q 时域信号, L=128
  输出:
    modal_dict = {
        'time_feat':  (B, 2, L)                            — 原始时域 (直通)
        'freq_feat':  (B, 2, n_freq) complex               — rfft 频域
        'stft_feat':  (B, 2, n_fft//2+1, n_frames) complex — STFT 时频域
        'ms_feat':    list[ (B, 2, L_i) ]                   — 多尺度下采样
    }
    geom_out:  (B, 2, L)   几何增强后的信号 (v2 永久 bypass)

=========================================================================
跨模块关系:
=========================================================================
  - ModalProjector 的 4 个输出键对应 domain_experts.py 中 4 个专家的 input_keys:
      FD_Expert → freq_feat + time_feat
      MS_Expert → ms_feat   + time_feat
      TF_Expert → stft_feat + time_feat
      TD_Expert → time_feat
  - GeometricAugmentor v2 始终 bypass，几何增强在 dataset 层完成
  - DataAugmentation 是 PhysicsMoEDiffusion.forward() 中调用的统一入口
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Modal Projector  —  多模态特征提取 (信号→4个物理域)
# ============================================================

class ModalProjector(nn.Module):
    """
    将 (B, 2, L) I/Q 信号投射到四个物理模态域:
      1. time_feat : 原始时域 (直通, 零计算开销)
      2. freq_feat : torch.fft.rfft → 1D 复数频谱
      3. stft_feat : torch.stft → 2D 时频复数表示
      4. ms_feat   : 多尺度平均池化 (stride-2 逐级下采样)
    
    所有输出均为 *可微* 变换, 保证梯度可回传到输入信号。
    
    【参数默认值选择依据】
      n_fft=32:       L=128 时, 频率分辨率 = 32/2+1 = 17 bin, 时间帧数适中
      hop_length=8:   每帧重叠 = n_fft - hop = 24 点, 时间分辨率 ≈ L/hop = 16 帧
      n_scales=3:     产生 L=128, 64, 32 三种尺度, 覆盖细/中/粗分辨率
    """
    
    def __init__(self, signal_length=2560, n_fft=128, hop_length=32, n_scales=5):
        """
        Args:
            signal_length (int): 输入信号长度, 默认 128
            n_fft (int):         STFT 的 FFT 点数, 默认 32
            hop_length (int):    STFT 帧移, 默认 8
            n_scales (int):      多尺度池化层数, 默认 3 → [1x, 2x, 4x] 下采样
        """
        super().__init__()
        self.signal_length = signal_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_scales = n_scales
        
        # ── STFT 窗函数 ─────────────────────────────────────────
        # register_buffer: 不参与梯度更新, 但跟随模型 .to(device)
        # Hann 窗是 STFT 中最常用的窗函数, 旁瓣衰减 > 30 dB
        self.register_buffer('stft_window', torch.hann_window(n_fft))
        
        # ── 多尺度平均池化 ───────────────────────────────────────
        # scale 0: AvgPool1d(kernel=1, stride=1) → L=128 (原始分辨率)
        # scale 1: AvgPool1d(kernel=2, stride=2) → L=64  (2x 下采样)
        # scale 2: AvgPool1d(kernel=4, stride=4) → L=32  (4x 下采样)
        # AvgPool1d 天然可微, 反向传播时梯度均匀分配到池化窗口内的元素
        self.ms_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2**i, stride=2**i)
            for i in range(n_scales)
        ])
    
    def forward(self, x):
        """
        四路并行变换: 时域直通 / rfft / STFT / 多尺度池化

        Args:
            x (Tensor): (B, 2, L) — I/Q 时域信号

        Returns:
            dict: {
                'time_feat': (B, 2, L)              — 原始时域
                'freq_feat': (B, 2, L//2+1) complex — rfft 频域
                'stft_feat': (B, 2, F, T)   complex — STFT 时频域
                'ms_feat':   list[(B, 2, L_i)]      — 多尺度特征
            }
        """
        B, C, L = x.shape
        
        # ── 路径 1: 时域直通 (零开销) ────────────────────────────
        time_feat = x  # (B, 2, L)
        
        # ── 路径 2: 频域 rfft ────────────────────────────────────
        # rfft: 实数 FFT, 仅保留正频率部分 (Hermitian 对称)
        # 输出长度 = L//2 + 1 = 65 (L=128 时)
        # 返回复数张量, 幅度|freq_feat|即为频谱, 相位 arg(freq_feat) 为相位谱
        freq_feat = torch.fft.rfft(x, dim=-1)  # (B, 2, L//2+1) complex
        
        # ── 路径 3: STFT 时频域 ──────────────────────────────────
        # torch.stft 要求输入为 (B, L), 因此逐通道 (I/Q) 分别计算
        # 输出: (B, n_fft//2+1, n_frames) complex = (B, 17, T)
        stft_list = []
        for ch in range(C):   # ch=0 → I 通道, ch=1 → Q 通道
            # s = torch.stft(
            #     x[:, ch, :],               # (B, L) 单通道
            #     n_fft=self.n_fft,           # FFT 点数 = 32
            #     hop_length=self.hop_length, # 帧移 = 8
            #     win_length=self.n_fft,      # 窗长 = n_fft
            #     window=self.stft_window,    # Hann 窗
            #     return_complex=True,        # 返回复数
            # )
            s = torch.functional.stft(
            x[:, ch, :],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.stft_window,
            center=True,          # 保持与旧版默认一致
            pad_mode='reflect',   # 保持与旧版默认一致
            normalized=False,     # 保持与旧版默认一致
            onesided=True,        # 保持与旧版默认一致
            return_complex=True,
            )
            stft_list.append(s)
        # 重新合并通道维度: (B, 2, F, T) complex
        stft_feat = torch.stack(stft_list, dim=1)
        
        # ── 路径 4: 多尺度平均池化 ───────────────────────────────
        # ms_feat[0]: (B, 2, 128) — 原始分辨率 (scale 0, pool=1)
        # ms_feat[1]: (B, 2, 64)  — 2x 下采样 (scale 1, pool=2)
        # ms_feat[2]: (B, 2, 32)  — 4x 下采样 (scale 2, pool=4)
        ms_feat = [pool(x) for pool in self.ms_pools]
        
        return {
            'time_feat': time_feat,      # (B, 2, L)           → TD_Expert, 所有专家
            'freq_feat': freq_feat,      # (B, 2, n_freq)      → FD_Expert
            'stft_feat': stft_feat,      # (B, 2, F, T)        → TF_Expert
            'ms_feat': ms_feat,          # list[(B, 2, L_i)]   → MS_Expert
        }


# ============================================================
#  Geometric Augmentor  —  几何域数据增强 (v2: 永久旁路)
# ============================================================

class GeometricAugmentor(nn.Module):
    """
    模拟真实通信场景中的几何变换:
      - 相位旋转: 随机旋转 I/Q 星座图 (等价于载波相位偏移)
      - IQ 交换: 互换 I/Q 通道 (等价于 90° 相位跳变)
    
    ======== v2 设计修改 (2026-03-03): 始终旁路 ========
    
    【为什么旁路?】
      原设计问题:
        1. GeometricAugmentor 仅作用于 noisy 信号 (x_noisy)
        2. 训练标签 x_clean 未做同步几何变换
        3. → 训练时 geom_out 被旋转角度 θ, 但 clean 未旋转
        4. → 扩散模型学到的映射包含 θ 偏移, 推理时 θ=0 导致 mismatch
      
      解决方案:
        几何增强改在数据集层 (_augment_pair) 中同步完成:
        对 (clean, noisy) 施加相同的随机变换, 保证标签一致性。
    
    【保留此类的原因】
      - 代码兼容性: PhysicsMoEDiffusion 仍调用 DataAugmentation
      - 未来可能: learnable augmentation / test-time augmentation
      - 架构完整性: Modal + Geometric 双路结构保持清晰
    """
    
    def __init__(self, rotation_prob=0.5, iq_swap_prob=0.3):
        """
        Args:
            rotation_prob (float):  相位旋转概率 (v2 中未使用, 保留接口)
            iq_swap_prob (float):   IQ 交换概率  (v2 中未使用, 保留接口)
        """
        super().__init__()
        self.rotation_prob = rotation_prob
        self.iq_swap_prob = iq_swap_prob
    
    def forward(self, x, training=True):
        """
        v2: 始终直通, 无论 training 模式如何

        Args:
            x (Tensor): (B, 2, L) — I (ch0), Q (ch1)
            training (bool): 当前版本忽略此参数, 始终旁路

        Returns:
            geom_out (Tensor): (B, 2, L) — 与输入完全相同
        """
        # v2: 始终旁路, 避免 train/test distribution mismatch
        # 几何增强已在数据集层 _augment_pair() 中同步完成 (clean + noisy 同步变换)
        return x


# ============================================================
#  DataAugmentation  —  组合入口 (ModalProjector + GeometricAugmentor)
# ============================================================

class DataAugmentation(nn.Module):
    """
    统一数据增强入口, 在 PhysicsMoEDiffusion.forward() 中被调用:
      modal_dict, geom_out = self.data_augmentation(x_noisy)

    内部包含两条并行路径:
      ┌─ Modal path  → ModalProjector → modal_dict
      │   提取 time/freq/stft/ms 四种物理域特征
      │   → 传递给 PhysicsMoE 中的各域专家
      │
      └─ Geometric path → GeometricAugmentor → geom_out
          v2: 始终直通, geom_out == x
          几何增强在数据集层 _augment_pair() 同步完成

    【调用链路】
      PhysicsMoEDiffusion.forward()
        → DataAugmentation(x_noisy)
          ├── ModalProjector(x_noisy) → modal_dict
          └── GeometricAugmentor(x_noisy) → geom_out (== x_noisy)
        → PhysicsMoE(modal_dict, ...) → moe_output
        → DiT denoiser(x_t, moe_output, ...) → noise prediction
    """
    
    def __init__(self, signal_length=2560, n_fft=128, hop_length=32, n_scales=5,
                 rotation_prob=0.5, iq_swap_prob=0.3):
        """
        Args:
            signal_length (int): 信号长度, 默认 128
            n_fft (int):         STFT FFT 点数, 默认 32
            hop_length (int):    STFT 帧移, 默认 8
            n_scales (int):      多尺度池化层数, 默认 3
            rotation_prob (float): 相位旋转概率 (v2 未使用)
            iq_swap_prob (float):  IQ 交换概率 (v2 未使用)
        """
        super().__init__()
        
        # ── 多模态投影器: 信号 → 4 个物理域 ──────────────────────
        self.modal_projector = ModalProjector(
            signal_length=signal_length,
            n_fft=n_fft,
            hop_length=hop_length,
            n_scales=n_scales,
        )
        
        # ── 几何增强器: v2 始终旁路 ──────────────────────────────
        self.geometric_augmentor = GeometricAugmentor(
            rotation_prob=rotation_prob,
            iq_swap_prob=iq_swap_prob,
        )
    
    def forward(self, x, training=True):
        """
        双路并行: 模态投影 + 几何增强

        Args:
            x (Tensor): (B, 2, L) — 输入 IQ 信号
            training (bool): 训练/推理标志 (v2 中对 Geometric 无影响)

        Returns:
            modal_dict (dict): 多模态特征字典 — 供各域专家使用
            geom_out (Tensor): (B, 2, L) — 几何增强后信号 (v2 == x)
        """
        modal_dict = self.modal_projector(x)                  # → dict
        geom_out = self.geometric_augmentor(x, training=training)  # → (B,2,L) == x
        return modal_dict, geom_out
