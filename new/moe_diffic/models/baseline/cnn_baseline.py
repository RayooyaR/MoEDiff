"""
cnn_baseline.py  —  PhysicsCNNBaseline
======================================
纯监督 CNN 干扰消除基线，无扩散过程，无 MoE。

作为对比实验的下界基线，量化以下两个维度的独立贡献：
  CNN → ICDM          : "扩散生成过程"本身带来多少增益
  CNN → MoE-DiffIC    : "扩散 + 物理专家"联合带来多少增益

架构：UNet 风格编解码器（1D）
  Encoder  : input_proj → enc1 → enc2 → enc3（逐层步长=2下采样）
  Bottleneck: 空洞卷积[dil=1,2,4] + 轻量全局自注意力
  Decoder  : dec3 → dec2 → dec1 → output_proj（转置卷积上采样 + 跳跃连接）

通道维度：
  L=128 → input_proj(2→C) → enc1(C→2C, L/2) → enc2(2C→4C, L/4)
        → enc3(4C→8C, L/8) → bottleneck(8C, L/8)
        → dec3(8C→4C, L/4) → dec2(4C→2C, L/2) → dec1(2C→C, L)
        → output_proj(C→2)

损失：MSE(时域) + 0.1 × MSE(频域幅度谱)
参数量（base_ch=32）：~0.9M

接口约定（与 train.py 兼容）：
  - train.py 传入: model(clean=clean, noisy=noisy)
  - forward 返回: dict，key 含 'loss'/'noise_pred'/'noise_target' 等
  - 推理接口: model.sample(noisy, channel, num_steps=...) → clean_pred
    （channel / num_steps 参数接收但不使用，保证与扩散模型签名一致）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  基础卷积模块
# ============================================================

class ConvBlock(nn.Module):
    """Conv1d + GroupNorm + SiLU（保持序列长度不变）"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      padding=pad, dilation=dilation),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """
    步长=2下采样块
    Conv1d(stride=2) + GroupNorm + SiLU + ConvBlock
    输出长度 = 输入长度 // 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act  = nn.SiLU()
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.act(self.norm(self.down(x))))


class UpBlock(nn.Module):
    """
    转置卷积×2上采样块
    ConvTranspose1d(stride=2) + skip concat + ConvBlock
    输出长度 = 输入长度 × 2
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, out_ch,
                                        kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act  = nn.SiLU()
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.up(x)))
        return self.conv(torch.cat([x, skip], dim=1))   # 跳跃连接


class LightSelfAttention(nn.Module):
    """
    轻量全局自注意力（瓶颈层使用）

    经过 3 次下采样后序列长度 L/8=16，
    MultiheadAttention 复杂度 O(L²d)=O(16²×256) ≈ 65K，开销极小。
    用于捕获 LFM 扫频等需要全局上下文的干扰特征。
    """

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(1, ch)   # LayerNorm 的 1D 等效
        self.attn = nn.MultiheadAttention(ch, num_heads,
                                           batch_first=True, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, L)
        h = self.norm(x).permute(0, 2, 1)         # (B, L, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)             # 残差 + (B, C, L)


class Bottleneck(nn.Module):
    """
    瓶颈层：扩张感受野 + 全局上下文

    空洞卷积 dilation=[1,2,4]：
      在长度 L=16 的序列上，有效感受野 = 3+(3-1)×2+(3-1)×4=17 > 16，
      每个位置能看到整条序列，无需更深的堆叠。

    自注意力：建模任意距离的位置间依赖，
      适合 LFM 线性扫频（频率随时间线性变化，需要全局视野识别趋势）。
    """

    def __init__(self, ch: int):
        super().__init__()
        self.dilated = nn.Sequential(
            ConvBlock(ch, ch, kernel=3, dilation=1),
            ConvBlock(ch, ch, kernel=3, dilation=2),
            ConvBlock(ch, ch, kernel=3, dilation=4),
        )
        self.attn = LightSelfAttention(ch, num_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.dilated(x))


# ============================================================
#  PhysicsCNNBaseline  —  主模型
# ============================================================

class PhysicsCNNBaseline(nn.Module):
    """
    UNet 风格 CNN 干扰消除基线

    【数据流】
      noisy (B, 2, 128)
        │
        ├── input_proj: 2→C,  L=128   ← skip0
        ├── enc1:       C→2C, L=64    ← skip1
        ├── enc2:       2C→4C,L=32    ← skip2
        ├── enc3:       4C→8C,L=16
        ├── bottleneck: 8C,   L=16    空洞conv+注意力
        ├── dec3:       8C→4C,L=32    + skip2
        ├── dec2:       4C→2C,L=64    + skip1
        ├── dec1:       2C→C, L=128   + skip0
        └── output_proj: C→2, L=128
        pred_clean (B, 2, 128)

    【训练接口】
      result = model(clean=clean_norm, noisy=noisy_norm)
      loss   = result['loss']   ← MSE(时域) + 0.1×MSE(频域幅度谱)

    【推理接口】
      pred = model.sample(noisy_norm, channel, num_steps=50)
      # channel / num_steps 被忽略，仅保持与扩散模型签名一致

    Args:
        in_channels  (int): IQ 通道数，固定为 2
        base_ch      (int): 基础通道数，默认 32
        signal_length(int): 时域采样点数，固定为 128
        freq_loss_weight(float): 频域感知损失权重，默认 0.1
    """

    def __init__(
        self,
        in_channels    : int   = 2,
        base_ch        : int   = 32,
        signal_length  : int   = 128,
        freq_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.signal_length   = signal_length
        self.freq_loss_weight = freq_loss_weight
        ch = base_ch    # 32

        # ── 输入投影 ──────────────────────────────────────────────
        self.input_proj = ConvBlock(in_channels, ch)    # 2→32, L=128

        # ── Encoder（每层长度 ÷2）────────────────────────────────
        self.enc1 = DownBlock(ch,     ch * 2)   # 32→64,   L=64
        self.enc2 = DownBlock(ch * 2, ch * 4)   # 64→128,  L=32
        self.enc3 = DownBlock(ch * 4, ch * 8)   # 128→256, L=16

        # ── Bottleneck（全局感受野）──────────────────────────────
        self.bottleneck = Bottleneck(ch * 8)    # 256, L=16

        # ── Decoder（每层长度 ×2 + 跳跃连接）────────────────────
        self.dec3 = UpBlock(ch * 8, ch * 4, ch * 4)    # 256+128→128, L=32
        self.dec2 = UpBlock(ch * 4, ch * 2, ch * 2)    # 128+64 →64,  L=64
        self.dec1 = UpBlock(ch * 2, ch,     ch)         # 64+32  →32,  L=128

        # ── 输出投影 ──────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            ConvBlock(ch, ch),
            nn.Conv1d(ch, in_channels, kernel_size=1),  # 32→2
        )

        # ── 权重初始化（Kaiming）─────────────────────────────────
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── 内部推断（带 skip 连接）──────────────────────────────────
    def _forward_core(self, noisy: torch.Tensor) -> torch.Tensor:
        """UNet 前向，返回预测的干净信号 (B, 2, L)"""
        x0 = self.input_proj(noisy)     # (B, C,   128)  ← skip0
        x1 = self.enc1(x0)              # (B, 2C,   64)  ← skip1
        x2 = self.enc2(x1)              # (B, 4C,   32)  ← skip2
        x3 = self.enc3(x2)              # (B, 8C,   16)

        b  = self.bottleneck(x3)        # (B, 8C,   16)

        d3 = self.dec3(b,  x2)          # (B, 4C,   32)
        d2 = self.dec2(d3, x1)          # (B, 2C,   64)
        d1 = self.dec1(d2, x0)          # (B, C,   128)

        return self.output_proj(d1)     # (B, 2,   128)

    # ── 训练/验证前向 ─────────────────────────────────────────────
    def forward(
        self,
        noisy : torch.Tensor,
        clean : torch.Tensor = None,
        # 以下参数仅用于接口兼容，CNN 本身不使用
        channel    = None,
        **kwargs,
    ) -> dict:
        """
        Args:
            noisy  (Tensor): (B, 2, L) 归一化后的受干扰信号
            clean  (Tensor): (B, 2, L) 归一化后的干净信号（训练时提供）
            channel: 忽略（兼容扩散模型接口）

        Returns:
            dict:
              'pred'         : (B, 2, L) 预测的干净信号
              'loss'         : 标量，时域MSE + freq_loss_weight×频域MSE
              'mse_loss'     : 标量，时域 MSE
              'freq_loss'    : 标量，频域幅度谱 MSE
              'noise_pred'   : 与 clean 相同形状的零张量（兼容 CombinedLoss 接口）
              'noise_target' : 与 clean 相同形状的零张量（兼容 CombinedLoss 接口）
              'rcl_loss'     : 0.0 标量（无 MoE，兼容接口）
              'balance_loss' : 0.0 标量（无 MoE，兼容接口）
              'gates'        : None
        """
        pred = self._forward_core(noisy)    # (B, 2, L)

        zero = torch.tensor(0.0, device=noisy.device, dtype=noisy.dtype)

        if clean is not None:
            # ── 时域 MSE ─────────────────────────────────────────
            mse_loss = F.mse_loss(pred, clean)

            # ── 频域感知损失 ──────────────────────────────────────
            # 比较预测与目标的幅度谱，防止 CNN 学到"时域均值"捷径
            # rfft 在最后一维（时间轴）做实数 FFT，得 L//2+1 个频点
            # [new-data] norm='ortho'：FFT 幅度不再随 √L 增长。原实现按 L=128 设计，L=2560 时频域项会比时域 MSE 大约 100 倍而淹没时域项
            pred_fft  = torch.fft.rfft(pred,  dim=-1, norm='ortho').abs()   # (B, 2, L//2+1)
            clean_fft = torch.fft.rfft(clean, dim=-1, norm='ortho').abs()
            freq_loss = F.mse_loss(pred_fft, clean_fft)

            loss = mse_loss + self.freq_loss_weight * freq_loss
        else:
            mse_loss = freq_loss = loss = zero

        return {
            'pred'         : pred,
            'loss'         : loss,
            'mse_loss'     : mse_loss,
            'freq_loss'    : freq_loss,
            # ── 以下字段保持与 CombinedLoss / train.py 接口兼容 ──
            'noise_pred'   : pred,       # CNN直接预测clean，此字段被train.py读取
            'noise_target' : clean if clean is not None else pred.detach(),
            'rcl_loss'     : zero,
            'balance_loss' : zero,
            'gates'        : None,
        }

    # ── 推理接口（与扩散模型 sample() 签名兼容）─────────────────
    @torch.no_grad()
    def sample(
        self,
        noisy      : torch.Tensor,
        channel    = None,     # 接收但不使用
        num_steps  : int = 50, # 接收但不使用
        **kwargs,
    ) -> torch.Tensor:
        """
        推理接口，与 ICDMBaseline.sample() 签名兼容。

        Args:
            noisy     (Tensor): (B, 2, L) 归一化后的受干扰信号
            channel   : 忽略（无信道条件）
            num_steps : 忽略（CNN 无迭代去噪）

        Returns:
            pred (Tensor): (B, 2, L) 预测的干净信号
        """
        return self._forward_core(noisy)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)