"""
restormer_baseline.py  —  Restormer1DBaseline
=============================================
Restormer（Zamir et al., CVPR 2022, "Restormer: Efficient Transformer for High-Resolution Image Restoration"，官方实现
https://github.com/swz30/Restormer 的 basicsr/models/archs/restormer_arch.py）到一维 I/Q 信号的忠实移植，作为"基于深度学习的直接回归"基线。

忠实保留（与官方一致）：
  • 4 级编解码 U 形结构；每级 TransformerBlock = [LayerNorm → MDTA] + [LayerNorm → GDFN]，均带残差
  • MDTA（多深度卷积头转置自注意力）：1×1 卷积 + 深度可分 3 卷积生成 Q/K/V，注意力在"通道"维度上计算（c×c 而不是 L×L），
    Q/K 沿序列维 L2 归一化，每头一个可学习温度
  • GDFN（门控深度卷积前馈）：1×1 升维(×2.66，分两路) → 深度 3 卷积 → GELU 门控 → 1×1 降维
  • 层归一化 WithBias；卷积无偏置；Downsample = 3 卷积 + 像素反洗牌，Upsample = 3 卷积 + 像素洗牌；
    解码端 skip 拼接后 1×1 卷积降通道（最高分辨率一级不降通道）；末端 refinement 块；输出 3 卷积 + 全局残差（out = f(x) + x）
  • 默认超参数同官方：dim=48，num_blocks=[4,6,6,8]，num_refinement_blocks=4，heads=[1,2,4,8]，ffn_expansion_factor=2.66

为 1D 做的必要改动：
  • Conv2d→Conv1d；PixelUnshuffle/PixelShuffle(2) 的 1D 等价写法（序列长度 ÷2 / ×2，通道 ×2 / ÷2）。2D 里 3×3 卷积先把通道
    减半再 4 倍反洗牌得到 2 倍通道；1D 反洗牌只有 2 倍，所以 Downsample 的卷积保持通道数不变、Upsample 同理。
  • 输入/输出通道 = 2（I/Q）；真实数据信号长度 2560（640 符号×4 采样点），经 3 次下采样 → 320，无需 padding（MDTA 在通道维做注意力，复杂度对序列长度线性）。
  • 不使用官方的渐进式训练/L1 损失/多尺度 patch 训练：训练损失、优化器、EMA、数据划分与本项目 CNN 基线完全一致
    （MSE + 0.1×频域幅度谱 MSE，AdamW lr=3e-4，bf16），以便只比较架构。

接口与 PhysicsCNNBaseline 一致：model(noisy, clean) -> dict(loss, pred, ...)；model.sample(noisy) -> 预测的干净信号。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm1d(nn.Module):
    """官方 WithBias / BiasFree LayerNorm：对每个位置在通道维做归一化。输入 (B, C, L)。"""

    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        x = x.transpose(1, 2)                                   # (B, L, C)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.bias is not None:
            mu = x.mean(-1, keepdim=True)
            y = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        else:
            y = x / torch.sqrt(sigma + 1e-5) * self.weight
        return y.transpose(1, 2)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""

    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv1d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv1d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv1d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class MDTA(nn.Module):
    """Multi-DConv head Transposed self-Attention：注意力矩阵是 (通道/头) × (通道/头)，复杂度对序列长度线性。"""

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv1d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, l = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.num_heads, c // self.num_heads, l) for t in (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.project_out((attn @ v).reshape(b, c, l))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, with_bias_ln=True):
        super().__init__()
        self.norm1 = LayerNorm1d(dim, with_bias_ln)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm1d(dim, with_bias_ln)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def unshuffle1d(x, r=2):          # (B, C, L) -> (B, C*r, L/r)，与 PixelUnshuffle 同一通道排列
    b, c, l = x.shape
    return x.reshape(b, c, l // r, r).permute(0, 1, 3, 2).reshape(b, c * r, l // r)


def shuffle1d(x, r=2):            # (B, C*r, L) -> (B, C, L*r)，PixelShuffle 的一维版本
    b, c, l = x.shape
    return x.reshape(b, c // r, r, l).permute(0, 1, 3, 2).reshape(b, c // r, l * r)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv = nn.Conv1d(n_feat, n_feat, 3, 1, 1, bias=False)

    def forward(self, x):
        return unshuffle1d(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv = nn.Conv1d(n_feat, n_feat, 3, 1, 1, bias=False)

    def forward(self, x):
        return shuffle1d(self.conv(x))


class Restormer1D(nn.Module):
    def __init__(self, inp_channels=2, out_channels=2, dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4, heads=(1, 2, 4, 8),
                 ffn_expansion_factor=2.66, bias=False, with_bias_ln=True):
        super().__init__()
        blk = lambda d, h, n: nn.Sequential(*[TransformerBlock(d, h, ffn_expansion_factor, bias, with_bias_ln) for _ in range(n)])
        self.patch_embed = nn.Conv1d(inp_channels, dim, 3, 1, 1, bias=bias)
        self.encoder_level1 = blk(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = blk(dim * 2, heads[1], num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = blk(dim * 4, heads[2], num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = blk(dim * 8, heads[3], num_blocks[3])
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv1d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = blk(dim * 4, heads[2], num_blocks[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv1d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = blk(dim * 2, heads[1], num_blocks[1])
        self.up2_1 = Upsample(dim * 2)                                   # 最高分辨率一级不降通道（与官方一致）
        self.decoder_level1 = blk(dim * 2, heads[0], num_blocks[0])
        self.refinement = blk(dim * 2, heads[0], num_refinement_blocks)
        self.output = nn.Conv1d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp):
        e1 = self.encoder_level1(self.patch_embed(inp))
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        lat = self.latent(self.down3_4(e3))
        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(lat), e3], 1)))
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1)))
        d1 = self.decoder_level1(torch.cat([self.up2_1(d2), e1], 1))
        return self.output(self.refinement(d1)) + inp                    # 全局残差（官方去噪/去模糊同此）


class Restormer1DBaseline(nn.Module):
    """与 PhysicsCNNBaseline 同接口的 Restormer 回归基线。"""

    def __init__(self, in_channels=2, dim=48, signal_length=128, freq_loss_weight=0.1):
        super().__init__()
        assert signal_length % 8 == 0, 'Restormer 需要序列长度能被 8 整除（3 次下采样）'
        self.signal_length, self.freq_loss_weight = signal_length, freq_loss_weight
        self.net = Restormer1D(in_channels, in_channels, dim=dim)

    def _forward_core(self, noisy):
        return self.net(noisy)

    def forward(self, noisy, clean=None, channel=None, **kwargs):
        pred = self._forward_core(noisy)
        zero = torch.tensor(0.0, device=noisy.device, dtype=noisy.dtype)
        if clean is not None:
            pred, clean = pred.float(), clean.float()
            mse = F.mse_loss(pred, clean)
            freq = F.mse_loss(torch.fft.rfft(pred, dim=-1, norm='ortho').abs(), torch.fft.rfft(clean, dim=-1, norm='ortho').abs())
            loss = mse + self.freq_loss_weight * freq
        else:
            mse = freq = loss = zero
        return {'pred': pred, 'loss': loss, 'mse_loss': mse, 'freq_loss': freq, 'noise_pred': pred,
                'noise_target': clean if clean is not None else pred.detach(), 'rcl_loss': zero, 'balance_loss': zero, 'gates': None}

    @torch.no_grad()
    def sample(self, noisy, channel=None, num_steps=50, **kwargs):
        return self._forward_core(noisy)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
