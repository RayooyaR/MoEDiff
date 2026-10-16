import torch
import torch.nn as nn
import torch.nn.functional as F


class FD_Expert(nn.Module):
    def __init__(self, in_channels=2, signal_length=2560, hidden_dim=64, feature_dim=256):
        super().__init__()
        _ = signal_length  

        self.freq_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.freq_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.freq_norm = nn.LayerNorm(hidden_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, freq_feat, time_feat=None):
        del time_feat

        # 频域幅度作为输入特征
        if torch.is_complex(freq_feat):
            # Avoid complex abs CUDA JIT path that depends on nvrtc-builtins DLL.
            mag = torch.sqrt(freq_feat.real.pow(2) + freq_feat.imag.pow(2) + 1e-12)
        else:
            mag = torch.abs(freq_feat)

        h = self.freq_conv(mag)                   # (B, H, F)
        h_t = h.transpose(1, 2)                   # (B, F, H)
        h_attn, _ = self.freq_attn(h_t, h_t, h_t)
        h_t = self.freq_norm(h_t + h_attn)        # (B, F, H)
        h = h_t.transpose(1, 2)                   # (B, H, F)

        pooled = self.pool(h).squeeze(-1)         # (B, H)
        feat = self.proj(pooled)                  # (B, D)
        return feat


class MS_Expert(nn.Module):

    def __init__(self, in_channels=2, signal_length=2560, n_scales=5, hidden_channels=128, feature_dim=256):
        super().__init__()
        _ = signal_length  
        self.n_scales = n_scales

        self.scale_encoders = nn.ModuleList()
        for _ in range(n_scales):
            self.scale_encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(4, hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(4, hidden_channels),
                    nn.SiLU(),
                )
            )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_channels * n_scales, feature_dim)

    def forward(self, ms_feat, time_feat=None):

        del time_feat

        pooled_list = []
        for feat, encoder in zip(ms_feat, self.scale_encoders):
            h = encoder(feat)                      # (B, H, L_i)
            h_pool = self.pool(h).squeeze(-1)     # (B, H)
            pooled_list.append(h_pool)

        multi_scale_feat = torch.cat(pooled_list, dim=-1)  # (B, H*n_scales)
        out = self.proj(multi_scale_feat)                  # (B, D)
        return out


class TF_Expert(nn.Module):

    def __init__(self, in_channels=2, signal_length=2560, n_fft=128, hop_length=32,
                 hidden_channels=128, internal_n_fft=None, feature_dim=256, n_tokens=1, bands=4):
        super().__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.internal_n_fft = n_fft
        self.internal_hop = hop_length
        # [exp] 实验 8：n_tokens>1 时输出 n_tokens 个保留时间位置的局部 token（频率方向池化成 bands 个子带），默认 1 与原行为一致
        self.n_tokens, self.bands = int(n_tokens), int(bands)

        self.mask_cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, hidden_channels),
            nn.SiLU(),
        )

        self.register_buffer("stft_window", torch.hann_window(self.internal_n_fft))
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_channels, feature_dim)
        if self.n_tokens > 1:
            self.tok_pool = nn.AdaptiveAvgPool2d((self.bands, self.n_tokens))
            self.tok_proj = nn.Linear(hidden_channels * self.bands, feature_dim)
            pos = torch.arange(self.n_tokens, dtype=torch.float32)[:, None]
            div = torch.exp(torch.arange(0, feature_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / feature_dim))
            pe = torch.zeros(self.n_tokens, feature_dim); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('tok_pos', pe, persistent=False)

    def forward(self, stft_feat, time_feat):
        del stft_feat

        bsz, channels, _ = time_feat.shape

        stft_list = []
        for ch in range(channels):
            s = torch.stft(
                time_feat[:, ch, :],
                n_fft=self.internal_n_fft,
                hop_length=self.internal_hop,
                win_length=self.internal_n_fft,
                window=self.stft_window,
                return_complex=True,
            )
            stft_list.append(s)

        stft_internal = torch.stack(stft_list, dim=1)      # (B, 2, F, T)
        if torch.is_complex(stft_internal):
            # Avoid complex abs CUDA JIT path that depends on nvrtc-builtins DLL.
            mag = torch.sqrt(stft_internal.real.pow(2) + stft_internal.imag.pow(2) + 1e-12)
        else:
            mag = torch.abs(stft_internal)                  # (B, 2, F, T)

        h = self.mask_cnn(mag)                              # (B, H, F, T)
        if self.n_tokens > 1:                               # [exp] 局部 TF token：(B, M, D)，带时间位置编码；每个 token 对应一段时间窗
            z = self.tok_pool(h)                            # (B, H, bands, M)
            z = z.permute(0, 3, 1, 2).reshape(bsz, self.n_tokens, -1)
            return self.tok_proj(z) + self.tok_pos.unsqueeze(0)
        pooled = self.pool2d(h).view(bsz, -1)              # (B, H)
        feat = self.proj(pooled)                            # (B, D)
        return feat


class TD_Expert(nn.Module):

    def __init__(self, in_channels=2, signal_length=2560, hidden_channels=128, feature_dim=256):
        super().__init__()
        _ = signal_length

        self.envelope_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GroupNorm(2, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GroupNorm(2, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(2, hidden_channels),
            nn.SiLU(),
        )

        self.threshold_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        # 鎷兼帴 [鍖呯粶鍏ㄥ眬鐗瑰緛, 闃堝€肩壒寰乚 鍚庢姇褰?
        self.proj = nn.Linear(hidden_channels + 1, feature_dim)

    def forward(self, time_feat):
        """
        Args:
            time_feat: (B, 2, L)

        Returns:
            feat: (B, D)
        """
        h = self.envelope_net(time_feat)                    # (B, H, L)
        h_pool = self.pool(h).squeeze(-1)                   # (B, H)
        threshold = self.threshold_net(h)                   # (B, 1)

        fused = torch.cat([h_pool, threshold], dim=-1)      # (B, H+1)
        feat = self.proj(fused)                             # (B, D)
        return feat


class GenericExpert(nn.Module):
    """[exp] 非物理的“通用”1D CNN 专家（控制实验）：与物理专家同样接受原始 I/Q，输出同维特征，但不使用任何域变换（FFT/STFT/多尺度）。
    用于检验“物理域表征本身”与“多一个同规模分支/参数量”的区别（实验 1、2 的对照）。"""

    def __init__(self, in_channels=2, signal_length=2560, hidden_channels=96, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 7, stride=2, padding=3), nn.GroupNorm(4, hidden_channels), nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, stride=2, padding=2), nn.GroupNorm(4, hidden_channels), nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, stride=2, padding=1), nn.GroupNorm(4, hidden_channels), nn.SiLU())
        self.proj = nn.Linear(2 * hidden_channels, feature_dim)

    def forward(self, time_feat):
        h = self.net(time_feat)
        return self.proj(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


class ExpertType:
    """Expert type constants."""

    FREQUENCY_DOMAIN = "FD"
    MULTI_SCALE = "MS"
    TIME_FREQUENCY = "TF"
    TIME_DOMAIN = "TD"
    GENERIC = ("GN", "GN2", "GN3", "GN4")      # [exp] 通用 CNN 专家（控制）
    CYCLO = "CYC"                              # [exp] 实验 M2：循环平稳专家
    CYCLO_CONTROL = "CYX"                      # [exp] M2 对照：离循环 α 的同流程专家


DOMAIN_EXPERT_SPECS = {
    "FD": {
        "class": FD_Expert,
        "description": "棰戝煙鐗瑰緛 (ST/MT)",
        "input_keys": ["freq_feat", "time_feat"],
    },
    "MS": {
        "class": MS_Expert,
        "description": "澶氬昂搴︾壒寰?(NB/NFM/WB)",
        "input_keys": ["ms_feat", "time_feat"],
    },
    "TF": {
        "class": TF_Expert,
        "description": "鏃堕鐗瑰緛 (LFM/RFM)",
        "input_keys": ["stft_feat", "time_feat"],
    },
    "TD": {
        "class": TD_Expert,
        "description": "鏃跺煙鐬€佺壒寰?(PL)",
        "input_keys": ["time_feat"],
    },
}

from .cyclo_expert import CycloExpert, CycloControlExpert  # noqa: E402
DOMAIN_EXPERT_SPECS["CYC"] = {"class": CycloExpert, "description": "cyclostationary (M2)", "input_keys": ["time_feat"]}
DOMAIN_EXPERT_SPECS["CYX"] = {"class": CycloControlExpert, "description": "off-cycle control (M2)", "input_keys": ["time_feat"]}

for _g in ExpertType.GENERIC:
    DOMAIN_EXPERT_SPECS[_g] = {"class": GenericExpert, "description": "generic CNN (control)", "input_keys": ["time_feat"]}

ALL_DOMAIN_EXPERT_NAMES = [k for k in DOMAIN_EXPERT_SPECS.keys() if k not in ExpertType.GENERIC and k not in ("CYC", "CYX")]     # 默认四个物理专家，GN* 仅显式指定时使用



