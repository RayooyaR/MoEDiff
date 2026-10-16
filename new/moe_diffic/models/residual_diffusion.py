"""实验 10：初始恢复 + 残差扩散。

   s_base = f_φ(y)（冻结的监督 CNN），r = (s − s_base) / res_scale，扩散只生成 r；
   ŝ = s_base + λ · res_scale · r̂ （λ 由 CorrectionGate 给出，默认 1）。
去噪网络输入通道 = [x_t(2), y(2), s_base(2)] = 6，条件仍为（可选的）MoE token。
训练残差的来源有两种：
   1) 'oof'  ：用 K 折交叉得到的“样本外”基础估计（make_oof_base.py 生成，按 clean_id 查表），避免过拟合基础恢复器造成的残差偏小；
   2) 'online'：在线用冻结的基础恢复器直接计算（对训练集会偏乐观，仅作对照/回退）。
推理时 s_base = f_φ(y) 对任何输入都是样本外的。
"""
import torch
import torch.nn as nn

from .baseline.cnn_baseline import PhysicsCNNBaseline
from .correction_gate import CorrectionGate
from .physics_moe_model import PhysicsMoEModel


class ResidualDiffusionModel(PhysicsMoEModel):
    def __init__(self, config, base_ch=32):
        config.denoiser.in_channels = 6
        super().__init__(config)
        self.base = PhysicsCNNBaseline(in_channels=2, base_ch=base_ch, signal_length=config.data.signal_length, freq_loss_weight=0.1)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.register_buffer('res_scale', torch.ones(()))
        self.gate = None
        self.gate_sir = None          # oracle 变体用的真实 SIR（仅分析）
        self._sb = None

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()                                        # 基础恢复器始终 eval
        return self

    def set_gate(self, gate):
        self.gate = gate

    @torch.no_grad()
    def base_pred(self, y):
        out = self.base(y)
        return out['pred'] if isinstance(out, dict) else out

    def cond_extra(self, x_noisy):
        return self._sb

    def forward(self, x_clean, x_noisy):
        sb = self._aux.get('base_pred')
        if sb is None:
            sb = self.base_pred(x_noisy)
        self._sb = sb.detach()
        r = (x_clean - self._sb) / self.res_scale
        out = super()._training_forward(r, x_noisy)
        out['s_base'] = self._sb
        return out

    @torch.no_grad()
    def sample(self, x_noisy, num_steps=None, return_intermediates=False, init_scale=1.0, return_parts=False):
        sb = self.base_pred(x_noisy)
        self._sb = sb
        r = super().sample(x_noisy, num_steps=num_steps, init_scale=init_scale) * self.res_scale
        lam = 1.0
        if self.gate is not None and self.gate.variant != 'none':
            lam = self.gate(x_noisy, sb, r, sir=self.gate_sir)
        self._sb = None
        out = sb + lam * r
        return (out, sb, r, lam) if return_parts else out


class ResidualCNNBaseline(nn.Module):
    """实验 10 的对照：冻结的 CNN 初始恢复 + 一个**普通残差 CNN**（输入 [y, s_base]，直接回归残差，不做扩散）。
    用于回答“收益来自扩散本身，还是仅来自‘初始恢复 + 残差修正’这种结构”。"""

    def __init__(self, config, base_ch=32):
        super().__init__()
        L = config.data.signal_length
        self.base = PhysicsCNNBaseline(in_channels=2, base_ch=base_ch, signal_length=L, freq_loss_weight=0.1)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.res = PhysicsCNNBaseline(in_channels=4, base_ch=base_ch, signal_length=L, freq_loss_weight=0.1)
        self.register_buffer('res_scale', torch.ones(()))
        self._aux = {}

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def set_aux(self, **kw):
        self._aux = {k: v for k, v in kw.items() if v is not None}

    @torch.no_grad()
    def base_pred(self, y):
        out = self.base(y)
        return out['pred'] if isinstance(out, dict) else out

    def forward(self, x_clean, x_noisy):
        sb = self._aux.get('base_pred')
        if sb is None:
            sb = self.base_pred(x_noisy)
        r_hat = self.res._forward_core(torch.cat([x_noisy, sb], dim=1))[:, :2].float()
        r = (x_clean - sb) / self.res_scale
        mse = torch.nn.functional.mse_loss(r_hat, r)
        freq = torch.nn.functional.mse_loss(torch.fft.rfft(r_hat, dim=-1, norm='ortho').abs(), torch.fft.rfft(r, dim=-1, norm='ortho').abs())
        loss = mse + 0.1 * freq
        zero = r_hat.new_zeros(())
        return {'pred': sb + r_hat * self.res_scale, 'loss': loss, 'mse_loss': mse.detach(), 'freq_loss': freq.detach(), 'noise_pred': r_hat, 'noise_target': r,
                'gates': None, 'balance_loss': zero, 'route_logits': None, 'diversity_loss': None}

    @torch.no_grad()
    def sample(self, x_noisy, num_steps=None, return_intermediates=False, return_parts=False, **kw):
        sb = self.base_pred(x_noisy)
        r = self.res._forward_core(torch.cat([x_noisy, sb], dim=1))[:, :2].float() * self.res_scale
        out = sb + r
        return (out, sb, r, 1.0) if return_parts else out
