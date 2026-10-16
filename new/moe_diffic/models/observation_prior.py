"""实验 12：物理一致性约束（可选，高风险）。两种“非恒等”的约束，均用于**采样期引导**（DPS 式，Chung et al., ICLR 2023）：

  band   带限约束：干净信号经 RRC 成形，带外（|f| > fmax_clean）几乎无能量。用 x̂0 的带外能量做惩罚。
         （推理时不知道每条样本的 roll-off，所以取保守的最大带宽 (1+0.5)/(2·SPS)。）
  obs    观测一致性：‖y − x̂0 − î‖²，其中 î 来自**独立训练的干扰估计器**（以 y 为输入、以干扰信号为目标的 CNN）。
         若直接令 î = y − x̂0，此项恒为 0、没有任何信息——这是计划中明确禁止的伪约束；这里 î 不依赖 x̂0，所以不是恒等式。

GuidedSampler 复用 DDIM 步骤，每步对 x_t 求引导项梯度并从更新量中减去：x_{t-1} ← DDIM(x_t) − w · ∇_{x_t} L(x̂0(x_t))。
"""
import torch

from .physics_moe_pipeline import PhysicsMoESampler


def band_penalty(x0, fmax=0.1875):
    """x0: (B,2,L)。带外能量占比（先 FFT 复信号）。"""
    c = torch.complex(x0[:, 0].float(), x0[:, 1].float())
    P = torch.fft.fft(c, dim=-1).abs() ** 2
    f = torch.fft.fftfreq(c.shape[-1], device=x0.device).abs()
    return (P[:, f > fmax].sum(-1) / (P.sum(-1) + 1e-8))


class GuidedSampler(PhysicsMoESampler):
    def __init__(self, model, scheduler, w_obs=0.0, w_band=0.0, i_hat=None, **kw):
        super().__init__(model, scheduler, **kw)
        self.w_obs, self.w_band, self.i_hat = float(w_obs), float(w_band), i_hat

    def _ddim_sample(self, x_noisy, num_steps):
        device = x_noisy.device
        B, _, L = x_noisy.shape
        with torch.no_grad():
            moe_cond, info = self.model.compute_moe_condition(x_noisy, training=False)
        cond_mask = info.get('cond_mask') if isinstance(info, dict) else None
        extra = self.model.cond_extra(x_noisy)
        self.scheduler.set_timesteps(num_steps, device=device)
        ac = self.scheduler.alphas_cumprod.to(device)
        x_t = self.init_scale * torch.randn(B, 2, L, device=device)
        for t in self.scheduler.timesteps:
            tb = t.expand(B)
            with torch.enable_grad():
                x_in = x_t.detach().requires_grad_(True)
                eps = self.model.denoise(x_in, x_noisy, tb, moe_cond, cond_mask, extra).float()
                a = ac[t]
                x0 = (x_in - (1 - a).sqrt() * eps) / a.sqrt()
                loss = 0.0
                if self.w_obs > 0 and self.i_hat is not None:
                    loss = loss + self.w_obs * (x_noisy - x0 - self.i_hat).flatten(1).norm(dim=1).sum()
                if self.w_band > 0:
                    loss = loss + self.w_band * band_penalty(x0).sum()
                grad = torch.autograd.grad(loss, x_in)[0] if torch.is_tensor(loss) else 0.0
            x_prev = self.scheduler.step(model_output=eps.detach(), timestep=t, sample=x_t.detach(), eta=self.eta).prev_sample
            x_t = (x_prev - grad).detach()
        return x_t
