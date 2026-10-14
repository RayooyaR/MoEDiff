import math
import torch
from diffusers import DDIMScheduler


class PhysicsMoEScheduler(DDIMScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        set_alpha_to_one: bool = False,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        **kwargs
    ):
        if beta_schedule == "sigmoid":
            s = torch.linspace(-6, 6, num_train_timesteps)
            betas = torch.sigmoid(s) * (beta_end - beta_start) + beta_start
            super().__init__(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule="linear",
                clip_sample=clip_sample,
                set_alpha_to_one=set_alpha_to_one,
                steps_offset=steps_offset,
                prediction_type=prediction_type,
                **kwargs
            )
            self.betas = betas
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        elif beta_schedule == "cosine":
            super().__init__(
                num_train_timesteps=num_train_timesteps,
                beta_schedule="scaled_linear",
                clip_sample=clip_sample,
                set_alpha_to_one=set_alpha_to_one,
                steps_offset=steps_offset,
                prediction_type=prediction_type,
                **kwargs
            )
            steps = num_train_timesteps + 1
            s = 0.008
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
            self.betas = betas
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        else:
            super().__init__(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                set_alpha_to_one=set_alpha_to_one,
                steps_offset=steps_offset,
                prediction_type=prediction_type,
                **kwargs
            )


def create_scheduler(config):
    return PhysicsMoEScheduler(
        num_train_timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
        # [new-data] 原为 True：DDIM 每步把 x0 估计裁到 [-1, 1]，而归一化后的信号峰值可达约 3（16QAM 有 36.8% 的点 > 1），
        # 会系统性压平恢复波形的峰值。已关闭。
        clip_sample=False,
        prediction_type="epsilon",
    )


class PhysicsMoESampler:
    def __init__(self, model, scheduler, eta=0.0, init_scale=1.0):
        self.model = model
        self.scheduler = scheduler
        self.eta = eta
        self.init_scale = float(init_scale)       # [new-data] 初始噪声缩放；1=标准，0=从零起步的确定性路径

    @torch.no_grad()
    def sample(self, x_noisy, num_steps=None):
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        num_steps = num_train_timesteps if num_steps is None else min(num_steps, num_train_timesteps)
        return self._ddim_sample(x_noisy, num_steps)

    def _ddim_sample(self, x_noisy, num_steps):
        device = x_noisy.device
        batch_size, _, signal_length = x_noisy.shape
        moe_cond, _ = self.model.compute_moe_condition(x_noisy, training=False)
        self.scheduler.set_timesteps(num_steps, device=device)
        x_t = self.init_scale * torch.randn(batch_size, 2, signal_length, device=device)

        for t in self.scheduler.timesteps:
            t_batch = t.expand(batch_size)
            noise_pred = self.model.denoiser(torch.cat([x_t, x_noisy], dim=1), t_batch, moe_cond)
            x_t = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=x_t,
                eta=self.eta,
            ).prev_sample

        return x_t


def create_sampler(model, config, eta=0.0):
    return PhysicsMoESampler(model, create_scheduler(config), eta=eta)
