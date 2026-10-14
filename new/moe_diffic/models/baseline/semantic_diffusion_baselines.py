import torch
import torch.nn as nn

from ..old.icdm_baseline import ICDMBaseline


class DiffCom1DBaseline(nn.Module):
    def __init__(self, config, dps_step_size: float = 0.1):
        super().__init__()
        self.core = ICDMBaseline(config)
        self.dps_step_size = float(dps_step_size)

    def forward(self, x_clean, x_noisy):
        return self.core(x_clean, x_clean)

    @torch.no_grad()
    def sample(self, x_noisy, num_steps=None, return_intermediates=False):
        if num_steps is None or num_steps >= self.core.num_timesteps:
            num_steps = int(self.core.num_timesteps)

        device = x_noisy.device
        bsz = x_noisy.shape[0]
        times = torch.linspace(self.core.num_timesteps - 1, 0, num_steps + 1, device=device).long()
        x_t = torch.randn(bsz, 2, x_noisy.shape[-1], device=device)

        for i in range(num_steps):
            t = times[i]
            t_next = times[i + 1]
            t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)

            with torch.enable_grad():
                x_t_req = x_t.detach().requires_grad_(True)
                x_input = torch.cat([x_t_req, x_noisy], dim=1)
                noise_pred = self.core.denoiser(x_input, t_batch)

                alpha_t = self.core.alphas_cumprod[t]
                x_0_pred = (x_t_req - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                loss_measurement = torch.mean((x_0_pred - x_noisy) ** 2)
                grad = torch.autograd.grad(loss_measurement, x_t_req, only_inputs=True)[0]
                x_t_corr = (x_t_req - self.dps_step_size * grad).detach()

            alpha_t = self.core.alphas_cumprod[t]
            x_0_pred = (x_t_corr - torch.sqrt(1 - alpha_t) * noise_pred.detach()) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -5.0, 5.0)

            if t_next <= 0:
                x_t = x_0_pred
                continue

            alpha_next = self.core.alphas_cumprod[t_next]
            sigma = torch.tensor(0.0, device=device)
            c = torch.sqrt(torch.clamp(1 - alpha_next - sigma ** 2, min=0.0))
            noise = torch.zeros_like(x_t_corr)
            x_t = torch.sqrt(alpha_next) * x_0_pred + c * noise_pred.detach() + sigma * noise

        return x_t


class ICDMDual1DBaseline(nn.Module):
    def __init__(self, config, dual_gamma: float = 1.0):
        super().__init__()
        self.model_s = ICDMBaseline(config)
        self.model_b = ICDMBaseline(config)
        self.dual_gamma = float(dual_gamma)

    def forward(self, x_clean, x_noisy):
        out_s = self.model_s(x_clean, x_clean)
        x_interf = x_noisy - x_clean
        out_b = self.model_b(x_interf, x_interf)

        noise_pred = 0.5 * (out_s['noise_pred'] + out_b['noise_pred'])
        noise_target = 0.5 * (out_s['noise_target'] + out_b['noise_target'])
        clean_pred = 0.5 * (out_s.get('moe_clean_pred', 0.0) + out_b.get('moe_clean_pred', 0.0))

        return {
            'noise_pred': noise_pred,
            'noise_target': noise_target,
            'moe_clean_pred': clean_pred,
            'gates': None,
            'balance_loss': torch.tensor(0.0, device=x_clean.device),
        }

    @torch.no_grad()
    def sample(self, x_noisy, num_steps=None, return_intermediates=False):
        if num_steps is None or num_steps >= self.model_s.num_timesteps:
            num_steps = int(self.model_s.num_timesteps)

        device = x_noisy.device
        bsz = x_noisy.shape[0]
        times = torch.linspace(self.model_s.num_timesteps - 1, 0, num_steps + 1, device=device).long()
        x_t = torch.randn(bsz, 2, x_noisy.shape[-1], device=device)

        for i in range(num_steps):
            t = times[i]
            t_next = times[i + 1]
            t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)

            x_input_s = torch.cat([x_t, x_noisy], dim=1)
            noise_s = self.model_s.denoiser(x_input_s, t_batch)

            b_t = x_noisy - x_t
            x_input_b = torch.cat([b_t, b_t], dim=1)
            noise_b = self.model_b.denoiser(x_input_b, t_batch)

            noise_comb = noise_s - self.dual_gamma * noise_b

            alpha_t = self.model_s.alphas_cumprod[t]
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_comb) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -5.0, 5.0)

            if t_next <= 0:
                x_t = x_0_pred
                continue

            alpha_next = self.model_s.alphas_cumprod[t_next]
            sigma = torch.tensor(0.0, device=device)
            c = torch.sqrt(torch.clamp(1 - alpha_next - sigma ** 2, min=0.0))
            noise = torch.zeros_like(x_t)
            x_t = torch.sqrt(alpha_next) * x_0_pred + c * noise_comb + sigma * noise

        return x_t
