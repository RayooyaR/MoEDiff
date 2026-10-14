import torch
import torch.nn as nn

from .modal_projector import ModalProjector
from .physics_moe import PhysicsMoE
from .dit_blocks import DiTDenoiser1D

class PhysicsMoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        signal_length = config.data.signal_length
        n_fft = getattr(config.modal, 'n_fft', 32) if hasattr(config, 'modal') else 32
        hop_length = getattr(config.modal, 'hop_length', 8) if hasattr(config, 'modal') else 8
        n_scales = getattr(config.modal, 'n_scales', 3) if hasattr(config, 'modal') else 3

        self.modal_projector = ModalProjector(signal_length=signal_length, n_fft=n_fft, hop_length=hop_length, n_scales=n_scales)

        self.moe_enabled = bool(getattr(config.moe, 'enabled', True)) if hasattr(config, 'moe') else True
        if self.moe_enabled:
            self.moe = PhysicsMoE(config)
            self.active_expert_names = self.moe.active_expert_names
            self.num_experts = self.moe.num_experts
        else:
            self.moe = None
            self.active_expert_names = []
            self.num_experts = 0

        hidden_size = config.denoiser.hidden_size
        moe_feature_dim = getattr(config.moe, 'expert_feature_dim', 256)
        
        # Linear 层会自动作用于序列张量的最后一个维度：(B, K, moe_feature_dim) -> (B, K, hidden_size)
        self.moe_cond_proj = nn.Identity() if moe_feature_dim == hidden_size else nn.Linear(moe_feature_dim, hidden_size)

        self.denoiser = DiTDenoiser1D(
            in_channels=4,
            out_channels=2,
            signal_length=signal_length,
            hidden_size=hidden_size,
            depth=config.denoiser.depth,
            num_heads=config.denoiser.num_heads,
            mlp_ratio=config.denoiser.mlp_ratio,
            patch_size=config.denoiser.patch_size,
        )

        from .physics_moe_pipeline import create_scheduler
        self.scheduler = create_scheduler(config)
        self.num_timesteps = config.diffusion.timesteps

    def compute_moe_condition(self, x_noisy, training=True):
        if not self.moe_enabled or self.moe is None:
            batch_size = x_noisy.shape[0]
            device = x_noisy.device
            hidden_size = self.denoiser.hidden_size
            return (
                torch.zeros(batch_size, 1, hidden_size, device=device, dtype=x_noisy.dtype), # 空序列
                {'gates': None, 'balance_loss': torch.zeros((), device=device, dtype=x_noisy.dtype), 'diversity_loss': None},
            )

        moe_outputs = self.moe(self.modal_projector(x_noisy), training=training)
        if isinstance(moe_outputs, (tuple, list)) and len(moe_outputs) == 4:
            moe_out, balance_loss, gates, diversity_loss = moe_outputs
        elif isinstance(moe_outputs, (tuple, list)) and len(moe_outputs) == 3:
            moe_out, balance_loss, gates = moe_outputs
            diversity_loss = None
        else:
            raise ValueError(f'Unexpected PhysicsMoE output structure: {type(moe_outputs)}')

        # 更新断言：此处应当是 (B, num_experts, D) 3维张量
        if not (torch.is_tensor(moe_out) and moe_out.ndim == 3):
            raise ValueError(f'Expected moe_out shape (B, K, D), got {getattr(moe_out, "shape", None)}')

        return self.moe_cond_proj(moe_out), {'gates': gates, 'balance_loss': balance_loss, 'diversity_loss': diversity_loss,
                                             'route_logits': getattr(self.moe, 'last_route_logits', None)}

    def forward(self, *args, **kwargs):
        if len(args) == 2:
            return self._training_forward(args[0], args[1])
        if len(args) == 3:
            return self._inference_forward(args[0], args[1], args[2], **kwargs)
        raise ValueError(f'Invalid number of arguments: {len(args)}. Expected 2 or 3 arguments')

    def _training_forward(self, x_clean, x_noisy):
        batch_size = x_clean.shape[0]
        device = x_clean.device
        moe_cond_seq, moe_info = self.compute_moe_condition(x_noisy, training=True)

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_clean)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        sqrt_alpha_prod = alphas_cumprod[t].sqrt().view(-1, *([1] * (x_clean.ndim - 1)))
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]).sqrt().view(-1, *([1] * (x_clean.ndim - 1)))
        x_t = sqrt_alpha_prod * x_clean + sqrt_one_minus_alpha_prod * noise
        
        # 传递条件序列给扩散网络
        noise_pred = self.denoiser(torch.cat([x_t, x_noisy], dim=1), t, moe_cond_seq)
        
        x0_pred = (x_t - sqrt_one_minus_alpha_prod * noise_pred) / (sqrt_alpha_prod + 1e-8)
        return {
            'noise_pred': noise_pred,
            'noise_target': noise,
            'x0_pred': x0_pred,
            'x0_target': x_clean,
            'x_t': x_t,
            't': t,
            'gates': moe_info['gates'],
            'balance_loss': moe_info['balance_loss'],
            'route_logits': moe_info.get('route_logits'),
            'diversity_loss': moe_info.get('diversity_loss'),
        }

    def _inference_forward(self, x_t, x_noisy, t, return_moe_info=False):
        moe_cond_seq, moe_info = self.compute_moe_condition(x_noisy, training=self.training)
        noise_pred = self.denoiser(torch.cat([x_t, x_noisy], dim=1), t, moe_cond_seq)
        return (noise_pred, moe_info) if return_moe_info else noise_pred

    def get_moe_enhanced(self, x_signal):
        if not self.moe_enabled:
            zero_loss = torch.zeros((), device=x_signal.device, dtype=x_signal.dtype)
            return x_signal, zero_loss, None

        moe_outputs = self.moe(self.modal_projector(x_signal), training=False)
        if isinstance(moe_outputs, (tuple, list)) and len(moe_outputs) == 4:
            enhanced, balance_loss, gates, _ = moe_outputs
        else:
            enhanced, balance_loss, gates = moe_outputs
        return enhanced, balance_loss, gates

    @torch.no_grad()
    def sample(self, x_noisy, num_steps=None, return_intermediates=False, init_scale=1.0):
        from .physics_moe_pipeline import PhysicsMoESampler
        return PhysicsMoESampler(self, self.scheduler, eta=0.0, init_scale=init_scale).sample(x_noisy, num_steps=num_steps)