import torch
import torch.nn as nn

from .modal_projector import ModalProjector
from .physics_moe import PhysicsMoE
from .dit_blocks import DiTDenoiser1D

def build_denoiser(config, in_channels, out_channels, signal_length, hidden_size, cond_mode='xattn'):
    """去噪主干工厂（实验 7/9）。默认 dit + xattn 与原版一致。"""
    bb = getattr(config.denoiser, 'backbone', 'dit')
    if bb == 'dit':
        kw = {} if cond_mode == 'xattn' else {'cond_mode': cond_mode}
        return DiTDenoiser1D(in_channels=in_channels, out_channels=out_channels, signal_length=signal_length, hidden_size=hidden_size,
                             depth=config.denoiser.depth, num_heads=config.denoiser.num_heads, mlp_ratio=config.denoiser.mlp_ratio,
                             patch_size=config.denoiser.patch_size, **kw)
    from .unet1d_denoiser import UNet1DDenoiser, DiffWave1DDenoiser
    if bb == 'unet1d':
        return UNet1DDenoiser(in_channels, out_channels, signal_length, hidden_size=hidden_size, base=int(getattr(config.denoiser, 'unet_base', 64)),
                              cond_mode=cond_mode)
    if bb == 'diffwave':
        return DiffWave1DDenoiser(in_channels, out_channels, signal_length, hidden_size=hidden_size, res_channels=int(getattr(config.denoiser, 'diffwave_res', 128)),
                                  layers=int(getattr(config.denoiser, 'diffwave_layers', 20)), cond_mode=cond_mode)
    raise ValueError('unknown denoiser backbone %s' % bb)


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

        self.extra_local = getattr(config.moe, 'extra_local', 'none')          # E1：局部干扰定位通道（oracle 上界 / 由 y 预测）
        self.e1_loss = None
        if self.extra_local == 'e1':
            from .local_cond import E1PredictorCNN
            self.e1 = E1PredictorCNN()
        self.oracle_cond = getattr(config.moe, 'oracle_cond', 'none')          # 诊断用：真实干扰类型/属性作为条件（信息上界）
        if self.oracle_cond != 'none':
            d_in = (8 if 'type' in self.oracle_cond else 0) + (12 if 'attrs' in self.oracle_cond else 0)
            self.oracle_proj = nn.Sequential(nn.Linear(d_in, 128), nn.SiLU(), nn.Linear(128, hidden_size))
        self.cond_mode = getattr(config.moe, 'cond_mode', 'xattn')
        self.true_uncond = bool(getattr(config.moe, 'true_uncond', False))
        self.denoiser_in_channels = int(getattr(config.denoiser, 'in_channels', 4))
        self.denoiser = build_denoiser(config, self.denoiser_in_channels, 2, signal_length, hidden_size, self.cond_mode)
        self._aux = {}

        from .physics_moe_pipeline import create_scheduler
        self.scheduler = create_scheduler(config)
        self.num_timesteps = config.diffusion.timesteps

    def compute_moe_condition(self, x_noisy, training=True):
        cond, info = self._compute_moe_condition(x_noisy, training)
        cf = getattr(self, 'cf_mode', None)                         # 诊断用反事实：把条件置零 / 在批内错位成别的样本的条件（默认 None，不影响任何正常流程）
        if cf is not None and cond is not None:
            cond = torch.zeros_like(cond) if cf == 'zero' else cond.roll(1, dims=0)
        return cond, info

    def _compute_moe_condition(self, x_noisy, training=True):
        if self.oracle_cond != 'none' and self.moe_enabled and self.moe is not None:       # TF 等专家 + oracle：oracle token 追加在专家 token 之后（静态等权，经 AdaLN 池化）
            cond, info = self._compute_moe_condition_experts(x_noisy, training)
            tok = self.oracle_proj(self._aux['oracle'].to(x_noisy.dtype)).unsqueeze(1)
            mask = info.get('cond_mask')
            if mask is not None:
                info = dict(info); info['cond_mask'] = torch.cat([mask, torch.zeros(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)], dim=1)
            return torch.cat([cond, tok], dim=1), info
        return self._compute_moe_condition_experts(x_noisy, training)

    def _compute_moe_condition_experts(self, x_noisy, training=True):
        if self.oracle_cond != 'none' and not (self.moe_enabled and self.moe is not None):
            tok = self.oracle_proj(self._aux['oracle'].to(x_noisy.dtype)).unsqueeze(1)      # (B, 1, hidden)；与 TF 专家同样经 cond_mode(AdaLN) 注入
            return tok, {'gates': None, 'balance_loss': torch.zeros((), device=x_noisy.device, dtype=x_noisy.dtype), 'diversity_loss': None}
        if not self.moe_enabled or self.moe is None:
            batch_size = x_noisy.shape[0]
            device = x_noisy.device
            hidden_size = self.denoiser.hidden_size
            if self.true_uncond:      # [exp] 真正的无条件分支：不传伪零 token（零 token 经交叉注意力的偏置投影仍会产生非零输出）
                return None, {'gates': None, 'balance_loss': torch.zeros((), device=device, dtype=x_noisy.dtype), 'diversity_loss': None}
            return (
                torch.zeros(batch_size, 1, hidden_size, device=device, dtype=x_noisy.dtype), # 空序列
                {'gates': None, 'balance_loss': torch.zeros((), device=device, dtype=x_noisy.dtype), 'diversity_loss': None},
            )

        self.moe.aux_in = self._aux
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
                                             'route_logits': getattr(self.moe, 'last_route_logits', None),
                                             'cond_mask': getattr(self.moe, 'last_mask', None), 'aux_losses': dict(self.moe.aux_out)}

    def set_aux(self, **kw):
        """训练期写入辅助标签（sir、attrs）；推理时必须清空（oracle_sir 变体除外，仅作分析）。"""
        self._aux = {k: v for k, v in kw.items() if v is not None}

    def cond_extra(self, x_noisy):
        """额外的去噪网络输入通道（残差扩散返回基础估计 s_base）。默认无。"""
        if self.extra_local == 'none':
            return None
        from .local_cond import interf_features, upsample_time
        L = x_noisy.shape[-1]
        if self.extra_local == 'oracle_mask':
            return upsample_time(interf_features(self._aux['interf']), L).to(x_noisy.dtype)
        pred = self.e1(x_noisy)                                                     # (B, 16, L/32)
        if self.training and self._aux.get('interf') is not None:
            tgt = interf_features(self._aux['interf'])                              # (B, 16, T)
            self.e1_loss = torch.nn.functional.mse_loss(upsample_time(pred.float(), tgt.shape[-1]), tgt)
        return upsample_time(pred, L)

    def denoise(self, x_t, x_noisy, t, cond, cond_mask=None, extra=None):
        """去噪网络调用的统一入口：输入通道 = [x_t, x_noisy(, extra...)]。"""
        inp = torch.cat([x_t, x_noisy] + ([extra] if extra is not None else []), dim=1)
        return self.denoiser(inp, t, cond, cond_mask) if cond_mask is not None else self.denoiser(inp, t, cond)

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
        noise_pred = self.denoise(x_t, x_noisy, t, moe_cond_seq, moe_info.get('cond_mask'), self.cond_extra(x_noisy)).float()      # bf16 自动混合精度下转回 fp32 再算 x0
        
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
            'aux_losses': moe_info.get('aux_losses', {}),
        }

    def _inference_forward(self, x_t, x_noisy, t, return_moe_info=False):
        moe_cond_seq, moe_info = self.compute_moe_condition(x_noisy, training=self.training)
        noise_pred = self.denoise(x_t, x_noisy, t, moe_cond_seq, moe_info.get('cond_mask'), self.cond_extra(x_noisy))
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