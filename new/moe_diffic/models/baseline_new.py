"""两个"前沿方法迁移"基线的重写版（1D 复基带信号）。

为什么重写：原始代码 models/baseline/semantic_diffusion_baselines.py 依赖不存在的 models/old/icdm_baseline.py，无法运行；
并且代码本身有两处会使基线失效的问题：
  (1) DiffCom 的 DPS 引导用 torch.mean（对整个 batch 取平均）算测量残差，梯度量级约 1/(B·2·L)≈1e-5，等于没有引导；
  (2) 训练时把干净信号当作条件通道输入 (model(x_clean, x_clean))，推理时却输入含干扰的信号，训练/推理分布不一致，
      且条件通道泄漏了目标。
本文件按方法原意实现，并保持与 train.py 的接口兼容（forward 返回 dict、sample(x_noisy, num_steps)）。

DiffCom1DBaseline  —— 干净信号的无条件扩散先验 + DPS 后验采样（测量 y = x + b，b 为干扰+噪声）：
    x_{t-1} = DDIM(x_t) − ζ · ∇_{x_t} ‖y − x̂0(x_t)‖₂        （逐样本的 L2 范数，标准 DPS）
ICDMDual1DBaseline —— 两个无条件扩散先验（信号 s、干扰 b）+ 分数解耦：
    ∇log p(x|y) ≈ ∇log p_s(x) − ∇log p_b(y − x)   ⇒   ε_comb = ε_s − γ·ε_b
    其中 b 模型在其自身的归一化尺度下训练，采样时按噪声水平对齐（见 _match_b_timestep）。
"""
import torch
import torch.nn as nn

from .dit_blocks import DiTDenoiser1D
from .physics_moe_pipeline import create_scheduler


class _UncondPrior(nn.Module):
    """无条件 ε 预测扩散先验：DiT 去噪器，输入只有 x_t。"""

    def __init__(self, config):
        super().__init__()
        self.denoiser = DiTDenoiser1D(
            in_channels=2, out_channels=2,
            signal_length=config.data.signal_length,
            hidden_size=config.denoiser.hidden_size,
            depth=config.denoiser.depth,
            num_heads=config.denoiser.num_heads,
            mlp_ratio=config.denoiser.mlp_ratio,
            patch_size=config.denoiser.patch_size,
        )

    def eps(self, x_t, t):
        return self.denoiser(x_t, t, None)


def _train_dict(prior, scheduler, x0):
    """标准 ε-预测训练步。键与 PhysicsMoEModel._training_forward 对齐。"""
    bsz, device = x0.shape[0], x0.device
    t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    ac = scheduler.alphas_cumprod.to(device)
    sa = ac[t].sqrt().view(-1, 1, 1)
    s1 = (1 - ac[t]).sqrt().view(-1, 1, 1)
    x_t = sa * x0 + s1 * noise
    eps = prior.eps(x_t, t)
    return dict(eps=eps, noise=noise, x0_pred=(x_t - s1 * eps) / (sa + 1e-8), t=t, x_t=x_t)


def _compat(d, x0, device):
    zero = torch.zeros((), device=device)
    return {'noise_pred': d['eps'], 'noise_target': d['noise'], 'x0_pred': d['x0_pred'], 'x0_target': x0,
            'x_t': d['x_t'], 't': d['t'], 'gates': None, 'balance_loss': zero, 'route_logits': None,
            'diversity_loss': None}


class DiffCom1DBaseline(nn.Module):
    def __init__(self, config, dps_step_size: float = 1.0):
        super().__init__()
        self.prior = _UncondPrior(config)
        self.scheduler = create_scheduler(config)
        self.num_timesteps = config.diffusion.timesteps
        self.dps_step_size = float(dps_step_size)

    def forward(self, x_clean, x_noisy):
        d = _train_dict(self.prior, self.scheduler, x_clean)       # 先验只看干净信号；x_noisy 不参与训练
        return _compat(d, x_clean, x_clean.device)

    def sample(self, x_noisy, num_steps=None, return_intermediates=False, init_scale=1.0):
        device = x_noisy.device
        sched = self.scheduler
        sched.set_timesteps(min(num_steps or self.num_timesteps, self.num_timesteps), device=device)
        ac = sched.alphas_cumprod.to(device)
        x_t = init_scale * torch.randn_like(x_noisy)
        for t in sched.timesteps:
            tb = t.expand(x_t.shape[0])
            with torch.enable_grad():
                x_in = x_t.detach().requires_grad_(True)
                eps = self.prior.eps(x_in, tb)
                a = ac[t]
                x0 = (x_in - (1 - a).sqrt() * eps) / a.sqrt()
                resid = (x_noisy - x0).flatten(1).norm(dim=1)       # 逐样本 L2 范数
                grad = torch.autograd.grad(resid.sum(), x_in)[0]
            x_prev = sched.step(eps.detach(), t, x_t.detach(), eta=0.0).prev_sample
            x_t = (x_prev - self.dps_step_size * grad).detach()
        return x_t


class ICDMDual1DBaseline(nn.Module):
    def __init__(self, config, dual_gamma: float = 1.0):
        super().__init__()
        self.model_s = _UncondPrior(config)      # 信号先验（输入 x/data_scale）
        self.model_b = _UncondPrior(config)      # 干扰(+噪声)先验（输入 b/data_scale/b_scale）
        self.scheduler = create_scheduler(config)
        self.num_timesteps = config.diffusion.timesteps
        self.dual_gamma = float(dual_gamma)
        self.register_buffer('b_scale', torch.ones(()))

    def set_interference_scale(self, value: float):
        """b 的全局标准差（相对于 data_scale 归一化后的单位），由训练脚本根据训练集设置。"""
        self.b_scale.fill_(float(value))

    def forward(self, x_clean, x_noisy):
        ds = _train_dict(self.model_s, self.scheduler, x_clean)
        b = (x_noisy - x_clean) / self.b_scale                     # 干扰+噪声，按自身尺度归一化
        db = _train_dict(self.model_b, self.scheduler, b)
        out = _compat(ds, x_clean, x_clean.device)
        # 总损失 = 两个先验的 ε 损失的平均；x0 空间项仅对信号先验计算（干扰先验的 x0 空间项由其自身损失近似）
        out['noise_pred'] = torch.cat([ds['eps'], db['eps']], dim=0)
        out['noise_target'] = torch.cat([ds['noise'], db['noise']], dim=0)
        return out

    def _match_b_timestep(self, sigma_eff, sig_table):
        """在 b 模型的时间轴上找噪声水平 sigma_eff/b_scale 对应的 t'。"""
        target = sigma_eff / self.b_scale
        idx = torch.searchsorted(sig_table, target.reshape(1)).clamp(0, sig_table.numel() - 1)
        return idx.reshape(())

    def sample(self, x_noisy, num_steps=None, return_intermediates=False, init_scale=1.0):
        device = x_noisy.device
        sched = self.scheduler
        sched.set_timesteps(min(num_steps or self.num_timesteps, self.num_timesteps), device=device)
        ac = sched.alphas_cumprod.to(device)
        sig_table = ((1 - ac) / ac).sqrt()                          # σ_eff(t) = sqrt((1-ᾱ)/ᾱ)，随 t 单调增
        x_t = init_scale * torch.randn_like(x_noisy)
        bsz = x_t.shape[0]
        for t in sched.timesteps:
            a = ac[t]
            eps_s = self.model_s.eps(x_t, t.expand(bsz))
            # VE 形式的信号估计 x̃ = x_t/√ᾱ，其对应的干扰估计 b̃ = y − x̃（噪声水平同为 σ_eff）
            b_tilde = x_noisy - x_t / a.sqrt()
            tp = self._match_b_timestep(sig_table[t], sig_table)    # b 模型的等价时间步
            a_p = ac[tp]
            b_in = a_p.sqrt() * (b_tilde / self.b_scale)            # 转回 VP 形式 √ᾱ' · b̃'
            eps_b = self.model_b.eps(b_in, tp.expand(bsz))
            eps = eps_s - self.dual_gamma * eps_b
            x_t = sched.step(eps, t, x_t, eta=0.0).prev_sample
        return x_t


class DiTRegressionBaseline(nn.Module):
    """同骨干（DiT）的直接回归对照：不做扩散，输入受干扰信号，直接回归干净信号。

    - config.moe.enabled=False：纯 DiT 回归（与 DiT 扩散基线同骨干、同参数规模，去掉扩散）；
    - config.moe.enabled=True ：DiT + MoE 多分支条件（与 MoE-DiffIC 同结构），也去掉扩散。
    时间步固定为 0（骨干结构与扩散版相同）；损失 = MSE + 0.1·频域幅度谱 MSE（ortho 归一化，与 CNN 基线一致）+（MoE 时）均衡损失。
    推理：sample() 直接返回预测（确定性、无随机性）。
    """

    def __init__(self, config, balance_coef=0.005):
        super().__init__()
        from .physics_moe_model import PhysicsMoEModel
        self.core = PhysicsMoEModel(config)                       # 复用其 modal_projector / MoE / 条件投影
        self.core.denoiser = DiTDenoiser1D(                        # 换成只接收受干扰信号的 2 通道 DiT
            in_channels=2, out_channels=2, signal_length=config.data.signal_length,
            hidden_size=config.denoiser.hidden_size, depth=config.denoiser.depth,
            num_heads=config.denoiser.num_heads, mlp_ratio=config.denoiser.mlp_ratio, patch_size=config.denoiser.patch_size)
        self.balance_coef = float(balance_coef)

    def _predict(self, x_noisy, training):
        cond, info = self.core.compute_moe_condition(x_noisy, training=training)
        t = torch.zeros(x_noisy.shape[0], dtype=torch.long, device=x_noisy.device)
        return self.core.denoiser(x_noisy, t, cond), info

    def forward(self, x_clean, x_noisy):
        pred, info = self._predict(x_noisy, self.training)
        mse = torch.nn.functional.mse_loss(pred, x_clean)
        freq = torch.nn.functional.mse_loss(torch.fft.rfft(pred, dim=-1, norm='ortho').abs(), torch.fft.rfft(x_clean, dim=-1, norm='ortho').abs())
        bal = info['balance_loss'] if torch.is_tensor(info.get('balance_loss')) else pred.new_zeros(())
        loss = mse + 0.1 * freq + self.balance_coef * bal
        zero = pred.new_zeros(())
        return {'pred': pred, 'loss': loss, 'mse_loss': mse.detach(), 'freq_loss': freq.detach(), 'noise_pred': pred, 'noise_target': x_clean,
                'gates': info['gates'], 'balance_loss': zero, 'route_logits': None, 'diversity_loss': None}

    @torch.no_grad()
    def sample(self, x_noisy, num_steps=None, return_intermediates=False, **kw):
        return self._predict(x_noisy, False)[0]


class MultiViewCNNBaseline(nn.Module):
    """多分支特征 + CNN 对照：在原 U-Net CNN 的瓶颈层，用交叉注意力注入与 MoE-DiffIC 完全相同的多分支专家特征。

    多分支特征 = MoE-DiffIC 里的 ModalProjector + PhysicsMoE（FFT / STFT / 多尺度 / 时域 四个专家 + 原路由），输出 4 个 token（256 维）；
    CNN 瓶颈特征 (B,256,L/8) 作为 query，对这 4 个 token 做交叉注意力（残差加回）。其余结构、损失（MSE + 0.1·频域 MSE + MoE 均衡）与 CNN 基线一致。
    不做扩散，直接回归；sample() 返回确定性预测。
    """

    def __init__(self, config, base_ch=32, balance_coef=0.005):
        super().__init__()
        from .baseline.cnn_baseline import PhysicsCNNBaseline
        from .physics_moe_model import PhysicsMoEModel
        self.core = PhysicsMoEModel(config)
        self.core.denoiser = nn.Identity()                           # 不需要扩散去噪器，只用其多分支条件提取部分
        self.cnn = PhysicsCNNBaseline(in_channels=2, base_ch=base_ch, signal_length=config.data.signal_length, freq_loss_weight=0.1)
        ch = base_ch * 8
        self.norm = nn.GroupNorm(1, ch)
        self.xattn = nn.MultiheadAttention(ch, 4, batch_first=True)
        self.balance_coef = float(balance_coef)

    def _predict(self, x_noisy, training):
        cond, info = self.core.compute_moe_condition(x_noisy, training=training)      # (B, 4, hidden)
        c = self.cnn
        x0 = c.input_proj(x_noisy)
        x1 = c.enc1(x0)
        x2 = c.enc2(x1)
        x3 = c.enc3(x2)
        b = c.bottleneck(x3)
        h = self.norm(b).permute(0, 2, 1)
        a, _ = self.xattn(h, cond, cond, need_weights=False)
        b = b + a.permute(0, 2, 1)
        d3 = c.dec3(b, x2)
        d2 = c.dec2(d3, x1)
        d1 = c.dec1(d2, x0)
        return c.output_proj(d1), info

    def forward(self, x_clean, x_noisy):
        pred, info = self._predict(x_noisy, self.training)
        mse = torch.nn.functional.mse_loss(pred, x_clean)
        freq = torch.nn.functional.mse_loss(torch.fft.rfft(pred, dim=-1, norm='ortho').abs(), torch.fft.rfft(x_clean, dim=-1, norm='ortho').abs())
        bal = info['balance_loss'] if torch.is_tensor(info.get('balance_loss')) else pred.new_zeros(())
        loss = mse + 0.1 * freq + self.balance_coef * bal
        zero = pred.new_zeros(())
        return {'pred': pred, 'loss': loss, 'mse_loss': mse.detach(), 'freq_loss': freq.detach(), 'noise_pred': pred, 'noise_target': x_clean,
                'gates': info['gates'], 'balance_loss': zero, 'route_logits': None, 'diversity_loss': None}

    @torch.no_grad()
    def sample(self, x_noisy, num_steps=None, return_intermediates=False, **kw):
        return self._predict(x_noisy, False)[0]
