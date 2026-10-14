"""验证 clip_sample 修复：用"理想噪声预测器"跑完整 DDIM 采样，峰值为 3 的信号应被原样还原。
需要在 new/moe_diffic 目录下运行（导入其中的 models.physics_moe_pipeline）。
"""
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0, '.')
from models.physics_moe_pipeline import create_scheduler, PhysicsMoEScheduler

cfg = SimpleNamespace(diffusion=SimpleNamespace(timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='cosine'))
torch.manual_seed(0)
x0 = torch.randn(4, 2, 2560) * 1.0
x0[:, :, ::97] = 3.0   # 构造峰值 3（归一化后的 16QAM 就有这个量级）

def run(sched, steps=50):
    sched.set_timesteps(steps)
    x_t = torch.randn_like(x0)
    for t in sched.timesteps:
        a = sched.alphas_cumprod[t]
        eps = (x_t - a.sqrt() * x0) / (1 - a).sqrt()      # 理想噪声预测
        x_t = sched.step(eps, t, x_t, eta=0.0).prev_sample
    return x_t

fixed = run(create_scheduler(cfg))
old = run(PhysicsMoEScheduler(num_train_timesteps=1000, beta_schedule='cosine', clip_sample=True))
print('修复后(create_scheduler): 最大 |误差| = %.2e, 输出峰值 = %.2f' % ((fixed - x0).abs().max(), fixed.max()))
print('原设置(clip_sample=True): 最大 |误差| = %.2f,  输出峰值 = %.2f' % ((old - x0).abs().max(), old.max()))
# 采样器返回最后一步的 x_t；调度表末端 alpha=0.9999，残留约 1% 的噪声（峰值 3 时约 0.04），与 clip 无关
assert (fixed - x0).abs().max() < 0.1, '修复后应能还原（残留噪声底除外）'
assert (old - x0).abs().max() > 1.0, '原设置应被截断'
print('PASS')
