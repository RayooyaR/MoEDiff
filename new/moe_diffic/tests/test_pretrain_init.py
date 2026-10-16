"""预训练权重迁移的正确性：denoiser.* 逐张量一致载入（取 EMA 权重），y 通道输入卷积置零，其余（TF 专家）重新初始化。"""
import os, subprocess, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); py = sys.executable
common = ['--model_type', 'moe_diffmic', '--backbone', 'unet1d', '--unet_base', '64', '--cond_mode', 'adaln', '--amp', 'bf16', '--new_types', 'ST', 'MT', 'NB', 'WB', 'PL', '--seed', '42']
pool = '/tmp/_pool_test.npz'
subprocess.run([py, os.path.join(ROOT, '..', 'gen', 'make_clean_pool.py'), '--n', '96', '--out', pool], check=True, capture_output=True)
subprocess.run([py, '-W', 'ignore', os.path.join(ROOT, 'pretrain_clean.py'), '--exp_name', '_t_pre', '--pretrain_pool', pool, '--pretrain_steps', '20', '--disable_moe', '--true_uncond'] + common, check=True, capture_output=True, cwd=ROOT)
r = subprocess.run([py, '-W', 'ignore', os.path.join(ROOT, 'exp_train.py'), '--exp_name', '_t_ft', '--experts', 'TF', '--init_from', '_t_pre', '--epochs', '1', '--train_per_cell', '1', '--val_per_cell', '1',
                    '--val_stride', '30', '--skip_final_eval', '--ddim_steps', '2', '--num_workers', '0', '--lr', '0', '--abort_chance_frac', '9'] + common, capture_output=True, text=True, cwd=ROOT)
assert r.returncode == 0, r.stderr[-500:]
assert '[init_from]' in r.stdout and 'denoiser.* 共' in r.stdout
ck_pre = torch.load(os.path.join(ROOT, 'runs', '_t_pre', 'best_model.pth'), map_location='cpu', weights_only=False)
ck_ft = torch.load(os.path.join(ROOT, 'runs', '_t_ft', 'best_model.pth'), map_location='cpu', weights_only=False)
# lr=0：微调后的 denoiser.* 应与预训练 EMA 权重逐位相等（y 通道的输入卷积除外，应为 0）；TF 专家等为重新初始化
sd_pre, sd_ft, ema_pre = ck_pre['model_state_dict'], ck_ft['model_state_dict'], ck_pre['ema_shadow']
den = [k for k in sd_pre if k.startswith('denoiser.')]
for k in den:
    ref = ema_pre.get(k, sd_pre[k]).cpu()
    got = sd_ft[k].cpu()
    if k == 'denoiser.inp.weight':
        assert torch.equal(got[:, :2], ref[:, :2]) and got[:, 2:4].abs().sum() == 0
    else:
        assert torch.equal(got, ref), k
assert any(k.startswith('moe.') for k in sd_ft) and not any(k.startswith('moe.') for k in sd_pre)
import shutil
for n in ('_t_pre', '_t_ft'): shutil.rmtree(os.path.join(ROOT, 'runs', n), ignore_errors=True)
print('OK 预训练迁移：denoiser.* %d 个张量键/形状一致，TF 专家重新初始化' % len(den))
