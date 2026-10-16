"""补全统一总表缺口的任务计划（机器 A，单卡）：生成 exp/plans/fill_A.json，随后用 launch.sh A fill <小时> 运行。
补的内容：U-Net 扩散+时频(J0) 列在各场景的缺口，及与之同口径的 DiT 无专家/时频专家缺口。均为新管线训练（bf16、步数对齐、平均 8 次采样）。
用法: python exp/fill_plan.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
FB = '--backbone unet1d --unet_base 64 --cond_mode adaln'          # 与 J0 / lof_*_tf 一致（B 机自动选出的主干）
U = '--experts TF ' + FB
ALL8 = ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM']
C = REG.COST
EP = lambda n, K=40, m=1: REG.epochs_for(P, n, K, m)
def tr(name, extra, types, K=40, mult=1, n=None, tier=1, splits=('test',), cost=None, data_root=None, holdout=None, epochs=None, cfg=''):
    d = REG.data_args(P, types, K, mult, holdout)
    if data_root: d += ' --new_data_root ../data/%s' % data_root
    ntypes = n or (len(types) if types else 8)
    R.train(name, 'FILL', 'A', 'moe_diffmic', extra, d, epochs or EP(ntypes, K, mult), int(name.rsplit('_s', 1)[1]) if '_s' in name and name.rsplit('_s', 1)[1].isdigit() else 42, cost or C['unet'], tier=tier, splits=splits,
            eval_extra=('--data_root ../data/%s' % data_root) if data_root else '', meta=dict(exp='FILL', cfg=cfg or name))

# ---- tier 1：U-Net 列的主要缺口 + zs4 的同口径对照
tr('full8_unet_s42', U, None, splits=('test', 'test_ood'), n=8)
tr('full8_tf_s42', '--experts TF', None, splits=('test', 'test_ood'), n=8, cost=C['moe'])
for z, types in (('zs1', ['NB', 'WB', 'PL', 'LFM', 'NFM', 'RFM']), ('zs3', ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM']), ('zs4', ['ST', 'MT', 'PL', 'LFM', 'NFM', 'RFM'])):
    tr(f'{z}_unet_s42', U, types)
tr('zs4_dit', '--disable_moe --true_uncond', ['ST', 'MT', 'PL', 'LFM', 'NFM', 'RFM'], cost=C['dit'])
tr('zs4_ex_TF', '--experts TF', ['ST', 'MT', 'PL', 'LFM', 'NFM', 'RFM'], cost=C['moe'])
K30, M30 = REG.SIZES[30]
for sd in (42, 43):
    tr('j0_p30_s%d' % sd, U, REG.ZS2_TRAIN, K30, M30, splits=('test', 'test_ood'))
for H in ('RFM', 'WB'):
    tr(f'lof_{H}_tf_s42', U, None, holdout=H, n=7)
# ---- tier 1：zs4_moe 已有权重，只补平均 8 次采样评估
R.add('zs4_moe_avg8', 'FILL', 'A', '%s evaluate_new.py --runs zs4_moe --splits test --stride 2 --avg_k 8 --tag _avg8 --amp bf16' % REG.PYV, 0.3, tier=1, meta=dict(kind='script', exp='FILL'))
# ---- tier 2：少样本 K 轴(步数对齐、不增强) + 其他数据集
for K, m in ((1, 40), (5, 8), (10, 4)):
    tr(f'fsK{K}_unet_s42', U, REG.ZS2_TRAIN, K, m, tier=1)
    tr(f'fs{K}_na_tf_s42', '--experts TF', REG.ZS2_TRAIN, K, m, tier=2, cost=C['moe'])
tr('ef_unet_s42', U, None, n=8, data_root='v5_effects', tier=2, splits=('test', 'test_ood'))
tr('ef_tf_s42', '--experts TF', None, n=8, data_root='v5_effects', tier=2, splits=('test', 'test_ood'), cost=C['moe'])
tr('ccir_unet_s42', U, None, n=1, data_root='v3b_ccir', tier=2, epochs=100)
tr('ccir_tf_s42', '--experts TF', None, n=1, data_root='v3b_ccir', tier=2, epochs=100, cost=C['moe'])

# ---- Restormer 基线（深度学习直接回归，22.8M 参数；每步约 81 ms，约为 CNN 的 11 倍）
def rest(name, types, K=40, mult=1, n=None, tier=2, splits=('test',), data_root=None, holdout=None, epochs=None):
    d = REG.data_args(P, types, K, mult, holdout)
    if data_root: d += ' --new_data_root ../data/%s' % data_root
    ntypes = n or (len(types) if types else 8)
    R.train(name, 'FILL', 'A', 'restormer', '--restormer_dim 48', d, epochs or REG.epochs_for(P, ntypes, K, mult, True),
            int(name.rsplit('_s', 1)[1]), 0.5, tier=tier, splits=splits, diffusion=False,
            eval_extra=('--data_root ../data/%s' % data_root) if data_root else '', meta=dict(exp='FILL', cfg=name))
for p_, (K_, m_) in REG.SIZES.items():
    for sd in (42, 43, 44):
        rest('rest_p%d_s%d' % (p_, sd), REG.ZS2_TRAIN, K_, m_, tier=1 if p_ == 10 else 2, splits=('test', 'test_ood'))
rest('rest_full8_s42', None, n=8, splits=('test', 'test_ood'))
for z, types in (('zs1', ['NB', 'WB', 'PL', 'LFM', 'NFM', 'RFM']), ('zs3', ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM']), ('zs4', ['ST', 'MT', 'PL', 'LFM', 'NFM', 'RFM'])):
    rest(f'rest_{z}_s42', types)
for K_, m_ in ((1, 40), (5, 8), (10, 4)):
    rest(f'rest_K{K_}_s42', REG.ZS2_TRAIN, K_, m_)
for H in ('LFM', 'RFM', 'MT', 'PL', 'WB'):
    rest(f'rest_lo_{H}_s42', None, holdout=H, n=7)
rest('rest_ef_s42', None, n=8, data_root='v5_effects', splits=('test', 'test_ood'))
rest('rest_ccir_s42', None, n=1, data_root='v3b_ccir', epochs=100)
jobs = R.jobs
names = {j['name'] for j in jobs}
json.dump(dict(profile='fill', machine='A', gpus=1, jobs=jobs), open(os.path.join(HERE, 'plans', 'fill_A.json'), 'w'), ensure_ascii=False, indent=0)
tot = sum(j['cost'] for j in jobs)
print('任务数 %d，总代价 %.1f 单位，tier1=%.1f tier2=%.1f' % (len(jobs), tot, sum(j['cost'] for j in jobs if j['tier'] == 1), sum(j['cost'] for j in jobs if j['tier'] == 2)))
for j in jobs: print(' ', j['tier'], j['name'], j['cost'])
