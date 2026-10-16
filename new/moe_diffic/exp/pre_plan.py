"""干净信号无条件预训练 + 10% 配对监督微调（P1/P0）。尚未启动，需确认后运行: bash exp/launch.sh A pre 8 --slots 5
  P1：1 万条额外独立干净波形(gen/make_clean_pool.py)预训练 4 万步 -> J0(TF+U-Net) 全参数微调；3 个微调种子共用同一个预训练权重。
  P0（对照）：只用该种子 10% 训练对里的 420 条干净信号做同样 4 万步预训练 -> 同样微调。分开“额外干净数据”与“额外预训练计算/初始化”。
  B0 = 现有 J0（j0_p10_s42/43/44，未预训练）。微调与 J0 完全相同(同样本、同步数、同 batch、同采样评估)，仅多 --init_from。
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
K, M = REG.SIZES[10]; z2 = REG.ZS2_TRAIN
D = REG.data_args(P, z2, K, M); EP = REG.epochs_for(P, len(z2), K, M)
J0 = '--experts TF --backbone unet1d --unet_base 64 --cond_mode adaln'
PRE = '--model_type moe_diffmic --disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln --amp bf16 --lr 3e-4 --batch_size 32 --new_types ' + ' '.join(z2)
POOL = '../data/clean_pool/pool_10k.npz'
R.add('make_pool_10k', 'PRE', 'A', '$PY ../gen/make_clean_pool.py --n 10000 --out %s --check' % POOL, 0.1, tier=1, meta=dict(kind='script', exp='PRE'))
R.add('pre_c10k', 'PRE', 'A', '$PY pretrain_clean.py --exp_name pre_c10k --pretrain_pool %s --pretrain_steps 40000 --seed 42 %s' % (POOL, PRE), 1.6, tier=1, deps=['make_pool_10k'], meta=dict(kind='train', exp='PRE', cfg='pre_c10k'))
for sd in (42, 43, 44):
    R.train('p1_p10_s%d' % sd, 'PRE', 'A', 'moe_diffmic', '%s --init_from pre_c10k' % J0, D, EP, sd, 0.8, tier=1, splits=('test', 'test_ood'), deps=['pre_c10k'], meta=dict(exp='PRE', cfg='P1', seed=sd))
for sd in (42, 43, 44):
    R.add('pre_t420_s%d' % sd, 'PRE', 'A', '$PY pretrain_clean.py --exp_name pre_t420_s%d --pretrain_pool train420 --pretrain_steps 40000 --train_per_cell %d --seed %d %s' % (sd, K, sd, PRE), 1.6, tier=2,
          meta=dict(kind='train', exp='PRE', cfg='pre_t420', seed=sd))
    R.train('p0_p10_s%d' % sd, 'PRE', 'A', 'moe_diffmic', '%s --init_from pre_t420_s%d' % (J0, sd), D, EP, sd, 0.8, tier=2, splits=('test', 'test_ood'), deps=['pre_t420_s%d' % sd], meta=dict(exp='PRE', cfg='P0', seed=sd))
json.dump(dict(profile='pre', machine='A', gpus=1, jobs=R.jobs), open(os.path.join(HERE, 'plans', 'pre_A.json'), 'w'), ensure_ascii=False, indent=0)
print('任务数 %d，总代价 %.1f（tier1=%.1f，tier2=%.1f）' % (len(R.jobs), sum(j['cost'] for j in R.jobs), sum(j['cost'] for j in R.jobs if j['tier'] == 1), sum(j['cost'] for j in R.jobs if j['tier'] == 2)))
