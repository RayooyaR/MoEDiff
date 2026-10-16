"""10% 样本监督场景：补齐三个缺数据的基线（Restormer / DiffCom / ICDM），每个 3 个种子，全部并行。
生成 exp/plans/m10_A.json；运行: bash exp/launch.sh A m10 4 --slots 5
设定与表里其它行完全一致：训练 5 类(ST/MT/NB/WB/PL)、每格 4 条(×10 虚拟放大，步数对齐)、bf16；评估 test + test_ood。
DiffCom：官方式热启动(训练集平均 SINR→起始步)；ζ 与 ICDM 的 γ 沿用既有协议在验证集(仅训练见过的类型)上从网格中选，最终测试做平均 8 次采样。
  • DiffCom 的 DPS 引导需要梯度，评估用 fp32（不用 bf16，保证引导精度）；ICDM 无梯度，评估用 bf16（与其它扩散模型一致）。
  • 这两个基线训练时不启用"训练过半仍是机会水平则中止"的早停（先验类模型训练中途的无引导/粗引导 BER 本来就接近机会水平）。
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
K, M = REG.SIZES[10]
z2 = REG.ZS2_TRAIN
D = REG.data_args(P, z2, K, M)
EP = REG.epochs_for(P, len(z2), K, M)
SP = 'test test_ood'
for sd in (42, 43, 44):
    # Restormer（回归，损失/优化器与 CNN 完全一致）
    R.train('rest_p10_s%d' % sd, 'M10', 'A', 'restormer', '--restormer_dim 48', D, REG.epochs_for(P, len(z2), K, M, True), sd, 0.5, tier=1,
            splits=('test', 'test_ood'), diffusion=False, meta=dict(exp='M10', cfg='rest_p10', seed=sd))
    # DiffCom（热启动）
    nm = 'fs_p10_diffcom_s%d' % sd
    common = REG.common(P).replace('--abort_chance_frac %s' % P['abort'], '--abort_chance_frac 9')
    cmd = ('%s exp_train.py --exp_name %s --model_type diffcom_1d %s %s --epochs %d --seed %d --dps_step_size 30 --dps_warm_start'
           % (REG.PYV, nm, common, D, EP, sd)
           + ' && %s tune_and_eval.py --run %s --kind diffcom --warm_start --val_stride 2 --final_avg8 --amp none --splits %s' % (REG.PYV, nm, SP)
           + ' && rm -f runs/%s/best_model.pth' % nm)
    R.add(nm, 'M10', 'A', cmd, 1.0, tier=1, meta=dict(kind='train', model='diffcom_1d', seed=sd, exp='M10', cfg='diffcom_p10'))
    # ICDM（双先验）
    nm = 'fs_p10_icdm_s%d' % sd
    cmd = ('%s exp_train.py --exp_name %s --model_type icdm_1d %s %s --epochs %d --seed %d --icdm_dual_gamma 0.75'
           % (REG.PYV, nm, common, D, EP, sd)
           + ' && %s tune_and_eval.py --run %s --kind icdm --val_stride 2 --final_avg8 --amp bf16 --splits %s' % (REG.PYV, nm, SP)
           + ' && rm -f runs/%s/best_model.pth' % nm)
    R.add(nm, 'M10', 'A', cmd, 1.3, tier=1, meta=dict(kind='train', model='icdm_1d', seed=sd, exp='M10', cfg='icdm_p10'))
jobs = R.jobs
json.dump(dict(profile='m10', machine='A', gpus=1, jobs=jobs), open(os.path.join(HERE, 'plans', 'm10_A.json'), 'w'), ensure_ascii=False, indent=0)
print('任务数 %d，总代价 %.1f' % (len(jobs), sum(j['cost'] for j in jobs)))
for j in jobs: print(' ', j['name'], j['cost'])
print(jobs[1]['cmd'])
