"""remix 增强的 3 种子复核（种子 43、44；种子 42 在 aug 计划里）。对照 un0_p10_s{42,43,44}。运行: bash exp/launch.sh A aug2 6 --slots 2"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
K, M = REG.SIZES[10]; D = REG.data_args(P, REG.ZS2_TRAIN, K, M); EP = REG.epochs_for(P, 5, K, M)
BASE = '--disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln'
for sd in (43, 44):
    R.train('au_remix_p10_s%d' % sd, 'AUG', 'A', 'moe_diffmic', '%s --phys_aug remix' % BASE, D, EP, sd, REG.COST['unet'], tier=1, splits=('test', 'test_ood'), meta=dict(exp='AUG', cfg='aug_remix', seed=sd, size=10))
json.dump(dict(profile='aug2', machine='A', gpus=1, jobs=R.jobs), open(os.path.join(HERE, 'plans', 'aug2_A.json'), 'w'), ensure_ascii=False, indent=0)
print(len(R.jobs))
