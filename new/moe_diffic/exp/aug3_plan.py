"""rot 与 rot+remix 的补种子（43、44）；种子 42 在 aug 计划里。对照 un0_p10_s{42,43,44}、au_remix_p10_s{42,43,44}。运行: bash exp/launch.sh A aug3 6 --slots 4"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
K, M = REG.SIZES[10]; D = REG.data_args(P, REG.ZS2_TRAIN, K, M); EP = REG.epochs_for(P, 5, K, M)
BASE = '--disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln'
for tag, ops in (('rotremix', 'rot,remix'), ('rot', 'rot')):
    for sd in (43, 44):
        R.train('au_%s_p10_s%d' % (tag, sd), 'AUG', 'A', 'moe_diffmic', '%s --phys_aug %s' % (BASE, ops), D, EP, sd, REG.COST['unet'], tier=1, splits=('test', 'test_ood'), meta=dict(exp='AUG', cfg='aug_' + tag, seed=sd, size=10))
json.dump(dict(profile='aug3', machine='A', gpus=1, jobs=R.jobs), open(os.path.join(HERE, 'plans', 'aug3_A.json'), 'w'), ensure_ascii=False, indent=0)
print(len(R.jobs), [j['name'] for j in R.jobs])
