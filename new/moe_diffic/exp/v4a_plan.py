"""锚定残差融合混合专家（TF 为锚；FD/MS/TD/CYC 以零初始化的静态标量+动态门控累加；求和而非平均；无负载均衡/Gumbel 路由）。起点逐位等价于 TF 单专家。
  v4a：v4_digital（信号型干扰）10% 监督 + remix，种子 42、43，对照 v4t（TF 单专家）与 v4b（无专家）。
  v2a：v2（ST/MT/NB/WB/PL）10% 监督 + remix，种子 42、43，对照 au_remix_p10_s*（无专家）与 tfr_p10_s*（TF+remix）。
运行: bash exp/launch.sh A v4a 4 --slots 4"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
M = '--experts TF FD MS TD CYC --fusion anchor --backbone unet1d --unet_base 64 --cond_mode adaln --phys_aug remix'
D4 = REG.data_args(P, None, 6, 10) + ' --new_data_root ../data/v4_digital'; EP4 = REG.epochs_for(P, 5, 6, 10)
K, Mm = REG.SIZES[10]; D2 = REG.data_args(P, REG.ZS2_TRAIN, K, Mm); EP2 = REG.epochs_for(P, 5, K, Mm)
for sd in (42, 43):
    R.train('v4a_p10_s%d' % sd, 'V4A', 'A', 'moe_diffmic', M, D4, EP4, sd, REG.COST['unet'] * 1.1, tier=1, splits=('test',), eval_extra='--data_root ../data/v4_digital', meta=dict(exp='V4A', cfg='v4_anchor', seed=sd))
    R.train('v2a_p10_s%d' % sd, 'V4A', 'A', 'moe_diffmic', M, D2, EP2, sd, REG.COST['unet'] * 1.1, tier=1, splits=('test', 'test_ood'), meta=dict(exp='V4A', cfg='v2_anchor', seed=sd))
p = dict(profile='v4a', machine='A', gpus=1, jobs=R.jobs)
for j in p['jobs']: j['cmd'] = j['cmd'].replace('--abort_chance_frac 0.5', '--abort_chance_frac 9')
json.dump(p, open(os.path.join(HERE, 'plans', 'v4a_A.json'), 'w'), ensure_ascii=False, indent=0)
print(len(R.jobs), [j['name'] for j in R.jobs])
