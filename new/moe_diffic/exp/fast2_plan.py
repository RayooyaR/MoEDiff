"""v4_digital 补种子 + TF 专家与 oracle 的组合：
  v4b/v4t/v4o 补种子 43、44（种子 42 在 fast 计划里）；v4to = TF 专家 + oracle(真实类型+12 维属性) 的组合（token 并列，经 AdaLN 池化），种子 42、43、44。
运行: bash exp/launch.sh A fast2 4 --slots 9"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
BASE = '--disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln'
J0 = '--experts TF --backbone unet1d --unet_base 64 --cond_mode adaln'
OR = '--disable_moe --oracle_cond type_attrs --backbone unet1d --unet_base 64 --cond_mode adaln'
TO = '--experts TF --oracle_cond type_attrs --backbone unet1d --unet_base 64 --cond_mode adaln'
D4 = REG.data_args(P, None, 6, 10) + ' --new_data_root ../data/v4_digital'; EP4 = REG.epochs_for(P, 5, 6, 10)
def job(name, extra, sd):
    R.train(name, 'FAST', 'A', 'moe_diffmic', extra + ' --phys_aug remix', D4, EP4, sd, REG.COST['unet'], tier=1, splits=('test',), eval_extra='--data_root ../data/v4_digital', meta=dict(exp='FAST', cfg=name, seed=sd))
for sd in (43, 44):
    job('v4b_p10_s%d' % sd, BASE, sd); job('v4t_p10_s%d' % sd, J0, sd); job('v4o_p10_s%d' % sd, OR, sd)
for sd in (42, 43, 44):
    job('v4to_p10_s%d' % sd, TO, sd)
p = dict(profile='fast2', machine='A', gpus=1, jobs=R.jobs)
for j in p['jobs']: j['cmd'] = j['cmd'].replace('--abort_chance_frac 0.5', '--abort_chance_frac 9')
json.dump(p, open(os.path.join(HERE, 'plans', 'fast2_A.json'), 'w'), ensure_ascii=False, indent=0)
print(len(R.jobs), [j['name'] for j in R.jobs])
