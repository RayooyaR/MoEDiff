"""v4_digital（信号型干扰，10% 监督=每格 6 条）：
  tier1（先做）remix 的 v4 验证：v4n=无专家无增强；v4ry=无专家 + remixys（仅用 (y,s) 的可部署版本）；对照已有 v4b（无专家 + remix）。种子 42、43。
  tier2 真正的"混合专家"：v4tc=TF+CYC（时频 + 循环平稳，静态等权）；v4q4u=FD+MS+TF+TD 四专家静态等权；v4q4r=四专家 + 软路由。全部 remix，种子 42、43。对照 v4t（TF 单专家）。
运行: bash exp/launch.sh A v4m 4 --slots 4  （tier1 先占满 4 个槽，tier2 随后自动补上）"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
U = '--backbone unet1d --unet_base 64 --cond_mode adaln'
D4 = REG.data_args(P, None, 6, 10) + ' --new_data_root ../data/v4_digital'; EP4 = REG.epochs_for(P, 5, 6, 10)
def job(name, extra, sd, tier):
    R.train(name, 'V4M', 'A', 'moe_diffmic', extra, D4, EP4, sd, REG.COST['unet'], tier=tier, splits=('test',), eval_extra='--data_root ../data/v4_digital', meta=dict(exp='V4M', cfg=name, seed=sd))
for sd in (42, 43):
    job('v4n_p10_s%d' % sd, '--disable_moe --true_uncond %s' % U, sd, 1)
    job('v4ry_p10_s%d' % sd, '--disable_moe --true_uncond %s --phys_aug remixys' % U, sd, 1)
for sd in (42, 43):
    job('v4tc_p10_s%d' % sd, '--experts TF CYC --fusion uniform %s --phys_aug remix' % U, sd, 2)
    job('v4q4u_p10_s%d' % sd, '--experts FD MS TF TD --fusion uniform %s --phys_aug remix' % U, sd, 2)
    job('v4q4r_p10_s%d' % sd, '--experts FD MS TF TD %s --phys_aug remix' % U, sd, 2)
p = dict(profile='v4m', machine='A', gpus=1, jobs=R.jobs)
for j in p['jobs']: j['cmd'] = j['cmd'].replace('--abort_chance_frac 0.5', '--abort_chance_frac 9')
json.dump(p, open(os.path.join(HERE, 'plans', 'v4m_A.json'), 'w'), ensure_ascii=False, indent=0)
print(len(R.jobs), [(j['name'], j['tier']) for j in R.jobs])
