"""Oracle 条件诊断 + U-Net 扩散的物理数据增强筛选（10% 监督训练集）。
  oracle_p10_s{42,43,44}：无专家 U-Net + 真实干扰类型 one-hot + 12 维干扰物理属性经 MLP 作为条件(AdaLN)——信息上界诊断，不可部署；
      对照：un0_p10_s{42,43,44}（同结构无条件）、j0（TF 专家）。只在见过类型上解读（oracle 的类型 one-hot 对未见类型无意义）。
  au_<op>_p10_s42：无专家 U-Net 基线(un0) + 单个物理增强算子（rot/shift/awgn/iqgain/remix）及 rot+remix 组合，筛选用单种子 42，对照 un0_p10_s42。
生成 exp/plans/aug_A.json；运行: bash exp/launch.sh A aug 6 --slots 5
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
K, M = REG.SIZES[10]; z2 = REG.ZS2_TRAIN
D = REG.data_args(P, z2, K, M); EP = REG.epochs_for(P, len(z2), K, M)
BASE = '--disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln'
def job(name, extra, cfg, sd):
    R.train(name, 'AUG', 'A', 'moe_diffmic', extra, D, EP, sd, REG.COST['unet'], tier=1, splits=('test', 'test_ood'), meta=dict(exp='AUG', cfg=cfg, seed=sd, size=10))
for sd in (42, 43, 44):
    job('oracle_p10_s%d' % sd, '--disable_moe --oracle_cond type_attrs --backbone unet1d --unet_base 64 --cond_mode adaln', 'oracle_type_attrs', sd)
for tag, ops in (('rot', 'rot'), ('shift', 'shift'), ('awgn', 'awgn'), ('iqgain', 'iqgain'), ('remix', 'remix'), ('rotremix', 'rot,remix')):
    job('au_%s_p10_s42' % tag, '%s --phys_aug %s' % (BASE, ops), 'aug_' + tag, 42)
json.dump(dict(profile='aug', machine='A', gpus=1, jobs=R.jobs), open(os.path.join(HERE, 'plans', 'aug_A.json'), 'w'), ensure_ascii=False, indent=0)
print('任务数 %d，总代价 %.1f' % (len(R.jobs), sum(j['cost'] for j in R.jobs)))
for j in R.jobs: print(' ', j['name'])
