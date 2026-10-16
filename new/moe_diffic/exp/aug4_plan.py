"""remix 打开后再测专家（区分"专家无效"是样本量不足还是结构/信息问题）：先 2 个种子（42、43），效果好再补 44。
  tfr_p10_s*：J0（TF 专家 + U-Net）+ remix；对照 au_remix_p10_s*（无专家 + remix）。
  orr_p10_s*：oracle(真实干扰类型+12 维属性；remix 后按新干扰重算) + remix；对照同上。
运行: bash exp/launch.sh A aug4 6 --slots 4"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
K, M = REG.SIZES[10]; D = REG.data_args(P, REG.ZS2_TRAIN, K, M); EP = REG.epochs_for(P, 5, K, M)
J0 = '--experts TF --backbone unet1d --unet_base 64 --cond_mode adaln'
OR = '--disable_moe --oracle_cond type_attrs --backbone unet1d --unet_base 64 --cond_mode adaln'
for sd in (42, 43):
    R.train('tfr_p10_s%d' % sd, 'AUG', 'A', 'moe_diffmic', '%s --phys_aug remix' % J0, D, EP, sd, REG.COST['unet'], tier=1, splits=('test', 'test_ood'), meta=dict(exp='AUG', cfg='tf_remix', seed=sd, size=10))
    R.train('orr_p10_s%d' % sd, 'AUG', 'A', 'moe_diffmic', '%s --phys_aug remix' % OR, D, EP, sd, REG.COST['unet'], tier=1, splits=('test', 'test_ood'), meta=dict(exp='AUG', cfg='oracle_remix', seed=sd, size=10))
json.dump(dict(profile='aug4', machine='A', gpus=1, jobs=R.jobs), open(os.path.join(HERE, 'plans', 'aug4_A.json'), 'w'), ensure_ascii=False, indent=0)
print(len(R.jobs), [j['name'] for j in R.jobs])
