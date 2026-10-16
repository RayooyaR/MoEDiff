"""彻底定位专家无效的原因 + 让专家起作用的尝试 + remix 是否依赖模拟器内部信息（均在 remix 下，U-Net 无专家为底，10% 监督，先 2 个种子）。
  om_p10_s*   ：remix + 局部干扰定位 oracle（真实干扰的 16 子带时频功率作为额外输入通道；信息上界，不可部署）
  e1_p10_s*   ：remix + E1 干扰定位专家（由 y 预测同样的通道，训练时用 y−s 派生的真实干扰特征做稠密辅助监督；推理只用 y）
  ry_p10_s*   ：remixys（仅依赖配对样本 (y,s)：不使用已分离的干扰/噪声和样本真实滚降系数）
  tfk_p10_s42 ：TF 专家 + remix，保留检查点，用于"反事实条件"诊断（把 TF 条件置零/错位后 BER 变化）
对照：au_remix_p10_s{42,43}（无专家 + remix）。运行: bash exp/launch.sh A loc 6 --slots 7"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
K, M = REG.SIZES[10]; D = REG.data_args(P, REG.ZS2_TRAIN, K, M); EP = REG.epochs_for(P, 5, K, M)
BASE = '--disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln'
J0 = '--experts TF --backbone unet1d --unet_base 64 --cond_mode adaln'
def job(name, extra, cfg, sd, keep=False):
    R.train(name, 'LOC', 'A', 'moe_diffmic', extra, D, EP, sd, REG.COST['unet'], tier=1, splits=('test', 'test_ood'), keep=keep, meta=dict(exp='LOC', cfg=cfg, seed=sd, size=10))
for sd in (42, 43):
    job('om_p10_s%d' % sd, '%s --phys_aug remix --extra_local oracle_mask' % BASE, 'oracle_mask_remix', sd)
    job('e1_p10_s%d' % sd, '%s --phys_aug remix --extra_local e1 --e1_weight 0.1' % BASE, 'e1_remix', sd)
    job('ry_p10_s%d' % sd, '%s --phys_aug remixys' % BASE, 'remixys', sd)
job('tfk_p10_s42', '%s --phys_aug remix' % J0, 'tf_remix_keep', 42, keep=True)
json.dump(dict(profile='loc', machine='A', gpus=1, jobs=R.jobs), open(os.path.join(HERE, 'plans', 'loc_A.json'), 'w'), ensure_ascii=False, indent=0)
print(len(R.jobs), [j['name'] for j in R.jobs])
