"""快速验证（单/双种子）：
  测试 1（信号型干扰场景 v4_digital：CCIR/FSK/OFDM/BURST/ACI，10% 监督≈每格 6 条共 630 条，remix 打开）：专家/oracle 是否有价值
     v4b：无专家 U-Net + remix（基线）；v4t：TF 专家 + remix；v4o：oracle(真实类型+12 维属性) + remix；v4m：局部 oracle(真实干扰时频) + remix。种子 42。
  测试 2（WB 是否拖累其他类型）：去掉 WB 的 4 类训练（ST/MT/NB/PL）+ remix，种子 42、43；对照 au_remix_p10_s{42,43}（5 类）在 ST/MT/NB/PL 上的 BER。
运行: bash exp/launch.sh A fast 4 --slots 6"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
BASE = '--disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln'
J0 = '--experts TF --backbone unet1d --unet_base 64 --cond_mode adaln'
# 测试 1：v4_digital，每格 6 条(10%)，虚拟放大 10 倍 -> 约 21000 步
D4 = REG.data_args(P, None, 6, 10) + ' --new_data_root ../data/v4_digital'
EP4 = REG.epochs_for(P, 5, 6, 10)
def job4(name, extra, cfg):
    R.train(name, 'FAST', 'A', 'moe_diffmic', extra + ' --phys_aug remix', D4, EP4, 42, REG.COST['unet'], tier=1, splits=('test',), eval_extra='--data_root ../data/v4_digital', meta=dict(exp='FAST', cfg=cfg, seed=42))
job4('v4b_p10_s42', BASE, 'v4_base_remix')
job4('v4t_p10_s42', J0, 'v4_tf_remix')
job4('v4o_p10_s42', '--disable_moe --oracle_cond type_attrs --backbone unet1d --unet_base 64 --cond_mode adaln', 'v4_oracle_remix')
job4('v4m_p10_s42', BASE + ' --extra_local oracle_mask', 'v4_oraclemask_remix')
# 测试 2：去掉 WB
types4 = ['ST', 'MT', 'NB', 'PL']
D2 = REG.data_args(P, types4, 4, 10); EP2 = REG.epochs_for(P, 4, 4, 10)
for sd in (42, 43):
    R.train('nw_p10_s%d' % sd, 'FAST', 'A', 'moe_diffmic', BASE + ' --phys_aug remix', D2, EP2, sd, REG.COST['unet'], tier=1, splits=('test',), meta=dict(exp='FAST', cfg='no_wb_remix', seed=sd))
json.dump(dict(profile='fast', machine='A', gpus=1, jobs=R.jobs), open(os.path.join(HERE, 'plans', 'fast_A.json'), 'w'), ensure_ascii=False, indent=0)
print(len(R.jobs), [(j['name']) for j in R.jobs], 'v4 epochs', EP4, 'noWB epochs', EP2)
