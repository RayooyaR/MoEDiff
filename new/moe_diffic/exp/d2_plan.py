"""阶段 1~3：基础对照 + D2（损失）实验。固定 10% 监督训练集(5 类、每格 4 条)、同样的步数/采样预算(25 步 DDIM、平均 8 次)。
  基础对照：U-Net 无专家(3 种子)、J0=U-Net+TF 的第 3 个种子(s44)
  D2：A0=只有 ε 损失；B=多分辨率幅度谱取代原谱项；C=B + 多分辨率复数谱。A(现有损失)=J0。
  tier1：基础对照 + D2 三个变体的种子 42（筛选）；tier2：D2 三个变体的种子 43、44（筛选通过后才有价值，可随时中止）。
M2（TF + Cyclo 专家，静态等权融合，不加 Router）：m2_cyc=TF+CYC；m2_gn=TF+参数更多的普通 CNN 分支(GN，对照)；m2_cyx=TF+离循环对照专家(CYX，特异性对照，tier2)。
  基线 J0=TF-only(同 D2 的 A)。三者均用 --fusion uniform。
生成 exp/plans/d2_A.json；运行: bash exp/launch.sh A d2 8 --slots 4
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import registry as REG
P = REG.profile('real'); R = REG.Reg(P)
K, M = REG.SIZES[10]; z2 = REG.ZS2_TRAIN
D = REG.data_args(P, z2, K, M); EP = REG.epochs_for(P, len(z2), K, M)
J0 = '--experts TF --backbone unet1d --unet_base 64 --cond_mode adaln'
def job(name, extra, tier, cfg, sd):
    R.train(name, 'D2', 'A', 'moe_diffmic', extra, D, EP, sd, REG.COST['unet'], tier=tier, splits=('test', 'test_ood'), meta=dict(exp='D2', cfg=cfg, seed=sd, size=10))
for sd in (42, 43, 44):
    job('un0_p10_s%d' % sd, '--disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln', 1, 'unet_noexpert', sd)
job('j0_p10_s44', J0, 1, 'J0', 44)
for tag, lv in (('a0', 'eps_only'), ('b', 'mrstft_mag'), ('c', 'mrstft_cplx')):
    for sd, tier in ((42, 1), (43, 2), (44, 2)):
        job('d2%s_p10_s%d' % (tag, sd), '%s --loss_variant %s --stft_weight 0.1' % (J0, lv), tier, 'd2' + tag, sd)
for tag, ex, tier in (('cyc', 'TF CYC', 1), ('gn', 'TF GN', 1), ('cyx', 'TF CYX', 2)):
    for sd in (42, 43, 44):
        job('m2_%s_p10_s%d' % (tag, sd), '--experts %s --fusion uniform --backbone unet1d --unet_base 64 --cond_mode adaln' % ex, tier, 'm2_' + tag, sd)
json.dump(dict(profile='d2', machine='A', gpus=1, jobs=R.jobs), open(os.path.join(HERE, 'plans', 'd2_A.json'), 'w'), ensure_ascii=False, indent=0)
print('任务数 %d，总代价 %.1f（tier1=%.1f tier2=%.1f）' % (len(R.jobs), sum(j['cost'] for j in R.jobs), sum(j['cost'] for j in R.jobs if j['tier'] == 1), sum(j['cost'] for j in R.jobs if j['tier'] == 2)))
for j in R.jobs: print(' ', j['tier'], j['name'])
