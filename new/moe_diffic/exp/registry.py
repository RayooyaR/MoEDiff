"""plan.md 全部实验的任务注册表：每个任务 = 一条完整的 bash 命令链（训练 -> 评估 -> 清理权重）+ 元数据。

分工原则（三台互不相通的服务器 A=1 卡 / B=3 卡 / C=2 卡）：有依赖的任务放同一台，独立任务分散，**零跨机同步**。
  B  依赖链：E1-E5（MoE 设计）、E7-E9（主干/局部TF/条件注入）、E10-E11（残差扩散与门控）、依赖胜出者的 E15/E16 行、胜出者的 LOTO 复核
  C  独立任务：逐类留出(E6)类型 MT、PL；E14 增强；E15 的基线行（CNN / 无专家 DiT / TF-only）
  A  独立任务：逐类留出类型 LFM；P0 审计；E13 采样消融；E12 物理一致性（可选）
  三台都跑“锚点任务”(TF-only seed42 / CNN seed42)，用来量化同一配置在不同机器上的差异（见 final_report.py）。
任务之间的依赖通过 deps 表达；“选择”任务(kind=select)按验证集 BER 从候选里选胜出者，写入 winners.json，
后续任务命令里的 {CAND:键} 在启动前被替换（缺失时用 DEFAULTS，所以缺少上游结果也不会卡住）。
命令里的 $PY 由 runner 注入（环境里的 python）。profile=smoke 用极小设置验证全流程。
"""
import math

MACHINE_GPUS = {'A': 1, 'B': 3, 'C': 2}
PYV = '$PY'
TYPES8 = ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM']
ZS2_TRAIN = ['ST', 'MT', 'NB', 'WB', 'PL']                 # 训练类型；未见类型 = LFM NFM RFM（调频类）
ZS2_UNSEEN = ['LFM', 'NFM', 'RFM']
LOTO_TYPES = {'A': ['LFM', 'RFM'], 'C': ['MT', 'PL', 'WB']}             # 逐类留出的类型分配（B 在最后阶段对胜出者三个类型全做）
DEFAULTS = {          # 缺少上游选择结果时使用的预先登记的默认值（plan.md 的完整设计）
    'bb_top1': '', 'bb_top2': '', 'bb_top1_is_dit': True, 'tok_top1': '', 'cond_top1': '',
    'backbone_final': '', 'tok_final': '', 'cond_final': '', 'final_base': '',
    'moe_top1': '--experts TF', 'moe_top2': '--experts FD MS TF TD', 'moe_compose': '--router_type dual --attr_mode joint --shared_expert',
    'moe_final': '--router_type dual --attr_mode joint --shared_expert', 'gate_variant': 'both', 'aug_top1': 'rot', 'aug_top2': 'remix',
    'aug_combo1': 'rot,remix', 'aug_combo2': 'remix,awgn', 'gate_variant_base': 'both', 'gate_local_flag': '', 'bb_top2_is_dit': True,
    'compose_dup': False, 'top1_dup': False, 'top2_dup': False, 'tok_default': True, 'cond_default': True, 'e12_w_obs': '0.3', 'e12_w_band': '0',
}

# ---- 配置 id -> 参数（相对默认：四专家、legacy 路由、xattn、DiT）
CFG = {
    'e1_fd': '--experts FD', 'e1_tf': '--experts TF', 'e1_td': '--experts TD', 'e1_ms': '--experts MS', 'e1_four': '',
    'e1_uniform4': '--fusion uniform', 'e1_generic4': '--experts GN GN2 GN3 GN4',
    'e2_shared4': '--shared_expert', 'e2_sharedTF': '--experts TF --shared_expert', 'e2_generic2': '--experts GN GN2',
    'e3_joint4': '--attr_mode joint', 'e3_prefrz4': '--attr_mode pretrain_freeze', 'e3_preft4': '--attr_mode pretrain_ft', 'e3_jointTF': '--experts TF --attr_mode joint',
    'e4_dual': '--router_type dual --dual_variant dual', 'e4_noint': '--router_type dual --dual_variant dual_noint', 'e4_morph': '--router_type dual --dual_variant morph',
    'e4_strength': '--router_type dual --dual_variant strength', 'e4_oracle': '--router_type dual --dual_variant oracle_sir', 'e4_prior': '--router_type prior',
    'e5_sum': '--fusion sum_single', 'e5_topk': '--router_type dual --dual_variant dual --fusion topk_mask',
    'e7_dit0': '--disable_moe --true_uncond', 'e7_u0': '--disable_moe --true_uncond --backbone unet1d --unet_base 48',
    'e7_uTF': '--experts TF --backbone unet1d --unet_base 64', 'e7_uTF48': '--experts TF --backbone unet1d --unet_base 48',
    'e7_dwTF': '--experts TF --backbone diffwave --diffwave_res 128', 'e7_dwTF192': '--experts TF --backbone diffwave --diffwave_res 192',
    'e8_tok16': '--experts TF --tf_tokens 16', 'e8_tok32': '--experts TF --tf_tokens 32',
    'e9_film': '--experts TF --cond_mode film', 'e9_adaln': '--experts TF --cond_mode adaln',
}
BACKBONE_ARGS = {'e1_tf': '', 'e7_uTF': '--backbone unet1d --unet_base 64', 'e7_uTF48': '--backbone unet1d --unet_base 48',
                 'e7_dwTF': '--backbone diffwave --diffwave_res 128', 'e7_dwTF192': '--backbone diffwave --diffwave_res 192'}
TOK_ARGS = {'e1_tf': '', 'e8_tok16': '--tf_tokens 16', 'e8_tok32': '--tf_tokens 32'}
COND_ARGS = {'e1_tf': '', 'e9_film': '--cond_mode film', 'e9_adaln': '--cond_mode adaln'}
MOE_POOL = ['e1_tf', 'e1_four', 'e2_shared4', 'e2_sharedTF', 'e3_joint4', 'e3_prefrz4', 'e3_preft4', 'e3_jointTF',
            'e4_dual', 'e4_noint', 'e4_morph', 'e4_strength', 'e4_prior', 'e5_sum', 'e5_topk']
GATE_VARIANTS = {'sir': ('sir', False), 'conf': ('conf', False), 'both': ('both', False), 'oracle': ('oracle', False), 'both_local': ('both', True)}
AUG_SINGLES = ['rot', 'shift', 'awgn', 'iqgain', 'remix']
LOTO_CFGS = {
    'cnn': ('cnn', ''), 'dit0': ('moe', '--disable_moe --true_uncond'), 'tf': ('moe', '--experts TF'), 'static4': ('moe', '--fusion uniform'),
    'legacy4': ('moe', ''), 'dual': ('moe', '--router_type dual'), 'dual_attr': ('moe', '--router_type dual --attr_mode joint'),
    'dual_shared': ('moe', '--router_type dual --shared_expert'),
}
LOTO_SEED2 = ['cnn', 'tf', 'legacy4', 'dual_attr']
SIZES = {10: (4, 10), 30: (12, 3), 100: (40, 1)}           # 训练数据比例% -> (每格样本数 K, 虚拟放大倍数 mult)，使优化步数大致相同

# ---- 代价（以“旧机 fp32 小时”为单位，仅用于排序与负载统计）
COST = {'moe': 0.75, 'dit': 0.5, 'unet': 0.8, 'dw': 0.7, 'cnn': 0.08, 'res': 0.9, 'rescnn': 0.15, 'oof': 1.2, 'gate': 0.15, 'eval': 0.1, 'script': 0.3}


def profile(name):
    if name in ('real', 'fill', 'm10', 'd2', 'aug', 'aug2', 'aug3', 'aug4', 'loc', 'fast', 'fast2', 'v4m', 'v4a', 'pre'):
        return dict(name='real', smoke=False, ddim=25, val_every=5, warm=5, abort=0.5, stride=2, avg_k=8, base_ep=60, folds=5, gate_steps=600,
                    unit_seconds=2800, sa_stride=8, sa_steps='10 25 50 100', sa_seeds=2, oof_extra='')
    if name == 'smoke':
        return dict(name='smoke', smoke=True, ddim=3, val_every=1, warm=0, abort=9, stride=60, avg_k=2, base_ep=2, folds=2, gate_steps=20,
                    unit_seconds=8, sa_stride=80, sa_steps='3 5', sa_seeds=2, oof_extra='--ddim_steps 3')
    raise ValueError(name)


def epochs_for(P, ntypes, K, mult, reg=False):
    if P['smoke']:
        return 2
    steps_per_epoch = math.ceil(ntypes * 21 * K * mult / 32)
    return int(round((12000 if reg else 21000) / steps_per_epoch))


def common(P, reg=False):
    return ('--lr 3e-4 --batch_size 32 --num_workers 0 --ddim_steps %d --eval_freq 5 --val_every %d --warmup_epochs %d --amp bf16 --val_stride 2 '
            '--abort_chance_frac %s --skip_final_eval' % (P['ddim'], P['val_every'], P['warm'], P['abort']))


def data_args(P, types=None, K=40, mult=1, holdout=None):
    a = []
    if types:
        a.append('--new_types ' + ' '.join(types))
    if holdout:
        a.append('--holdout ' + holdout)
    if P['smoke']:
        a.append('--train_per_cell 1 --val_per_cell 1')
    elif K is not None and K < 40:
        a.append('--train_per_cell %d' % K)
    if mult > 1 and not P['smoke']:
        a.append('--aug_multiplier %d' % mult)
    return ' '.join(a)


def ev(P, name, splits=('test',), diffusion=True, tag='_avg8', extra=''):
    return '%s evaluate_new.py --runs %s --splits %s --stride %d --avg_k %d --tag %s --amp bf16 %s' % (
        PYV, name, ' '.join(splits), P['stride'], P['avg_k'] if diffusion else 1, tag, extra)


class Reg:
    def __init__(self, P):
        self.P, self.jobs, self.names = P, [], set()

    def add(self, name, group, machine, cmd, cost, tier=1, deps=(), skip_if=None, meta=None, mode='all_done'):
        assert name not in self.names, 'duplicate job name %s' % name
        self.names.add(name)
        self.jobs.append(dict(name=name, group=group, machine=machine, cmd=cmd, cost=round(cost, 3), tier=tier, deps=list(deps), skip_if=skip_if,
                              meta=meta or {}, deps_mode=mode))
        return name

    # ---- 训练 + 评估 + 清理 的标准链
    def train(self, name, group, machine, model_type, extra, data, epochs, seed, cost, tier=1, deps=(), splits=('test',), keep=False, skip_if=None,
              meta=None, eval_extra='', diffusion=None, gate=None):
        P = self.P
        diffusion = model_type in ('moe_diffmic', 'res_diff') if diffusion is None else diffusion
        cmd = '%s exp_train.py --exp_name %s --model_type %s %s %s --epochs %d --seed %d %s' % (
            PYV, name, model_type, common(P), data, epochs, seed, extra)
        cmd += ' && ' + ev(P, name, splits, diffusion, extra=eval_extra)
        if not keep:
            cmd += ' && rm -f runs/%s/best_model.pth' % name
        m = dict(kind='train', model=model_type, seed=seed)
        m.update(meta or {})
        return self.add(name, group, machine, cmd, cost, tier, deps, skip_if, m)


def build(profile_name='real'):
    P = profile(profile_name)
    R = Reg(P)
    z2 = ZS2_TRAIN
    K0, EP = 40, epochs_for(P, len(z2), 40, 1)
    D2 = data_args(P, z2)

    def mcost(cfg_args, backbone_hint=''):
        if 'unet1d' in cfg_args or 'unet1d' in backbone_hint:
            return COST['unet']
        if 'diffwave' in cfg_args:
            return COST['dw']
        if '--disable_moe' in cfg_args:
            return COST['dit']
        return COST['moe']

    # ======================================================== 每台机器都要的：环境测试 + 锚点
    for M in 'ABC':
        R.add('sys_tests_%s' % M, 'SYS', M, 'mkdir -p exp_state && (cd tests && %s run_all.py --fast) > exp_state/sys_tests_%s.log 2>&1; true' % (PYV, M), 0.2, tier=0, meta=dict(kind='script'))
        R.train('anc_tf_%s' % M, 'ANCHOR', M, 'moe_diffmic', '--experts TF', D2, EP, 42, COST['moe'], tier=0, deps=[], meta=dict(cfg='anchor_tf', exp='ANCHOR', machine=M))
        R.train('anc_cnn_%s' % M, 'ANCHOR', M, 'cnn', '--cnn_base_ch 32', D2, epochs_for(P, len(z2), 40, 1, True) if not P['smoke'] else 2, 42, COST['cnn'], tier=0,
                deps=[], meta=dict(cfg='anchor_cnn', exp='ANCHOR', machine=M), diffusion=False)

    # ======================================================== A：P0 审计、E13、E12、LOTO(LFM)
    R.add('p0_audit', 'P0', 'A', '%s audit_baseline.py --repro zs2_ex_TF_s44' % PYV if not P['smoke'] else '%s audit_baseline.py --skip_repro' % PYV, COST['script'] * 3, tier=1,
          deps=[], meta=dict(kind='script', exp='P0'))
    R.add('e13_sampling', 'E13', 'A', '%s sampling_ablation.py --runs zs2_ex_TF_s44 zs2_dit --steps %s --avg 1 8 --seeds %d --stride %d --out ../reports/e13_sampling_ablation.csv'
          % (PYV, P['sa_steps'], P['sa_seeds'], P['sa_stride']), 2.5, tier=1, deps=[], meta=dict(kind='script', exp='E13'))
    R.add('model_costs', 'E7', 'A', '%s model_costs.py --out ../reports/model_costs.csv' % PYV, COST['script'], tier=1, deps=[], meta=dict(kind='script', exp='E7'))
    # E12 可选（tier 2）：干扰估计器 + 引导
    R.train('e12_est', 'E12', 'A', 'cnn', '--target interf --cnn_base_ch 32', D2, 100 if not P['smoke'] else 2, 42, COST['cnn'], tier=2, keep=True, deps=[], diffusion=False,
            meta=dict(cfg='e12_est', exp='E12'))
    R.add('e12_guid_val', 'E12', 'A', '%s eval_guidance.py --run zs2_ex_TF_s44 --est e12_est --w_obs 0 0.3 1 3 --w_band 0 1 --split val --stride 2 --ddim_steps %d --out ../reports/e12_val.json'
          % (PYV, P['ddim']), 1.0, tier=2, deps=['e12_est'], meta=dict(kind='script', exp='E12'))
    R.add('e12_select', 'E12', 'A', '%s exp/selection.py e12' % PYV, 0.01, tier=2, deps=['e12_guid_val'], meta=dict(kind='select'))
    R.add('e12_guid_test', 'E12', 'A', '%s eval_guidance.py --run zs2_ex_TF_s44 --est e12_est --w_obs {CAND:e12_w_obs} --w_band {CAND:e12_w_band} --split test --stride 2 --ddim_steps %d --out ../reports/e12_test.json'
          % (PYV, P['ddim']), 1.0, tier=2, deps=['e12_select'], meta=dict(kind='script', exp='E12'))

    # ======================================================== LOTO（E6）：A 做 LFM，C 做 MT/PL；B 在最后做胜出者
    def loto(M, H, tier):
        for cfg, (mt, extra) in LOTO_CFGS.items():
            seeds = [42] + ([43, 44] if cfg in LOTO_SEED2 else [])
            for sd in seeds:
                name = 'lo_%s_%s_s%d' % (H, cfg, sd)
                keep = sd == 42 and cfg in ('cnn', 'tf', 'dual_attr')
                if mt == 'cnn':
                    R.train(name, 'E6', M, 'cnn', '--cnn_base_ch 32', data_args(P, None, holdout=H), epochs_for(P, 7, 40, 1, True) if not P['smoke'] else 2, sd, COST['cnn'], tier=tier,
                            keep=keep, diffusion=False, deps=[], meta=dict(exp='E6', cfg=cfg, holdout=H, seed=sd))
                else:
                    R.train(name, 'E6', M, 'moe_diffmic', extra, data_args(P, None, holdout=H), epochs_for(P, 7, 40, 1), sd, mcost(extra), tier=tier, keep=keep,
                            deps=[], meta=dict(exp='E6', cfg=cfg, holdout=H, seed=sd))
        for cfg in ('tf', 'dual_attr'):
            R.add('lo_%s_adapt_%s' % (H, cfg), 'E6', M, '%s loto_adapt.py --run lo_%s_%s_s42 --holdout %s --cnn_run lo_%s_cnn_s42 --out runs/lo_%s_%s_s42/loto_adapt.json' % (PYV, H, cfg, H, H, H, cfg),
                  COST['script'], tier=tier, deps=['lo_%s_%s_s42' % (H, cfg), 'lo_%s_cnn_s42' % H], meta=dict(kind='script', exp='E6', holdout=H))
            R.add('lo_%s_clean_%s' % (H, cfg), 'E6', M, 'rm -f runs/lo_%s_%s_s42/best_model.pth' % (H, cfg), 0.001, tier=tier, deps=['lo_%s_adapt_%s' % (H, cfg)], meta=dict(kind='script'))
        R.add('lo_%s_clean_cnn' % H, 'E6', M, 'rm -f runs/lo_%s_cnn_s42/best_model.pth' % H, 0.001, tier=tier, deps=['lo_%s_adapt_tf' % H, 'lo_%s_adapt_dual_attr' % H], meta=dict(kind='script'))

    for H in LOTO_TYPES['A']:
        loto('A', H, 1)
    for H in LOTO_TYPES['C']:
        loto('C', H, 1)

    # ======================================================== C：E14 增强（10% 数据，固定优化步数）、E15 基线行
    K10, M10 = SIZES[10]
    EP10 = epochs_for(P, len(z2), K10, M10)
    D10 = data_args(P, z2, K10, M10)
    for aug in ['none'] + AUG_SINGLES:
        R.train('e14_%s_s42' % aug, 'E14', 'C', 'moe_diffmic', '--experts TF --num_workers 2 --phys_aug %s' % aug, D10, EP10, 42, COST['moe'] * 0.9, tier=1, splits=('test', 'test_ood'), deps=[],
                meta=dict(exp='E14', cfg='e14_' + aug, seed=42, size=10))
    R.add('e14_select', 'E14', 'C', '%s exp/selection.py aug' % PYV, 0.01, tier=1, deps=['e14_%s_s42' % a for a in ['none'] + AUG_SINGLES], mode='any_terminal', meta=dict(kind='select'))
    for i in (1, 2):
        R.train('e14_combo%d_s42' % i, 'E14', 'C', 'moe_diffmic', '--experts TF --num_workers 2 --phys_aug {CAND:aug_combo%d}' % i, D10, EP10, 42, COST['moe'] * 0.9, tier=1, splits=('test', 'test_ood'),
                deps=['e14_select'], meta=dict(exp='E14', cfg='e14_combo%d' % i, seed=42, size=10))
    R.add('e14_augstats', 'E14', 'C', '%s aug_stats.py --n %d --out ../reports/e14_aug_stats.csv' % (PYV, 100 if P['smoke'] else 600), 0.3, tier=1, deps=[], meta=dict(kind='script', exp='E14'))
    for sd in (43, 44):
        for tag, ar in (('none', 'none'), ('top1', '{CAND:aug_top1}'), ('combo1', '{CAND:aug_combo1}')):
            R.train('e14_%s_s%d' % (tag, sd), 'E14', 'C', 'moe_diffmic', '--experts TF --num_workers 2 --phys_aug %s' % ar, D10, EP10, sd, COST['moe'] * 0.9, tier=2, splits=('test', 'test_ood'),
                    deps=['e14_select'], meta=dict(exp='E14', cfg='e14_' + tag, seed=sd, size=10))
    # E15 基线行：CNN / 无专家 DiT / TF-only × 10,30,100% × 3 种子（与其对应的依赖行在 B；见 README 的跨机说明）
    for p, (K, mult) in SIZES.items():
        for sd in (42, 43, 44):
            Dp = data_args(P, z2, K, mult)
            R.train('de_cnn_p%d_s%d' % (p, sd), 'E15', 'C', 'cnn', '--cnn_base_ch 32', Dp, epochs_for(P, len(z2), K, mult, True) if not P['smoke'] else 2, sd, COST['cnn'], tier=1, splits=('test', 'test_ood'),
                    diffusion=False, deps=[], meta=dict(exp='E15', cfg='cnn', size=p, seed=sd))
            R.train('de_dit_p%d_s%d' % (p, sd), 'E15', 'C', 'moe_diffmic', '--disable_moe --true_uncond', Dp, epochs_for(P, len(z2), K, mult), sd, COST['dit'], tier=1, splits=('test', 'test_ood'),
                    deps=[], meta=dict(exp='E15', cfg='dit', size=p, seed=sd))
            R.train('de_tf_p%d_s%d' % (p, sd), 'E15', 'C', 'moe_diffmic', '--experts TF', Dp, epochs_for(P, len(z2), K, mult), sd, COST['moe'], tier=1, splits=('test', 'test_ood'),
                    deps=[], meta=dict(exp='E15', cfg='tf', size=p, seed=sd))

    # ======================================================== B：依赖链
    sB = []
    # ---- 第 1 轮（种子 42 初筛）
    # OOF 基础估计 + 残差扩散/残差CNN：样本外基础估计必须与残差模型使用**同一训练子集**（同种子、同比例），否则查表会缺样本，
    # 且用全量数据训练的基础恢复器会泄漏 10%/30% 实验的数据预算。故按 (比例, 种子) 建立；全量(100%)数据在正式 profile 下所有种子共用 base_zs2。
    def base_for(p, sd):
        nm = 'base_zs2' if (p == 100 and not P['smoke']) else 'base_p%d_s%d' % (p, sd)
        jn = 'oof_zs2' if nm == 'base_zs2' else 'oof_' + nm
        if jn not in R.names:
            K, mult = SIZES[p]
            ep = P['base_ep'] if P['smoke'] else int(round(P['base_ep'] * 40.0 / (K * mult)))
            cmd = '%s make_oof_base.py --name %s --types %s --epochs %d --folds %d --amp bf16 --seed %d --val_per_cell %d %s' % (
                PYV, nm, ' '.join(z2), ep, P['folds'], sd, 1 if P['smoke'] else 5, P['oof_extra'])
            if P['smoke']:
                cmd += ' --train_per_cell 1'
            elif K < 40:
                cmd += ' --train_per_cell %d --aug_multiplier %d' % (K, mult)
            R.add(jn, 'E10', 'B', cmd, COST['oof'] if p == 100 else 0.8, tier=1, deps=sB, meta=dict(kind='script', exp='E10', size=p, seed=sd))
        return jn, nm, '--base_ckpt %s --base_oof runs/%s_oof.npz' % (nm, nm)
    J42, BN42, BASE = base_for(100, 42)
    s1_jobs = []
    for cfg, args in CFG.items():
        name = '%s_s42' % cfg
        extra = args + (' --attr_pretrain_epochs %d' % (15 if not P['smoke'] else 1) if 'pretrain' in args else '')
        R.train(name, cfg.split('_')[0].upper(), 'B', 'moe_diffmic', extra, D2, EP, 42, mcost(args), tier=1, deps=sB, keep=cfg.startswith('e3_') or cfg in ('e1_four', 'e4_dual'),
                meta=dict(exp=cfg.split('_')[0].upper(), cfg=cfg, seed=42))
        s1_jobs.append(name)
    # E1/E4：专家屏蔽影响（固定 checkpoint）与门控诊断（路由携带的是干扰类型还是 SIR）
    for cfg in ('e1_four', 'e4_dual'):
        R.add('imp_%s' % cfg, 'E1', 'B', '%s expert_impact.py --run %s_s42 --unseen LFM NFM RFM --stride %d --out ../reports/expert_impact_%s_s42.csv && %s ../analysis/gate_diagnostics.py --run %s_s42 --suffix _avg8 --experts FD MS TF TD > ../reports/gate_diag_%s_s42.txt; rm -f runs/%s_s42/best_model.pth'
              % (PYV, cfg, 40 if P['smoke'] else 4, cfg, PYV, cfg, cfg, cfg), COST['script'], tier=1, deps=['%s_s42' % cfg], meta=dict(kind='script', exp='E1', cfg=cfg + '_impact'))
    # E3 的“冻结专家特征 + 线性头”迁移检验（对未见类型 LFM NFM RFM），以通用 CNN 特征(base_zs2)与随机专家为对照
    for cfg in ('e3_joint4', 'e3_prefrz4', 'e3_preft4', 'e3_jointTF'):
        R.add('%s_probe' % cfg, 'E3', 'B', '%s loto_adapt.py --run %s_s42 --holdout LFM NFM RFM --cnn_run %s --out runs/%s_s42/loto_adapt.json && rm -f runs/%s_s42/best_model.pth'
              % (PYV, cfg, BN42, cfg, cfg), COST['script'], tier=1, deps=['%s_s42' % cfg, J42], meta=dict(kind='script', exp='E3', cfg=cfg + '_probe'))
    R.train('e10_rdTF_s42', 'E10', 'B', 'res_diff', '--experts TF ' + BASE, D2, EP, 42, COST['res'], tier=1, deps=[J42], keep=True, eval_extra='--parts', meta=dict(exp='E10', cfg='e10_rdTF', seed=42))
    R.train('e10_rdU_s42', 'E10', 'B', 'res_diff', '--disable_moe --true_uncond ' + BASE, D2, EP, 42, COST['res'] * 0.75, tier=1, deps=[J42], eval_extra='--parts', meta=dict(exp='E10', cfg='e10_rdU', seed=42))
    R.train('e10_rc_s42', 'E10', 'B', 'res_cnn', BASE, D2, epochs_for(P, len(z2), 40, 1, True) if not P['smoke'] else 2, 42, COST['rescnn'], tier=1, deps=[J42], diffusion=False, eval_extra='--parts',
            meta=dict(exp='E10', cfg='e10_rc', seed=42))
    R.add('e10_base_eval', 'E10', 'B', ev(P, BN42, ('test',), False), 0.05, tier=1, deps=[J42], meta=dict(kind='script', exp='E10', cfg='e10_base'))
    # E11 门控：在 rdTF 上训练各变体并评估
    for gname, (gv, loc) in GATE_VARIANTS.items():
        R.add('e11_gate_%s' % gname, 'E11', 'B', '%s train_gate.py --run e10_rdTF_s42 --variant %s %s --name %s --steps %d --ddim_steps %d' % (
            PYV, gv, '--local' if loc else '', gname, P['gate_steps'], P['ddim']), COST['gate'], tier=1, deps=['e10_rdTF_s42'], meta=dict(kind='script', exp='E11', cfg=gname))
        R.add('e11_eval_%s' % gname, 'E11', 'B', ev(P, 'e10_rdTF_s42', ('test',), True, tag='_avg8g_%s' % gname, extra='--gate %s --parts' % gname), COST['eval'], tier=1,
              deps=['e11_gate_%s' % gname], meta=dict(kind='script', exp='E11', cfg=gname))
    R.add('select1', 'SEL', 'B', '%s exp/selection.py s1' % PYV, 0.01, tier=1, mode='any_terminal',
          deps=s1_jobs + ['e10_rdTF_s42', 'e10_rdU_s42', 'e10_rc_s42'] + ['e11_gate_%s' % g for g in GATE_VARIANTS], meta=dict(kind='select'))

    # ---- 第 2 轮（补种子与复核；依赖 select1）
    s2_jobs = []
    for cfg in ('e1_tf', 'e1_four', 'e1_generic4', 'e1_uniform4'):
        for sd in (43, 44):
            s2_jobs.append(R.train('%s_s%d' % (cfg, sd), 'E1', 'B', 'moe_diffmic', CFG[cfg], D2, EP, sd, mcost(CFG[cfg]), tier=1, deps=['select1'], meta=dict(exp='E1', cfg=cfg, seed=sd)))
    for key in ('moe_compose', 'moe_top1', 'moe_top2'):
        for sd in ((42, 43, 44) if key == 'moe_compose' else (43, 44)):
            s2_jobs.append(R.train('c_%s_s%d' % (key, sd), 'MOE', 'B', 'moe_diffmic', '{CAND:%s}' % key, D2, EP, sd, COST['moe'] * 1.1, tier=1, deps=['select1'],
                                   skip_if={'moe_compose': 'compose_dup', 'moe_top1': 'top1_dup', 'moe_top2': 'top2_dup'}[key], meta=dict(exp='MOE', cfg='c_' + key, seed=sd)))
    for key in ('bb_top1', 'bb_top2'):
        for sd in (43, 44):
            s2_jobs.append(R.train('c_%s_s%d' % (key, sd), 'E7', 'B', 'moe_diffmic', '--experts TF {CAND:%s}' % key, D2, EP, sd, COST['unet'], tier=1, deps=['select1'],
                                   skip_if={'bb_top1': 'bb_top1_is_dit', 'bb_top2': 'bb_top2_is_dit'}[key], meta=dict(exp='E7', cfg='c_' + key, seed=sd)))
    for key in ('tok_top1',):
        for sd in (43, 44):
            s2_jobs.append(R.train('c_%s_s%d' % (key, sd), 'E8', 'B', 'moe_diffmic', '--experts TF {CAND:%s}' % key, D2, EP, sd, COST['moe'], tier=1, deps=['select1'], skip_if='tok_default',
                                   meta=dict(exp='E8', cfg='c_' + key, seed=sd)))
    for key in ('cond_top1',):
        for sd in (43, 44):
            s2_jobs.append(R.train('c_%s_s%d' % (key, sd), 'E9', 'B', 'moe_diffmic', '--experts TF {CAND:%s}' % key, D2, EP, sd, COST['moe'], tier=1, deps=['select1'], skip_if='cond_default',
                                   meta=dict(exp='E9', cfg='c_' + key, seed=sd)))
    for sd in (42, 43):          # 主干/局部TF/条件注入三者的组合（只评一次，避免把单项收益简单相加）
        s2_jobs.append(R.train('c_basecombo_s%d' % sd, 'E7', 'B', 'moe_diffmic', '--experts TF {CAND:bb_top1} {CAND:tok_top1} {CAND:cond_top1}', D2, EP, sd, COST['unet'], tier=1,
                               deps=['select1'], meta=dict(exp='E7', cfg='c_basecombo', seed=sd)))
    # 若胜出主干不是 DiT：在胜出主干上重做残差扩散
    for sd in (43, 44):
        jn, bn, bargs = base_for(100, sd)
        s2_jobs.append(R.train('e10_rdTF_s%d' % sd, 'E10', 'B', 'res_diff', '--experts TF ' + bargs, D2, EP, sd, COST['res'], tier=1, deps=['select1', jn], eval_extra='--parts', meta=dict(exp='E10', cfg='e10_rdTF', seed=sd)))
        s2_jobs.append(R.train('e10_rc_s%d' % sd, 'E10', 'B', 'res_cnn', bargs, D2, epochs_for(P, len(z2), 40, 1, True) if not P['smoke'] else 2, sd, COST['rescnn'], tier=1, deps=['select1', jn], diffusion=False, eval_extra='--parts',
                               meta=dict(exp='E10', cfg='e10_rc', seed=sd)))
    s2_jobs.append(R.train('e10w_rdTF_s42', 'E10', 'B', 'res_diff', '--experts TF {CAND:bb_top1} ' + BASE, D2, EP, 42, COST['res'], tier=1, deps=['select1'], skip_if='bb_top1_is_dit', eval_extra='--parts',
                           meta=dict(exp='E10', cfg='e10w_rdTF', seed=42)))
    R.add('select2', 'SEL', 'B', '%s exp/selection.py s2' % PYV, 0.01, tier=1, mode='any_terminal', deps=s2_jobs, meta=dict(kind='select'))

    # ---- 第 3 轮（最终：依赖行的 E15/E16 + 胜出者 LOTO + 对照）
    FB = '{CAND:final_base}'

    def gate_cmd(run, name, exp, cfg, p, sd, J, sp):
        R.add(name, exp, 'B', '%s train_gate.py --run %s --variant {CAND:gate_variant_base} {CAND:gate_local_flag} --name final --steps %d --ddim_steps %d' % (PYV, run, P['gate_steps'], P['ddim'])
              + ' && ' + ev(P, run, sp, True, tag='_avg8g', extra='--gate final --parts') + ' && rm -f runs/%s/best_model.pth' % run, COST['gate'] + COST['eval'], tier=1, deps=[run],
              meta=dict(kind='script', exp=exp, cfg=cfg + '_gate', size=p, seed=sd, J=J))
    MF = '{CAND:moe_final}'
    GV = '{CAND:gate_variant}'
    rd_pref = '--experts TF ' + FB + ' ' + BASE
    for p, (K, mult) in SIZES.items():
        Dp, EPp = data_args(P, z2, K, mult), epochs_for(P, len(z2), K, mult)
        for sd in (42, 43):
            sp = ('test', 'test_ood')
            # J1：新 MoE + 普通扩散
            R.train('de_moe_p%d_s%d' % (p, sd), 'E15', 'B', 'moe_diffmic', '%s %s' % (FB, MF), Dp, EPp, sd, COST['moe'], tier=1, splits=sp, deps=['select2'],
                    meta=dict(exp='E15', cfg='moe', size=p, seed=sd, J='J1'))
            # J2/J3 的残差扩散：训练 -> (评估 λ=1) -> 训练门控 -> 评估(门控)
            for tag, cond, J in (('res', '--experts TF %s' % FB, 'J2'), ('joint', '%s %s' % (FB, MF), 'J3')):
                nm = 'de_%s_p%d_s%d' % (tag, p, sd)
                jn, bn, bargs = base_for(p, sd)
                R.train(nm, 'E15', 'B', 'res_diff', cond + ' ' + bargs, Dp, EPp, sd, COST['res'] * 1.1, tier=1, splits=sp, deps=['select2', jn], keep=True, eval_extra='--parts',
                        meta=dict(exp='E15', cfg=tag, size=p, seed=sd, J=J))
                gate_cmd(nm, nm + '_gate', 'E15', tag, p, sd, J, sp)
    for p in (10, 100):
        K, mult = SIZES[p]
        Dp, EPp = data_args(P, z2, K, mult), epochs_for(P, len(z2), K, mult)
        for sd in (42, 43):
            sp = ('test', 'test_ood')
            R.train('j0_p%d_s%d' % (p, sd), 'E16', 'B', 'moe_diffmic', '--experts TF %s' % FB, Dp, EPp, sd, COST['moe'], tier=1, splits=sp, deps=['select2'], meta=dict(exp='E16', cfg='J0', size=p, seed=sd, J='J0'))
            nm = 'j4_p%d_s%d' % (p, sd)
            jn, bn, bargs = base_for(p, sd)
            R.train(nm, 'E16', 'B', 'res_diff', '--fusion uniform %s %s' % (FB, bargs), Dp, EPp, sd, COST['res'] * 1.1, tier=1, splits=sp, deps=['select2', jn], keep=True, eval_extra='--parts',
                    meta=dict(exp='E16', cfg='J4', size=p, seed=sd, J='J4'))
            gate_cmd(nm, nm + '_gate', 'E16', 'J4', p, sd, 'J4', sp)
            R.train('j5_p%d_s%d' % (p, sd), 'E16', 'B', 'mv_cnn', '%s' % MF, Dp, epochs_for(P, len(z2), K, mult, True) if not P['smoke'] else 2, sd, COST['cnn'] * 3, tier=1, splits=sp,
                    diffusion=False, deps=['select2'], meta=dict(exp='E16', cfg='J5', size=p, seed=sd, J='J5'))
    # B 上的同机 E15 基线行（种子 42，tier 2 填充）：保证 B 上的依赖行与基线在同一台机器上可直接比较
    for p, (K, mult) in SIZES.items():
        Dp = data_args(P, z2, K, mult)
        R.train('deb_cnn_p%d_s42' % p, 'E15', 'B', 'cnn', '--cnn_base_ch 32', Dp, epochs_for(P, len(z2), K, mult, True) if not P['smoke'] else 2, 42, COST['cnn'], tier=2, splits=('test', 'test_ood'),
                diffusion=False, deps=[], meta=dict(exp='E15', cfg='cnn_b', size=p, seed=42))
        R.train('deb_dit_p%d_s42' % p, 'E15', 'B', 'moe_diffmic', '--disable_moe --true_uncond', Dp, epochs_for(P, len(z2), K, mult), 42, COST['dit'], tier=2, splits=('test', 'test_ood'),
                deps=[], meta=dict(exp='E15', cfg='dit_b', size=p, seed=42))
        R.train('deb_tf_p%d_s42' % p, 'E15', 'B', 'moe_diffmic', '--experts TF', Dp, epochs_for(P, len(z2), K, mult), 42, COST['moe'], tier=2, splits=('test', 'test_ood'),
                deps=[], meta=dict(exp='E15', cfg='tf_b', size=p, seed=42))
    # 胜出者的 LOTO（B 机上与 tf / cnn 同机对照）
    for H in ('LFM', 'MT', 'PL'):
        R.train('lof_%s_moe_s42' % H, 'E6', 'B', 'moe_diffmic', '%s %s' % (FB, MF), data_args(P, None, holdout=H), epochs_for(P, 7, 40, 1), 42, COST['moe'], tier=1, deps=['select2'],
                meta=dict(exp='E6F', cfg='moe_final', holdout=H, seed=42))
        R.train('lof_%s_tf_s42' % H, 'E6', 'B', 'moe_diffmic', '--experts TF %s' % FB, data_args(P, None, holdout=H), epochs_for(P, 7, 40, 1), 42, COST['moe'], tier=1, deps=['select2'],
                meta=dict(exp='E6F', cfg='tf', holdout=H, seed=42))
        R.train('lof_%s_cnn_s42' % H, 'E6', 'B', 'cnn', '--cnn_base_ch 32', data_args(P, None, holdout=H), epochs_for(P, 7, 40, 1, True) if not P['smoke'] else 2, 42, COST['cnn'], tier=1, deps=['select2'],
                diffusion=False, meta=dict(exp='E6F', cfg='cnn', holdout=H, seed=42))
    # 收尾：删除遗留权重
    for M in 'ABC':
        R.add('cleanup_%s' % M, 'SYS', M, 'BUILTIN:cleanup', 0.001, tier=3, deps=[], mode='any_terminal', meta=dict(kind='script'))
    return R.jobs


if __name__ == '__main__':
    import collections
    jobs = build('real')
    load = collections.defaultdict(float)
    for j in jobs:
        load[j['machine']] += j['cost']
    print({m: round(v, 1) for m, v in load.items()}, len(jobs))
