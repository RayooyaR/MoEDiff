"""汇总报告：把三台服务器的结果（合并到同一份 runs/ 之后）整理成 ../reports/plan_report.md。
每个实验一节：多种子“均值±标准差”（BER 全部/见过/未见类型）、ΔBER（相对未恢复输入）、负增益比例、参数量，以及预先登记的配对对比（按 clean_id 分组 Bootstrap 95% CI）。
缺失的 run 显示为 —；失败/跳过的任务在“执行概况”里列出。不做任何基于 test 的选择——选择只发生在 selection.py（验证集）。
用法: python exp/final_report.py [--profile real] [--out ../reports/plan_report.md]"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, '..', 'analysis'))
import registry as REG  # noqa: E402
from aggregate_results import align, load, paired, summarize, type_mask  # noqa: E402

UN = REG.ZS2_UNSEEN


def runs_dir():
    return os.path.join(ROOT, 'runs')


def lo(name, split='test', suffix='_avg8'):
    return load(name, split, suffix, runs_dir())


def ms(vals, fmt='%.4f'):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return '—'
    return (fmt % np.mean(v)) + ('±' + fmt % np.std(v, ddof=1) if len(v) > 1 else '') + ('' if len(v) > 1 else '')


def run_stats(name, split='test', suffix='_avg8', unseen=UN, holdout=None):
    d = lo(name, split, suffix)
    if d is None:
        return None
    out = {'all': summarize(d)}
    if holdout:
        out['unseen'] = summarize(d, type_mask(d, types=holdout)); out['seen'] = summarize(d, type_mask(d, exclude=holdout))
    elif unseen:
        out['seen'] = summarize(d, type_mask(d, exclude=unseen)); out['unseen'] = summarize(d, type_mask(d, types=unseen))
    return out


def params_of(name):
    f = os.path.join(runs_dir(), name, 'run_info.json')
    return json.load(open(f))['params'] if os.path.exists(f) else None


def table(title, rows, split='test', suffix='_avg8', unseen=UN, holdout=None, note=''):
    """rows: [(标签, [run 名...])]；同一行的多个 run 视为多个种子。"""
    L = ['### ' + title, '']
    if note:
        L += [note, '']
    L += ['| 配置 | 种子数 | 参数量(M) | BER 全部 | BER 见过类型 | BER 未见类型 | ΔBER 未见 | 负增益比例(未见) |', '|---|---|---|---|---|---|---|---|']
    for lab, names in rows:
        st = [run_stats(n, split, suffix, unseen, holdout) for n in names]
        st = [s for s in st if s]
        if not st:
            L.append('| %s | 0 | — | — | — | — | — | — |' % lab); continue
        p = params_of(names[0])
        key = 'unseen' if 'unseen' in st[0] else 'all'
        L.append('| %s | %d | %s | %s | %s | %s | %s | %s |' % (lab, len(st), '%.2f' % (p / 1e6) if p else '—', ms([s['all']['ber'] for s in st]),
                                                              ms([s['seen']['ber'] for s in st if 'seen' in s]), ms([s['unseen']['ber'] for s in st if 'unseen' in s]),
                                                              ms([s[key]['delta_ber'] for s in st], '%+.4f'), ms([s[key]['neg_gain_ratio'] for s in st], '%.3f')))
    return L + ['']


def seeds_avg(names, split='test', suffix='_avg8'):
    """多种子：按 clean_id 对齐后对逐样本 BER 取种子平均，返回伪 d（供配对检验）。"""
    ds = [lo(n, split, suffix) for n in names]
    ds = [d for d in ds if d is not None]
    if not ds:
        return None
    base = ds[0]
    if len(ds) == 1:
        return base
    out = dict(base)
    ids = {int(c): i for i, c in enumerate(base['clean_id'])}
    acc = np.zeros(len(base['clean_id'])); cnt = 0
    for d in ds:
        idx = {int(c): i for i, c in enumerate(d['clean_id'])}
        if set(idx) != set(ids):
            continue
        acc += d['ber_model'][[idx[int(c)] for c in base['clean_id']]]; cnt += 1
    out['ber_model'] = acc / max(cnt, 1)
    return out


def pairs(title, spec, split='test', suffix='_avg8', unseen=UN, holdout=None):
    """spec: [(A 标签, [A runs], B 标签, [B runs])]"""
    L = ['### ' + title, '', '| 对比（A − B，负=A 更好） | 子集 | 配对 BER 差 [95% CI]（▼ A显著更好 ▲ A显著更差 · 不显著） |', '|---|---|---|']
    for la, na, lb, nb in spec:
        a, b = seeds_avg(na, split, suffix), seeds_avg(nb, split, suffix)
        if a is None or b is None:
            L.append('| %s − %s | — | 缺数据 |' % (la, lb)); continue
        subs = [('全部', None)]
        if holdout:
            subs += [('见过', type_mask(a, exclude=holdout)), ('留出', type_mask(a, types=holdout))]
        elif unseen:
            subs += [('见过', type_mask(a, exclude=unseen)), ('未见', type_mask(a, types=unseen))]
        for sn, sel in subs:
            m, lo_, hi, n = paired(a, b, sel, n_boot=1000)
            L.append('| %s − %s | %s | %+.4f [%+.4f,%+.4f] %s |' % (la, lb, sn, m, lo_, hi, '▼' if hi < 0 else ('▲' if lo_ > 0 else '·')))
    return L + ['']


def sec_overview_(plans):
    L = ['## 0. 执行概况与跨机一致性', '']
    for M in 'ABC':
        f = os.path.join(ROOT, 'exp_state', M, 'summary.json')
        if os.path.exists(f):
            s = json.load(open(f))
            L.append('- 机器 %s：用时 %.1f h；任务状态 %s；失败 %s；早期中止 %s；因缺上游被跳过 %d；因时间不够跳过 %s' % (M, s['hours'], s['counts'], s['failed'] or '无', s['aborted'] or '无', len(s['dep_failed']), s['skipped_time'] or '无'))
        else:
            L.append('- 机器 %s：没有 summary.json（尚未合并或未运行）' % M)
    L += ['', '**锚点任务**（同一配置 TF-only / CNN、种子 42、zs2 设置，在三台机器各跑一次）：同一配置在不同机器上的 BER 差异就是“机器/环境噪声”的量级；跨机器比较时应以此为参照。', '',
          '| 锚点 | 机器 | BER 见过 | BER 未见 |', '|---|---|---|---|']
    for a in ('anc_tf', 'anc_cnn'):
        for M in 'ABC':
            s = run_stats('%s_%s' % (a, M))
            L.append('| %s | %s | %s | %s |' % (a, M, '%.4f' % s['seen']['ber'] if s else '—', '%.4f' % s['unseen']['ber'] if s else '—'))
    sel = os.path.join(ROOT, 'exp_state', 'B', 'selection_log.txt')
    if os.path.exists(sel):
        L += ['', '**自动选择记录（B 机，验证集）**', '', '```', open(sel).read().strip(), '```']
    w = os.path.join(ROOT, 'exp_state', 'B', 'winners.json')
    if os.path.exists(w):
        L += ['', '胜出者：`' + json.dumps(json.load(open(w)), ensure_ascii=False) + '`']
    return L + ['']


def sec_moe():
    L = ['## E1–E5 MoE 设计（zs2 设置：训练 ST/MT/NB/WB/PL，未见 LFM/NFM/RFM；种子 42 初筛，E1 关键配置补 43/44）', '']
    s3 = lambda c: ['%s_s%d' % (c, s) for s in (42, 43, 44)]
    L += table('E1 单专家与融合对照（含通用 CNN 专家控制）', [('FD-only', s3('e1_fd')), ('TF-only', s3('e1_tf')), ('TD-only', s3('e1_td')), ('MS-only', s3('e1_ms')), ('四专家(legacy 路由)', s3('e1_four')),
                                                    ('四专家静态均匀融合', s3('e1_uniform4')), ('4 个通用 CNN 专家(控制)', s3('e1_generic4'))])
    L += table('E2 共享专家', [('四专家', s3('e1_four')), ('共享+四专家', ['e2_shared4_s42']), ('共享+TF', ['e2_sharedTF_s42']), ('两个通用分支(参数匹配控制)', ['e2_generic2_s42'])])
    L += table('E3 物理属性辅助监督', [('四专家(仅恢复损失)', ['e1_four_s42']), ('联合训练', ['e3_joint4_s42']), ('预训练后冻结', ['e3_prefrz4_s42']), ('预训练后微调', ['e3_preft4_s42']), ('TF+属性(联合)', ['e3_jointTF_s42'])])
    pr = []
    for c in ('e3_joint4', 'e3_prefrz4', 'e3_preft4', 'e3_jointTF'):
        f = os.path.join(runs_dir(), c + '_s42', 'loto_adapt.json')
        if os.path.exists(f):
            d = json.load(open(f))
            pr.append('| %s | %s | %s | %s |' % (c, ' / '.join('%s %.2f' % (g, v) for g, v in d['zero_shot']['trained'].items()), ' / '.join('%s %.2f' % (g, v) for g, v in d['zero_shot'].get('random', {}).items()),
                                                  ' / '.join('%s %.2f' % (g, v) for g, v in d['zero_shot'].get('cnn', {}).items())))
    if pr:
        L += ['**E3 机制检验：冻结专家特征 + 线性头，零样本预测未见类型(LFM/NFM/RFM)的物理属性（标准化 RMSE，越小越好；对照=随机初始化专家 / 通用 CNN 特征）**', '', '| 配置 | 训练后专家 | 随机专家 | 通用CNN特征 |', '|---|---|---|---|'] + pr + ['']
    L += table('E4 形态—强度双因素路由', [('legacy 路由', ['e1_four_s42']), ('先验路由(prior)', ['e4_prior_s42']), ('双因素 dual', ['e4_dual_s42']), ('无交互项', ['e4_noint_s42']), ('仅形态', ['e4_morph_s42']),
                                   ('仅强度', ['e4_strength_s42']), ('oracle SIR（上界，不可部署）', ['e4_oracle_s42'])])
    L += table('E5 融合机制', [('xattn(四专家)', ['e1_four_s42']), ('静态均匀', ['e1_uniform4_s42']), ('加权求和→单 token', ['e5_sum_s42']), ('双因素路由(软权重)', ['e4_dual_s42']), ('Top-K + 真 mask', ['e5_topk_s42']), ('TF-only', ['e1_tf_s42'])])
    L += table('候选复核（3 种子）', [('组合 compose', ['c_moe_compose_s%d' % s for s in (42, 43, 44)]), ('top1', ['c_moe_top1_s43', 'c_moe_top1_s44']), ('top2', ['c_moe_top2_s43', 'c_moe_top2_s44'])])
    L += pairs('配对对比（种子平均后）', [('TF-only', s3('e1_tf'), '四专家', s3('e1_four')), ('四专家', s3('e1_four'), '4 通用CNN专家', s3('e1_generic4')), ('四专家', s3('e1_four'), '静态均匀', s3('e1_uniform4')),
                                  ('TF-only', s3('e1_tf'), '4 通用CNN专家', s3('e1_generic4'))])
    return L


def sec_backbone():
    L = ['## E7–E9 去噪主干 / 局部 TF / 条件注入', '']
    L += table('E7 主干（TF 条件；无条件对照为真无条件分支）', [('DiT 无条件', ['e7_dit0_s42']), ('DiT+TF', ['e1_tf_s42']), ('U-Net(48) 无条件', ['e7_u0_s42']), ('U-Net(64)+TF', ['e7_uTF_s42']), ('U-Net(48)+TF', ['e7_uTF48_s42']),
                                      ('DiffWave(128)+TF', ['e7_dwTF_s42']), ('DiffWave(192)+TF', ['e7_dwTF192_s42'])])
    L += table('E8 局部时频 token', [('全局 TF(1 token)', ['e1_tf_s42']), ('16 个局部 token', ['e8_tok16_s42']), ('32 个局部 token', ['e8_tok32_s42'])])
    L += table('E9 条件注入方式', [('交叉注意力', ['e1_tf_s42']), ('FiLM', ['e9_film_s42']), ('AdaLN', ['e9_adaln_s42'])])
    L += table('组合复核', [('默认', ['e1_tf_s%d' % s for s in (42, 43)]), ('主干+局部TF+注入 组合', ['c_basecombo_s42', 'c_basecombo_s43'])])
    f = os.path.join(ROOT, '..', 'reports', 'e13_sampling_ablation.csv')
    if os.path.exists(f):
        L += ['## E13 采样步数 × 平均次数（既有 checkpoint，不重训）', '', '| run | 步数 | 平均次数 | BER | BER 种子间std | MSE | ms/样本 |', '|---|---|---|---|---|---|---|']
        for r in csv.DictReader(open(f)):
            L.append('| %s | %s | %s | %.4f | %s | %.4f | %.1f |' % (r['run'], r['steps'], r['avg_k'], float(r['ber']), r['ber_seed_std'][:7], float(r['mse']), float(r['ms_per_sample'])))
        L.append('')
    return L


def sec_residual():
    L = ['## E10–E11 残差扩散与自适应修正', '']
    rows = [('CNN 基础恢复器 base_zs2', ['base_zs2']), ('普通残差 CNN', ['e10_rc_s%d' % s for s in (42, 43, 44)]), ('直接扩散 TF-only', ['e1_tf_s%d' % s for s in (42, 43, 44)]),
            ('残差扩散 TF (λ≡1)', ['e10_rdTF_s%d' % s for s in (42, 43, 44)]), ('残差扩散 无条件', ['e10_rdU_s42'])]
    L += table('E10', rows)
    L += ['**E10 残差扩散内部：基础估计本身 vs 加上残差后（种子 42，逐样本 ber_base）**', '']
    d = lo('e10_rdTF_s42')
    if d is not None and 'ber_base' in d:
        a = np.asarray(d['ber_base']); b = d['ber_model']; i = d['ber_mf']
        un = type_mask(d, types=UN)
        L += ['| 子集 | 未恢复输入 | 基础估计(CNN) | 加残差扩散 | 基础→加残差 的变化 |', '|---|---|---|---|---|']
        for nm, sel in (('全部', np.ones(len(a), bool)), ('见过', ~un), ('未见', un)):
            L.append('| %s | %.4f | %.4f | %.4f | %+.4f |' % (nm, i[sel].mean(), a[sel].mean(), b[sel].mean(), (b - a)[sel].mean()))
        L.append('')
    L += ['### E11 门控变体（在 rdTF 上；交叉验证 BER 来自验证集，测试结果为 _avg8g_<名> 文件）', '', '| 门控 | 验证集 cv_ber | λ≡1 验证 BER | λ≡0 验证 BER | 测试 BER 见过 | 测试 BER 未见 | 测试 ΔBER 未见 | 测试负增益比例(未见) |', '|---|---|---|---|---|---|---|---|']
    for g in REG.GATE_VARIANTS:
        f = os.path.join(runs_dir(), 'e10_rdTF_s42', 'gate_%s.json' % g)
        j = json.load(open(f)) if os.path.exists(f) else None
        d = lo('e10_rdTF_s42', 'test', '_avg8g_%s' % g)
        s = (run_stats('e10_rdTF_s42', 'test', '_avg8g_%s' % g) or {})
        L.append('| %s%s | %s | %s | %s | %s | %s | %s | %s |' % (g, ' (oracle 上界)' if g == 'oracle' else '', '%.4f' % j['cv_ber'] if j else '—', '%.4f' % j['ber_lambda1'] if j else '—', '%.4f' % j['ber_lambda0'] if j else '—',
                                                              '%.4f' % s['seen']['ber'] if s else '—', '%.4f' % s['unseen']['ber'] if s else '—', '%+.4f' % s['unseen']['delta_ber'] if s else '—',
                                                              '%.3f' % s['unseen']['neg_gain_ratio'] if s else '—'))
    L.append('')
    # 弱干扰负增益（按 SIR）
    L += ['**按 SIR 的负增益比例（P[BER_恢复 > BER_输入]）**', '', '| 方法 | ' + ' | '.join('%+d dB' % s for s in (-20, -15, -10, -5, 0, 5, 10)) + ' |', '|---|' + '---|' * 7]
    for lab, nm, suf in (('直接扩散 TF-only', 'e1_tf_s42', '_avg8'), ('CNN 基础', 'base_zs2', '_avg8'), ('残差扩散 λ≡1', 'e10_rdTF_s42', '_avg8')) + tuple(('残差+门控 %s' % g, 'e10_rdTF_s42', '_avg8g_%s' % g) for g in ('both', 'both_local')):
        d = lo(nm, 'test', suf)
        if d is None:
            continue
        sir = np.round(d['sir']).astype(int)
        L.append('| %s | ' % lab + ' | '.join('%.3f' % (d['ber_model'][sir == s] > d['ber_mf'][sir == s] + 1e-12).mean() for s in (-20, -15, -10, -5, 0, 5, 10)) + ' |')
    return L + ['']


def sec_loto():
    L = ['## E6 严格逐类留出（Leave-One-Type-Out；训练完全不含留出类型；单独报告留出类型）', '']
    cfgs = list(REG.LOTO_CFGS)
    for M, hs in REG.LOTO_TYPES.items():
        for H in hs:
            rows = [(c, ['lo_%s_%s_s%d' % (H, c, s) for s in (42, 43, 44)]) for c in cfgs]
            L += table('留出 %s（机器 %s；“未见”=留出类型，“见过”=其余 7 类）' % (H, M), rows, unseen=None, holdout=[H])
            L += pairs('留出 %s 的配对对比' % H, [('dual_attr', ['lo_%s_dual_attr_s%d' % (H, s) for s in (42, 43, 44)], 'tf-only', ['lo_%s_tf_s%d' % (H, s) for s in (42, 43, 44)]),
                                          ('tf-only', ['lo_%s_tf_s%d' % (H, s) for s in (42, 43, 44)], 'cnn', ['lo_%s_cnn_s%d' % (H, s) for s in (42, 43, 44)]),
                                          ('legacy4', ['lo_%s_legacy4_s%d' % (H, s) for s in (42, 43, 44)], 'tf-only', ['lo_%s_tf_s%d' % (H, s) for s in (42, 43, 44)])], unseen=None, holdout=[H])
            for c in ('tf', 'dual_attr'):
                f = os.path.join(runs_dir(), 'lo_%s_%s_s42' % (H, c), 'loto_adapt.json')
                if os.path.exists(f):
                    d = json.load(open(f))
                    L.append('- 迁移补充（%s，冻结专家特征→物理属性，标准化 RMSE）零样本：训练后 %s | 随机 %s | 通用CNN %s | 常数基线 %s' % (c, d['zero_shot']['trained'], d['zero_shot'].get('random'), d['zero_shot'].get('cnn'), d['const_baseline']))
            L.append('')
    L += ['### 胜出者的 LOTO 复核（B 机；与 tf-only、cnn 同机）', '']
    for H in ('LFM', 'MT', 'PL'):
        L += table('留出 %s' % H, [('胜出 MoE', ['lof_%s_moe_s42' % H]), ('TF-only', ['lof_%s_tf_s42' % H]), ('CNN', ['lof_%s_cnn_s42' % H])], unseen=None, holdout=[H])
    return L


def sec_aug_data():
    L = ['## E14 物理增强与干扰重新混合（TF-only，10% 训练数据，固定优化步数）', '']
    aug = ['none'] + REG.AUG_SINGLES + ['combo1', 'combo2']
    L += table('E14', [(a, ['e14_%s_s%d' % (a, s) for s in (42, 43, 44)]) for a in aug] + [('top1 单项', ['e14_top1_s43', 'e14_top1_s44']), ])
    wf = os.path.join(ROOT, 'exp_state', 'C', 'winners.json')
    top1 = json.load(open(wf)).get('aug_top1', '') if os.path.exists(wf) else ''
    none3 = ['e14_none_s%d' % s for s in (42, 43, 44)]
    spec = []
    for a in REG.AUG_SINGLES:
        names = ['e14_%s_s42' % a] + (['e14_top1_s43', 'e14_top1_s44'] if a == top1 else [])
        spec.append((a + ('（%d 种子）' % len(names)), names, 'none（同种子集）', none3[:len(names)]))
    spec.append(('combo1（3 种子）', ['e14_combo1_s%d' % s for s in (42, 43, 44)], 'none（3 种子）', none3))
    spec.append(('combo2（1 种子）', ['e14_combo2_s42'], 'none（种子 42）', ['e14_none_s42']))
    L += pairs('E14 配对（相对不增强；按相同种子数的平均比较。注意：置信区间只反映测试样本的随机性，不含训练种子间波动，单种子对比容易被种子运气误导）', spec)
    f = os.path.join(ROOT, '..', 'reports', 'e14_aug_stats.csv')
    if os.path.exists(f):
        L += ['**增强后有效的带内 SIR/SNR 分布（由数组重新计算）**', '', '| 算子 | SIR 均值±std [min,max] | SNR 均值±std [min,max] |', '|---|---|---|']
        for r in csv.DictReader(open(f)):
            L.append('| %s | %.2f±%.2f [%.1f,%.1f] | %.2f±%.2f [%.1f,%.1f] |' % (r['ops'], float(r['sir_mean']), float(r['sir_std']), float(r['sir_min']), float(r['sir_max']), float(r['snr_mean']), float(r['snr_std']), float(r['snr_min']), float(r['snr_max'])))
        L.append('')
    L += ['## E15 训练数据比例（10%/30%/100%）— 未见类型 BER 与 ΔBER', '',
          '行说明：cnn/dit/tf = 基线（C 机，3 种子；`*_b` = B 机同机种子 42）；moe = J1；res = J2（残差扩散+门控，TF 条件）；joint = J3（胜出 MoE + 残差扩散 + 门控）。', '',
          '| 模型 | 数据比例 | 种子数 | BER 见过 | BER 未见 | ΔBER 未见 | 负增益(未见) | test_ood BER |', '|---|---|---|---|---|---|---|---|']
    def names(cfg, p):
        if cfg in ('cnn', 'dit', 'tf'):
            return ['de_%s_p%d_s%d' % (cfg, p, s) for s in (42, 43, 44)], '_avg8'
        if cfg in ('cnn_b', 'dit_b', 'tf_b'):
            return ['deb_%s_p%d_s42' % (cfg[:-2], p)], '_avg8'
        if cfg == 'moe':
            return ['de_moe_p%d_s%d' % (p, s) for s in (42, 43)], '_avg8'
        return ['de_%s_p%d_s%d' % (cfg, p, s) for s in (42, 43)], '_avg8g'
    for cfg in ('cnn', 'dit', 'tf', 'cnn_b', 'dit_b', 'tf_b', 'moe', 'res', 'joint'):
        for p in (10, 30, 100):
            nm, suf = names(cfg, p)
            st = [run_stats(n, 'test', suf) for n in nm]; st = [s for s in st if s]
            od = [run_stats(n, 'test_ood', suf) for n in nm]; od = [s for s in od if s]
            if not st:
                continue
            L.append('| %s | %d%% | %d | %s | %s | %s | %s | %s |' % (cfg, p, len(st), ms([s['seen']['ber'] for s in st]), ms([s['unseen']['ber'] for s in st]), ms([s['unseen']['delta_ber'] for s in st], '%+.4f'),
                                                               ms([s['unseen']['neg_gain_ratio'] for s in st], '%.3f'), ms([s['all']['ber'] for s in od]) if od else '—'))
    L.append('')
    return L


def sec_joint():
    L = ['## E16 两项创新的独立/联合贡献（B 机，同机比较；未见类型 = 调频类）', '', '| 编号 | MoE | Diffusion | 数据比例 | BER 见过 | BER 未见 | ΔBER 未见 | 负增益(未见) |', '|---|---|---|---|---|---|---|---|']
    spec = {'J0': ('无(TF 条件)', '普通扩散', lambda p: (['j0_p%d_s%d' % (p, s) for s in (42, 43)], '_avg8')), 'J1': ('新 MoE', '普通扩散', lambda p: (['de_moe_p%d_s%d' % (p, s) for s in (42, 43)], '_avg8')),
            'J2': ('无(TF 条件)', '自适应残差扩散', lambda p: (['de_res_p%d_s%d' % (p, s) for s in (42, 43)], '_avg8g')), 'J3': ('新 MoE', '自适应残差扩散', lambda p: (['de_joint_p%d_s%d' % (p, s) for s in (42, 43)], '_avg8g')),
            'J4': ('静态多分支', '自适应残差扩散', lambda p: (['j4_p%d_s%d' % (p, s) for s in (42, 43)], '_avg8g')), 'J5': ('新 MoE', '直接监督 CNN(不扩散)', lambda p: (['j5_p%d_s%d' % (p, s) for s in (42, 43)], '_avg8'))}
    for p in (10, 100):
        for J, (m, dname, fn) in spec.items():
            nm, suf = fn(p)
            st = [run_stats(n, 'test', suf) for n in nm]; st = [s for s in st if s]
            if st:
                L.append('| %s | %s | %s | %d%% | %s | %s | %s | %s |' % (J, m, dname, p, ms([s['seen']['ber'] for s in st]), ms([s['unseen']['ber'] for s in st]), ms([s['unseen']['delta_ber'] for s in st], '%+.4f'),
                                                                     ms([s['unseen']['neg_gain_ratio'] for s in st], '%.3f')))
    L += ['', '**交互量 δ = (L_J0−L_J3) − (L_J0−L_J1) − (L_J0−L_J2)**（plan 的定义，L=BER；等价于 (B_J1−B_J0)+(B_J2−B_J0)−(B_J3−B_J0)）：**δ>0 表示协同**（联合收益大于两项独立收益之和），δ≈0 表示可加，δ<0 表示互相抵消；按 clean_id 分组 Bootstrap 95% CI', '', '| 数据比例 | 子集 | δ [95% CI] |', '|---|---|---|']
    for p in (10, 100):
        arr = {J: seeds_avg(spec[J][2](p)[0], 'test', spec[J][2](p)[1]) for J in ('J0', 'J1', 'J2', 'J3')}
        if any(v is None for v in arr.values()):
            L.append('| %d%% | — | 缺数据 |' % p); continue
        base = arr['J0']
        for sn, sel in (('全部', None), ('未见', type_mask(base, types=UN)), ('见过', type_mask(base, exclude=UN))):
            diffs = {}
            ok = True
            for J in ('J1', 'J2', 'J3'):
                ia, ib = align(arr[J], base)
                if len(ia) != len(base['clean_id']):
                    ok = False; break
                diffs[J] = arr[J]['ber_model'][ia] - base['ber_model'][ib]
            if not ok:
                continue
            delta = diffs['J1'] + diffs['J2'] - diffs['J3']
            keep = sel if sel is not None else np.ones(len(delta), bool)
            groups = base['clean_id'][keep]; dv = delta[keep]
            u, inv = np.unique(groups, return_inverse=True)
            gs, gc = np.bincount(inv, weights=dv), np.bincount(inv)
            rng = np.random.default_rng(0)
            bs = [gs[k].sum() / gc[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(1000))]
            lo_, hi = np.percentile(bs, [2.5, 97.5])
            L.append('| %d%% | %s | %+.4f [%+.4f,%+.4f] |' % (p, sn, dv.mean(), lo_, hi))
    L.append('')
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='real')
    ap.add_argument('--out', default=os.path.join(ROOT, '..', 'reports', 'plan_report.md'))
    a = ap.parse_args()
    plans = {M: json.load(open(os.path.join(HERE, 'plans', '%s_%s.json' % (a.profile, M)))) for M in 'ABC' if os.path.exists(os.path.join(HERE, 'plans', '%s_%s.json' % (a.profile, M)))}
    L = ['# plan.md 全部实验结果汇总', '', '- 所有数值：扩散类为“随机采样 8 次取平均”（avg8，测试集每隔一条的一半样本）；BER 越低越好；ΔBER = 未恢复输入 BER − 恢复后 BER；负增益比例 = 恢复后 BER 高于输入的样本比例。',
         '- “见过/未见”：zs2 设置下训练类型 ST/MT/NB/WB/PL 为见过，LFM/NFM/RFM（调频类）为未见；逐类留出一节按各自留出类型。',
         '- 种子：± 为种子间标准差（≥2 个种子才有）。除非特别说明，同一张表里的 run 来自同一台机器；跨机比较请参考“锚点”。', '']
    for sec in (lambda: sec_overview_(plans), sec_moe, sec_backbone, sec_residual, sec_loto, sec_aug_data, sec_joint):
        try:
            L += sec()
        except Exception as e:             # 某一节出错不影响其他节
            L += ['> 该节生成失败：%r' % e, '']
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, 'w', encoding='utf-8').write('\n'.join(L))
    print('written', a.out, len(L), 'lines')


if __name__ == '__main__':
    main()
