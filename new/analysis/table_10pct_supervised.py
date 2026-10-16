"""10% 样本 · 监督场景：基线对比总表（第一列=方法，第一行=全部指标；无数据留空）。
场景：训练 5 类干扰(ST/MT/NB/WB/PL)，每个(类型×调制×信干比)格 4 条(共 420 条，约全量 10%)，步数与满数据对齐；
      测试集 A（与训练同参数区间）中"训练见过的 5 类"=监督场景（2100 条，每隔一条）。
用法（在 new/moe_diffic 下）： python ../analysis/table_10pct_supervised.py  -> ../reports/table_10pct_supervised.md / .csv
"""
import csv, os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from aggregate_results import load
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs'); OUT = os.path.join(HERE, '..', 'reports')
FM = ['LFM', 'NFM', 'RFM']
S = lambda b, ss: ['%s_s%d' % (b, s) for s in ss]
ROWS = [('传统', '匹配滤波 MF（不做抑制，直接解调）', 'MF'),
        ('传统', 'LMMSE（滑窗维纳滤波）', ['zs2_lmmse']),
        ('深度学习', 'CNN（直接回归）', S('de_cnn_p10', (42, 43, 44))),
        ('深度学习', 'Restormer（Transformer 直接回归）', S('rest_p10', (42, 43, 44))),
        ('深度学习', 'DiT 条件扩散（无专家）', S('de_dit_p10', (42, 43, 44))),
        ('最新', 'DiffCom（无条件先验+测量一致性引导，官方式热启动）', S('fs_p10_diffcom', (42, 43, 44))),
        ('最新', 'ICDM（信号/干扰双模型分数解耦）', S('fs_p10_icdm', (42, 43, 44))),
        ('本文方法', 'MoE-DiffIC：DiT 扩散 + 时频专家', S('de_tf_p10', (42, 43, 44))),
        ('本文方法', 'MoE-DiffIC：U-Net 扩散 + 时频专家（J0）', S('j0_p10', (42, 43)))]
METR = [('ber', 'BER↓', '%.4f', min), ('delta', 'ΔBER↑', '%+.4f', max), ('neg', '负增益比例↓', '%.3f', min), ('evm', 'EVM↓', '%.3f', min),
        ('mse', 'MSE↓', '%.3f', min), ('mae', 'MAE↓', '%.3f', min), ('sinr', '带内SINR(dB)↑', '%.2f', max)]

def cell(runs, mf=False):
    vs = []
    for r in runs:
        d = load(r, 'test', '_avg8', RUNS)
        if d is None:
            d = load(r, 'test', '', RUNS)
            if d is None: continue
            n = len(d['type']); d = {k: (v[::2] if getattr(v, 'ndim', 0) >= 1 and len(v) == n else v) for k, v in d.items()}
        names = [str(t) for t in d['type_names']]
        m = ~np.isin(d['type'], [i for i, t in enumerate(names) if t in FM])          # 监督场景：训练见过的 5 类
        b, bi = d['ber_model'][m], d['ber_mf'][m]
        if mf: vs.append(dict(ber=bi.mean(), delta=0.0, neg=float('nan'), evm=d['evm_mf'][m].mean(), mse=d['mse_mf'][m].mean(), mae=d['mae_mf'][m].mean(), sinr=d['sinr_inband_mf'][m].mean(), n=int(m.sum())))
        else: vs.append(dict(ber=b.mean(), delta=(bi - b).mean(), neg=(b > bi + 1e-12).mean(), evm=d['evm_model'][m].mean(), mse=d['mse_model'][m].mean(), mae=d['mae_model'][m].mean(), sinr=d['sinr_inband_model'][m].mean(), n=int(m.sum())))
    if not vs: return None
    o = {k: float(np.mean([v[k] for v in vs])) for k in ('ber', 'delta', 'neg', 'evm', 'mse', 'mae', 'sinr')}
    o['ns'] = len(vs); o['n'] = vs[0]['n']; o['sd'] = float(np.std([v['ber'] for v in vs], ddof=1)) if len(vs) > 1 else None
    return o

res = []
for cat, name, runs in ROWS:
    c = cell(['de_cnn_p10_s42'], mf=True) if runs == 'MF' else cell(runs)
    res.append((cat, name, c))
base = [c for cat, _, c in res if c and cat != '本文方法' and cat != '传统']          # 加粗：深度学习/最新基线之间的最优
allv = [c for _, _, c in res if c]
best = {k: f(c[k] for c in allv if c[k] == c[k]) for k, _, _, f in METR}
def pm(runs):
    for x in (runs if isinstance(runs, list) else ['de_cnn_p10_s42']):
        f = os.path.join(RUNS, x, 'run_info.json')
        if os.path.exists(f): return '%.2fM' % (json.load(open(f))['params'] / 1e6)
    return '—'
md = ['# 10% 样本 · 监督场景：基线对比\n',
      '> 场景：训练只含 5 类干扰（ST/MT/NB/WB/PL），每个（类型×调制×信干比）格 4 条，共 420 条（约全量 10%），训练步数与满数据对齐。',
      '> 测试：测试集 A（与训练同参数区间）中这 5 类干扰（**监督场景**：测试类型都在训练里出现过），2100 条，每隔一条取样。',
      '> 扩散类均为平均 8 次采样、25 步 DDIM；多个随机种子先分别计算再取均值（BER 后括号为种子间标准差）。**— = 该设定下没有数据，未补。** 粗体 = 该列最优（含本文方法）。\n',
      '| 方法 | 种子 | 参数量 | ' + ' | '.join(m[1] for m in METR) + ' |', '|---|---|---|' + '---|' * len(METR)]
csv_rows = []; last = None
for (cat, name, c), (_, _, runs) in zip(res, ROWS):
    if cat != last:
        md.append('| **【%s】** |' % cat + ' |' * (2 + len(METR))); last = cat
    if c is None:
        md.append('| %s | — | — | ' % name + ' | '.join('—' for _ in METR) + ' |'); csv_rows.append([cat, name, 0] + [''] * len(METR)); continue
    cs = []
    for k, _, f, _ in METR:
        v = c[k]; t = '—' if v != v else f % v
        if k == 'ber' and c['sd'] is not None: t += '(±%.4f)' % c['sd']
        if v == v and abs(v - best[k]) < 1e-12 and runs != 'MF': t = '**' + t + '**'
        cs.append(t)
    md.append('| %s | %s | %s | ' % (name, c['ns'] if runs != 'MF' else '—', pm(runs) if runs != 'MF' else '—') + ' | '.join(cs) + ' |')
    csv_rows.append([cat, name, c['ns']] + ['%.6f' % c[k] if c[k] == c[k] else '' for k, _, _, _ in METR])
md.append('\n**说明**\n- LMMSE 不依赖训练数据量（滑窗自适应滤波），数值与其他数据量设定相同；MF = 不处理直接解调，作为下限参照。\n- DiffCom：官方式热启动（起始步由训练集平均 SINR=-5.5 dB 按官方公式给出，t0≈491），ζ 在验证集（仅训练见过的类型）上选，3 个种子选出 ζ=10、10、3.33；ICDM 的 γ 同样在验证集上选，选出 0.5、0.5、0.75。两者均平均 8 次采样；DiffCom 在 fp32 下评估。\n- Restormer：官方结构的 1D 移植，损失/优化器与 CNN 一致，22.77M 参数。\n- "负增益比例" = 处理之后比不处理更差的样本占比；"带内 SINR" = 恢复后带内信号与残余干扰加噪声功率比。')
open(os.path.join(OUT, 'table_10pct_supervised.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
with open(os.path.join(OUT, 'table_10pct_supervised.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['类别', '方法', '种子数'] + [m[1] for m in METR]); w.writerows(csv_rows)
print('\n'.join(md))
