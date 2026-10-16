"""10% 样本少样本设定：全部基线 × 全部指标（无数据的格子留空）。
用法（在 new/moe_diffic 下）： python ../analysis/fewshot10_table.py   -> ../reports/fewshot10_table.md / .csv
设定：训练只含 5 类干扰(ST/MT/NB/WB/PL)，每个(类型×调制×信干比)格 4 条(共 420 条，约占全量 10%)，优化步数与 100% 对齐；LFM/NFM/RFM 三类整组未见。
"""
import csv, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from aggregate_results import load
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs'); OUT = os.path.join(HERE, '..', 'reports')
FM = ['LFM', 'NFM', 'RFM']
def S(b, ss): return ['%s_s%d' % (b, s) for s in ss]
# (显示名, [run], 后缀, 备注)
M = [('LMMSE（传统滑窗维纳，不训练，与数据量无关）', ['zs2_lmmse'], '', ''),
     ('CNN（直接回归）', S('de_cnn_p10', (42, 43, 44)), '_avg8', ''),
     ('DiffCom（无条件先验+测量引导）', [], '', '10% 设定下未运行'),
     ('ICDM 双模型', [], '', '10% 设定下未运行'),
     ('DiT 扩散，无专家（真无条件）', S('de_dit_p10', (42, 43, 44)), '_avg8', ''),
     ('MoE-DiffIC 四专家 + 旧路由（DiT）', [], '', '10% 设定下未运行'),
     ('DiT 扩散 + 时频专家', S('de_tf_p10', (42, 43, 44)), '_avg8', ''),
     ('U-Net 扩散 + 时频专家（J0）', S('j0_p10', (42, 43)), '_avg8', ''),
     ('U-Net 扩散 + 共享+四专家（J1）', S('de_moe_p10', (42, 43)), '_avg8', ''),
     ('CNN 打底 + 残差扩散（时频，λ=1）', S('de_res_p10', (42, 43)), '_avg8', ''),
     ('　　+ 自适应门控（J2）', S('de_res_p10', (42, 43)), '_avg8g', ''),
     ('CNN 打底 + 新MoE + 残差扩散（λ=1）', S('de_joint_p10', (42, 43)), '_avg8', ''),
     ('　　+ 自适应门控（J3）', S('de_joint_p10', (42, 43)), '_avg8g', ''),
     ('静态多分支 + 残差扩散 + 门控（J4）', S('j4_p10', (42, 43)), '_avg8g', ''),
     ('新MoE + 直接回归 CNN，不用扩散（J5）', S('j5_p10', (42, 43)), '_avg8', '')]
METR = [('ber', 'BER↓', '%.4f'), ('delta', 'ΔBER↑', '%+.4f'), ('neg', '负增益比例↓', '%.3f'), ('evm', 'EVM↓', '%.3f'), ('mse', 'MSE↓', '%.3f'), ('mae', 'MAE↓', '%.3f'), ('sinr', '带内SINR(dB)↑', '%.2f')]
BEST = {'ber': min, 'delta': max, 'neg': min, 'evm': min, 'mse': min, 'mae': min, 'sinr': max}
SUBS = [('测试集A · 见过类型(ST/MT/NB/WB/PL)', 'test', 'seen'), ('测试集A · 未见类型(LFM/NFM/RFM，零样本)', 'test', 'held'), ('测试集A · 全部类型', 'test', 'all'), ('测试集B · 未见参数区间(OOD)，全部类型', 'test_ood', 'all')]

def cell(runs, split, suf, which):
    vs = []
    for r in runs:
        d = load(r, split, suf, RUNS)
        if d is not None and suf == '':                 # 非扩散：完整测试集每隔一条，与 avg8 同一批样本
            n = len(d['type']); d = {k: (v[::2] if getattr(v, 'ndim', 0) >= 1 and len(v) == n else v) for k, v in d.items()}
        if d is None: continue
        names = [str(t) for t in d['type_names']]
        h = np.isin(d['type'], [i for i, t in enumerate(names) if t in FM])
        m = np.ones(len(d['type']), bool) if which == 'all' else (h if which == 'held' else ~h)
        b, bi = d['ber_model'][m], d['ber_mf'][m]
        vs.append(dict(ber=b.mean(), delta=(bi - b).mean(), neg=(b > bi + 1e-12).mean(), evm=d['evm_model'][m].mean(), mse=d['mse_model'][m].mean(), mae=d['mae_model'][m].mean(), sinr=d['sinr_inband_model'][m].mean(),
                       mf=dict(ber=bi.mean(), delta=0.0, neg=float('nan'), evm=d['evm_mf'][m].mean(), mse=d['mse_mf'][m].mean(), mae=d['mae_mf'][m].mean(), sinr=d['sinr_inband_mf'][m].mean()), n=int(m.sum())))
    if not vs: return None
    o = {k: float(np.mean([v[k] for v in vs])) for k in ('ber', 'delta', 'neg', 'evm', 'mse', 'mae', 'sinr')}
    o['mf'] = {k: float(np.mean([v['mf'][k] for v in vs])) for k in vs[0]['mf']}; o['ns'] = len(vs); o['n'] = vs[0]['n']
    o['sd'] = float(np.std([v['ber'] for v in vs], ddof=1)) if len(vs) > 1 else None
    return o

def main():
    md = ['# 10% 样本少样本设定：全部基线 × 全部指标\n',
          '> 由 `new/analysis/fewshot10_table.py` 生成。训练只含 5 类干扰(ST/MT/NB/WB/PL)，每个(类型×调制×信干比)格 4 条，共 420 条（约全量 10%），优化步数与 100% 对齐；LFM/NFM/RFM 三类整组不参与训练。',
          '> 扩散类均为平均 8 次采样、25 步；测试集每隔一条（A=3360 条，B=1680 条）。**空白(—)=该设定下没有数据，未补。** 种子数见"种子"列，BER 后括号为种子间标准差。粗体=该列最优。\n']
    rows = []
    for title, split, which in SUBS:
        res = [(n, cell(r, split, suf, which), note) for n, r, suf, note in M]
        if which == 'all' and split == 'test_ood':        # LMMSE 只有 test，B 集用全量监督 LMMSE（同样不依赖数据量）
            res[0] = (M[0][0], cell(['exp_lmmse'], 'test_ood', '', which), '')
        ref = next(c for _, c, _ in res if c)
        md.append(f'\n## {title}（样本 {ref["n"]}）\n')
        md.append('| 方法 | 种子 | 参数量 | ' + ' | '.join(m[1] for m in METR) + ' |'); md.append('|' + '---|' * (3 + len(METR)))
        best = {k: BEST[k]([c[k] for _, c, _ in res if c]) for k, _, _ in METR}
        def row(n, c, note, mf=False):
            if c is None: return '| %s | — | — | ' % n + ' | '.join('—' for _ in METR) + ' |'
            cs = []
            for k, _, f in METR:
                v = c[k] if not mf else c['mf'][k]
                t = '—' if (v != v) else f % v
                if k == 'ber' and not mf and c['sd'] is not None: t += '(±%.4f)' % c['sd']
                if not mf and abs(v - best[k]) < 1e-12: t = '**' + t + '**'
                cs.append(t)
            return '| %s | %s | %s | ' % (n, '—' if mf else c['ns'], '—') + ' | '.join(cs) + ' |'
        md.append(row('不处理(匹配滤波，MF)', dict(ref, mf=ref['mf']), '', mf=True))
        for n, c, note in res:
            md.append(row(n, c, note))
            if c: rows.append([title, n, c['ns']] + ['%.6f' % c[k] for k, _, _ in METR])
            else: rows.append([title, n, 0] + [''] * len(METR))
    # 参数量列：从 run_info 取
    import json
    pm = {}
    for n, r, suf, _ in M:
        for x in r:
            f = os.path.join(RUNS, x, 'run_info.json')
            if os.path.exists(f): pm[n] = json.load(open(f))['params'] / 1e6; break
    txt = '\n'.join(md) + '\n'
    for n, p in pm.items():
        txt = txt.replace('| %s | ' % n, '| %s | ' % n, 1)
    # 填参数量（把每个方法行第三列的 — 换成参数量）
    out = []
    for line in txt.split('\n'):
        for n, p in pm.items():
            if line.startswith('| %s | ' % n):
                parts = line.split(' | '); parts[2] = '%.2fM' % p; line = ' | '.join(parts)
        out.append(line)
    txt = '\n'.join(out)
    txt += '\n\n## 参考：旧管线每格 K=5 条（约 12.5%，不增强，150 轮，步数未对齐）下有 DiffCom / ICDM / 四专家数据\n\n仅作参考，**不是 10% 设定**；训练配方不同，不能与上表直接比较。\n'
    ref_md = ['| 方法 | 见过类型 BER | 未见类型 BER |', '|---|---|---|']
    for n, r in (('LMMSE', 'fs_K5_na_lmmse'), ('CNN', 'fs_K5_na_cnn'), ('DiffCom', 'fs_K5_na_diffcom'), ('ICDM', 'fs_K5_na_icdm'), ('DiT 无专家', 'fs_K5_na_dit'), ('MoE 四专家', 'fs_K5_na_moe')):
        a = cell([r], 'test', '_avg8' if r.split('_')[-1] not in ('lmmse', 'cnn') else '', 'seen'); b = cell([r], 'test', '_avg8' if r.split('_')[-1] not in ('lmmse', 'cnn') else '', 'held')
        ref_md.append('| %s | %s | %s |' % (n, '—' if a is None else '%.4f' % a['ber'], '—' if b is None else '%.4f' % b['ber']))
    txt += '\n'.join(ref_md) + '\n'
    open(os.path.join(OUT, 'fewshot10_table.md'), 'w', encoding='utf-8').write(txt)
    with open(os.path.join(OUT, 'fewshot10_table.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['测试集', '方法', '种子数'] + [m[1] for m in METR]); w.writerows(rows)
    print('ok', len(rows))
main()
