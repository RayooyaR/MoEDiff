"""与小样本整合报告（report_fs.py）同样格式的对照：留出整类干扰（zs）与更难场景（ef）。

用法（在 new/moe_diffic 下）：
    python ../analysis/report_generic.py zs   # 留出整类：A.测试集全部为训练见过的类型 / B.全部为训练未见的类型；列 = zs2(去调频) zs1(去单音/多音) zs3(去脉冲)
    python ../analysis/report_generic.py ef   # 更难场景（干扰叠加随机效应）：A.test（与训练同参数区间）/ B.test_ood（未见参数区间）
扩散类方法（DiT/ICDM、DiffCom、ICDM双模型、MoE-DiffIC）的主结果 = 随机采样取平均（每样本独立采样 8 次再平均，*_avg8.npz，使用测试集每隔一条的一半样本）；
非扩散方法取同一批样本。附加行给出 DiT/MoE 的单次随机采样与确定性采样作为参照。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
SEEN = {'zs1': ['NB', 'WB', 'LFM', 'NFM', 'RFM', 'PL'], 'zs2': ['ST', 'MT', 'NB', 'WB', 'PL'], 'zs3': ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM']}
LAB = {'zs2': 'zs2 去调频', 'zs1': 'zs1 去单音/多音', 'zs3': 'zs3 去脉冲'}
METRICS = [('ber', 'BER（越低越好）', '%.4f'), ('evm', 'EVM（越低越好）', '%.3f'), ('mse', 'MSE（越低越好）', '%.3f'), ('mae', 'MAE（越低越好）', '%.3f'),
           ('sinr_inband', '带内SINR dB（越高越好）', '%.2f')]
ROWS = [('MF 匹配滤波', 'mf', None, ''), ('LMMSE', 'sub', 'lmmse', ''), ('CNN', 'sub', 'cnn', ''),
        ('DiT扩散(ICDM) ·平均8次', 'avg', 'dit', '_avg8'), ('DiffCom ·平均8次', 'avg', 'diffcom', '_avg8'),
        ('ICDM双模型 ·平均8次', 'avg', 'icdm', '_avg8'), ('MoE-DiffIC ·平均8次', 'avg', 'moe', '_avg8'),
        ('  参照 DiT 单次随机', 'sub', 'dit', ''), ('  参照 DiT 确定性', 'sub', 'dit', '_det'),
        ('  参照 MoE 单次随机', 'sub', 'moe', ''), ('  参照 MoE 确定性', 'sub', 'moe', '_det')]


def load(run, split, suf):
    f = os.path.join(RUNS, run, 'per_sample_%s%s.npz' % (split, suf))
    return {k: v for k, v in np.load(f).items()} if os.path.exists(f) else None


def get(row, prefix, split):
    disp, kind, name, suf = row
    if kind == 'mf':
        d = load(prefix + '_cnn', split, '')
        if d is None:
            return None
        out = {k.replace('_mf', '_model'): v[::2] for k, v in d.items() if k.endswith('_mf')}
        out.update({k: d[k][::2] for k in ('type', 'sir', 'mod')})
        out.update({k: d[k] for k in ('type_names', 'mod_names')})
        return out
    d = load('%s_%s' % (prefix, name), split, suf)
    if d is None:
        return None
    if kind == 'avg':
        return d
    return {k: (v[::2] if getattr(v, 'ndim', 0) == 1 and len(v) == len(d['type']) else v) for k, v in d.items()}


def print_table(title, fmt, cols, key, getter):
    print('\n[%s]' % title)
    print('%-26s' % '' + ''.join('%16s' % c for c in cols))
    vals = {}
    for row in ROWS:
        vals[row[0]] = [getter(row, i) for i in range(len(cols))]
    main = [r[0] for r in ROWS if not r[0].startswith('  ')]
    best = [(np.nanmax if key == 'sinr_inband' else np.nanmin)([vals[n][i] for n in main]) for i in range(len(cols))]
    for row in ROWS:
        n = row[0]; s = '%-26s' % n
        for i, v in enumerate(vals[n]):
            mark = '*' if (n in main and not np.isnan(v) and abs(v - best[i]) < 1e-12) else ' '
            s += ('--' if np.isnan(v) else (fmt % v) + mark).rjust(16)
        print(s)


def sig(a, b):
    d = a - b; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return '%+.4f%s' % (m, '▲' if m - 1.96 * se > 0 else ('▼' if m + 1.96 * se < 0 else '·'))


def mask_for(d, seen, want_seen):
    types = [str(t) for t in d['type_names']]
    uns = [i for i, t in enumerate(types) if t not in seen]
    return (~np.isin(d['type'], uns)) if want_seen else np.isin(d['type'], uns)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'zs'
    if mode == 'zs':
        sets = ['zs2', 'zs1', 'zs3']
        scopes = [('A. 测试集全部为【训练见过的干扰类型】（新样本）', True), ('B. 测试集全部为【训练从未见过的干扰类型】', False)]
        cols = [LAB[s] for s in sets]
        for title, want_seen in scopes:
            print('\n' + '#' * 112 + '\n' + title + '\n' + '#' * 112)
            for key, mt, fmt in METRICS:
                def getter(row, i, key=key, want_seen=want_seen):
                    d = get(row, sets[i], 'test')
                    if d is None:
                        return np.nan
                    return d[key + '_model'][mask_for(d, SEEN[sets[i]], want_seen)].mean()
                print_table(mt, fmt, cols, key, getter)
        print('\n' + '#' * 112 + '\nC. MoE-DiffIC(平均8次) 相对各方法的 BER 差（对方 − MoE；正=MoE 更好；▲▼显著），分见过 / 未见类型\n' + '#' * 112)
        for want_seen, lab in ((True, '见过类型'), (False, '未见类型')):
            print('\n[%s]' % lab)
            print('%-26s' % '' + ''.join('%16s' % c for c in cols))
            for row in ROWS[:7]:
                if row[0].startswith('MoE'):
                    continue
                s = '%-26s' % row[0]
                for z in sets:
                    a, m = get(row, z, 'test'), get(ROWS[6], z, 'test')
                    if a is None or m is None:
                        s += '--'.rjust(16); continue
                    mk = mask_for(m, SEEN[z], want_seen)
                    s += sig(a['ber_model'][mk], m['ber_model'][mk]).rjust(16)
                print(s)
    else:
        cols = ['A.test(同参数区间)', 'B.test_ood(未见参数)']
        splits = ['test', 'test_ood']
        print('#' * 112 + '\n更难场景 v5（干扰叠加 7 种随机效应）：所有方法 × 所有指标；A=与训练同参数区间的 test，B=未见参数区间的 test_ood\n' + '#' * 112)
        for key, mt, fmt in METRICS:
            def getter(row, i, key=key):
                d = get(row, 'ef', splits[i])
                return np.nan if d is None else d[key + '_model'].mean()
            print_table(mt, fmt, cols, key, getter)
        print('\nC. MoE-DiffIC(平均8次) 相对各方法的 BER 差（对方 − MoE；正=MoE 更好；▲▼显著）')
        print('%-26s' % '' + ''.join('%16s' % c for c in cols))
        for row in ROWS[:7]:
            if row[0].startswith('MoE'):
                continue
            s = '%-26s' % row[0]
            for sp in splits:
                a, m = get(row, 'ef', sp), get(ROWS[6], 'ef', sp)
                s += '--'.rjust(16) if (a is None or m is None) else sig(a['ber_model'], m['ber_model']).rjust(16)
            print(s)
        # 按 SIR
        for sp in splits:
            m = get(ROWS[6], 'ef', sp)
            if m is None:
                continue
            S = m['sir']; sirs = [-20, -15, -10, -5, 0, 5, 10]
            print('\n[%s BER 按 SIR]' % sp)
            print('%-26s' % '' + ''.join('%9s' % ('%+d' % s) for s in sirs))
            for row in ROWS[:7]:
                d = get(row, 'ef', sp)
                if d is not None:
                    print('%-26s' % row[0] + ''.join('%9.3f' % d['ber_model'][S == s].mean() for s in sirs))


if __name__ == '__main__':
    main()
