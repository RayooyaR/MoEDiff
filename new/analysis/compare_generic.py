"""通用对比：所有方法 × 所有指标（BER / EVM / MSE / MAE / 全带 SINR / 带内 SINR），含见过/未见类型、按 SIR、按类型、按调制、显著性。

用法（在 new/moe_diffic 下）：
    python ../analysis/compare_generic.py zs2          # 留出调频家族，全部基线（zs1/zs2/zs3 同理）
    python ../analysis/compare_generic.py fs1|fs5|fs10  # 小样本 K=1/5/10（留出调频家族，含增强）
    python ../analysis/compare_generic.py fsall         # 小样本：BER 及全部指标随 K 的变化
    python ../analysis/compare_generic.py ef            # 更难场景 v5（干扰叠加随机信号影响），test 与 test_ood
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
SIRS = [-20, -15, -10, -5, 0, 5, 10]
KEYS = [('ber', 'BER'), ('evm', 'EVM'), ('mse', 'MSE'), ('mae', 'MAE'), ('sinr', 'SINR'), ('sinr_inband', '带内SINR')]
SEEN = {'zs1': ['NB', 'WB', 'LFM', 'NFM', 'RFM', 'PL'], 'zs2': ['ST', 'MT', 'NB', 'WB', 'PL'], 'zs3': ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM'],
        'fs': ['ST', 'MT', 'NB', 'WB', 'PL']}
MAIN = 'MoE-DiffIC 确定'


def methods_for(prefix):
    p = prefix
    return [('MF 匹配滤波', None, ''), ('LMMSE', p + '_lmmse', ''), ('CNN', p + '_cnn', ''), ('DiT扩散(ICDM) 随机', p + '_dit', ''),
            ('DiT扩散(ICDM) 确定', p + '_dit', '_det'), ('DiffCom', p + '_diffcom', ''), ('ICDM双模型', p + '_icdm', ''),
            ('MoE-DiffIC 随机', p + '_moe', ''), ('MoE-DiffIC 确定', p + '_moe', '_det')]


def load(run, split, suf):
    f = os.path.join(RUNS, run, 'per_sample_%s%s.npz' % (split, suf))
    return {k: v for k, v in np.load(f).items()} if os.path.exists(f) else None


def col(d, k, mf):
    return d[k + ('_mf' if mf else '_model')]


def sig(a, b):
    d = a - b; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return '%+.4f%s' % (m, '▲' if m - 1.96 * se > 0 else ('▼' if m + 1.96 * se < 0 else '·'))


def collect(prefix, split):
    D = {}
    ref = None
    for name, run, suf in methods_for(prefix):
        if run is None:
            continue
        d = load(run, split, suf)
        if d is not None:
            D[name] = (d, False); ref = ref or d
    if ref is not None:
        D = {'MF 匹配滤波': (ref, True), **D}
    return D, ref


def table(title, D, mask):
    print('\n' + title)
    print('%-20s' % '' + ''.join('%10s' % n for _, n in KEYS))
    for name, (d, mf) in D.items():
        print('%-20s' % name + ''.join('%10.4f' % col(d, k, mf)[mask].mean() if k not in ('sinr', 'sinr_inband') else '%10.2f' % col(d, k, mf)[mask].mean() for k, _ in KEYS))


def breakdown(D, ref, split_label, seen=None):
    types = [str(t) for t in ref['type_names']]; mods = [str(m) for m in ref['mod_names']]
    T, Mo, S = ref['type'], ref['mod'], ref['sir']
    allm = np.ones(len(T), bool)
    table('【%s 全部样本】' % split_label, D, allm)
    if seen:
        sm = np.isin(T, [types.index(t) for t in seen]); um = ~sm
        table('【%s 见过的类型 %s】' % (split_label, seen), D, sm)
        table('【%s 未见类型★ %s】' % (split_label, [t for t in types if t not in seen]), D, um)
    else:
        um = allm
    for title, keys, lab, sel in (('按 SIR (dB)', SIRS, ['%+d' % s for s in SIRS], lambda k: S == k),
                                  ('按干扰类型', range(len(types)), [t + ('★' if seen and t not in seen else '') for t in types], lambda k: T == k),
                                  ('按调制', range(len(mods)), mods, lambda k: Mo == k)):
        print('\n【%s BER：%s（%s）】' % (split_label, title, '仅未见类型' if seen and title.startswith('按 SIR') else '全部样本'))
        print('%-20s' % '' + ''.join('%9s' % x for x in lab))
        base = um if (seen and title.startswith('按 SIR')) else allm
        vals = {n: [col(d, 'ber', mf)[sel(k) & base].mean() for k in keys] for n, (d, mf) in D.items()}
        best = [min(vals[n][i] for n in vals) for i in range(len(list(keys)))]
        for n, v in vals.items():
            print('%-20s' % n + ''.join(('%.3f%s' % (x, '*' if abs(x - best[i]) < 1e-12 else ' ')).rjust(9) for i, x in enumerate(v)))
    if MAIN in D:
        print('\n【%s 显著性：基线 BER − %s BER（正=本文更好；▲▼显著）】' % (split_label, MAIN))
        sc = [('全部', allm)] + ([('未见类型★', um)] if seen else []) + [('低SIR≤-10', S <= -10), ('中SIR -5~0', (S >= -5) & (S <= 0)), ('高SIR≥5', S >= 5)]
        print('%-20s' % '' + ''.join('%14s' % n for n, _ in sc))
        o = D[MAIN][0]['ber_model']
        for n, (d, mf) in D.items():
            if n == MAIN:
                continue
            print('%-20s' % n + ''.join(sig(col(d, 'ber', mf)[m], o[m]).rjust(14) for _, m in sc))


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else 'zs2'
    if key in ('zs1', 'zs2', 'zs3'):
        D, ref = collect(key, 'test')
        print('=' * 100 + '\n%s：全部基线 × 全部指标（训练只用见过的类型，测试在全部 8 类上）\n' % key + '=' * 100)
        breakdown(D, ref, key, SEEN[key])
    elif key in ('fs1', 'fs5', 'fs10'):
        K = key[2:]
        D, ref = collect('fs_K' + K, 'test')
        print('=' * 100 + '\n小样本 K=%s 条/格（留出调频家族 LFM/NFM/RFM，训练含增强）：全部基线 × 全部指标\n' % K + '=' * 100)
        breakdown(D, ref, key, SEEN['fs'])
    elif key in ('fsna1', 'fsna5', 'fsna10'):
        K = key[4:]
        D, ref = collect('fs_K' + K + '_na', 'test')
        print('=' * 100 + '\n小样本 K=%s 条/格 · 无增强对照（其余设置与增强版完全相同）：全部基线 × 全部指标\n' % K + '=' * 100)
        breakdown(D, ref, key, SEEN['fs'])
    elif key == 'fsaug':
        print('=' * 100 + '\n增强的贡献：同一 K 下 增强版 vs 无增强版（未见类型★ 与 见过类型；Δ=增强−无增强，BER/EVM/MSE/MAE 负=增强更好，SINR 正=增强更好）\n' + '=' * 100)
        names = [('LMMSE', 'lmmse', ''), ('CNN', 'cnn', ''), ('DiT扩散(ICDM) 确定', 'dit', '_det'), ('DiffCom', 'diffcom', ''), ('ICDM双模型', 'icdm', ''),
                 ('MoE-DiffIC 确定', 'moe', '_det')]
        for K in (1, 5, 10):
            rows = {}
            for scope in ('未见类型★', '见过类型'):
                print('\nK=%d  %s' % (K, scope))
                print('%-20s' % '' + ''.join('%9s' % n + ' ' * 0 for _, n in KEYS) + '   (每指标：增强 / 无增强 / Δ)')
                for nm, m, suf in names:
                    a, b = load('fs_K%d_%s' % (K, m), 'test', suf), load('fs_K%d_na_%s' % (K, m), 'test', suf)
                    if a is None or b is None:
                        continue
                    T = a['type']; types = [str(t) for t in a['type_names']]
                    um = ~np.isin(T, [types.index(t) for t in SEEN['fs']]); mask = um if scope.startswith('未见') else ~um
                    line = '%-20s' % nm
                    for k, _ in KEYS:
                        x, y = a[k + '_model'][mask].mean(), b[k + '_model'][mask].mean()
                        line += '  %.3f/%.3f/%+.3f' % (x, y, x - y) if k not in ('sinr', 'sinr_inband') else '  %.2f/%.2f/%+.2f' % (x, y, x - y)
                    d = a['ber_model'][mask] - b['ber_model'][mask]
                    se = d.std(ddof=1) / np.sqrt(len(d)); mm = d.mean()
                    line += '   BER Δ显著性: ' + ('增强显著更好' if mm + 1.96 * se < 0 else ('增强显著更差' if mm - 1.96 * se > 0 else '无显著差异'))
                    print(line)
    elif key == 'fsall':
        print('=' * 100 + '\n小样本曲线：各方法随 K（条/格）的变化（未见类型★ / 见过类型 / 全部），全部指标\n' + '=' * 100)
        for K in (1, 5, 10):
            D, ref = collect('fs_K%d' % K, 'test')
            if ref is None:
                continue
            T = ref['type']; types = [str(t) for t in ref['type_names']]
            um = ~np.isin(T, [types.index(t) for t in SEEN['fs']]); sm = ~um
            table('K=%d  未见类型★' % K, D, um)
        for K in (1, 5, 10):
            D, ref = collect('fs_K%d' % K, 'test')
            if ref is None:
                continue
            T = ref['type']; types = [str(t) for t in ref['type_names']]
            sm = np.isin(T, [types.index(t) for t in SEEN['fs']])
            table('K=%d  见过的类型' % K, D, sm)
    elif key == 'ef':
        for split in ('test', 'test_ood'):
            D, ref = collect('ef', split)
            if ref is None:
                continue
            print('=' * 100 + '\n更难场景 v5（干扰叠加随机信号影响）%s：全部基线 × 全部指标\n' % split + '=' * 100)
            breakdown(D, ref, split)
            # 按效应出现与否 / 个数分析（按 loader 的样本顺序重建）
            root = os.path.join(HERE, '..', 'data', 'v5_effects', split)
            eff = []
            for f in sorted(glob.glob(os.path.join(root, '*', '*', '*.npz'))):
                z = np.load(f)
                eff += [json.loads(p)['effects'] for p in z['interference_params']]
            if len(eff) == len(ref['type']):
                ne = np.array([len(e) for e in eff])
                names = ['nonlinearity', 'iq_imbalance', 'phase_noise', 'cfo', 'drift', 'multipath', 'fading']
                print('\n【%s 按干扰上叠加的效应个数 的 BER】' % split)
                groups = [('0~1个', ne <= 1), ('2~3个', (ne >= 2) & (ne <= 3)), ('4~5个', (ne >= 4) & (ne <= 5)), ('6~7个', ne >= 6)]
                print('%-20s' % '' + ''.join('%10s' % g for g, _ in groups))
                for n, (d, mf) in D.items():
                    print('%-20s' % n + ''.join('%10.3f' % col(d, 'ber', mf)[m].mean() for _, m in groups))
                print('\n【%s 各效应出现时 vs 不出现时的 BER（出现 / 不出现）】' % split)
                print('%-20s' % '' + ''.join('%16s' % e[:12] for e in names))
                for n, (d, mf) in D.items():
                    if n in ('MF 匹配滤波', 'CNN', 'LMMSE', MAIN):
                        b = col(d, 'ber', mf)
                        row = ''
                        for e in names:
                            pres = np.array([e in x for x in eff])
                            row += '%16s' % ('%.3f/%.3f' % (b[pres].mean(), b[~pres].mean()))
                        print('%-20s' % n + row)
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
