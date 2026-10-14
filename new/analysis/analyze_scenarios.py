"""场景分析：读取 evaluate_new.py 的逐样本结果，找出模型相对匹配滤波基线 / 相对无 MoE 的优势区域。

用法（在 new/moe_diffic 下）：  python ../analysis/analyze_scenarios.py [主模型 run 名，默认 exp_moe_legacy]
所有比较均为同一批样本上的配对比较；置信区间为均值差的正态近似 95% CI（只反映测试样本抽样误差，不含训练随机种子的波动）。
符号： ▲ 模型显著更好   ▼ 模型显著更差   · 差异不显著
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
MAIN = sys.argv[1] if len(sys.argv) > 1 else 'exp_moe_legacy'
NAMES = {'exp_moe_legacy': '原MoE', 'exp_nomoe': '无MoE', 'exp_moe_prior': '新路由'}
SIRS = [-20, -15, -10, -5, 0, 5, 10]


def load(run, split):
    d = np.load(os.path.join(RUNS, run, 'per_sample_%s.npz' % split))
    return {k: d[k] for k in d.files}


def mark(diff, lo=None):
    """diff = 基准 - 模型（正=模型更好）。返回 (均值, 符号)。"""
    m = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.inf
    if m - 1.96 * se > 0:
        return m, '▲'
    if m + 1.96 * se < 0:
        return m, '▼'
    return m, '·'


def grid(title, rows, cols, rowlab, collab, sel, a, b, key='ber', fmt='%+.3f', scale=1.0):
    """a、b 为两组逐样本数组；格子内容 = 均值(a - b)（a 为基准，正值=b 更好）。"""
    print('\n' + title)
    print('%-8s' % '' + ''.join('%9s' % c for c in collab))
    for r, rl in zip(rows, rowlab):
        line = '%-8s' % rl
        for c in cols:
            m = sel(r, c)
            if m.sum() < 2:
                line += '%9s' % 'n/a'
                continue
            v, s = mark(a[m] - b[m])
            line += ('%s%s' % (fmt % (v * scale), s)).rjust(9)
        print(line)


def main():
    D = {sp: {r: load(r, sp) for r in NAMES} for sp in ('test', 'test_ood')}
    ref = D['test'][MAIN]
    types, mods = list(ref['type_names']), list(ref['mod_names'])
    print('主模型: %s（%s）；所有格子比较的是同一批样本\n' % (MAIN, NAMES.get(MAIN, MAIN)))

    # ---------- A 全局多指标 ----------
    print('=' * 78 + '\n【A】多指标汇总（test / test_ood）。MF = 不处理直接解调（匹配滤波基线）；SINR 越大越好，其余越小越好\n' + '=' * 78)
    for sp in ('test', 'test_ood'):
        d = D[sp][MAIN]
        print('\n%s' % sp)
        print('%-8s %8s %8s %9s %9s %10s %12s' % ('', 'BER', 'EVM', 'MSE', 'MAE', 'SINR(dB)', '带内SINR(dB)'))
        print('%-8s %8.4f %8.4f %9.4f %9.4f %10.2f %12.2f' % ('MF基线', d['ber_mf'].mean(), d['evm_mf'].mean(), d['mse_mf'].mean(),
                                                           d['mae_mf'].mean(), d['sinr_mf'].mean(), d['sinr_inband_mf'].mean()))
        for r, n in NAMES.items():
            x = D[sp][r]
            print('%-8s %8.4f %8.4f %9.4f %9.4f %10.2f %12.2f' % (n, x['ber_model'].mean(), x['evm_model'].mean(), x['mse_model'].mean(),
                                                                x['mae_model'].mean(), x['sinr_model'].mean(), x['sinr_inband_model'].mean()))

    d = ref
    # ---------- B 主模型 vs MF ----------
    print('\n' + '=' * 78 + '\n【B】主模型相对匹配滤波基线（test）。格子 = BER(MF) − BER(模型)；正值且 ▲ = 模型显著更好\n' + '=' * 78)
    T, M, S = d['type'], d['mod'], d['sir']
    grid('B1 干扰类型 × SIR', range(len(types)), SIRS, types, ['%+d' % s for s in SIRS], lambda r, c: (T == r) & (S == c), d['ber_mf'], d['ber_model'])
    grid('B2 调制 × SIR', range(len(mods)), SIRS, mods, ['%+d' % s for s in SIRS], lambda r, c: (M == r) & (S == c), d['ber_mf'], d['ber_model'])
    grid('B3 干扰类型 × 调制（对全部 SIR 平均）', range(len(types)), range(len(mods)), types, mods, lambda r, c: (T == r) & (M == c), d['ber_mf'], d['ber_model'])
    sn = d['snr']
    bins = [(10, 15), (15, 20), (20, 25), (25, 30.01)]
    grid('B4 带内 SNR 区间 × SIR', range(len(bins)), SIRS, ['SNR%d-%d' % (a, round(b)) for a, b in bins], ['%+d' % s for s in SIRS],
         lambda r, c: (sn >= bins[r][0]) & (sn < bins[r][1]) & (S == c), d['ber_mf'], d['ber_model'])

    # 同一场景下用别的指标看：带内 SINR 提升
    grid('B5 带内 SINR 提升 (dB) = 模型 − 不处理；干扰类型 × SIR（▲=模型把带内误差显著压得更小）', range(len(types)), SIRS, types,
         ['%+d' % s for s in SIRS], lambda r, c: (T == r) & (S == c), d['sinr_inband_model'], d['sinr_inband_mf'], fmt='%+.1f')

    # ---------- C 泛化 ----------
    print('\n' + '=' * 78 + '\n【C】未见干扰参数（test_ood）相对 MF：干扰类型 × SIR\n' + '=' * 78)
    o = D['test_ood'][MAIN]
    grid('C1 BER(MF) − BER(模型)', range(len(types)), SIRS, types, ['%+d' % s for s in SIRS], lambda r, c: (o['type'] == r) & (o['sir'] == c),
         o['ber_mf'], o['ber_model'])
    print('\n模型 test→test_ood 的 BER 变化（按类型）:')
    for i, t in enumerate(types):
        print('  %-4s %.3f → %.3f' % (t, d['ber_model'][T == i].mean(), o['ber_model'][o['type'] == i].mean()))

    # ---------- D MoE 的增益 ----------
    print('\n' + '=' * 78 + '\n【D】MoE 的增益：BER(无MoE) − BER(MoE)，正值且 ▲ = MoE 显著更好（test）\n' + '=' * 78)
    nm = D['test']['exp_nomoe']
    for r in ('exp_moe_legacy', 'exp_moe_prior'):
        x = D['test'][r]
        print('\n--- %s 相对 无MoE：全局 %+.4f %s' % (NAMES[r], *mark(nm['ber_model'] - x['ber_model'])))
        grid('按干扰类型 × SIR', range(len(types)), SIRS, types, ['%+d' % s for s in SIRS], lambda a, c: (T == a) & (S == c), nm['ber_model'], x['ber_model'])
        grid('按调制 × SIR', range(len(mods)), SIRS, mods, ['%+d' % s for s in SIRS], lambda a, c: (M == a) & (S == c), nm['ber_model'], x['ber_model'])
    x, y = D['test']['exp_moe_legacy'], D['test']['exp_moe_prior']
    print('\n--- 原MoE vs 新路由（BER(新路由) − BER(原MoE)，正=原MoE更好）：全局 %+.4f %s' % mark(y['ber_model'] - x['ber_model']))
    grid('按干扰类型 × SIR', range(len(types)), SIRS, types, ['%+d' % s for s in SIRS], lambda a, c: (T == a) & (S == c), y['ber_model'], x['ber_model'])

    # ---------- D2 不同指标下的模型排序 ----------
    print('\n' + '=' * 78 + '\n【D2】不同指标下，MoE 相对无MoE 是否更好（test，全局配对检验；正且▲=MoE更好）\n' + '=' * 78)
    for r in ('exp_moe_legacy', 'exp_moe_prior'):
        x = D['test'][r]
        print('\n%s 相对 无MoE:' % NAMES[r])
        for name, k, sign in (('BER', 'ber_model', 1), ('EVM', 'evm_model', 1), ('MSE', 'mse_model', 1), ('MAE', 'mae_model', 1),
                              ('SINR(dB)', 'sinr_model', -1), ('带内SINR(dB)', 'sinr_inband_model', -1)):
            diff = sign * (nm[k] - x[k])
            v, sym = mark(diff)
            print('  %-12s 改善量 %+.4f %s' % (name, v, sym))

    # ---------- E 优势区域总览 ----------
    print('\n' + '=' * 78 + '\n【E】总览：模型相对 MF 的相对 BER 降低（%），按 SIR 段；▲▼ 同上（test）\n' + '=' * 78)
    bands = {'低 SIR (-20~-10)': [-20, -15, -10], '中 SIR (-5~0)': [-5, 0], '高 SIR (+5~+10)': [5, 10]}
    print('%-18s' % '' + ''.join('%14s' % t for t in types))
    for bn, ss in bands.items():
        line = '%-18s' % bn
        for i in range(len(types)):
            m = (T == i) & np.isin(S, ss)
            _, s = mark(d['ber_mf'][m] - d['ber_model'][m])
            rel = 100 * (d['ber_mf'][m].mean() - d['ber_model'][m].mean()) / max(d['ber_mf'][m].mean(), 1e-9)
            line += ('%+.0f%%%s' % (rel, s)).rjust(14)
        print(line)


if __name__ == '__main__':
    main()
