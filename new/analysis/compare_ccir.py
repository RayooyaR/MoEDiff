"""CCIR 场景（异速率同信道数字干扰）的方法对比。读取 evaluate_new.py / diag_sampling.py 的输出。

用法（在 new/moe_diffic 下）：  python ../analysis/compare_ccir.py [diag 用的 stride，默认 4]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SIRS = [-20, -15, -10, -5, 0, 5, 10]
M = [('MF 匹配滤波', None), ('LMMSE', 'hardb_lmmse'), ('CNN', 'hardb_cnn'), ('DiT扩散(无MoE)', 'ccir_nomoe'),
     ('DiffCom(DPS)', 'ccir_diffcom'), ('ICDM双模型', 'ccir_icdm'), ('MoE-DiffIC', 'ccir_moe')]


def load(run):
    if run is None:
        return None
    f = os.path.join(RUNS, run, 'per_sample_test.npz')
    return {k: v for k, v in np.load(f).items()} if os.path.exists(f) else None


def main():
    D = {n: (load(r), r is None) for n, r in M if r is None or load(r) is not None}
    ref = D['CNN'][0]
    # MF 列取自任一可用 run
    D['MF 匹配滤波'] = (ref, True)
    names = [n for n, _ in M if n in D]
    print('缺少结果的方法:', [n for n, _ in M if n not in D], '\n')
    col = lambda d, k, mf: d[k + ('_mf' if mf else '_model')]
    S, Mo, mods = ref['sir'], ref['mod'], list(ref['mod_names'])
    print('=' * 96 + '\n【1】全局（CCIR, test）\n' + '=' * 96)
    print('%-16s %8s %8s %9s %8s %10s' % ('方法', 'BER', 'EVM', 'MSE', 'MAE', '带内SINR(dB)'))
    for n in names:
        d, mf = D[n]
        print('%-16s %8.4f %8.3f %9.3f %8.3f %10.2f' % (n, col(d, 'ber', mf).mean(), col(d, 'evm', mf).mean(), col(d, 'mse', mf).mean(),
                                                          col(d, 'mae', mf).mean(), col(d, 'sinr_inband', mf).mean()))
    for title, keys, lab, sel in (('【2】按 SIR (dB)', SIRS, ['%+d' % s for s in SIRS], lambda k: S == k),
                                  ('【3】按（目标信号的）调制', range(len(mods)), mods, lambda k: Mo == k)):
        print('\n' + '=' * 96 + '\n' + title + '：BER，每列最低标 *\n' + '=' * 96)
        print('%-16s' % '' + ''.join('%9s' % x for x in lab))
        vals = {n: [col(D[n][0], 'ber', D[n][1])[sel(k)].mean() for k in keys] for n in names}
        best = [min(vals[n][i] for n in names) for i in range(len(list(keys)))]
        for n in names:
            print('%-16s' % n + ''.join(('%.3f%s' % (v, '*' if abs(v - best[i]) < 1e-12 else ' ')).rjust(9) for i, v in enumerate(vals[n])))

    print('\n' + '=' * 96 + '\n【4】MoE-DiffIC 相对各方法的配对 BER 改善（对方 − MoE-DiffIC；正=MoE-DiffIC 更好；▲▼=显著）\n' + '=' * 96)
    if 'MoE-DiffIC' in D:
        o = D['MoE-DiffIC'][0]['ber_model']
        bands = [('全部', SIRS), ('低SIR', [-20, -15, -10]), ('中SIR', [-5, 0]), ('高SIR', [5, 10])]
        print('%-16s' % '' + ''.join('%14s' % b[0] for b in bands))
        for n in names:
            if n == 'MoE-DiffIC':
                continue
            line = '%-16s' % n
            for _, ss in bands:
                m = np.isin(S, ss); d = col(D[n][0], 'ber', D[n][1])[m] - o[m]
                mu, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
                line += ('%+.4f%s' % (mu, '▲' if mu - 1.96 * se > 0 else ('▼' if mu + 1.96 * se < 0 else '·'))).rjust(14)
            print(line)

    print('\n' + '=' * 96 + '\n【5】多次采样平均（K=16）的诊断：单次 → 均值，与 CNN 在同一子集上的对比\n' + '=' * 96)
    for n, r in (('DiT扩散(无MoE)', 'ccir_nomoe'), ('MoE-DiffIC', 'ccir_moe')):
        f = os.path.join(RUNS, r, 'diag_sampling_test.npz')
        if not os.path.exists(f):
            continue
        d = np.load(f); sl = slice(None, None, STRIDE); c = D['CNN'][0]
        print('%-16s K=1: BER %.4f MSE %.4f | K=16: BER %.4f MSE %.4f | CNN: BER %.4f MSE %.4f | 样本间方差 %.4f' % (
            n, d['ber_k1'].mean(), d['mse_k1'].mean(), d['ber_k16'].mean(), d['mse_k16'].mean(),
            c['ber_model'][sl].mean(), c['mse_model'][sl].mean(), d['var'].mean()))


if __name__ == '__main__':
    main()
