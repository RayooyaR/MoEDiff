"""全部方法对比：传统 / 深度学习 / 前沿 / 本文方法。读取 evaluate_new.py 生成的逐样本结果。

用法（在 new/moe_diffic 下）：  python ../analysis/compare_methods.py [ours 的 run 名，默认 exp_moe_legacy]
置信区间：同一批样本上的配对差，均值差的正态近似 95% CI（只含测试样本抽样误差，不含训练种子波动）。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
OURS = sys.argv[1] if len(sys.argv) > 1 else 'exp_moe_legacy'
SIRS = [-20, -15, -10, -5, 0, 5, 10]

# (显示名, 类别, run 名; None 表示匹配滤波 = 取任一 run 的 *_mf 列)
METHODS = [
    ('MF 匹配滤波', '传统', None),
    ('LMMSE', '传统', 'exp_lmmse'),
    ('CNN', '深度学习', 'exp_cnn'),
    ('DiT扩散(ICDM,无MoE)', '深度学习', 'exp_nomoe'),
    ('DiffCom(DPS)', '前沿', 'exp_diffcom_1d'),
    ('ICDM双模型', '前沿', 'exp_icdm_1d'),
    ('MoE-DiffIC(原路由)', '本文', 'exp_moe_legacy'),
    ('MoE-DiffIC(新路由)', '本文', 'exp_moe_prior'),
]
TAG = {'dps': '', 'dual': ''}   # 如需使用调参后的结果文件，可在此填后缀


def load(run, split):
    f = os.path.join(RUNS, run, 'per_sample_%s.npz' % split)
    if not os.path.exists(f):
        return None
    d = np.load(f)
    return {k: d[k] for k in d.files}


def params_of(run):
    import torch
    if run is None:
        return 0
    p = os.path.join(RUNS, run, 'best_model.pth')
    if os.path.exists(p):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        return sum(v.numel() for v in ck['model_state_dict'].values() if v.dtype.is_floating_point)
    q = os.path.join(RUNS, run, 'lmmse.pt')
    return int(torch.load(q, map_location='cpu')['W'].numel()) if os.path.exists(q) else 0


def col(d, key, is_mf):
    return d[key + ('_mf' if is_mf else '_model')]


def main():
    data = {sp: {} for sp in ('test', 'test_ood')}
    ref = None
    for name, cat, run in METHODS:
        for sp in data:
            d = load(run, sp) if run else None
            if run is None:
                d = load(OURS, sp)          # 取任一 run 的 MF 列
            if d is not None:
                data[sp][name] = (d, run is None)
                ref = ref or d
    avail = [m for m in METHODS if m[0] in data['test']]
    miss = [m[0] for m in METHODS if m[0] not in data['test']]
    if miss:
        print('（尚无结果，已跳过: %s）\n' % ', '.join(miss))
    types, mods = list(ref['type_names']), list(ref['mod_names'])
    S, T, M = ref['sir'], ref['type'], ref['mod']
    nm = [m[0] for m in avail]

    print('=' * 100 + '\n【1】全局多指标（越小越好: BER/EVM/MSE/MAE；越大越好: SINR）\n' + '=' * 100)
    print('%-22s %-8s %8s | %7s %7s %8s %7s %9s %11s' % ('方法', '类别', '参数量', 'BER', 'EVM', 'MSE', 'MAE', 'SINR(dB)', '带内SINR(dB)'))
    for sp in ('test', 'test_ood'):
        print('--- %s' % sp)
        for name, cat, run in avail:
            d, mf = data[sp][name]
            pc = params_of(run)
            print('%-22s %-8s %8s | %7.4f %7.3f %8.3f %7.3f %9.2f %11.2f' % (
                name, cat, ('%.2fM' % (pc / 1e6)) if pc else '-', col(d, 'ber', mf).mean(), col(d, 'evm', mf).mean(),
                col(d, 'mse', mf).mean(), col(d, 'mae', mf).mean(), col(d, 'sinr', mf).mean(), col(d, 'sinr_inband', mf).mean()))

    def table(title, rows, rlab, sel, sp='test'):
        print('\n' + '=' * 100 + '\n' + title + '（BER，每列最低者标 *）\n' + '=' * 100)
        print('%-22s' % '' + ''.join('%9s' % r for r in rlab))
        vals = {n: [col(data[sp][n][0], 'ber', data[sp][n][1])[sel(r)].mean() for r in rows] for n in nm}
        best = [min(vals[n][i] for n in nm) for i in range(len(rows))]
        for n in nm:
            print('%-22s' % n + ''.join(('%.3f%s' % (v, '*' if abs(v - best[i]) < 1e-12 else ' ')).rjust(9) for i, v in enumerate(vals[n])))

    table('【2】按 SIR (dB), test', SIRS, ['%+d' % s for s in SIRS], lambda r: S == r)
    table('【3】按干扰类型, test', range(len(types)), types, lambda r: T == r)
    table('【4】按调制, test', range(len(mods)), mods, lambda r: M == r)
    table('【5】按 SIR (dB), test_ood（未见干扰参数）', SIRS, ['%+d' % s for s in SIRS], lambda r: data['test_ood'][nm[0]][0]['sir'] == r, sp='test_ood')
    o = data['test_ood'][nm[0]][0]
    table('【6】按干扰类型, test_ood', range(len(types)), types, lambda r: o['type'] == r, sp='test_ood')

    print('\n' + '=' * 100 + '\n【7】本文方法(%s) 相对各基线的 BER 改善（基线 − 本文，正=本文更好，▲/▼=显著）\n' % OURS + '=' * 100)
    bands = [('全部', list(SIRS)), ('低SIR -20~-10', [-20, -15, -10]), ('中SIR -5~0', [-5, 0]), ('高SIR +5~+10', [5, 10])]
    ours_name = [m[0] for m in METHODS if m[2] == OURS][0]
    od, om = data['test'][ours_name]
    print('%-22s' % '' + ''.join('%18s' % b[0] for b in bands) + '%18s' % 'test_ood 全部')
    for name in nm:
        if name == ours_name:
            continue
        line = '%-22s' % name
        for sp, bd in [('test', b) for b in bands] + [('test_ood', bands[0])]:
            d, mf = data[sp][name]
            dd, _ = data[sp][ours_name]
            sel = np.isin(dd['sir'], bd[1])
            diff = col(d, 'ber', mf)[sel] - col(dd, 'ber', False)[sel]
            m_, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
            sym = '▲' if m_ - 1.96 * se > 0 else ('▼' if m_ + 1.96 * se < 0 else '·')
            line += ('%+.4f%s' % (m_, sym)).rjust(18)
        print(line)


if __name__ == '__main__':
    main()
