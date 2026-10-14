"""同信道数字干扰家族（v4_digital: CCIR/FSK/OFDM/BURST/ACI）：全类型训练对比 + 留出整个子类的泛化。

用法（在 new/moe_diffic 下）： python ../analysis/compare_digital.py
runs: v4_{moe,cnn}_all（全类型）；v4_hf_*（留出 FSK）；v4_ho_*（留出 OFDM）；v4_lmmse。_det 后缀为确定性采样。
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
SIRS = [-20, -15, -10, -5, 0, 5, 10]


def load(run, suf=''):
    f = os.path.join(RUNS, run, 'per_sample_test%s.npz' % suf)
    return {k: v for k, v in np.load(f).items()} if os.path.exists(f) else None


def sig(a, b):
    d = a - b; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return '%+.4f%s' % (m, '▲' if m - 1.96 * se > 0 else ('▼' if m + 1.96 * se < 0 else '·'))


def main():
    M = {'MF': None, 'LMMSE': load('v4_lmmse'), 'CNN': load('v4_cnn_all'), 'MoE 随机': load('v4_moe_all'), 'MoE 确定': load('v4_moe_all', '_det')}
    ref = M['CNN'] or M['MoE 确定'] or M['MoE 随机']
    if ref is None:
        print('尚无结果'); return
    types = [str(t) for t in ref['type_names']]; T, S, Mo = ref['type'], ref['sir'], ref['mod']; mods = [str(m) for m in ref['mod_names']]
    print('=' * 100 + '\n【1】全类型训练：test 全局与分解（BER）\n' + '=' * 100)
    rows = {'MF': ref['ber_mf']}
    rows.update({k: v['ber_model'] for k, v in M.items() if v is not None and k != 'MF'})
    print('%-10s %8s' % ('', '全部') + ''.join('%9s' % t for t in types) + ''.join('%8s' % m for m in mods))
    for k, b in rows.items():
        print('%-10s %8.4f' % (k, b.mean()) + ''.join('%9.3f' % b[T == i].mean() for i in range(len(types))) + ''.join('%8.3f' % b[Mo == i].mean() for i in range(len(mods))))
    print('\n按 SIR：' + ' '.join('%7g' % s for s in SIRS))
    for k, b in rows.items():
        print('%-10s ' % k + ' '.join('%7.3f' % b[S == s].mean() for s in SIRS))
    if M['CNN'] is not None and M['MoE 确定'] is not None:
        print('\nMoE(确定) − CNN（全类型, 配对）: 全部 %s | 低SIR %s | 高SIR %s' % (
            sig(M['MoE 确定']['ber_model'], M['CNN']['ber_model']),
            sig(M['MoE 确定']['ber_model'][S <= -10], M['CNN']['ber_model'][S <= -10]),
            sig(M['MoE 确定']['ber_model'][S >= 5], M['CNN']['ber_model'][S >= 5])))

    for tag, unseen_name, label in (('hf', 'FSK', '留出 FSK'), ('ho', 'OFDM', '留出 OFDM')):
        cnn, moe, moe_d = load('v4_%s_cnn' % tag), load('v4_%s_moe' % tag), load('v4_%s_moe' % tag, '_det')
        if cnn is None and moe is None:
            continue
        ui = types.index(unseen_name)
        print('\n' + '=' * 100 + '\n【2】%s（训练只用其余 4 个子类；★=未见）：test BER\n' % label + '=' * 100)
        print('%-14s %6s' % ('', 'MF') + ''.join('%9s' % (t + ('★' if i == ui else '')) for i, t in enumerate(types)) + '%9s' % '全部')
        lines = [('MF', ref['ber_mf'])]
        for nm, d in (('CNN(留出)', cnn), ('MoE 随机(留出)', moe), ('MoE 确定(留出)', moe_d), ('CNN(全类型)', M['CNN']), ('MoE 确定(全类型)', M['MoE 确定'])):
            if d is not None:
                lines.append((nm, d['ber_model']))
        for nm, b in lines:
            print('%-14s' % nm + '       ' + ''.join('%9.3f' % b[T == i].mean() for i in range(len(types))) + '%9.3f' % b.mean())
        m = T == ui
        print('\n未见子类 %s 上的配对 BER 差：' % unseen_name)
        for nm, d in (('MoE 随机', moe), ('MoE 确定', moe_d)):
            if d is not None and cnn is not None:
                print('  %-10s − CNN(留出): %s   （▼=MoE 显著更好）' % (nm, sig(d['ber_model'][m], cnn['ber_model'][m])))
        print('未见子类按 SIR（MF / CNN / MoE随机 / MoE确定）:')
        for s in (-20, -10, 0, 10):
            mm = m & (S == s)
            print('  %+3d dB  ' % s + ' / '.join('%.3f' % d[mm].mean() for d in [ref['ber_mf']] + [x['ber_model'] for x in (cnn, moe, moe_d) if x is not None]))
        if cnn is not None and M['CNN'] is not None:
            print('恶化倍数（未见子类，相对全类型训练）: CNN ×%.1f%s' % (cnn['ber_model'][m].mean() / M['CNN']['ber_model'][m].mean(),
                  ('；MoE 确定 ×%.1f' % (moe_d['ber_model'][m].mean() / M['MoE 确定']['ber_model'][m].mean())) if moe_d is not None and M['MoE 确定'] is not None else ''))


if __name__ == '__main__':
    main()
