"""留出整类干扰泛化对比（第二版）：含确定性采样、多种子、四种留出方式。

用法（在 new/moe_diffic 下）： python ../analysis/compare_zeroshot2.py
留出方式：zs1 去掉 ST/MT；zs2 去掉 LFM/NFM/RFM；zs3 去掉 PL；zs4 去掉 NB/WB。★ 标记训练时没见过的类型。
文件命名：runs/zsK_{cnn,moe,prior}[_s43|_s44]/per_sample_test[_det].npz（_det 为确定性采样）。
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
ALL = ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM']
SEEN = {'zs1': ['NB', 'WB', 'LFM', 'NFM', 'RFM', 'PL'], 'zs2': ['ST', 'MT', 'NB', 'WB', 'PL'],
        'zs3': ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM'], 'zs4': ['ST', 'MT', 'PL', 'LFM', 'NFM', 'RFM']}
SEEDS = ['', '_s43', '_s44']


def load(run, suf=''):
    f = os.path.join(RUNS, run, 'per_sample_test%s.npz' % suf)
    return {k: v for k, v in np.load(f).items()} if os.path.exists(f) else None


def seeds(zs, kind, suf):
    return [d for d in (load('%s_%s%s' % (zs, kind, s), suf) for s in SEEDS) if d is not None]


def main():
    ref = {'CNN': load('exp_cnn'), 'MoE 确定': load('exp_moe_legacy', '_det'), 'Prior 确定': load('exp_moe_prior', '_det')}
    base = ref['CNN']
    types = list(base['type_names']); T = base['type']; S = base['sir']
    for zs, seen in SEEN.items():
        meth = {'CNN': seeds(zs, 'cnn', ''), 'MoE 随机': seeds(zs, 'moe', ''), 'MoE 确定': seeds(zs, 'moe', '_det'),
                'Prior 确定': seeds(zs, 'prior', '_det')}
        meth = {k: v for k, v in meth.items() if v}
        if not meth:
            continue
        unseen = [t for t in types if t not in seen]
        mu = np.isin(T, [types.index(t) for t in unseen]); ms = ~mu
        print('=' * 104 + '\n%s：训练类型 %s ；未见★ %s\n' % (zs, seen, [str(u) for u in unseen]) + '=' * 104)
        print('\n【1】逐类型 BER（各方法为该设置下训练；多种子取平均）；右侧为全类型训练的参照')
        print('%-6s %6s' % ('类型', 'MF') + ''.join('%11s' % k for k in meth) + ' |' + ''.join('%11s' % ('全:' + k[:5]) for k in ('CNN', 'MoE 确定')))
        for i, t in enumerate(types):
            m = T == i
            avg = lambda ds: np.mean([d['ber_model'][m].mean() for d in ds])
            print('%-6s %6.3f' % (str(t) + ('★' if t in unseen else ''), base['ber_mf'][m].mean()) + ''.join('%11.3f' % avg(v) for v in meth.values())
                  + ' |' + ''.join('%11.3f' % ref[k]['ber_model'][m].mean() for k in ('CNN', 'MoE 确定')))
        print('\n【2】汇总：见过 / 未见★ 的平均 BER（多种子：均值 ±标准差，括号内为种子数）；未见类型相对全类型训练的恶化倍数')
        print('%-12s %18s %24s %s' % ('', '见过', '未见★', '恶化倍数'))
        print('%-12s %18.3f %24.3f' % ('MF', base['ber_mf'][ms].mean(), base['ber_mf'][mu].mean()))
        for k, ds in meth.items():
            a = np.array([d['ber_model'][ms].mean() for d in ds]); b = np.array([d['ber_model'][mu].mean() for d in ds])
            r = ref.get(k, ref['MoE 确定'] if k.startswith('MoE') else ref['CNN'])['ber_model'][mu].mean()
            print('%-12s %18s %24s   ×%.1f (%d)' % (k, '%.3f±%.3f' % (a.mean(), a.std()), '%.3f±%.3f' % (b.mean(), b.std()), b.mean() / r, len(ds)))
        if 'CNN' in meth:
            print('\n【3】未见类型上，各 MoE 变体相对 CNN 的 BER 差（>0 表示该变体更差；逐种子配对，合计未见）')
            for k, ds in meth.items():
                if k == 'CNN':
                    continue
                row = []
                for j, d in enumerate(ds):
                    c = meth['CNN'][j] if j < len(meth['CNN']) else meth['CNN'][0]
                    diff = d['ber_model'][mu] - c['ber_model'][mu]
                    m_, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
                    row.append('%+.4f%s' % (m_, '▲' if m_ - 1.96 * se > 0 else ('▼' if m_ + 1.96 * se < 0 else '·')))
                print('  %-12s 逐种子: %s   （▼=该变体显著优于 CNN）' % (k, '  '.join(row)))
            print('\n【4】未见类型按 SIR 的 BER（种子平均）：MF / ' + ' / '.join(meth))
            for s in (-20, -10, 0, 10):
                m = mu & (S == s)
                print('  %+3d dB  ' % s + ' / '.join('%.3f' % x for x in [base['ber_mf'][m].mean()] + [np.mean([d['ber_model'][m].mean() for d in ds]) for ds in meth.values()]))
    print('\n注：★ 未见类型的路由分析（新路由）仍见 compare_zeroshot.py。')


if __name__ == '__main__':
    main()
