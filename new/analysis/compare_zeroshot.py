"""留出整类干扰（zero-shot）泛化对比。读取 evaluate_new.py 在完整 v2 test 上的逐样本结果。

用法（在 new/moe_diffic 下）：  python ../analysis/compare_zeroshot.py
对每种留出方式 zs1 / zs2：训练只用"见过"的类型，测试在全部 8 类上，★ 标记训练时没见过的类型。
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
SPLITS = {'zs1': ['NB', 'WB', 'LFM', 'NFM', 'RFM', 'PL'], 'zs2': ['ST', 'MT', 'NB', 'WB', 'PL']}
REF = {'CNN': 'exp_cnn', 'MoE-DiffIC(原路由)': 'exp_moe_legacy', 'MoE-DiffIC(新路由)': 'exp_moe_prior'}


def load(run):
    f = os.path.join(RUNS, run, 'per_sample_test.npz')
    return {k: v for k, v in np.load(f).items()} if os.path.exists(f) else None


def main():
    ref = {n: load(r) for n, r in REF.items()}
    base = ref['CNN']
    types = list(base['type_names']); T = base['type']; S = base['sir']
    for zs, seen in SPLITS.items():
        runs = {'CNN': load(zs + '_cnn'), 'MoE-DiffIC(原路由)': load(zs + '_moe'), 'MoE-DiffIC(新路由)': load(zs + '_prior')}
        runs = {n: d for n, d in runs.items() if d is not None}
        if not runs:
            continue
        unseen = [t for t in types if t not in seen]
        print('=' * 100 + '\n留出方式 %s：训练类型 %s ；未见类型 %s\n' % (zs, seen, unseen) + '=' * 100)
        print('\n【1】逐类型 BER（左：本留出设置下训练；右：用全部类型训练的同一方法，作参照）')
        hdr = '%-6s %7s' % ('类型', 'MF') + ''.join('%14s' % n[:12] for n in runs) + ' |' + ''.join('%14s' % ('全类型' + n[:6]) for n in runs)
        print(hdr)
        for i, t in enumerate(types):
            m = T == i
            line = '%-6s %7.3f' % (t + ('★' if t in unseen else ''), base['ber_mf'][m].mean())
            line += ''.join('%14.3f' % d['ber_model'][m].mean() for d in runs.values()) + ' |'
            line += ''.join('%14.3f' % ref[n]['ber_model'][m].mean() for n in runs)
            print(line)
        mu = np.isin(T, [types.index(t) for t in unseen]); ms = ~mu
        print('\n【2】汇总（见过类型 / 未见类型 平均 BER）')
        print('%-18s %12s %12s %16s' % ('', '见过', '未见★', '未见：相对全类型训练的恶化'))
        print('%-18s %12.3f %12.3f' % ('MF', base['ber_mf'][ms].mean(), base['ber_mf'][mu].mean()))
        for n, d in runs.items():
            a, b, c = d['ber_model'][ms].mean(), d['ber_model'][mu].mean(), ref[n]['ber_model'][mu].mean()
            print('%-18s %12.3f %12.3f %16s' % (n, a, b, '%.3f→%.3f (×%.1f)' % (c, b, b / c)))
        print('\n【3】未见类型上的配对 BER 差（行方法 − CNN；负=该方法更好；▲▼显著）')
        if 'CNN' in runs:
            for n, d in runs.items():
                if n == 'CNN':
                    continue
                for scope, sel in [('未见类型合计', mu)] + [(t + '★', T == types.index(t)) for t in unseen]:
                    diff = d['ber_model'][sel] - runs['CNN']['ber_model'][sel]
                    mu_, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
                    sym = '▲' if mu_ - 1.96 * se > 0 else ('▼' if mu_ + 1.96 * se < 0 else '·')
                    print('  %-18s %-10s %+.4f%s   (CNN 更好=▲ 表示该方法更差)' % (n, scope, mu_, sym))
        print('\n【4】未见类型按 SIR 的 BER（MF / CNN / ' + ' / '.join(n for n in runs if n != 'CNN') + '）')
        for s in (-20, -10, 0, 10):
            m = mu & (S == s)
            print('  %+3d dB  ' % s + ' / '.join('%.3f' % x for x in [base['ber_mf'][m].mean()] + [d['ber_model'][m].mean() for d in runs.values()]))
        # 路由
        d = runs.get('MoE-DiffIC(新路由)')
        if d is not None and 'gates' in d:
            print('\n【5】新路由对未见类型的路由（平均权重 FD/MS/TF/TD；先验主专家吻合率）')
            from_prior = {'ST': 0, 'MT': 0, 'NB': 0, 'PL': 3, 'LFM': 2, 'NFM': 2, 'RFM': 2, 'WB': 1}
            for t in types:
                i = types.index(t); m = T == i
                hit = (d['gates'][m].argmax(1) == from_prior[t]).mean()
                print('  %-4s %s  先验主专家吻合 %.2f %s' % (t, np.round(d['gates'][m].mean(0), 2), hit, '★未见' if t in unseen else ''))


if __name__ == '__main__':
    main()
