"""实验 15：训练数据比例（10%/30%/100%）— BER 曲线与 OOD 增益。
run 命名约定：<前缀>_<模型>_p<比例>_s<种子>，如 de_tf_p10_s42。逐模型、逐比例汇总种子均值±标准差；
子集：全部 / 见过类型 / 未见类型（--unseen）。附 ΔBER（相对未恢复输入）与负增益比例。
用法: python data_efficiency.py --prefix de --models cnn dit tf moe --sizes 10 30 100 --seeds 42 43 --suffix _avg8 --unseen LFM NFM RFM [--png ../reports/data_eff.png]"""
import argparse
import os

import numpy as np

from aggregate_results import load, summarize, type_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='de')
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[10, 30, 100])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--suffix', default='_avg8')
    ap.add_argument('--split', default='test')
    ap.add_argument('--unseen', nargs='*', default=None)
    ap.add_argument('--png', default=None)
    a = ap.parse_args()
    table = {}
    for m in a.models:
        for p in a.sizes:
            for sd in a.seeds:
                d = load('%s_%s_p%d_s%d' % (a.prefix, m, p, sd), a.split, a.suffix) or load('%s_%s_p%d_s%d' % (a.prefix, m, p, sd), a.split, '')
                if d is None:
                    continue
                for sub, fn in (('all', None), ('seen', (lambda d_: type_mask(d_, exclude=a.unseen)) if a.unseen else None), ('unseen', (lambda d_: type_mask(d_, types=a.unseen)) if a.unseen else None)):
                    if fn is None and sub != 'all':
                        continue
                    table.setdefault((m, p, sub), []).append(summarize(d, fn(d) if fn else None))
    print('%-8s %-6s %-7s %4s %-16s %-16s %-14s' % ('模型', '比例%', '子集', '种子', 'BER 均值±std', 'ΔBER 均值±std', '负增益比例'))
    for (m, p, sub), L in sorted(table.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        b = [x['ber'] for x in L]; dl = [x['delta_ber'] for x in L]; ng = [x['neg_gain_ratio'] for x in L]
        print('%-8s %-6d %-7s %4d %.4f±%.4f   %+.4f±%.4f   %.3f' % (m, p, sub, len(L), np.mean(b), np.std(b, ddof=1) if len(b) > 1 else 0, np.mean(dl), np.std(dl, ddof=1) if len(dl) > 1 else 0, np.mean(ng)))
    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        subs = ['seen', 'unseen'] if a.unseen else ['all']
        fig, axs = plt.subplots(1, len(subs), figsize=(5.2 * len(subs), 4), squeeze=False)
        for ax, sub in zip(axs[0], subs):
            for m in a.models:
                xs = [p for p in a.sizes if (m, p, sub) in table]
                ax.errorbar(xs, [np.mean([x['ber'] for x in table[(m, p, sub)]]) for p in xs], yerr=[np.std([x['ber'] for x in table[(m, p, sub)]], ddof=1) if len(table[(m, p, sub)]) > 1 else 0 for p in xs], marker='o', label=m, capsize=3)
            ax.set_xlabel('训练数据比例 (%)'); ax.set_ylabel('BER'); ax.set_title(sub); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); os.makedirs(os.path.dirname(os.path.abspath(a.png)), exist_ok=True); fig.savefig(a.png, dpi=150)
        print('saved', a.png)


if __name__ == '__main__':
    main()
