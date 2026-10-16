"""BER—SIR 曲线（含未恢复输入）。需要 matplotlib（系统 python3 有）。
用法: python3 plot_ber_by_sir.py --runs A:标签A B:标签B --suffix _avg8 --split test [--unseen LFM NFM RFM --subset seen|unseen] --out ../reports/ber_by_sir.png"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='run 名或 run名:标签')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--split', default='test')
    ap.add_argument('--unseen', nargs='*', default=None)
    ap.add_argument('--subset', choices=['all', 'seen', 'unseen'], default='all')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); drawn_in = False
    for spec in a.runs:
        run, _, lab = spec.partition(':')
        f = os.path.join(RUNS, run, 'per_sample_%s%s.npz' % (a.split, a.suffix))
        if not os.path.exists(f):
            continue
        d = np.load(f); names = [str(t) for t in d['type_names']]
        sel = np.ones(len(d['type']), bool)
        if a.unseen and a.subset != 'all':
            u = np.isin(d['type'], [i for i, t in enumerate(names) if t in a.unseen])
            sel = u if a.subset == 'unseen' else ~u
        sir = np.round(d['sir']).astype(int)
        xs = sorted(set(sir[sel].tolist()))
        ax.plot(xs, [d['ber_model'][sel & (sir == s)].mean() for s in xs], marker='o', label=lab or run)
        if not drawn_in:
            ax.plot(xs, [d['ber_mf'][sel & (sir == s)].mean() for s in xs], 'k--', marker='x', label='未恢复输入')
            drawn_in = True
    ax.set_xlabel('带内 SIR (dB)'); ax.set_ylabel('BER'); ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    fig.tight_layout(); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); fig.savefig(a.out, dpi=150)
    print('saved', a.out)


if __name__ == '__main__':
    main()
