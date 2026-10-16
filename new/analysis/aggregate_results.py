"""逐样本配对统计（plan.md §2.2）：BER、ΔBER、负增益比例、按 干扰类型 / 调制 / SIR 的细分，以及两个 run 之间的配对差与
按 clean_id 分组的 Bootstrap 置信区间。

读取 runs/<名>/per_sample_<split><后缀>.npz（evaluate_new.py 的输出；avg8 文件 = 测试集每隔一条的一半样本）。
定义：
  ΔBER_i      = BER_input_i − BER_restored_i          （>0 表示恢复有益）
  负增益比例   = 样本中 BER_restored_i > BER_input_i 的比例（把已正确的符号“修坏”的比例）
  配对差       = mean_i (BER_A_i − BER_B_i)，负=A 更好；CI 由按 clean_id 分组的 Bootstrap（默认 2000 次）给出
库用法: from aggregate_results import load, summarize, paired
命令行: python aggregate_results.py --split test --suffix _avg8 --unseen LFM NFM RFM --runs A B C --pairs A:B A:C
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')


def load(run, split='test', suffix='', runs_dir=RUNS):
    f = os.path.join(runs_dir, run, 'per_sample_%s%s.npz' % (split, suffix))
    if not os.path.exists(f):
        return None
    return {k: v for k, v in np.load(f).items()}


def align(d_a, d_b):
    """把两个 run 的逐样本数组按 clean_id 对齐（求交集），返回索引。"""
    ia = {int(c): i for i, c in enumerate(d_a['clean_id'])}
    ib = {int(c): i for i, c in enumerate(d_b['clean_id'])}
    common = sorted(set(ia) & set(ib))
    return np.array([ia[c] for c in common]), np.array([ib[c] for c in common])


def type_mask(d, types=None, exclude=None):
    names = [str(t) for t in d['type_names']]
    sel = np.ones(len(d['type']), dtype=bool)
    if types:
        sel &= np.isin(d['type'], [i for i, t in enumerate(names) if t in types])
    if exclude:
        sel &= ~np.isin(d['type'], [i for i, t in enumerate(names) if t in exclude])
    return sel


def summarize(d, sel=None):
    sel = np.ones(len(d['type']), dtype=bool) if sel is None else sel
    b, bi = d['ber_model'][sel], d['ber_mf'][sel]
    return {'n': int(sel.sum()), 'ber': float(b.mean()), 'ber_input': float(bi.mean()), 'delta_ber': float((bi - b).mean()),
            'neg_gain_ratio': float((b > bi + 1e-12).mean()), 'mse': float(d['mse_model'][sel].mean()), 'evm': float(d['evm_model'][sel].mean()),
            'sinr_inband': float(d['sinr_inband_model'][sel].mean())}


def breakdown(d, sel=None, by='type'):
    """by in type|mod|sir -> {名: summarize}"""
    sel = np.ones(len(d['type']), dtype=bool) if sel is None else sel
    names = {'type': [str(t) for t in d['type_names']], 'mod': [str(t) for t in d['mod_names']]}
    out = {}
    if by == 'sir':
        for s in sorted(set(np.round(d['sir']).astype(int).tolist())):
            out['%+d dB' % s] = summarize(d, sel & (np.round(d['sir']).astype(int) == s))
    else:
        arr = d[by]
        for i, n in enumerate(names[by]):
            m = sel & (arr == i)
            if m.any():
                out[n] = summarize(d, m)
    return out


def paired(d_a, d_b, sel_a=None, n_boot=2000, seed=0, key='ber_model'):
    """A − B 的配对差（按 clean_id 对齐），返回 (均值, ci_low, ci_high, n_clean_ids)。"""
    ia, ib = align(d_a, d_b)
    if sel_a is not None:
        keep = sel_a[ia]
        ia, ib = ia[keep], ib[keep]
    diff = d_a[key][ia] - d_b[key][ib]
    groups = d_a['clean_id'][ia]
    uniq, inv = np.unique(groups, return_inverse=True)
    gsum = np.bincount(inv, weights=diff); gcnt = np.bincount(inv)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        means.append(gsum[pick].sum() / gcnt[pick].sum())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), len(uniq)


def fmt_pair(res):
    m, lo, hi, n = res
    sig = '▼' if hi < 0 else ('▲' if lo > 0 else '·')
    return '%+.4f [%+.4f,%+.4f]%s' % (m, lo, hi, sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--unseen', nargs='*', default=None, help='未见类型（其余视为见过）；不给则不分')
    ap.add_argument('--pairs', nargs='*', default=[], help='A:B，输出 A−B 的配对差')
    ap.add_argument('--breakdown', choices=['type', 'mod', 'sir'], default=None)
    a = ap.parse_args()
    D = {r: load(r, a.split, a.suffix) for r in a.runs}
    groups = [('全部', None)]
    if a.unseen:
        d0 = next(d for d in D.values() if d is not None)
        groups += [('见过类型', lambda d: type_mask(d, exclude=a.unseen)), ('未见类型', lambda d: type_mask(d, types=a.unseen))]
    print('%-22s %-9s %6s %8s %8s %8s %9s %8s' % ('run', '子集', 'n', 'BER', '输入BER', 'ΔBER', '负增益比例', 'MSE'))
    for r, d in D.items():
        if d is None:
            print('%-22s 无数据' % r); continue
        for gn, fn in groups:
            s = summarize(d, fn(d) if fn else None)
            print('%-22s %-9s %6d %8.4f %8.4f %+8.4f %9.3f %8.3f' % (r, gn, s['n'], s['ber'], s['ber_input'], s['delta_ber'], s['neg_gain_ratio'], s['mse']))
        if a.breakdown:
            for k, v in breakdown(d, by=a.breakdown).items():
                print('    %-8s n=%4d BER=%.4f ΔBER=%+.4f 负增益=%.3f' % (k, v['n'], v['ber'], v['delta_ber'], v['neg_gain_ratio']))
    for p in a.pairs:
        x, y = p.split(':')
        if D.get(x) is None or D.get(y) is None:
            print('配对 %s:%s 缺数据' % (x, y)); continue
        for gn, fn in groups:
            print('配对差 %s − %s [%s]: %s' % (x, y, gn, fmt_pair(paired(D[x], D[y], fn(D[x]) if fn else None))))


if __name__ == '__main__':
    main()
