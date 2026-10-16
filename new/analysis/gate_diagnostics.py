"""专家与路由诊断：从 per_sample_*.npz 里的 gates（以及双因素路由的 sir_est）统计
  - 专家被选中频率（门控>1e-6 的比例）与平均权重、门控熵
  - 门控方差被 干扰类型 / SIR / 调制 解释的比例（η²）；由门控判断“干扰类型”“SIR≥0”的线性可分程度（最近均值分类）
  - 同一类型随 SIR 变化的门控（固定类型）与固定 SIR 下不同类型的门控（固定 SIR）
  - 双因素路由：SIR 估计的偏差/MAE（按真实 SIR 分档）
用法: python gate_diagnostics.py --run zs2_moe [--suffix _avg8] [--experts FD MS TF TD]"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')


def eta2(G, labels):
    tot = ((G - G.mean(0)) ** 2).sum()
    between = 0.0
    for l in np.unique(labels):
        m = labels == l
        between += m.sum() * ((G[m].mean(0) - G.mean(0)) ** 2).sum()
    return float(between / (tot + 1e-12))


def nearest_mean_acc(G, labels):
    cls = np.unique(labels)
    mu = np.stack([G[labels == c].mean(0) for c in cls])
    pred = cls[np.argmin(((G[:, None] - mu[None]) ** 2).sum(-1), 1)]
    return float((pred == labels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--experts', nargs='*', default=None)
    a = ap.parse_args()
    d = np.load(os.path.join(RUNS, a.run, 'per_sample_%s%s.npz' % (a.split, a.suffix)))
    if 'gates' not in d:
        print('该文件没有 gates（模型无 MoE，或评估时未记录）'); return
    G = d['gates'].astype(np.float64); n, K = G.shape
    names = a.experts or ['E%d' % i for i in range(K)]
    types = [str(t) for t in d['type_names']]
    sir = np.round(d['sir']).astype(int)
    print('样本数 %d，专家数 %d' % (n, K))
    print('平均门控      : ' + '  '.join('%s=%.3f' % (nm, v) for nm, v in zip(names, G.mean(0))))
    print('被选中频率    : ' + '  '.join('%s=%.2f' % (nm, v) for nm, v in zip(names, (G > 1e-6).mean(0))))
    P = G / (G.sum(1, keepdims=True) + 1e-12)
    print('门控熵(均值)  : %.3f  (最大 %.3f)' % (float((-(P * np.log(P + 1e-12)).sum(1)).mean()), np.log(K)))
    print('门控方差解释比例 η²：干扰类型 %.3f | SIR %.3f | 调制 %.3f' % (eta2(G, d['type']), eta2(G, sir), eta2(G, d['mod'])))
    print('最近均值分类：由门控判干扰类型 acc=%.3f（随机 %.3f）；判 SIR≥0dB acc=%.3f' % (nearest_mean_acc(G, d['type']), 1.0 / len(np.unique(d['type'])), nearest_mean_acc(G, (sir >= 0).astype(int))))
    print('\n固定类型随 SIR 变化的平均门控（行=类型，列=SIR 低→高 的各专家）：')
    for ti, t in enumerate(types):
        m = d['type'] == ti
        if m.any():
            print('  %-4s ' % t + ' | '.join(' '.join('%.2f' % v for v in G[m & (sir == s)].mean(0)) for s in sorted(set(sir.tolist())) if (m & (sir == s)).any()))
    print('\n固定 SIR 下各类型的平均门控 (SIR=0 dB)：')
    for ti, t in enumerate(types):
        m = (d['type'] == ti) & (sir == 0)
        if m.any():
            print('  %-4s ' % t + ' '.join('%s=%.2f' % (nm, v) for nm, v in zip(names, G[m].mean(0))))
    if 'sir_est' in d:
        e = d['sir_est'] - d['sir']
        print('\n双因素路由的 SIR 估计：总体 MAE=%.2f dB，偏差=%+.2f dB' % (np.abs(e).mean(), e.mean()))
        for s in sorted(set(sir.tolist())):
            m = sir == s
            print('  真实 %+3d dB: 估计均值 %+6.2f  MAE %.2f' % (s, d['sir_est'][m].mean(), np.abs(e[m]).mean()))


if __name__ == '__main__':
    main()
