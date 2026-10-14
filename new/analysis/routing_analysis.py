"""MoE（原路由）学到的路由策略分析：读取各 MoE 模型在测试集上记录的门控权重（per_sample_test_det.npz 里的 gates）。

专家顺序：FD 频域 / MS 多尺度 / TF 时频 / TD 时域。
输出：按干扰类型 / SIR / 调制 的平均权重、占优专家分布、门控熵与有效专家数、
      路由对"干扰类型"的信息量（最近质心分类准确率、占优专家与类型的互信息）、各因素对门控方差的解释比例。
用法（在 new/moe_diffic 下）： python ../analysis/routing_analysis.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
EX = ['FD', 'MS', 'TF', 'TD']
SEEN = {'zs1': ['NB', 'WB', 'LFM', 'NFM', 'RFM', 'PL'], 'zs2': ['ST', 'MT', 'NB', 'WB', 'PL'], 'zs3': ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM']}


def load(run, f='per_sample_test_det.npz'):
    p = os.path.join(RUNS, run, f)
    return {k: v for k, v in np.load(p).items()} if os.path.exists(p) else None


def eta2(G, groups):
    """各因素解释的门控方差比例（组间平方和 / 总平方和，对 4 个专家求和）。"""
    tot = ((G - G.mean(0)) ** 2).sum()
    ss = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss += m.sum() * ((G[m].mean(0) - G.mean(0)) ** 2).sum()
    return ss / tot


def mutual_info(a, b):
    ua, ub = np.unique(a), np.unique(b)
    p = np.array([[np.mean((a == x) & (b == y)) for y in ub] for x in ua])
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    return mi, ha


def centroid_acc(G, y, rng):
    idx = rng.permutation(len(y)); half = len(y) // 2
    tr, te = idx[:half], idx[half:]
    cls = np.unique(y)
    C = np.stack([G[tr][y[tr] == c].mean(0) for c in cls])
    pred = cls[np.argmin(((G[te][:, None] - C[None]) ** 2).sum(-1), 1)]
    return (pred == y[te]).mean(), 1.0 / len(cls)


def report(title, d, names_key='type_names', seen=None):
    G, T = d['gates'], d['type']
    types = [str(t) for t in d[names_key]]
    present = sorted(set(T.tolist()))
    print('\n' + '=' * 100 + '\n' + title + '\n' + '=' * 100)
    ent = -(np.clip(G, 1e-8, 1) * np.log(np.clip(G, 1e-8, 1))).sum(1)
    print('整体平均权重 %s ；门控熵均值 %.3f（4 专家均匀=1.386）；有效专家数 %.2f；占优专家权重均值 %.2f'
          % (dict(zip(EX, G.mean(0).round(3))), ent.mean(), np.exp(ent).mean(), G.max(1).mean()))
    print('\n[按干扰类型] 平均权重 FD/MS/TF/TD | 占优专家占比 FD/MS/TF/TD')
    cent = {}
    for t in present:
        m = T == t
        cent[t] = G[m].mean(0)
        dom = np.bincount(G[m].argmax(1), minlength=4) / m.sum()
        tag = ''
        if seen is not None:
            tag = '  [见过]' if types[t] in seen else '  [未见★]'
        print('  %-6s %s | %s%s' % (types[t], ' '.join('%.2f' % x for x in G[m].mean(0)), ' '.join('%.2f' % x for x in dom), tag))
    L1 = max(np.abs(cent[a] - cent[b]).sum() for a in present for b in present)
    print('  类型间平均门控向量的最大 L1 距离 %.3f（0=完全相同，2=完全不同）' % L1)
    print('\n[按 SIR] 平均权重 FD/MS/TF/TD')
    S = d['sir']
    for s in sorted(set(S.tolist())):
        print('  %+3d dB  %s' % (s, ' '.join('%.2f' % x for x in G[S == s].mean(0))))
    print('\n[按调制] ' + '  '.join('%s: %s' % (str(n), ' '.join('%.2f' % x for x in G[d['mod'] == i].mean(0))) for i, n in enumerate(d['mod_names'])))
    rng = np.random.default_rng(0)
    acc, chance = centroid_acc(G, T, rng)
    mi, h = mutual_info(G.argmax(1), T)
    sir_acc, sir_chance = centroid_acc(G, (S >= 0).astype(int), rng)
    print('\n[路由携带了什么信息]')
    print('  由门控向量判断"干扰类型"的准确率 %.1f%%（随机 %.1f%%）；占优专家与类型的互信息占类型熵 %.1f%%' % (100 * acc, 100 * chance, 100 * mi / h))
    print('  由门控向量判断"信干比是否≥0 dB"的准确率 %.1f%%（随机 50%%）' % (100 * sir_acc))
    print('  门控方差被各因素解释的比例：干扰类型 %.1f%% | 信干比 %.1f%% | 调制 %.1f%%'
          % (100 * eta2(G, T), 100 * eta2(G, S), 100 * eta2(G, d['mod'])))
    return cent


def main():
    d = load('exp_moe_legacy')
    report('A. 标准 8 类数据，全类型训练（原路由 MoE，exp_moe_legacy，test 6720 条）', d)
    for z in ('zs2', 'zs1', 'zs3'):
        d = load(z + '_moe')
        types = [str(t) for t in d['type_names']]
        cent = report('B. 留出整类 %s（训练只见过 %s；★=未见）' % (z, SEEN[z]), d, seen=SEEN[z])
        # 未见类型的路由与哪个见过类型最相似
        un = [i for i, t in enumerate(types) if t not in SEEN[z]]
        sn = [i for i, t in enumerate(types) if t in SEEN[z]]
        print('\n[未见类型的路由与哪个"见过类型"最相似（门控向量 L1 距离）]')
        for u in un:
            best = min(sn, key=lambda s: np.abs(cent[u] - cent[s]).sum())
            print('  %-4s → 最像 %-4s (L1=%.3f)' % (types[u], types[best], np.abs(cent[u] - cent[best]).sum()))
    d = load('ef_moe')
    report('C. 更难场景 v5（干扰叠加随机效应，ef_moe，test）', d)
    d = load('v4_moe_all')
    if d is not None:
        report('D. 数字干扰家族（CCIR/FSK/OFDM/BURST/ACI，v4_moe_all，test）', d)
    # 种子之间路由是否稳定（zs2 的三个种子）
    print('\n' + '=' * 100 + '\nE. 路由的稳定性：zs2 的三个训练种子，各干扰类型的平均权重\n' + '=' * 100)
    for run in ('zs2_moe', 'zs2_moe_s43', 'zs2_moe_s44'):
        d = load(run)
        types = [str(t) for t in d['type_names']]
        print(run)
        for t in sorted(set(d['type'].tolist())):
            print('   %-4s %s' % (types[t], ' '.join('%.2f' % x for x in d['gates'][d['type'] == t].mean(0))))


if __name__ == '__main__':
    main()
