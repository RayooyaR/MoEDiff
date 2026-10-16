"""v4_digital（信号型干扰）：基线 / TF / oracle / TF+oracle 的 2 种子配对比较（在 new/moe_diffic 下运行）。
同时给出：每个种子的总 BER、均值±标准差、逐种子配对差、逐类型方向一致性，以及以测试样本为单位的配对 bootstrap 95% 置信区间
（先把同一测试样本在 2 个种子上的差值取平均，再对样本重抽样；这只反映测试样本的随机性，种子间波动由"逐种子配对差"体现）。"""
import sys, numpy as np
sys.path.insert(0, '.')
from aggregate_results import load, type_mask
S = (42, 43)
cfg = [('基线(无专家+remix)', 'v4b_p10_s%d'), ('TF 专家', 'v4t_p10_s%d'), ('oracle(类型+属性)', 'v4o_p10_s%d'), ('TF 专家 + oracle', 'v4to_p10_s%d')]
D = {}
for n, p in cfg:
    ds = [load(p % s, 'test', '_avg8', 'runs') for s in S]
    D[n] = ds if all(d is not None for d in ds) else None
base = D['基线(无专家+remix)']
tn = [str(t) for t in base[0]['type_names']] if base else []
print('%-18s %-30s %-10s %s' % ('配置', '各种子总BER', '均值±sd', '各类型均值 ' + '/'.join(tn)))
for n, _ in cfg:
    ds = D[n]
    if ds is None: print('%-18s 尚无结果' % n); continue
    b = [d['ber_model'].mean() for d in ds]
    print('%-18s %-30s %.4f±%.4f %s' % (n, np.round(b, 4), np.mean(b), np.std(b, ddof=1), np.round([np.mean([d['ber_model'][type_mask(d, types=[t])].mean() for d in ds]) for t in tn], 3)))
rng = np.random.default_rng(0)
print('\n配对比较（相对基线；负=更好）')
for n, _ in cfg[1:]:
    ds = D[n]
    if ds is None or base is None: continue
    pairs = [x['ber_model'].mean() - y['ber_model'].mean() for x, y in zip(ds, base)]
    per_type = [np.mean([x['ber_model'][type_mask(x, types=[t])].mean() - y['ber_model'][type_mask(y, types=[t])].mean() for x, y in zip(ds, base)]) for t in tn]
    diff = np.mean([x['ber_model'] - y['ber_model'] for x, y in zip(ds, base)], 0)            # 每个测试样本的 2 种子平均差
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    rel = 100 * np.mean(pairs) / np.mean([y['ber_model'].mean() for y in base])
    print('%-18s 逐种子差=%s 均值 %+.4f(%.1f%%) 种子间sd=%.4f 类型方向 %d/%d 为负；样本bootstrap 95%%CI [%+.4f, %+.4f]' % (n, np.round(pairs, 4), np.mean(pairs), rel, np.std(pairs, ddof=1), sum(v < 0 for v in per_type), len(tn), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
