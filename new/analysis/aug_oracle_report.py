"""Oracle 条件诊断 + 数据增强筛选的汇总（在 new/moe_diffic 下运行）。只看测试集 A 的 5 类见过类型。"""
import sys, numpy as np
sys.path.insert(0, '.')
from aggregate_results import load, type_mask
SEEN = ['ST', 'MT', 'NB', 'WB', 'PL']; UN = ['LFM', 'NFM', 'RFM']
def row(runs):
    ds = [d for d in (load(r, 'test', '_avg8', 'runs') for r in runs) if d is not None]
    if not ds: return None
    sm = lambda d: type_mask(d, exclude=UN)
    pt = np.mean([[d['ber_model'][type_mask(d, types=[t])].mean() for t in SEEN] for d in ds], 0)
    seen = [d['ber_model'][sm(d)].mean() for d in ds]
    lo = np.mean([d['ber_model'][sm(d) & (d['sir'] <= -10)].mean() for d in ds])
    return dict(n=len(ds), seen=np.mean(seen), sd=(np.std(seen, ddof=1) if len(seen) > 1 else float('nan')), lo=lo, evm=np.mean([d['evm_model'][sm(d)].mean() for d in ds]),
                sinr=np.mean([d['sinr_inband_model'][sm(d)].mean() for d in ds]), pt=pt, per=seen)
def show(name, runs):
    r = row(runs)
    if r is None: print('%-28s 尚无结果' % name); return
    print('%-28s n=%d seen=%.4f(±%.4f) 强干扰≤-10dB=%.4f EVM=%.3f SINR=%.2f | ST/MT/NB/WB/PL=%s' % (name, r['n'], r['seen'], r['sd'], r['lo'], r['evm'], r['sinr'], np.round(r['pt'], 3)))
S3 = (42, 43, 44)
print('## Oracle 条件诊断（3 种子）')
show('无专家 U-Net (un0)', ['un0_p10_s%d' % s for s in S3]); show('U-Net + TF 专家 (J0)', ['j0_p10_s%d' % s for s in S3])
show('U-Net + oracle(类型+12属性)', ['oracle_p10_s%d' % s for s in S3])
print('## 数据增强筛选（种子 42；对照 un0_p10_s42）')
show('无增强 un0_s42', ['un0_p10_s42'])
for t in ('rot', 'shift', 'awgn', 'iqgain', 'remix', 'rotremix'): show('增强 ' + t, ['au_%s_p10_s42' % t])
