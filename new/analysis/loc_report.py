"""专家无效原因定位 + E1 + remixys 的汇总（在 new/moe_diffic 下运行；2 个种子 42、43，对照 au_remix_p10_s{42,43}）。"""
import sys, numpy as np
sys.path.insert(0, '.')
from aggregate_results import load, type_mask
UN = ['LFM', 'NFM', 'RFM']; SEEN = ['ST', 'MT', 'NB', 'WB', 'PL']; S = (42, 43)
def row(pat):
    out = []
    for s in S:
        d, o = load(pat % s, 'test', '_avg8', 'runs'), load(pat % s, 'test_ood', '_avg8', 'runs')
        if d is None: return None
        sm = type_mask(d, exclude=UN); nw = type_mask(d, types=['ST', 'MT', 'NB', 'PL'])
        out.append([d['ber_model'][sm].mean(), d['ber_model'][nw].mean(), d['ber_model'][type_mask(d, types=UN)].mean(), o['ber_model'].mean() if o is not None else np.nan,
                    d['ber_model'][sm & (d['sir'] <= -10)].mean(), d['evm_model'][sm].mean(), d['sinr_inband_model'][sm].mean()] + [d['ber_model'][type_mask(d, types=[t])].mean() for t in SEEN])
    return np.array(out)
nm = ['见过', '非WB', '未见', 'OOD', '强干扰', 'EVM', 'SINR', 'ST', 'MT', 'NB', 'WB', 'PL']
cfg = [('无专家+remix(基线)', 'au_remix_p10_s%d'), ('局部 oracle+remix(上界)', 'om_p10_s%d'), ('E1 预测专家+remix', 'e1_p10_s%d'), ('remixys(仅用(y,s))', 'ry_p10_s%d')]
print('%-22s' % '', ' '.join('%-7s' % n for n in nm))
R = {}
for n, p in cfg:
    r = row(p)
    if r is None: print('%-22s 尚无结果' % n); continue
    R[n] = r; print('%-22s' % n, ' '.join(('%-7.4f' % v) if i != 6 else ('%-7.2f' % v) for i, v in enumerate(r.mean(0))), ' 各种子见过', np.round(r[:, 0], 4))
b = R.get('无专家+remix(基线)')
if b is not None:
    for n in R:
        if n != '无专家+remix(基线)': print('配对差(见过BER) %-20s' % n, np.round(R[n][:, 0] - b[:, 0], 4), ' 非WB', np.round(R[n][:, 1] - b[:, 1], 4))
