"""测试 1（v4_digital 信号型干扰：专家/oracle 是否有价值）与测试 2（去掉 WB 是否影响其他类型）的汇总。在 new/moe_diffic 下运行。"""
import sys, numpy as np
sys.path.insert(0, '.')
from aggregate_results import load, type_mask
print('## 测试 1：v4_digital（CCIR/FSK/OFDM/BURST/ACI），10% 监督(630 条)+remix，种子 42，测试集 A')
names = [('无专家+remix(基线)', 'v4b_p10_s42'), ('TF 专家+remix', 'v4t_p10_s42'), ('oracle(类型+属性)+remix', 'v4o_p10_s42'), ('局部 oracle(真实干扰时频)+remix', 'v4m_p10_s42')]
base = None
for n, r in names:
    d = load(r, 'test', '_avg8', 'runs')
    if d is None: print('%-30s 尚无结果' % n); continue
    tn = [str(t) for t in d['type_names']]
    pt = {t: d['ber_model'][type_mask(d, types=[t])].mean() for t in tn}
    lo = d['ber_model'][d['sir'] <= -10].mean()
    print('%-30s BER=%.4f 强干扰=%.4f EVM=%.3f SINR=%.2f | %s' % (n, d['ber_model'].mean(), lo, d['evm_model'].mean(), d['sinr_inband_model'].mean(), {k: round(float(v), 3) for k, v in pt.items()}))
    if base is None: base = d['ber_model'].mean(); d0 = d; print('   不处理(输入) BER=%.4f' % d['ber_mf'].mean())
    else: print('   相对基线 %+.4f' % (d['ber_model'].mean() - base))
for r in ('v4_cnn_all', 'v4_moe_all'):
    d = load(r, 'test', '_avg8', 'runs') or load(r, 'test', '', 'runs')
    if d is not None:
        sel = np.arange(len(d['ber_model']))[::2] if load(r, 'test', '_avg8', 'runs') is None else slice(None)
        print('参考(100%% 数据旧机) %-12s BER=%.4f' % (r, d['ber_model'][sel].mean()))
print('\n## 测试 2：去掉 WB 的 4 类训练 vs 5 类训练（均 remix，10%），在 ST/MT/NB/PL 上的 BER')
T4 = ['ST', 'MT', 'NB', 'PL']; res = {}
for tag, pat in (('5类(含WB)', 'au_remix_p10_s%d'), ('4类(去WB)', 'nw_p10_s%d')):
    rows = []
    for s in (42, 43):
        d = load(pat % s, 'test', '_avg8', 'runs')
        if d is None: rows = None; break
        rows.append([d['ber_model'][type_mask(d, types=T4)].mean()] + [d['ber_model'][type_mask(d, types=[t])].mean() for t in T4] + [d['ber_model'][type_mask(d, types=T4) & (d['sir'] <= -10)].mean()])
    if rows is None: print(tag, '尚无结果'); continue
    res[tag] = np.array(rows); a = res[tag]
    print('%-10s 非WB均值=%.4f 各类型 ST/MT/NB/PL=%s 强干扰=%.4f 各种子=%s' % (tag, a[:, 0].mean(), np.round(a[:, 1:5].mean(0), 4), a[:, 5].mean(), np.round(a[:, 0], 4)))
if len(res) == 2: print('配对差(4类−5类) 非WB BER:', np.round(res['4类(去WB)'][:, 0] - res['5类(含WB)'][:, 0], 4))
