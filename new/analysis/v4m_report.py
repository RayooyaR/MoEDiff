"""v4_digital：remix 验证 + 真正的混合专家（在 new/moe_diffic 下运行）。种子 42、43；测试集 A（5 类信号型干扰）。"""
import sys, numpy as np
sys.path.insert(0, '.')
from aggregate_results import load, type_mask
S = (42, 43)
def stats(pat):
    ds = [load(pat % s, 'test', '_avg8', 'runs') for s in S]
    if any(d is None for d in ds): return None
    tn = [str(t) for t in ds[0]['type_names']]
    return dict(ds=ds, ber=np.array([d['ber_model'].mean() for d in ds]), tn=tn,
                pt=np.mean([[d['ber_model'][type_mask(d, types=[t])].mean() for t in tn] for d in ds], 0),
                lo=np.mean([d['ber_model'][d['sir'] <= -10].mean() for d in ds]), evm=np.mean([d['evm_model'].mean() for d in ds]), sinr=np.mean([d['sinr_inband_model'].mean() for d in ds]),
                inp=np.mean([d['ber_mf'].mean() for d in ds]))
cfg = [('无专家，无增强', 'v4n_p10_s%d'), ('无专家 + remix', 'v4b_p10_s%d'), ('无专家 + remixys(仅(y,s))', 'v4ry_p10_s%d'),
       ('TF 单专家 + remix', 'v4t_p10_s%d'), ('TF+CYC 静态 + remix', 'v4tc_p10_s%d'), ('四专家 静态等权 + remix', 'v4q4u_p10_s%d'), ('四专家 软路由 + remix', 'v4q4r_p10_s%d'), ('锚定残差混合专家(TF+FD+MS+TD+CYC) + remix', 'v4a_p10_s%d')]
R = {}
print('%-26s %-18s %-9s %-8s %-7s %-6s %s' % ('配置', '各种子总BER', '均值', '强干扰', 'EVM', 'SINR', '各类型 ' ))
for n, p in cfg:
    r = stats(p); R[n] = r
    if r is None: print('%-26s 尚无结果' % n); continue
    print('%-26s %-18s %.4f   %.4f   %.3f  %.2f  %s' % (n, np.round(r['ber'], 4), r['ber'].mean(), r['lo'], r['evm'], r['sinr'], dict(zip(r['tn'], np.round(r['pt'], 3)))))
if R.get('无专家，无增强'): print('不处理(输入) BER = %.4f' % R['无专家，无增强']['inp'])
def pair(a, b):
    if R.get(a) is None or R.get(b) is None: return
    d = R[a]['ber'] - R[b]['ber']
    print('配对差 %s − %s: 逐种子 %s 均值 %+.4f (%.1f%%)' % (a, b, np.round(d, 4), d.mean(), 100 * d.mean() / R[b]['ber'].mean()))
print('\n## remix 在 v4 上的效果')
pair('无专家 + remix', '无专家，无增强'); pair('无专家 + remixys(仅(y,s))', '无专家，无增强'); pair('无专家 + remixys(仅(y,s))', '无专家 + remix')
print('\n## 混合专家 vs 单专家（均 remix）')
for a in ('TF+CYC 静态 + remix', '四专家 静态等权 + remix', '四专家 软路由 + remix', '锚定残差混合专家(TF+FD+MS+TD+CYC) + remix'):
    pair(a, 'TF 单专家 + remix'); pair(a, '无专家 + remix')
r = R.get('四专家 软路由 + remix')
if r is not None:
    G = np.concatenate([d['gates'] for d in r['ds']]); T = np.concatenate([d['type'] for d in r['ds']]); sir = np.concatenate([d['sir'] for d in r['ds']])
    G = G.reshape(len(G), -1)
    print('\n## 软路由的门控权重（专家顺序 FD,MS,TF,TD；%d 个测试样本）' % len(G))
    print('总体平均权重', np.round(G.mean(0), 3), ' 样本间标准差', np.round(G.std(0), 3))
    for ti, t in enumerate(r['tn']): print('  %-6s' % t, np.round(G[T == ti].mean(0), 3))
    ss = ((G - G.mean(0)) ** 2).sum(0)
    eta_t = [sum((T == k).sum() * (G[T == k, j].mean() - G[:, j].mean()) ** 2 for k in np.unique(T)) / max(ss[j], 1e-12) for j in range(G.shape[1])]
    sr = np.round(sir).astype(int); eta_s = [sum((sr == k).sum() * (G[sr == k, j].mean() - G[:, j].mean()) ** 2 for k in np.unique(sr)) / max(ss[j], 1e-12) for j in range(G.shape[1])]
    print('门控被"干扰类型"解释的比例 η²(每专家)', np.round(eta_t, 3), ' 被"SIR"解释', np.round(eta_s, 3))

# ---- 锚定残差混合专家的专家贡献（α）与 v2 对照
r = R.get('锚定残差混合专家(TF+FD+MS+TD+CYC) + remix')
if r is not None:
    G = np.concatenate([d['gates'] for d in r['ds']]).reshape(-1, 5)
    print('\n## v4 锚定混合专家的 α（专家顺序 TF,FD,MS,TD,CYC；锚 TF 恒为 1）：均值', np.round(G.mean(0), 3), ' 样本间sd', np.round(G.std(0), 3))
print('\n## v2 (ST/MT/NB/WB/PL) 10% + remix：锚定混合专家 vs 对照')
SEEN = ['ST', 'MT', 'NB', 'WB', 'PL']; UN = ['LFM', 'NFM', 'RFM']
for n, pat in (('无专家+remix', 'au_remix_p10_s%d'), ('TF+remix', 'tfr_p10_s%d'), ('锚定混合专家+remix', 'v2a_p10_s%d')):
    ds = [load(pat % s_, 'test', '_avg8', 'runs') for s_ in S]
    if any(d is None for d in ds): print(n, '尚无结果'); continue
    seen = [d['ber_model'][type_mask(d, exclude=UN)].mean() for d in ds]; nw = [d['ber_model'][type_mask(d, types=['ST', 'MT', 'NB', 'PL'])].mean() for d in ds]
    un = [d['ber_model'][type_mask(d, types=UN)].mean() for d in ds]
    print('%-20s 见过=%s 均值%.4f 非WB=%.4f 未见=%.4f' % (n, np.round(seen, 4), np.mean(seen), np.mean(nw), np.mean(un)))
