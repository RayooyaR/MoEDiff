"""统一总表：多场景 × 全部基线 × 全部指标（缺失单元格标 —，并输出覆盖矩阵，供补数据使用）。

用法（在 new/moe_diffic 下）： python ../analysis/unified_table.py
输出： ../reports/unified_table.md  （总览BER表 + 覆盖/缺口清单 + 各指标详表）
       ../reports/unified_table.csv （长表：场景/子集/指标/方法/数值/种子数/样本数）
取数：runs/<run>/per_sample_<split>[_avg8].npz；扩散类优先 _avg8（8 次采样平均，测试集每隔一条）；
      非扩散(MF/LMMSE/CNN)取完整测试集每隔一条，与 avg8 同一批样本；多个种子先分别计算再取均值。
"""
import csv, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from aggregate_results import load
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs'); OUT = os.path.join(HERE, '..', 'reports')

METHODS = [('mf', '不处理(MF)'), ('lmmse', 'LMMSE'), ('cnn', 'CNN'), ('rest', 'Restormer'), ('diffcom', 'DiffCom'), ('icdm', 'ICDM'),
           ('dit', 'DiT扩散(无专家)'), ('moe4', 'MoE-DiffIC四专家'), ('tf', '仅时频专家'), ('unet', 'U-Net扩散+时频')]
METRICS = [('ber', 'BER↓', '%.4f', 'min'), ('delta', 'ΔBER↑', '%+.4f', 'max'), ('neg', '负增益比例↓', '%.3f', 'min'), ('evm', 'EVM↓', '%.3f', 'min'),
           ('mse', 'MSE↓', '%.3f', 'min'), ('mae', 'MAE↓', '%.3f', 'min'), ('sinr', '带内SINR(dB)↑', '%.2f', 'max')]
FM = ['LFM', 'NFM', 'RFM']; ZS5 = ['ST', 'MT', 'NB', 'WB', 'PL']

def S(base, seeds): return ['%s_s%d' % (base, s) for s in seeds]

def get(run, split, diffusion):
    d = load(run, split, '_avg8', RUNS)
    if d is not None: return d, False
    d = load(run, split, '', RUNS)
    if d is None: return None, False
    n = len(d['type']); d = {k: (v[::2] if getattr(v, 'ndim', 0) >= 1 and len(v) == n else v) for k, v in d.items()}
    return d, diffusion                  # 扩散类没有 avg8 -> 单次采样，标 †

DIFF = {'diffcom', 'icdm', 'dit', 'moe4', 'tf', 'unet'}
def cell(key, runs, split, held, which):
    """返回 {metric: 值}, 种子数, 样本数；runs 为空或文件缺失 -> None"""
    vs = []; single = False
    for r in runs:
        d, sg = get(r, split, key in DIFF)
        if d is None: continue
        single |= sg
        names = [str(t) for t in d['type_names']]
        isheld = np.isin(d['type'], [i for i, t in enumerate(names) if t in (held or [])])
        m = np.ones(len(d['type']), bool) if which == 'all' else (isheld if which == 'held' else ~isheld)
        if not m.any(): continue
        b, bi = d['ber_model'][m], d['ber_mf'][m]
        vs.append(dict(ber=b.mean(), delta=(bi - b).mean(), neg=(b > bi + 1e-12).mean(), evm=d['evm_model'][m].mean(), mse=d['mse_model'][m].mean(), mae=d['mae_model'][m].mean(),
                       sinr=d['sinr_inband_model'][m].mean(), mf=dict(ber=bi.mean(), delta=0.0, neg=0.0, evm=d['evm_mf'][m].mean(), mse=d['mse_mf'][m].mean(), mae=d['mae_mf'][m].mean(), sinr=d['sinr_inband_mf'][m].mean()), n=int(m.sum())))
    if not vs: return None
    out = {k: float(np.mean([v[k] for v in vs])) for k in ('ber', 'delta', 'neg', 'evm', 'mse', 'mae', 'sinr')}
    out['_mf'] = {k: float(np.mean([v['mf'][k] for v in vs])) for k in vs[0]['mf']}
    out['_nseed'] = len(vs); out['_n'] = vs[0]['n']; out['_single'] = single
    out['_sd'] = float(np.std([v['ber'] for v in vs], ddof=1)) if len(vs) > 1 else None
    return out

# ---------------- 场景定义： (组, 场景名, split, 子集[(名, held, which)], {方法: [runs]}, 备注) ----------------
SC = []
def add(group, name, split, subs, runs, note=''): SC.append((group, name, split, subs, runs, note))
def zs(z): return {'lmmse': [f'{z}_lmmse'], 'cnn': [f'{z}_cnn'], 'diffcom': [f'{z}_diffcom'], 'icdm': [f'{z}_icdm'], 'dit': [f'{z}_dit'], 'moe4': [f'{z}_moe'], 'tf': [f'{z}_ex_TF']}

# 1 监督（8 类全部参与训练）
FULL = {'lmmse': ['exp_lmmse'], 'cnn': ['exp_cnn'], 'diffcom': ['exp_diffcom_1d'], 'icdm': ['exp_icdm_1d'], 'dit': ['exp_nomoe'], 'moe4': ['exp_moe_legacy'], 'tf': ['full8_tf_s42'], 'unet': ['full8_unet_s42'], 'rest': ['rest_full8_s42']}
add('1 监督', '全量监督(8类全训练)·测试集A同参数区间', 'test', [('全部8类', None, 'all')], FULL)
add('1 监督', '全量监督(8类全训练)·测试集B未见参数区间', 'test_ood', [('全部8类', None, 'all')], FULL)
# 2 整类留出（零样本）
for z, held, lab in (('zs2', FM, '留出调频类(LFM/NFM/RFM)'), ('zs1', ['ST', 'MT'], '留出单音/多音'), ('zs3', ['PL'], '留出脉冲'), ('zs4', ['NB', 'WB'], '留出窄带/宽带')):
    r = zs(z); r['unet'] = ['j0_p100_s42', 'j0_p100_s43'] if z == 'zs2' else [f'{z}_unet_s42']
    r['rest'] = S('rest_p100', (42, 43, 44)) if z == 'zs2' else [f'rest_{z}_s42']
    add('2 整类留出', lab, 'test', [('见过类型', held, 'seen'), ('未见类型(零样本)', held, 'held')], r)
# 3 少样本·旧管线（每格 K 条）
for K in (1, 5, 10):
    for aug, pre in (('增强', f'fs_K{K}_'), ('不增强', f'fs_K{K}_na_')):
        r = {m: [pre + m] for m in ('lmmse', 'cnn', 'diffcom', 'icdm')}; r['dit'] = [pre + 'dit']; r['moe4'] = [pre + 'moe']
        r['tf'] = [f'fs{K}_{"" if aug == "增强" else "na_"}tf_s42']
        r['unet'] = [f'fsK{K}_unet_s42'] if aug == '不增强' else []
        r['rest'] = [f'rest_K{K}_s42'] if aug == '不增强' else []
        add('3 少样本(每格K条)', f'每格{K}条·{aug}', 'test', [('见过类型', FM, 'seen'), ('未见类型(零样本)', FM, 'held')], r,
            '旧管线:150轮,步数未对齐' if aug == '增强' else '旧管线(U-Net列:步数对齐,不增强)')
# 4 少样本·新管线（10/30/100% 数据，步数对齐）
for P in (10, 30, 100):
    for split_, subs in (('test', [('见过类型', FM, 'seen'), ('未见类型(零样本)', FM, 'held')]), ('test_ood', [('全部(未见参数区间OOD)', None, 'all')])):
        r = {'lmmse': ['zs2_lmmse'] if split_ == 'test' else ['exp_lmmse'], 'cnn': S(f'de_cnn_p{P}', (42, 43, 44)), 'dit': S(f'de_dit_p{P}', (42, 43, 44)), 'tf': S(f'de_tf_p{P}', (42, 43, 44)),
             'diffcom': ['zs2_diffcom'] if P == 100 and split_ == 'test' else (S('fs_p10_diffcom', (42, 43, 44)) if P == 10 else [f'pct{P}_diffcom_s42']), 'icdm': ['zs2_icdm'] if P == 100 and split_ == 'test' else (S('fs_p10_icdm', (42, 43, 44)) if P == 10 else [f'pct{P}_icdm_s42']),
             'rest': S(f'rest_p{P}', (42, 43, 44)), 'moe4': (S('e1_four', (42, 43, 44)) if split_ == 'test' else ['zs2_moe']) if P == 100 else [f'pct{P}_moe4_s42'], 'unet': S(f'j0_p{P}', (42, 43))}
        add('4 少样本(训练数据%)', f'训练数据{P}%·' + ('测试集A' if split_ == 'test' else '测试集B'), split_, subs, r, 'LMMSE与数据量无关(复用)')
# 5 逐类留出 LOTO
for H in ('LFM', 'RFM', 'MT', 'PL', 'WB'):
    r = {'cnn': S(f'lo_{H}_cnn', (42, 43, 44)), 'dit': [f'lo_{H}_dit0_s42'], 'moe4': S(f'lo_{H}_legacy4', (42, 43, 44)), 'tf': S(f'lo_{H}_tf', (42, 43, 44)),
         'unet': [f'lof_{H}_tf_s42'], 'rest': [f'rest_lo_{H}_s42'], 'lmmse': ['exp_lmmse'], 'diffcom': [f'lo_{H}_diffcom_s42'], 'icdm': [f'lo_{H}_icdm_s42']}
    add('5 逐类留出(只拿走1类)', f'留出{H}', 'test', [(f'被留出类型{H}', [H], 'held')], r)
# 6 更难场景/其他
EF = {'lmmse': ['ef_lmmse'], 'cnn': ['ef_cnn'], 'diffcom': ['ef_diffcom'], 'icdm': ['ef_icdm'], 'dit': ['ef_dit'], 'moe4': ['ef_moe'], 'unet': ['ef_unet_s42'], 'tf': ['ef_tf_s42'], 'rest': ['rest_ef_s42']}
add('6 其他场景', '更难场景(叠加7种随机效应)·测试集A', 'test', [('全部8类', None, 'all')], EF)
add('6 其他场景', '更难场景(叠加7种随机效应)·测试集B', 'test_ood', [('全部8类', None, 'all')], EF)
add('6 其他场景', 'CCIR异速率同信道数字干扰', 'test', [('全部', None, 'all')],
    {'lmmse': ['hardb_lmmse'], 'cnn': ['hardb_cnn'], 'diffcom': ['ccir_diffcom'], 'icdm': ['ccir_icdm'], 'dit': ['ccir_nomoe'], 'moe4': ['ccir_moe'], 'unet': ['ccir_unet_s42'], 'tf': ['ccir_tf_s42'], 'rest': ['rest_ccir_s42']})
add('6 其他场景', '数字干扰家族(CCIR/FSK/OFDM/BURST/ACI)', 'test', [('全部5类', None, 'all')], {'lmmse': ['v4_lmmse'], 'cnn': ['v4_cnn_all'], 'moe4': ['v4_moe_all']})

def fmt(f, v): return '—' if v is None else f % v

def main():
    os.makedirs(OUT, exist_ok=True)
    rows_long = []; over = []; detail = []; gaps = []
    for group, name, split, subs, runs, note in SC:
        for sname, held, which in subs:
            res = {}
            for mk, _ in METHODS:
                if mk == 'mf': continue
                res[mk] = cell(mk, runs.get(mk, []), split, held, which)
            anyr = next((v for v in res.values() if v), None)
            if anyr is None: continue
            res['mf'] = dict(anyr['_mf'], _nseed=0, _n=anyr['_n'], _sd=None)
            # 总览行
            vals = {k: v['ber'] for k, v in res.items() if v and k != 'mf'}
            best = min(vals, key=vals.get) if vals else None
            cells = []
            for mk, _ in METHODS:
                v = res.get(mk)
                if v is None:
                    cells.append('—'); 
                    if mk != 'mf': gaps.append((group, name, sname, mk))
                else:
                    t = '%.4f' % v['ber'] + ('(%d)' % v['_nseed'] if v['_nseed'] > 1 else '') + ('†' if v.get('_single') else '')
                    cells.append('**' + t + '**' if mk == best else t)
            over.append('| %s | %s | %s | %d | ' % (group, name, sname, anyr['_n']) + ' | '.join(cells) + ' |')
            # 详表
            for mkey, mname, f, direction in METRICS:
                vals = {k: v[mkey] for k, v in res.items() if v and k != 'mf'}
                bestk = (min if direction == 'min' else max)(vals, key=vals.get) if vals else None
                cs = []
                for mk, _ in METHODS:
                    v = res.get(mk)
                    if v is None: cs.append('—'); continue
                    x = v[mkey] if mk != 'mf' else v[mkey] if mkey in v else None
                    t = fmt(f, x) + ('†' if v.get('_single') else ''); cs.append('**' + t + '**' if mk == bestk else t)
                    rows_long.append([group, name, split, sname, mname, dict(METHODS)[mk], '%.6f' % x if x is not None else '', v['_nseed'], v['_n']])
                detail.append('| %s | %s | %s | %s | ' % (name, sname, mname, '') + ' | '.join(cs) + ' |')
    hdr = '| 组 | 场景 | 子集 | 样本数 | ' + ' | '.join(n for _, n in METHODS) + ' |\n|' + '---|' * (4 + len(METHODS)) + '\n'
    with open(os.path.join(OUT, 'unified_table.md'), 'w', encoding='utf-8') as f:
        f.write('# 统一总表：多场景 × 全部基线 × 全部指标\n\n> 由 `new/analysis/unified_table.py` 生成。单元格 = BER（括号内为种子数，>1 个时为均值）；† = 扩散类只有单次随机采样（未做平均 8 次）；— = 尚无数据；粗体 = 该行最优。\n> 扩散类均为平均 8 次采样；测试集每隔一条；MF=不处理直接解调。\n\n## 一、总览（BER）\n\n')
        f.write(hdr + '\n'.join(over) + '\n\n## 二、缺口清单（—）\n\n')
        from collections import defaultdict
        g = defaultdict(list)
        for grp, nm, sn, mk in gaps: g[(grp, nm)].append(dict(METHODS)[mk]) if dict(METHODS)[mk] not in g[(grp, nm)] else None
        for (grp, nm), ms in g.items(): f.write('- %s · %s：缺 %s\n' % (grp, nm, '、'.join(ms)))
        f.write('\n## 三、全部指标详表\n\n| 场景 | 子集 | 指标 | | ' + ' | '.join(n for _, n in METHODS) + ' |\n|' + '---|' * (4 + len(METHODS)) + '\n' + '\n'.join(detail) + '\n')
    with open(os.path.join(OUT, 'unified_table.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['组', '场景', '划分', '子集', '指标', '方法', '数值', '种子数', '样本数']); w.writerows(rows_long)
    print('over rows', len(over), 'gaps', len(gaps), 'long', len(rows_long))
if __name__ == '__main__': main()
