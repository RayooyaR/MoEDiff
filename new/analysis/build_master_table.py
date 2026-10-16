"""汇总全部实验：所有基线 × 所有测试集 × 所有指标，输出一张长表（Markdown + CSV）。

用法（在 new/moe_diffic 下）： python ../analysis/build_master_table.py
输出： ../reports/all_results.md 、 ../reports/all_results.csv

取数规则：
  - 全部读取 runs/<名>/per_sample_<split>[_avg8].npz（每个测试样本一行）。
  - 扩散类方法优先用“随机采样 8 次取平均”（_avg8，只用测试集每隔一条的一半样本）；没有则退回单次随机采样并在数值后标 †。
  - 其余方法（匹配滤波、LMMSE、CNN、回归类）有完整测试集，统一取每隔一条（[::2]），与 _avg8 用同一批样本。
  - 匹配滤波(MF)＝对受干扰信号不做任何抑制直接解调，数值取自同一批样本的 *_mf 字段。
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
OUT = os.path.join(HERE, '..', 'reports')

METRICS = [('ber', 'BER 误码率↓', '%.4f'), ('evm', 'EVM 误差向量幅度↓', '%.3f'), ('mse', 'MSE 均方误差↓', '%.3f'),
           ('mae', 'MAE 平均绝对误差↓', '%.3f'), ('sinr_inband', '带内SINR(dB)↑', '%.2f')]

# 列：(key, 显示名, 是否扩散类)
COLS = [('mf', 'MF 匹配滤波', False), ('lmmse', 'LMMSE', False), ('cnn', 'CNN', False),
        ('dit', 'DiT扩散(无专家)', True), ('diffcom', 'DiffCom', True), ('icdm', 'ICDM双模型', True),
        ('moe', 'MoE-DiffIC(四专家)', True), ('tf', 'MoE-仅时频专家', True),
        ('ditL', '对照:DiT扩散-272(容量匹配)', True), ('dreg', '对照:DiT直接回归', False), ('mreg', '对照:MoE直接回归', False),
        ('mvcnn', '对照:CNN+专家特征', False), ('cnnL', '对照:加宽CNN', False),
        ('prior', '对照:MoE先验路由', True), ('ditLong', 'DiT扩散-延长训练', True), ('moeLong', 'MoE-延长训练', True)]

SEEN_ALL = ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM']
FS_SEEN = ['ST', 'MT', 'NB', 'WB', 'PL']                       # 小样本/zs2 训练用类型


def zs_runs(z):
    return {'lmmse': f'{z}_lmmse', 'cnn': f'{z}_cnn', 'dit': f'{z}_dit', 'diffcom': f'{z}_diffcom', 'icdm': f'{z}_icdm', 'moe': f'{z}_moe',
            'tf': f'{z}_ex_TF', 'ditL': f'{z}_ditL', 'dreg': f'{z}_dreg', 'mreg': f'{z}_mreg', 'mvcnn': f'{z}_mvcnn', 'cnnL': f'{z}_cnnL'}


# 每个场景： (场景名, 数据集, 划分, 方法->run, [(子集名, 过滤: None 全部 / ('in'|'out', 类型列表))])
SCEN = []
SCEN.append(('标准全量训练(8类干扰全部参与训练)', 'v2', 'test',
             {'lmmse': 'exp_lmmse', 'cnn': 'exp_cnn', 'dit': 'exp_nomoe', 'diffcom': 'exp_diffcom_1d', 'icdm': 'exp_icdm_1d', 'moe': 'exp_moe_legacy', 'prior': 'exp_moe_prior'},
             [('测试集A:与训练同参数区间(全部8类)', None)]))
SCEN.append(('标准全量训练(8类干扰全部参与训练)', 'v2', 'test_ood',
             {'lmmse': 'exp_lmmse', 'cnn': 'exp_cnn', 'dit': 'exp_nomoe', 'diffcom': 'exp_diffcom_1d', 'icdm': 'exp_icdm_1d', 'moe': 'exp_moe_legacy', 'prior': 'exp_moe_prior'},
             [('测试集B:未见干扰参数区间(全部8类)', None)]))
for z, held, lab in (('zs2', ['LFM', 'NFM', 'RFM'], '整类留出-留出调频类(线性/非线性/随机调频不参与训练)'),
                     ('zs1', ['ST', 'MT'], '整类留出-留出单音/多音(不参与训练)'),
                     ('zs3', ['PL'], '整类留出-留出脉冲(不参与训练)'),
                     ('zs4', ['NB', 'WB'], '整类留出-留出窄带/宽带(不参与训练)')):
    seen = [t for t in SEEN_ALL if t not in held]
    SCEN.append((lab, 'v2', 'test', zs_runs(z), [('全部为训练见过的干扰类型(%s)' % '/'.join(seen), ('out', held)), ('全部为训练未见过的干扰类型(%s)' % '/'.join(held), ('in', held))]))
for K in (1, 5, 10):
    for aug, pre in (('用增强', 'fs_K%d_' % K), ('不增强', 'fs_K%d_na_' % K)):
        r = {m: pre + m for m in ('lmmse', 'cnn', 'dit', 'diffcom', 'icdm', 'moe')}
        SCEN.append(('小样本-每个(类型×调制×信干比)格子仅%d条训练样本-%s(训练类型同"留出调频类")' % (K, aug), 'v2', 'test', r,
                     [('全部为训练见过的干扰类型(ST/MT/NB/WB/PL)', ('out', ['LFM', 'NFM', 'RFM'])), ('全部为训练未见过的干扰类型(LFM/NFM/RFM)', ('in', ['LFM', 'NFM', 'RFM']))]))
EF = {'lmmse': 'ef_lmmse', 'cnn': 'ef_cnn', 'dit': 'ef_dit', 'diffcom': 'ef_diffcom', 'icdm': 'ef_icdm', 'moe': 'ef_moe'}
SCEN.append(('更难场景(干扰叠加7种随机效应,8类全部训练)', 'v5_effects', 'test', EF, [('测试集A:与训练同参数区间(全部8类)', None)]))
SCEN.append(('更难场景(干扰叠加7种随机效应,8类全部训练)', 'v5_effects', 'test_ood', EF, [('测试集B:未见干扰参数区间(全部8类)', None)]))
SCEN.append(('CCIR场景(异速率同信道数字干扰,单一类别)', 'v3b_ccir', 'test',
             {'lmmse': 'hardb_lmmse', 'cnn': 'hardb_cnn', 'dit': 'ccir_nomoe', 'diffcom': 'ccir_diffcom', 'icdm': 'ccir_icdm', 'moe': 'ccir_moe',
              'ditLong': 'ccir_nomoe_long', 'moeLong': 'ccir_moe_long'}, [('全部(CCIR)', None)]))
SCEN.append(('早期场景(同信道干扰CCI与混合干扰MIX)', 'v3_hard', 'test', {'lmmse': 'hard_lmmse', 'cnn': 'hard_cnn'}, [('全部(CCI+MIX)', None)]))
SCEN.append(('数字干扰家族-全类型训练(CCIR/FSK/OFDM/BURST/ACI)', 'v4_digital', 'test', {'lmmse': 'v4_lmmse', 'cnn': 'v4_cnn_all', 'moe': 'v4_moe_all'}, [('全部5类', None)]))
SCEN.append(('数字干扰家族-留出FSK(频移键控)', 'v4_digital', 'test', {'cnn': 'v4_hf_cnn', 'moe': 'v4_hf_moe'},
             [('训练见过的类型(CCIR/OFDM/BURST/ACI)', ('out', ['FSK'])), ('训练未见过的类型(FSK)', ('in', ['FSK']))]))
SCEN.append(('数字干扰家族-留出OFDM(正交频分复用)', 'v4_digital', 'test', {'cnn': 'v4_ho_cnn', 'moe': 'v4_ho_moe'},
             [('训练见过的类型(CCIR/FSK/BURST/ACI)', ('out', ['OFDM'])), ('训练未见过的类型(OFDM)', ('in', ['OFDM']))]))


def load_one(run, split):
    """返回 (数据, 是否单次随机采样)；优先 _avg8。"""
    for suf, single in (('_avg8', False), ('', True)):
        f = os.path.join(RUNS, run, 'per_sample_%s%s.npz' % (split, suf))
        if os.path.exists(f):
            return {k: v for k, v in np.load(f).items()}, single, suf
    return None, False, None


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, csv_rows, warns = [], [], []
    for scen, data, split, runs, subsets in SCEN:
        loaded = {}
        n_full = None
        for key, run in runs.items():
            d, single, suf = load_one(run, split)
            if d is None:
                continue
            n = len(d['type'])
            if suf != '_avg8':
                n_full = n_full or n
            loaded[key] = (d, single, suf)
        if n_full is None:                                            # 全部都是 avg8：完整长度 = 2n 或 2n-1，仅需对齐
            n_full = 2 * len(next(iter(loaded.values()))[0]['type'])
        ref_types = None
        sub = {}
        for key, (d, single, suf) in loaded.items():
            n = len(d['type'])
            if suf == '_avg8':
                idx = slice(None)
                if abs(n - n_full / 2) > 1:
                    warns.append('%s %s 长度不符(%d vs %d)' % (runs[key], split, n, n_full)); continue
            else:
                idx = slice(None, None, 2)
                if n != n_full:
                    warns.append('%s %s 长度不符(%d vs %d)' % (runs[key], split, n, n_full)); continue
            dd = {k: (v[idx] if getattr(v, 'ndim', 0) == 1 and len(v) == n else v) for k, v in d.items()}
            tn = [str(t) for t in dd['type_names']]
            if ref_types is None:
                ref_types = tn
            sub[key] = (dd, single, tn)
        # MF：取任一方法的 *_mf
        any_key = next(iter(sub))
        for sname, filt in subsets:
            def mask(dd, tn):
                if filt is None:
                    return np.ones(len(dd['type']), dtype=bool)
                m = np.isin(dd['type'], [i for i, t in enumerate(tn) if t in filt[1]])
                return m if filt[0] == 'in' else ~m
            ns = {}
            for mk, mname, fmt in METRICS:
                vals = {}
                for ck, cname, diff in COLS:
                    if ck == 'mf':
                        dd, single, tn = sub[any_key]
                        vals[ck] = (float(dd[mk + '_mf'][mask(dd, tn)].mean()), False)
                    elif ck in sub:
                        dd, single, tn = sub[ck]
                        vals[ck] = (float(dd[mk + '_model'][mask(dd, tn)].mean()), bool(single and diff))
                    ns[ck] = int(mask(*[sub[ck][0], sub[ck][2]]).sum()) if ck in sub else None
                best = None
                cand = {k: v[0] for k, v in vals.items() if not k.startswith(('ditLong', 'moeLong'))}
                if cand:
                    best = (max if mk == 'sinr_inband' else min)(cand, key=cand.get)
                row = [scen, sname, mname]
                for ck, cname, diff in COLS:
                    if ck in vals:
                        v, dag = vals[ck]
                        s = fmt % v + ('†' if dag else '')
                        row.append('**' + s + '**' if ck == best else s)
                        csv_rows.append([scen, data, split, sname, mk, ck, cname, '%.6f' % v, ns.get(ck) if ck != 'mf' else ns.get(any_key), '单次随机采样' if dag else ''])
                    else:
                        row.append('—')
                rows.append(row)
    # 写 Markdown
    legend = open(os.path.join(HERE, 'master_table_legend.md'), encoding='utf-8').read() if os.path.exists(os.path.join(HERE, 'master_table_legend.md')) else ''
    with open(os.path.join(OUT, 'all_results.md'), 'w', encoding='utf-8') as f:
        f.write('# 全部实验结果汇总（所有基线 × 所有测试集 × 所有指标）\n\n' + legend + '\n## 结果表\n\n')
        f.write('| 场景 | 测试集 | 指标 | ' + ' | '.join(c[1] for c in COLS) + ' |\n')
        f.write('|' + '---|' * (3 + len(COLS)) + '\n')
        last = None
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')
    with open(os.path.join(OUT, 'all_results.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['场景', '数据集', '划分', '测试集', '指标', '方法键', '方法', '数值', '样本数', '备注'])
        w.writerows(csv_rows)
    print('rows', len(rows), 'csv', len(csv_rows))
    for w_ in warns:
        print('WARN', w_)


if __name__ == '__main__':
    main()
