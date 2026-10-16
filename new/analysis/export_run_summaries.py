"""把所有已完成的 run 的逐样本结果汇总成两张小表（便于上传/复核，不含权重与逐样本 npz）。
用法（在 new/moe_diffic 下）： python ../analysis/export_run_summaries.py
输出： ../results/run_summary_by_subset.csv   —— 每个 run × 测试集 × 子集（全部/见过5类/非WB/未见调频类）的 BER、输入BER、ΔBER、负增益比例、EVM、MSE、MAE、带内SINR
       ../results/run_by_type_sir.csv         —— 每个 run × 测试集 × 干扰类型 × SIR 的 BER 与输入BER、样本数
        ../results/runs_index.csv              —— run 名、模型类型、种子、关键参数（来自 args.json）、参数量
取数规则同 unified_table：扩散类取 per_sample_<split>_avg8.npz（平均 8 次采样，测试集每隔一条）。"""
import csv, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs'); OUT = os.path.join(HERE, '..', 'results'); os.makedirs(OUT, exist_ok=True)
UN = ['LFM', 'NFM', 'RFM']; SEEN5 = ['ST', 'MT', 'NB', 'WB', 'PL']; NONWB = ['ST', 'MT', 'NB', 'PL']
sub_rows, ts_rows, idx_rows = [], [], []
for rdir in sorted(glob.glob(os.path.join(RUNS, '*'))):
    run = os.path.basename(rdir)
    files = {os.path.basename(f)[11:-4]: f for f in glob.glob(os.path.join(rdir, 'per_sample_*.npz')) if f.endswith('_avg8.npz') or f.endswith('_avg8g.npz')}
    if not files: continue
    a = json.load(open(os.path.join(rdir, 'args.json'))) if os.path.exists(os.path.join(rdir, 'args.json')) else {}
    ri = json.load(open(os.path.join(rdir, 'run_info.json'))) if os.path.exists(os.path.join(rdir, 'run_info.json')) else {}
    idx_rows.append([run, a.get('model_type'), a.get('seed'), a.get('backbone'), a.get('experts'), a.get('fusion'), a.get('cond_mode'), a.get('phys_aug'), a.get('oracle_cond'), a.get('extra_local'),
                     a.get('loss_variant'), a.get('train_per_cell'), a.get('aug_multiplier'), a.get('epochs'), a.get('new_data_root'), ri.get('params')])
    for split, f in sorted(files.items()):
        d = {k: v for k, v in np.load(f).items()}
        names = [str(t) for t in d['type_names']]
        isT = lambda ts: np.isin(d['type'], [i for i, t in enumerate(names) if t in ts])
        subsets = [('全部', np.ones(len(d['type']), bool))]
        if any(t in names for t in SEEN5) and any(t in names for t in UN):
            subsets += [('见过5类', isT(SEEN5)), ('非WB四类', isT(NONWB)), ('未见调频类', isT(UN))]
        for sn, m in subsets:
            if not m.any(): continue
            b, bi = d['ber_model'][m], d['ber_mf'][m]
            sub_rows.append([run, split, sn, int(m.sum()), '%.5f' % b.mean(), '%.5f' % bi.mean(), '%.5f' % (bi - b).mean(), '%.4f' % (b > bi + 1e-12).mean(), '%.4f' % d['evm_model'][m].mean(),
                             '%.4f' % d['mse_model'][m].mean(), '%.4f' % d['mae_model'][m].mean(), '%.3f' % d['sinr_inband_model'][m].mean()])
        sir = np.round(d['sir']).astype(int)
        for ti, t in enumerate(names):
            for s in sorted(set(sir.tolist())):
                m = (d['type'] == ti) & (sir == s)
                if m.any(): ts_rows.append([run, split, t, s, int(m.sum()), '%.5f' % d['ber_model'][m].mean(), '%.5f' % d['ber_mf'][m].mean()])
def w(name, head, rows):
    with open(os.path.join(OUT, name), 'w', newline='', encoding='utf-8-sig') as f:
        c = csv.writer(f); c.writerow(head); c.writerows(rows)
    print(name, len(rows), '行', '%.2f MB' % (os.path.getsize(os.path.join(OUT, name)) / 1e6))
w('run_summary_by_subset.csv', ['run', 'split', 'subset', 'n', 'BER', 'BER_input', 'delta_BER', 'neg_gain_ratio', 'EVM', 'MSE', 'MAE', 'SINR_inband_dB'], sub_rows)
w('run_by_type_sir.csv', ['run', 'split', 'type', 'SIR_dB', 'n', 'BER', 'BER_input'], ts_rows)
w('runs_index.csv', ['run', 'model_type', 'seed', 'backbone', 'experts', 'fusion', 'cond_mode', 'phys_aug', 'oracle_cond', 'extra_local', 'loss_variant', 'train_per_cell', 'aug_multiplier', 'epochs', 'data_root', 'params'], idx_rows)
