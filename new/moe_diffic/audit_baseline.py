"""P0：实验资产审计与数据一致性。一键输出 ../reports/baseline_audit.md（以及 ../reports/experiment_index.csv）。

内容：
 1. 实验索引：runs/* -> 模型/专家/路由/训练类型/种子/每格样本数/epoch/更新步数/采样步数/噪声模式/参数量/可用评估文件
 2. 划分检查：各数据集 train/val/test/test_ood 的 clean_id 两两不相交
 3. 数据定义检查：noisy = clean + interf + noise；带内 SIR/SNR 与保存值一致；干净信号经已知同步解调 BER=0
 4. 复现检查：用 evaluate_new（同样本、同种子、avg8）重算选定 run 的 BER，与已保存的逐样本结果比较
 5. 评估/路由路径说明与实测：eval() 采样用 training=False 的路由；Top-K“乘零”与“真屏蔽”的差别（调用 tests/test_router_masks.py）
用法: python audit_baseline.py [--repro zs2_ex_TF zs2_moe] [--skip_repro]
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
import numpy as np  # noqa: E402

RUNS = os.path.join(HERE, 'runs')
DATA = os.path.abspath(os.path.join(HERE, '..', 'data'))
REPORTS = os.path.abspath(os.path.join(HERE, '..', 'reports'))


def git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'n/a'


def index_runs():
    rows = []
    for r in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, r, 'args.json')
        if not os.path.exists(p):
            continue
        a = json.load(open(p))
        files = sorted(os.path.basename(f)[len('per_sample_'):-4] for f in glob.glob(os.path.join(RUNS, r, 'per_sample_*.npz')))
        info = {}
        if os.path.exists(os.path.join(RUNS, r, 'run_info.json')):
            info = json.load(open(os.path.join(RUNS, r, 'run_info.json')))
        rows.append(dict(run=r, model_type=a.get('model_type'), experts=' '.join(a.get('experts') or ['FD', 'MS', 'TF', 'TD']) if not a.get('disable_moe') else 'none',
                         router=a.get('router_type'), backbone=a.get('backbone', 'dit'), cond_mode=a.get('cond_mode', 'xattn'), data=os.path.basename(str(a.get('new_data_root'))),
                         train_types=' '.join(a.get('new_types') or ['all']), holdout=' '.join(a.get('holdout') or []), seed=a.get('seed'), train_per_cell=a.get('train_per_cell'),
                         augment=a.get('augment') or a.get('phys_aug', 'none'), epochs=a.get('epochs'), ddim_steps=a.get('ddim_steps'), noise_mode=a.get('noise_mode'),
                         amp=a.get('amp', 'none'), params=info.get('params'), eval_files=' '.join(files)))
    return rows


def check_splits():
    out = []
    for d in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, d)
        if not os.path.isfile(os.path.join(p, 'metadata.json')):
            continue
        ids = {}
        for sp in ('train', 'val', 'test', 'test_ood'):
            fs = glob.glob(os.path.join(p, sp, '*', '*', '*.npz'))
            if fs:
                s = set()
                for f in fs:
                    s |= set(np.load(f)['clean_id'].tolist())
                ids[sp] = s
        names = list(ids)
        bad = [(names[i], names[j], len(ids[names[i]] & ids[names[j]])) for i in range(len(names)) for j in range(i + 1, len(names)) if ids[names[i]] & ids[names[j]]]
        out.append((d, {k: len(v) for k, v in ids.items()}, bad))
    return out


def check_definitions(n_per_split=60):
    from metrics_new import matched_filter, symbols_to_bits
    res = []
    for d in ('v2',):
        meta = json.load(open(os.path.join(DATA, d, 'metadata.json')))
        SPS, NSYM, SPAN = int(meta['SPS']), int(meta['N_SYM']), int(meta['RRC_SPAN'])
        fm_of = lambda ro: (1 + ro) / (2 * SPS)
        for sp in ('train', 'test', 'test_ood'):
            fs = sorted(glob.glob(os.path.join(DATA, d, sp, '*', '*', '*.npz')))[::max(1, len(glob.glob(os.path.join(DATA, d, sp, '*', '*', '*.npz'))) // 12)][:12]
            e_mix, e_sir, e_snr, e_ber, n = [], [], [], [], 0
            for f in fs:
                z = np.load(f)
                for i in range(min(5, len(z['clean_id']))):
                    c, j, nz, mix = z['clean_signal'][i], z['interference_signal'][i], z['noise_signal'][i], z['mixture_signal'][i]
                    e_mix.append(float(np.abs(mix - (c + j + nz)).max()))
                    L = len(c); mask = np.abs(np.fft.fftfreq(L)) <= fm_of(float(z['rolloff'][i]))
                    P = lambda x: (np.abs(np.fft.fft(x)[mask]) ** 2).sum() / L ** 2
                    e_sir.append(abs(10 * np.log10(P(c) / P(j)) - float(z['sir_db_inband'][i])))
                    e_snr.append(abs(10 * np.log10(P(c) / P(nz)) - float(z['snr_db_inband'][i])))
                    mod = str(z['modulation'][i])
                    y = matched_filter(c, float(z['rolloff'][i]), float(z['timing_offset'][i]), float(z['cfo'][i]), float(z['phase'][i]), SPS, NSYM, SPAN) / float(z['demod_gain'][i])
                    bits = symbols_to_bits(y, mod); ref = z['bit_stream'][i]
                    bps = {'BPSK': 1, 'QPSK': 2, '16QAM': 4}[mod]; k0, k1 = 6 * bps, (NSYM - 6) * bps
                    e_ber.append(float(np.mean(bits[k0:k1] != ref[k0:k1]))); n += 1
            res.append(dict(dataset=d, split=sp, n=n, max_mix_err=max(e_mix), sir_err_mean=float(np.mean(e_sir)), sir_err_max=float(np.max(e_sir)),
                            snr_err_max=float(np.max(e_snr)), clean_ber_max=float(np.max(e_ber))))
    return res


def repro(runs, tol=0.002):
    out = []
    for r in runs:
        ref_f = os.path.join(RUNS, r, 'per_sample_test_avg8.npz')
        if not os.path.exists(ref_f):
            out.append((r, None, None, 'no reference')); continue
        ref = np.load(ref_f)
        t0 = time.time()
        subprocess.run([sys.executable, '-W', 'ignore', os.path.join(HERE, 'evaluate_new.py'), '--runs', r, '--splits', 'test', '--stride', '2', '--avg_k', '8', '--tag', '_audit'],
                       capture_output=True, text=True, cwd=HERE)
        f = os.path.join(RUNS, r, 'per_sample_test_audit.npz')
        if not os.path.exists(f):
            out.append((r, float(ref['ber_model'].mean()), None, 'eval failed')); continue
        new = np.load(f)
        os.remove(f)
        d = abs(float(new['ber_model'].mean()) - float(ref['ber_model'].mean()))
        out.append((r, float(ref['ber_model'].mean()), float(new['ber_model'].mean()), 'PASS' if d <= tol else 'FAIL (|Δ|=%.4f)' % d, round(time.time() - t0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro', nargs='*', default=['zs2_ex_TF'])
    ap.add_argument('--skip_repro', action='store_true')
    a = ap.parse_args()
    os.makedirs(REPORTS, exist_ok=True)
    rows = index_runs()
    with open(os.path.join(REPORTS, 'experiment_index.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sp = check_splits(); defs = check_definitions()
    rp = [] if a.skip_repro else repro(a.repro)
    mask_line = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(HERE, 'tests', 'test_router_masks.py')], capture_output=True, text=True, cwd=os.path.join(HERE, 'tests'))
    L = ['# 基线审计（P0）', '', '- 生成时间：%s；git 版本：%s；实验数：%d（详见 experiment_index.csv）' % (time.strftime('%Y-%m-%d %H:%M'), git_sha(), len(rows)), '',
         '## 1. 划分检查（clean_id 两两不相交）', '', '| 数据集 | 各划分 clean_id 数 | 结果 |', '|---|---|---|']
    for d, cnt, bad in sp:
        L.append('| %s | %s | %s |' % (d, cnt, 'PASS' if not bad else 'FAIL %s' % bad))
    L += ['', '## 2. 数据定义检查（v2 抽样）', '', '| 划分 | 抽样数 | max\\|mixture−(clean+interf+noise)\\| | 带内SIR误差 均值/最大(dB) | 带内SNR误差 最大(dB) | 干净信号解调 BER 最大 |', '|---|---|---|---|---|---|']
    for r in defs:
        L.append('| %s | %d | %.2e | %.4f / %.4f | %.4f | %.4f |' % (r['split'], r['n'], r['max_mix_err'], r['sir_err_mean'], r['sir_err_max'], r['snr_err_max'], r['clean_ber_max']))
    L += ['', '## 3. 复现检查（evaluate_new，同样本/同种子/avg8）', '', '| run | 已保存 BER | 重算 BER | 结果 |', '|---|---|---|---|']
    for r in rp:
        L.append('| %s | %s | %s | %s |' % (r[0], '%.4f' % r[1] if r[1] is not None else '-', '%.4f' % r[2] if r[2] is not None else '-', r[3]))
    if a.skip_repro:
        L.append('| (已跳过) | | | |')
    L += ['', '## 4. 路径与屏蔽说明', '',
          '- 训练：`PhysicsMoEModel._training_forward` 调用 `compute_moe_condition(training=True)`（路由器带噪声/Gumbel）；采样与评估走 `sample()`，其中 `compute_moe_condition(training=False)`，路由无噪声。',
          '- 评估使用 EMA 参数、`model.eval()`；逐样本结果含 `ber_mf`（未恢复输入的 BER），因此 ΔBER 与负增益比例可由 aggregate_results.py 直接得到。',
          '- Top-K 屏蔽：`' + (mask_line.stdout.strip().splitlines() or ['(测试失败)'])[-1] + '`',
          '- 旧“无 MoE”基线传入的是全零伪 token（经交叉注意力的偏置投影仍产生非零输出）；`--true_uncond` 才是真正的无条件分支。已有的 zsK_dit 系列基线用的是伪零 token 版本。', '']
    open(os.path.join(REPORTS, 'baseline_audit.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
