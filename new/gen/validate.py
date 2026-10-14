"""数据验收：  python validate.py [数据目录，默认 ../data/v2]

输出：控制台表格 + ../reports/<目录名>_report.json + ../reports/<目录名>_psd.png
解调为"已知同步参数"的匹配滤波（oracle 同步），只衡量干扰+噪声的影响。
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

import config as C
import signal_gen as SG

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'v2')
NAME = os.path.basename(ROOT.rstrip(os.sep))
REPORT_DIR = os.path.join(HERE, '..', 'reports')


def load_all():
    S = []
    for f in sorted(glob.glob(os.path.join(ROOT, '*', '*', '*', '*.npz'))):
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        parts = f.split(os.sep)
        for i in range(len(d['clean_signal'])):
            e = {k: v[i] for k, v in d.items()}
            e['split'], e['type'], e['mod'] = parts[-4], parts[-3], parts[-2]
            S.append(e)
    return S


def inband(x, fmax):
    X = np.fft.fft(x)
    m = np.abs(np.fft.fftfreq(len(x))) <= fmax
    return (np.abs(X[m]) ** 2).sum() / len(x) ** 2


def args(s):
    return (s['mod'], float(s['rolloff']), float(s['timing_offset']), float(s['cfo']), float(s['phase']), float(s['demod_gain']))


def ber(s, x):
    return SG.ber_interior(SG.demod_oracle(x, *args(s)), s['bit_stream'], s['mod'])


def grid(title, rows, rowkeys, sirs=None):
    sirs = sirs or C.SIR_DB
    print('\n' + title)
    print('%-9s' % '' + ' '.join('%7g' % s for s in sirs))
    for r in rowkeys:
        print('%-9s' % r + ' '.join(('%.3f' % np.mean(rows[(r, s)])).rjust(7) if (r, s) in rows else '    n/a' for s in sirs))


def main():
    S = load_all()
    cnt = {k: sum(1 for s in S if s['split'] == k) for k in C.SPLITS}
    print('样本总数 %d；split: %s' % (len(S), cnt))
    print('类型: %s；调制: %s' % (sorted({s['type'] for s in S}), sorted({s['mod'] for s in S})))
    rep = {'counts': cnt}

    # 1. 字段一致性
    res = max(np.linalg.norm(s['mixture_signal'] - s['clean_signal'] - s['interference_signal'] - s['noise_signal'])
              / np.linalg.norm(s['mixture_signal']) for s in S)
    print('\n[1] mixture - clean - interference - noise 最大相对残差: %.2e (complex64 精度约 1e-7)' % res)
    rep['field_residual_max'] = float(res)

    # 2. 独立重算带内 SIR / SNR
    e_sir, e_snr = [], []
    for s in S:
        fmax = (1 + float(s['rolloff'])) / (2 * C.SPS)
        ps = inband(s['clean_signal'].astype(np.complex128), fmax)
        e_sir.append(10 * np.log10(ps / inband(s['interference_signal'].astype(np.complex128), fmax)) - float(s['sir_db_target']))
        e_snr.append(10 * np.log10(ps / inband(s['noise_signal'].astype(np.complex128), fmax)) - float(s['snr_db_target']))
    print('[2] 重算带内 SIR 与目标最大偏差 %.4f dB；带内 SNR 偏差：均值 %+.2f dB, 最大 |.| %.2f dB（噪声为随机实现，有涨落）'
          % (np.max(np.abs(e_sir)), np.mean(e_snr), np.max(np.abs(e_snr))))
    rep['sir_err_max_db'] = float(np.max(np.abs(e_sir)))
    rep['snr_err_mean_db'] = float(np.mean(e_snr))

    # 3. 划分互不相交 + 每条样本 clean_id 唯一
    ids = {k: [int(s['clean_id']) for s in S if s['split'] == k] for k in C.SPLITS}
    ks = list(ids)
    ov = {'%s&%s' % (a, b): len(set(ids[a]) & set(ids[b])) for i, a in enumerate(ks) for b in ks[i + 1:]}
    dup = sum(len(v) - len(set(v)) for v in ids.values())
    print('[3] clean_id 两两交集:', ov, '| split 内重复 clean_id 数:', dup)
    rep['clean_id_overlap'] = ov
    rep['clean_id_dup'] = dup

    # 4. 干净信号自检：各调制 oracle 解调 BER 必须为 0；符号/比特一致
    bad = defaultdict(float)
    for s in S[::5]:
        bad[s['mod']] = max(bad[s['mod']], ber(s, s['clean_signal']))
    sym_ok = all(np.array_equal(SG.symbols_to_bits(s['symbol_stream'].astype(np.complex128), s['mod']), s['bit_stream']) for s in S[::50])
    print('[4] 干净信号 oracle 解调 BER 最大值（抽样，按调制）:', dict(bad), '| symbol_stream 与 bit_stream 一致:', sym_ok)
    rep['clean_ber_max'] = dict(bad)

    # 5. 干扰带内占比 / 峰值
    print('\n[5] 干扰能量落在信号带内的比例（中位数）/ 混合信号峰值幅度（中位 / 最大）')
    for t in C.TYPES:
        xs = [s for s in S if s['type'] == t]
        print('  %-4s 带内占比 %.2f | 峰值 %.1f / %.1f' % (
            t, np.median([float(s['interference_inband_frac']) for s in xs]),
            np.median([np.abs(s['mixture_signal']).max() for s in xs]), max(np.abs(s['mixture_signal']).max() for s in xs)))

    # 6. MF BER
    for split in ('test', 'test_ood'):
        by_t, by_m, by_tm = defaultdict(list), defaultdict(list), defaultdict(list)
        for s in S:
            if s['split'] != split:
                continue
            b = ber(s, s['mixture_signal'])
            sir = int(s['sir_db_target'])
            by_t[(s['type'], sir)].append(b)
            by_m[(s['mod'], sir)].append(b)
        grid('[6] %s —— MF BER：行=干扰类型（对三种调制取平均），列=带内 SIR dB' % split, by_t, C.TYPES)
        grid('[6] %s —— MF BER：行=调制（对全部干扰类型取平均）' % split, by_m, C.MODS)
        rep['mf_ber_by_type_' + split] = {'%s@%d' % k: float(np.mean(v)) for k, v in by_t.items()}
        rep['mf_ber_by_mod_' + split] = {'%s@%d' % k: float(np.mean(v)) for k, v in by_m.items()}
    # 6b. 去噪声：干扰单独的影响（test）
    by_t0 = defaultdict(list)
    for s in S:
        if s['split'] == 'test':
            by_t0[(s['type'], int(s['sir_db_target']))].append(ber(s, s['mixture_signal'] - s['noise_signal']))
    grid('[6b] test —— MF BER（去掉噪声，仅干扰影响；行=干扰类型）', by_t0, C.TYPES)

    # 6c. test 与 test_ood 逐格差异（类型×SIR，对调制取平均）
    d = max(abs(np.mean(rep_a) - np.mean(rep_b)) for rep_a, rep_b in [(a, b) for a, b in
            [(rep['mf_ber_by_type_test'][k], rep['mf_ber_by_type_test_ood'][k]) for k in rep['mf_ber_by_type_test']]])
    print('\n[6c] test 与 test_ood 的 MF BER 逐格（类型×SIR）最大差异: %.3f' % d)
    diffs = sorted(((abs(rep['mf_ber_by_type_test'][k] - rep['mf_ber_by_type_test_ood'][k]), k) for k in rep['mf_ber_by_type_test']), reverse=True)[:4]
    print('     差异最大的 4 格:', [('%s' % k, round(v, 3)) for v, k in diffs])
    rep['test_vs_ood_max_diff'] = float(d)

    # 7. 无干扰仅噪声的 BER（按调制）
    bn = defaultdict(list)
    for s in S:
        if s['split'] == 'test':
            bn[s['mod']].append(ber(s, s['clean_signal'] + s['noise_signal']))
    print('\n[7] 只有噪声（无干扰）MF BER（test，按调制）:', {k: round(float(np.mean(v)), 4) for k, v in bn.items()})
    rep['noise_only_ber_test'] = {k: float(np.mean(v)) for k, v in bn.items()}

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, '%s_report.json' % NAME), 'w') as f:
        json.dump(rep, f, indent=2)

    # 8. PSD 图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 4, figsize=(20, 7))
        for a, t in zip(ax.ravel(), C.TYPES):
            xs = [s for s in S if s['split'] == 'test' and s['type'] == t and s['mod'] == 'QPSK' and int(s['sir_db_target']) == -10][:3]
            f = np.fft.fftshift(np.fft.fftfreq(C.SIG_LEN))
            for s in xs:
                a.plot(f, 10 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(s['interference_signal'])) ** 2) / C.SIG_LEN + 1e-12), lw=0.6)
                fmax = (1 + float(s['rolloff'])) / (2 * C.SPS)
                a.axvline(-fmax, color='k', ls=':', lw=0.8)
                a.axvline(fmax, color='k', ls=':', lw=0.8)
            a.plot(f, 10 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(xs[0]['clean_signal'])) ** 2) / C.SIG_LEN + 1e-12),
                   color='gray', lw=0.6, alpha=0.7)
            a.set_title('%s (SIR_in=-10 dB; gray = clean)' % t, fontsize=9)
            a.set_xlabel('f [cycles/sample]')
            a.set_ylabel('PSD [dB]')
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, '%s_psd.png' % NAME), dpi=100)
        # 星座图（干净符号 + 匹配滤波后的混合信号）
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        for a, m in zip(ax, C.MODS):
            s = [x for x in S if x['split'] == 'test' and x['mod'] == m and x['type'] == 'ST' and int(x['sir_db_target']) == 10][0]
            y = SG.matched_filter(s['mixture_signal'], *args(s)[1:5]) / float(s['demod_gain'])
            k = slice(C.EDGE_SKIP, C.N_SYM - C.EDGE_SKIP)
            a.scatter(y.real[k], y.imag[k], s=6, alpha=0.6)
            a.scatter(s['symbol_stream'].real, s['symbol_stream'].imag, s=40, marker='x', color='r')
            a.set_title('%s after MF (ST, SIR=+10 dB); red x = ideal' % m, fontsize=9)
            a.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, '%s_constellation.png' % NAME), dpi=100)
        print('\n图: reports/%s_psd.png, reports/%s_constellation.png' % (NAME, NAME))
    except Exception as e:
        print('绘图跳过:', e)


if __name__ == '__main__':
    main()
