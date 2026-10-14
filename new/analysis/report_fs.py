"""小样本整合报告：增强 vs 无增强 × （全部见过类型 / 全部未见类型）× 所有方法 × 所有指标。

用法（在 new/moe_diffic 下）： python ../analysis/report_fs.py
扩散类方法（DiT/ICDM、DiffCom、ICDM双模型、MoE-DiffIC）的主结果 = 随机采样取平均（每样本独立采样 8 次再平均，runs/<名>/per_sample_test_avg8.npz，
使用测试集每隔一条的一半样本）；非扩散方法取同一批样本。附加行给出 DiT/MoE 的单次随机采样与确定性采样作为参照。
训练：zs2 场景（见过类型 ST/MT/NB/WB/PL，未见类型 LFM/NFM/RFM），每格 K 条，增强 = 旋转/循环时移/I-Q 微拉伸/仅输入侧加噪声。
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
SEEN = ['ST', 'MT', 'NB', 'WB', 'PL']
KS = [1, 5, 10]
METRICS = [('ber', 'BER（越低越好）', '%.4f'), ('evm', 'EVM（越低越好）', '%.3f'), ('mse', 'MSE（越低越好）', '%.3f'), ('mae', 'MAE（越低越好）', '%.3f'),
           ('sinr_inband', '带内SINR dB（越高越好）', '%.2f')]
ROWS = [  # (显示名, 后缀类型, run 后缀, 文件后缀)
    ('MF 匹配滤波', 'mf', None, ''), ('LMMSE', 'sub', 'lmmse', ''), ('CNN', 'sub', 'cnn', ''),
    ('DiT扩散(ICDM) ·平均8次', 'avg', 'dit', '_avg8'), ('DiffCom ·平均8次', 'avg', 'diffcom', '_avg8'),
    ('ICDM双模型 ·平均8次', 'avg', 'icdm', '_avg8'), ('MoE-DiffIC ·平均8次', 'avg', 'moe', '_avg8'),
    ('  参照 DiT 单次随机', 'sub', 'dit', ''), ('  参照 DiT 确定性', 'sub', 'dit', '_det'),
    ('  参照 MoE 单次随机', 'sub', 'moe', ''), ('  参照 MoE 确定性', 'sub', 'moe', '_det')]


def load(name, suf):
    f = os.path.join(RUNS, name, 'per_sample_test%s.npz' % suf)
    return {k: v for k, v in np.load(f).items()} if os.path.exists(f) else None


def get(row, K, aug):
    disp, kind, run, suf = row
    pre = 'fs_K%d%s' % (K, '' if aug else '_na')
    if kind == 'mf':
        d = load(pre + '_cnn', '')
        return None if d is None else {k.replace('_mf', '_model'): v[::2] for k, v in d.items() if k.endswith('_mf') or k in ('type', 'sir', 'mod')}
    d = load('%s_%s' % (pre, run), suf)
    if d is None:
        return None
    return d if kind == 'avg' else {k: (v[::2] if getattr(v, 'ndim', 0) == 1 and len(v) == len(d['type']) else v) for k, v in d.items()}


def main():
    ref = get(ROWS[2], 1, True)
    if ref is None:
        print('尚无结果'); return
    types = [str(t) for t in ref['type_names']]
    unseen_idx = [i for i, t in enumerate(types) if t not in SEEN]
    cols = [(K, a) for K in KS for a in (True, False)]
    for scope, want_seen in (('A. 测试集全部为【训练见过的干扰类型】（新样本，类型：%s）' % SEEN, True), ('B. 测试集全部为【训练从未见过的干扰类型】（%s）' % [types[i] for i in unseen_idx], False)):
        print('\n' + '#' * 110 + '\n' + scope + '\n' + '#' * 110)
        for key, title, fmt in METRICS:
            print('\n[%s]' % title)
            print('%-26s' % '' + ''.join('%11s' % ('K%d·%s' % (K, '增强' if a else '无增强')) for K, a in cols))
            vals = {}
            for row in ROWS:
                line = []
                for K, a in cols:
                    d = get(row, K, a)
                    if d is None:
                        line.append(np.nan); continue
                    T = d['type']
                    m = ~np.isin(T, unseen_idx) if want_seen else np.isin(T, unseen_idx)
                    line.append(d[key + '_model'][m].mean())
                vals[row[0]] = line
            main_rows = [r[0] for r in ROWS if not r[0].startswith('  ')]
            best = [(np.nanmax if key == 'sinr_inband' else np.nanmin)([vals[n][i] for n in main_rows]) for i in range(len(cols))]
            for row in ROWS:
                n = row[0]
                s = '%-26s' % n
                for i, v in enumerate(vals[n]):
                    mark = '*' if (n in main_rows and not np.isnan(v) and abs(v - best[i]) < 1e-12) else ' '
                    s += (('--' if np.isnan(v) else (fmt % v) + mark)).rjust(11)
                print(s)
    print('\n' + '#' * 110 + '\nC. MoE-DiffIC(平均8次) 相对 CNN 的 BER 差（CNN − MoE；正=MoE 更好；▲▼显著）\n' + '#' * 110)
    print('%-12s' % '' + ''.join('%14s' % ('K%d·%s' % (K, '增强' if a else '无增强')) for K, a in cols))
    for lab, want_seen in (('见过类型', True), ('未见类型', False)):
        s = '%-12s' % lab
        for K, a in cols:
            m_, c_ = get(ROWS[6], K, a), get(ROWS[2], K, a)
            if m_ is None or c_ is None:
                s += '--'.rjust(14); continue
            T = m_['type']; mk = ~np.isin(T, unseen_idx) if want_seen else np.isin(T, unseen_idx)
            d = c_['ber_model'][mk] - m_['ber_model'][mk]; mu = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
            s += ('%+.4f%s' % (mu, '▲' if mu - 1.96 * se > 0 else ('▼' if mu + 1.96 * se < 0 else '·'))).rjust(14)
        print(s)
    print('\nD. 增强的贡献（未见类型 BER：增强 − 无增强；负=增强更好）')
    print('%-26s' % '' + ''.join('%11s' % ('K=%d' % K) for K in KS))
    for row in ROWS[1:7]:
        s = '%-26s' % row[0]
        for K in KS:
            a, b = get(row, K, True), get(row, K, False)
            if a is None or b is None:
                s += '--'.rjust(11); continue
            mk = np.isin(a['type'], unseen_idx)
            s += ('%+.4f' % (a['ber_model'][mk].mean() - b['ber_model'][mk].mean())).rjust(11)
        print(s)


if __name__ == '__main__':
    main()
