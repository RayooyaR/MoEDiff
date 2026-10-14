"""控制变量对照：MoE 的收益来自哪里？（容量？扩散目标？Transformer 骨干？多分支特征？）

用法（在 new/moe_diffic 下）： python ../analysis/report_controls.py
对每种留出设置 zs2/zs1/zs3，比较（测试集分为 A.全部见过类型 / B.全部未见类型）：
  CNN；CNN 加宽(2.5M)；多分支特征+CNN(MV-CNN, 2.7M)；DiT 扩散(256, 7.0M)；DiT 扩散(272, 8.0M, 容量匹配)；MoE-DiffIC 扩散(7.8M)；DiT 直接回归(7.0M)；MoE 直接回归(7.8M)
扩散类用随机采样取平均 8 次（*_avg8.npz，测试集每隔一条的一半样本），回归类取同一批样本。
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs')
SEEN = {'zs1': ['NB', 'WB', 'LFM', 'NFM', 'RFM', 'PL'], 'zs2': ['ST', 'MT', 'NB', 'WB', 'PL'], 'zs3': ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM']}
LAB = {'zs2': 'zs2 去调频', 'zs1': 'zs1 去单音/多音', 'zs3': 'zs3 去脉冲'}
METRICS = [('ber', 'BER（越低越好）', '%.4f'), ('evm', 'EVM（越低越好）', '%.3f'), ('mse', 'MSE（越低越好）', '%.3f'), ('mae', 'MAE（越低越好）', '%.3f'),
           ('sinr_inband', '带内SINR dB（越高越好）', '%.2f')]
# (显示名, run 后缀, 文件后缀, 类型: avg=已是 stride2, sub=需取 [::2])
ROWS = [('CNN (1.6M)', 'cnn', '', 'sub'), ('CNN 加宽 base40 (2.5M)', 'cnnL', '', 'sub'), ('多分支特征+CNN (MV-CNN 2.7M)', 'mvcnn', '', 'sub'),
        ('DiT 扩散 256 (7.0M)', 'dit', '_avg8', 'avg'), ('DiT 扩散 272 容量匹配 (8.0M)', 'ditL', '_avg8', 'avg'),
        ('MoE-DiffIC 扩散 (7.8M)', 'moe', '_avg8', 'avg'), ('DiT 直接回归 (7.0M)', 'dreg', '', 'sub'), ('MoE 直接回归 (7.8M)', 'mreg', '', 'sub'),
        ('  参照 DiT-272 单次随机', 'ditL', '', 'sub'), ('  参照 DiT-272 确定性', 'ditL', '_det', 'sub')]


def load(run, suf):
    f = os.path.join(RUNS, run, 'per_sample_test%s.npz' % suf)
    return {k: v for k, v in np.load(f).items()} if os.path.exists(f) else None


def get(z, row):
    d = load('%s_%s' % (z, row[1]), row[2])
    if d is None or row[3] == 'avg':
        return d
    return {k: (v[::2] if getattr(v, 'ndim', 0) == 1 and len(v) == len(d['type']) else v) for k, v in d.items()}


def mask(d, z, seen):
    types = [str(t) for t in d['type_names']]
    uns = [i for i, t in enumerate(types) if t not in SEEN[z]]
    return (~np.isin(d['type'], uns)) if seen else np.isin(d['type'], uns)


def sig(a, b):
    d = a - b; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return '%+.4f%s' % (m, '▲' if m - 1.96 * se > 0 else ('▼' if m + 1.96 * se < 0 else '·'))


def main():
    sets = ['zs2', 'zs1', 'zs3']
    for title, seen in (('A. 测试集全部为【训练见过的干扰类型】', True), ('B. 测试集全部为【训练从未见过的干扰类型】', False)):
        print('\n' + '#' * 110 + '\n' + title + '\n' + '#' * 110)
        for key, mt, fmt in METRICS:
            print('\n[%s]' % mt)
            print('%-32s' % '' + ''.join('%16s' % LAB[z] for z in sets))
            vals = {r[0]: [] for r in ROWS}
            for z in sets:
                for r in ROWS:
                    d = get(z, r)
                    vals[r[0]].append(np.nan if d is None else d[key + '_model'][mask(d, z, seen)].mean())
            main_rows = [r[0] for r in ROWS if not r[0].startswith('  ')]
            best = [(np.nanmax if key == 'sinr_inband' else np.nanmin)([vals[n][i] for n in main_rows]) for i in range(len(sets))]
            for r in ROWS:
                s = '%-32s' % r[0]
                for i, v in enumerate(vals[r[0]]):
                    mark = '*' if (r[0] in main_rows and not np.isnan(v) and abs(v - best[i]) < 1e-12) else ' '
                    s += ('--' if np.isnan(v) else (fmt % v) + mark).rjust(16)
                print(s)
    print('\n' + '#' * 110 + '\nC. 关键对比的配对 BER 差（左 − 右；负=左更好；▲▼显著）\n' + '#' * 110)
    pairs = [('MoE 扩散 − DiT-272 扩散', 5, 4, '容量匹配后 MoE 是否仍优于同容量 DiT'), ('MoE 扩散 − DiT-256 扩散', 5, 3, '（对照）'),
             ('DiT-272 扩散 − DiT-256 扩散', 4, 3, '加容量本身的作用'), ('DiT 回归 − DiT 扩散(256)', 6, 3, '去掉扩散（同骨干）'),
             ('MoE 回归 − MoE 扩散', 7, 5, '去掉扩散（带 MoE）'), ('MoE 回归 − DiT 回归', 7, 6, '回归下 MoE 是否有益'),
             ('DiT 回归 − CNN', 6, 0, '同训练目标下 Transformer 骨干 vs 卷积'),
             ('MV-CNN − CNN', 2, 0, '给 CNN 加多分支特征是否有益'), ('MV-CNN − CNN加宽', 2, 1, '同参数量：多分支 vs 单纯加宽'),
             ('CNN加宽 − CNN', 1, 0, '加宽本身的作用'), ('MV-CNN − MoE 扩散', 2, 5, '多分支+CNN 能否追平 MoE-DiffIC')]
    for seen, lab in ((False, '未见类型'), (True, '见过类型')):
        print('\n[%s]' % lab)
        print('%-30s' % '' + ''.join('%16s' % LAB[z] for z in sets) + '   含义')
        for name, a, b, note in pairs:
            s = '%-30s' % name
            for z in sets:
                da, db = get(z, ROWS[a]), get(z, ROWS[b])
                if da is None or db is None:
                    s += '--'.rjust(16); continue
                m = mask(da, z, seen)
                s += sig(da['ber_model'][m], db['ber_model'][m]).rjust(16)
            print(s + '   ' + note)


if __name__ == '__main__':
    main()
