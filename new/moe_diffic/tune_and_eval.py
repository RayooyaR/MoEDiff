"""在验证集上为 DiffCom 的 ζ / ICDM 双模型的 γ 调参，再用最优值在 test 上评估。

用法： python tune_and_eval.py --run zs2_diffcom --kind diffcom [--data_root ...] [--splits test]
调参只用 val（val 只含训练见过的类型，不泄漏）；最优值落在网格边界时向外扩一档。结果写入 runs/<run>/chosen.json。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# 留出整类实验中各模型训练时见过的类型；调参只在这些类型的验证样本上进行（val 中的未见类型不参与调参）
SEEN = {'zs1': ['NB', 'WB', 'LFM', 'NFM', 'RFM', 'PL'], 'zs2': ['ST', 'MT', 'NB', 'WB', 'PL'],
        'zs3': ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM'], 'zs4': ['ST', 'MT', 'PL', 'LFM', 'NFM', 'RFM'],
        'fs': ['ST', 'MT', 'NB', 'WB', 'PL']}


def seen_types(run):
    return SEEN.get(run.split('_')[0])


def ev(run, split, flag, val, tag, data_root, extra=()):
    cmd = [PY, '-W', 'ignore', 'evaluate_new.py', '--runs', run, '--splits', split, '--tag', tag] + ([flag, str(val)] if flag else []) + list(extra)
    if split == 'val' and seen_types(run):
        cmd += ['--types'] + seen_types(run)
    if data_root:
        cmd += ['--data_root', data_root]
    subprocess.run(cmd, check=True, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.join(HERE, 'runs', run, 'per_sample_%s%s.npz' % (split, tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--kind', choices=['diffcom', 'icdm'], required=True)
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--splits', nargs='+', default=['test'])
    ap.add_argument('--warm_start', action='store_true', help='DiffCom：官方式热启动')
    ap.add_argument('--val_stride', type=int, default=1, help='验证集调参时每隔几条取一条')
    ap.add_argument('--final_avg8', action='store_true', help='最终评估直接做平均 8 次采样（stride 2，tag _avg8），不再做单次采样评估')
    ap.add_argument('--amp', default='none')
    a = ap.parse_args()
    base_extra = (['--warm_start'] if a.warm_start else []) + ['--amp', a.amp]
    flag, grid = ('--dps_step', [10, 30, 100]) if a.kind == 'diffcom' else ('--dual_gamma', [0.5, 0.75, 1.0])
    seen = {}

    def score(v):
        if v not in seen:
            f = ev(a.run, 'val', flag, v, '_tune%g' % v, a.data_root, extra=base_extra + ['--stride', str(a.val_stride)])
            seen[v] = float(np.load(f)['ber_model'].mean())
        return seen[v]
    for _ in range(3):                                   # 最多向外扩 3 次
        for v in grid:
            score(v)
        best = min(seen, key=seen.get)
        if best == min(grid):
            grid = [best / 3.0 if a.kind == 'diffcom' else round(best - 0.25, 3)] + grid
            if grid[0] <= 0:
                break
        elif best == max(grid):
            grid = grid + [best * 3.0 if a.kind == 'diffcom' else round(best + 0.5, 3)]
        else:
            break
    best = min(seen, key=seen.get)
    json.dump({'kind': a.kind, 'chosen': best, 'val_ber': seen}, open(os.path.join(HERE, 'runs', a.run, 'chosen.json'), 'w'), indent=1)
    for sp in a.splits:
        if a.final_avg8:
            ev(a.run, sp, flag, best, '_avg8', a.data_root, extra=base_extra + ['--stride', '2', '--avg_k', '8'])
        else:
            ev(a.run, sp, flag, best, '', a.data_root, extra=base_extra)
    print(a.run, a.kind, 'chosen', best, seen)


if __name__ == '__main__':
    main()
