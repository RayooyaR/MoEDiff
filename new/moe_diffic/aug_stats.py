"""实验 14：报告各增强算子作用后“有效”的带内 SIR / SNR 分布（由数组重新计算），以及与原始分布的差异。
用法: python aug_stats.py --ops none rot shift awgn iqgain remix rot,remix --n 600 --out ../reports/e14_aug_stats.csv"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
import numpy as np  # noqa: E402

from moe_dataset import MoEDiffDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ops', nargs='+', default=['none', 'rot', 'shift', 'awgn', 'iqgain', 'remix', 'rot,remix', 'remix,awgn'])
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--types', nargs='*', default=['ST', 'MT', 'NB', 'WB', 'PL'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = []
    for op in a.ops:
        ds = MoEDiffDataset(a.data_root, 'train', types=a.types, max_per_cell=4, aug_ops=None if op == 'none' else op)
        idx = np.random.RandomState(0).choice(len(ds), min(a.n, len(ds)), replace=False)
        sir = np.array([float(ds[i]['sir_db']) for i in idx]); snr = np.array([float(ds[i]['snr_db']) for i in idx])
        rows.append(dict(ops=op, n=len(idx), sir_mean=sir.mean(), sir_std=sir.std(), sir_min=sir.min(), sir_max=sir.max(), snr_mean=snr.mean(), snr_std=snr.std(), snr_min=snr.min(), snr_max=snr.max()))
        print('%-12s SIR %6.2f±%5.2f [%6.2f,%6.2f] | SNR %6.2f±%5.2f [%6.2f,%6.2f]' % (op, sir.mean(), sir.std(), sir.min(), sir.max(), snr.mean(), snr.std(), snr.min(), snr.max()), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
