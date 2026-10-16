"""实验 13：扩散采样稳定性与推理成本（不重训）。
对每个 run 扫描 DDIM 步数 × 平均次数（avg_k），同时报告 BER 与 MSE（分别报告：多次平均可能改善 MSE 却改变星座判决）、
单样本端到端时延（含条件提取与全部采样步）、以及 avg_k=1 时不同随机种子之间 BER 的波动（采样可靠性）。
用法: python sampling_ablation.py --runs zs2_ex_TF zs2_ditL --steps 10 25 50 100 --avg 1 8 --seeds 3 --stride 4 --out ../reports/sampling_ablation.csv"""
import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate_new as EV  # noqa: E402
from metrics_new import batch_ber  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402


def run_one(model, ds, scale, dev, steps, k, seed, bs=64, amp=True):
    torch.manual_seed(seed)
    bers, mses, tsum = [], [], 0.0
    for i in range(0, len(ds.samples), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds.samples)))]
        b = {kk: torch.stack([it[kk] for it in items]) for kk in items[0]}
        y = b['noisy'].to(dev) / scale
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            rec = torch.stack([model.sample(y, num_steps=steps).float() for _ in range(k)]).mean(0) * scale
        torch.cuda.synchronize(); tsum += time.time() - t0
        bers += batch_ber(rec.cpu(), b, ds.meta, ds.mod_names).tolist()
        mses += ((rec.cpu() - b['clean']) ** 2).mean(dim=(1, 2)).tolist()
    return float(np.mean(bers)), float(np.mean(mses)), tsum / len(ds.samples) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--steps', type=int, nargs='+', default=[10, 25, 50, 100])
    ap.add_argument('--avg', type=int, nargs='+', default=[1, 8])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--split', default='test')
    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda')
    rows = []
    for run in a.runs:
        ds = MoEDiffDataset(a.data_root, a.split)
        ds.samples = ds.samples[::a.stride]
        model, scale, mt = EV.build_model(run, ds.meta, dev)
        for st in a.steps:
            for k in a.avg:
                res = [run_one(model, ds, scale, dev, st, k, 1000 + s, amp=bool(a.amp)) for s in range(a.seeds if k == 1 else 1)]
                ber = np.mean([r[0] for r in res])
                rows.append(dict(run=run, steps=st, avg_k=k, ber=ber, ber_seed_std=float(np.std([r[0] for r in res], ddof=1)) if len(res) > 1 else float('nan'),
                                 mse=float(np.mean([r[1] for r in res])), ms_per_sample=float(np.mean([r[2] for r in res])), n=len(ds.samples)))
                print({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in rows[-1].items()}, flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
