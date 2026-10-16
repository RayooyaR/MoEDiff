"""实验 12：采样期物理一致性引导的评估。
  python eval_guidance.py --run <扩散 run> [--est <干扰估计器 run>] --w_obs 0 0.3 1 --w_band 0 1 --split val|test --stride 4
对每个 (w_obs, w_band) 组合输出 BER/MSE（权重只在 val 上选，test 只报告选定权重）；同时记录每批耗时（引导需要对去噪网络求梯度，推理开销显著更高）。
干扰估计器用 `exp_train.py --model_type cnn --target interf` 训练（学习目标是干扰信号，输入是 y）。"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate_new as EV  # noqa: E402
from metrics_new import batch_ber  # noqa: E402
from models.observation_prior import GuidedSampler  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--est', default=None)
    ap.add_argument('--w_obs', type=float, nargs='+', default=[0.0])
    ap.add_argument('--w_band', type=float, nargs='+', default=[0.0])
    ap.add_argument('--split', default='val')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--ddim_steps', type=int, default=25)
    ap.add_argument('--avg_k', type=int, default=1)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda')
    ck = torch.load(os.path.join(HERE, 'runs', a.run, 'best_model.pth'), map_location='cpu', weights_only=False)
    targs = ck['args']
    types = None if a.split != 'val' else ([t for t in (targs.get('new_types') or []) if not (targs.get('holdout') and t in targs['holdout'])] or None)
    ds = MoEDiffDataset(a.data_root, a.split, types=types, noise_mode=targs.get('noise_mode', 'with'))
    ds.samples = ds.samples[::a.stride]
    model, scale, mt = EV.build_model(a.run, ds.meta, dev)
    est = None
    if a.est:
        est, escale, _ = EV.build_model(a.est, ds.meta, dev)
    rows = []
    for wo in a.w_obs:
        for wb in a.w_band:
            if wo > 0 and est is None:
                continue
            torch.manual_seed(0); bers, mses, t0 = [], [], time.time()
            for i in range(0, len(ds.samples), 32):
                items = [ds[j] for j in range(i, min(i + 32, len(ds.samples)))]
                b = {k: torch.stack([it[k] for it in items]) for k in items[0]}
                y = b['noisy'].to(dev) / scale
                i_hat = (est.sample(y * scale / escale) * escale / scale) if est is not None else None
                recs = []
                for _ in range(a.avg_k):
                    smp = GuidedSampler(model, model.scheduler, w_obs=wo, w_band=wb, i_hat=i_hat, eta=0.0)
                    recs.append(smp.sample(y, num_steps=a.ddim_steps).detach())
                rec = torch.stack(recs).mean(0) * scale
                bers += batch_ber(rec.cpu(), b, ds.meta, ds.mod_names).tolist()
                mses.append(float(((rec.cpu() - b['clean']) ** 2).mean()))
            rows.append({'w_obs': wo, 'w_band': wb, 'ber': float(np.mean(bers)), 'mse': float(np.mean(mses)), 'seconds': round(time.time() - t0, 1), 'n': len(ds.samples)})
            print(json.dumps(rows[-1]), flush=True)
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
