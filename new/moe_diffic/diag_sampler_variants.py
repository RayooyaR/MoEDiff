"""不重新训练，只改采样方式：比较 (DDIM 步数, 初始噪声缩放) 的组合对 BER/MSE/EVM 的影响。

用法：  python diag_sampler_variants.py --run exp_moe_legacy [--data_root ...] [--stride 4]
init_scale=0 时采样路径完全确定（不再依赖随机初始噪声），推理成本与单次采样相同。
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from evaluate_new import build_model  # noqa: E402
from metrics_new import batch_ber, batch_evm  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--steps', nargs='+', type=int, default=[5, 10, 25, 50])
    ap.add_argument('--scales', nargs='+', type=float, default=[1.0, 0.7, 0.5, 0.3, 0.0])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    a = ap.parse_args()
    dev = torch.device('cuda')
    ds = MoEDiffDataset(a.data_root, a.split)
    ds.samples = ds.samples[::a.stride]
    model, scale, mt = build_model(a.run, ds.meta, dev)
    items = [ds[j] for j in range(len(ds))]
    batches = [{k: torch.stack([it[k] for it in items[i:i + a.batch_size]]) for k in items[0]} for i in range(0, len(items), a.batch_size)]
    print('== %s，%d 条样本；表中为 BER / MSE / EVM' % (a.run, len(ds)))
    print('%8s ' % 'steps\\scale' + ''.join('%26s' % ('init_scale=%g' % s) for s in a.scales))
    out = {}
    for st in a.steps:
        row = '%8d ' % st
        for sc in a.scales:
            torch.manual_seed(0)
            ber, mse, evm = [], [], []
            for b in batches:
                with torch.no_grad():
                    est = model.sample(b['noisy'].to(dev) / scale, num_steps=st, init_scale=sc) * scale
                ber += batch_ber(est, b, ds.meta, ds.mod_names).tolist()
                evm += batch_evm(est, b, ds.meta, ds.mod_names).tolist()
                mse += ((est.cpu() - b['clean']) ** 2).mean(dim=(1, 2)).tolist()
            out[(st, sc)] = (np.mean(ber), np.mean(mse), np.mean(evm))
            row += '%26s' % ('%.4f / %.3f / %.3f' % out[(st, sc)])
        print(row, flush=True)
    best = min(out, key=lambda k: out[k][0])
    print('BER 最低: steps=%d, init_scale=%g → BER %.4f MSE %.3f EVM %.3f' % (best[0], best[1], *out[best]))


if __name__ == '__main__':
    main()
