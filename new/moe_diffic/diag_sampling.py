"""诊断：扩散模型的误差有多少来自"每次只取一个后验样本"。

对同一批测试样本，各模型独立采样 K 次；比较
    单次样本 / K 次样本的平均(后验均值的近似) / CNN 的点估计
的 BER、EVM、MSE、SINR，并分解 MSE = (均值的误差) + (样本间方差)。
用法：  python diag_sampling.py --runs exp_moe_legacy exp_nomoe --stride 4 --K 16
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
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--ddim_steps', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    a = ap.parse_args()
    dev = torch.device('cuda')
    ds = MoEDiffDataset(a.data_root, a.split)
    ds.samples = ds.samples[::a.stride]
    Ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= a.K]
    for run in a.runs:
        model, scale, mt = build_model(run, ds.meta, dev)
        torch.manual_seed(0)
        acc = {k: dict(ber=[], evm=[], mse=[]) for k in Ks}
        var_list, sir_all, type_all = [], [], []
        for i in range(0, len(ds), a.batch_size):
            items = [ds[j] for j in range(i, min(i + a.batch_size, len(ds)))]
            batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
            noisy = batch['noisy'].to(dev)
            with torch.no_grad():
                outs = torch.stack([model.sample(noisy / scale, num_steps=a.ddim_steps) * scale for _ in range(a.K)])   # (K,B,2,L)
            clean = batch['clean'].to(dev)
            for k in Ks:
                est = outs[:k].mean(0)
                acc[k]['ber'] += batch_ber(est, batch, ds.meta, ds.mod_names).tolist()
                acc[k]['evm'] += batch_evm(est, batch, ds.meta, ds.mod_names).tolist()
                acc[k]['mse'] += ((est - clean) ** 2).mean(dim=(1, 2)).tolist()
            var_list += outs.var(0, unbiased=True).mean(dim=(1, 2)).tolist()       # 样本间方差（逐样本）
            sir_all += batch['sir_db'].tolist()
            type_all += batch['interference_type'].tolist()
        print('\n== %s (%s)，%d 条样本，%d 次采样/样本' % (run, mt, len(ds), a.K))
        print('   K    BER     EVM     MSE')
        for k in Ks:
            print('  %2d  %.4f  %.4f  %.4f' % (k, np.mean(acc[k]['ber']), np.mean(acc[k]['evm']), np.mean(acc[k]['mse'])))
        print('  样本间方差（单个样本的随机性）= %.4f ；单次 MSE − 平均后 MSE(K=%d) = %.4f' % (
            np.mean(var_list), Ks[-1], np.mean(acc[1]['mse']) - np.mean(acc[Ks[-1]]['mse'])))
        sir = np.array(sir_all)
        print('  按 SIR（单次 → K=%d 平均）BER:' % Ks[-1])
        print('   ' + '  '.join('%+d:%.3f→%.3f' % (s, np.mean(np.array(acc[1]['ber'])[sir == s]), np.mean(np.array(acc[Ks[-1]]['ber'])[sir == s]))
                               for s in sorted(set(sir.tolist()))))
        np.savez(os.path.join(HERE, 'runs', run, 'diag_sampling_%s.npz' % a.split), Ks=np.array(Ks), sir=sir,
                 **{'ber_k%d' % k: np.array(acc[k]['ber']) for k in Ks}, **{'mse_k%d' % k: np.array(acc[k]['mse']) for k in Ks},
                 var=np.array(var_list))


if __name__ == '__main__':
    main()
