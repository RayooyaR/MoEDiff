"""拟合并评估 LMMSE 基线：  python lmmse_fit.py --exp_name exp_lmmse

步骤：训练集上精确估计窗口协方差 R_yy、R_xy → 对一组岭参数求维纳滤波器 W = R_xy (R_yy + ωI)^-1
→ 在验证集上按 BER 选 ω → 保存 runs/<exp_name>/lmmse.pt。测试集评估由 evaluate_new.py 统一完成。
岭参数候选包括原始代码的设置 ω = κ²（κ² = 训练集上干扰功率/信号功率），以及一组相对 tr(R_yy)/D 的比例。
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
from metrics_new import batch_ber  # noqa: E402
from models.lmmse_new import LMMSEFilter, extract_windows  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_name', default='exp_lmmse')
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--window', type=int, default=256)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--types', nargs='+', default=None, help='只用这些干扰类型拟合/验证（留出整类实验用）')
    ap.add_argument('--train_per_cell', type=int, default=None, help='小样本：训练集每格取 K 条')
    ap.add_argument('--val_per_cell', type=int, default=None, help='验证集每格取多少条')
    ap.add_argument('--aug_views', type=int, default=0, help='>0 时对训练样本做增强，每个样本生成这么多个视图参与协方差估计')
    a = ap.parse_args()
    dev = torch.device(a.device)

    tr = MoEDiffDataset(a.data_root, 'train', types=a.types, max_per_cell=a.train_per_cell, augment=a.aug_views > 0, virtual_multiplier=max(1, a.aug_views))
    va = MoEDiffDataset(a.data_root, 'val', types=a.types, max_per_cell=a.val_per_cell)
    its = [tr[i] for i in range(len(tr))]                # 含增强视图（若开启）
    X = torch.stack([it['clean'] for it in its]).to(dev).double()
    Y = torch.stack([it['noisy'] for it in its]).to(dev).double()
    xw, yw = extract_windows(X, a.window, a.overlap), extract_windows(Y, a.window, a.overlap)
    n, D = xw.shape
    R_yy, R_xy = yw.T @ yw / n, xw.T @ yw / n
    kappa2 = float(((Y - X) ** 2).mean() / (X ** 2).mean())
    tr_over_d = float(torch.trace(R_yy)) / D
    print('训练窗口数 %d，维度 %d，κ²=%.2f，tr(R_yy)/D=%.2f' % (n, D, kappa2, tr_over_d))

    cands = ([('kappa2(原始设置)', kappa2)] + [('%g·κ²' % m, m * kappa2) for m in (3, 10, 30, 100, 1000)]
             + [('%g·tr/D' % r, r * tr_over_d) for r in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)])
    items = [va[i] for i in range(len(va))]
    batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    noisy = batch['noisy'].to(dev)
    results, best = [], None
    for name, omega in cands:
        W = torch.linalg.solve(R_yy + omega * torch.eye(D, device=dev, dtype=R_yy.dtype), R_xy.T).T
        f = LMMSEFilter(W.float(), a.window, a.overlap, X.shape[-1]).to(dev)
        est = f.sample(noisy)
        ber = float(batch_ber(est, batch, va.meta, va.mod_names).mean())
        mse = float(((est.cpu() - batch['clean']) ** 2).mean())
        results.append((name, omega, ber, mse))
        print('  ω=%-18s (%.4g)  val BER=%.4f  val MSE=%.4f' % (name, omega, ber, mse))
        if best is None or ber < best[2]:
            best = (name, omega, ber, W.float().cpu())
    print('MF（不处理）val BER = %.4f' % float(batch_ber(noisy, batch, va.meta, va.mod_names).mean()))
    print('选定: ω=%s (%.4g)，val BER=%.4f' % (best[0], best[1], best[2]))
    out = os.path.join(HERE, 'runs', a.exp_name)
    os.makedirs(out, exist_ok=True)
    torch.save(dict(W=best[3], window=a.window, overlap=a.overlap, length=int(X.shape[-1]), omega_name=best[0], omega=best[1],
                    kappa2=kappa2, candidates=results), os.path.join(out, 'lmmse.pt'))
    print('saved', os.path.join(out, 'lmmse.pt'))


if __name__ == '__main__':
    main()
