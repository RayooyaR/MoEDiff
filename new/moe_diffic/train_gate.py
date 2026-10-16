"""实验 11：训练修正门控 λ（ŝ = s_base + λ·r̂）。

只使用训练允许的数据：在**验证集**（与训练集 clean_id 不相交、仅含训练见过的类型）上拟合小门控——不在训练集上拟合是因为扩散模型已见过训练样本，
其残差估计在训练集上偏乐观，会让门控过度信任 r̂；绝不使用测试集。门控很小（约 1.5 万参数），用按 clean_id 分组的 5 折交叉验证
报告样本外的 BER/MSE（`cv_ber`），并据此在各变体间选择（见 exp/selection.py）。
用法: python train_gate.py --run <res_diff 的 run 名> --variant both [--local] [--name both] [--steps 600]
输出: runs/<run>/gate_<name>.pt 与 gate_<name>.json（cv 结果；含 λ≡1 与 λ≡0 两个参照）
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'loader'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate_new as EV  # noqa: E402
from metrics_new import batch_ber  # noqa: E402
from models.correction_gate import CorrectionGate  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402


def gen_parts(model, ds, scale, dev, steps, bs=64, seed=0):
    """对数据集每条样本采样一次残差，返回 y, s, s_base, r(已乘尺度), sir（均为 CPU 张量，归一化域）。"""
    torch.manual_seed(seed)
    Y, S, SB, R, SIR = [], [], [], [], []
    for i in range(0, len(ds.samples), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds.samples)))]
        b = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        y = b['noisy'].to(dev) / scale
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            _, sb, r, _ = model.sample(y, num_steps=steps, return_parts=True)
        Y.append(y.cpu()); S.append((b['clean'] / scale)); SB.append(sb.float().cpu()); R.append(r.float().cpu()); SIR.append(b['sir_db'])
    return [torch.cat(x) for x in (Y, S, SB, R, SIR)]


def fit_gate(variant, local, Y, S, SB, R, SIR, dev, steps=600, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    g = CorrectionGate(variant, local).to(dev)
    Y, S, SB, R, SIR = [t.to(dev).float() for t in (Y, S, SB, R, SIR)]
    opt = torch.optim.Adam([p for p in g.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    for _ in range(steps if variant != 'none' else 0):
        lam = g(Y, SB, R, sir=SIR)
        loss = ((SB + lam * R - S) ** 2).mean()
        if variant in ('sir', 'both'):
            loss = loss + 0.1 * g.sir_nll(Y, SIR)
        opt.zero_grad(); loss.backward(); opt.step()
    return g.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--variant', required=True, choices=['none', 'sir', 'conf', 'both', 'oracle'])
    ap.add_argument('--local', action='store_true')
    ap.add_argument('--name', default=None)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--ddim_steps', type=int, default=25)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    a = ap.parse_args()
    name = a.name or (a.variant + ('_local' if a.local else ''))
    dev = torch.device('cuda')
    ck = torch.load(os.path.join(HERE, 'runs', a.run, 'best_model.pth'), map_location='cpu', weights_only=False)
    targs = ck['args']
    types = [t for t in (targs.get('new_types') or []) if not (targs.get('holdout') and t in targs['holdout'])] or None
    ds = MoEDiffDataset(a.data_root, 'val', types=types, mods=targs.get('new_mods'), noise_mode=targs.get('noise_mode', 'with'))
    model, scale, mt = EV.build_model(a.run, ds.meta, dev)
    assert mt == 'res_diff', mt
    Y, S, SB, R, SIR = gen_parts(model, ds, scale, dev, a.ddim_steps)
    ids = np.array([s['clean_id'] for s in ds.samples]); fold = ids % a.folds
    oof_hat = torch.zeros_like(S)
    for k in range(a.folds):
        tr, te = torch.from_numpy(fold != k), torch.from_numpy(fold == k)
        g = fit_gate(a.variant, a.local, Y[tr], S[tr], SB[tr], R[tr], SIR[tr], dev, a.steps)
        with torch.no_grad():
            lam = g(Y[te].to(dev), SB[te].to(dev), R[te].to(dev), sir=SIR[te].to(dev)).cpu()
        oof_hat[te] = SB[te] + lam * R[te]

    def ber_of(est):
        out = []
        for i in range(0, len(ds.samples), 128):
            items = [ds[j] for j in range(i, min(i + 128, len(ds.samples)))]
            b = {k: torch.stack([it[k] for it in items]) for k in items[0]}
            out += batch_ber((est[i:i + 128] * scale), b, ds.meta, ds.mod_names).tolist()
        return float(np.mean(out))
    res = {'variant': a.variant, 'local': a.local, 'n': len(ds.samples), 'cv_ber': ber_of(oof_hat), 'cv_mse': float(((oof_hat - S) ** 2).mean()),
           'ber_lambda1': ber_of(SB + R), 'ber_lambda0': ber_of(SB), 'mse_lambda1': float(((SB + R - S) ** 2).mean()), 'mse_lambda0': float(((SB - S) ** 2).mean())}
    gfull = fit_gate(a.variant, a.local, Y, S, SB, R, SIR, dev, a.steps)
    torch.save({'variant': a.variant, 'local': a.local, 'state': gfull.state_dict()}, os.path.join(HERE, 'runs', a.run, 'gate_%s.pt' % name))
    json.dump(res, open(os.path.join(HERE, 'runs', a.run, 'gate_%s.json' % name), 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
