"""实验 6 的迁移性补充测试：冻结专家特征 + 极小线性适配头，检验“在已见类学得的专家表征能否迁移到留出类”。

被测表征（全部冻结、不更新）：
  trained   训练好的专家 token（本 run，训练时完全没见过留出类型）
  random    同结构、随机初始化的专家（控制：表征是否只是网络结构的随机特征）
  cnn       通用 CNN 特征（--cnn_run 指定的监督 CNN 的瓶颈层均值+最大池化；控制：通用卷积特征）
适配任务：由特征线性预测干扰的 12 维物理属性（loader/attributes.py，标签来自干扰信号本身，对任何类型都有定义）。
协议（两张表分开报告，不混用）：
  zero-shot   线性头只在**已见类型**样本上拟合，直接测留出类型（不使用任何留出类标签）
  few-shot    线性头在已见类型 + 留出类型的 K 条/格 训练样本（取自 train 划分里的留出类型）上拟合，测留出类型的 test 样本
指标：标准化 RMSE（单位=已见类型上该属性的标准差，越小越好；再按 FD/TF/TD/MS 分组取平均）。另给“常数基线”=始终预测已见类型均值的 RMSE 作参照。
用法: python loto_adapt.py --run <以 --holdout 训练的 run> --holdout LFM [--cnn_run <cnn run>] [--k 1 5 10] --out runs/<run>/loto_adapt.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate_new as EV  # noqa: E402
from attributes import ATTR_GROUPS, ATTR_NAMES  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402


@torch.no_grad()
def feats_trained(model, ds, scale, dev, kind, cnn=None):
    out = []
    for i in range(0, len(ds.samples), 128):
        items = [ds[j] for j in range(i, min(i + 128, len(ds.samples)))]
        y = torch.stack([it['noisy'] for it in items]).to(dev) / scale
        if kind == 'cnn':
            c = cnn
            b = c.bottleneck(c.enc3(c.enc2(c.enc1(c.input_proj(y)))))
            f = torch.cat([b.mean(-1), b.amax(-1)], 1)
        else:
            cond, _ = model.compute_moe_condition(y, training=False)
            f = cond.flatten(1)
        out.append(f.float().cpu())
    return torch.cat(out)


def ridge_fit(X, Y, lam=1e-2):
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    ym = Y.mean(0, keepdim=True)
    A = Xn.T @ Xn + lam * Xn.shape[0] * torch.eye(X.shape[1])
    W = torch.linalg.solve(A, Xn.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym


def rmse(y, p):
    """标准化单位（已见类型的属性标准差）下的 RMSE，越小越好。不用 R²：留出类型内部有些属性几乎是常数（如调频类的占空比≡1），R² 分母为 0 会爆炸。"""
    return ((y - p) ** 2).mean(0).sqrt()


def per_group(r):
    return {g: float(np.mean([r[i] for i in idx])) for g, idx in ATTR_GROUPS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--holdout', nargs='+', required=True)
    ap.add_argument('--cnn_run', default=None)
    ap.add_argument('--k', type=int, nargs='+', default=[1, 5, 10])
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda')
    meta = json.load(open(os.path.join(a.data_root, 'metadata.json')))
    seen = [t for t in meta['TYPES'] if t not in a.holdout]
    tr_seen = MoEDiffDataset(a.data_root, 'train', types=seen, return_aux=True, max_per_cell=10)
    stats = (tr_seen.attr_mean, tr_seen.attr_std)
    te_hold = MoEDiffDataset(a.data_root, 'test', types=a.holdout, return_aux=True, attr_stats=stats)
    ytr_seen = torch.stack([torch.from_numpy((s['attrs_raw'] - stats[0]) / stats[1]) for s in tr_seen.samples])
    yte = torch.stack([torch.from_numpy((s['attrs_raw'] - stats[0]) / stats[1]) for s in te_hold.samples])
    model, scale, mt = EV.build_model(a.run, meta, dev)
    reps = {'trained': (model, None)}
    from models import PhysicsMoEModel
    import train as T
    ck = torch.load(os.path.join(HERE, 'runs', a.run, 'best_model.pth'), map_location='cpu', weights_only=False)
    cfg = T.build_config(__import__('argparse').Namespace(**ck['args'])); cfg.data.signal_length = int(meta['SIG_LEN'])
    torch.manual_seed(0)
    reps['random'] = (PhysicsMoEModel(cfg).to(dev).eval(), None)
    if a.cnn_run:
        cnn, cscale, _ = EV.build_model(a.cnn_run, meta, dev)
        reps['cnn'] = (cnn, cnn)
    res = {'run': a.run, 'holdout': a.holdout, 'attr_names': ATTR_NAMES, 'metric': 'standardized RMSE (lower is better)', 'zero_shot': {}, 'few_shot': {},
           'const_baseline': per_group(rmse(yte, ytr_seen.mean(0, keepdim=True).expand_as(yte)).numpy())}
    F = {}
    for name, (m, cnn) in reps.items():
        sc = scale if name != 'cnn' else cscale
        F[name] = (feats_trained(m, tr_seen, sc, dev, 'cnn' if name == 'cnn' else 'moe', cnn), feats_trained(m, te_hold, sc, dev, 'cnn' if name == 'cnn' else 'moe', cnn))
        f = ridge_fit(F[name][0], ytr_seen)
        res['zero_shot'][name] = per_group(rmse(yte, f(F[name][1])).numpy())
    for k in a.k:
        ds_h = MoEDiffDataset(a.data_root, 'train', types=a.holdout, return_aux=True, attr_stats=stats, max_per_cell=k)
        yh = torch.stack([torch.from_numpy((s['attrs_raw'] - stats[0]) / stats[1]) for s in ds_h.samples])
        res['few_shot'][str(k)] = {}
        for name, (m, cnn) in reps.items():
            sc = scale if name != 'cnn' else cscale
            fh = feats_trained(m, ds_h, sc, dev, 'cnn' if name == 'cnn' else 'moe', cnn)
            f = ridge_fit(torch.cat([F[name][0], fh]), torch.cat([ytr_seen, yh]))
            res['few_shot'][str(k)][name] = per_group(rmse(yte, f(F[name][1])).numpy())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
