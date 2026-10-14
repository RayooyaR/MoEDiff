"""线性探针：MoE 的专家特征里，有没有 DiT 骨干本身没有的信息？

对每个训练好的扩散模型（run），在 test 集（全部 8 类干扰）上取特征，然后用「线性」读出头预测：
  - 干扰类型（8 类准确率；另给“训练从未见过的类型”那几类的准确率）
  - 调制类型（3 类）
  - 带内 SIR（回归，R²）
  - 干扰的谱质心、谱宽度（回归，R²；由样本里保存的干扰信号直接算出，对所有类型都适用）
特征来源：
  raw     输入信号的 log 幅度谱（池化到 64 点）+ 时域包络统计 —— 不含任何学习的“原始信息量”参照
  FD/MS/TF/TD  MoE 的各专家输出（256 维，门控前后均可取；这里取门控后的条件 token）
  experts 四个专家拼接（1024 维）
  dit_mid / dit_last  DiT 中间层 / 最后一层的 patch 特征做平均池化（256 维），在固定噪声水平 t 下取
  dit_*_msx           同上但用 均值+标准差+最大值 池化（768 维）
  注意 t 要取接近纯噪声（如 990）：t 小时 x_t 里含真实干净信号，DiT 能直接相减得到干扰，对比不公平
读出头在 5 折交叉验证上评估（按 clean_id 分组，避免同一条干净信号同时出现在训练与验证折里）。
注意：探针只能说明“特征里线性可读出多少信息”，不能说明模型是否用到了这些信息。

用法（在 new/moe_diffic 下）：
  python probe_features.py --runs zs2_moe zs2_dit zs2_ditL --out ../reports/probe_zs2.txt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
from evaluate_new import build_model  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402

SEEN = {'zs1': ['NB', 'WB', 'LFM', 'NFM', 'RFM', 'PL'], 'zs2': ['ST', 'MT', 'NB', 'WB', 'PL'], 'zs3': ['ST', 'MT', 'NB', 'WB', 'LFM', 'NFM', 'RFM']}


def interference_targets(root, split, ds):
    """按与 MoEDiffDataset 相同的文件顺序，计算干扰信号的谱质心 / 谱宽度（归一化频率，fftshift 后）。"""
    from pathlib import Path
    cen, wid = [], []
    for f in sorted((Path(root) / split).glob('*/*/*.npz')):
        z = np.load(f, allow_pickle=False)
        b = z['interference_signal']
        c = b[:, 0] + 1j * b[:, 1] if b.ndim == 3 else b
        sp = np.abs(np.fft.fftshift(np.fft.fft(c, axis=-1), axes=-1)) ** 2
        fr = np.fft.fftshift(np.fft.fftfreq(c.shape[-1]))
        p = sp / (sp.sum(-1, keepdims=True) + 1e-20)
        m = (p * fr).sum(-1)
        cen.append(m)
        wid.append(np.sqrt((p * (fr - m[:, None]) ** 2).sum(-1)))
    return np.concatenate(cen), np.concatenate(wid)


@torch.no_grad()
def extract(run, ds, dev, t_level):
    model, scale, mtype = build_model(run, ds.meta, dev)
    assert mtype == 'moe_diffmic', mtype
    ac = model.scheduler.alphas_cumprod.to(dev)
    feats = {}
    blocks = model.denoiser.blocks
    store = {}
    hooks = [blocks[len(blocks) // 2 - 1].register_forward_hook(lambda m, i, o: store.__setitem__('mid', o)),
             blocks[-1].register_forward_hook(lambda m, i, o: store.__setitem__('last', o))]
    g = torch.Generator(device=dev).manual_seed(0)
    for i in range(0, len(ds.samples), 128):
        items = [ds[j] for j in range(i, min(i + 128, len(ds.samples)))]
        clean = torch.stack([it['clean'] for it in items]).to(dev) / scale
        noisy = torch.stack([it['noisy'] for it in items]).to(dev) / scale
        # raw：log 幅度谱池化 + 包络统计
        c = torch.complex(noisy[:, 0], noisy[:, 1])
        sp = torch.log1p(torch.fft.fftshift(torch.fft.fft(c, dim=-1), dim=-1).abs() / c.shape[-1] ** 0.5)
        sp = torch.nn.functional.adaptive_avg_pool1d(sp.unsqueeze(1), 64).squeeze(1)
        env = c.abs()
        raw = torch.cat([sp, env.mean(1, keepdim=True), env.std(1, keepdim=True), env.amax(1, keepdim=True)], 1)
        feats.setdefault('raw', []).append(raw.cpu())
        cond, info = model.compute_moe_condition(noisy, training=False)
        if getattr(model, 'moe_enabled', False):
            for k, name in enumerate(model.active_expert_names):
                feats.setdefault(name, []).append(cond[:, k].cpu())
            feats.setdefault('experts', []).append(cond.flatten(1).cpu())
        t = torch.full((clean.shape[0],), t_level, dtype=torch.long, device=dev)
        noise = torch.randn(clean.shape, device=dev, generator=g)
        x_t = ac[t].sqrt().view(-1, 1, 1) * clean + (1 - ac[t]).sqrt().view(-1, 1, 1) * noise
        model.denoiser(torch.cat([x_t, noisy], 1), t, cond)
        feats.setdefault('dit_mid', []).append(store['mid'].mean(1).cpu())
        feats.setdefault('dit_last', []).append(store['last'].mean(1).cpu())
        for k in ('mid', 'last'):                      # 均值+标准差+最大值池化（维度 768），与四专家拼接（1024）更可比
            h = store[k]
            feats.setdefault('dit_%s_msx' % k, []).append(torch.cat([h.mean(1), h.std(1), h.amax(1)], 1).cpu())
    for h in hooks:
        h.remove()
    return {k: torch.cat(v).float() for k, v in feats.items()}


def folds(groups, k=5, seed=0):
    u = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(u)
    fid = {g: i % k for i, g in enumerate(u)}
    return np.array([fid[g] for g in groups])


def fit_eval(X, y, fold, task, dev, n_cls=None, wd=1e-3):
    """5 折交叉验证的线性读出头，返回所有样本的验证集预测。"""
    X = X.to(dev)
    pred = np.zeros(len(y), dtype=np.float64) if task == 'reg' else np.zeros(len(y), dtype=np.int64)
    for f in range(fold.max() + 1):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if task == 'cls':
            ytr = torch.as_tensor(y[tr], device=dev)
            W = torch.zeros(X.shape[1], n_cls, device=dev, requires_grad=True)
            b = torch.zeros(n_cls, device=dev, requires_grad=True)
            opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr) + wd * (W ** 2).sum()
                loss.backward()
                return loss
            opt.step(closure)
            pred[te] = (Xte @ W + b).argmax(1).cpu().numpy()
        else:
            ym = y[tr].mean()
            ytr = torch.as_tensor(y[tr] - ym, device=dev, dtype=torch.float32)
            A = Xtr.T @ Xtr + wd * Xtr.shape[0] * torch.eye(X.shape[1], device=dev)
            w = torch.linalg.solve(A, Xtr.T @ ytr)
            pred[te] = (Xte @ w).cpu().numpy() + ym
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--t_level', type=int, default=300, help='DiT 特征所取的噪声水平（扩散时间步）')
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    cpu = torch.device('cpu')                  # 读出头在 CPU 上拟合（小矩阵，GPU 被训练任务占满时反而慢）
    torch.set_num_threads(8)
    ds = MoEDiffDataset(a.data_root, 'test', noise_mode='with')
    n = len(ds.samples)
    ytype = np.array([s['interference_type'] for s in ds.samples])
    ymod = np.array([s['modulation'] for s in ds.samples])
    ysir = np.array([s['sir_db'] for s in ds.samples])
    gid = np.array([s['clean_id'] for s in ds.samples])
    cen, wid = interference_targets(a.data_root, 'test', ds)
    assert len(cen) == n
    fold = folds(gid)
    lines = []
    for run in a.runs:
        zname = run.split('_')[0]
        unseen = [i for i, t in enumerate(ds.type_names) if t not in SEEN.get(zname, [])]
        F = extract(run, ds, dev, a.t_level)
        lines.append('\n===== %s ；特征取自 test 集 %d 条（t=%d）；未见类型 %s' % (run, n, a.t_level, [ds.type_names[i] for i in unseen]))
        lines.append('%-10s %5s | %8s %10s | %7s | %7s | %8s %8s' % ('特征', '维度', '类型acc', '未见类型acc', '调制acc', 'SIR R²', '谱质心R²', '谱宽度R²'))
        for name, X in F.items():
            print('  fit', run, name, flush=True)
            pt = fit_eval(X, ytype, fold, 'cls', cpu, len(ds.type_names))
            pm = fit_eval(X, ymod, fold, 'cls', cpu, len(ds.mod_names))
            ps = fit_eval(X, ysir, fold, 'reg', cpu)
            pc = fit_eval(X, cen, fold, 'reg', cpu)
            pw = fit_eval(X, wid, fold, 'reg', cpu)
            um = np.isin(ytype, unseen)
            lines.append('%-10s %5d | %8.3f %10.3f | %7.3f | %7.3f | %8.3f %8.3f' % (
                name, X.shape[1], (pt == ytype).mean(), (pt[um] == ytype[um]).mean() if um.any() else float('nan'),
                (pm == ymod).mean(), r2(ysir, ps), r2(cen, pc), r2(wid, pw)))
        print('\n'.join(lines[-(len(F) + 2):]), flush=True)
    if a.out:
        open(a.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
