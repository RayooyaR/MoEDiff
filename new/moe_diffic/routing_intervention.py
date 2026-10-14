"""路由干预实验：在推理时强行改变 MoE 的门控权重，看恢复质量如何变化，以检验路由是否真的在起作用。

模式：normal 原路由 / uniform 四专家均匀 0.25 / mean 固定为数据集平均权重 / shuffle 在样本之间随机打乱（破坏"哪个样本用哪组权重"的对应）/
      top1 只保留占优专家 / drop_FD|MS|TF|TD 关掉某个专家（该专家权重置 0）。
采样用确定性（init_scale=0，25 步），测试集每隔 4 条取一条。
用法： python routing_intervention.py --run exp_moe_legacy [--data_root ...]
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
import torch.nn as nn  # noqa: E402
from evaluate_new import build_model  # noqa: E402
from metrics_new import batch_ber  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402

EX = ['FD', 'MS', 'TF', 'TD']


class GateOverride(nn.Module):
    def __init__(self, orig, mode, const=None):
        super().__init__()
        self.orig, self.mode, self.const = orig, mode, const
        self.gen = torch.Generator().manual_seed(0)

    def _apply(self, g):
        m = self.mode
        if m == 'normal':
            return g
        if m == 'uniform':
            return torch.full_like(g, 1.0 / g.shape[1])
        if m == 'mean':
            return self.const.to(g.device).expand_as(g).clone()
        if m == 'shuffle':
            perm = torch.randperm(g.shape[0], generator=self.gen).to(g.device)
            return g[perm]
        if m == 'top1':
            return torch.zeros_like(g).scatter_(1, g.argmax(1, keepdim=True), 1.0)
        if m.startswith('drop_'):
            g = g.clone(); g[:, EX.index(m[5:])] = 0.0
            return g
        raise ValueError(m)

    def forward(self, x, training=True):
        out = self.orig(x, training=training)
        if isinstance(out, dict):                  # 新路由（PriorGuidedRouter）返回字典
            out = dict(out)
            out['gates'] = self._apply(out['gates'])
            return out
        return self._apply(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seen', nargs='+', default=None, help='训练见过的类型（用于分开汇报见过/未见）')
    a = ap.parse_args()
    dev = torch.device('cuda')
    ds = MoEDiffDataset(a.data_root, 'test')
    ds.samples = ds.samples[::a.stride]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(ds.samples))                       # 打乱顺序，使每个 batch 混合不同类型/信干比
    items = [ds[int(j)] for j in order]
    batches = [{k: torch.stack([it[k] for it in items[i:i + a.batch_size]]) for k in items[0]} for i in range(0, len(items), a.batch_size)]
    model, scale, mt = build_model(a.run, ds.meta, dev)
    orig = model.moe.router
    p = os.path.join(HERE, 'runs', a.run, 'per_sample_test_det.npz')
    const = torch.tensor(np.load(p)['gates'].mean(0), dtype=torch.float32).view(1, -1) if os.path.exists(p) else torch.full((1, 4), 0.25)
    types = [str(t) for t in ds.type_names]
    modes = ['normal', 'uniform', 'mean', 'shuffle', 'top1'] + ['drop_' + e for e in EX]
    print('== %s ；测试样本 %d ；平均权重 %s' % (a.run, len(items), dict(zip(EX, const.view(-1).numpy().round(3)))))
    res = {}
    for md in modes:
        model.moe.router = GateOverride(orig, md, const)
        ber, mse, ty = [], [], []
        for b in batches:
            with torch.no_grad():
                est = model.sample(b['noisy'].to(dev) / scale, num_steps=25, init_scale=0.0) * scale
            ber += batch_ber(est, b, ds.meta, ds.mod_names).tolist()
            mse += ((est.cpu() - b['clean']) ** 2).mean(dim=(1, 2)).tolist()
            ty += b['interference_type'].tolist()
        res[md] = (np.array(ber), np.array(mse), np.array(ty))
        print('  %-9s BER %.4f  MSE %.4f' % (md, np.mean(ber), np.mean(mse)), flush=True)
    print('\n逐类型 BER（列=干扰类型%s）' % ('；★=未见' if a.seen else ''))
    names = [t + ('★' if a.seen and t not in a.seen else '') for t in types]
    print('%-10s' % '' + ''.join('%8s' % n for n in names))
    for md in modes:
        b, _, ty = res[md]
        print('%-10s' % md + ''.join('%8.3f' % b[ty == i].mean() for i in range(len(types))))
    if a.seen:
        un = [i for i, t in enumerate(types) if t not in a.seen]
        print('\n见过类型 / 未见类型 平均 BER')
        for md in modes:
            b, _, ty = res[md]
            m = np.isin(ty, un)
            print('  %-9s 见过 %.4f   未见 %.4f' % (md, b[~m].mean(), b[m].mean()))
    np.savez(os.path.join(HERE, 'runs', a.run, 'routing_intervention.npz'), modes=np.array(modes),
             **{'ber_' + m: res[m][0] for m in modes}, **{'type_' + m: res[m][2] for m in modes})


if __name__ == '__main__':
    main()
