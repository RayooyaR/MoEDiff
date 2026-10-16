"""实验 1：专家屏蔽影响（固定 checkpoint，依次屏蔽各专家），输出 expert_impact.csv。
两种屏蔽：
  zero    把该专家的 token 置零（仍参与交叉注意力，经 Key/Value 偏置投影仍有贡献——不是真正的移除）
  remove  用 key_padding_mask 真正移除（需模型支持；film/adaln 池化时同样不计入）
同一批样本、同一随机初始噪声（逐批固定种子）-> 屏蔽前后配对比较。
区分：本测试衡量“当前模型对该专家的依赖”；“该专家单独能学到什么”需要从头训练的单专家 run（见任务清单 E1）。
用法: python expert_impact.py --run zs2_moe --modes zero remove --unseen LFM NFM RFM --stride 4 --out ../reports/expert_impact_zs2_moe.csv"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate_new as EV  # noqa: E402
from metrics_new import batch_ber  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402


def token_positions(moe, name):
    off = 1 if moe.shared_expert is not None else 0
    ei = moe.active_expert_names.index(name)
    return (moe.group_idx == ei).nonzero().flatten().add(off).tolist()


def patch_mask(moe, name, mode):
    orig = moe.forward
    pos = token_positions(moe, name)

    def fwd(modal_dict, training=True):
        out = orig(modal_dict, training)
        cond = out[0].clone(); cond[:, pos] = 0
        if mode == 'remove':
            m = torch.zeros(cond.shape[0], cond.shape[1], dtype=torch.bool, device=cond.device)
            m[:, pos] = True
            if moe.last_mask is not None:
                m = m | moe.last_mask
            moe.last_mask = m
        return (cond,) + tuple(out[1:])
    moe.forward = fwd
    return lambda: setattr(moe, 'forward', orig)


def evaluate(model, ds, scale, dev, steps, seed=0, bs=64, amp=True):
    out = {k: [] for k in ('ber', 'type', 'mod', 'sir')}
    for i in range(0, len(ds.samples), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds.samples)))]
        b = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        torch.manual_seed(seed + i)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            rec = model.sample(b['noisy'].to(dev) / scale, num_steps=steps).float() * scale
        out['ber'] += batch_ber(rec.cpu(), b, ds.meta, ds.mod_names).tolist()
        out['type'] += b['interference_type'].tolist(); out['mod'] += b['modulation'].tolist(); out['sir'] += b['sir_db'].tolist()
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--modes', nargs='+', default=['zero', 'remove'])
    ap.add_argument('--unseen', nargs='*', default=[])
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--ddim_steps', type=int, default=25)
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda')
    ds = MoEDiffDataset(a.data_root, 'test'); ds.samples = ds.samples[::a.stride]
    model, scale, mt = EV.build_model(a.run, ds.meta, dev)
    moe = model.moe
    assert moe is not None, '该模型没有 MoE'
    base = evaluate(model, ds, scale, dev, a.ddim_steps)
    tn, mn = ds.type_names, ds.mod_names
    rows = []

    def add(expert, mode, res):
        def grp(label, sel):
            if sel.any():
                rows.append(dict(run=a.run, expert=expert, mode=mode, group=label, n=int(sel.sum()), ber=float(res['ber'][sel].mean()), ber_full=float(base['ber'][sel].mean()),
                                 delta=float(res['ber'][sel].mean() - base['ber'][sel].mean())))
        grp('all', np.ones(len(res['ber']), bool))
        if a.unseen:
            u = np.isin(res['type'], [i for i, t in enumerate(tn) if t in a.unseen])
            grp('seen_types', ~u); grp('unseen_types', u)
        for i, t in enumerate(tn): grp('type:' + t, res['type'] == i)
        for i, m_ in enumerate(mn): grp('mod:' + m_, res['mod'] == i)
        for s in sorted(set(np.round(res['sir']).astype(int).tolist())): grp('sir:%+d' % s, np.round(res['sir']).astype(int) == s)
    add('none', 'full', base)
    for name in moe.active_expert_names:
        for mode in a.modes:
            undo = patch_mask(moe, name, mode)
            try:
                add(name, mode, evaluate(model, ds, scale, dev, a.ddim_steps))
            finally:
                undo(); moe.last_mask = None
    for r in rows:
        if r['group'] in ('all', 'seen_types', 'unseen_types'):
            print('%-6s %-7s %-12s BER %.4f (完整 %.4f) Δ=%+.4f' % (r['expert'], r['mode'], r['group'], r['ber'], r['ber_full'], r['delta']))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
