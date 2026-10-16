"""采样耗时（评估主要成本）：python sample.py <run> <fp32|bf16> ；批 64、25 步 DDIM、K=1，打印每批耗时与折算的 avg8 评估用时（3360 样本）。"""
import sys, time
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, ROOT + '/moe_diffic'); sys.path.insert(0, ROOT + '/loader')
import torch
from evaluate_new import build_model
from moe_dataset import MoEDiffDataset
run, mode = sys.argv[1], sys.argv[2]; bs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
dev = torch.device('cuda')
ds = MoEDiffDataset(ROOT + '/data/v2', 'test', types=['ST'], max_per_cell=2)
m, scale, mt = build_model(run, ds.meta, dev)
noisy = torch.stack([ds[i % len(ds.samples)]['noisy'] for i in range(bs)]).to(dev) / scale
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(mode == 'bf16')):
    m.sample(noisy, num_steps=25)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3): m.sample(noisy, num_steps=25)
    torch.cuda.synchronize(); per = (time.time() - t) / 3
print('%s %s 批%d: %.2f s/批 -> avg8 评估 3360 样本约 %.1f 分钟' % (run, mode, bs, per, per / bs * 3360 * 8 / 60), flush=True)
