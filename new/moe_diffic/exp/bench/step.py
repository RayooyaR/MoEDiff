"""训练步耗时：python step.py <run> <fp32|tf32|bf16> [steps]；run 取自 moe_diffic/runs（用其 args.json 重建模型）。"""
import sys, time, os
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, ROOT + '/moe_diffic'); sys.path.insert(0, ROOT + '/loader')
import torch
from evaluate_new import build_model
from moe_dataset import MoEDiffDataset
run, mode = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
dev = torch.device('cuda')
ds = MoEDiffDataset(ROOT + '/data/v2', 'train', types=['ST'], max_per_cell=2)
m, scale, mt = build_model(run, ds.meta, dev); m.train()
batch = torch.stack([ds[i % len(ds.samples)]['clean'] for i in range(32)]).to(dev); noisy = torch.stack([ds[i % len(ds.samples)]['noisy'] for i in range(32)]).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'; torch.backends.cudnn.allow_tf32 = mode == 'tf32'
def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(mode == 'bf16')):
        out = m(batch / scale, noisy / scale)
        loss = ((out['noise_pred'] - out['noise_target']) ** 2).mean() if mt not in ('cnn', 'dit_reg') else out['loss']
    loss.backward(); opt.step()
for _ in range(8): step()
torch.cuda.synchronize()
import os
if os.environ.get('START_AT'):
    while time.time() < float(os.environ['START_AT']): time.sleep(0.01)
t = time.time()
for _ in range(n): step()
torch.cuda.synchronize(); print('%s %s %.1f ms/step' % (run, mode, (time.time() - t) / n * 1000), flush=True)
