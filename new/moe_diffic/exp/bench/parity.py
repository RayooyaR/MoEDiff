"""bf16 推理一致性（不训练）：同一批样本、同一初始噪声，比较 fp32 与 bf16 采样结果。
对照量：两个不同随机种子的 fp32 采样之间的差（采样本身的随机性）。bf16 引入的差应远小于它。"""
import sys, time
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, ROOT + '/moe_diffic'); sys.path.insert(0, ROOT + '/loader')
import numpy as np, torch
from evaluate_new import build_model
from moe_dataset import MoEDiffDataset
from metrics_new import batch_ber
dev = torch.device('cuda')
ds = MoEDiffDataset(ROOT + '/data/v2', 'test', noise_mode='with'); ds.samples = ds.samples[::13][:512]
meta, mods = ds.meta, ds.mod_names
for run in sys.argv[1:]:
    m, scale, mt = build_model(run, meta, dev)
    outs = {k: [] for k in ('fp32', 'bf16', 'fp32b')}; bers = {k: [] for k in outs}
    t = time.time()
    for i in range(0, len(ds), 64):
        items = [ds[j] for j in range(i, min(i + 64, len(ds)))]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        noisy = batch['noisy'].to(dev) / scale
        for tag, mode, seed in (('fp32', 'fp32', 0), ('bf16', 'bf16', 0), ('fp32b', 'fp32', 1)):
            torch.manual_seed(seed)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(mode == 'bf16')):
                rec = m.sample(noisy, num_steps=25).float() * scale
            outs[tag].append(rec.cpu()); bers[tag] += batch_ber(rec.cpu(), batch, meta, mods).tolist()
    O = {k: torch.cat(v) for k, v in outs.items()}
    rel = lambda a, b: float(((O[a] - O[b]) ** 2).sum() / (O[b] ** 2).sum())
    print('%s (512 条, %.0f 秒): BER fp32 %.4f | bf16 %.4f | fp32 另一随机种子 %.4f' % (run, time.time() - t, np.mean(bers['fp32']), np.mean(bers['bf16']), np.mean(bers['fp32b'])))
    print('   相对差(‖a-b‖²/‖b‖²): bf16 vs fp32(同噪声) = %.5f ; fp32 不同种子 vs fp32 = %.5f ; 比值 %.3f' % (rel('bf16', 'fp32'), rel('fp32b', 'fp32'), rel('bf16', 'fp32') / rel('fp32b', 'fp32')), flush=True)
