"""干净通信信号的无条件扩散预训练（自监督思想：不需要受干扰观测、不需要配对标签）。

做法：用与 J0 完全相同的 PhysicsMoEModel（U-Net 去噪网络、余弦 1000 步噪声调度、ε 预测），把受干扰观测通道 y 置零、关闭 MoE，
只学干净信号的分布 p(s)：  s_t = √ᾱ s + √(1-ᾱ) ε，loss = MSE(ε̂, ε)。
迁移：保存的 best_model.pth 与 exp_train 同格式（model_state_dict / ema_shadow / data_scale / args）；微调时 `exp_train.py --init_from <本 run>` 逐张量加载 denoiser.*。
数据源（--pretrain_pool）：
  gen/make_clean_pool.py 生成的 npz   —— 额外独立生成的干净波形（P1）
  train420                            —— 只用本次 10% 训练对里的干净信号（对照 P0：同样的预训练计算量、无额外干净数据）
验证：用验证集(val split)的干净信号计算固定噪声网格上的 ε-MSE，按验证 ε-MSE 保存最佳 EMA 权重（不看测试集）。
用法示例：
  python pretrain_clean.py --exp_name pre_clean10k --pretrain_pool ../data/clean_pool/pool_10k.npz --pretrain_steps 40000 \\
     --model_type moe_diffmic --disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln --amp bf16 --lr 3e-4 --batch_size 32 --seed 42
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
import train as T  # noqa: E402
from exp_train import get_args  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402


def main(argv=None):
    args = get_args(argv)
    assert args.pretrain_pool, '需要 --pretrain_pool'
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(HERE, 'runs', args.exp_name); os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, 'args.json'), 'w'), indent=1)
    log = open(os.path.join(out_dir, 'train.log'), 'w', encoding='utf-8')
    def P(*a):
        s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    root = os.path.abspath(os.path.join(HERE, args.new_data_root)); meta = json.load(open(os.path.join(root, 'metadata.json')))
    train_types = [t for t in (args.new_types or meta['TYPES'])]
    # ---- 干净信号源
    if args.pretrain_pool == 'train420':
        kw = dict(types=train_types, mods=args.new_mods, noise_mode=args.noise_mode, max_per_cell=args.train_per_cell, seed=args.seed)
        ds = MoEDiffDataset(root, 'train', **kw)
        X = np.stack([s['clean'] for s in ds.samples]).astype(np.float32)
        P('[数据] train420：本次 10%% 训练对的干净信号 %d 条（每条重复使用）' % len(X))
    else:
        z = np.load(args.pretrain_pool); X = z['x'].astype(np.float32)
        P('[数据] 干净波形池 %s：%d 条，clean_id %d..%d' % (args.pretrain_pool, len(X), z['clean_id'].min(), z['clean_id'].max()))
    data_scale = float(np.std(X)); P('data_scale = %.6f' % data_scale)
    X = torch.from_numpy(X / data_scale).to(dev)
    vds = MoEDiffDataset(root, 'val', types=train_types, noise_mode=args.noise_mode)
    XV = torch.from_numpy(np.stack([s['clean'] for s in vds.samples]).astype(np.float32) / data_scale).to(dev)
    # ---- 模型：与 J0 同一个类，同一份 config
    config = T.build_config(args)
    config.data.signal_length = int(meta['SIG_LEN']); config.data.samples_per_symbol = config.data.sps = int(meta['SPS']); config.data.rrc_span = int(meta['RRC_SPAN'])
    from models import PhysicsMoEModel
    model = PhysicsMoEModel(config).to(dev)
    P('Params: %s' % f'{sum(p.numel() for p in model.parameters()):,}', ' 无条件预训练：y 通道置零')
    ema = T.EMA(model, decay=args.ema_rate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total, warm = args.pretrain_steps, 500
    sch = optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(min((s - warm) / max(total - warm, 1), 1.0) * math.pi)))
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    gv = torch.Generator(device=dev)

    @torch.no_grad()
    def val_eps():                                                    # 固定噪声/时间步的验证 ε-MSE（EMA 权重）
        model.eval(); ema.apply_shadow(); gv.manual_seed(1234); tot, n = 0.0, 0
        for i in range(0, len(XV), 128):
            x = XV[i:i + 128]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                r = model(x, torch.zeros_like(x))
            tot += float(F.mse_loss(r['noise_pred'].float(), r['noise_target'].float(), reduction='sum')); n += r['noise_pred'].numel()
        ema.restore(); model.train(); return tot / n
    t0, best, run_loss, run_n = time.time(), float('inf'), 0.0, 0
    model.train()
    for step in range(total):
        idx = torch.randint(0, len(X), (args.batch_size,), device=dev, generator=g)
        x = X[idx]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
            r = model(x, torch.zeros_like(x))
        loss = F.mse_loss(r['noise_pred'].float(), r['noise_target'].float())
        if not torch.isfinite(loss): P('loss 非有限，中止'); sys.exit(3)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, args.gradient_clip); opt.step(); sch.step(); ema.update()
        run_loss += float(loss); run_n += 1
        if (step + 1) % 1000 == 0 or step + 1 == total:
            v = val_eps()
            P('step %d/%d train_eps_mse=%.5f val_eps_mse=%.5f (%.0fs)' % (step + 1, total, run_loss / run_n, v, time.time() - t0)); run_loss, run_n = 0.0, 0
            if v < best:
                best = v
                torch.save({'epoch': step + 1, 'model_state_dict': model.state_dict(), 'ema_shadow': ema.shadow, 'data_scale': data_scale,
                            'best_metrics': {'val_eps_mse': v, 'steps': step + 1}, 'args': vars(args)}, os.path.join(out_dir, 'best_model.pth'))
    json.dump({'params': sum(p.numel() for p in model.parameters()), 'steps': total, 'n_clean': int(len(X)), 'seconds': round(time.time() - t0, 1), 'best_val_eps_mse': best, 'data_scale': data_scale},
              open(os.path.join(out_dir, 'run_info.json'), 'w'), indent=1)
    P('done best_val_eps_mse=%.5f' % best)


if __name__ == '__main__':
    main()
