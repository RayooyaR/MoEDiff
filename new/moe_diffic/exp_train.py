"""plan.md 实验统一训练入口（新）。不改动 train.py 的训练流程，复用其参数解析、config 构建与 EMA。

支持的 --model_type：
  moe_diffmic  条件扩散（DiT / UNet1D / DiffWave 主干；专家/路由/融合/条件注入/属性监督等扩展见 exp_args.py）
  res_diff     初始恢复 + 残差扩散（实验 10）
  cnn / mv_cnn / dit_reg  直接回归类（与 train.py 相同的模型与损失）
输出（runs/<exp_name>/）：args.json、train.log、best_model.pth（--keep_ckpt 或由任务清单决定是否保留）、val_log.csv、
  val_summary.json（最佳验证 BER 等，供自动选择脚本读取）、run_info.json（参数量/耗时）、test_results.json（快速测试，按干扰类型）。
退出码：0 正常；3 发散（loss 非有限）；4 早期中止（训练过半验证 BER 仍为机会水平）。
"""
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'loader'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train as T  # noqa: E402
from exp_args import add_exp_args  # noqa: E402
from exp_config import build_oracle  # noqa: E402
from metrics_new import batch_ber  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402
from training.losses import CombinedLoss  # noqa: E402

REGRESSION = ('cnn', 'mv_cnn', 'dit_reg', 'res_cnn', 'restormer')


def get_args(argv=None):
    p = T.build_parser()
    add_exp_args(p)
    return p.parse_args(argv)


def build_model_for_training(args, config):
    mt = args.model_type
    if mt == 'moe_diffmic':
        from models import PhysicsMoEModel
        return PhysicsMoEModel(config)
    if mt == 'res_diff':
        from models.residual_diffusion import ResidualDiffusionModel
        return ResidualDiffusionModel(config, base_ch=args.cnn_base_ch)
    if mt == 'res_cnn':
        from models.residual_diffusion import ResidualCNNBaseline
        return ResidualCNNBaseline(config, base_ch=args.cnn_base_ch)
    if mt == 'cnn':
        from models.baseline.cnn_baseline import PhysicsCNNBaseline
        return PhysicsCNNBaseline(in_channels=2, base_ch=args.cnn_base_ch, signal_length=config.data.signal_length, freq_loss_weight=0.1)
    if mt == 'restormer':
        from models.baseline.restormer_baseline import Restormer1DBaseline
        return Restormer1DBaseline(in_channels=2, dim=args.restormer_dim, signal_length=config.data.signal_length)
    if mt == 'diffcom_1d':
        from models.baseline_new import DiffCom1DBaseline
        return DiffCom1DBaseline(config, dps_step_size=args.dps_step_size, warm_start=args.dps_warm_start)
    if mt == 'icdm_1d':
        from models.baseline_new import ICDMDual1DBaseline
        return ICDMDual1DBaseline(config, dual_gamma=args.icdm_dual_gamma)
    if mt == 'mv_cnn':
        from models.baseline_new import MultiViewCNNBaseline
        return MultiViewCNNBaseline(config, base_ch=args.cnn_base_ch, balance_coef=args.balance_loss_coef)
    if mt == 'dit_reg':
        from models.baseline_new import DiTRegressionBaseline
        return DiTRegressionBaseline(config, balance_coef=args.balance_loss_coef)
    raise ValueError('exp_train 不支持 model_type=%s（请用 train.py）' % mt)


def load_frozen_base(model, run, device):
    """把 runs/<run>/best_model.pth 里的 CNN（EMA 权重）装进 ResidualDiffusionModel.base。"""
    ck = torch.load(os.path.join(HERE, 'runs', run, 'best_model.pth'), map_location=device, weights_only=False)
    sd = dict(ck['model_state_dict'])
    for n, v in ck.get('ema_shadow', {}).items():
        if n in sd:
            sd[n] = v
    model.base.load_state_dict(sd)
    return float(ck['data_scale'])


def main(argv=None):
    args = get_args(argv)
    t_start = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(HERE, 'runs', args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, 'args.json'), 'w'), indent=1)
    import logging
    logger = logging.getLogger('exp_train'); logger.handlers.clear(); logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    for h in (logging.FileHandler(os.path.join(out_dir, 'train.log'), encoding='utf-8'), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); logger.addHandler(h)
    logger.info('Experiment: %s  model_type=%s  args=%s' % (args.exp_name, args.model_type, json.dumps({k: v for k, v in vars(args).items() if v not in (None, False)}, ensure_ascii=False)[:600]))

    # ------------------------------------------------------------ 数据
    root = os.path.abspath(os.path.join(HERE, args.new_data_root)) if not os.path.isabs(args.new_data_root) else args.new_data_root
    meta = json.load(open(os.path.join(root, 'metadata.json')))
    all_types = list(meta['TYPES'])
    train_types = [t for t in (args.new_types or all_types) if not (args.holdout and t in args.holdout)]
    need_aux = args.attr_mode != 'none' or args.router_type == 'dual' or args.target == 'interf' or args.oracle_cond != 'none' or args.extra_local != 'none'
    kw = dict(types=train_types, mods=args.new_mods, noise_mode=args.noise_mode, max_per_cell=args.max_per_cell, seed=args.seed)
    kw_tr, kw_va = dict(kw), dict(kw)
    if args.train_per_cell is not None:
        kw_tr['max_per_cell'] = args.train_per_cell
    if args.val_per_cell is not None:
        kw_va['max_per_cell'] = args.val_per_cell
    train_ds = MoEDiffDataset(root, 'train', augment=args.augment, virtual_multiplier=args.aug_multiplier, return_aux=need_aux,
                              aug_ops=args.phys_aug if args.phys_aug != 'none' else None, **kw_tr)
    held_samples = []
    if args.oof_fold >= 0:
        held_samples = [s for s in train_ds.samples if s['clean_id'] % args.oof_folds == args.oof_fold]
        train_ds.samples = [s for s in train_ds.samples if s['clean_id'] % args.oof_folds != args.oof_fold]
        logger.info('[oof] fold %d/%d：训练 %d 条，样本外 %d 条' % (args.oof_fold, args.oof_folds, len(train_ds.samples), len(held_samples)))
    stats = (train_ds.attr_mean, train_ds.attr_std) if need_aux else None
    val_ds = MoEDiffDataset(root, 'val', return_aux=need_aux, attr_stats=stats, **kw_va)
    if need_aux:
        np.savez(os.path.join(out_dir, 'attr_stats.npz'), mean=train_ds.attr_mean, std=train_ds.attr_std)     # 评估时需用同一套标准化统计量
    if args.val_stride > 1:
        val_ds.samples = val_ds.samples[::args.val_stride]
    mod_names, type_names = list(train_ds.mod_names), list(train_ds.type_names)
    logger.info('[data] root=%s train_types=%s holdout=%s train=%d val=%d' % (root, train_types, args.holdout, len(train_ds), len(val_ds)))
    data_scale = float(np.std(np.concatenate([s['clean'] for s in train_ds.samples], axis=0)))
    data_scale = data_scale if data_scale > 1e-6 else 1.0
    logger.info('data_scale = %.6f' % data_scale)

    config = T.build_config(args)
    config.data.signal_length = int(meta['SIG_LEN']); config.data.samples_per_symbol = config.data.sps = int(meta['SPS']); config.data.rrc_span = int(meta['RRC_SPAN'])
    model = build_model_for_training(args, config).to(device)
    base_model = model
    if args.init_from:                                # 用预训练（干净信号无条件扩散）的 EMA 权重初始化去噪网络；逐张量核对，不用 strict=False 忽略不匹配
        ck0 = torch.load(os.path.join(HERE, 'runs', args.init_from, 'best_model.pth'), map_location=device, weights_only=False)
        sd0 = dict(ck0['model_state_dict'])
        for n_, v_ in ck0['ema_shadow'].items():
            if n_ in sd0:
                sd0[n_] = v_
        own = model.state_dict()
        den_own = [k for k in own if k.startswith('denoiser.')]
        miss = [k for k in den_own if k not in sd0]
        bad = [k for k in den_own if k in sd0 and own[k].shape != sd0[k].shape]
        assert not miss and not bad, '预训练权重与当前去噪网络不兼容：缺失 %s 形状不符 %s' % (miss[:5], bad[:5])
        with torch.no_grad():
            for k in den_own:
                own[k].copy_(sd0[k])
            if args.backbone == 'unet1d':               # y 通道的输入卷积权重在预训练中没有梯度：置零，使微调起点 = 预训练的无条件模型
                own['denoiser.inp.weight'][:, 2:4].zero_()
        new_keys = [k for k in own if not k.startswith('denoiser.')]
        sc0 = float(ck0['data_scale'])
        logger.info('[init_from] %s：载入 denoiser.* 共 %d 个张量（含 EMA 权重）；重新初始化（不属于去噪网络）%d 个张量，前缀 %s；y 通道输入卷积置零；预训练 data_scale=%.5f 当前 %.5f'
                    % (args.init_from, len(den_own), len(new_keys), sorted({k.split('.')[0] for k in new_keys}), sc0, data_scale))
        if abs(sc0 - data_scale) / data_scale > 0.01:
            logger.warning('预训练与当前 data_scale 相差 >1%%（%.5f vs %.5f）' % (sc0, data_scale))
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Params: %s (trainable %s)' % (f'{n_params:,}', f'{n_train:,}'))

    if args.model_type == 'icdm_1d':               # 干扰先验按自身尺度归一化（与 train.py 一致）
        _b = np.concatenate([(s_['noisy'] - s_['clean']).ravel() for s_ in train_ds.samples])
        model.set_interference_scale(float(np.std(_b)) / data_scale)
        logger.info('[ICDM-dual] interference scale (relative to data_scale) = %.3f' % (float(np.std(_b)) / data_scale))
    if args.model_type == 'diffcom_1d' and args.dps_warm_start:
        from models.baseline_new import estimate_obs_snr_db
        model.obs_snr_db = estimate_obs_snr_db(train_ds.samples)
        json.dump({'obs_snr_db': model.obs_snr_db}, open(os.path.join(out_dir, 'obs_snr.json'), 'w'))
        logger.info('[DiffCom] 热启动：训练集上 y 相对干净信号的平均 SINR = %.2f dB，t0 = %d' % (model.obs_snr_db, model.warm_t_start(model.scheduler.alphas_cumprod)))
    is_reg = args.model_type in REGRESSION
    oof_map = None
    if args.model_type in ('res_diff', 'res_cnn'):
        if not args.base_ckpt:
            raise ValueError('res_diff 需要 --base_ckpt（冻结的基础恢复器）')
        base_scale = load_frozen_base(model, args.base_ckpt, device)
        if abs(base_scale - data_scale) / data_scale > 1e-3:
            logger.warning('基础恢复器的 data_scale (%.5f) 与当前 (%.5f) 不同；残差尺度已按当前 data_scale 归一化重算' % (base_scale, data_scale))
        if args.base_oof:
            z = np.load(args.base_oof)
            oof_map = {int(c): i for i, c in enumerate(z['clean_id'])}
            oof_pred = torch.from_numpy(z['pred']).float()
        # 残差尺度：训练集上 (clean - s_base) / data_scale 的标准差
        res = []
        with torch.no_grad():
            for i in range(0, len(train_ds.samples), 256):
                items = train_ds.samples[i:i + 256]
                cl = torch.from_numpy(np.stack([s['clean'] for s in items])).to(device) / data_scale
                no = torch.from_numpy(np.stack([s['noisy'] for s in items])).to(device) / data_scale
                if oof_map is not None:
                    idx = [oof_map[s['clean_id']] for s in items]
                    sb = oof_pred[idx].to(device) / data_scale
                else:
                    sb = model.base_pred(no)
                res.append((cl - sb).flatten().cpu())
        model.res_scale.fill_(float(torch.cat(res).std()))
        resv = []                                                           # 验证集上的残差尺度（基础恢复器对验证样本天然是样本外的）：与训练集残差尺度对比，检查隐性泄漏/偏乐观
        with torch.no_grad():
            for i in range(0, len(val_ds.samples), 256):
                items = val_ds.samples[i:i + 256]
                cl = torch.from_numpy(np.stack([s['clean'] for s in items])).to(device) / data_scale
                no = torch.from_numpy(np.stack([s['noisy'] for s in items])).to(device) / data_scale
                resv.append((cl - model.base_pred(no)).flatten().cpu())
        res_val_std = float(torch.cat(resv).std())
        logger.info('残差尺度：训练集 %.4f | 验证集 %.4f（比值 %.3f；明显小于 1 说明训练残差偏乐观）' % (float(model.res_scale), res_val_std, float(model.res_scale) / max(res_val_std, 1e-8)))
        logger.info('res_scale = %.4f (%s)' % (float(model.res_scale), 'OOF 样本外基础估计' if oof_map is not None else '在线基础估计（训练集上偏乐观）'))

    route_prior = None
    if args.router_type == 'prior' and not args.disable_moe:
        from models.prior_router import build_prior_table
        route_prior = build_prior_table(type_names, list(config.moe.active_experts)).to(device)

    criterion = CombinedLoss(config)
    stft_loss = None
    if args.loss_variant == 'eps_only':
        criterion.reconstruction_weight = 0.0; criterion.spectral_weight = 0.0
    elif args.loss_variant in ('mrstft_mag', 'mrstft_cplx'):
        from training.losses import MultiResSTFTLoss
        criterion.spectral_weight = 0.0                       # 多分辨率项取代原来的单分辨率幅度谱项；x̂0 的 SmoothL1 重建项保留
        stft_loss = MultiResSTFTLoss(complex_term=(args.loss_variant == 'mrstft_cplx'))
        logger.info('[D2] loss_variant=%s stft_weight=%g（取代原谱项，保留 x0 SmoothL1 重建项）' % (args.loss_variant, args.stft_weight))
    bs = min(args.batch_size, len(train_ds))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    steps_per_epoch = len(loader)

    # ---------------------------------------------------- 属性预训练阶段（实验 3）
    def aux_set(batch):
        if hasattr(model, 'set_aux'):
            model.set_aux(sir=batch['sir_db'].to(device) if 'sir_db' in batch else None, attrs=batch['attrs'].to(device) if 'attrs' in batch else None,
                          base_pred=None, oracle=(build_oracle(batch, args.oracle_cond, device) if args.oracle_cond != 'none' else None),
                          interf=((batch['interf'].to(device) / data_scale) if (args.extra_local != 'none' and 'interf' in batch) else None))

    if args.attr_mode in ('pretrain_freeze', 'pretrain_ft'):
        params = [p for n, p in model.named_parameters() if n.startswith(('moe.experts', 'moe.attr_heads', 'modal_projector'))]
        opt0 = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        model.train()
        for ep in range(args.attr_pretrain_epochs):
            tot, n = 0.0, 0
            for batch in loader:
                noisy = batch['noisy'].to(device) / data_scale
                aux_set(batch)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                    _, info = model.compute_moe_condition(noisy, training=True)
                loss = info['aux_losses']['attr_loss'].float()
                opt0.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, args.gradient_clip); opt0.step()
                tot += loss.item(); n += 1
            logger.info('[attr-pretrain] epoch %d/%d attr_loss=%.4f' % (ep + 1, args.attr_pretrain_epochs, tot / max(n, 1)))
        if args.attr_mode == 'pretrain_freeze':
            for n_, p in model.named_parameters():
                if n_.startswith(('moe.experts', 'moe.attr_heads', 'modal_projector')):
                    p.requires_grad_(False)
            logger.info('[attr-pretrain] 专家与属性头已冻结')

    trainable = [p for p in model.parameters() if p.requires_grad]
    ema = T.EMA(model, decay=args.ema_rate)
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total_steps, warm = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        return 0.5 * (1 + math.cos(min((step - warm) / max(total_steps - warm, 1), 1.0) * math.pi))
    sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def sample(noisy_norm):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
            out = model.sample(noisy_norm, num_steps=args.ddim_steps)
        return out.float()

    def base_pred_for(batch, noisy):
        """res_diff 的基础估计：OOF 查表（训练集）或在线计算。"""
        if args.model_type not in ('res_diff', 'res_cnn') or oof_map is None:
            return None
        ids = batch['clean_id'].tolist()
        if all(i in oof_map for i in ids) and args.phys_aug == 'none' and not args.augment:
            return (oof_pred[[oof_map[i] for i in ids]].to(device) / data_scale)
        return None

    history, best_ber, best_epoch, best_metrics = [], float('inf'), -1, {}
    csv_path = os.path.join(out_dir, 'val_log.csv')
    csv_f = open(csv_path, 'w', newline=''); cw = csv.writer(csv_f); cw.writerow(['epoch', 'train_loss', 'val_loss', 'val_mse', 'val_ber', 'val_ber_input', 'time_s'])
    status = 0

    # ------------------------------------------------------------ 训练循环
    for epoch in range(args.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in loader:
            clean = batch['interf' if args.target == 'interf' else 'clean'].to(device) / data_scale
            noisy = batch['noisy'].to(device) / data_scale
            aux_set(batch)
            if args.model_type in ('res_diff', 'res_cnn'):
                model._aux['base_pred'] = base_pred_for(batch, noisy)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):
                result = model(noisy, clean) if args.model_type in ('cnn', 'restormer') else model(clean, noisy)
            if args.amp == 'bf16':
                result = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in result.items()}
            if is_reg:
                loss = result['loss']
            else:
                loss, _ = criterion(
                    noise_pred=result['noise_pred'], noise_target=result['noise_target'], x0_pred=result.get('x0_pred'), x0_target=result.get('x0_target'),
                    gates=(None if args.fusion == 'anchor' else result.get('gates')), balance_loss=result.get('balance_loss'),
                    x0_weight=(base_model.scheduler.alphas_cumprod.to(device)[result['t']] if args.x0_loss_weighting == 'alpha' and 't' in result else None),
                    route_logits=result.get('route_logits') if args.router_type == 'prior' else None,
                    route_target=(route_prior[batch['interference_type'].to(device)] if args.router_type == 'prior' and route_prior is not None else None),
                    diversity_loss=result.get('diversity_loss') if args.router_type == 'prior' else None)
                if stft_loss is not None and result.get('x0_pred') is not None:
                    _w = base_model.scheduler.alphas_cumprod.to(device)[result['t']] if args.x0_loss_weighting == 'alpha' and 't' in result else None
                    loss = loss + args.stft_weight * stft_loss(result['x0_pred'], result['x0_target'], _w)
                al = result.get('aux_losses') or {}
                if 'attr_loss' in al and args.attr_mode in ('joint',):
                    loss = loss + args.attr_weight * al['attr_loss'].float()
                if args.extra_local == 'e1' and getattr(base_model, 'e1_loss', None) is not None:
                    loss = loss + args.e1_weight * base_model.e1_loss.float()
                if 'sir_nll' in al:
                    loss = loss + args.sir_weight * al['sir_nll'].float()
                if 'morph_attr' in al:
                    loss = loss + args.morph_attr_weight * al['morph_attr'].float()
            if not torch.isfinite(loss):
                logger.error('loss 非有限 (epoch %d)，中止' % epoch); status = 3; break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step(); sched.step(); ema.update()
            ep_loss += loss.item(); nb += 1
        if status:
            break
        train_loss = ep_loss / max(nb, 1)

        # ---------------- 验证（EMA 权重）
        if (epoch + 1) % max(1, args.val_every) == 0 or epoch == args.epochs - 1:
            model.eval(); ema.apply_shadow()
            if hasattr(model, 'set_aux'):
                model.set_aux()                        # 推理期不带任何标签（oracle_sir 变体除外，它在验证中取真实 SIR）
            vls, ber_l, ber_in, mse_l = 0.0, [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    clean_v, noisy_v = batch['interf' if args.target == 'interf' else 'clean'].to(device), batch['noisy'].to(device)
                    nv = noisy_v / data_scale
                    if args.oracle_cond != 'none' or args.extra_local == 'oracle_mask':
                        model.set_aux(oracle=(build_oracle(batch, args.oracle_cond, device) if args.oracle_cond != 'none' else None),
                                      interf=((batch['interf'].to(device) / data_scale) if args.extra_local == 'oracle_mask' else None))
                    if hasattr(model, 'set_aux') and args.router_type == 'dual' and args.dual_variant == 'oracle_sir':
                        model.set_aux(sir=batch['sir_db'].to(device))
                    if args.model_type in ('res_diff', 'res_cnn'):
                        model._aux['base_pred'] = None
                    rv = model(nv, clean_v / data_scale) if args.model_type in ('cnn', 'restormer') else model(clean_v / data_scale, nv)
                    if is_reg:
                        vls += float(rv['loss'])
                    else:
                        vl, _ = criterion(noise_pred=rv['noise_pred'], noise_target=rv['noise_target'], gates=(None if args.fusion == 'anchor' else rv.get('gates')), balance_loss=rv.get('balance_loss'))
                        vls += float(vl)
                    if args.model_type in ('res_diff', 'res_cnn') and hasattr(model, 'gate_sir'):
                        model.gate_sir = batch['sir_db'].to(device)
                    rec = sample(nv) * data_scale
                    if args.target == 'clean':
                        ber_l += batch_ber(rec, batch, meta, mod_names).tolist()
                        ber_in += batch_ber(noisy_v, batch, meta, mod_names).tolist()
                    else:
                        ber_l += [0.0]; ber_in += [0.0]
                    mse_l.append(float(((rec - clean_v) ** 2).mean()))
            ema.restore()
            if hasattr(model, 'set_aux'):
                model.set_aux()
            val_loss, val_ber, val_in, val_mse = vls / len(val_loader), float(np.mean(ber_l)), float(np.mean(ber_in)), float(np.mean(mse_l))
            sqi = 10 * math.log10(float(np.mean([((b['noisy'] - b['clean']) ** 2).mean() for b in val_ds.samples[:64]])) / (val_mse + 1e-10))
            logger.info('Epoch %d: train_loss=%.6f  val_loss=%.6f' % (epoch, train_loss, val_loss))
            logger.info('Epoch %d: [EVAL] MSE=%.6f, SQI=%.4f dB, BER=%.4f (不处理直接解调: %.4f)' % (epoch, val_mse, sqi, val_ber, val_in))
            cw.writerow([epoch, train_loss, val_loss, val_mse, val_ber, val_in, round(time.time() - t_start, 1)]); csv_f.flush()
            score = val_ber if (args.select_metric == 'ber' and args.target == 'clean') else (val_mse if args.target == 'interf' else -sqi)
            if epoch >= args.min_best_epoch and score < best_ber:
                best_ber, best_epoch = score, epoch
                best_metrics = {'ber': val_ber, 'mse': val_mse, 'sqi': sqi, 'ber_input': val_in, 'epoch': epoch}
                torch.save({'epoch': epoch, 'model_state_dict': base_model.state_dict(), 'ema_shadow': ema.shadow, 'data_scale': data_scale,
                            'best_metrics': best_metrics, 'args': vars(args)}, os.path.join(out_dir, 'best_model.pth'))
                logger.info('  ★ Saved best model (val BER %.4f)' % val_ber)
            if (epoch + 1) >= args.abort_chance_frac * args.epochs and val_ber >= 0.45 and not is_reg and args.target == 'clean':
                logger.error('训练过半验证 BER 仍为机会水平 (%.3f)，早期中止' % val_ber); status = 4; break
        else:
            logger.info('Epoch %d: train_loss=%.6f  val=SKIPPED' % (epoch, train_loss))
    csv_f.close()

    # ------------------------------------------------------------ 总结与快速测试
    elapsed = time.time() - t_start
    summary = {'exp_name': args.exp_name, 'status': status, 'best_val_ber': best_metrics.get('ber'), 'best_epoch': best_epoch, 'val_ber_input': best_metrics.get('ber_input'),
               'val_mse': best_metrics.get('mse'), 'epochs': args.epochs, 'train_types': train_types, 'holdout': args.holdout, 'params': n_params, 'trainable_params': n_train,
               'seconds': round(elapsed, 1), 'model_type': args.model_type}
    json.dump(summary, open(os.path.join(out_dir, 'val_summary.json'), 'w'), indent=1)
    extra_info = {'res_scale': float(model.res_scale), 'res_val_std': res_val_std} if args.model_type in ('res_diff', 'res_cnn') else {}
    json.dump({'params': n_params, 'trainable': n_train, 'seconds': round(elapsed, 1), 'data_scale': data_scale, **extra_info}, open(os.path.join(out_dir, 'run_info.json'), 'w'))
    if status == 0 and not args.skip_final_eval and args.target == 'clean' and os.path.exists(os.path.join(out_dir, 'best_model.pth')):
        ck = torch.load(os.path.join(out_dir, 'best_model.pth'), map_location=device, weights_only=False)
        for n_, p in model.named_parameters():
            if n_ in ck['ema_shadow']:
                p.data = ck['ema_shadow'][n_].to(device)
        model.eval()
        if hasattr(model, 'set_aux'):
            model.set_aux()
        res = {}
        for split in ('test', 'test_ood'):
            ds = MoEDiffDataset(root, split, noise_mode=args.noise_mode, mods=args.new_mods)
            ds.samples = ds.samples[::4]
            per = {}
            with torch.no_grad():
                for i in range(0, len(ds.samples), 128):
                    items = [ds[j] for j in range(i, min(i + 128, len(ds.samples)))]
                    batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
                    nv = batch['noisy'].to(device) / data_scale
                    if args.model_type in ('res_diff', 'res_cnn'):
                        model.gate_sir = batch['sir_db'].to(device)
                    if args.oracle_cond != 'none' or args.extra_local == 'oracle_mask':
                        model.set_aux(oracle=(build_oracle(batch, args.oracle_cond, device) if args.oracle_cond != 'none' else None),
                                      interf=((batch['interf'].to(device) / data_scale) if args.extra_local == 'oracle_mask' else None))
                    if hasattr(model, 'set_aux') and args.router_type == 'dual' and args.dual_variant == 'oracle_sir':
                        model.set_aux(sir=batch['sir_db'].to(device))
                    rec = sample(nv) * data_scale
                    b = batch_ber(rec, batch, meta, mod_names)
                    for t_i, v in zip(batch['interference_type'].tolist(), b.tolist()):
                        per.setdefault(ds.type_names[t_i], []).append(v)
            res[split] = {'ber': float(np.mean([v for l in per.values() for v in l])), 'per_type': {k: float(np.mean(v)) for k, v in per.items()}, 'n': len(ds.samples)}
            logger.info('[quick %s] BER=%.4f  per_type=%s' % (split, res[split]['ber'], {k: round(v, 3) for k, v in res[split]['per_type'].items()}))
        json.dump(res, open(os.path.join(out_dir, 'test_results.json'), 'w'), indent=1)
    if status == 0 and held_samples:
        ck = torch.load(os.path.join(out_dir, 'best_model.pth'), map_location=device, weights_only=False)
        for n_, p in model.named_parameters():
            if n_ in ck['ema_shadow']:
                p.data = ck['ema_shadow'][n_].to(device)
        model.eval(); preds, ids = [], []
        with torch.no_grad():
            for i in range(0, len(held_samples), 128):
                it = held_samples[i:i + 128]
                nv = torch.from_numpy(np.stack([s['noisy'] for s in it])).to(device) / data_scale
                preds.append((sample(nv) * data_scale).cpu().numpy()); ids += [s['clean_id'] for s in it]
        np.savez(os.path.join(out_dir, 'oof_fold.npz'), clean_id=np.array(ids), pred=np.concatenate(preds).astype(np.float32))
        logger.info('[oof] 已保存样本外预测 %d 条' % len(ids))
    if status == 0 and not args.keep_ckpt:
        pass            # 是否删除 best_model.pth 由任务清单决定（评估完成后）
    logger.info('Done status=%d best_val_ber=%s @ep%d (%.0fs)' % (status, best_metrics.get('ber'), best_epoch, elapsed))
    return status


if __name__ == '__main__':
    sys.exit(main())
