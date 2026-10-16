"""新数据集的完整评估：加载训练好的模型，在 test / test_ood 上采样，并保存逐样本指标。

用法（在 new/moe_diffic 下）：
    python evaluate_new.py --runs exp_moe_legacy exp_nomoe exp_moe_prior --splits test test_ood

逐样本指标（模型输出 与 不处理的输入[=匹配滤波基线] 各一份）：
    BER、EVM(符号域)、MSE、MAE、全带 SINR、带内 SINR（越大越好）。
输出：runs/<run>/per_sample_<split>.npz
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'loader'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import train as T  # noqa: E402
from exp_config import build_oracle  # noqa: E402
from models import PhysicsMoEModel  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402
from metrics_new import batch_ber, batch_evm, batch_inband_sinr  # noqa: E402


def load_gate(model, run, name, device):
    """加载 train_gate.py 保存的修正门控（runs/<run>/gate_<name>.pt）。"""
    from models.correction_gate import CorrectionGate
    d = torch.load(os.path.join(HERE, 'runs', run, 'gate_%s.pt' % name), map_location=device)
    g = CorrectionGate(d['variant'], d['local']).to(device)
    g.load_state_dict(d['state'])
    model.set_gate(g.eval())
    return d['variant']


def build_model(run, meta, device, dps_step=None, dual_gamma=None, warm_start=False, data_root=None):
    """按训练时保存的参数重建任意类型的模型，返回 (model, data_scale, model_type)。"""
    rdir = os.path.join(HERE, 'runs', run)
    if os.path.exists(os.path.join(rdir, 'lmmse.pt')):
        from models.lmmse_new import LMMSEFilter
        d = torch.load(os.path.join(rdir, 'lmmse.pt'), map_location=device)
        return LMMSEFilter(d['W'], d['window'], d['overlap'], d['length']).to(device).eval(), 1.0, 'lmmse'
    ck = torch.load(os.path.join(rdir, 'best_model.pth'), map_location=device, weights_only=False)
    a = argparse.Namespace(**ck['args'])
    cfg = T.build_config(a)
    cfg.data.signal_length = int(meta['SIG_LEN'])
    cfg.data.samples_per_symbol = cfg.data.sps = int(meta['SPS'])
    cfg.data.rrc_span = int(meta['RRC_SPAN'])
    mt = a.model_type
    if mt == 'moe_diffmic':
        model = PhysicsMoEModel(cfg)
    elif mt == 'res_cnn':
        from models.residual_diffusion import ResidualCNNBaseline
        model = ResidualCNNBaseline(cfg, base_ch=getattr(a, 'cnn_base_ch', 32))
    elif mt == 'res_diff':
        from models.residual_diffusion import ResidualDiffusionModel
        model = ResidualDiffusionModel(cfg, base_ch=getattr(a, 'cnn_base_ch', 32))
    elif mt == 'mv_cnn':
        from models.baseline_new import MultiViewCNNBaseline
        model = MultiViewCNNBaseline(cfg, base_ch=getattr(a, 'cnn_base_ch', 32))
    elif mt == 'dit_reg':
        from models.baseline_new import DiTRegressionBaseline
        model = DiTRegressionBaseline(cfg)
    elif mt == 'restormer':
        from models.baseline.restormer_baseline import Restormer1DBaseline
        model = Restormer1DBaseline(in_channels=2, dim=getattr(a, 'restormer_dim', 48), signal_length=cfg.data.signal_length)
    elif mt == 'cnn':
        from models.baseline.cnn_baseline import PhysicsCNNBaseline
        model = PhysicsCNNBaseline(in_channels=2, base_ch=getattr(a, 'cnn_base_ch', 32), signal_length=cfg.data.signal_length)
    elif mt == 'diffcom_1d':
        from models.baseline_new import DiffCom1DBaseline
        warm = bool(warm_start or getattr(a, 'dps_warm_start', False))
        model = DiffCom1DBaseline(cfg, dps_step_size=a.dps_step_size if dps_step is None else dps_step, warm_start=warm)
        if warm:                                                    # 训练集平均 SINR：优先读训练时记录的值，否则按该 run 的训练类型重新统计
            jf = os.path.join(rdir, 'obs_snr.json')
            if os.path.exists(jf):
                model.obs_snr_db = json.load(open(jf))['obs_snr_db']
            else:
                from models.baseline_new import estimate_obs_snr_db
                tt = [t for t in (getattr(a, 'new_types', None) or meta['TYPES']) if t not in (getattr(a, 'holdout', None) or [])]
                tr = MoEDiffDataset(data_root, 'train', types=tt, noise_mode='with', max_per_cell=getattr(a, 'train_per_cell', None) or 4)
                model.obs_snr_db = estimate_obs_snr_db(tr.samples)
                json.dump({'obs_snr_db': model.obs_snr_db}, open(jf, 'w'))
    elif mt == 'icdm_1d':
        from models.baseline_new import ICDMDual1DBaseline
        model = ICDMDual1DBaseline(cfg, dual_gamma=a.icdm_dual_gamma if dual_gamma is None else dual_gamma)
    else:
        raise ValueError(mt)
    model = model.to(device)
    model.load_state_dict(ck['model_state_dict'])
    for n, p in model.named_parameters():              # 评估使用 EMA 参数（与训练时一致）
        if n in ck['ema_shadow']:
            p.data = ck['ema_shadow'][n].to(device)
    return model.eval(), ck['data_scale'], mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--splits', nargs='+', default=['test', 'test_ood'])
    ap.add_argument('--data_root', default=os.path.abspath(os.path.join(HERE, '..', 'data', 'v2')))
    ap.add_argument('--ddim_steps', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dps_step', type=float, default=None, help='DiffCom 的 DPS 步长（覆盖训练时的值）')
    ap.add_argument('--warm_start', action='store_true', help='DiffCom：官方式热启动（从接收信号起步）')
    ap.add_argument('--cf', choices=['zero', 'shuffle'], default=None, help='诊断：反事实条件（zero=条件置零；shuffle=条件换成别的样本的）')
    ap.add_argument('--dual_gamma', type=float, default=None, help='ICDM 双模型的干扰分数权重 γ（覆盖训练时的值）')
    ap.add_argument('--stride', type=int, default=1, help='每隔 stride 条取一条（调参时用，加快速度）')
    ap.add_argument('--tag', default='', help='输出文件名后缀')
    ap.add_argument('--types', nargs='+', default=None, help='只评估这些干扰类型（调参时限定为训练见过的类型，避免泄漏）')
    ap.add_argument('--avg_k', type=int, default=1, help='扩散类模型：每个样本独立随机采样 K 次再取平均（后验均值的蒙特卡洛近似），K=1 即单次采样')
    ap.add_argument('--amp', choices=['none', 'bf16'], default='none', help='采样用 bf16 自动混合精度（推理一致性已验证）；默认 none')
    ap.add_argument('--gate', default=None, help='residual diffusion：加载 runs/<run>/gate_<名>.pt 的修正门控')
    ap.add_argument('--parts', action='store_true', help='res_diff：额外保存基础估计的 BER/MSE（ber_base/mse_base）与平均 λ')
    ap.add_argument('--init_scale', type=float, default=1.0, help='扩散采样的初始噪声缩放（仅 MoE-DiffIC 模型）；1=标准，0=确定性')
    a_ = ap.parse_args()
    dev = torch.device(a_.device)

    for run in a_.runs:
        for split in a_.splits:
            _ra = json.load(open(os.path.join(HERE, 'runs', run, 'args.json'))) if os.path.exists(os.path.join(HERE, 'runs', run, 'args.json')) else {}
            oracle = _ra.get('oracle_cond', 'none')
            xl = _ra.get('extra_local', 'none')
            if oracle != 'none' or xl == 'oracle_mask':                                        # 诊断用 oracle 条件：需要干扰类型与属性（用训练集统计量标准化）
                _st = np.load(os.path.join(HERE, 'runs', run, 'attr_stats.npz'))
                ds = MoEDiffDataset(a_.data_root, split, noise_mode='with', types=a_.types, return_aux=True, attr_stats=(_st['mean'], _st['std']))
            else:
                ds = MoEDiffDataset(a_.data_root, split, noise_mode='with', types=a_.types)
            meta, mods = ds.meta, ds.mod_names
            model, scale, mtype = build_model(run, meta, dev, a_.dps_step, a_.dual_gamma, a_.warm_start, a_.data_root)
            if a_.stride > 1:
                ds.samples = ds.samples[::a_.stride]
            if a_.cf:
                model.cf_mode = a_.cf
            torch.manual_seed(a_.seed)
            if a_.gate and mtype == 'res_diff':
                load_gate(model, run, a_.gate, dev)
            if a_.amp == 'bf16':
                _orig_sample = model.sample

                def _amp_sample(*args, **kw):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out_ = _orig_sample(*args, **kw)
                    return tuple(o_.float() if torch.is_tensor(o_) else o_ for o_ in out_) if isinstance(out_, tuple) else out_.float()
                model.sample = _amp_sample

            R = {k: [] for k in ['type', 'mod', 'sir', 'snr', 'clean_id',
                                 'ber_model', 'ber_mf', 'evm_model', 'evm_mf', 'mse_model', 'mse_mf', 'mae_model', 'mae_mf',
                                 'sinr_model', 'sinr_mf', 'sinr_inband_model', 'sinr_inband_mf']}
            t0 = time.time()
            gate_list = []
            sir_est_list = []
            for i in range(0, len(ds), a_.batch_size):
                items = [ds[j] for j in range(i, min(i + a_.batch_size, len(ds)))]
                batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
                clean, noisy = batch['clean'].to(dev), batch['noisy'].to(dev)
                sir_t = batch['sir_db'].to(dev)
                if oracle != 'none' or xl == 'oracle_mask':
                    model.set_aux(oracle=(build_oracle(batch, oracle, dev) if oracle != 'none' else None),
                                  interf=((batch['interf'].to(dev) / scale) if xl == 'oracle_mask' else None))
                if hasattr(model, 'gate_sir'):
                    model.gate_sir = sir_t                          # 仅 oracle 门控使用（上界分析）
                if getattr(model, 'moe_enabled', False) and getattr(getattr(model, 'moe', None), 'router_type', '') == 'dual' and model.moe.router.variant == 'oracle_sir':
                    model.set_aux(sir=sir_t)                       # 仅 oracle_sir 路由变体使用（上界分析）
                with torch.no_grad():
                    if mtype in ('res_diff', 'res_cnn'):
                        K = max(1, a_.avg_k) if mtype == 'res_diff' else 1
                        parts = [model.sample(noisy / scale, num_steps=a_.ddim_steps, return_parts=True) for _ in range(K)]
                        rec = torch.stack([p_[0] for p_ in parts]).mean(0) * scale
                        sb_ = parts[0][1] * scale
                        lam_ = torch.stack([(p_[3].float().flatten(1).mean(1) if (torch.is_tensor(p_[3]) and p_[3].ndim > 0) else torch.ones(noisy.shape[0], device=dev)) for p_ in parts]).mean(0)
                        R.setdefault('ber_base', []).extend(batch_ber(sb_.cpu(), batch, meta, mods).tolist())
                        R.setdefault('mse_base', []).extend(((sb_.cpu() - clean.cpu()) ** 2).mean(dim=(1, 2)).tolist())
                        R.setdefault('lam_mean', []).extend(lam_.cpu().tolist())
                    elif mtype in ('moe_diffmic', 'diffcom_1d', 'icdm_1d') and a_.init_scale != 1.0:
                        rec = model.sample(noisy / scale, num_steps=a_.ddim_steps, init_scale=a_.init_scale) * scale
                    elif mtype in ('moe_diffmic', 'diffcom_1d', 'icdm_1d') and a_.avg_k > 1:
                        rec = torch.stack([model.sample(noisy / scale, num_steps=a_.ddim_steps) for _ in range(a_.avg_k)]).mean(0) * scale
                    else:
                        rec = model.sample(noisy / scale, num_steps=a_.ddim_steps) * scale
                    if mtype == 'moe_diffmic' and getattr(model, 'moe_enabled', False):     # 记录路由权重
                        _, _info = model.compute_moe_condition(noisy / scale, training=False)
                        if _info['gates'] is not None:
                            gate_list.append(_info['gates'].cpu().numpy())
                        _mu = (_info.get('aux_losses') or {}).get('sir_mu')
                        if _mu is not None:
                            sir_est_list.append((_mu.float() * 10.0).cpu().numpy())
                cl, no, re = clean.cpu(), noisy.cpu(), rec.cpu()
                pc = (cl ** 2).sum(dim=(1, 2))
                for tag, est in (('model', re), ('mf', no)):
                    err = est - cl
                    R['ber_' + tag] += batch_ber(est, batch, meta, mods).tolist()
                    R['evm_' + tag] += batch_evm(est, batch, meta, mods).tolist()
                    R['mse_' + tag] += (err ** 2).mean(dim=(1, 2)).tolist()
                    R['mae_' + tag] += err.abs().mean(dim=(1, 2)).tolist()
                    R['sinr_' + tag] += (10 * torch.log10(pc / ((err ** 2).sum(dim=(1, 2)) + 1e-20))).tolist()
                    R['sinr_inband_' + tag] += batch_inband_sinr(cl, est, batch, meta).tolist()
                R['type'] += batch['interference_type'].tolist()
                R['mod'] += batch['modulation'].tolist()
                R['sir'] += batch['sir_db'].tolist()
                R['snr'] += batch['snr_db'].tolist()
                R['clean_id'] += batch['clean_id'].tolist()
            out = os.path.join(HERE, 'runs', run, 'per_sample_%s%s.npz' % (split, a_.tag))
            extra = {'gates': np.concatenate(gate_list)} if gate_list else {}
            if sir_est_list:
                extra['sir_est'] = np.concatenate(sir_est_list)
            np.savez(out, **{k: np.array(v) for k, v in R.items()}, type_names=np.array(ds.type_names), mod_names=np.array(mods), **extra)
            print('%s %s: %d 条, BER=%.4f (MF %.4f), %.0fs -> %s' % (run, split, len(ds), np.mean(R['ber_model']), np.mean(R['ber_mf']),
                                                                   time.time() - t0, os.path.relpath(out, HERE)), flush=True)


if __name__ == '__main__':
    main()
