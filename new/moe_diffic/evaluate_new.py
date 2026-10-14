"""新数据集的完整评估：加载训练好的模型，在 test / test_ood 上采样，并保存逐样本指标。

用法（在 new/moe_diffic 下）：
    python evaluate_new.py --runs exp_moe_legacy exp_nomoe exp_moe_prior --splits test test_ood

逐样本指标（模型输出 与 不处理的输入[=匹配滤波基线] 各一份）：
    BER、EVM(符号域)、MSE、MAE、全带 SINR、带内 SINR（越大越好）。
输出：runs/<run>/per_sample_<split>.npz
"""
import argparse
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
from models import PhysicsMoEModel  # noqa: E402
from moe_dataset import MoEDiffDataset  # noqa: E402
from metrics_new import batch_ber, batch_evm, batch_inband_sinr  # noqa: E402


def build_model(run, meta, device, dps_step=None, dual_gamma=None):
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
    elif mt == 'mv_cnn':
        from models.baseline_new import MultiViewCNNBaseline
        model = MultiViewCNNBaseline(cfg, base_ch=getattr(a, 'cnn_base_ch', 32))
    elif mt == 'dit_reg':
        from models.baseline_new import DiTRegressionBaseline
        model = DiTRegressionBaseline(cfg)
    elif mt == 'cnn':
        from models.baseline.cnn_baseline import PhysicsCNNBaseline
        model = PhysicsCNNBaseline(in_channels=2, base_ch=getattr(a, 'cnn_base_ch', 32), signal_length=cfg.data.signal_length)
    elif mt == 'diffcom_1d':
        from models.baseline_new import DiffCom1DBaseline
        model = DiffCom1DBaseline(cfg, dps_step_size=a.dps_step_size if dps_step is None else dps_step)
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
    ap.add_argument('--dual_gamma', type=float, default=None, help='ICDM 双模型的干扰分数权重 γ（覆盖训练时的值）')
    ap.add_argument('--stride', type=int, default=1, help='每隔 stride 条取一条（调参时用，加快速度）')
    ap.add_argument('--tag', default='', help='输出文件名后缀')
    ap.add_argument('--types', nargs='+', default=None, help='只评估这些干扰类型（调参时限定为训练见过的类型，避免泄漏）')
    ap.add_argument('--avg_k', type=int, default=1, help='扩散类模型：每个样本独立随机采样 K 次再取平均（后验均值的蒙特卡洛近似），K=1 即单次采样')
    ap.add_argument('--init_scale', type=float, default=1.0, help='扩散采样的初始噪声缩放（仅 MoE-DiffIC 模型）；1=标准，0=确定性')
    a_ = ap.parse_args()
    dev = torch.device(a_.device)

    for run in a_.runs:
        for split in a_.splits:
            ds = MoEDiffDataset(a_.data_root, split, noise_mode='with', types=a_.types)
            meta, mods = ds.meta, ds.mod_names
            model, scale, mtype = build_model(run, meta, dev, a_.dps_step, a_.dual_gamma)
            if a_.stride > 1:
                ds.samples = ds.samples[::a_.stride]
            torch.manual_seed(a_.seed)

            R = {k: [] for k in ['type', 'mod', 'sir', 'snr', 'clean_id',
                                 'ber_model', 'ber_mf', 'evm_model', 'evm_mf', 'mse_model', 'mse_mf', 'mae_model', 'mae_mf',
                                 'sinr_model', 'sinr_mf', 'sinr_inband_model', 'sinr_inband_mf']}
            t0 = time.time()
            gate_list = []
            for i in range(0, len(ds), a_.batch_size):
                items = [ds[j] for j in range(i, min(i + a_.batch_size, len(ds)))]
                batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
                clean, noisy = batch['clean'].to(dev), batch['noisy'].to(dev)
                with torch.no_grad():
                    if mtype in ('moe_diffmic', 'diffcom_1d', 'icdm_1d') and a_.init_scale != 1.0:
                        rec = model.sample(noisy / scale, num_steps=a_.ddim_steps, init_scale=a_.init_scale) * scale
                    elif mtype in ('moe_diffmic', 'diffcom_1d', 'icdm_1d') and a_.avg_k > 1:
                        rec = torch.stack([model.sample(noisy / scale, num_steps=a_.ddim_steps) for _ in range(a_.avg_k)]).mean(0) * scale
                    else:
                        rec = model.sample(noisy / scale, num_steps=a_.ddim_steps) * scale
                    if mtype == 'moe_diffmic' and getattr(model, 'moe_enabled', False):     # 记录路由权重
                        _, _info = model.compute_moe_condition(noisy / scale, training=False)
                        if _info['gates'] is not None:
                            gate_list.append(_info['gates'].cpu().numpy())
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
            np.savez(out, **{k: np.array(v) for k, v in R.items()}, type_names=np.array(ds.type_names), mod_names=np.array(mods), **extra)
            print('%s %s: %d 条, BER=%.4f (MF %.4f), %.0fs -> %s' % (run, split, len(ds), np.mean(R['ber_model']), np.mean(R['ber_mf']),
                                                                   time.time() - t0, os.path.relpath(out, HERE)), flush=True)


if __name__ == '__main__':
    main()
