#!/usr/bin/env python3
"""
增强评估脚本：添加 MAE, ΔSIRdb, BER 等指标
支持仿真和真实数据集评估
"""
import argparse
import json
import os
import time
from types import SimpleNamespace
from typing import Dict, Tuple

import glob
import numpy as np
import torch
import torch.nn as nn


def _sample_model(model, noisy_norm: torch.Tensor, num_steps: int) -> torch.Tensor:
    if isinstance(model, nn.DataParallel):
        return model.module.sample(noisy_norm, num_steps=num_steps)
    return model.sample(noisy_norm, num_steps=num_steps)


def _rebuild_model_from_ckpt(ckpt: dict, device: torch.device, multi_gpu: bool = False):
    saved_args = ckpt.get('args', {})
    model_type = saved_args.get('model_type', 'moe_diffmic')
    data_scale = float(ckpt.get('data_scale', 1.0))
    signal_length = int(saved_args.get('signal_length', 128))
    pos_embed = ckpt.get('model_state_dict', {}).get('denoiser.pos_embed', None)
    if isinstance(pos_embed, torch.Tensor) and pos_embed.ndim == 3:
        patch_size = 4
        signal_length = int(pos_embed.shape[1]) * patch_size

    config = SimpleNamespace(
        data=SimpleNamespace(signal_length=signal_length, num_channels=2),
        model=SimpleNamespace(
            type='cddm' if model_type == 'cddm' else ('cnn' if model_type == 'cnn' else 'physics_moe'),
            condition_dim=256,
            freq_loss_weight=saved_args.get('rcl_coef', 0.1) if model_type == 'cnn' else 0.0,
        ),
        moe=SimpleNamespace(
            num_experts=4, input_dim=256, hidden_dim=256, expert_output_dim=256, condition_dim=256,
            k=saved_args.get('moe_k', 2), enabled=not saved_args.get('disable_moe', False),
            noisy_gating=True, noise_epsilon=0.01, balance_loss_coef=saved_args.get('balance_loss_coef', 0.005),
        ),
        modal=SimpleNamespace(n_fft=32, hop_length=8, n_scales=3),
        channel=SimpleNamespace(channel_dim=64, hidden_dim=128),
        denoiser=SimpleNamespace(backbone='dit', hidden_size=256, num_heads=4, depth=4, mlp_ratio=4.0, patch_size=4),
        diffusion=SimpleNamespace(
            timesteps=saved_args.get('diffusion_steps', 1000),
            test_timesteps=saved_args.get('ddim_steps', 20),
            beta_schedule=saved_args.get('beta_schedule', 'cosine'),
            beta_start=0.0001, beta_end=0.02, model_var_type='fixedsmall',
        ),
    )

    if model_type == 'moe_diffmic':
        from models import PhysicsMoEModel
        model = PhysicsMoEModel(config)
    elif model_type == 'icdm':
        from models import ICDMBaseline
        model = ICDMBaseline(config)
    elif model_type == 'diffcom_1d':
        from models import DiffCom1DBaseline
        model = DiffCom1DBaseline(config, dps_step_size=float(saved_args.get('dps_step_size', 0.10)))
    elif model_type == 'icdm_1d':
        from models import ICDMDual1DBaseline
        model = ICDMDual1DBaseline(config, dual_gamma=float(saved_args.get('icdm_dual_gamma', 1.0)))
    elif model_type == 'cnn':
        from models import PhysicsCNNBaseline
        model = PhysicsCNNBaseline(
            in_channels=config.data.num_channels, base_ch=32, signal_length=config.data.signal_length,
            freq_loss_weight=config.model.freq_loss_weight,
        )
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    ema_shadow = ckpt.get('ema_shadow', {})
    state_dict = ckpt['model_state_dict']
    if ema_shadow:
        merged = {k: ema_shadow.get(k, v) for k, v in state_dict.items()}
        model.load_state_dict(merged, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if multi_gpu and device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    return model, data_scale


def _rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    n = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64)
    t = n / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-12:
            taps[i] = beta / np.sqrt(2.0) * ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / (den + 1e-12)
    taps = taps / np.sqrt(np.sum(taps ** 2) + 1e-12)
    return taps.astype(np.float32)


def _complex_to_iq(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag], axis=1).astype(np.float32)


def _iq_to_complex(x_iq: np.ndarray) -> np.ndarray:
    return (x_iq[:, 0, :] + 1j * x_iq[:, 1, :]).astype(np.complex64)


def _load_type_npz(type_dir: str) -> Dict[str, np.ndarray]:
    npz_list = sorted(glob.glob(os.path.join(type_dir, '*.npz')))
    if not npz_list:
        raise FileNotFoundError(f'No npz under: {type_dir}')
    clean_all, noisy_all, bits_all, sir_all = [], [], [], []
    for p in npz_list:
        d = np.load(p, allow_pickle=True)
        clean_all.append(np.asarray(d['clean_signal']).astype(np.complex64))
        noisy_all.append(np.asarray(d['mixture_signal']).astype(np.complex64))
        bits_all.append(np.asarray(d['bit_stream']).astype(np.uint8))
        sir_all.append(np.asarray(d.get('sir_db_target', np.zeros((clean_all[-1].shape[0],), dtype=np.float32))).reshape(-1).astype(np.float32))
    return {
        'clean': np.concatenate(clean_all, axis=0),
        'noisy': np.concatenate(noisy_all, axis=0),
        'bits': np.concatenate(bits_all, axis=0),
        'sir_db': np.concatenate(sir_all, axis=0),
    }


def _match_filter_batch(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    y = np.empty_like(x, dtype=np.complex64)
    for i in range(x.shape[0]):
        y[i] = np.convolve(x[i], taps, mode='same').astype(np.complex64)
    return y


def _bits_from_symbols_qpsk(sym: np.ndarray) -> np.ndarray:
    b0 = (sym.real < 0).astype(np.uint8)
    b1 = (sym.imag < 0).astype(np.uint8)
    out = np.empty((sym.shape[0], sym.shape[1] * 2), dtype=np.uint8)
    out[:, 0::2] = b0
    out[:, 1::2] = b1
    return out


def _sample_symbols(x_mf: np.ndarray, num_symbols: int, sps: int, offset: int) -> np.ndarray:
    idx = offset + np.arange(num_symbols) * sps
    if idx[-1] >= x_mf.shape[1]:
        idx = idx[idx < x_mf.shape[1]]
    return x_mf[:, idx]


def _ber_with_qpsk_phase_search(sym: np.ndarray, bits_ref: np.ndarray) -> np.ndarray:
    rotations = [1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j]
    ber_best = np.full((sym.shape[0],), np.inf, dtype=np.float32)
    for r in rotations:
        bits_hat = _bits_from_symbols_qpsk((sym * r).astype(np.complex64))
        ber = (bits_hat != bits_ref).mean(axis=1).astype(np.float32)
        ber_best = np.minimum(ber_best, ber)
    return ber_best


def _estimate_best_offset(clean: np.ndarray, bits: np.ndarray, taps: np.ndarray, sps: int) -> int:
    num_symbols = bits.shape[1] // 2
    clean_mf = _match_filter_batch(clean, taps)
    best_off, best_ber = 0, 1.0
    for off in range(sps):
        sym = _sample_symbols(clean_mf, num_symbols, sps, off)
        if sym.shape[1] != num_symbols:
            continue
        ber = _ber_with_qpsk_phase_search(sym, bits).mean()
        if ber < best_ber:
            best_ber = float(ber)
            best_off = off
    return best_off


def _compute_ber_after_mf(rec: np.ndarray, bits: np.ndarray, taps: np.ndarray, sps: int) -> np.ndarray:
    offset = sps // 2
    rec_mf = _match_filter_batch(rec, taps)
    sym_rec = _sample_symbols(rec_mf, bits.shape[1] // 2, sps, offset)
    sym_rec = sym_rec / np.sqrt(float(sps))
    return (_bits_from_symbols_qpsk(sym_rec) != bits).mean(axis=1).astype(np.float32)


def _init_weighted_stats() -> Dict[str, float]:
    return {
        'count': 0,
        'mse_sum': 0.0,
        'mae_sum': 0.0,
        'sinr_sum': 0.0,
        'delta_sir_sum': 0.0,
        'ber_sum': 0.0,
    }


def _accumulate_weighted_stats(stats: Dict[str, float], mse: np.ndarray, mae: np.ndarray, sinr: np.ndarray,
                               delta_sir: np.ndarray, ber: np.ndarray) -> None:
    stats['count'] += int(mse.shape[0])
    stats['mse_sum'] += float(mse.sum())
    stats['mae_sum'] += float(mae.sum())
    stats['sinr_sum'] += float(sinr.sum())
    stats['delta_sir_sum'] += float(delta_sir.sum())
    stats['ber_sum'] += float(ber.sum())


def _finalize_weighted_stats(stats: Dict[str, float]) -> Dict[str, float]:
    cnt = max(int(stats.get('count', 0)), 1)
    return {
        'mse_mean': float(stats.get('mse_sum', 0.0) / cnt),
        'mae_mean': float(stats.get('mae_sum', 0.0) / cnt),
        'sinr_mean': float(stats.get('sinr_sum', 0.0) / cnt),
        'delta_sir_db': float(stats.get('delta_sir_sum', 0.0) / cnt),
        'ber_mean': float(stats.get('ber_sum', 0.0) / cnt),
        'sample_count': int(stats.get('count', 0)),
    }


def evaluate_model(run_dir: str, model_name: str, test_root: str, device: torch.device, batch_size: int, ddim_steps: int, output_suffix: str) -> Dict:
    """评估单个模型"""
    ckpt_path = os.path.join(run_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, data_scale = _rebuild_model_from_ckpt(ckpt, device)
    saved_args = ckpt.get('args', {})
    if ddim_steps <= 0:
        ddim_steps = int(saved_args.get('ddim_steps', 20))

    signal_types = sorted(
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    )
    all_type_results = {}
    global_stats = _init_weighted_stats()
    global_sir_stats: Dict[int, Dict[str, float]] = {}

    for sig_type in signal_types:
        type_dir = os.path.join(test_root, sig_type)
        if not os.path.isdir(type_dir):
            print(f'      [SKIP] {sig_type}')
            continue
        
        try:
            print(f'      {sig_type}...', end='', flush=True)
            data = _load_type_npz(type_dir)
            clean, noisy, bits, sir_db = data['clean'], data['noisy'], data['bits'], data['sir_db']
            n = clean.shape[0]
            rec = np.empty_like(clean, dtype=np.complex64)

            t0 = time.time()
            for st in range(0, n, batch_size):
                ed = min(st + batch_size, n)
                noisy_iq = torch.from_numpy(_complex_to_iq(noisy[st:ed])).to(device)
                with torch.no_grad():
                    pred_iq = _sample_model(model, noisy_iq / data_scale, num_steps=ddim_steps) * data_scale
                rec[st:ed] = _iq_to_complex(pred_iq.detach().cpu().numpy())
            elapsed = time.time() - t0

            clean_iq = _complex_to_iq(clean)
            noisy_iq = _complex_to_iq(noisy)
            rec_iq = _complex_to_iq(rec)
            
            # 先对模型输出波形做匹配滤波，再进行符号抽样和 BER 计算
            taps = _rrc_taps(beta=0.5, sps=16, span=4)
            # align with score_based_source_separation: fixed offset, no phase search
            ber = _compute_ber_after_mf(rec, bits, taps=taps, sps=16)
            
            # 计算指标
            mse = ((rec_iq - clean_iq) ** 2).mean(axis=(1, 2))
            mae = np.abs(rec_iq - clean_iq).mean(axis=(1, 2))
            sig_pow = (clean_iq ** 2).mean(axis=(1, 2))
            sinr = 10.0 * np.log10(sig_pow / (mse + 1e-10))
            
            # ΔSIRdb
            nb = ((noisy_iq - clean_iq) ** 2).mean(axis=(1, 2))
            sinr_noisy = 10.0 * np.log10(sig_pow / (nb + 1e-10))
            delta_sir = sinr - sinr_noisy

            sir_int = np.round(sir_db).astype(np.int32)
            sir_results = {}
            for k in sorted(np.unique(sir_int).tolist()):
                m = sir_int == k
                sir_results[str(int(k))] = {
                    'mse_mean': float(mse[m].mean()),
                    'mae_mean': float(mae[m].mean()),
                    'sinr_mean': float(sinr[m].mean()),
                    'delta_sir_db': float(delta_sir[m].mean()),
                    'ber_mean': float(ber[m].mean()),
                    'count': int(m.sum()),
                }
                bucket = global_sir_stats.setdefault(int(k), _init_weighted_stats())
                _accumulate_weighted_stats(bucket, mse[m], mae[m], sinr[m], delta_sir[m], ber[m])
            
            all_type_results[sig_type] = {
                'global': {
                    'mse_mean': float(mse.mean()),
                    'mae_mean': float(mae.mean()),
                    'sinr_mean': float(sinr.mean()),
                    'delta_sir_db': float(delta_sir.mean()),
                    'ber_mean': float(ber.mean()),
                    'count': int(n),
                    'elapsed_s': float(round(elapsed, 2)),
                },
                'sir_results': sir_results,
            }
            
            # 积累全局指标（按样本数加权）
            _accumulate_weighted_stats(global_stats, mse, mae, sinr, delta_sir, ber)

            print(f' OK')
        except Exception as e:
            print(f' FAILED: {e}')

    result = {
        'model_name': model_name,
        'checkpoint': ckpt_path,
        'ddim_steps': ddim_steps,
        'data_scale': data_scale,
        'global': _finalize_weighted_stats(global_stats),
        'global_sir_results': {str(k): _finalize_weighted_stats(v) for k, v in sorted(global_sir_stats.items())},
        'per_type': all_type_results,
    }

    # 保存JSON结果
    output_json = os.path.join(run_dir, f'eval_n100_{output_suffix}.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n        ✓ 保存到: {output_json}')

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', type=str, required=True)
    p.add_argument('--runs_root', type=str, default='runs')
    p.add_argument('--test_root', type=str, required=True)
    p.add_argument('--dataset_type', type=str, default='sim', help='sim or realdata')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--ddim_steps', type=int, default=20)
    args = p.parse_args()

    with open(args.model_config, 'r') as f:
        model_configs = json.load(f)

    dev = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    print(f'\n设备: {dev}')
    print(f'数据集类型: {args.dataset_type}')
    print(f'测试集: {args.test_root}\n')

    results = []
    for cfg in model_configs:
        model_dir = cfg['dir']
        model_name = cfg['name']
        output_suffix = cfg['suffix']
        
        run_dir = os.path.join(args.runs_root, model_dir)
        if not os.path.isdir(run_dir):
            print(f'[跳过] {model_dir} 不存在')
            continue
        
        print(f'▶ 评估: {model_name}')
        try:
            result = evaluate_model(run_dir, model_name, args.test_root, dev, args.batch_size, args.ddim_steps, output_suffix)
            results.append(result)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'[错误] {e}')
            torch.cuda.empty_cache()

    print(f'\n\n{"="*100}')
    print(f'综合结果 ({args.dataset_type.upper()} - N100):')
    print(f'{"="*100}')
    print(f'{"模型":<30} | {"MSE":<10} | {"MAE":<10} | {"SINR":<8} | {"ΔSIRdb":<8} | {"BER":<10}')
    print("-"*100)
    for r in results:
        g = r['global']
        print(f"{r['model_name']:<30} | {g['mse_mean']:<10.6f} | {g['mae_mean']:<10.6f} | {g['sinr_mean']:<8.3f} | {g['delta_sir_db']:<8.3f} | {g['ber_mean']:<10.6f}")


if __name__ == '__main__':
    main()
