#!/usr/bin/env python3
import argparse
import glob
import json
import os
import time
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch

from models import create_lmmse_baseline, create_matched_filtering_baseline


def _rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    n = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64)
    t = n / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-12:
            taps[i] = beta / np.sqrt(2.0) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
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


def _load_type_npz(type_dir: str):
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


def _compute_ber(method: str, model, noisy_iq: np.ndarray, rec: np.ndarray, clean: np.ndarray, bits: np.ndarray, device: torch.device) -> np.ndarray:
    sps = 16
    offset = sps // 2
    if method == 'mf':
        rec_mf = rec
    else:
        taps = _rrc_taps(beta=0.5, sps=sps, span=4)
        rec_mf = _match_filter_batch(rec, taps)
    sym_rec = _sample_symbols(rec_mf, bits.shape[1] // 2, sps, offset)
    sym_rec = sym_rec / np.sqrt(float(sps))
    return (_bits_from_symbols_qpsk(sym_rec) != bits).mean(axis=1).astype(np.float32)


def _fixed_symbol_offset(sps: int, tau_s: int = 8) -> int:
    return int(tau_s % sps)


def _agg_by_sir(sir_db, clean_iq, rec_iq, mse, mae, sinr, delta_sir, ber):
    out = {}
    sir_int = np.round(sir_db).astype(np.int32)
    for k in sorted(np.unique(sir_int).tolist()):
        m = sir_int == k
        clean_sel = clean_iq[m]
        rec_sel = rec_iq[m]
        ss_res = float(((rec_sel - clean_sel) ** 2).sum())
        mean_clean = float(clean_sel.mean())
        ss_tot = float(((clean_sel - mean_clean) ** 2).sum())
        out[str(int(k))] = {
            'mse_mean': float(mse[m].mean()),
            'mae_mean': float(mae[m].mean()),
            'sinr_mean': float(sinr[m].mean()),
            'delta_sir_db': float(delta_sir[m].mean()),
            'ber_mean': float(ber[m].mean()),
            'count': int(m.sum()),
        }
    return out


def _update_global_sir_stats(stats, sir_db, clean_iq, rec_iq, mse, mae, sinr, delta_sir, ber):
    sir_int = np.round(sir_db).astype(np.int32)
    for k in np.unique(sir_int):
        m = sir_int == k
        clean_sel = clean_iq[m]
        rec_sel = rec_iq[m]
        e = stats.setdefault(int(k), {
            'count': 0,
            'mse_sum': 0.0,
            'mae_sum': 0.0,
            'sinr_sum': 0.0,
            'delta_sir_sum': 0.0,
            'ber_sum': 0.0,
            'ss_res_sum': 0.0,
            'clean_sum': 0.0,
            'clean_sumsq': 0.0,
            'clean_elem_count': 0,
        })
        e['count'] += int(m.sum())
        e['mse_sum'] += float(mse[m].sum())
        e['mae_sum'] += float(mae[m].sum())
        e['sinr_sum'] += float(sinr[m].sum())
        e['delta_sir_sum'] += float(delta_sir[m].sum())
        e['ber_sum'] += float(ber[m].sum())
        e['ss_res_sum'] += float(((rec_sel - clean_sel) ** 2).sum())
        e['clean_sum'] += float(clean_sel.sum())
        e['clean_sumsq'] += float((clean_sel ** 2).sum())
        e['clean_elem_count'] += int(clean_sel.size)


def _finalize_global_sir(stats):
    out = {}
    for k in sorted(stats.keys()):
        s = stats[k]
        cnt = max(int(s['count']), 1)
        elem_cnt = max(int(s['clean_elem_count']), 1)
        ss_tot = float(s['clean_sumsq'] - (s['clean_sum'] ** 2) / elem_cnt)
        out[str(int(k))] = {
            'mse_mean': float(s['mse_sum'] / cnt),
            'mae_mean': float(s['mae_sum'] / cnt),
            'sinr_mean': float(s['sinr_sum'] / cnt),
            'delta_sir_db': float(s['delta_sir_sum'] / cnt),
            'ber_mean': float(s['ber_sum'] / cnt),
            'count': int(s['count']),
        }
    return out


def _infer_signal_length(train_root: str) -> int:
    for t in sorted(d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))):
        npzs = sorted(glob.glob(os.path.join(train_root, t, '*.npz')))
        if npzs:
            d = np.load(npzs[0], allow_pickle=True)
            return int(np.asarray(d['clean_signal']).shape[-1])
    raise RuntimeError(f'Cannot infer signal_length from {train_root}')


def _create_model(method: str, signal_length: int):
    config = SimpleNamespace(data=SimpleNamespace(signal_length=signal_length, num_channels=2))
    if method == 'mf':
        return create_matched_filtering_baseline(config)
    if method == 'lmmse':
        return create_lmmse_baseline(config)
    raise ValueError(f'Unknown method: {method}')


def _train_traditional(model, train_root: str, device: torch.device):
    model = model.to(device)
    model.train()
    type_dirs = sorted(d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d)))
    seen = 0
    for t in type_dirs:
        dct = _load_type_npz(os.path.join(train_root, t))
        clean = torch.from_numpy(_complex_to_iq(dct['clean'])).to(device)
        noisy = torch.from_numpy(_complex_to_iq(dct['noisy'])).to(device)
        with torch.no_grad():
            _ = model(clean, noisy)
        seen += clean.shape[0]
    model.eval()
    return model, int(seen)


def evaluate_one(method: str, dataset_name: str, train_root: str, test_root: str, output_root: str, batch_size: int, device: str, fit_stats: bool):
    dev = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    signal_length = _infer_signal_length(train_root)
    model = _create_model(method, signal_length)
    if fit_stats:
        model, train_seen = _train_traditional(model, train_root, dev)
    else:
        model = model.to(dev)
        model.eval()
        train_seen = 0

    signal_types = sorted(d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d)))
    all_type_results = {}
    global_sir_stats = {}

    global_mse_list = []
    global_mae_list = []
    global_sinr_list = []
    global_delta_sir_list = []
    global_ber_list = []
    global_count = 0
    global_mse_sum = 0.0
    global_mae_sum = 0.0
    global_sinr_sum = 0.0
    global_delta_sir_sum = 0.0
    global_ber_sum = 0.0

    for sig_type in signal_types:
        type_dir = os.path.join(test_root, sig_type)
        data = _load_type_npz(type_dir)
        clean, noisy, bits, sir_db = data['clean'], data['noisy'], data['bits'], data['sir_db']
        n = clean.shape[0]
        rec = np.empty_like(clean, dtype=np.complex64)

        t0 = time.time()
        for st in range(0, n, batch_size):
            ed = min(st + batch_size, n)
            noisy_iq_t = torch.from_numpy(_complex_to_iq(noisy[st:ed])).to(dev)
            with torch.no_grad():
                pred_iq = model.sample(noisy_iq_t)
            rec[st:ed] = _iq_to_complex(pred_iq.detach().cpu().numpy())
        elapsed = time.time() - t0

        clean_iq = _complex_to_iq(clean)
        noisy_iq = _complex_to_iq(noisy)
        rec_iq = _complex_to_iq(rec)

        ber = _compute_ber(method, model, noisy_iq, rec, clean, bits, dev)

        mse = ((rec_iq - clean_iq) ** 2).mean(axis=(1, 2))
        mae = np.abs(rec_iq - clean_iq).mean(axis=(1, 2))
        sig_pow = (clean_iq ** 2).mean(axis=(1, 2))
        sinr = 10.0 * np.log10(sig_pow / (mse + 1e-10))
        nb = ((noisy_iq - clean_iq) ** 2).mean(axis=(1, 2))
        sinr_noisy = 10.0 * np.log10(sig_pow / (nb + 1e-10))
        delta_sir = sinr - sinr_noisy

        ss_res = float(((rec_iq - clean_iq) ** 2).sum())
        ss_tot = float(((clean_iq - clean_iq.mean()) ** 2).sum())

        sir_results = _agg_by_sir(sir_db, clean_iq, rec_iq, mse, mae, sinr, delta_sir, ber)

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

        _update_global_sir_stats(global_sir_stats, sir_db, clean_iq, rec_iq, mse, mae, sinr, delta_sir, ber)

        global_mse_list.append(float(mse.mean()))
        global_mae_list.append(float(mae.mean()))
        global_sinr_list.append(float(sinr.mean()))
        global_delta_sir_list.append(float(delta_sir.mean()))
        global_ber_list.append(float(ber.mean()))
        global_mse_sum += float(mse.sum())
        global_mae_sum += float(mae.sum())
        global_sinr_sum += float(sinr.sum())
        global_delta_sir_sum += float(delta_sir.sum())
        global_ber_sum += float(ber.sum())
        global_count += int(n)

    result = {
        'method': method,
        'dataset': dataset_name,
        'train_root': train_root,
        'test_root': test_root,
        'global': {
            'mse_mean': float(global_mse_sum / max(global_count, 1)),
            'mae_mean': float(global_mae_sum / max(global_count, 1)),
            'sinr_mean': float(global_sinr_sum / max(global_count, 1)),
            'delta_sir_db': float(global_delta_sir_sum / max(global_count, 1)),
            'ber_mean': float(global_ber_sum / max(global_count, 1)),
            'sample_count': int(global_count),
        },
        'global_sir_results': _finalize_global_sir(global_sir_stats),
        'per_type': all_type_results,
        'training_stats': {
            'mode': 'fit_stats' if fit_stats else 'no_train',
            'samples_seen': int(train_seen),
        },
    }

    os.makedirs(output_root, exist_ok=True)
    out_json = os.path.join(output_root, f'eval_n100_{method}_{dataset_name}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_json, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--methods', nargs='+', default=['mf', 'lmmse'])
    p.add_argument('--run_root', type=str, default='runs/traditional_methods')
    p.add_argument('--fit_stats', action='store_true', default=False,
                   help='传统方法是否先用训练集估计统计量；默认关闭(直接测试集推理)')
    args = p.parse_args()

    jobs = [
        ('sim2560', 'dataset/sim_2560/train_2560/M005', 'dataset/sim_2560/test_2560/N100'),
        ('realdata', 'dataset/realdata/train/M005', 'dataset/realdata/test/N100'),
    ]
    jobs = [job for job in jobs if os.path.isdir(job[1]) and os.path.isdir(job[2])]
    if not jobs:
        raise FileNotFoundError('No valid train/test job pairs found for traditional evaluation')

    all_out = []
    for method in args.methods:
        for dname, train_root, test_root in jobs:
            out_dir = os.path.join(args.run_root, f'{method}_{dname}')
            print(f'\n=== Evaluate {method} on {dname} ===')
            out_json, res = evaluate_one(
                method=method,
                dataset_name=dname,
                train_root=train_root,
                test_root=test_root,
                output_root=out_dir,
                batch_size=args.batch_size,
                device=args.device,
                fit_stats=args.fit_stats,
            )
            print(f"saved: {out_json}")
            print(f"global: samples={res['global']['sample_count']} mse={res['global']['mse_mean']:.6f} ber={res['global']['ber_mean']:.6f}")
            all_out.append(out_json)

    summary_path = os.path.join(args.run_root, 'eval_n100_traditional_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'files': all_out}, f, indent=2, ensure_ascii=False)
    print(f'\nsummary: {summary_path}')


if __name__ == '__main__':
    main()
