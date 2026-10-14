"""新数据集的 BER 计算：已知同步参数（频偏/初相/定时/增益）的匹配滤波硬判决。

与 new/gen/signal_gen.py 中的 demod_oracle / ber_interior 逻辑一致，但自包含（不依赖 gen/ 目录），
参数 (sps, n_sym, span, edge_skip) 来自数据集 metadata.json。
"""
import numpy as np

_T16 = 2.0 / np.sqrt(10.0)


def rrc(t, beta):
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    eps = 1e-8
    s0 = np.abs(t) < eps
    s1 = np.abs(np.abs(t) - 1.0 / (4 * beta)) < eps
    ns = ~(s0 | s1)
    tn = t[ns]
    out[ns] = (np.sin(np.pi * tn * (1 - beta)) + 4 * beta * tn * np.cos(np.pi * tn * (1 + beta))) / (
        np.pi * tn * (1 - (4 * beta * tn) ** 2))
    out[s0] = 1 - beta + 4 * beta / np.pi
    out[s1] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                   + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    return out


def symbols_to_bits(y, mod):
    """y: 已按增益归一化的复符号估计；返回硬判决 bit。映射与 gen/signal_gen.py 一致（Gray）。"""
    if mod == 'BPSK':
        return (y.real < 0).astype(np.uint8)
    if mod == 'QPSK':
        b = np.empty(2 * len(y), dtype=np.uint8)
        b[0::2] = (y.real < 0)
        b[1::2] = (y.imag < 0)
        return b
    if mod == '16QAM':
        b = np.empty(4 * len(y), dtype=np.uint8)
        b[0::4] = (y.real < 0)
        b[1::4] = (np.abs(y.real) < _T16)
        b[2::4] = (y.imag < 0)
        b[3::4] = (np.abs(y.imag) < _T16)
        return b
    raise ValueError(mod)


def _bps(mod):
    return {'BPSK': 1, 'QPSK': 2, '16QAM': 4}[mod]


def matched_filter(r, beta, tau, cfo, phase, sps, n_sym, span):
    n = np.arange(len(r))
    r = np.asarray(r, dtype=np.complex128) * np.exp(-1j * (2 * np.pi * cfo * n + phase))
    w = span * sps // 2
    m = np.arange(-w, w + 1)
    g = rrc((m - tau) / sps, beta)
    rp = np.concatenate([np.zeros(w), r, np.zeros(w)])
    idx = np.arange(n_sym)[:, None] * sps + m[None, :] + w
    return (rp[idx] * g[None, :]).sum(1)


def batch_ber(sig, batch, meta, mod_names):
    """sig: (B, 2, L) 原始尺度的待解调信号（torch 或 numpy）；batch 为 MoEDiffDataset 的 batch。返回 (B,) 的 BER。

    忽略首尾 EDGE_SKIP 个符号（滤波器边缘效应）。
    """
    sig = np.asarray(sig.detach().cpu() if hasattr(sig, 'detach') else sig, dtype=np.float64)
    demod = np.asarray(batch['demod'].cpu() if hasattr(batch['demod'], 'cpu') else batch['demod'], dtype=np.float64)
    sym = np.asarray(batch['symbols'].cpu() if hasattr(batch['symbols'], 'cpu') else batch['symbols'], dtype=np.float64)
    modi = np.asarray(batch['modulation'].cpu() if hasattr(batch['modulation'], 'cpu') else batch['modulation'])
    sps, n_sym, span, skip = int(meta['SPS']), int(meta['N_SYM']), int(meta['RRC_SPAN']), int(meta['EDGE_SKIP'])
    out = np.empty(sig.shape[0])
    for b in range(sig.shape[0]):
        mod = mod_names[int(modi[b])]
        rolloff, tau, cfo, phase, gain = demod[b]
        r = sig[b, 0] + 1j * sig[b, 1]
        y = matched_filter(r, rolloff, tau, cfo, phase, sps, n_sym, span) / gain
        hat = symbols_to_bits(y, mod)
        ref = symbols_to_bits(sym[b, 0] + 1j * sym[b, 1], mod)
        k = _bps(mod)
        out[b] = np.mean(hat[k * skip:k * (n_sym - skip)] != ref[k * skip:k * (n_sym - skip)])
    return out


def batch_evm(sig, batch, meta, mod_names):
    """符号域 EVM（匹配滤波+同步参数已知后，估计符号与理想符号的均方根误差，相对理想符号 RMS）。返回 (B,)。"""
    sig = np.asarray(sig.detach().cpu() if hasattr(sig, 'detach') else sig, dtype=np.float64)
    demod = np.asarray(batch['demod'].cpu() if hasattr(batch['demod'], 'cpu') else batch['demod'], dtype=np.float64)
    sym = np.asarray(batch['symbols'].cpu() if hasattr(batch['symbols'], 'cpu') else batch['symbols'], dtype=np.float64)
    sps, n_sym, span, skip = int(meta['SPS']), int(meta['N_SYM']), int(meta['RRC_SPAN']), int(meta['EDGE_SKIP'])
    out = np.empty(sig.shape[0])
    sl = slice(skip, n_sym - skip)
    for b in range(sig.shape[0]):
        rolloff, tau, cfo, phase, gain = demod[b]
        r = sig[b, 0] + 1j * sig[b, 1]
        y = matched_filter(r, rolloff, tau, cfo, phase, sps, n_sym, span) / gain
        s = sym[b, 0] + 1j * sym[b, 1]
        out[b] = np.sqrt(np.mean(np.abs(y[sl] - s[sl]) ** 2) / np.mean(np.abs(s[sl]) ** 2))
    return out


def batch_inband_sinr(x_clean, x_est, batch, meta):
    """带内 SINR (dB)：10log10(P_clean_inband / P_error_inband)，带宽按各样本 roll-off。返回 (B,)。"""
    c = np.asarray(x_clean.detach().cpu() if hasattr(x_clean, 'detach') else x_clean, dtype=np.float64)
    e = np.asarray(x_est.detach().cpu() if hasattr(x_est, 'detach') else x_est, dtype=np.float64) - c
    roll = np.asarray(batch['demod'].cpu() if hasattr(batch['demod'], 'cpu') else batch['demod'], dtype=np.float64)[:, 0]
    L = c.shape[-1]
    f = np.abs(np.fft.fftfreq(L))
    out = np.empty(c.shape[0])
    for b in range(c.shape[0]):
        m = f <= (1 + roll[b]) / (2 * int(meta['SPS']))
        pc = (np.abs(np.fft.fft(c[b, 0] + 1j * c[b, 1])[m]) ** 2).sum()
        pe = (np.abs(np.fft.fft(e[b, 0] + 1j * e[b, 1])[m]) ** 2).sum()
        out[b] = 10 * np.log10(pc / (pe + 1e-20))
    return out
