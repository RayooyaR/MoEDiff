"""BPSK/QPSK/16QAM + RRC 成形 + 简单信道（频偏/初相/定时偏移），以及带已知同步参数的匹配滤波解调。

映射（均 Gray，平均符号功率 1）：
  BPSK : b -> 1-2b
  QPSK : (b0,b1) -> ((1-2*b0) + j(1-2*b1)) / sqrt(2)
  16QAM: I 取 (b0,b1)，Q 取 (b2,b3)；每轴 00:+3, 01:+1, 11:-1, 10:-3，再除以 sqrt(10)
"""
import numpy as np

import config as C

_T16 = 2.0 / np.sqrt(10.0)   # 16QAM 判决门限（归一化后）


def rrc(t, beta):
    """RRC 脉冲，t 以符号周期为单位。"""
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


def _lvl(b0, b1):
    return np.where(b0 == 0, np.where(b1 == 0, 3.0, 1.0), np.where(b1 == 1, -1.0, -3.0))


def bits_to_symbols(bits, mod):
    if mod == 'BPSK':
        return (1.0 - 2.0 * bits.astype(np.float64)) + 0j
    if mod == 'QPSK':
        b = bits.reshape(-1, 2).astype(np.float64)
        return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)
    if mod == '16QAM':
        b = bits.reshape(-1, 4)
        return (_lvl(b[:, 0], b[:, 1]) + 1j * _lvl(b[:, 2], b[:, 3])) / np.sqrt(10)
    raise ValueError(mod)


def symbols_to_bits(y, mod):
    """y 为已按增益归一化的符号估计，返回硬判决 bit 序列。"""
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


def shape(symbols, beta, tau):
    """符号 k 的脉冲峰值位于 n = k*SPS + tau（tau 为分数采样点定时偏移）。"""
    up = np.zeros(C.SIG_LEN, dtype=np.complex128)
    up[::C.SPS] = symbols
    w = C.RRC_SPAN * C.SPS // 2
    m = np.arange(-w, w + 1)
    h = rrc((m - tau) / C.SPS, beta)
    return np.convolve(up, h, mode='same')


def matched_filter(r, beta, tau, cfo, phase):
    """已知 频偏/初相/定时 的匹配滤波，返回每个符号的未归一化估计 y (N_SYM,)。"""
    n = np.arange(len(r))
    r = np.asarray(r, dtype=np.complex128) * np.exp(-1j * (2 * np.pi * cfo * n + phase))
    w = C.RRC_SPAN * C.SPS // 2
    m = np.arange(-w, w + 1)
    g = rrc((m - tau) / C.SPS, beta)
    rp = np.concatenate([np.zeros(w), r, np.zeros(w)])
    idx = np.arange(C.N_SYM)[:, None] * C.SPS + m[None, :] + w
    return (rp[idx] * g[None, :]).sum(1)


def make_clean(clean_id, mod):
    """按 (clean_id, mod) 确定性生成一条干净信号（含信道效应，不含干扰和噪声）。

    demod_gain：匹配滤波输出相对符号的增益（由干净信号最小二乘得到），16QAM 判决需要它。
    """
    rng = np.random.default_rng(np.random.SeedSequence([C.SEED, 7, int(clean_id)]))
    beta = float(rng.uniform(*C.ROLLOFF))
    tau = float(rng.uniform(0.0, 1.0))
    cfo = float(rng.uniform(-C.CFO_MAX, C.CFO_MAX))
    phase = float(rng.uniform(0.0, 2 * np.pi))
    bits = rng.integers(0, 2, size=C.BITS_PER_SYM[mod] * C.N_SYM).astype(np.uint8)
    sym = bits_to_symbols(bits, mod)
    x = shape(sym, beta, tau)
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    n = np.arange(C.SIG_LEN)
    x = x * np.exp(1j * (2 * np.pi * cfo * n + phase))
    y = matched_filter(x, beta, tau, cfo, phase)
    k = slice(C.EDGE_SKIP, C.N_SYM - C.EDGE_SKIP)
    gain = float(np.real(np.sum(np.conj(sym[k]) * y[k])) / np.sum(np.abs(sym[k]) ** 2))
    return dict(signal=x.astype(np.complex64), bits=bits, symbols=sym.astype(np.complex64), beta=beta, tau=tau,
                cfo=cfo, phase=phase, gain=gain)


def demod_oracle(r, mod, beta, tau, cfo, phase, gain):
    """已知同步参数与增益的匹配滤波硬判决，返回 bit 序列。

    用途：隔离"干扰+噪声"对解调的影响，不含同步误差。
    """
    return symbols_to_bits(matched_filter(r, beta, tau, cfo, phase) / gain, mod)


def ber_interior(bits_hat, bits_ref, mod):
    """忽略首尾 EDGE_SKIP 个符号后的 BER。"""
    bps = C.BITS_PER_SYM[mod]
    a, b = bps * C.EDGE_SKIP, bps * (C.N_SYM - C.EDGE_SKIP)
    return float(np.mean(bits_hat[a:b] != bits_ref[a:b]))
