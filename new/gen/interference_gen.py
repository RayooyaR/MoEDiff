"""八类带内干扰（复基带）。每个生成函数返回 (单位功率复信号, 参数字典)。

所有频率参数相对信号带宽半宽 fmax 取值，保证干扰落在信号频带内/附近。
regime 'A' 为训练/验证/常规测试区间，'B' 为留出区间（只用于 test_ood）。
"""
import numpy as np

import config as C


def _unit(x):
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def _white(rng, L):
    return rng.standard_normal(L) + 1j * rng.standard_normal(L)


def gen_ST(rng, L, fmax, regime):
    # 训练区间取中心附近，留出区间取偏离中心但仍在平坦通带内（<=0.6 fmax），保证难度相当
    if regime == 'A':
        f = rng.uniform(-0.3, 0.3) * fmax
    else:
        f = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.6) * fmax
    ph = rng.uniform(0, 2 * np.pi)
    x = np.exp(1j * (2 * np.pi * f * np.arange(L) + ph))
    return _unit(x), dict(f=float(f))


def gen_MT(rng, L, fmax, regime):
    if regime == 'A':
        k, span = int(rng.integers(2, 4)), 0.6
    else:
        k, span = int(rng.integers(4, 7)), 0.95
    fs = rng.uniform(-span, span, size=k) * fmax
    amp = rng.uniform(0.5, 1.5, size=k)
    ph = rng.uniform(0, 2 * np.pi, size=k)
    n = np.arange(L)
    x = sum(a * np.exp(1j * (2 * np.pi * f * n + p)) for a, f, p in zip(amp, fs, ph))
    return _unit(x), dict(k=k, f=[float(v) for v in fs], amp=[float(v) for v in amp])


def gen_PL(rng, L, fmax, regime):
    if regime == 'A':
        npulse, width = int(rng.integers(2, 7)), rng.uniform(40, 160)
    else:
        npulse, width = int(rng.integers(13, 20)), rng.uniform(12, 40)
    fc = rng.uniform(-0.6, 0.6) * fmax
    sigma = width / 5.0
    n = np.arange(L)
    centers = rng.uniform(width, L - width, size=npulse)
    x = np.zeros(L, dtype=np.complex128)
    for c in centers:
        env = np.exp(-0.5 * ((n - c) / sigma) ** 2)
        x += env * np.exp(1j * (2 * np.pi * fc * n + rng.uniform(0, 2 * np.pi)))
    return _unit(x), dict(npulse=npulse, width=float(width), fc=float(fc))


def gen_LFM(rng, L, fmax, regime):
    # 锯齿扫频：f 从 f0 线性扫到 f1，周期 T 个采样点后重复
    while True:
        f0, f1 = rng.uniform(-0.95, 0.95, size=2) * fmax
        if abs(f1 - f0) >= 0.5 * fmax:
            break
    T = rng.uniform(640, 2560) if regime == 'A' else rng.uniform(160, 500)
    off = rng.uniform(0, T)
    n = np.arange(L)
    f_inst = f0 + (f1 - f0) * (((n + off) % T) / T)
    x = np.exp(1j * 2 * np.pi * np.cumsum(f_inst))
    return _unit(x), dict(f0=float(f0), f1=float(f1), T=float(T))


def gen_NB(rng, L, fmax, regime):
    bw = (rng.uniform(0.05, 0.15) if regime == 'A' else rng.uniform(0.2, 0.35)) * 2 * fmax
    fc = rng.uniform(-0.6, 0.6) * fmax
    f = np.fft.fftfreq(L)
    H = np.exp(-0.5 * ((f - fc) / (bw / 2)) ** 2)
    x = np.fft.ifft(np.fft.fft(_white(rng, L)) * H)
    return _unit(x), dict(bw=float(bw), fc=float(fc))


def gen_WB(rng, L, fmax, regime):
    bw = (rng.uniform(0.8, 1.3) if regime == 'A' else rng.uniform(0.35, 0.65)) * 2 * fmax
    fc = rng.uniform(-0.1, 0.1) * fmax
    f = np.fft.fftfreq(L)
    H = 1.0 / (1.0 + np.exp((np.abs(f - fc) - bw / 2) / 0.005))
    x = np.fft.ifft(np.fft.fft(_white(rng, L)) * H)
    return _unit(x), dict(bw=float(bw), fc=float(fc))


def gen_NFM(rng, L, fmax, regime):
    # 窄带调频：phase = 2*pi*fc*n + beta*sin(2*pi*fm*n)，频偏 df = beta*fm
    fc = rng.uniform(-0.5, 0.5) * fmax
    if regime == 'A':
        fm, beta = rng.uniform(0.03, 0.08) * fmax, rng.uniform(0.5, 2.0)
    else:
        fm, beta = rng.uniform(0.1, 0.2) * fmax, rng.uniform(0.5, 1.5)
    n = np.arange(L)
    x = np.exp(1j * (2 * np.pi * fc * n + beta * np.sin(2 * np.pi * fm * n + rng.uniform(0, 2 * np.pi))))
    return _unit(x), dict(fc=float(fc), fm=float(fm), beta=float(beta))


def gen_RFM(rng, L, fmax, regime):
    # 随机调频：瞬时频率 = fc + 低通高斯噪声（标准差 sig_f，截止 cut）
    fc = rng.uniform(-0.4, 0.4) * fmax
    if regime == 'A':
        sig_f, cut = rng.uniform(0.15, 0.3) * fmax, rng.uniform(0.02, 0.05) * fmax
    else:
        sig_f, cut = rng.uniform(0.3, 0.45) * fmax, rng.uniform(0.06, 0.12) * fmax
    f = np.fft.fftfreq(L)
    w = np.fft.ifft(np.fft.fft(rng.standard_normal(L)) * np.exp(-0.5 * (f / cut) ** 2)).real
    w = w / np.std(w) * sig_f
    x = np.exp(1j * 2 * np.pi * np.cumsum(fc + w))
    return _unit(x), dict(fc=float(fc), sig_f=float(sig_f), cut=float(cut))


GENERATORS = dict(ST=gen_ST, MT=gen_MT, PL=gen_PL, LFM=gen_LFM, NB=gen_NB, WB=gen_WB, NFM=gen_NFM, RFM=gen_RFM)


def inband_mask(L, fmax):
    return np.abs(np.fft.fftfreq(L)) <= fmax


def inband_power(x, mask):
    """带内功率（Parseval：全带时等于 mean|x|^2）。"""
    X = np.fft.fft(x)
    return float((np.abs(X[mask]) ** 2).sum() / len(x) ** 2)


def generate(itype, rng, L, fmax, regime):
    """生成干扰；带内能量占比不足则重采样参数。返回 (j, params, inband_frac)。"""
    mask = inband_mask(L, fmax)
    for _ in range(50):
        j, params = GENERATORS[itype](rng, L, fmax, regime)
        frac = inband_power(j, mask) / float(np.mean(np.abs(j) ** 2))
        if frac >= C.MIN_INBAND_FRAC:
            return j, params, frac
    raise RuntimeError('cannot sample %s with in-band fraction >= %.2f' % (itype, C.MIN_INBAND_FRAC))
