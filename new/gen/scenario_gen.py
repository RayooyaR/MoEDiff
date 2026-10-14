"""候选"难场景"的干扰生成器：同信道数字干扰 CCI、复合干扰 MIX。

CCI  另一路数字调制信号叠加进来：调制随机(QPSK/BPSK/16QAM)、roll-off/定时/频偏/初相均独立，符号率与目标信号相同(4 点/符号)。
     频谱与目标信号几乎完全重合，线性滤波无法分离，需要利用星座与脉冲成形的结构做联合判决。
MIX  随机选 2~3 种现有干扰(ST/MT/PL/LFM/NB/WB/NFM/RFM)同时叠加，各成分功率比随机。
两者都返回单位功率的复信号，随后由 make_dataset 按带内 SIR 缩放。
"""
import numpy as np

import config as C
import interference_gen as IG
import signal_gen as SG


def gen_CCI(rng, L, fmax, regime):
    mod = str(rng.choice(C.MODS))
    cid = int(rng.integers(10 ** 8, 2 * 10 ** 9))            # 与目标信号的 clean_id 区段不重叠
    c = SG.make_clean(cid, mod)
    x = c['signal'].astype(np.complex128)
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), dict(mod=mod, rolloff=c['beta'], tau=c['tau'], cfo=c['cfo'])


_BASE = ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM']


def gen_MIX(rng, L, fmax, regime):
    k = int(rng.integers(2, 4))
    names = [str(n) for n in rng.choice(_BASE, size=k, replace=False)]
    w = rng.dirichlet(np.ones(k))
    x = np.zeros(L, dtype=np.complex128)
    for n, wi in zip(names, w):
        xi, _ = IG.GENERATORS[n](rng, L, fmax, 'A')
        x += np.sqrt(wi) * xi
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), dict(components=names, weights=[float(v) for v in w])


def register():
    IG.GENERATORS['CCI'] = gen_CCI
    IG.GENERATORS['MIX'] = gen_MIX
    IG.GENERATORS['CCIR'] = gen_CCIR


def gen_CCIR(rng, L, fmax, regime):
    """异速率同信道数字干扰：符号周期 T_i = SPS·u（u∈[0.6,1.7]，排除 [0.9,1.1]），其余(调制/roll-off/定时/频偏/初相)随机。

    目标信号的符号周期固定为 SPS 个采样点，干扰的周期性与之不同，因此目标在原理上可被识别
    （同速率时两路信号统计性质完全相同，信号功率相近时无法区分谁是目标）。
    """
    mod = str(rng.choice(C.MODS))
    while True:
        u = float(rng.uniform(0.6, 1.7))
        if not (0.9 <= u <= 1.1):
            break
    T = C.SPS * u
    beta = float(rng.uniform(*C.ROLLOFF))
    tau = float(rng.uniform(0, T))
    cfo = float(rng.uniform(-C.CFO_MAX, C.CFO_MAX))
    phase = float(rng.uniform(0, 2 * np.pi))
    n_sym = int(np.ceil((L + 8 * T) / T)) + 2
    bps = C.BITS_PER_SYM[mod]
    bits = rng.integers(0, 2, size=bps * n_sym).astype(np.uint8)
    sym = SG.bits_to_symbols(bits, mod)
    half = 4.0 * T                                          # RRC 跨度 8 个符号
    x = np.zeros(L, dtype=np.complex128)
    for k in range(n_sym):
        c = k * T + tau - 2 * T                             # 让信号从 −2T 开始，避免起始边缘效应
        lo, hi = max(0, int(np.ceil(c - half))), min(L - 1, int(np.floor(c + half)))
        if hi < lo:
            continue
        n = np.arange(lo, hi + 1)
        x[lo:hi + 1] += sym[k] * SG.rrc((n - c) / T, beta)
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    n = np.arange(L)
    x = x * np.exp(1j * (2 * np.pi * cfo * n + phase))
    return x, dict(mod=mod, rate_ratio=1.0 / u, rolloff=beta)


# ---------------------------------------------------------------------------------------------
# 同信道数字干扰家族的其他子类（FSK / OFDM / BURST / ACI）
# ---------------------------------------------------------------------------------------------
def _smooth(x, sigma):
    k = int(np.ceil(4 * sigma)) * 2 + 1
    t = np.arange(k) - k // 2
    g = np.exp(-0.5 * (t / sigma) ** 2)
    return np.convolve(x, g / g.sum(), mode='same')


def gen_FSK(rng, L, fmax, regime):
    """恒包络数字调制：2-FSK / 4-FSK，相位连续，频率脉冲经高斯平滑。符号率与目标不同。"""
    M = int(rng.choice([2, 4]))
    h = float(rng.uniform(0.5, 1.0))
    while True:
        u = float(rng.uniform(0.6, 1.7))
        if not (0.9 <= u <= 1.1):
            break
    T = C.SPS * u
    fc = float(rng.uniform(-0.5, 0.5) * fmax)
    n = np.arange(L)
    off = rng.uniform(0, T)
    nsym = int(np.ceil((L + T) / T)) + 2
    a = rng.choice(np.arange(-(M - 1), M, 2), size=nsym) / float(M - 1)       # 取值 ∈ [-1, 1]
    f_sym = a[np.minimum(((n + off) // T).astype(int), nsym - 1)]
    f_inst = fc + (h / (2.0 * T)) * _smooth(f_sym, 0.35 * T)
    x = np.exp(1j * (2 * np.pi * np.cumsum(f_inst) + rng.uniform(0, 2 * np.pi)))
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), dict(M=M, h=h, rate_ratio=1.0 / u, fc=fc)


def gen_OFDM(rng, L, fmax, regime):
    """多载波：nfft∈{32,64,128}，循环前缀 1/4，占用带宽 (0.6~1.3)×信号带宽，各子载波 QPSK。"""
    nfft = int(rng.choice([32, 64, 128]))
    ncp = nfft // 4
    bw = float(rng.uniform(0.6, 1.3) * 2 * fmax)
    nsc = int(np.clip(round(bw * nfft), 4, nfft - 2))
    fc = float(rng.uniform(-0.3, 0.3) * fmax)
    nsym = int(np.ceil(L / (nfft + ncp))) + 3
    idx = (np.arange(nsc) - nsc // 2) % nfft
    sig = []
    for _ in range(nsym):
        spec = np.zeros(nfft, dtype=np.complex128)
        spec[idx] = (rng.choice([-1.0, 1.0], nsc) + 1j * rng.choice([-1.0, 1.0], nsc)) / np.sqrt(2)
        t = np.fft.ifft(spec) * np.sqrt(nfft)
        sig.append(np.concatenate([t[-ncp:], t]))
    x = np.concatenate(sig)
    start = int(rng.integers(0, nfft + ncp))
    x = x[start:start + L]
    n = np.arange(L)
    x = x * np.exp(1j * (2 * np.pi * fc * n + rng.uniform(0, 2 * np.pi)))
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), dict(nfft=nfft, nsc=nsc, bw=bw, fc=fc)


def gen_BURST(rng, L, fmax, regime):
    """突发数字干扰：异速率单载波信号被随机的数据包门控（占空比 20%~70%，各包幅度不同）。"""
    x, p = gen_CCIR(rng, L, fmax, regime)
    mask = np.zeros(L)
    target_duty = rng.uniform(0.2, 0.7)
    while mask.mean() < target_duty:
        ln = int(rng.integers(200, 900))
        s = int(rng.integers(0, max(1, L - ln)))
        mask[s:s + ln] = rng.uniform(0.5, 1.5)
    mask = _smooth(mask, 8.0)
    x = x * mask
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), dict(duty=float((mask > 0.05).mean()), **{k: p[k] for k in ('mod', 'rate_ratio')})


def gen_ACI(rng, L, fmax, regime):
    """相邻信道干扰：异速率单载波信号整体平移 ±(0.25~0.7)×信号带宽，只有一部分落入目标信号带内。"""
    x, p = gen_CCIR(rng, L, fmax, regime)
    shift = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.7) * 2 * fmax)
    x = x * np.exp(1j * 2 * np.pi * shift * np.arange(L))
    return x, dict(shift=shift, **{k: p[k] for k in ('mod', 'rate_ratio')})


def register_digital():
    register()
    IG.GENERATORS.update(FSK=gen_FSK, OFDM=gen_OFDM, BURST=gen_BURST, ACI=gen_ACI)


# ---------------------------------------------------------------------------------------------
# 干扰上的随机信号影响（更难场景 v5）：7 种效应，每种独立以 P_EFFECT 概率出现，可同时出现多种，也可能一种都没有
# 衰落类：multipath, fading；频率类：cfo, drift, phase_noise；硬件类：iq_imbalance, nonlinearity
# ---------------------------------------------------------------------------------------------
P_EFFECT = 0.5
EFFECTS = ['nonlinearity', 'iq_imbalance', 'phase_noise', 'cfo', 'drift', 'multipath', 'fading']   # 按发射机→信道的物理顺序依次施加


def _norm(x):
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def _apply_effect(name, x, rng, L, fmax):
    n = np.arange(L)
    if name == 'nonlinearity':                                  # Rapp 软限幅/压缩
        A = rng.uniform(0.7, 1.5) * np.sqrt(np.mean(np.abs(x) ** 2))
        p = int(rng.choice([1, 2, 3]))
        return x / (1 + (np.abs(x) / A) ** (2 * p)) ** (1.0 / (2 * p)), dict(A=float(A), p=p)
    if name == 'iq_imbalance':                                  # y = μx + ν·conj(x)
        g = 1 + rng.choice([-1, 1]) * rng.uniform(0.05, 0.3)
        psi = np.deg2rad(rng.uniform(3, 20))
        mu, nu = (1 + g * np.exp(-1j * psi)) / 2, (1 - g * np.exp(1j * psi)) / 2
        return mu * x + nu * np.conj(x), dict(g=float(g), psi_deg=float(np.rad2deg(psi)))
    if name == 'phase_noise':                                   # 维纳相位噪声
        s = rng.uniform(0.01, 0.06)
        return x * np.exp(1j * np.cumsum(rng.normal(0, s, L))), dict(sigma=float(s))
    if name == 'cfo':                                           # 大频偏
        d = rng.choice([-1, 1]) * rng.uniform(0.1, 0.5) * fmax
        return x * np.exp(1j * 2 * np.pi * d * n), dict(delta=float(d))
    if name == 'drift':                                         # 线性频率漂移（多普勒）
        D = rng.choice([-1, 1]) * rng.uniform(0.1, 0.4) * fmax
        return x * np.exp(1j * 2 * np.pi * (D / (2.0 * L)) * n ** 2), dict(total_drift=float(D))
    if name == 'multipath':                                     # 2~4 径，功率指数衰减，复高斯增益
        k = int(rng.integers(2, 5))
        delays = np.concatenate([[0], np.sort(rng.integers(1, 17, size=k - 1))])
        gains = (rng.standard_normal(k) + 1j * rng.standard_normal(k)) / np.sqrt(2) * np.exp(-np.arange(k) * rng.uniform(0.2, 0.7))
        h = np.zeros(delays.max() + 1, dtype=np.complex128)
        np.add.at(h, delays, gains)
        return np.convolve(x, h)[:L], dict(paths=k, max_delay=int(delays.max()))
    if name == 'fading':                                        # 慢时变瑞利衰落（含相位变化）
        fd = rng.uniform(0.0005, 0.004)
        f = np.fft.fftfreq(L)
        g = np.fft.ifft(np.fft.fft(rng.standard_normal(L) + 1j * rng.standard_normal(L)) * np.exp(-0.5 * (f / fd) ** 2))
        return x * (g / np.sqrt(np.mean(np.abs(g) ** 2))), dict(fd=float(fd))
    raise ValueError(name)


def apply_random_effects(rng, x, L, fmax):
    on = [e for e in EFFECTS if rng.random() < P_EFFECT]
    info = {}
    for e in EFFECTS:                                            # 保持固定的施加顺序
        if e in on:
            x, p = _apply_effect(e, x, rng, L, fmax)
            info[e] = p
    return _norm(x), info


def register_effects():
    """把 8 类基础干扰的生成器包一层：先生成基础干扰，再叠加随机效应。类型名保持不变。"""
    base = dict(IG.GENERATORS)

    def wrap(fn):
        def g(rng, L, fmax, regime):
            x, params = fn(rng, L, fmax, regime)
            x, eff = apply_random_effects(rng, x, L, fmax)
            params = dict(params)
            params['effects'] = eff
            params['n_effects'] = len(eff)
            return x, params
        return g
    for name in ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM']:
        IG.GENERATORS[name] = wrap(base[name])
