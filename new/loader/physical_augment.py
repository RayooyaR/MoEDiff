"""实验 14：物理合理的数据增强与干扰重新混合（仅用于训练集）。

约定：一条样本是 dict，含 clean/interf/noisy 三个 (2, L) float32 数组（noisy = clean + interf + 残余噪声）、
demod(5,)=[rolloff, timing_offset, cfo, phase, gain]、sps（过采样率）、sir_db、snr_db（带内）。
所有算子返回新的 dict，并**重新由数组计算带内 SIR/SNR**（增强后的有效值），以便报告增强后的 SIR/SNR 分布。

算子（ops 用逗号分隔的名字选择，按顺序作用）：
  rot    共同相位旋转：y,x,i 同乘 e^{jφ}，demod 的 phase += φ                       （目标与输入同变换，标签一致）
  shift  合法时移：循环移位 s（取 sps 的整数倍，|s|<=4*sps），demod 的 phase -= 2π·cfo·s（符号序列随之滚动，不用于 BER 评估）
  awgn   输入侧附加噪声：y += n，带内附加 SNR ~ U(20, 35) dB（相对 clean 带内功率），目标 clean 不变（目标定义为不含噪声）
  iqgain 接收机 I/Q 增益失配：y 的 I×(1+e)、Q×(1−e)，e~U(-0.05,0.05)，**仅输入侧**，目标不变
  remix  独立干扰重新混合：y' = clean + g·interf_j + (原残余噪声)，interf_j 取自**训练池**中另一条（不同 clean_id）的干扰，
         g 使带内 SIR 等于从 SIR 档位集合随机抽取的目标值；干扰来源仅限训练允许的类型（池由调用方给定）
"""
import numpy as np

OPS = ('rot', 'shift', 'awgn', 'iqgain', 'remix', 'remixys')
SIR_GRID = (-20, -15, -10, -5, 0, 5, 10)


def to_c(a):
    return a[0].astype(np.float64) + 1j * a[1].astype(np.float64)


def to_iq(c):
    return np.stack([c.real, c.imag], 0).astype(np.float32)


def fmax_of(rolloff, sps):
    return (1.0 + float(rolloff)) / (2.0 * sps)


def inband_power(c, fmax):
    L = len(c)
    mask = np.abs(np.fft.fftfreq(L)) <= fmax
    X = np.fft.fft(c)
    return float((np.abs(X[mask]) ** 2).sum() / L ** 2) + 1e-30


def actual_sir_snr(clean, interf, noisy, rolloff, sps):
    """由数组重新计算带内 SIR/SNR（dB）。noise = noisy - clean - interf。"""
    fm = fmax_of(rolloff, sps)
    c, i = to_c(clean), to_c(interf)
    n = to_c(noisy) - c - i
    ps, pi, pn = inband_power(c, fm), inband_power(i, fm), inband_power(n, fm)
    return 10 * np.log10(ps / pi), 10 * np.log10(ps / pn)


def _finish(s, clean, interf, noisy, demod):
    out = dict(s)
    out.update(clean=clean.astype(np.float32), interf=interf.astype(np.float32), noisy=noisy.astype(np.float32), demod=demod.astype(np.float32))
    sir, snr = actual_sir_snr(out['clean'], out['interf'], out['noisy'], demod[0], s['sps'])
    out['sir_db'], out['snr_db'] = float(sir), float(snr)
    return out


def op_rot(s, rng):
    phi = rng.uniform(0, 2 * np.pi)
    r = np.exp(1j * phi)
    d = s['demod'].copy(); d[3] = (d[3] + phi) % (2 * np.pi)
    return _finish(s, to_iq(to_c(s['clean']) * r), to_iq(to_c(s['interf']) * r), to_iq(to_c(s['noisy']) * r), d)


def op_shift(s, rng):
    sps = int(s['sps'])
    k = int(rng.integers(-4, 5)) * sps
    if k == 0:
        return s
    roll = lambda a: np.roll(a, k, axis=-1)
    d = s['demod'].copy(); d[3] = (d[3] - 2 * np.pi * d[2] * k) % (2 * np.pi)
    return _finish(s, roll(s['clean']), roll(s['interf']), roll(s['noisy']), d)


def op_awgn(s, rng):
    fm = fmax_of(s['demod'][0], s['sps'])
    L = s['clean'].shape[-1]
    snr = rng.uniform(20.0, 35.0)
    ps = inband_power(to_c(s['clean']), fm)
    # 白噪声的带内功率 = σ² × (带内频点占比)；按带内 SNR 定标
    frac = float(np.mean(np.abs(np.fft.fftfreq(L)) <= fm))
    sigma2 = ps / 10 ** (snr / 10.0) / frac
    n = np.sqrt(sigma2 / 2) * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
    return _finish(s, s['clean'], s['interf'], to_iq(to_c(s['noisy']) + n), s['demod'])


def op_iqgain(s, rng):
    e = rng.uniform(-0.05, 0.05)
    y = s['noisy'].copy(); y[0] *= (1 + e); y[1] *= (1 - e)
    return _finish(s, s['clean'], s['interf'], y, s['demod'])


def op_remix(s, rng, pool):
    """pool: 训练池样本列表（dict，含 interf 与 clean_id）。"""
    for _ in range(20):
        j = pool[int(rng.integers(len(pool)))]
        if j['clean_id'] != s['clean_id']:
            break
    else:
        return s
    fm = fmax_of(s['demod'][0], s['sps'])
    target = float(rng.choice(SIR_GRID))
    ps = inband_power(to_c(s['clean']), fm)
    pj = inband_power(to_c(j['interf']), fm)
    g = np.sqrt(ps / (pj * 10 ** (target / 10.0)))
    noise = to_c(s['noisy']) - to_c(s['clean']) - to_c(s['interf'])               # 保留原残余噪声
    ni = to_iq(g * to_c(j['interf']))
    out = _finish(s, s['clean'], ni, to_iq(to_c(s['clean']) + g * to_c(j['interf']) + noise), s['demod'])
    out['interference_type'] = j['interference_type']                              # 干扰形态来自 j
    out['attrs'] = None                                                            # 属性需按新干扰重算
    return out


def op_remixys(s, rng, pool):
    """仅依赖配对样本 (y, s) 的 remix（不使用模拟器内部信息，用来检验 remix 的收益是否依赖"干扰/噪声已分开存放"这一条件）：
    把另一条训练样本的 y_j - s_j（干扰+噪声一起）当作"干扰"，缩放到随机目标 SIR 后加到本条干净信号上；不再另加噪声；
    带内 SIR 用固定带宽 fmax=(1+0.5)/(2*sps)（最大滚降）而不是该样本的真实滚降系数。"""
    for _ in range(20):
        j = pool[int(rng.integers(len(pool)))]
        if j['clean_id'] != s['clean_id']:
            break
    else:
        return s
    fm = fmax_of(0.5, s['sps'])
    target = float(rng.choice(SIR_GRID))
    ps = inband_power(to_c(s['clean']), fm)
    res_j = to_c(j['noisy']) - to_c(j['clean'])
    pj = inband_power(res_j, fm)
    g = np.sqrt(ps / (pj * 10 ** (target / 10.0)))
    ni = to_iq(g * res_j)
    out = _finish(s, s['clean'], ni, to_iq(to_c(s['clean']) + g * res_j), s['demod'])
    out['interference_type'] = j['interference_type']
    out['attrs'] = None
    return out


def apply_ops(ops, s, rng, pool=None):
    for op in ops:
        if op == 'rot':
            s = op_rot(s, rng)
        elif op == 'shift':
            s = op_shift(s, rng)
        elif op == 'awgn':
            s = op_awgn(s, rng)
        elif op == 'iqgain':
            s = op_iqgain(s, rng)
        elif op == 'remix':
            s = op_remix(s, rng, pool)
        elif op == 'remixys':
            s = op_remixys(s, rng, pool)
        else:
            raise ValueError('unknown op %s; choose from %s' % (op, OPS))
    return s


def parse_ops(spec):
    """'rot,remix' -> ('rot','remix')；'none' 或空 -> ()"""
    if not spec or spec == 'none':
        return ()
    ops = tuple(x for x in spec.split(',') if x)
    for o in ops:
        if o not in OPS:
            raise ValueError('unknown op %s' % o)
    return ops
