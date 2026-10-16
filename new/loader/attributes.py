"""由「干扰信号本身」计算的 12 维物理属性（实验 3 的辅助监督标签；仅训练期使用，推理不需要）。

标签来源（明确说明）：这些量由数据集保存的 interference_signal 直接计算，**不是**由混合信号估计，也**不是**类别标签的重新包装。
它们都是连续的、对所有干扰类型都有定义的物理量，因此可以在未见类型上做探针迁移检验。

分组（每个专家 3 个）：
  FD 频域  0 谱宽度(std of 归一化频率)   1 谱峰数量   2 频带集中度(最大 5% 频点占总功率比例)
  TF 时频  3 频率变化量(STFT 质心对时间的线性斜率×帧数)   4 质心起伏(std)   5 轨迹锐度(每帧最大频点功率占比的均值)
  TD 时域  6 占空比(功率超过峰值 10% 的采样点比例)   7 峰均比(dB)   8 功率时间质心(0~1)
  MS 多尺度 9 粗尺度能量占比(池化 16)   10 中尺度能量占比(池化 4 去掉池化 16)   11 尺度间包络相关(池化 4 与池化 16 的包络)
"""
import numpy as np

ATTR_NAMES = ['fd_width', 'fd_peaks', 'fd_conc', 'tf_slope', 'tf_cstd', 'tf_sharp',
              'td_duty', 'td_papr_db', 'td_tcentroid', 'ms_coarse', 'ms_mid', 'ms_corr']
ATTR_GROUPS = {'FD': [0, 1, 2], 'TF': [3, 4, 5], 'TD': [6, 7, 8], 'MS': [9, 10, 11]}
N_ATTR = len(ATTR_NAMES)
_EPS = 1e-12


def _pool(p, k):
    """沿最后一维做不重叠平均池化（长度截到 k 的整数倍）。"""
    n = (p.shape[-1] // k) * k
    return p[..., :n].reshape(*p.shape[:-1], n // k, k).mean(-1)


def compute_attrs(b):
    """b: (N, L) complex 或 (N, 2, L) real，返回 (N, 12) float32。"""
    b = np.asarray(b)
    if b.ndim == 3:
        b = b[:, 0] + 1j * b[:, 1]
    b = b.astype(np.complex128)
    n, L = b.shape
    out = np.zeros((n, N_ATTR), dtype=np.float64)
    p_t = np.abs(b) ** 2                                        # (N, L) 瞬时功率
    tot = p_t.sum(1) + _EPS

    # ---- FD ----
    P = np.abs(np.fft.fftshift(np.fft.fft(b, axis=1), axes=1)) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(L))
    pn = P / (P.sum(1, keepdims=True) + _EPS)
    mu = (pn * f).sum(1)
    out[:, 0] = np.sqrt((pn * (f - mu[:, None]) ** 2).sum(1))
    k = 8
    ps = np.apply_along_axis(lambda r: np.convolve(r, np.ones(k) / k, mode='same'), 1, pn)
    thr = 0.1 * ps.max(1, keepdims=True)
    loc = (ps[:, 1:-1] > ps[:, :-2]) & (ps[:, 1:-1] >= ps[:, 2:]) & (ps[:, 1:-1] > thr)
    out[:, 1] = np.minimum(loc.sum(1), 8)
    top = max(1, int(0.05 * L))
    out[:, 2] = np.sort(pn, axis=1)[:, -top:].sum(1)

    # ---- TF ----
    nfft, hop = 64, 32
    nfr = (L - nfft) // hop + 1
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]            # (T, nfft)
    fr = b[:, idx] * np.hanning(nfft)[None, None, :]                           # (N, T, nfft)
    S = np.abs(np.fft.fftshift(np.fft.fft(fr, axis=2), axes=2)) ** 2
    ff = np.fft.fftshift(np.fft.fftfreq(nfft))
    fp = S.sum(2) + _EPS
    cen = (S * ff[None, None, :]).sum(2) / fp                                  # (N, T) 帧质心
    tt = np.arange(nfr) - (nfr - 1) / 2.0
    w = fp / fp.sum(1, keepdims=True)                                          # 以帧功率加权
    cm = (w * cen).sum(1, keepdims=True)
    tm = (w * tt[None, :]).sum(1, keepdims=True)
    slope = (w * (cen - cm) * (tt[None, :] - tm)).sum(1) / ((w * (tt[None, :] - tm) ** 2).sum(1) + _EPS)
    out[:, 3] = slope * nfr
    out[:, 4] = np.sqrt((w * (cen - cm) ** 2).sum(1))
    out[:, 5] = (w * (S.max(2) / fp)).sum(1)

    # ---- TD ----
    out[:, 6] = (p_t > 0.1 * p_t.max(1, keepdims=True)).mean(1)
    out[:, 7] = 10 * np.log10(p_t.max(1) / (p_t.mean(1) + _EPS) + _EPS)
    out[:, 8] = (p_t * (np.arange(L)[None, :] / L)).sum(1) / tot

    # ---- MS ----
    p16, p4 = _pool(b.real, 16) + 1j * _pool(b.imag, 16), _pool(b.real, 4) + 1j * _pool(b.imag, 4)
    e16 = (np.abs(p16) ** 2).sum(1) * 16
    e4 = (np.abs(p4) ** 2).sum(1) * 4
    out[:, 9] = e16 / tot
    out[:, 10] = np.maximum(e4 - e16, 0) / tot
    env4, env16 = _pool(np.abs(b), 4), np.repeat(_pool(np.abs(b), 16), 4, axis=1)
    m = min(env4.shape[1], env16.shape[1])
    a_, c_ = env4[:, :m] - env4[:, :m].mean(1, keepdims=True), env16[:, :m] - env16[:, :m].mean(1, keepdims=True)
    out[:, 11] = (a_ * c_).sum(1) / (np.sqrt((a_ ** 2).sum(1) * (c_ ** 2).sum(1)) + _EPS)
    return out.astype(np.float32)
