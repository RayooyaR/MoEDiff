"""物理增强单元测试：标签一致性 / 能量守恒 / 重新混合的 SIR 精度。用法: python tests/test_augmentation_physics.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..', 'loader'))
import numpy as np
from moe_dataset import MoEDiffDataset
from physical_augment import apply_ops, to_c, SIR_GRID
from metrics_new import matched_filter

ROOT = os.path.join(HERE, '..', '..', 'data', 'v2')
ds = MoEDiffDataset(ROOT, 'train', types=['ST', 'LFM', 'PL', 'NB'], mods=['QPSK'], max_per_cell=2, noise_mode='with')
meta = ds.meta; SPS, NSYM, SPAN = ds.sps, ds.n_sym, int(meta['RRC_SPAN'])
rng = np.random.default_rng(0)

def base(i):
    s = ds.samples[i]
    return dict(clean=s['clean'], interf=s['interf'], noisy=s['noisy'], demod=s['demod'], sps=SPS, sir_db=s['sir_db'], snr_db=s['snr_db'],
                clean_id=s['clean_id'], interference_type=s['interference_type'], attrs=None)

def mf(c, d):
    return matched_filter(to_c(c), d[0], d[1], d[2], d[3], SPS, NSYM, SPAN)

n_ok = 0
for i in range(0, len(ds.samples), 7):
    s = base(i); ref = mf(s['clean'], s['demod'])
    # rot：共同旋转后用更新的 phase 匹配滤波，输出必须与旋转前相同
    t = apply_ops(['rot'], s, rng)
    assert np.allclose(mf(t['clean'], t['demod']), ref, atol=1e-3), 'rot 后匹配滤波输出应不变'
    assert abs(np.sum(t['clean'] ** 2) - np.sum(s['clean'] ** 2)) / np.sum(s['clean'] ** 2) < 1e-5, 'rot 应保持能量'
    assert abs(t['sir_db'] - s['sir_db']) < 0.05, 'rot 不应改变带内 SIR'
    # shift：匹配滤波输出 = 原输出按符号滚动（只比较远离回绕边缘的符号）
    t = apply_ops(['shift'], s, rng)
    out_t = mf(t['clean'], t['demod'])
    ok = any(np.allclose(out_t[8 + 4:NSYM - 8 - 4], np.roll(ref, sh // SPS)[8 + 4:NSYM - 8 - 4], atol=1e-2) for sh in range(-16, 17, SPS))
    assert ok, 'apply_ops(shift) 的输出应与某个合法符号滚动一致'
    for sh in range(-16, 17, SPS):
        if sh == 0: continue
        sc = np.roll(s['clean'], sh, axis=-1); d = s['demod'].copy(); d[3] = (d[3] - 2 * np.pi * d[2] * sh) % (2 * np.pi)
        out = mf(sc, d); exp = np.roll(ref, sh // SPS)
        sl = slice(8 + abs(sh) // SPS, NSYM - 8 - abs(sh) // SPS)
        assert np.allclose(out[sl], exp[sl], atol=1e-2), '合法时移 + 相位修正后匹配滤波输出应随符号滚动 (sh=%d)' % sh
    # awgn：输入侧加噪，目标不变，SNR 降低但有限
    t = apply_ops(['awgn'], s, rng)
    assert np.array_equal(t['clean'], s['clean']) and t['snr_db'] <= s['snr_db'] + 0.1 and t['snr_db'] > 5, 'awgn 只改输入、不改目标'
    # iqgain：目标/干扰不变
    t = apply_ops(['iqgain'], s, rng)
    assert np.array_equal(t['clean'], s['clean']) and np.array_equal(t['interf'], s['interf']), 'iqgain 只作用于输入'
    # remix：目标不变；新 SIR 落在档位集合内（容差 0.2dB）；残余噪声保持
    t = apply_ops(['remix'], s, rng, pool=ds.samples)
    assert np.array_equal(t['clean'], s['clean'])
    assert min(abs(t['sir_db'] - g) for g in SIR_GRID) < 0.3, 'remix 的有效 SIR 应等于目标档位 (got %.2f)' % t['sir_db']
    r0 = to_c(s['noisy']) - to_c(s['clean']) - to_c(s['interf']); r1 = to_c(t['noisy']) - to_c(t['clean']) - to_c(t['interf'])
    assert np.allclose(r0, r1, atol=1e-3), 'remix 应保留原残余噪声'
    n_ok += 1
# 数据集接口：aug_ops 不改变长度；同一 worker 种子下两次取样结果不同
ds2 = MoEDiffDataset(ROOT, 'train', types=['ST', 'LFM'], mods=['QPSK'], max_per_cell=2, aug_ops='rot,remix', return_aux=True)
a, b = ds2[0], ds2[0]
assert a['clean'].shape == (2, 2560) and a['attrs'].shape == (12,) and a['interf'].shape == (2, 2560)
print('PASS test_augmentation_physics (%d 条样本检查)' % n_ok)
