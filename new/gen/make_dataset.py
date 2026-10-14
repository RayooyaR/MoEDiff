"""生成试点数据集：  python make_dataset.py [输出目录，默认 ../data/pilot]

样本关系（逐点精确成立）：
    mixture_signal = clean_signal + interference_signal + noise_signal
interference_signal 是已按带内 SIR 缩放后的干扰；noise_signal 是已缩放后的 AWGN。
无噪声版本 = clean_signal + interference_signal。
"""
import csv
import json
import os
import sys

import numpy as np

import config as C
import interference_gen as IG
import signal_gen as SG


def sir_tag(sir):
    return ('neg%02ddB' % -sir) if sir < 0 else ('pos%02ddB' % sir)


def build_sample(split_idx, ti, mi, si, i, clean_id, regime):
    mod = C.MODS[mi]
    clean = SG.make_clean(clean_id, mod)
    x = clean['signal'].astype(np.complex128)
    L = C.SIG_LEN
    fmax = (1 + clean['beta']) / (2 * C.SPS)
    mask = IG.inband_mask(L, fmax)
    p_s_in = IG.inband_power(x, mask)

    rng = np.random.default_rng(np.random.SeedSequence([C.SEED, 11, split_idx, ti, mi, si, i]))
    itype, sir = C.TYPES[ti], C.SIR_DB[si]
    j, params, frac = IG.generate(itype, rng, L, fmax, regime)
    p_j_in = IG.inband_power(j, mask)
    alpha = float(np.sqrt(p_s_in / (p_j_in * 10 ** (sir / 10.0))))
    jj = alpha * j

    snr = float(rng.uniform(*C.SNR_DB_RANGE))
    r = float(mask.mean())
    sigma2 = p_s_in / (10 ** (snr / 10.0) * r)          # 白噪声复方差；带内噪声功率 = sigma2 * r
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(L) + 1j * rng.standard_normal(L))

    mix = x + jj + noise
    return dict(
        clean=x, interference=jj, noise=noise, mixture=mix, bits=clean['bits'], symbols=clean['symbols'], gain=clean['gain'],
        alpha=alpha, sir_target=float(sir), snr_target=snr,
        sir_in=10 * np.log10(p_s_in / IG.inband_power(jj, mask)),
        snr_in=10 * np.log10(p_s_in / IG.inband_power(noise, mask)),
        sir_full=10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(jj) ** 2)),
        inband_frac=frac, beta=clean['beta'], tau=clean['tau'], cfo=clean['cfo'], phase=clean['phase'],
        params=json.dumps(params), clean_id=clean_id,
    )


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'v2')
    out = os.path.abspath(out)
    manifest = []
    for split_idx, (split, (id_base, n_per, regime)) in enumerate(C.SPLITS.items()):
        counter = id_base
        for ti, itype in enumerate(C.TYPES):
            for mi, mod in enumerate(C.MODS):
                for si, sir in enumerate(C.SIR_DB):
                    S = []
                    for i in range(n_per):
                        S.append(build_sample(split_idx, ti, mi, si, i, counter, regime))
                        counter += 1
                    d = os.path.join(out, split, itype, mod)
                    os.makedirs(d, exist_ok=True)
                    path = os.path.join(d, 'sir_%s.npz' % sir_tag(sir))

                    def stack(key, dtype=None):
                        a = np.array([s[key] for s in S])
                        return a.astype(dtype) if dtype else a
                    np.savez_compressed(
                        path,
                        clean_signal=stack('clean', np.complex64),
                        mixture_signal=stack('mixture', np.complex64),
                        interference_signal=stack('interference', np.complex64),
                        noise_signal=stack('noise', np.complex64),
                        bit_stream=stack('bits', np.uint8),
                        symbol_stream=stack('symbols', np.complex64),
                        demod_gain=stack('gain', np.float32),
                        modulation=np.array([mod] * n_per),
                        bits_per_symbol=np.full(n_per, C.BITS_PER_SYM[mod], dtype=np.int32),
                        sir_db_target=stack('sir_target', np.float32),
                        sir_db_inband=stack('sir_in', np.float32),
                        sir_db_fullband=stack('sir_full', np.float32),
                        snr_db_target=stack('snr_target', np.float32),
                        snr_db_inband=stack('snr_in', np.float32),
                        alpha=stack('alpha', np.float32),
                        interference_inband_frac=stack('inband_frac', np.float32),
                        rolloff=stack('beta', np.float32),
                        timing_offset=stack('tau', np.float32),
                        cfo=stack('cfo', np.float32),
                        phase=stack('phase', np.float32),
                        clean_id=stack('clean_id', np.int64),
                        interference_type=np.array([itype] * n_per),
                        interference_params=np.array([s['params'] for s in S]),
                        regime=np.array([regime] * n_per),
                        sample_index=np.arange(n_per, dtype=np.int32),
                    )
                    manifest.append((split, itype, mod, sir, n_per, os.path.relpath(path, out)))

    with open(os.path.join(out, 'manifest.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'interference_type', 'modulation', 'sir_db_inband', 'num_samples', 'npz_path'])
        w.writerows(manifest)
    meta = {k: getattr(C, k) for k in dir(C) if k.isupper()}
    meta['definitions'] = {
        'band': '|f| <= fmax, fmax=(1+rolloff)/(2*SPS) cycles/sample',
        'SIR': '10log10(P_clean_inband / P_interference_inband)',
        'SNR': '10log10(P_clean_inband / P_noise_inband), white noise',
        'mixture': 'clean + interference + noise',
    }
    with open(os.path.join(out, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    n = sum(m[4] for m in manifest)
    print('wrote %d files, %d samples -> %s' % (len(manifest), n, out))


if __name__ == '__main__':
    main()
