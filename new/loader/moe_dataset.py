"""读取 new/data/v2 的 PyTorch 数据集。

    from moe_dataset import MoEDiffDataset
    ds = MoEDiffDataset('new/data/v2', 'train')                # 默认：含噪声输入
    ds = MoEDiffDataset('new/data/v2', 'test', noise_mode='without', mods=['QPSK'], types=['ST', 'MT'])

每个样本是一个定长字典（可直接用默认 collate 组 batch）：
    clean             (2, 2560) float32   目标：无干扰无噪声的信号（I/Q 两通道）
    noisy             (2, 2560) float32   输入：clean + interference (+ noise，取决于 noise_mode)
    sir_db            ()        float32   带内 SIR 目标值
    snr_db            ()        float32   带内 SNR 目标值
    interference_type ()        int64     下标见 ds.type_names
    modulation        ()        int64     下标见 ds.mod_names
    symbols           (2, 640)  float32   理想发射符号（I/Q），可用于星座损失或离线算 BER
    demod             (5,)      float32   [rolloff, timing_offset, cfo, phase, demod_gain]，供已知同步的匹配滤波解调
    clean_id          ()        int64     干净信号唯一编号
    sample_index      ()        int64     本数据集内的位置

另外保留 ds.samples（字典列表，含 'clean' 的 numpy (2, L) 数组），与旧训练脚本里
"用 train_dataset.samples 计算 data_scale" 的写法兼容。

不含任何数据增强：本加载器只读取已生成的样本。
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from attributes import compute_attrs, N_ATTR            # noqa: E402
from physical_augment import apply_ops, parse_ops        # noqa: E402

NOISE_MODES = ('with', 'without')


def augment_pair(clean, noisy, rng):
    """训练增强（参考原始代码 data/augmentation.py 的四种算子；clean/noisy 为 (2, L) float32）。

    旋转 / 循环时移 / I-Q 微拉伸：对 clean 与 noisy 使用完全相同的参数（保持 noisy = clean + 干扰 + 噪声 的关系）。
    高斯白噪声：**只加到输入侧**（SNR 相对 clean 功率 20~40 dB），目标保持干净——与本项目"目标不含噪声"的任务定义一致
    （原始代码把噪声同步加到输入和目标上）。
    触发逻辑沿用原始设计：先从 {旋转 0.45, 噪声 0.45, 拉伸 0.10} 按权重无放回抽 1~2 个；时移独立以 50% 概率触发。
    """
    c = clean[0].astype(np.float64) + 1j * clean[1]
    y = noisy[0].astype(np.float64) + 1j * noisy[1]
    k = int(rng.integers(1, 3))
    chosen = set(rng.choice(['rotation', 'awgn', 'stretch'], size=k, replace=False, p=[0.45, 0.45, 0.10]).tolist())
    if 'rotation' in chosen:
        r = np.exp(1j * rng.uniform(0, 2 * np.pi))
        c, y = c * r, y * r
    if 'stretch' in chosen:
        e = rng.uniform(-0.05, 0.05)
        c = c.real * (1 + e) + 1j * c.imag * (1 - e)
        y = y.real * (1 + e) + 1j * y.imag * (1 - e)
    if rng.random() < 0.5:
        sh = int(rng.integers(-16, 17))
        c, y = np.roll(c, sh), np.roll(y, sh)
    if 'awgn' in chosen:
        snr = rng.uniform(20.0, 40.0)
        sigma2 = np.mean(np.abs(c) ** 2) / 10 ** (snr / 10.0)
        y = y + np.sqrt(sigma2 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    return _iq(c), _iq(y)


def _iq(x):
    return np.stack([x.real, x.imag], axis=0).astype(np.float32)


class MoEDiffDataset(Dataset):
    def __init__(self, root, split, types=None, mods=None, sirs=None, noise_mode='with',
                 max_per_cell=None, seed=0, augment=False, virtual_multiplier=1,
                 return_aux=False, aug_ops=None, attr_stats=None):
        """
        root          数据目录（含 metadata.json 与 <split>/<类型>/<调制>/sir_*.npz）
        split         'train' | 'val' | 'test' | 'test_ood'
        types, mods   只加载这些干扰类型 / 调制（默认全部）
        sirs          只加载这些 SIR 档位（dB，默认全部）
        noise_mode    'with'：noisy = mixture_signal（含噪声）；'without'：noisy = clean + interference
        max_per_cell  每个 (类型, 调制, SIR) 格子最多取多少条。按固定种子的排列取前 n 条，
                      因此同一种子下 n 小的集合包含于 n 大的集合（小样本子集）。
        augment       True 时每次取样返回随机增强后的视图（见 augment_pair），仅用于训练集
        virtual_multiplier  数据集长度放大倍数（每个样本在一个 epoch 内被取到这么多次，每次增强不同）
        return_aux    True 时每条额外返回 interf(2,L)=干扰信号 与 attrs(12,)=由干扰计算的物理属性（实验 3 辅助监督；用 attr_stats=(mean,std) 标准化）
        aug_ops       物理增强算子，如 'rot,remix'（见 physical_augment.py）；仅用于训练集，与旧的 augment 互斥
        """
        if noise_mode not in NOISE_MODES:
            raise ValueError('noise_mode must be one of %s' % (NOISE_MODES,))
        self.root = Path(root)
        self.augment = bool(augment)
        self.aug_ops = parse_ops(aug_ops) if isinstance(aug_ops, str) or aug_ops is None else tuple(aug_ops)
        self.return_aux = bool(return_aux)
        if self.aug_ops and self.augment:
            raise ValueError('augment 与 aug_ops 互斥')
        self.sps = None
        self.virtual_multiplier = max(1, int(virtual_multiplier))
        self.split = split
        self.noise_mode = noise_mode
        meta = json.load(open(self.root / 'metadata.json'))
        self.type_names = list(meta['TYPES'])
        self.mod_names = list(meta['MODS'])
        self.sir_levels = list(meta['SIR_DB'])
        self.meta = meta
        self.sig_len = int(meta['SIG_LEN'])
        self.n_sym = int(meta['N_SYM'])
        self.sps = int(meta['SPS'])

        for name, want, have in (('types', types, self.type_names), ('mods', mods, self.mod_names)):
            bad = [w for w in (want or []) if w not in have]
            if bad:
                raise ValueError('unknown %s %s; available: %s' % (name, bad, have))
        want_sirs = None if sirs is None else {float(s) for s in sirs}

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError('split not found: %s' % split_dir)
        files = sorted(split_dir.glob('*/*/*.npz'))
        if not files:
            raise FileNotFoundError('no npz under %s' % split_dir)

        self.samples = []
        for fi, f in enumerate(files):
            itype, mod = f.parent.parent.name, f.parent.name
            if itype not in self.type_names or mod not in self.mod_names:
                raise ValueError('unexpected directory names in %s' % f)
            if (types is not None and itype not in types) or (mods is not None and mod not in mods):
                continue
            z = np.load(f, allow_pickle=False)
            sir = float(z['sir_db_target'][0])
            if want_sirs is not None and sir not in want_sirs:
                continue
            # 文件内字段与目录名必须一致，避免目录被误放
            if str(z['interference_type'][0]) != itype or str(z['modulation'][0]) != mod:
                raise ValueError('file contents disagree with its directory: %s' % f)

            clean, interf, noise = z['clean_signal'], z['interference_signal'], z['noise_signal']
            noisy = z['mixture_signal'] if noise_mode == 'with' else clean + interf
            n = len(clean)
            order = np.arange(n)
            if max_per_cell is not None:
                order = np.random.RandomState(seed * 100003 + fi).permutation(n)[:max_per_cell]
            for i in order:
                self.samples.append(dict(
                    clean=_iq(clean[i]), noisy=_iq(noisy[i]),
                    sir_db=float(z['sir_db_target'][i]), snr_db=float(z['snr_db_target'][i]),
                    interference_type=self.type_names.index(itype), modulation=self.mod_names.index(mod),
                    symbols=_iq(z['symbol_stream'][i]),
                    demod=np.array([z['rolloff'][i], z['timing_offset'][i], z['cfo'][i], z['phase'][i],
                                    z['demod_gain'][i]], dtype=np.float32),
                    clean_id=int(z['clean_id'][i]),
                    interf=_iq(interf[i]),
                ))
        if not self.samples:
            raise ValueError('no samples selected (check types/mods/sirs filters)')
        self.attr_mean = self.attr_std = None
        if self.return_aux:
            A = compute_attrs(np.stack([s_['interf'] for s_ in self.samples]))
            for s_, a_ in zip(self.samples, A):
                s_['attrs_raw'] = a_
            if attr_stats is None:
                attr_stats = (A.mean(0), A.std(0) + 1e-6)
            self.attr_mean, self.attr_std = np.asarray(attr_stats[0], np.float32), np.asarray(attr_stats[1], np.float32)

    def __len__(self):
        return len(self.samples) * self.virtual_multiplier      # 不增强时为同一样本的重复（用于保持各设置的每 epoch 步数一致）

    def __getitem__(self, index):
        s = self.samples[index % len(self.samples)]
        clean, noisy = s['clean'], s['noisy']
        interf, sir_db, snr_db, demod, itype, attrs = s['interf'], s['sir_db'], s['snr_db'], s['demod'], s['interference_type'], s.get('attrs_raw')
        if self.aug_ops:
            rng = np.random.default_rng([int(torch.initial_seed()) % (2 ** 32), int(index)])
            t = apply_ops(self.aug_ops, dict(clean=clean, interf=interf, noisy=noisy, demod=demod, sps=self.sps, sir_db=sir_db, snr_db=snr_db,
                                             clean_id=s['clean_id'], interference_type=itype, attrs=attrs), rng, pool=self.samples)
            clean, noisy, interf, demod, sir_db, snr_db, itype, attrs = t['clean'], t['noisy'], t['interf'], t['demod'], t['sir_db'], t['snr_db'], t['interference_type'], t.get('attrs')
            if self.return_aux and attrs is None:
                attrs = compute_attrs(interf[None])[0]
        if self.augment:
            # 每个 worker 的 torch 种子在每个 epoch 都会变化，因此同一样本在不同 epoch 得到不同视图，同时整体可复现
            rng = np.random.default_rng([int(torch.initial_seed()) % (2 ** 32), int(index)])
            clean, noisy = augment_pair(clean, noisy, rng)
        item = {
            'clean': torch.from_numpy(np.ascontiguousarray(clean)),
            'noisy': torch.from_numpy(np.ascontiguousarray(noisy)),
            'sir_db': torch.tensor(sir_db, dtype=torch.float32),
            'snr_db': torch.tensor(snr_db, dtype=torch.float32),
            'interference_type': torch.tensor(itype, dtype=torch.long),
            'modulation': torch.tensor(s['modulation'], dtype=torch.long),
            'symbols': torch.from_numpy(s['symbols']),
            'demod': torch.from_numpy(np.ascontiguousarray(demod)),
            'clean_id': torch.tensor(s['clean_id'], dtype=torch.long),
            'sample_index': torch.tensor(index, dtype=torch.long),
        }
        if self.return_aux:
            item['interf'] = torch.from_numpy(np.ascontiguousarray(interf))
            item['attrs'] = torch.from_numpy(((np.asarray(attrs, np.float32) - self.attr_mean) / self.attr_std).astype(np.float32))
        return item

    def summary(self):
        """各 (类型, 调制) 的样本数，便于核对。"""
        from collections import Counter
        c = Counter((self.type_names[s['interference_type']], self.mod_names[s['modulation']]) for s in self.samples)
        return dict(sorted(c.items()))
