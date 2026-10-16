#!/usr/bin/env python
"""M2-0：循环平稳（谱相关）特征在本项目数据上的可分辨性分析（纯 CPU）。

用法：
    /root/autodl-tmp/envs/moediff_bw/bin/python m2_0_cyclo.py compute   # 计算特征+探针，写 reports/m2_0_cache/
    /root/autodl-tmp/envs/moediff_bw/bin/python m2_0_cyclo.py report    # 由缓存写 reports/m2_0_cyclo_analysis.md
    /root/miniconda3/bin/python m2_0_cyclo.py plot                      # 画图（moediff_bw 无 matplotlib）

------------------------------------------------------------------------
【预先固定的判据（先写下，不再改）】
 主统计量 = Nw=128 汉宁窗、hop=32、非共轭谱相关、alpha=0.25、|f|<=0.0625 内各 bin 的
            "相干系数按相位对齐后相加"(s_coh)；零假设 = 同一样本在近邻离循环
            alpha∈{0.1875,0.21875,0.28125,0.3125} 上的同一统计量。
 (i)  对 ST/MT/NB/PL 之一，在 SIR<=-10 dB 的至少一档上 d'>=3（主统计量；其他变体仅作敏感性报告）。
 (ii) 探针(主模型=MLP)里 B 相对 A：带内 SINR 回归 R2 提升>=0.03 或 SIR 7 分类准确率提升>=0.02，
      且增量 > 2×(逐折配对差的标准误)。（线性模型、全带 SINR 目标仅作敏感性报告）
 (i) 与 (ii) 同时满足才"通过"，否则"不通过，建议终止 M2"。
------------------------------------------------------------------------
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys, json, glob, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../new
DATA = os.path.join(ROOT, 'data', 'v2')
REP = os.path.join(ROOT, 'reports')
CACHE = os.path.join(REP, 'm2_0_cache')
SEED = 20261015
SPS = 4
TYPES = ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM']
MODS = ['QPSK', 'BPSK', '16QAM']
SIRS = [-20, -15, -10, -5, 0, 5, 10]
NARROW = ['ST', 'MT', 'NB', 'PL']
FBAND = 0.0625                                      # |f| 上限
CFO_MAX = 0.002


# ----------------------------------------------------------------------------
# 数据读取（与 MoEDiffDataset 同顺序：sorted glob，noisy=mixture_signal）
# ----------------------------------------------------------------------------
def load_split(split):
    files = sorted(glob.glob(os.path.join(DATA, split, '*', '*', '*.npz')))
    cols = dict(clean=[], noisy=[], typ=[], mod=[], sir=[], snr=[], beta=[], cfo=[], cid=[], par=[])
    for f in files:
        z = np.load(f, allow_pickle=False)
        t, m = f.split(os.sep)[-3], f.split(os.sep)[-2]
        n = len(z['clean_signal'])
        cols['clean'].append(z['clean_signal']); cols['noisy'].append(z['mixture_signal'])
        cols['typ'] += [TYPES.index(t)] * n; cols['mod'] += [MODS.index(m)] * n
        cols['sir'] += list(z['sir_db_target']); cols['snr'] += list(z['snr_db_target'])
        cols['beta'] += list(z['rolloff']); cols['cfo'] += list(z['cfo']); cols['cid'] += list(z['clean_id'])
        cols['par'] += [json.loads(s) for s in z['interference_params']]
    d = dict(clean=np.concatenate(cols['clean']).astype(np.complex64), noisy=np.concatenate(cols['noisy']).astype(np.complex64))
    for k in ('typ', 'mod', 'cid'):
        d[k] = np.array(cols[k], dtype=np.int64)
    for k in ('sir', 'snr', 'beta', 'cfo'):
        d[k] = np.array(cols[k], dtype=np.float64)
    d['sirc'] = np.array([SIRS.index(int(round(s))) for s in d['sir']])
    d['par'] = cols['par']
    return d


def check_against_dataset():
    """与 MoEDiffDataset 抽样核对（只做一次）。"""
    sys.path.insert(0, os.path.join(ROOT, 'loader'))
    from moe_dataset import MoEDiffDataset
    ds = MoEDiffDataset(DATA, 'test', noise_mode='with')
    d = load_split('test')
    assert len(ds.samples) == len(d['typ'])
    for i in range(0, len(ds.samples), 497):
        s = ds.samples[i]
        assert np.allclose(s['noisy'][0], d['noisy'][i].real) and np.allclose(s['noisy'][1], d['noisy'][i].imag)
        assert s['interference_type'] == d['typ'][i] and s['modulation'] == d['mod'][i]
    print('与 MoEDiffDataset 核对一致，N=%d' % len(ds.samples))


# ----------------------------------------------------------------------------
# 谱相关估计（时间平滑法，汉宁窗滑动 FFT）
# ----------------------------------------------------------------------------
def frames_fft(z, Nw, hop):
    """z:(B,L) complex -> (B,K,Nw) complex64"""
    L = z.shape[1]
    K = (L - Nw) // hop + 1
    idx = np.arange(Nw)[None, :] + hop * np.arange(K)[:, None]
    w = np.hanning(Nw + 1)[:-1].astype(np.float32)           # 周期汉宁窗
    return np.fft.fft(z[:, idx] * w, axis=-1).astype(np.complex64)


def scf_stats(X, Nw, m, conj):
    """alpha=m/Nw (m 为偶数)。非共轭 S(f)=<X(f+a/2)X*(f-a/2)>；共轭 S(f)=<X(a/2+f)X(a/2-f)>。
    f=j/Nw, |f|<=FBAND。返回若干标量统计量 (B,) 及 5 子带相干系数 (B,5)。"""
    jm = int(round(FBAND * Nw))
    j = np.arange(-jm, jm + 1)
    h = m // 2
    K = X.shape[1]
    if not conj:
        ip, im = (j + h) % Nw, (j - h) % Nw
        Xp, Xm = X[:, :, ip], X[:, :, im]
        S = (Xp * np.conj(Xm)).mean(1)[:, None, :]                 # (B,1,J)
    else:
        # 共轭循环频率 = alpha+2*CFO（CFO 最大 0.002）：对帧序列做过采样 FFT，在 |delta|<=0.004 内搜索
        ip, im = (h + j) % Nw, (h - j) % Nw
        Xp, Xm = X[:, :, ip], X[:, :, im]
        hop = Nw // 4
        NF = 512
        qm = int(np.ceil(2 * CFO_MAX * hop * NF))
        q = np.arange(-qm, qm + 1) % NF
        S = (np.fft.fft(Xp * Xm, n=NF, axis=1) / K)[:, q, :]       # (B,Nd,J)
    Dp, Dm = (np.abs(Xp) ** 2).mean(1), (np.abs(Xm) ** 2).mean(1)
    D = (np.sqrt(Dp * Dm) + 1e-20)[:, None, :]
    Ptot = (np.abs(X) ** 2).mean((1, 2)) + 1e-20
    J = len(j)
    g = S / D
    sc = np.abs(g.sum(2)) / J                                      # (B,Nd)
    ix = sc.argmax(1)
    ar = np.arange(len(ix))
    g, S = g[ar, ix], S[ar, ix]                                    # (B,J)
    out = dict(
        s_coh=sc[ar, ix],                      # 相干系数按相位对齐后相加（主）
        s_inc=np.abs(g).mean(1),               # 相干系数幅度的平均（不要求相位一致）
        s_pow=np.abs(S.sum(1)) / (J * Ptot),   # 功率归一化（相干）：|ΣS|/(J·全 bin 平均功率)
    )
    groups = np.array_split(np.arange(J), 5)
    out['bands'] = np.stack([np.abs(g[:, gi]).mean(1) for gi in groups], 1)
    return out


def alpha_plan(Nw):
    """(名字 -> m)；off1..4 是零假设用的近邻离循环 alpha，far=0.4。"""
    a = lambda al: int(round(al * Nw / 2.0)) * 2          # 取最近的偶数 m
    return dict(a025=a(0.25), a05=a(0.5), a0=0,
                off1=a(0.1875), off2=a(0.21875), off3=a(0.28125), off4=a(0.3125), far=a(0.40625))


NEAR = ['off1', 'off2', 'off3', 'off4']


def compute_feats(sig, Nw, hop=None, bs=256):
    """sig:(N,L) complex -> {(conj,aname,stat): (N,...)}"""
    hop = hop or Nw // 4
    plan = alpha_plan(Nw)
    res = {}
    for s in range(0, len(sig), bs):
        X = frames_fft(sig[s:s + bs], Nw, hop)
        for cj in (0, 1):
            for an, m in plan.items():
                st = scf_stats(X, Nw, m, bool(cj))
                for k, v in st.items():
                    res.setdefault((cj, an, k), []).append(v)
    return {k: np.concatenate(v) for k, v in res.items()}


def whole_window_moment(z, alpha=0.25):
    n = np.arange(z.shape[1])
    p = (np.abs(z) ** 2)
    return np.abs((p * np.exp(-2j * np.pi * alpha * n)).sum(1)) / p.sum(1)


# ----------------------------------------------------------------------------
# 探针特征
# ----------------------------------------------------------------------------
def tf_feat(z):
    """A：与 TF 专家同口径（I/Q 两通道分别做 n_fft=128,hop=32 的 STFT 幅度），
    池化到 16(频)×8(时)×2 通道=256 维，取 log。信号先按 RMS 归一。"""
    import torch
    torch.set_num_threads(4)
    out = []
    win = torch.hann_window(128)
    for s in range(0, len(z), 512):
        zz = z[s:s + 512]
        zz = zz / np.sqrt((np.abs(zz) ** 2).mean(1, keepdims=True))
        x = torch.from_numpy(np.stack([zz.real, zz.imag], 1).astype(np.float32))      # (B,2,L)
        B = x.shape[0]
        st = torch.stft(x.reshape(B * 2, -1), n_fft=128, hop_length=32, win_length=128, window=win, return_complex=True)
        mag = torch.sqrt(st.real ** 2 + st.imag ** 2 + 1e-12).reshape(B, 2, st.shape[-2], st.shape[-1])
        p = torch.nn.functional.adaptive_avg_pool2d(mag, (16, 8)).reshape(B, -1)
        out.append(torch.log(p + 1e-4).numpy())
    return np.concatenate(out)


def cyc_blocks(F):
    """由 compute_feats(Nw=128) 的结果拼 B_cyc(44 维) 与对照 C_off(32 维)。"""
    lg = lambda x: np.log(np.asarray(x) + 1e-3)

    def blk(cj, an):
        return np.concatenate([lg(F[(cj, an, 's_coh')])[:, None], lg(F[(cj, an, 's_inc')])[:, None],
                               lg(F[(cj, an, 's_pow')])[:, None], lg(F[(cj, an, 'bands')])], 1)      # 8 维
    nearmean = lambda cj, k: np.mean([F[(cj, a, k)] for a in NEAR], 0)
    cyc = [blk(0, 'a025'), blk(1, 'a025'), blk(1, 'a0')]                                                # 24
    for cj in (0, 1):
        cyc.append(np.stack([lg(nearmean(cj, 's_coh')), lg(nearmean(cj, 's_inc')), lg(nearmean(cj, 's_pow'))], 1))   # 6
    for cj in (0, 1):
        cyc.append(lg(nearmean(cj, 'bands')))                                                           # 10
    cyc.append(np.stack([lg(F[(0, 'a05', 's_coh')]), lg(F[(0, 'a05', 's_inc')]),
                         lg(F[(1, 'a05', 's_coh')]), lg(F[(1, 'a05', 's_inc')])], 1))                  # 4
    Bc = np.concatenate(cyc, 1)
    Co = np.concatenate([blk(0, 'off3'), blk(1, 'off3'), blk(0, 'off2'), blk(1, 'off2')], 1)           # 32
    return Bc.astype(np.float32), Co.astype(np.float32)


def sinr_targets(d):
    L = d['noisy'].shape[1]
    res = d['noisy'] - d['clean']
    fr = np.abs(np.fft.fftfreq(L))
    fmax = (1 + d['beta']) / (2 * SPS)
    mask = fr[None, :] <= fmax[:, None]
    Pc = (np.abs(np.fft.fft(d['clean'], axis=1)) ** 2 * mask).sum(1)
    Pr = (np.abs(np.fft.fft(res, axis=1)) ** 2 * mask).sum(1)
    in_db = 10 * np.log10(Pc / Pr)
    full_db = 10 * np.log10((np.abs(d['clean']) ** 2).sum(1) / (np.abs(res) ** 2).sum(1))
    return in_db, full_db


# ----------------------------------------------------------------------------
# 干扰频率落区统计
# ----------------------------------------------------------------------------
def interferer_zones(d, Nw=128):
    """对 ST/MT/NB/PL 逐样本判断干扰主谱成分是否落在
       DC 区 |f|<=beta/8（任务书原假设的"被破坏区"）与 边缘区 ||f|-0.125|<=beta/8+1/Nw
       （按 SCF 的定义，真正污染 alpha0 所用 bin X(f±0.125) 的区域）。"""
    n = len(d['typ'])
    dc = np.zeros(n, bool); edge = np.zeros(n, bool)
    for i in range(n):
        t, p, b = TYPES[d['typ'][i]], d['par'][i], d['beta'][i]
        hw = b / 8.0
        if t == 'ST':
            fs, ext = [p['f']], 0.0
        elif t == 'MT':
            fs, ext = p['f'], 0.0
        elif t == 'NB':
            fs, ext = [p['fc']], p['bw'] / 2.0            # 高斯 sigma=bw/2
        elif t == 'PL':
            fs, ext = [p['fc']], 2 * 5.0 / (2 * np.pi * p['width'])   # 2σ_f，σ_t=width/5
        else:
            continue
        for f in fs:
            lo, hi = abs(f) - ext, abs(f) + ext
            if lo <= hw:
                dc[i] = True
            e = 0.125
            if (lo <= e + hw + 1.0 / Nw) and (hi >= e - hw - 1.0 / Nw):
                edge[i] = True
    return dc, edge


# ----------------------------------------------------------------------------
# 检测指标
# ----------------------------------------------------------------------------
def auc(pos, neg):
    """P(pos>neg)，并列取一半（Mann-Whitney）"""
    pos, neg = np.asarray(pos), np.asarray(neg)
    s = np.sort(neg)
    lo = np.searchsorted(s, pos, 'left'); hi = np.searchsorted(s, pos, 'right')
    return float(((lo + hi) / 2.0).sum() / (len(pos) * len(neg)))


def dprime(pos, off):
    return float((np.mean(pos) - np.mean(off)) / (np.std(off) + 1e-12))


# ----------------------------------------------------------------------------
# 探针（无 sklearn：torch CPU MLP / 线性）
# ----------------------------------------------------------------------------
def fit_predict(Xtr, ytr, Xte, task, kind, seed, ncls=None):
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32)); Xte = torch.from_numpy(((Xte - mu) / sd).astype(np.float32))
    d = Xtr.shape[1]
    if task == 'cls':
        out, y = ncls, torch.from_numpy(ytr.astype(np.int64))
    else:
        ym, ys = ytr.mean(), ytr.std()
        out, y = 1, torch.from_numpy(((ytr - ym) / ys).astype(np.float32))
    if kind == 'mlp':
        net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, out))
        epochs, lr, wd = EPOCHS, 2e-3, 5e-2
    else:
        net = nn.Linear(d, out)
        epochs, lr, wd = EPOCHS, 5e-3, 1e-2
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr); bs = 128
    steps = epochs * ((n + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            o = net(Xtr[b])
            loss = nn.functional.cross_entropy(o, y[b]) if task == 'cls' else nn.functional.mse_loss(o.squeeze(1), y[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        o = net(Xte)
    if task == 'cls':
        return o.argmax(1).numpy()
    return o.squeeze(1).numpy() * ys + ym


EPOCHS = 60


def group_folds(cid, k=5, seed=0):
    ug = np.unique(cid)
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in cid])


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ----------------------------------------------------------------------------
def stage_features():
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    check_against_dataset()
    te = load_split('test'); tr = load_split('train')
    print('loaded', time.time() - t0)
    R = {}
    R['ww_clean'] = whole_window_moment(te['clean'])
    R['ww_noisy'] = whole_window_moment(te['noisy'])
    neg_te = (te['noisy'] - te['clean']).astype(np.complex64)
    sets = dict(noisy=te['noisy'], neg=neg_te, clean=te['clean'])
    for Nw in (64, 128):
        for nm, sig in sets.items():
            F = compute_feats(sig, Nw)
            for k, v in F.items():
                R['F%d|full|%s|%d|%s|%s' % (Nw, nm, k[0], k[1], k[2])] = v
            print('scf Nw=%d %s' % (Nw, nm), time.time() - t0, flush=True)
    for wn, sl in (('h1', slice(0, 1280)), ('h2', slice(1280, 2560))):
        for nm, sig in sets.items():
            F = compute_feats(np.ascontiguousarray(sig[:, sl]), 128)
            for k, v in F.items():
                R['F128|%s|%s|%d|%s|%s' % (wn, nm, k[0], k[1], k[2])] = v
    print('half windows done', time.time() - t0, flush=True)
    for nm, d in (('train', tr), ('test', te)):
        A = tf_feat(d['noisy'])
        F = compute_feats(d['noisy'], 128)
        Bc, Co = cyc_blocks(F)
        sin, sfull = sinr_targets(d)
        for k, v in (('A', A), ('Bc', Bc), ('Co', Co), ('sinr_in', sin), ('sinr_full', sfull)):
            R['P_%s_%s' % (nm, k)] = v
        for k in ('typ', 'mod', 'sirc', 'cid', 'sir', 'snr', 'beta', 'cfo'):
            R['M_%s_%s' % (nm, k)] = d[k]
        print('probe feats', nm, A.shape, Bc.shape, Co.shape, time.time() - t0, flush=True)
    R['zone_dc'], R['zone_edge'] = interferer_zones(te)
    # ST/MT/NB/PL 干扰主频（供落区分析）：ST: f；其他: 取 |f| 最小者之外的信息已在 zone 里
    R['st_f'] = np.array([p.get('f', np.nan) if isinstance(p.get('f', None), float) else np.nan for p in te['par']])
    np.savez_compressed(os.path.join(CACHE, 'feats.npz'), **R)
    print('saved feats', time.time() - t0)


def stage_probe(epochs=None):
    global EPOCHS
    if epochs:
        EPOCHS = epochs
    t0 = time.time()
    R = np.load(os.path.join(CACHE, 'feats.npz'))
    P = {}
    for k in ('A', 'Bc', 'Co', 'sinr_in', 'sinr_full', 'typ', 'mod', 'sirc', 'cid'):
        pre = 'P_' if k in ('A', 'Bc', 'Co', 'sinr_in', 'sinr_full') else 'M_'
        P[k] = np.concatenate([R[pre + 'train_' + k], R[pre + 'test_' + k]])
    sets_x = dict(A=P['A'], B=np.concatenate([P['A'], P['Bc']], 1), C=np.concatenate([P['A'], P['Co']], 1))
    folds = group_folds(P['cid'], 5, SEED)
    tasks = dict(sir=('cls', P['sirc'], 7), mod=('cls', P['mod'], 3), typ=('cls', P['typ'], 8),
                 sinr_in=('reg', P['sinr_in'], None), sinr_full=('reg', P['sinr_full'], None))
    out, preds = {}, {}
    for kind in ('mlp', 'lin'):
        for tn, (task, y, nc) in tasks.items():
            for fs, X in sets_x.items():
                pred = np.zeros(len(y), dtype=np.int64 if task == 'cls' else np.float64)
                fm = []
                for f in range(5):
                    tri, tei = folds != f, folds == f
                    p = fit_predict(X[tri], y[tri], X[tei], task, kind, SEED + f, nc)
                    pred[tei] = p
                    fm.append(float((p == y[tei]).mean()) if task == 'cls' else r2(y[tei], p))
                key = '%s|%s|%s' % (kind, tn, fs)
                out[key] = dict(folds=fm, mean=float(np.mean(fm)),
                                mae=None if task == 'cls' else float(np.abs(pred - y).mean()))
                preds[key] = pred
                print(key, '%.4f' % out[key]['mean'], np.round(fm, 3), '%.0fs' % (time.time() - t0), flush=True)
    np.savez_compressed(os.path.join(CACHE, 'probe_preds.npz'), **{k.replace('|', '__'): v for k, v in preds.items()},
                        y_sir=P['sirc'], y_mod=P['mod'], y_typ=P['typ'], y_sinr_in=P['sinr_in'], y_sinr_full=P['sinr_full'],
                        folds=folds, n_dims=np.array([P['A'].shape[1], P['Bc'].shape[1], P['Co'].shape[1]]))
    json.dump(out, open(os.path.join(CACHE, 'probe.json'), 'w'), indent=1)
    print('probe done', time.time() - t0)


def stage_ww():
    """整窗口循环矩在 alpha=0.25 与近邻离循环 alpha 上的取值（对照"频带分辨"特征）。"""
    te = load_split('test')
    out = {}
    for nm, sig in (('noisy', te['noisy']), ('neg', te['noisy'] - te['clean']), ('clean', te['clean'])):
        for a in (0.25, 0.1875, 0.21875, 0.28125, 0.3125):
            out['%s|%.5f' % (nm, a)] = whole_window_moment(sig, a)
    np.savez_compressed(os.path.join(CACHE, 'ww.npz'), **out)
    print('ww saved')


def stage_probe_extra():
    """补充消融：只用循环特征(Bonly)、最小循环特征集(Bmin=nc/共轭@0.25 各 8 维)。"""
    t0 = time.time()
    R = np.load(os.path.join(CACHE, 'feats.npz'))
    cat = lambda k, pre: np.concatenate([R[pre + 'train_' + k], R[pre + 'test_' + k]])
    A, Bc = cat('A', 'P_'), cat('Bc', 'P_')
    sirc, cid = cat('sirc', 'M_'), cat('cid', 'M_')
    sin = cat('sinr_in', 'P_')
    folds = group_folds(cid, 5, SEED)
    sets_x = dict(Bonly=Bc, Bmin=np.concatenate([A, Bc[:, :16]], 1))
    out = {}
    for tn, (task, y, nc) in dict(sir=('cls', sirc, 7), sinr_in=('reg', sin, None)).items():
        for fs, X in sets_x.items():
            fm = []
            for f in range(5):
                tri, tei = folds != f, folds == f
                p = fit_predict(X[tri], y[tri], X[tei], task, 'mlp', SEED + f, nc)
                fm.append(float((p == y[tei]).mean()) if task == 'cls' else r2(y[tei], p))
            out['mlp|%s|%s' % (tn, fs)] = dict(folds=fm, mean=float(np.mean(fm)))
            print(tn, fs, np.round(fm, 3), time.time() - t0, flush=True)
    json.dump(out, open(os.path.join(CACHE, 'probe_extra.json'), 'w'), indent=1)

# ----------------------------------------------------------------------------
# 分析汇总
# ----------------------------------------------------------------------------
def analyze():
    R = np.load(os.path.join(CACHE, 'feats.npz'))
    W = np.load(os.path.join(CACHE, 'ww.npz'))
    typ, mod, sirc, beta = R['M_test_typ'], R['M_test_mod'], R['M_test_sirc'], R['M_test_beta']
    dcz, edz = R['zone_dc'], R['zone_edge']
    F = lambda Nw, win, which, cj, a, s: R['F%d|%s|%s|%d|%s|%s' % (Nw, win, which, cj, a, s)]
    res = {}

    def dp_auc(Nw, win, cj, stat, k, a0='a025'):
        pos = F(Nw, win, 'noisy', cj, a0, stat)[k]
        off = np.concatenate([F(Nw, win, 'noisy', cj, a, stat)[k] for a in NEAR])
        neg = F(Nw, win, 'neg', cj, a0, stat)[k]
        return dprime(pos, off), auc(pos, neg)

    # ---- Q2 矩阵 ----
    variants = {'primary': (128, 'full', 0, 's_coh'), 'pow128': (128, 'full', 0, 's_pow'), 'inc128': (128, 'full', 0, 's_inc'),
                'coh64': (64, 'full', 0, 's_coh'), 'pow64': (64, 'full', 0, 's_pow'), 'cj128': (128, 'full', 1, 's_coh'),
                'h1': (128, 'h1', 0, 's_coh'), 'h2': (128, 'h2', 0, 's_coh')}
    res['mat'] = {}
    for vn, (Nw, win, cj, stat) in variants.items():
        D = np.zeros((8, 7)); A = np.zeros((8, 7))
        for t in range(8):
            for s in range(7):
                k = (typ == t) & (sirc == s)
                if cj:
                    k = k & (mod == MODS.index('BPSK'))
                D[t, s], A[t, s] = dp_auc(Nw, win, cj, stat, k)
        res['mat'][vn] = dict(d=D.tolist(), auc=A.tolist())
    # 整窗口矩阵
    D = np.zeros((8, 7)); A = np.zeros((8, 7))
    for t in range(8):
        for s in range(7):
            k = (typ == t) & (sirc == s)
            pos = W['noisy|0.25000'][k]
            off = np.concatenate([W['noisy|%.5f' % a][k] for a in (0.1875, 0.21875, 0.28125, 0.3125)])
            D[t, s] = dprime(pos, off); A[t, s] = auc(pos, W['neg|0.25000'][k])
    res['mat']['whole'] = dict(d=D.tolist(), auc=A.tolist())
    res['ww'] = dict(clean=float(W['clean|0.25000'].mean()),
                     noisy_mean=[float(R['ww_noisy'][sirc == s].mean()) for s in range(7)],
                     noisy_std=[float(R['ww_noisy'][sirc == s].std()) for s in range(7)])
    # 逐调制（主统计量）
    res['bymod'] = {}
    for t in range(8):
        for mi, mn in enumerate(MODS):
            res['bymod']['%s|%s' % (TYPES[t], mn)] = [dp_auc(128, 'full', 0, 's_coh', (typ == t) & (mod == mi) & (sirc == s)) for s in range(7)]
    # 特征均值随 SIR（图用）
    res['coh_vs_sir'] = {TYPES[t]: dict(
        pos=[float(F(128, 'full', 'noisy', 0, 'a025', 's_coh')[(typ == t) & (sirc == s)].mean()) for s in range(7)],
        null=[float(np.mean([F(128, 'full', 'noisy', 0, a, 's_coh')[(typ == t) & (sirc == s)].mean() for a in NEAR])) for s in range(7)],
        neg=[float(F(128, 'full', 'neg', 0, 'a025', 's_coh')[(typ == t) & (sirc == s)].mean()) for s in range(7)]) for t in range(8)}

    # ---- Q1 干净信号 ----
    q1 = []
    for Nw in (128, 64):
        for mi, mn in enumerate(MODS):
            for lo, hi in ((0.25, 0.375), (0.375, 0.5001)):
                k = (mod == mi) & (beta >= lo) & (beta < hi)
                g = lambda cj, a, s: F(Nw, 'full', 'clean', cj, a, s)[k]
                nearm = lambda cj, s: np.mean([g(cj, a, s) for a in NEAR], 0)
                q1.append(dict(Nw=Nw, mod=mn, beta='%.3f-%.3f' % (lo, min(hi, 0.5)), n=int(k.sum()),
                               nc_coh=float(g(0, 'a025', 's_coh').mean()), nc_coh_sd=float(g(0, 'a025', 's_coh').std()),
                               nc_pow=float(g(0, 'a025', 's_pow').mean()), nc_null_pow=float(nearm(0, 's_pow').mean()),
                               nc_null_coh=float(nearm(0, 's_coh').mean()),
                               a05_pow=float(g(0, 'a05', 's_pow').mean()), far_pow=float(g(0, 'far', 's_pow').mean()),
                               cj_coh=float(g(1, 'a025', 's_coh').mean()), cj_null_coh=float(nearm(1, 's_coh').mean()),
                               cj_pow=float(g(1, 'a025', 's_pow').mean()), cj_null_pow=float(nearm(1, 's_pow').mean()),
                               cj0_pow=float(g(1, 'a0', 's_pow').mean())))
    res['q1'] = q1

    # ---- 干扰落区 ----
    zone = {}
    for t in range(8):
        if TYPES[t] not in NARROW:
            continue
        k = typ == t
        zone[TYPES[t]] = dict(dc=float(dcz[k].mean()), edge=float(edz[k].mean()), either=float((dcz | edz)[k].mean()))
    res['zone_frac'] = zone
    zs = {}
    low = sirc <= 2                                         # SIR<=-10
    for t in range(8):
        if TYPES[t] not in NARROW:
            continue
        for zn, zv in (('edge', edz), ('dc', dcz)):
            for hit in (True, False):
                k = (typ == t) & low & (zv == hit)
                if k.sum() >= 20:
                    d_, a_ = dp_auc(128, 'full', 0, 's_coh', k)
                    zs['%s|%s|%s' % (TYPES[t], zn, 'hit' if hit else 'miss')] = dict(n=int(k.sum()), d=d_, auc=a_)
    res['zone_perf'] = zs

    # ---- Q5 稳定性 ----
    stab = {}
    for t in range(8):
        for s in (0, 2, 4):
            cvs = []
            for mi in range(3):
                k = (typ == t) & (mod == mi) & (sirc == s)
                v = F(128, 'full', 'noisy', 0, 'a025', 's_coh')[k]
                cvs.append(float(v.std() / v.mean()))
            stab['%s|%d' % (TYPES[t], SIRS[s])] = float(np.mean(cvs))
    res['cv_noisy'] = stab
    cv_clean = {}
    for mi, mn in enumerate(MODS):
        for lo, hi in ((0.25, 0.375), (0.375, 0.5001)):
            k = (mod == mi) & (beta >= lo) & (beta < hi)
            v = F(128, 'full', 'clean', 0, 'a025', 's_coh')[k]
            cv_clean['%s|%.3f' % (mn, lo)] = float(v.std() / v.mean())
    res['cv_clean'] = cv_clean
    win = {}
    for t in range(8):
        for s in (0, 2):
            k = (typ == t) & (sirc == s)
            row = {}
            for wn in ('full', 'h1', 'h2'):
                pos = F(128, wn, 'noisy', 0, 'a025', 's_coh')[k]
                off = np.concatenate([F(128, wn, 'noisy', 0, a, 's_coh')[k] for a in NEAR])
                d_, a_ = dp_auc(128, wn, 0, 's_coh', k)
                row[wn] = dict(pos_sd=float(pos.std()), null_sd=float(off.std()), null_mean=float(off.mean()), pos_mean=float(pos.mean()), d=d_, auc=a_)
            win['%s|%d' % (TYPES[t], SIRS[s])] = row
    res['window'] = win
    # 全局零假设方差比
    res['null_sd_ratio'] = float(np.mean([np.concatenate([F(128, 'h1', 'noisy', 0, a, 's_coh') for a in NEAR]).std() /
                                          np.concatenate([F(128, 'full', 'noisy', 0, a, 's_coh') for a in NEAR]).std()]))

    # ---- 各 alpha 的平均特征（干净/噪声）----
    alp = {}
    for an in ('a025', 'a05', 'a0', 'off1', 'off2', 'off3', 'off4', 'far'):
        alp[an] = dict(
            clean_pow=float(F(128, 'full', 'clean', 0, an, 's_pow').mean()), noisy_pow=float(F(128, 'full', 'noisy', 0, an, 's_pow').mean()),
            noisy_coh=float(F(128, 'full', 'noisy', 0, an, 's_coh').mean()), noisy_coh_sd=float(F(128, 'full', 'noisy', 0, an, 's_coh').std()),
            neg_coh=float(F(128, 'full', 'neg', 0, an, 's_coh').mean()),
            bpsk_cj_clean_pow=float(F(128, 'full', 'clean', 1, an, 's_pow')[mod == 1].mean()),
            bpsk_cj_noisy_coh=float(F(128, 'full', 'noisy', 1, an, 's_coh')[mod == 1].mean()))
    res['alpha'] = alp
    # ---- 单个循环特征能否当"SIR 计"：按干扰类型，对 带内 SINR 做 (log s_pow, log s_coh) 的二元线性拟合 ----
    sin = R['P_test_sinr_in']
    meter = {}
    for t in range(8):
        k = typ == t
        Xm = np.stack([np.ones(k.sum()), np.log10(F(128, 'full', 'noisy', 0, 'a025', 's_pow')[k] + 1e-6),
                       np.log10(F(128, 'full', 'noisy', 0, 'a025', 's_coh')[k] + 1e-6), (beta[k] - 0.375) * 10], 1)
        w, *_ = np.linalg.lstsq(Xm, sin[k], rcond=None)
        pr_ = Xm @ w
        c1 = np.corrcoef(np.log10(F(128, 'full', 'noisy', 0, 'a025', 's_pow')[k] + 1e-6), sin[k])[0, 1]
        meter[TYPES[t]] = dict(r2=float(1 - ((sin[k] - pr_) ** 2).sum() / ((sin[k] - sin[k].mean()) ** 2).sum()),
                               mae=float(np.abs(sin[k] - pr_).mean()), corr_pow=float(c1), sd=float(sin[k].std()))
    res['meter'] = meter
    return res


def probe_tables():
    P = json.load(open(os.path.join(CACHE, 'probe.json')))
    ex = json.load(open(os.path.join(CACHE, 'probe_extra.json'))) if os.path.exists(os.path.join(CACHE, 'probe_extra.json')) else {}
    return P, ex


def paired(P, kind, tn, a, b):
    fa, fb = np.array(P['%s|%s|%s' % (kind, tn, a)]['folds']), np.array(P['%s|%s|%s' % (kind, tn, b)]['folds'])
    d = fb - fa
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def verdict(res, P):
    D = np.array(res['mat']['primary']['d'])
    best = max((D[TYPES.index(t), s], t, SIRS[s]) for t in NARROW for s in range(3))
    i_ok = best[0] >= 3.0
    sir_d, sir_se = paired(P, 'mlp', 'sir', 'A', 'B')
    sin_d, sin_se = paired(P, 'mlp', 'sinr_in', 'A', 'B')
    ii_sir = sir_d >= 0.02 and sir_d > 2 * sir_se
    ii_sin = sin_d >= 0.03 and sin_d > 2 * sin_se
    return dict(best=best, i_ok=bool(i_ok), sir=(sir_d, sir_se), sinr=(sin_d, sin_se), ii_sir=bool(ii_sir), ii_sinr=bool(ii_sin),
                ii_ok=bool(ii_sir or ii_sin), passed=bool(i_ok and (ii_sir or ii_sin)))

# ----------------------------------------------------------------------------
# 画图（需要 matplotlib：用 /root/miniconda3/bin/python）
# ----------------------------------------------------------------------------
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C4 = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']          # 参考调色板前几个槽位
C4B = ['#e87ba4', '#008300', '#eda100', '#e34948']
MK = ['o', 's', '^', 'D']


def _style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#8a8984')
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def stage_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    res = analyze()
    # fig1 热图
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, key, ttl, vmax, fmt in ((axs[0], 'd', "d' (primary stat, Nw=128)", 30, '%.1f'), (axs[1], 'auc', 'AUC (signal+interf+noise vs interf+noise)', 1.0, '%.2f')):
        M = np.array(res['mat']['primary'][key])
        vmin = 0 if key == 'd' else 0.4
        im = ax.imshow(np.clip(M, vmin, vmax), cmap='Blues', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(7)); ax.set_xticklabels(SIRS); ax.set_yticks(range(8)); ax.set_yticklabels(TYPES)
        ax.set_xlabel('in-band SIR (dB)', color=INK2); ax.set_title(ttl, color=INK, fontsize=11, loc='left')
        for i in range(8):
            for j in range(7):
                v = M[i, j]
                dark = np.clip(v, vmin, vmax) > vmin + 0.6 * (vmax - vmin)
                ax.text(j, i, fmt % v, ha='center', va='center', fontsize=8, color='white' if dark else INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(REP, 'm2_0_fig1_dprime_auc.png'), dpi=140); plt.close(fig)

    # fig2 相干系数随 SIR
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, grp, ttl in ((axs[0], ['ST', 'MT', 'NB', 'PL'], 'Narrowband interference'), (axs[1], ['WB', 'LFM', 'RFM', 'NFM'], 'Wideband / FM-type interference')):
        for i, t in enumerate(grp):
            v = res['coh_vs_sir'][t]
            ax.plot(SIRS, v['pos'], color=(C4 if grp[0] == 'ST' else C4B)[i], marker=MK[i], lw=2, ms=6, label=t)
        nl = np.mean([res['coh_vs_sir'][t]['null'] for t in grp], 0)
        ax.plot(SIRS, nl, color='#8a8984', lw=1.5, ls='--', label='off-cycle null')
        ax.set_xlabel('in-band SIR (dB)', color=INK2); ax.set_title(ttl, color=INK, fontsize=11, loc='left')
        ax.set_ylim(0, 0.85); ax.legend(frameon=False, fontsize=9, ncol=5, loc='upper left'); _style(ax); ax.set_xticks(SIRS)
    axs[0].set_ylabel('alpha=0.25 coherence (s_coh)', color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(REP, 'm2_0_fig2_coh_vs_sir.png'), dpi=140); plt.close(fig)

    # fig3 探针
    P, ex = probe_tables()
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.9))
    cols = [C4[0], C4[1], '#8a8984']
    for ax, (tn, ttl) in zip(axs, (('sir', 'SIR 7-class accuracy'), ('mod', 'Modulation 3-class accuracy'), ('typ', 'Interf. type 8-class accuracy'),
                                  ('sinr_in', 'In-band SINR regression R2'), ('sinr_full', 'Full-band SINR regression R2'))):
        for ki, kind in enumerate(('mlp', 'lin')):
            for si, fs in enumerate(('A', 'B', 'C')):
                v = np.array(P['%s|%s|%s' % (kind, tn, fs)]['folds'])
                x = ki * 4 + si
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(5), color=cols[si], marker=['o', 's', '^'][si], ms=8, capsize=3, lw=1.5)
                ax.text(x, v.mean() + 0.012 * (1 if tn != 'sinr_in' and tn != 'sinr_full' else 0.4), '%.3f' % v.mean(), ha='center', fontsize=7, color=INK2)
        ax.set_xticks([1, 5]); ax.set_xticklabels(['MLP', 'Linear']); ax.set_xlim(-1, 7)
        lo = min(P['%s|%s|%s' % (k, tn, f)]['mean'] for k in ('mlp', 'lin') for f in 'ABC')
        ax.set_ylim(max(0, lo - 0.08), min(1.0, max(P['%s|%s|%s' % (k, tn, f)]['mean'] for k in ('mlp', 'lin') for f in 'ABC') + 0.06))
        ax.set_title(ttl, color=INK, fontsize=10, loc='left'); _style(ax)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=cols[i], marker=['o', 's', '^'][i], lw=0, ms=8, label=l) for i, l in
                        enumerate(('A: TF magnitude', 'B: A + cyclic feats', 'C: A + off-cycle control'))],
               loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(REP, 'm2_0_fig3_probe.png'), dpi=140); plt.close(fig)

    # fig4 按 SIR 分层的 SINR MAE
    pr = np.load(os.path.join(CACHE, 'probe_preds.npz'))
    y, sc = pr['y_sinr_in'], pr['y_sir']
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for si, (fs, c) in enumerate((('A', C4[0]), ('B', C4[1]), ('C', '#b9b8b2'))):
        p = pr['mlp__sinr_in__' + fs]
        mae = [np.abs(p - y)[sc == s].mean() for s in range(7)]
        ax.plot(SIRS, mae, color=c, marker=MK[si], lw=2, ms=6, label={'A': 'A: TF magnitude', 'B': 'B: A + cyclic feats', 'C': 'C: A + off-cycle control'}[fs])
    ax.set_xlabel('in-band SIR (dB)', color=INK2); ax.set_ylabel('in-band SINR MAE (dB)', color=INK2)
    ax.set_xticks(SIRS); ax.legend(frameon=False, fontsize=9); _style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(REP, 'm2_0_fig4_sinr_mae_by_sir.png'), dpi=140); plt.close(fig)
    print('figures saved')

# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------
def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def mat_table(M, fmt, aucM=None):
    rows = []
    for t in range(8):
        cells = []
        for s in range(7):
            c = fmt % M[t][s]
            if aucM is not None:
                c += ' / %.2f' % aucM[t][s]
            cells.append(c)
        rows.append([TYPES[t]] + cells)
    return md_table(['类型 \\ SIR(dB)'] + [str(s) for s in SIRS], rows)


def stage_report():
    res = analyze()
    P, ex = probe_tables()
    V = verdict(res, P)
    pr = np.load(os.path.join(CACHE, 'probe_preds.npz'))
    nd = pr['n_dims']
    M = res['mat']
    L = []
    w = L.append

    def pm(kind, tn, fs):
        v = np.array(P['%s|%s|%s' % (kind, tn, fs)]['folds'])
        return '%.3f±%.3f' % (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))

    def dl(kind, tn, a, b):
        d, se = paired(P, kind, tn, a, b)
        return '%+.3f±%.3f' % (d, se)

    D = np.array(M['primary']['d']); Au = np.array(M['primary']['auc'])
    ti = lambda t: TYPES.index(t)
    rel = lambda a, b: (a, b)
    sin_mae = lambda fs: P['mlp|sinr_in|%s' % fs]['mae']

    # ---------------- 结论 ----------------
    w('# M2-0 循环平稳（谱相关）特征的可分辨性分析')
    w('')
    w('> 纯 CPU 实验，脚本 `new/analysis/m2_0_cyclo.py`（固定种子，可复现）。数据：`new/data/v2`，test 全部 6720 条（未抽样；每个 类型×调制×SIR 格 40 条）用于 Q1/Q2/Q5；探针用 train+test 共 13440 条做 5 折交叉验证（按 clean_id 分组）。')
    w('')
    w('## 0. 结论先行')
    w('')
    verdict_txt = '通过' if V['passed'] else '不通过，建议终止 M2'
    w('**判定（按预先固定的判据）：%s。**' % verdict_txt)
    w('')
    w('- 判据 (i) %s：窄带型干扰里最好的格子是 %s @ SIR=%d dB，d\'=%.1f（要求 ≥3）；ST/MT/PL/NB 在 **所有** SIR 档（含 -20 dB）d\'都 ≥%.1f，AUC≈1.00。'
      % ('满足' if V['i_ok'] else '不满足', V['best'][1], V['best'][2], V['best'][0],
         min(D[ti(t), s] for t in NARROW for s in range(7))))
    w('- 判据 (ii) %s：探针 MLP 上 B 相对 A，SIR 7 分类准确率 %s（阈值 +0.02 且 >2SE，%s）；带内 SINR 回归 R² %s（阈值 +0.03 且 >2SE，%s）。对照 C（A+同维度的离循环特征）相对 A 的增量为 SIR 准确率 %s、R² %s，说明增量来自"循环性"本身而不是多了几十维谱统计。'
      % ('满足' if V['ii_ok'] else '不满足', dl('mlp', 'sir', 'A', 'B'), '达标' if V['ii_sir'] else '未达标',
         dl('mlp', 'sinr_in', 'A', 'B'), '达标' if V['ii_sinr'] else '未达标', dl('mlp', 'sir', 'A', 'C'), dl('mlp', 'sinr_in', 'A', 'C')))
    w('')
    w('最关键的 3 个数字：')
    w('')
    w('1. 频带分辨的 α=0.25 相干特征对 ST/MT/PL/NB 在 SIR=-20 dB 的 d\'≈%.0f–%.0f（AUC=1.00）；整窗口循环矩同一格 d\' 只有 %.1f–%.1f（AUC %.2f–%.2f）。'
      % (min(D[ti(t), 0] for t in NARROW), max(D[ti(t), 0] for t in NARROW),
         min(M['whole']['d'][ti(t)][0] for t in NARROW), max(M['whole']['d'][ti(t)][0] for t in NARROW),
         min(M['whole']['auc'][ti(t)][0] for t in NARROW), max(M['whole']['auc'][ti(t)][0] for t in NARROW)))
    w('2. 探针（MLP）SIR 7 分类准确率 A→B：%s → %s；带内 SINR 回归 R²：%s → %s（MAE %.2f → %.2f dB）。'
      % (pm('mlp', 'sir', 'A'), pm('mlp', 'sir', 'B'), pm('mlp', 'sinr_in', 'A'), pm('mlp', 'sinr_in', 'B'), sin_mae('A'), sin_mae('B')))
    w('3. 对宽带 WB 和扫频 LFM，特征在 SIR≤-10 dB 基本失效（WB d\'≤%.1f；LFM d\'=%.1f/%.1f/%.1f @ -20/-15/-10 dB）——这两类恰好是"干扰覆盖信号带边"的类型。'
      % (max(D[ti('WB'), :3]), D[ti('LFM'), 0], D[ti('LFM'), 1], D[ti('LFM'), 2]))
    w('')
    w('通俗地说：这个特征本质上是在信号带边（归一化频率 ±0.125 附近）找"符号率留下的指纹"，并用局部相干系数把干扰功率"除掉"。对窄带干扰，干扰几乎不碰带边，指纹清晰可见；对宽带干扰，带边被淹没，指纹消失。')
    w('')
    w('### 对任务书"理论要点"的核实与修正（以数据为准）')
    w('')
    w('| 理论要点 | 核实结果 |')
    w('|---|---|')
    w('| S^α0(f) 只存在于 α0=0.25，支撑 |f|≤β/8 | **成立**。α=0.25 处特征强（见 §2）；α=0.5 处功率归一化特征 ≈ %.1e，离循环处 ≈ 0.02–0.1，说明 **没有 2 次谐波**：RRC 带宽 (1+β)/8≤0.1875<0.25，S^0.5 的两个谱支不重叠，理论上就是 0，不能当作循环特征用（只能当零假设）。 |' % res['alpha']['a05']['clean_pow'])
    w('| 特征集中在"基带 DC 附近一个窄区域" | **需修正**：|f|≤β/8 是谱相关函数的 **频率变量 f**（两个谱分量的中点），实际取用的谱 bin 是 X(f+0.125) 与 X(f−0.125)，即位于 **信号带边 ±0.125±β/8**（RRC 滚降区），而不是 DC。因此"干扰频率落在 |f|≤β/8 内就被破坏"是错的：被污染的是干扰落在 ±0.125±β/8 的情形（见 §3.3，实测 DC 区命中的样本检测反而不差）。 |')
    w('| 整窗口循环矩在 SIR≤-15 dB 几乎测不出 | **粗查数值复现**（干净 %.3f；混合 %s；样本间标准差 %.3f–%.3f）。但"测不出"需细化：与纯干扰+噪声相比 AUC 仍有 %.2f–%.2f（-20/-15 dB, ST），只是 d\'小（%.1f/%.1f），动态范围被干扰功率稀释；频带分辨 + 相干系数归一化后 d\' 提高一个数量级。 |'
      % (res['ww']['clean'], '/'.join('%.3f' % x for x in res['ww']['noisy_mean'][:5]), min(res['ww']['noisy_std']), max(res['ww']['noisy_std']),
         M['whole']['auc'][0][0], M['whole']['auc'][0][1], M['whole']['d'][0][0], M['whole']['d'][0][1]))
    w('| 共轭版本利用 BPSK 非圆特性 | **成立但需 CFO 搜索**：BPSK 的共轭循环频率是 0.25+2·CFO（|CFO|≤0.002 → 偏移最多 0.004，约 0.5 个 FFT bin），固定在 0.25 时特征几乎消失；沿帧序列做过采样 FFT 在 ±0.004 内搜索后，BPSK 共轭相干系数 ≈ 0.43（β<0.375）/0.83（β≥0.375），QPSK/16QAM 无共轭特征（≈零假设）。 |')
    w('| 窗口 2560 点是否足够 | 足够（见 §5）。 |')
    w('')
    w('### 推荐特征集（后续接入模型用）')
    w('')
    w('- 估计器：Nw=128 汉宁窗、hop=32 的时间平滑谱相关；f 取 |f|≤0.0625 共 17 个 bin（f=j/128, j=-8..8），按 5 个子带分组。')
    w('- α 集合：**α=0.25 非共轭**（主）、**α=0.25 共轭，带 ±0.004 的 CFO 搜索**（BPSK 专用）、**α=0 共轭**（非圆性，BPSK 专用，已含在 B 里）。零假设参照：近邻离循环 α∈{0.1875, 0.21875, 0.28125, 0.3125} 的同一统计量的均值（用来做"减去基线"）。α=0.5 及其他谐波不要用。')
    w('- 每个 (α, 共轭类型) 取 8 维：相干系数相位对齐求和 s_coh、幅度平均 s_inc、功率归一化 s_pow（|ΣS|/(J·平均功率)，这是"信号功率占总功率"的估计）、5 个子带的相干系数幅度；取 log(x+1e-3)。')
    w('- 最终 B_cyc = 3×8（α=0.25 非共轭 / α=0.25 共轭 / α=0 共轭）+ 2×3（离循环基线 s_coh/s_inc/s_pow，非共轭与共轭）+ 2×5（离循环基线子带）+ 4（α=0.5 参照）= **%d 维**；最小版本（仅 α=0.25 非共轭+共轭，16 维）见 §4.3。' % nd[1])
    w('- 只用 s_pow（1 维）也能当粗略 SIR 计：按干扰类型，log s_pow 与带内 SINR 的相关系数 %.2f–%.2f（WB 例外 %.2f），见 §4.2。' % (min(v['corr_pow'] for k, v in res['meter'].items() if k != 'WB'), max(v['corr_pow'] for k, v in res['meter'].items() if k != 'WB'), res['meter']['WB']['corr_pow']))
    w('')

    # ---------------- 1 方法 ----------------
    w('## 1. 方法口径')
    w('')
    w('- 谱相关：时间平滑法。z=I+jQ，Nw∈{64,128}（周期汉宁窗），hop=Nw/4；S(f)=⟨X_k(f+α/2)X_k*(f−α/2)⟩_k（非共轭）、⟨X_k(α/2+f)X_k(α/2−f)⟩_k（共轭）；α=m/Nw（m 取偶数，最近网格）。α 取 0.25、0.5、0、近邻离循环 {0.1875,0.21875,0.28125,0.3125}（≈任务书的 0.19/0.31，加两个更近的）、远离循环 0.40625（≈0.4）。')
    w('- 标量统计量（|f|≤0.0625，J 个 bin）：`s_coh`=|Σ_f S/√(S0(f+)S0(f−))|/J（局部相干系数，相位对齐后相加）；`s_inc`=相干系数幅度的平均；`s_pow`=|Σ_f S|/(J·P̄)，P̄ 为窗内全部 bin 平均功率（功率归一化）。')
    w('- d\'=（α0 处均值 − 同一格内同样本在近邻离循环 α 处的均值）/离循环处标准差；AUC=P(混合信号的 α0 特征 > 纯干扰+噪声（noisy−clean）的 α0 特征)。每格 = 一个干扰类型 × 一个 SIR，合并 3 种调制与 2 种滚降共 120 条。')
    w('- 注意 α=0 的非共轭相干系数按定义恒为 1（只是功率谱），只作对照。干净信号在 α=0.5 处 s_coh 数值上没有意义（分母≈0，数值泄漏），故 α=0.5 一律用 s_pow 看。')
    w('- TF 特征 A：与 `TF_Expert` 同口径——I、Q 两通道分别 torch.stft（n_fft=128, hop=32, 汉宁窗）取幅度，样本先按 RMS 归一，平均池化到 16(频)×8(时)×2(通道)=256 维，取 log。B = A + B_cyc（%d 维）；C（对照）= A + 同形状的 α=0.28125/0.21875 离循环特征（%d 维）。' % (nd[1], nd[2]))
    w('- 探针：torch CPU，MLP（256-128，GELU，dropout 0.2，AdamW，60 epoch）与线性模型（softmax/线性回归）；特征按训练折标准化；超参只依据 A 的设定，B、C 沿用。5 折按 clean_id 分组（本数据每条样本的 clean_id 唯一，等价于普通 5 折）。')
    w('- 回归目标：带内 SINR(dB)=10log10(P_clean_带内 / P_(noisy−clean)_带内)，带内 = |f|≤(1+β)/8（主）；另报全带时域 SINR（补充）。')
    w('')

    # ---------------- 2 Q1 ----------------
    w('## 2. Q1 干净信号的 α0 特征有多强')
    w('')
    w('**结论：α=0.25 的非共轭特征在干净信号上很强，与调制方式无关（QPSK/BPSK/16QAM 一致），只随滚降 β 变化；共轭特征只有 BPSK 有；α=0.5 为 0。**')
    w('')
    rows = []
    for r in res['q1']:
        if r['Nw'] != 128:
            continue
        rows.append([r['mod'], r['beta'], r['n'], '%.2f±%.2f' % (r['nc_coh'], r['nc_coh_sd']), '%.3f' % r['nc_null_coh'], '%.2f' % r['nc_pow'], '%.3f' % r['nc_null_pow'],
                     '%.1e' % r['a05_pow'], '%.2f' % r['cj_coh'], '%.2f' % r['cj_null_coh'], '%.2f' % r['cj_pow']])
    w(md_table(['调制', 'β 区间', 'n', 's_coh@0.25 非共轭', '离循环 s_coh', 's_pow@0.25 非共轭', '离循环 s_pow', 's_pow@0.5', 's_coh@0.25 共轭', '离循环(共轭) s_coh', 's_pow@0.25 共轭'], rows))
    w('')
    w('（Nw=128，干净信号。Nw=64 的结果见 JSON 缓存；s_coh 在 β 小时偏小，是因为固定带 |f|≤0.0625 里只有 |f|≤β/8（0.031~0.0625）有支撑，其余 bin 是 0/0 噪声。）')
    w('')
    w('- 整窗口循环矩 |R(0.25,0)|/总功率：干净 %.3f（与任务书 0.12 一致）。' % res['ww']['clean'])
    w('- 非共轭：β 越大特征越强且宽（β=0.5 时在整个 |f|≤0.0625 内相干系数≈0.85，β=0.25 时只在 ±0.031 内有，平均 0.44）。三种调制数值几乎相同，说明它测的是 **符号率/脉冲成形**，不是调制。')
    w('- 共轭：BPSK 的 α=0.25 共轭特征与非共轭一样强（β≥0.375 时 0.83），QPSK 与 16QAM ≈ 零假设。α=0 共轭 s_pow（非圆性）BPSK ≈ %.1f，其余 ≈ 0.5，可当 BPSK 判别特征。' % res['q1'][2]['cj0_pow'])
    w('')
    w('各 α 上的平均特征（Nw=128，非共轭，全部 test 样本）：')
    w('')
    rows = [[an, '%.3g' % v['clean_pow'], '%.3g' % v['noisy_pow'], '%.3f±%.3f' % (v['noisy_coh'], v['noisy_coh_sd']), '%.3f' % v['neg_coh']]
            for an, v in res['alpha'].items()]
    w(md_table(['α 名', '干净 s_pow', '混合 s_pow', '混合 s_coh', '纯干扰+噪声 s_coh'], rows))
    w('')
    w('（a025=0.25，a05=0.5，a0=0（相干系数恒为 1），off1..4=0.1875/0.21875/0.28125/0.3125，far=0.40625。）')
    w('')

    # ---------------- 3 Q2 ----------------
    w('## 3. Q2 在混合信号上的可检测性')
    w('')
    w('### 3.1 主统计量 d\' / AUC（Nw=128，非共轭 α=0.25，s_coh；合并 3 种调制）')
    w('')
    w('![d\'/AUC 热图](m2_0_fig1_dprime_auc.png)')
    w('')
    w(mat_table(M['primary']['d'], '%.1f', M['primary']['auc']))
    w('')
    w('格式：d\' / AUC。结论（按类型）：')
    w('')
    def rng(t):
        return '%.0f–%.0f' % (D[ti(t)].min(), D[ti(t)].max())
    w('- **ST/MT/PL/NB**（窄带类）：所有 SIR 档 d\'=%s，AUC=1.00（NB 在 -20 dB 略低 %.1f，AUC %.3f）——**SIR 低到 -20 dB 特征依然清晰，且几乎不随 SIR 变化**（干扰集中在带内某处，不碰带边 bin，局部相干系数把它"除掉"）。'
      % ('/'.join(rng(t) for t in NARROW), D[ti('NB'), 0], Au[ti('NB'), 0]))
    w('- **NFM**：与窄带类同级（d\'≥%.0f）；**RFM**：SIR 低时变差（d\'=%.1f @ -20 dB，AUC %.2f），≥-10 dB 后 AUC=1.00。' % (D[ti('NFM')].min(), D[ti('RFM'), 0], Au[ti('RFM'), 0]))
    w('- **LFM**（扫频，周期内扫过整个频带，包括带边）：d\'=%.1f/%.1f/%.1f/%.1f @ -20/-15/-10/-5 dB，AUC=%.2f/%.2f/%.2f/%.2f，≤-15 dB 基本测不出。' % (D[ti('LFM'), 0], D[ti('LFM'), 1], D[ti('LFM'), 2], D[ti('LFM'), 3], Au[ti('LFM'), 0], Au[ti('LFM'), 1], Au[ti('LFM'), 2], Au[ti('LFM'), 3]))
    w('- **WB**（宽带，覆盖全频带）：d\'≈0（AUC≈0.5）直到 SIR=-5 dB（d\'=%.1f）；到 0 dB 才有 d\'=%.1f，与"宽带较差"的预期一致，而且比预期更差——SIR≤-10 dB 完全无效。' % (D[ti('WB'), 3], D[ti('WB'), 4]))
    w('- 窄带类 d\'整体 >20 的数值很大，是因为同一样本的离循环零假设标准差很小（≈0.02），不要把"d\'=25"理解成"某个下游增益"。')
    w('')
    w('### 3.2 其他统计量变体（敏感性）')
    w('')
    for vn, ttl in (('pow128', 'Nw=128, s_pow（功率归一化）'), ('inc128', 'Nw=128, s_inc（不相位对齐）'), ('coh64', 'Nw=64, s_coh'),
                    ('cj128', 'Nw=128 共轭 s_coh（仅 BPSK，40 条/格）'), ('whole', '整窗口循环矩 |R(0.25,0)|/P（零假设取同样本近邻离循环 α）')):
        w('**%s**' % ttl)
        w('')
        w(mat_table(M[vn]['d'], '%.1f', M[vn]['auc']))
        w('')
    w('读法：①功率归一化 s_pow 的 d\'随 SIR 单调变化（窄带类 -20→10 dB 约 1.5→14），因为它的分母是含干扰的总功率——所以 **s_pow 携带 SIR 信息，s_coh 携带"信号是否存在/带边是否干净"的信息**，二者互补；②Nw=64 的 d\'比 128 低 3 倍左右（频率分辨不足，β=0.25 时只有 ±2 个 bin 在支撑内）；③整窗口矩在窄带类 AUC 仍高（≥-15 dB 时 ≥0.96），但 d\'只有 1~4，且 WB 同样失效；④共轭特征在 BPSK 上和非共轭同级，但样本少、对 PL 稍弱。')
    w('')
    w('**分调制的主统计量 d\'（SIR=-20 / -10 dB）**')
    w('')
    rows = []
    for t in TYPES:
        rows.append([t] + ['%.1f / %.1f' % (res['bymod']['%s|%s' % (t, mn)][0][0], res['bymod']['%s|%s' % (t, mn)][2][0]) for mn in MODS])
    w(md_table(['类型'] + MODS, rows))
    w('')
    w('（每格 40 条；调制之间差别不大，说明非共轭 α=0.25 特征对调制方式不敏感。）')
    w('')
    w('### 3.3 干扰频率落区统计（修正后的"污染区"）')
    w('')
    w('两种"区"：**DC 区**=干扰（ST 的单音、MT 任一音、NB 中心±bw/2、PL 中心±2σ_f）与 |f|≤β/8 相交（任务书的假设）；**边缘区**=与 ||f|−0.125|≤β/8+1/Nw 相交（按 SCF 定义真正污染 X(f±0.125) 的位置）。')
    w('')
    rows = [[t, '%.0f%%' % (100 * v['dc']), '%.0f%%' % (100 * v['edge']), '%.0f%%' % (100 * v['either'])] for t, v in res['zone_frac'].items()]
    w(md_table(['类型', '落入 DC 区的样本占比', '落入边缘区的样本占比', '二者之一'], rows))
    w('')
    w('SIR≤-10 dB（-20/-15/-10 合并）时，按是否命中分层的 d\' / AUC（零假设取同一子集）：')
    w('')
    rows = []
    for t in NARROW:
        row = [t]
        for zn in ('edge', 'dc'):
            for hit in ('hit', 'miss'):
                v = res['zone_perf'].get('%s|%s|%s' % (t, zn, hit))
                row.append('—' if v is None else '%.1f / %.3f (n=%d)' % (v['d'], v['auc'], v['n']))
        rows.append(row)
    w(md_table(['类型', '边缘区-命中', '边缘区-未命中', 'DC 区-命中', 'DC 区-未命中'], rows))
    w('')
    w('结论：DC 区命中并 **不会** 破坏特征——命中 DC 区的样本 d\'反而更高，而"DC 区-未命中"（干扰中心远离 DC、更靠近带边）的 d\'更低（NB 11.6 vs 22.5）；边缘区命中同样使 d\'下降（NB 13.8 vs 22.7），但仍远大于 3、AUC≈1。这与"污染区在 ±0.125±β/8 而不是 DC"的修正一致。ST 由数据生成方式（|f|≤0.3·f_max）决定从不落入边缘区；MT/PL/NB 约 50% 的样本会碰到边缘区。注意：测试集 regime A 的干扰频率偏向带中央（≤0.6 f_max），**test_ood（regime B）的 ST 在 0.3~0.6 f_max、MT 最远到 0.95 f_max，更常碰到边缘区**，本实验未评估，是遗留风险。')
    w('')
    w('![coherence vs SIR](m2_0_fig2_coh_vs_sir.png)')
    w('')

    # ---------------- 4 Q3 ----------------
    w('## 4. Q3 增量信息探针（A=TF 幅度 vs B=A+循环特征）')
    w('')
    w('![probe](m2_0_fig3_probe.png)')
    w('')
    w('### 4.1 总表（5 折均值±标准误；括号内为 B−A、C−A 的逐折配对差±标准误）')
    w('')
    rows = []
    names = dict(sir='SIR 7 类（准确率）', mod='调制 3 类（准确率）', typ='干扰类型 8 类（准确率）', sinr_in='带内 SINR 回归（R²）', sinr_full='全带 SINR 回归（R²，补充）')
    for kind, kn in (('mlp', 'MLP'), ('lin', '线性')):
        for tn in ('sir', 'mod', 'typ', 'sinr_in', 'sinr_full'):
            rows.append([kn, names[tn], pm(kind, tn, 'A'), pm(kind, tn, 'B'), pm(kind, tn, 'C'), dl(kind, tn, 'A', 'B'), dl(kind, tn, 'A', 'C'),
                         ('%.2f→%.2f' % (P['%s|%s|A' % (kind, tn)]['mae'], P['%s|%s|B' % (kind, tn)]['mae'])) if tn.startswith('sinr') else ''])
    w(md_table(['模型', '任务', 'A', 'B', 'C(对照)', 'B−A', 'C−A', 'MAE(dB) A→B'], rows))
    w('')
    w('判据 (ii)（MLP）：SIR 分类 B−A=%+.3f±%.3f，%s；带内 SINR R² B−A=%+.3f±%.3f，%s。' % (V['sir'][0], V['sir'][1], '达标' if V['ii_sir'] else '未达标', V['sinr'][0], V['sinr'][1], '达标' if V['ii_sinr'] else '未达标'))
    w('')
    # 分层
    y_sir, y_mod, y_typ, y_sin = pr['y_sir'], pr['y_mod'], pr['y_typ'], pr['y_sinr_in']
    w('### 4.2 按 SIR 档分层（MLP，交叉验证预测汇总）')
    w('')
    rows = []
    for s in range(7):
        k = y_sir == s
        r = [SIRS[s]]
        for fs in 'AB':
            r.append('%.3f' % (pr['mlp__sir__' + fs][k] == s).mean())
        for fs in 'AB':
            r.append('%.2f' % np.abs(pr['mlp__sinr_in__' + fs] - y_sin)[k].mean())
        for fs in 'AB':
            r.append('%.3f' % (pr['mlp__mod__' + fs][k] == y_mod[k]).mean())
        for fs in 'AB':
            r.append('%.3f' % (pr['mlp__typ__' + fs][k] == y_typ[k]).mean())
        rows.append(r)
    w(md_table(['SIR(dB)', 'SIR档召回 A', 'B', 'SINR MAE(dB) A', 'B', '调制准确率 A', 'B', '类型准确率 A', 'B'], rows))
    w('')
    w('![mae](m2_0_fig4_sinr_mae_by_sir.png)')
    w('')
    w('按干扰类型分层的带内 SINR MAE（dB，MLP）与"单特征 SIR 计"（对 log s_pow、log s_coh、β 的线性拟合，仅循环特征，无 TF）：')
    w('')
    rows = []
    for t in range(8):
        k = y_typ == t
        rows.append([TYPES[t], '%.2f' % np.abs(pr['mlp__sinr_in__A'] - y_sin)[k].mean(), '%.2f' % np.abs(pr['mlp__sinr_in__B'] - y_sin)[k].mean(),
                     '%.2f' % res['meter'][TYPES[t]]['mae'], '%.3f' % res['meter'][TYPES[t]]['r2']])
    w(md_table(['类型', 'MAE A', 'MAE B', '线性 SIR 计 MAE', '线性 SIR 计 R²'], rows))
    w('')
    w('（线性 SIR 计在 test 集内拟合，只用于说明信息量，不是泛化指标。）')
    w('')
    w('### 4.3 补充消融（MLP，同样 5 折）')
    w('')
    if ex:
        rows = []
        for tn, nm in (('sir', 'SIR 7 类准确率'), ('sinr_in', '带内 SINR R²')):
            rows.append([nm, pm('mlp', tn, 'A'), pm('mlp', tn, 'B'),
                         '%.3f±%.3f' % (np.mean(ex['mlp|%s|Bmin' % tn]['folds']), np.std(ex['mlp|%s|Bmin' % tn]['folds'], ddof=1) / np.sqrt(5)),
                         '%.3f±%.3f' % (np.mean(ex['mlp|%s|Bonly' % tn]['folds']), np.std(ex['mlp|%s|Bonly' % tn]['folds'], ddof=1) / np.sqrt(5))])
        w(md_table(['任务', 'A', 'B (A+%d维)' % nd[1], 'Bmin (A+16维：α=0.25 非共轭+共轭)', 'Bonly (仅 %d 维循环特征，无 TF)' % nd[1]], rows))
    w('')

    # ---------------- 5 Q5 ----------------
    w('## 5. Q5 稳定性')
    w('')
    w('**样本内变异（变异系数 CV=std/mean，主统计量，同一 类型×调制×SIR 内，3 种调制平均）**')
    w('')
    rows = []
    for t in TYPES:
        rows.append([t] + ['%.2f' % res['cv_noisy']['%s|%d' % (t, s)] for s in (-20, -10, 0)])
    w(md_table(['类型', 'SIR=-20', 'SIR=-10', 'SIR=0'], rows))
    w('')
    w('干净信号的 CV：' + '；'.join('%s β≥%s：%.2f' % (k.split('|')[0], k.split('|')[1], v) for k, v in res['cv_clean'].items()) + '。')
    w('窄带类 CV≈0.2~0.3，与干净信号的 CV（0.1~0.29）同量级，**主要来自滚降 β 在 0.25~0.5 之间随机取值**（β 小时固定带内的有效 bin 少），不是噪声；WB、LFM、RFM 在低 SIR 时 CV 变大（0.5~1.0）是特征被淹没的表现。')
    w('')
    w('**窗口长度：前 1280 点 / 后 1280 点 / 全 2560 点（Nw=128，主统计量）**')
    w('')
    rows = []
    for k, v in res['window'].items():
        t, s = k.split('|')
        if t in ('ST', 'MT', 'PL', 'NB', 'LFM', 'WB', 'RFM'):
            rows.append([t, s] + ['%.4f' % v['full']['null_sd'], '%.4f' % v['h1']['null_sd'], '%.1f / %.2f' % (v['full']['d'], v['full']['auc']),
                                  '%.1f / %.2f' % (v['h1']['d'], v['h1']['auc']), '%.1f / %.2f' % (v['h2']['d'], v['h2']['auc'])])
    w(md_table(['类型', 'SIR', '零假设 sd 2560', '零假设 sd 1280(前)', "d'/AUC 2560", "d'/AUC 前1280", "d'/AUC 后1280"], rows))
    w('')
    w('零假设标准差从 2560 点到 1280 点放大 %.2f 倍（≈√2=1.41，符合 1/√点数 的估计方差规律）；窄带类 d\' 从 ~22 降到 ~15，仍远高于 3。**2560 点足够**，甚至 1280 点也够（但会损失约 30%% 的 d\'）。' % res['null_sd_ratio'])
    w('')

    # ---------------- 6 判定 ----------------
    w('## 6. 判定与理由')
    w('')
    w('| 判据 | 要求 | 实测 | 结果 |')
    w('|---|---|---|---|')
    w('| (i) 窄带型干扰 SIR≤-10 dB 某档 d\'≥3 | ≥3 | 最好格 %s@%d dB: %.1f；窄带型全部格最小值 %.1f | %s |'
      % (V['best'][1], V['best'][2], V['best'][0], min(D[ti(t), s] for t in NARROW for s in range(3)), '满足' if V['i_ok'] else '不满足'))
    w('| (ii-a) SIR 分类 B−A（MLP） | ≥+0.02 且 >2SE | %+.3f±%.3f | %s |' % (V['sir'][0], V['sir'][1], '满足' if V['ii_sir'] else '不满足'))
    w('| (ii-b) 带内 SINR R² B−A（MLP） | ≥+0.03 且 >2SE | %+.3f±%.3f | %s |' % (V['sinr'][0], V['sinr'][1], '满足' if V['ii_sinr'] else '不满足'))
    w('| **总判定** | (i) 且 ((ii-a) 或 (ii-b)) | | **%s** |' % verdict_txt)
    w('')
    w('@@NARRATIVE@@')
    w('')
    w('## 7. 复现与局限')
    w('')
    w('- 复现：`/root/autodl-tmp/envs/moediff_bw/bin/python new/analysis/m2_0_cyclo.py compute`（特征+整窗口矩+探针+补充消融，约 30 分钟，4 线程，机器负载高时更久）→ `... report`；画图用 `/root/miniconda3/bin/python new/analysis/m2_0_cyclo.py plot`（moediff_bw 没有 matplotlib；系统里没有中文字体，图内文字为英文）。中间缓存在 `new/reports/m2_0_cache/`。')
    w('- 未抽样：test 6720 条全部使用，每格 40 条（合并调制后每个 类型×SIR 格 120 条）；探针用 train+test 13440 条。')
    w('- 局限：①只评估了 regime A 的 test，未评估 test_ood；②探针是"单样本、信息量"测试，不是端到端抑制性能——B 对 A 的增量说明特征含 SIR 相关的新信息，不保证接入模型后 BER 改善；③TF 特征 A 被压缩到 256 维并取 log，真实 TF 专家的 CNN 可能提取到更多；④线性 SIR 计在 test 内拟合，仅作信息量说明。')
    open(os.path.join(REP, 'm2_0_cyclo_analysis.md'), 'w').write('\n'.join(L).replace('@@NARRATIVE@@', NARRATIVE))
    print('report written')


NARRATIVE = """### 6.1 怎么读这个"通过"

1. **(i) 是压倒性的满足，(ii) 只是"部分满足"**：SIR 分类 +0.067（MLP，线性模型 +0.238）远超 +0.02；但带内 SINR 回归 R² 只 +0.020（MLP），低于 +0.03 的阈值（线性模型 +0.155 远超）。MLP 的 A 基线本来就很强（R²=0.957，MAE 1.39 dB），所以 R² 的天花板效应明显；MAE 仍从 1.39 降到 1.02 dB（-27%），在 SIR=-5~5 dB 档降幅最大（-33%~-45%）。判据按"或"关系，所以正式结论是通过。
2. **增量确实来自"循环性"**：对照 C（同样多的离循环特征）对 A 几乎没有增量（SIR 准确率 -0.013，R² -0.001，MLP），说明 B 的收益不是"多给了几十维谱统计"；仅 44 维循环特征（Bonly，不含 TF）的 SIR 准确率（0.912）和 SINR R²（0.973）就已超过 256 维 TF 幅度特征 A（0.855 / 0.957）。16 维的最小版本 Bmin 与完整 B 几乎无差别（0.918 vs 0.922；0.977 vs 0.978），所以接入模型时 16 维就够。
3. **特征的两种信息要分开看**：(a) 相干系数 s_coh/s_inc 回答"信号带边是否被干扰覆盖"，对窄带干扰与 SIR 几乎无关（d\'≈20+，从 -20 到 +10 dB 不变），对 WB/LFM/RFM 随 SIR 单调下降——这是"可检测性"；(b) 功率归一化 s_pow 的分母含干扰总功率，所以与带内 SINR 的相关系数 0.95–0.97（WB 0.84）——这是 SIR 计。探针的收益主要来自 (b) 与 (a) 的组合，不是"检测到信号"本身。
4. **调制识别也受益**（MLP 0.507→0.648，线性 0.345→0.647）：来自 BPSK 的共轭特征与 α=0 非圆性，QPSK 与 16QAM 之间没有增量（二者都没有二阶循环差异）。这对 M2 的主任务不是重点。

### 6.2 风险与不足（建议在接入前知道）

- **WB 与 LFM 在 SIR≤-10 dB 完全无效**（d\'≤0.7 / ≤7.5，AUC 0.5~0.77）。这两类恰好是低 SIR 下最难的类型；循环特征不能帮助它们的"检测"，探针里对 WB 的 SINR 回归虽有改善（MAE 2.69→1.98 dB），推测来自 s_pow 的整体功率信息而非循环指纹。如果 M2 的价值主张是"在最难的宽带/扫频干扰上给出额外先验"，这个特征做不到。
- **只测了 regime A 的 test**。test_ood（regime B）的干扰更频繁落在 ±0.125 带边，特征可能退化，需要在接入前补测（用同一脚本换 split 即可）。
- **探针不是端到端**：B 对 A 的增量只证明"特征含有 TF 幅度特征没有（或难以）提取的 SIR 信息"。TF 专家的 CNN 可能能从原始 STFT 学到类似信息；建议在 M2 的第一个模型实验里直接用 Bmin（16 维）作为额外条件输入，与不加对比。
- 交叉验证标准误（逐折）会低估真实不确定度（各折训练集高度重叠），但本实验的主增量（SIR 分类 +0.067、线性模型 +0.15~0.30）比标准误大一个数量级，结论不依赖于此。
- 共轭特征需要 ±0.004 的 CFO 搜索，数据 CFO 上限 0.002（`CFO_MAX`）；若实际系统 CFO 更大需放宽搜索范围，零假设的均值也会随之抬高。
"""

if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'compute'
    if st == 'compute':
        stage_features(); stage_ww(); stage_probe(); stage_probe_extra()
    elif st == 'features':
        stage_features()
    elif st == 'probe':
        stage_probe(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif st == 'ww':
        stage_ww()
    elif st == 'probe_extra':
        stage_probe_extra()
    elif st == 'report':
        stage_report()
    elif st == 'plot':
        stage_plot()
