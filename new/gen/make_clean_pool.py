"""干净通信波形池（自监督/无条件预训练用）：直接调用项目原生成器 signal_gen.make_clean(clean_id, mod)，只输出干净信号，不含干扰与噪声，不构造配对。

  - 与数据集的干净目标完全同分布：同一函数（BPSK/QPSK/16QAM 映射、RRC 滚降 U(0.25,0.5)、定时 U(0,1)、频偏 ±0.002、随机初相、单位功率）。
  - clean_id 从 ID_BASE=1,000,000 起，与 train(0..)/val(100000..)/test(200000..)/test_ood(300000..) 的 clean_id 区段互不重叠，
    且 make_clean 由 (SEED, 7, clean_id) 唯一决定波形，所以与任何验证/测试波形不会重复。
  - 调制按 BPSK/QPSK/16QAM 轮流，三种各占 1/3。
用法: python gen/make_clean_pool.py --n 10000 --out data/clean_pool/pool_10k.npz [--check]
输出 npz：x (N,2,2560) float32（第 0 行 I、第 1 行 Q）、mod (N,) 调制索引（顺序见 mods）、clean_id、beta、tau、cfo、phase。
存储：每条 2×2560×4B=20 KB，1 万条≈205 MB，3 万条≈614 MB。
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import config as C
import signal_gen as SG

ID_BASE = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--id_base', type=int, default=ID_BASE)
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'data', 'clean_pool', 'pool_10k.npz'))
    ap.add_argument('--check', action='store_true', help='与训练集干净目标做分布一致性检查')
    a = ap.parse_args()
    assert a.id_base >= 1_000_000, 'id_base 必须避开已有划分的 clean_id 区段'
    mods = list(C.MODS)
    x = np.zeros((a.n, 2, C.SIG_LEN), np.float32); meta = {k: np.zeros(a.n, np.float64) for k in ('beta', 'tau', 'cfo', 'phase')}
    mi = np.zeros(a.n, np.int8); cid = np.arange(a.id_base, a.id_base + a.n)
    for i in range(a.n):
        m = mods[i % len(mods)]
        s = SG.make_clean(int(cid[i]), m)
        x[i, 0], x[i, 1] = s['signal'].real, s['signal'].imag
        mi[i] = i % len(mods)
        for k in meta: meta[k][i] = s[k]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, x=x, mod=mi, mods=np.array(mods), clean_id=cid, **meta)
    p = (x[:, 0] ** 2 + x[:, 1] ** 2).mean(1)
    print('已生成 %d 条 -> %s  (%.0f MB)  单位功率均值 %.4f(±%.4f)  I/Q 标准差 %.4f' % (a.n, a.out, os.path.getsize(a.out) / 1e6, p.mean(), p.std(), x.std()))
    if a.check:
        sys.path.insert(0, os.path.join(HERE, '..', 'loader'))
        from moe_dataset import MoEDiffDataset
        ds = MoEDiffDataset(os.path.join(HERE, '..', 'data', 'v2'), 'train', noise_mode='with', types=['ST'])
        ref = np.stack([s['clean'] for s in ds.samples[:1200]]).astype(np.float32)
        def psd(z): return (np.abs(np.fft.fft(z[:, 0] + 1j * z[:, 1], axis=-1)) ** 2).mean(0) / z.shape[-1]
        pr, pn = psd(ref), psd(x[:2000])
        print('[分布检查] 平均功率谱最大相对偏差 %.3f（越小越好）；I/Q 标准差 池 %.4f 对 训练集 %.4f；峰值因子分位数(池/训练集) 50%%: %.2f/%.2f  99%%: %.2f/%.2f' % (
            np.abs(pn - pr).max() / pr.max(), x.std(), ref.std(),
            *[v for q in (50, 99) for v in (np.percentile(np.abs(x[:, 0] + 1j * x[:, 1]).max(1) / np.sqrt(p), q), np.percentile(np.abs(ref[:, 0] + 1j * ref[:, 1]).max(1), q))]))
        ids = set(range(0, 400000)); print('[隔离检查] 池 clean_id 区间 [%d, %d]，与 train/val/test/test_ood 区段(<400000)重叠: %s' % (cid.min(), cid.max(), bool(ids & set(cid.tolist()))))


if __name__ == '__main__':
    main()
