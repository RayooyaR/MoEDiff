"""加载器测试：  python test_loader.py [数据目录，默认 ../data/v2]    （需要 torch；CPU 即可）"""
import os
import sys
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', message='CUDA initialization')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'gen'))
from moe_dataset import MoEDiffDataset  # noqa: E402
import config as C  # noqa: E402
import signal_gen as SG  # noqa: E402

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'v2')
ok = True


def check(name, cond, extra=''):
    global ok
    ok &= bool(cond)
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, extra))


# 1 规模与形状
expect = {'train': 6720, 'val': 840, 'test': 6720, 'test_ood': 3360}
sets = {}
for sp, n in expect.items():
    t0 = time.time()
    sets[sp] = MoEDiffDataset(ROOT, sp)
    check('split %s 条数' % sp, len(sets[sp]) == n, '(%d, 加载 %.1fs)' % (len(sets[sp]), time.time() - t0))
ds = sets['test']
x = ds[0]
check('字段与形状', x['clean'].shape == (2, 2560) and x['noisy'].shape == (2, 2560) and x['symbols'].shape == (2, 640)
      and x['demod'].shape == (5,) and x['clean'].dtype == torch.float32 and x['interference_type'].dtype == torch.long)
check('类型/调制名', ds.type_names == C.TYPES and ds.mod_names == C.MODS, str(ds.type_names))
check('每个(类型,调制)均衡', set(ds.summary().values()) == {7 * 40}, str(set(ds.summary().values())))

# 2 与原始 npz 逐点一致
import glob  # noqa: E402
f = sorted(glob.glob(os.path.join(ROOT, 'test', 'LFM', '16QAM', '*.npz')))[3]
z = np.load(f)
sub = MoEDiffDataset(ROOT, 'test', types=['LFM'], mods=['16QAM'], sirs=[float(z['sir_db_target'][0])])
check('过滤后条数 = 单文件条数', len(sub) == len(z['clean_signal']), str(len(sub)))
i = 7
a = sub[i]
check('clean 与 npz 一致', np.allclose(a['clean'][0].numpy(), z['clean_signal'][i].real) and np.allclose(a['clean'][1].numpy(), z['clean_signal'][i].imag))
check('noisy(with) 与 mixture 一致', np.allclose(a['noisy'][0].numpy(), z['mixture_signal'][i].real) and np.allclose(a['noisy'][1].numpy(), z['mixture_signal'][i].imag))

# 3 noise_mode
wo = MoEDiffDataset(ROOT, 'test', types=['LFM'], mods=['16QAM'], sirs=[float(z['sir_db_target'][0])], noise_mode='without')
d = (sub[i]['noisy'] - wo[i]['noisy']).numpy()
nz = np.stack([z['noise_signal'][i].real, z['noise_signal'][i].imag])
check('with - without = noise', np.allclose(d, nz, atol=1e-5))
check('without = clean + interference', np.allclose(wo[i]['noisy'][0].numpy(), (z['clean_signal'][i] + z['interference_signal'][i]).real, atol=1e-5))

# 4 过滤器 / max_per_cell 嵌套 / 错误处理
m = MoEDiffDataset(ROOT, 'val', types=['PL', 'RFM'], mods=['BPSK'], sirs=[-20, 10])
check('类型×调制×SIR 过滤', len(m) == 2 * 1 * 2 * 5, str(len(m)))
a3 = MoEDiffDataset(ROOT, 'train', max_per_cell=3, seed=1)
a5 = MoEDiffDataset(ROOT, 'train', max_per_cell=5, seed=1)
id3, id5 = {s['clean_id'] for s in a3.samples}, {s['clean_id'] for s in a5.samples}
check('max_per_cell 嵌套且条数正确', len(a3) == 3 * 168 and len(a5) == 5 * 168 and id3 <= id5)
a3b = MoEDiffDataset(ROOT, 'train', max_per_cell=3, seed=1)
check('固定种子可复现', [s['clean_id'] for s in a3.samples] == [s['clean_id'] for s in a3b.samples])
for name, fn in (('未知 split', lambda: MoEDiffDataset(ROOT, 'nope')),
                 ('未知类型', lambda: MoEDiffDataset(ROOT, 'val', types=['XX'])),
                 ('空选择', lambda: MoEDiffDataset(ROOT, 'val', sirs=[99])),
                 ('错误 noise_mode', lambda: MoEDiffDataset(ROOT, 'val', noise_mode='x'))):
    try:
        fn()
        check('报错: ' + name, False)
    except (ValueError, FileNotFoundError):
        check('报错: ' + name, True)

# 5 DataLoader（多进程、打乱）
dl = DataLoader(sets['train'], batch_size=32, shuffle=True, num_workers=2)
b = next(iter(dl))
check('DataLoader batch', b['clean'].shape == (32, 2, 2560) and b['symbols'].shape == (32, 2, 640) and b['demod'].shape == (32, 5))
check('batch 数值有限', all(torch.isfinite(v.float()).all() for v in b.values()))

# 6 经加载器取出的数据做解调：干净信号 BER 必须为 0，且与直接读 npz 的结果一致
def ber_from_item(item, key):
    r = item[key][0].numpy().astype(np.complex128) + 1j * item[key][1].numpy()
    mod = ds.mod_names[int(item['modulation'])]
    rolloff, tau, cfo, phase, gain = [float(v) for v in item['demod']]
    sym = item['symbols'][0].numpy() + 1j * item['symbols'][1].numpy()
    bits = SG.symbols_to_bits(sym, mod)
    return SG.ber_interior(SG.demod_oracle(r, mod, rolloff, tau, cfo, phase, gain), bits, mod)

idx = np.random.RandomState(0).choice(len(ds), 60, replace=False)
check('经加载器的干净信号 BER = 0', max(ber_from_item(ds[int(k)], 'clean') for k in idx) == 0.0)
f2 = sorted(glob.glob(os.path.join(ROOT, 'test', 'ST', 'QPSK', '*neg05dB.npz')))[0]   # SIR = -5 dB，BER 明显不为 0
z2 = np.load(f2)
s2 = MoEDiffDataset(ROOT, 'test', types=['ST'], mods=['QPSK'], sirs=[-5])
j = 5
direct = SG.ber_interior(SG.demod_oracle(z2['mixture_signal'][j], 'QPSK', float(z2['rolloff'][j]), float(z2['timing_offset'][j]),
                                         float(z2['cfo'][j]), float(z2['phase'][j]), float(z2['demod_gain'][j])), z2['bit_stream'][j], 'QPSK')
check('加载器路径与直接读 npz 的 BER 一致且非零', abs(ber_from_item(s2[j], 'noisy') - direct) < 1e-9 and direct > 0.1, '(%.4f)' % direct)

# 7 split 之间 clean_id 不相交
ids = {sp: {s['clean_id'] for s in d_.samples} for sp, d_ in sets.items()}
check('各 split clean_id 不相交', all(not (ids[a_] & ids[b_]) for a_ in ids for b_ in ids if a_ < b_))

print('\n全部通过' if ok else '\n存在失败项')
sys.exit(0 if ok else 1)
