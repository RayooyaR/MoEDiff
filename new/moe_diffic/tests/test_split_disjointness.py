"""train / val / test / test_ood 的 clean_id 两两不相交（逐数据集检查）。"""
import glob, os, sys
import numpy as np
from _common import *
DATA = os.path.abspath(os.path.join(ROOT, '..', 'data'))
n_ds = 0
for d in sorted(os.listdir(DATA)):
    p = os.path.join(DATA, d)
    if not os.path.isfile(os.path.join(p, 'metadata.json')):
        continue
    ids = {}
    for sp in ('train', 'val', 'test', 'test_ood'):
        fs = glob.glob(os.path.join(p, sp, '*', '*', '*.npz'))
        if not fs:
            continue
        s = set()
        for f in fs:
            s |= set(np.load(f)['clean_id'].tolist())
        ids[sp] = s
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = ids[names[i]] & ids[names[j]]
            assert not inter, '%s: %s 与 %s 的 clean_id 有 %d 个重叠' % (d, names[i], names[j], len(inter))
    n_ds += 1
    print('  %-14s ' % d + ' '.join('%s=%d' % (k, len(v)) for k, v in ids.items()) + ' 互不相交')
print('PASS test_split_disjointness (%d 个数据集)' % n_ds)
