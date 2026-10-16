"""把本机的结果打包成小文件（不含权重）：每个 run 的 args/summary/log/per_sample/gate 等，reports、exp_state、logs/jobs。
用法: python exp/pack_results.py --machine B [--profile real]  -> /root/autodl-tmp/exp_results_<机器>_<profile>.tar.gz
拷到汇总机器后: python exp/merge_results.py /path/to/exp_results_*.tar.gz && python exp/final_report.py"""
import argparse
import glob
import json
import os
import re
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
KEEP_EXT = ('.json', '.npz', '.log', '.csv', '.pt', '.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', required=True)
    ap.add_argument('--profile', default='real')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    plan = json.load(open(os.path.join(HERE, 'plans', '%s_%s.json' % (a.profile, a.machine))))
    names = set()
    for j in plan['jobs']:
        for m in re.finditer(r'--(?:exp_name|runs|run) (\w+)', j['cmd']):
            names.add(m.group(1))
        for m in re.finditer(r'--name (\w+)', j['cmd']):
            names.add(m.group(1))
        names.add(j['name'])
    extra = {'base_zs2', 'base_zs2_f0', 'base_zs2_f1', 'base_zs2_f2', 'base_zs2_f3', 'base_zs2_f4'}
    out = a.out or os.path.join(os.path.dirname(ROOT), '..', 'exp_results_%s_%s.tar.gz' % (a.machine, a.profile))
    out = os.path.abspath(out)
    n = 0
    with tarfile.open(out, 'w:gz') as tf:
        for r in sorted(names | extra):
            d = os.path.join(ROOT, 'runs', r)
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, '*')):
                if os.path.isfile(f) and f.endswith(KEEP_EXT) and os.path.basename(f) != 'best_model.pth' and not os.path.basename(f).startswith('events.out'):
                    tf.add(f, arcname=os.path.relpath(f, os.path.dirname(ROOT))); n += 1
        for f in glob.glob(os.path.join(ROOT, 'runs', 'base_zs2_oof.npz')):
            tf.add(f, arcname=os.path.relpath(f, os.path.dirname(ROOT)))
        for sub in ('exp_state/%s' % a.machine,):
            for f in glob.glob(os.path.join(ROOT, sub, '**', '*'), recursive=True):
                if os.path.isfile(f):
                    tf.add(f, arcname=os.path.relpath(f, os.path.dirname(ROOT)))
        for f in glob.glob(os.path.join(ROOT, 'exp_state', 'calib_%s.json' % a.machine)) + glob.glob(os.path.join(ROOT, 'exp_state', 'sys_tests_%s.log' % a.machine)):
            tf.add(f, arcname=os.path.relpath(f, os.path.dirname(ROOT)))
        for f in glob.glob(os.path.join(ROOT, 'logs', 'jobs', '*.log')):
            tf.add(f, arcname=os.path.relpath(f, os.path.dirname(ROOT)))
        for f in glob.glob(os.path.join(os.path.dirname(ROOT), 'reports', '*')):
            if os.path.isfile(f):
                tf.add(f, arcname=os.path.relpath(f, os.path.dirname(ROOT)))
    print('打包 %d 个 run 文件 -> %s (%.1f MB)' % (n, out, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
