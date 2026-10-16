"""清理某个 profile 在本机产生的 run / 状态 / 日志（用于 smoke 冒烟后恢复干净状态，避免污染后续真实实验）。
用法: python exp/clean_profile.py --profile smoke [--yes]"""
import argparse
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', required=True)
    ap.add_argument('--yes', action='store_true')
    a = ap.parse_args()
    names = set()
    for M in 'ABC':
        f = os.path.join(HERE, 'plans', '%s_%s.json' % (a.profile, M))
        if os.path.exists(f):
            for j in json.load(open(f))['jobs']:
                names.add(j['name'])
                for m in re.finditer(r'--(?:exp_name|name) (\w+)', j['cmd']):          # 只删本计划“产出”的 run；--run/--runs/--cnn_run 是输入（可能是既有实验），绝不删除
                    names.add(m.group(1))
    names |= {'base_zs2', 'base_zs2_oof.npz'} | {'base_zs2_f%d' % i for i in range(8)}
    protected = {n for n in os.listdir(os.path.join(ROOT, 'runs')) if n.startswith(('zs', 'fs_', 'exp_', 'ef_', 'ccir_', 'v4_', 'hard', 'ab_', 'smk_'))} - {n for n in names}
    PROTECT = ('zs', 'fs_', 'exp_', 'ef_', 'ccir_', 'v4_', 'hard', 'ab_', 'l1_', 'l2')
    todo = [n for n in names if os.path.exists(os.path.join(ROOT, 'runs', n)) and not n.startswith(PROTECT)]
    print('将删除 runs 下 %d 个目录/文件（受保护的既有实验不会被触及）' % len(todo))
    if not a.yes:
        print('加 --yes 执行'); return
    for n in todo:
        p = os.path.join(ROOT, 'runs', n)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for sub in ('exp_state', os.path.join('logs', 'jobs')):
        p = os.path.join(ROOT, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    import glob as _g
    extra = [os.path.basename(p) for p in _g.glob(os.path.join(ROOT, '..', 'reports', 'expert_impact_*.csv')) + _g.glob(os.path.join(ROOT, '..', 'reports', 'gate_diag_*.txt'))]
    for f in ['e12_val.json', 'e12_test.json', 'e13_sampling_ablation.csv', 'baseline_audit.md', 'experiment_index.csv', 'e14_aug_stats.csv', 'model_costs.csv', 'plan_report.md'] + extra:
        p = os.path.join(ROOT, '..', 'reports', f)
        if os.path.exists(p) and a.profile == 'smoke':
            os.remove(p)
    print('已清理')


if __name__ == '__main__':
    main()
