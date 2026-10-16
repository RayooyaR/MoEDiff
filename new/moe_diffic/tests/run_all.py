"""运行全部单元测试与回归基准。用法: python tests/run_all.py [--fast]（--fast 跳过 32 变体冒烟）"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
tests = ['test_tensor_shapes.py', 'test_router_masks.py', 'test_no_oracle_leakage.py', 'test_split_disjointness.py', 'test_augmentation_physics.py', 'test_sampler_reproducibility.py', 'test_restormer.py', 'test_mrstft.py', 'test_cyclo_expert.py', 'test_pretrain_init.py', 'test_local_cond.py']
bad = []
for t in tests:
    r = subprocess.run([py, '-W', 'ignore', os.path.join(HERE, t)], capture_output=True, text=True, cwd=HERE)
    line = (r.stdout.strip().splitlines() or ['(无输出)'])[-1]
    print('%-34s %s  %s' % (t, 'OK ' if r.returncode == 0 else 'FAIL', line if r.returncode == 0 else (r.stderr.strip().splitlines() or ['?'])[-1][:200]))
    if r.returncode:
        bad.append(t)
g = os.path.join('/root/autodl-tmp/bench', 'golden.py')
if os.path.exists(g):
    r = subprocess.run([py, '-W', 'ignore', g, 'check'], capture_output=True, text=True)
    print('golden 回归                         ', (r.stdout.strip().splitlines() or ['?'])[-1])
    if 'OK' not in r.stdout:
        bad.append('golden')
if '--fast' not in sys.argv:
    r = subprocess.run([py, '-W', 'ignore', os.path.join(HERE, 'smoke_variants.py')], capture_output=True, text=True, cwd=HERE)
    print('smoke_variants                      ', (r.stdout.strip().splitlines() or ['?'])[-1])
    if 'ALL OK' not in r.stdout:
        bad.append('smoke_variants')
print('ALL TESTS PASS' if not bad else 'FAILED: %s' % bad)
sys.exit(1 if bad else 0)
