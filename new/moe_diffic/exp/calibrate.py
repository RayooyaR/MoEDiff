"""校准（约 5-8 分钟）：实测本机的单任务速度与 MPS 并行收益，给出推荐的 --slots 和 --unit_seconds，写入 exp_state/calib_<机器>.json。
参照值（本项目开发机 RTX PRO 6000 实测）：MoE bf16 单进程 34.4 步/秒，3 进程 MPS 聚合 42.1 步/秒；对应每卡每小时约 3.86 个代价单位。
用法: python exp/calibrate.py --machine B"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
REF_SINGLE, REF_AGG3, REF_UNITS = 34.4, 42.1, 3.86


def run_conc(n, py, mps, run='zs2_ex_TF', steps=150):
    start = time.time() + 20
    env = dict(os.environ, START_AT=str(start), CUDA_VISIBLE_DEVICES='0')
    if mps:
        env.update(CUDA_MPS_PIPE_DIRECTORY='/tmp/mps_pipe_g0', CUDA_MPS_LOG_DIRECTORY='/tmp/mps_log_g0')
    ps = [subprocess.Popen([py, '-W', 'ignore', os.path.join(HERE, 'bench', 'step.py'), run, 'bf16', str(steps)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True) for _ in range(n)]
    ms = []
    for p in ps:
        out = p.communicate()[0].strip().splitlines()
        if out and 'ms/step' in out[-1]:
            ms.append(float(out[-1].split()[-2]))
    return sum(1000.0 / m for m in ms), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', required=True)
    ap.add_argument('--no_mps', action='store_true')
    a = ap.parse_args()
    py = os.environ.get('PY') or sys.executable
    single, ms1 = run_conc(1, py, not a.no_mps)
    agg3, ms3 = run_conc(3, py, not a.no_mps)
    gain = agg3 / single if single else 0
    slots = 3 if gain >= 1.12 and not a.no_mps else 1
    thr = agg3 if slots == 3 else single
    units_per_hour = REF_UNITS * thr / REF_AGG3
    unit_seconds = 3600.0 * slots / units_per_hour
    res = dict(machine=a.machine, single_steps_s=round(single, 1), agg3_steps_s=round(agg3, 1), mps_gain=round(gain, 2), slots=slots,
               units_per_hour_per_gpu=round(units_per_hour, 2), unit_seconds=round(unit_seconds), ms_single=ms1, ms_3=ms3)
    os.makedirs(os.path.join(ROOT, 'exp_state'), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, 'exp_state', 'calib_%s.json' % a.machine), 'w'), indent=1)
    print(json.dumps(res, indent=1))
    print('\n推荐: --slots %d --unit_seconds %d' % (slots, unit_seconds))
    print('MPS 并行收益 %.2fx（参照机 %.2fx）；每卡约 %.2f 个代价单位/小时（参照机 %.2f）' % (gain, REF_AGG3 / REF_SINGLE, units_per_hour, REF_UNITS))
    print('按 %s 台 12 小时窗口估算：必做负载占比 ≈ 计划代价 / (卡数 × %.2f × 11 × 0.9)' % (a.machine, units_per_hour))


if __name__ == '__main__':
    main()
