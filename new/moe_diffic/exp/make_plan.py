"""生成任务计划 JSON：exp/plans/<profile>_<A|B|C>.json，并打印每台机器的负载（旧机 fp32 小时单位）与对应窗口容量。
用法: python exp/make_plan.py [--profile real|smoke]"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import registry as REG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='real')
    ap.add_argument('--speed', type=float, default=3.86, help='每张卡每小时可处理的代价单位（bf16+MPS 实测推算≈3.86）')
    ap.add_argument('--window_h', type=float, default=11.0)
    a = ap.parse_args()
    jobs = REG.build(a.profile)
    names = {j['name'] for j in jobs}
    for j in jobs:
        for d in j['deps']:
            assert d in names, '依赖不存在: %s -> %s' % (j['name'], d)
    os.makedirs(os.path.join(HERE, 'plans'), exist_ok=True)
    print('profile=%s 任务总数 %d' % (a.profile, len(jobs)))
    for M, g in REG.MACHINE_GPUS.items():
        mj = [j for j in jobs if j['machine'] == M]
        # 机内依赖必须在同机（零跨机同步）
        mn = {j['name'] for j in mj}
        for j in mj:
            assert all(d in mn for d in j['deps']), '跨机依赖: %s -> %s' % (j['name'], [d for d in j['deps'] if d not in mn])
        json.dump(dict(profile=a.profile, machine=M, gpus=g, jobs=mj), open(os.path.join(HERE, 'plans', '%s_%s.json' % (a.profile, M)), 'w'), ensure_ascii=False, indent=0)
        by_tier = collections.Counter()
        for j in mj:
            by_tier[j['tier']] += j['cost']
        cap = g * a.speed * a.window_h * 0.9
        tot = sum(by_tier.values())
        print('  机器 %s（%d 卡）：%3d 个任务，代价 %.1f（tier0=%.1f tier1=%.1f tier2=%.1f tier3=%.2f），容量约 %.0f，必做(tier<=1)负载 %.0f%%，含 15%% 缓冲 %.0f%%'
              % (M, g, len(mj), tot, by_tier[0], by_tier[1], by_tier[2], by_tier[3], cap, 100 * (by_tier[0] + by_tier[1]) / cap, 115 * (by_tier[0] + by_tier[1]) / cap))


if __name__ == '__main__':
    main()
