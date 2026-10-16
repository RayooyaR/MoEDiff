"""为每台服务器生成一页“照着敲”的启动提示：exp/MACHINE_A.md / MACHINE_B.md / MACHINE_C.md（由任务计划自动统计）。"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import registry as REG  # noqa: E402

ROLE = {
    'A': '1 卡机（开发机）：逐类留出类型 LFM / RFM、P0 基线审计、E13 采样消融、E12 物理一致性（可选，最低优先级）、模型代价统计',
    'B': '3 卡机：依赖链（E1–E5 MoE 设计、E7–E9 主干/局部TF/条件注入、E10–E11 残差扩散与门控）→ 两次自动选择 → 依赖胜出者的 E15/E16(J0–J5) → 胜出者的 LOTO 复核',
    'C': '2 卡机：逐类留出类型 MT / PL / WB、E14 物理增强与干扰重新混合、E15 基线行（CNN / 无专家 DiT / TF-only，3 种子）',
}
OUTS = {
    'A': ['../reports/baseline_audit.md  (P0)', '../reports/e13_sampling_ablation.csv  (E13)', '../reports/e12_val.json / e12_test.json  (E12)', '../reports/model_costs.csv', 'runs/lo_LFM_* , runs/lo_RFM_*  (E6)'],
    'B': ['exp_state/B/winners.json 与 selection_log.txt  (自动选择结果)', '../reports/expert_impact_*.csv , gate_diag_*.txt  (E1/E4)', 'runs/e1_* … e11_* , c_* , de_* , j*_p* , lof_*  (各实验)'],
    'C': ['../reports/e14_aug_stats.csv  (增强后的有效 SIR/SNR)', 'runs/lo_MT_* , lo_PL_* , lo_WB_*  (E6)', 'runs/e14_* , de_cnn_* , de_dit_* , de_tf_*  (E14/E15)'],
}


def main():
    for M, g in REG.MACHINE_GPUS.items():
        plan = json.load(open(os.path.join(HERE, 'plans', 'real_%s.json' % M)))
        cnt, cost = collections.Counter(), collections.Counter()
        for j in plan['jobs']:
            cnt[j['group']] += 1; cost[j['group']] += j['cost']
        tot = sum(cost.values())
        L = ['# 服务器 %s —— 启动提示' % M, '', '**角色**：%s' % ROLE[M], '', '**GPU**：%d 张（runner 每张卡并发 3 个任务，需 MPS）；预计必做负载见 `python exp/make_plan.py`；deadline 默认 11 小时。' % g, '',
             '## 照着敲', '', '```bash', 'cd /root/autodl-tmp/MoEDiff/new/moe_diffic', 'export PY=/root/autodl-tmp/envs/moediff_bw/bin/python     # 该 python 必须能在本机 GPU 上运行 CUDA（Blackwell 需 cu128+）',
             '$PY exp/preflight.py --machine %s        # 1) 体检：GPU 数/显存/MPS/磁盘/数据完整性/代码哈希（三台代码哈希必须相同）' % M,
             '$PY exp/calibrate.py --machine %s        # 2) 校准 5-8 分钟：实测速度与 MPS 收益（可跳过，用默认值）' % M,
             'exp/launch_%s.sh real 11                 # 3) 正式启动（后台运行，SSH 断开不影响）；先冒烟：exp/launch_%s.sh smoke 1 ，之后 $PY exp/clean_profile.py --profile smoke --yes' % (M, M), '```', '',
             '## 监控（不需要人工介入）', '', '```bash', 'cat exp_state/%s/progress.txt' % M, 'tail -f logs/runner_%s.out' % M, 'grep -l \'"failed"\' exp_state/%s/jobs/*.json     # 失败任务' % M,
             'tail -n 40 logs/jobs/<任务名>.log', 'exp/launch_%s.sh real 8 --retry_failed      # 中断后续跑；并重试失败任务（已完成的自动跳过）' % M, '```', '',
             '## 本机承担的任务（共 %d 个，总代价 %.1f）' % (len(plan['jobs']), tot), '', '| 实验组 | 任务数 | 代价 |', '|---|---|---|']
        for k in sorted(cnt):
            L.append('| %s | %d | %.1f |' % (k, cnt[k], cost[k]))
        L += ['', '## 结果在哪里', ''] + ['- ' + o for o in OUTS[M]]
        L += ['', '## 结束后', '', '```bash', '$PY exp/pack_results.py --machine %s          # 打包（不含权重）-> /root/autodl-tmp/exp_results_%s_real.tar.gz' % (M, M),
              '# 三个 tar.gz 拷到同一台机器后: $PY exp/merge_results.py exp_results_A_real.tar.gz exp_results_B_real.tar.gz exp_results_C_real.tar.gz ; $PY exp/final_report.py', '```', '',
              '详细手册与审查要点见 `exp/README_SERVERS.md`。']
        open(os.path.join(HERE, 'MACHINE_%s.md' % M), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
        print('written MACHINE_%s.md' % M)


if __name__ == '__main__':
    main()
