# 服务器 A —— 启动提示

**角色**：1 卡机（开发机）：逐类留出类型 LFM / RFM、P0 基线审计、E13 采样消融、E12 物理一致性（可选，最低优先级）、模型代价统计

**GPU**：1 张（runner 每张卡并发 3 个任务，需 MPS）；预计必做负载见 `python exp/make_plan.py`；deadline 默认 11 小时。

## 照着敲

```bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
export PY=/root/autodl-tmp/envs/moediff_bw/bin/python     # 该 python 必须能在本机 GPU 上运行 CUDA（Blackwell 需 cu128+）
$PY exp/preflight.py --machine A        # 1) 体检：GPU 数/显存/MPS/磁盘/数据完整性/代码哈希（三台代码哈希必须相同）
$PY exp/calibrate.py --machine A        # 2) 校准 5-8 分钟：实测速度与 MPS 收益（可跳过，用默认值）
exp/launch_A.sh real 11                 # 3) 正式启动（后台运行，SSH 断开不影响）；先冒烟：exp/launch_A.sh smoke 1 ，之后 $PY exp/clean_profile.py --profile smoke --yes
```

## 监控（不需要人工介入）

```bash
cat exp_state/A/progress.txt
tail -f logs/runner_A.out
grep -l '"failed"' exp_state/A/jobs/*.json     # 失败任务
tail -n 40 logs/jobs/<任务名>.log
exp/launch_A.sh real 8 --retry_failed      # 中断后续跑；并重试失败任务（已完成的自动跳过）
```

## 本机承担的任务（共 53 个，总代价 27.5）

| 实验组 | 任务数 | 代价 |
|---|---|---|
| ANCHOR | 2 | 0.8 |
| E12 | 4 | 2.1 |
| E13 | 1 | 2.5 |
| E6 | 42 | 20.7 |
| E7 | 1 | 0.3 |
| P0 | 1 | 0.9 |
| SYS | 2 | 0.2 |

## 结果在哪里

- ../reports/baseline_audit.md  (P0)
- ../reports/e13_sampling_ablation.csv  (E13)
- ../reports/e12_val.json / e12_test.json  (E12)
- ../reports/model_costs.csv
- runs/lo_LFM_* , runs/lo_RFM_*  (E6)

## 结束后

```bash
$PY exp/pack_results.py --machine A          # 打包（不含权重）-> /root/autodl-tmp/exp_results_A_real.tar.gz
# 三个 tar.gz 拷到同一台机器后: $PY exp/merge_results.py exp_results_A_real.tar.gz exp_results_B_real.tar.gz exp_results_C_real.tar.gz ; $PY exp/final_report.py
```

详细手册与审查要点见 `exp/README_SERVERS.md`。
