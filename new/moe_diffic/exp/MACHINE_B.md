# 服务器 B —— 启动提示

**角色**：3 卡机：依赖链（E1–E5 MoE 设计、E7–E9 主干/局部TF/条件注入、E10–E11 残差扩散与门控）→ 两次自动选择 → 依赖胜出者的 E15/E16(J0–J5) → 胜出者的 LOTO 复核

**GPU**：3 张（runner 每张卡并发 3 个任务，需 MPS）；预计必做负载见 `python exp/make_plan.py`；deadline 默认 11 小时。

## 照着敲

```bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
export PY=/root/autodl-tmp/envs/moediff_bw/bin/python     # 该 python 必须能在本机 GPU 上运行 CUDA（Blackwell 需 cu128+）
$PY exp/preflight.py --machine B        # 1) 体检：GPU 数/显存/MPS/磁盘/数据完整性/代码哈希（三台代码哈希必须相同）
$PY exp/calibrate.py --machine B        # 2) 校准 5-8 分钟：实测速度与 MPS 收益（可跳过，用默认值）
exp/launch_B.sh real 11                 # 3) 正式启动（后台运行，SSH 断开不影响）；先冒烟：exp/launch_B.sh smoke 1 ，之后 $PY exp/clean_profile.py --profile smoke --yes
```

## 监控（不需要人工介入）

```bash
cat exp_state/B/progress.txt
tail -f logs/runner_B.out
grep -l '"failed"' exp_state/B/jobs/*.json     # 失败任务
tail -n 40 logs/jobs/<任务名>.log
exp/launch_B.sh real 8 --retry_failed      # 中断后续跑；并重试失败任务（已完成的自动跳过）
```

## 本机承担的任务（共 157 个，总代价 93.7）

| 实验组 | 任务数 | 代价 |
|---|---|---|
| ANCHOR | 2 | 0.8 |
| E1 | 17 | 11.8 |
| E10 | 14 | 9.2 |
| E11 | 10 | 1.2 |
| E15 | 39 | 23.4 |
| E16 | 16 | 8.9 |
| E2 | 3 | 2.2 |
| E3 | 8 | 4.2 |
| E4 | 6 | 4.5 |
| E5 | 2 | 1.5 |
| E6 | 9 | 4.7 |
| E7 | 12 | 9.1 |
| E8 | 4 | 3.0 |
| E9 | 4 | 3.0 |
| MOE | 7 | 5.8 |
| SEL | 2 | 0.0 |
| SYS | 2 | 0.2 |

## 结果在哪里

- exp_state/B/winners.json 与 selection_log.txt  (自动选择结果)
- ../reports/expert_impact_*.csv , gate_diag_*.txt  (E1/E4)
- runs/e1_* … e11_* , c_* , de_* , j*_p* , lof_*  (各实验)

## 结束后

```bash
$PY exp/pack_results.py --machine B          # 打包（不含权重）-> /root/autodl-tmp/exp_results_B_real.tar.gz
# 三个 tar.gz 拷到同一台机器后: $PY exp/merge_results.py exp_results_A_real.tar.gz exp_results_B_real.tar.gz exp_results_C_real.tar.gz ; $PY exp/final_report.py
```

详细手册与审查要点见 `exp/README_SERVERS.md`。
