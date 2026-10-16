# 服务器 C —— 启动提示

**角色**：2 卡机：逐类留出类型 MT / PL / WB、E14 物理增强与干扰重新混合、E15 基线行（CNN / 无专家 DiT / TF-only，3 种子）

**GPU**：2 张（runner 每张卡并发 3 个任务，需 MPS）；预计必做负载见 `python exp/make_plan.py`；deadline 默认 11 小时。

## 照着敲

```bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
export PY=/root/autodl-tmp/envs/moediff_bw/bin/python     # 该 python 必须能在本机 GPU 上运行 CUDA（Blackwell 需 cu128+）
$PY exp/preflight.py --machine C        # 1) 体检：GPU 数/显存/MPS/磁盘/数据完整性/代码哈希（三台代码哈希必须相同）
$PY exp/calibrate.py --machine C        # 2) 校准 5-8 分钟：实测速度与 MPS 收益（可跳过，用默认值）
exp/launch_C.sh real 11                 # 3) 正式启动（后台运行，SSH 断开不影响）；先冒烟：exp/launch_C.sh smoke 1 ，之后 $PY exp/clean_profile.py --profile smoke --yes
```

## 监控（不需要人工介入）

```bash
cat exp_state/C/progress.txt
tail -f logs/runner_C.out
grep -l '"failed"' exp_state/C/jobs/*.json     # 失败任务
tail -n 40 logs/jobs/<任务名>.log
exp/launch_C.sh real 8 --retry_failed      # 中断后续跑；并重试失败任务（已完成的自动跳过）
```

## 本机承担的任务（共 110 个，总代价 53.8）

| 实验组 | 任务数 | 代价 |
|---|---|---|
| ANCHOR | 2 | 0.8 |
| E14 | 16 | 9.8 |
| E15 | 27 | 12.0 |
| E6 | 63 | 31.0 |
| SYS | 2 | 0.2 |

## 结果在哪里

- ../reports/e14_aug_stats.csv  (增强后的有效 SIR/SNR)
- runs/lo_MT_* , lo_PL_* , lo_WB_*  (E6)
- runs/e14_* , de_cnn_* , de_dit_* , de_tf_*  (E14/E15)

## 结束后

```bash
$PY exp/pack_results.py --machine C          # 打包（不含权重）-> /root/autodl-tmp/exp_results_C_real.tar.gz
# 三个 tar.gz 拷到同一台机器后: $PY exp/merge_results.py exp_results_A_real.tar.gz exp_results_B_real.tar.gz exp_results_C_real.tar.gz ; $PY exp/final_report.py
```

详细手册与审查要点见 `exp/README_SERVERS.md`。
