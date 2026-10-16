# 本文件夹/压缩包的结果来自：服务器 **C**（2 卡机）

> 这是 plan.md 全部实验在三台服务器（A / B / C）分工中 **C 机** 的结果。**只包含 C 机负责的任务**，A、B 两台的结果在各自的压缩包里。

## 来源信息
- 机器：C（主机名 `autodl-container-y4c4dlvw4m-0cea6795`），2 × NVIDIA RTX PRO 6000 Blackwell（sm_120），每卡并发 3 个任务（MPS）
- 运行：profile=real，deadline 11 小时；2026-10-15 约 07:53 启动 → 14:17 结束，实际用时约 6.4 小时
- 代码哈希：`03d32b25a5ea`（105 个 py 文件；三台服务器必须一致才可合并比较）
- 任务：共 110 个，全部 done（失败 0、中止 0、因时间跳过 0）；pytorch 2.11.0+cu128，bf16 混合精度训练/采样
- 校准：单进程 53 ms/步，3 进程 MPS 并行收益 2.94×，每卡约 5.09 个代价单位/小时

## C 机承担的任务（对应 plan.md）
| 实验组 | run 数 | 打包文件数 |
|---|---|---|
| 逐类留出 E6（留出 MT / PL / WB） | 48 | 294 |
| E14 物理增强（TF-only，10% 数据） | 14 | 98 |
| E15 基线行：CNN | 9 | 63 |
| E15 基线行：无专家 DiT | 9 | 63 |
| E15 基线行：TF-only | 9 | 63 |
| 锚点（TF-only / CNN，种子 42） | 2 | 12 |

- 逐类留出（E6）：留出类型 MT、PL、WB；每个类型 8 种配置（cnn、dit0=无专家 DiT、tf=仅时频专家、static4=静态四专家、legacy4=四专家 legacy 路由、dual=双因素路由、dual_attr=双因素+属性监督、dual_shared=双因素+共享专家），其中 cnn/tf/legacy4/dual_attr 有 3 个种子（42/43/44），其余种子 42；另有冻结专家特征的迁移适配结果 `runs/lo_<类型>_{tf,dual_attr}_s42/loto_adapt.json`
- E14：TF-only，10% 训练数据（每格 4 条），固定优化步数；none/rot/shift/awgn/iqgain/remix 单项 + 自动选出的两个组合；部分补了种子 43/44；增强后有效 SIR/SNR 分布见 `reports/e14_aug_stats.csv`
- E15 基线行：CNN / 无专家 DiT / TF-only × 10%/30%/100% 数据 × 种子 42/43/44；同时评估 test 与 test_ood
- 锚点：`anc_tf_C`、`anc_cnn_C`（用于量化同一配置在不同机器上的差异）

## 压缩包里有什么（均为小文件，不含模型权重 best_model.pth）
- `moe_diffic/runs/<run名>/`（相对 `new/` 目录）：args.json、val_summary.json、run_info.json、train.log、val_log.csv、`per_sample_test_avg8.npz`（逐样本结果，扩散类为随机采样 8 次取平均、测试集每隔一条的一半样本）、`per_sample_test_ood_avg8.npz`（E15）、loto_adapt.json 等
- `exp_state/C/`：每个任务的状态 json、progress.txt、summary.json；`exp_state/calib_C.json`、`sys_tests_C.log`（单元测试日志）
- `logs/jobs/*.log`：每个任务的完整日志
- `reports/`：本说明文件 `RESULTS_FROM_MACHINE_C.md` 与本机产出（如 e14_aug_stats.csv）

## 如何使用
```bash
# 在汇总机器上（路径按实际调整），把 A、B、C 三个包都合并进同一份 runs/ 之后：
python exp/merge_results.py exp_results_A_real.tar.gz exp_results_B_real.tar.gz exp_results_C_real.tar.gz
python exp/final_report.py        # -> ../reports/plan_report.md
# 只看 C 机结果也可以直接用：
python ../analysis/aggregate_results.py --runs lo_MT_tf_s42 lo_MT_tf_s43 lo_MT_tf_s44 --suffix _avg8 --unseen MT
```

## 已知情况与注意
- 本机多卡 MPS 的一个 bug（客户端 CUDA_VISIBLE_DEVICES 需为 0）在正式启动前发现并修复；第一次启动被打断后用修复后的代码完整重跑，因此结果全部来自修复后的代码。A、B 机需同步该修复。
- 所有结果为单台机器、1–3 个种子；同一配置在不同机器上可能有 0.02 量级差异，跨机器比较请参考锚点任务。
- 旧机 TF-only 种子 42 的检查点（zs2_ex_TF/best_model.pth）此前被误删，与本包无关。
