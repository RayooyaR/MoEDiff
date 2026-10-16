# 基线审计（P0）

- 生成时间：2026-10-15 08:05；git 版本：c20e5d1；实验数：139（详见 experiment_index.csv）

## 1. 划分检查（clean_id 两两不相交）

| 数据集 | 各划分 clean_id 数 | 结果 |
|---|---|---|
| v1 | {} | PASS |
| v2 | {'train': 6720, 'val': 840, 'test': 6720, 'test_ood': 3360} | PASS |
| v3_hard | {'train': 4200, 'val': 420, 'test': 1680, 'test_ood': 42} | PASS |
| v3b_ccir | {'train': 4200, 'val': 420, 'test': 1680, 'test_ood': 21} | PASS |
| v4_digital | {'train': 6300, 'val': 840, 'test': 3150, 'test_ood': 105} | PASS |
| v5_effects | {'train': 6720, 'val': 840, 'test': 6720, 'test_ood': 3360} | PASS |

## 2. 数据定义检查（v2 抽样）

| 划分 | 抽样数 | max\|mixture−(clean+interf+noise)\| | 带内SIR误差 均值/最大(dB) | 带内SNR误差 最大(dB) | 干净信号解调 BER 最大 |
|---|---|---|---|---|---|
| train | 60 | 1.92e-06 | 0.0000 / 0.0000 | 0.0000 | 0.0000 |
| test | 60 | 1.07e-06 | 0.0000 / 0.0000 | 0.0000 | 0.0000 |
| test_ood | 60 | 1.07e-06 | 0.0000 / 0.0000 | 0.0000 | 0.0000 |

## 3. 复现检查（evaluate_new，同样本/同种子/avg8）

| run | 已保存 BER | 重算 BER | 结果 |
|---|---|---|---|
| zs2_ex_TF_s44 | 0.0983 | 0.0983 | PASS |

## 4. 路径与屏蔽说明

- 训练：`PhysicsMoEModel._training_forward` 调用 `compute_moe_condition(training=True)`（路由器带噪声/Gumbel）；采样与评估走 `sample()`，其中 `compute_moe_condition(training=False)`，路由无噪声。
- 评估使用 EMA 参数、`model.eval()`；逐样本结果含 `ber_mf`（未恢复输入的 BER），因此 ΔBER 与负增益比例可由 aggregate_results.py 直接得到。
- Top-K 屏蔽：`PASS test_router_masks：真屏蔽有效；“乘零不屏蔽”与“真屏蔽”的最大输出差 = 3.773e-02（>0 说明乘零并非真稀疏）`
- 旧“无 MoE”基线传入的是全零伪 token（经交叉注意力的偏置投影仍产生非零输出）；`--true_uncond` 才是真正的无条件分支。已有的 zsK_dit 系列基线用的是伪零 token 版本。
