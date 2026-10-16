# 10% 小样本监督：U-Net 扩散干扰恢复 —— 代码与结果整理

> 范围：本文件整理 **v2（ST/MT/NB/WB/PL 训练，LFM/NFM/RFM 未见）** 与 **v4_digital（CCIR/FSK/OFDM/BURST/ACI 信号型干扰）** 两个场景下 10% 小样本监督实验的代码、复现命令与已完成结果。
> 数据、模型权重、逐样本 npz 不在仓库中（体积大，见文末“未上传内容”）。所有数字均为测试集 A 的平均 8 次采样结果；**种子数已逐处标注**，种子少的结论不能视为确认。

## 1. 场景定义
| 场景 | 数据 | 训练集（10% 监督） | 测试 |
|---|---|---|---|
| v2 | `data/v2`（3 种调制 × 8 类干扰，2560 点=640 符号×4 点/符号，SIR −20…+10 dB 7 档，SNR 10–30 dB） | 训练 5 类 ST/MT/NB/WB/PL，每个（类型×调制×SIR）格 4 条，共 **420 条**（训练时重复 10 倍使步数对齐 ≈21000 步） | 测试集 A：见过 5 类 2100 条 + 未见调频类 1260 条（每隔一条取样）；测试集 B：未见干扰参数区间 |
| v4_digital | `data/v4_digital`（同信道数字干扰 CCIR/FSK/OFDM/BURST/ACI，格式同 v2） | 每格 6 条，共 **630 条**，×10 重复 | 测试集 A 全部 5 类（1575 条，每隔一条） |

指标：BER（首要）、ΔBER、负增益比例、EVM、MSE、MAE、带内 SINR。扩散模型：25 步 DDIM，每个测试样本独立采样 8 次取平均，bf16。

## 2. 代码地图（`new/`）
| 路径 | 内容 |
|---|---|
| `gen/` | 数据生成：`make_dataset.py`（v2）、`make_scenarios_c.py`（**v4_digital**）、`make_scenarios_d.py`（v5 随机效应）、`signal_gen.py`（干净信号生成 `make_clean`）、`interference_gen.py`、`make_clean_pool.py`（干净波形池，预训练用，**最终未采用**） |
| `loader/` | `moe_dataset.py`（数据加载；`return_aux` 返回干扰与 12 维属性）、`physical_augment.py`（**remix / remixys / rot / shift / awgn / iqgain 增强算子**）、`attributes.py` |
| `moe_diffic/exp_train.py` | 统一训练入口（复用 `train.py` 的解析与 EMA）；评估入口 `evaluate_new.py`；`tune_and_eval.py`（DiffCom ζ / ICDM γ 验证集调参） |
| `moe_diffic/models/` | `physics_moe_model.py`（扩散主模型：U-Net/DiT/DiffWave 去噪、MoE 条件、oracle、局部定位通道、反事实钩子）、`physics_moe.py`（专家与融合：uniform/sum_single/topk_mask/**anchor 锚定残差融合**）、`domain_experts.py`、`cyclo_expert.py`（循环平稳专家 CYC）、`local_cond.py`（局部干扰定位特征、E1 预测器）、`unet1d_denoiser.py`、`baseline_new.py`（DiffCom / ICDM 的 1D 迁移实现，DiffCom 含官方式热启动）、`baseline/cnn_baseline.py`、`baseline/restormer_baseline.py`（**Restormer 的 1D 忠实移植**） |
| `moe_diffic/exp/` | 任务计划与调度：`registry.py`、`runner.py`、`launch.sh`，以及各阶段计划 `*_plan.py`（`m10`=10% 基线补齐，`aug*`=增强与 oracle 诊断，`d2`=损失与 M2，`loc`=定位与 E1，`fast*`/`v4m`/`v4a`=v4_digital 与混合专家）；`plans/*.json` |
| `moe_diffic/training/losses.py` | 损失，含多分辨率 STFT 损失 `MultiResSTFTLoss` |
| `moe_diffic/tests/` | 单元测试与回归（`python tests/run_all.py --fast`；默认流程由 golden 回归保证逐位不变） |
| `analysis/` | 汇总脚本：`unified_table.py`（多场景×基线×指标总表）、`table_10pct_supervised.py`、`fewshot10_table.py`、`export_run_summaries.py`（导出 `results/`）、`aug_oracle_report.py`、`loc_report.py`、`fast_report.py`、`fast2_report.py`、`v4m_report.py`、`m2_0_cyclo.py`（循环平稳特征可分辨性） |
| `reports/` | 报告：`fewshot-report.md`、`baseline_audit_icdm_diffcom_restormer.md`、`m2_0_cyclo_analysis.md`、`table_10pct_supervised.md`、`unified_table.md/csv`、`fewshot_all_baselines.md/csv` 等 |
| `results/` | **全部已完成 run 的紧凑结果**：`run_summary_by_subset.csv`、`run_by_type_sir.csv`、`runs_index.csv`（由 `analysis/export_run_summaries.py` 生成） |

## 3. 环境与复现
- Python 环境：torch 2.11 + cu128（Blackwell sm_120）；bf16 autocast；单卡 + NVIDIA MPS 多任务并行。
- 数据生成：`cd new/gen && python make_dataset.py`（v2）；`python make_scenarios_c.py`（v4_digital）。
- 典型训练+评估（v2，10%，无专家 U-Net + remix，种子 42）：
```bash
cd new/moe_diffic
python exp_train.py --exp_name au_remix_p10_s42 --model_type moe_diffmic --lr 3e-4 --batch_size 32 --num_workers 0 --ddim_steps 25 \
  --val_every 5 --warmup_epochs 5 --amp bf16 --val_stride 2 --skip_final_eval \
  --new_types ST MT NB WB PL --train_per_cell 4 --aug_multiplier 10 --epochs 159 --seed 42 \
  --disable_moe --true_uncond --backbone unet1d --unet_base 64 --cond_mode adaln --phys_aug remix
python evaluate_new.py --runs au_remix_p10_s42 --splits test test_ood --stride 2 --avg_k 8 --tag _avg8 --amp bf16
```
  v4_digital 把数据参数换为 `--train_per_cell 6 --aug_multiplier 10 --new_data_root ../data/v4_digital --epochs 107`，评估加 `--data_root ../data/v4_digital`。
- 批量运行：`python exp/<阶段>_plan.py && bash exp/launch.sh A <阶段> <小时> --slots <并行数>`；各阶段的配置与对照写在各 `*_plan.py` 开头注释里。
- 汇总：`python analysis/unified_table.py`、`python analysis/export_run_summaries.py`。

## 4. 已完成结果（v2，10% 监督，测试集 A 见过 5 类，BER）
**基线对比**（`reports/table_10pct_supervised.md`）

| 方法 | 种子 | BER | EVM | 带内 SINR(dB) |
|---|---|---|---|---|
| 匹配滤波（不处理） | — | 0.2254 | 3.120 | −5.17 |
| LMMSE | 1 | 0.2576 | 0.984 | 0.14 |
| CNN | 3 | 0.1407 | 0.577 | 5.77 |
| Restormer（1D 移植，22.8M） | 3 | 0.1112 | 0.465 | 8.22 |
| DiT 条件扩散（无专家） | 3 | 0.1216 | 0.465 | 8.23 |
| DiffCom（官方式热启动） | 3 | 0.2255 | 3.132 | −5.27 |
| ICDM | 3 | 0.1478 | 0.566 | 6.52 |
| U-Net 扩散，无专家 | 3 | 0.0887 | 0.348 | 13.81 |
| U-Net 扩散 + TF 专家（J0） | 3 | 0.0912 | 0.350 | 13.95 |

**数据增强（U-Net 无专家为底）**：无增强 0.0887（3 种子）；**remix 0.0746（3 种子，−16%）**；rot 0.0835（3）；rot+remix 0.0752（3，不叠加）；remixys（仅用 (y,s)，可部署版本）0.0768（2，保留 remix 收益的约 84%）；shift 0.0876 / awgn 0.0927 / iqgain 0.0904（单种子，无效）。remix 同时改善未见调频类（0.1185→0.1032）和未见参数区间（0.1458→0.1276）。
**误差预算**：remix 后 WB 占总误差 73%，且 WB 在每个 SIR 档都不优于不处理（0.2735 对 0.248）；非 WB 四类平均 BER 0.0248。去掉 WB 训练对其他类型无影响（0.0237 对 0.0248，2 种子）。建议主指标用“非 WB 平均 BER”，WB 单独作安全性测试。
**专家/条件是否有用（v2）**：TF 专家 0.0912、循环平稳专家 0.0913、真实类型+属性 oracle 0.0927 均不优于无专家 0.0887；remix 打开后 TF 专家 0.0747、oracle 0.0750 对 0.0746（2 种子），仍无收益；局部 oracle（真实干扰时频曲线）0.0651（上界，增益集中在 WB 与未见类型），由 y 预测的 E1 专家 0.0747（=基线）；TF 条件反事实（置零/错位）BER 变化 <0.001；多分辨率 STFT 损失与 ε-only 损失均无增益（单种子）。
**remix 是否泄漏**：remix 只用训练集 420 条的组件，按随机目标 SIR 重新混合，与模拟器的“相加、独立”结构一致，是物理混合增强而非训练/测试泄漏；remixys 版本只依赖配对样本 (y,s)，验证了它不依赖模拟器内部信息。

## 5. 已完成结果（v4_digital，10% 监督=630 条，BER，2 个种子 42/43）
| 配置 | 种子 42 / 43 | 均值 | 相对基线 |
|---|---|---|---|
| 无专家 + remix（基线） | 0.1282 / 0.1356 | 0.1319 | — |
| TF 专家 + remix | 0.1256 / 0.1249 | 0.1252 | −5.0% |
| oracle（类型+12 维属性）+ remix | 0.1233 / 0.1262 | 0.1248 | −5.4% |
| TF 专家 + oracle | 0.1237 / 0.1281 | 0.1259 | −4.5% |
不处理 BER 0.2411；100% 数据旧机 CNN 0.1338。专家带来约 5% 的小改善，两个种子方向一致，**仍属“有迹象、未确认”**；TF 与 oracle 几乎相同，组合无叠加。

## 6. 进行中（本文件提交时尚未出结果）
- v4_digital 上 remix 的验证（无增强 / remix / remixys），以及真正的混合专家（TF+CYC 静态、四专家静态、四专家软路由）。
- **锚定残差混合专家**（TF 为锚、其余专家零初始化累加，起点逐位等价于 TF 单专家；`--fusion anchor`）在 v4 与 v2 上的测试。
- 结果出来后会更新 `results/` 与本文件。

## 7. 重要说明
- 基线实现核对见 `reports/baseline_audit_icdm_diffcom_restormer.md`：ICDM 为合理的简化迁移；DiffCom 结构忠实、但在“未知强干扰”下似然项与任务不匹配，结果≈不处理；Restormer 已通过与官方模块的数值对照测试。
- 干净信号预训练（`gen/make_clean_pool.py`、`moe_diffic/pretrain_clean.py`、`--init_from`）已实现并通过测试，但**未运行正式实验**（按要求取消）。
- 补数据队列（U-Net 在全量监督、整类留出等场景的补齐）已取消，`unified_table.md` 中相应单元格为空。

## 8. 未上传内容
`data/`（约 4.9 GB）、`moe_diffic/runs/`（约 13 GB：权重、逐样本 npz）、`exp_state/`、`logs/`、`reports/m2_0_cache/`（51 MB）、`exp_results_A_real.tar.gz`。紧凑结果见 `results/`。
