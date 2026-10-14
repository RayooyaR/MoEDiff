# MoE_DiffIC 适配新数据集 v2 —— 修改方案

状态：**仅方案，尚未修改任何代码。** 依据：读取 `MoE_DiffIC/` 源码 + 在 conda 环境 `moediff` 中的实测（标注"实测"的条目有实际运行结果）。

## 0. 执行方式

- 不改 `MoE_DiffIC/` 原目录。把它复制为 `new/moe_diffic/`，所有修改在副本上做。
- 新数据加载器已完成：`new/loader/moe_dataset.py`（`test_loader.py` 23 项全过）。
- 副本中需要改动的文件：`train.py`、`models/physics_moe_pipeline.py`、`training/losses.py`；新增 `evaluate_new.py`。其余文件不动。

## 1. 环境（实测）

| 项 | 现状 | 方案 |
|---|---|---|
| Python / 库 | `moediff` 环境：Python 3.10.22，torch 2.14.1，diffusers 0.41.0，numpy 2.2.6。旧代码核心导入（模型、损失、旧加载器）通过 | 沿用 |
| **GPU** | **不可用。** torch 是 CUDA 13.0 构建，驱动 550.120 只支持到 CUDA 12.4（torch 报 "driver too old"） | 在 `moediff` 中重装 CUDA 12.4（或 12.1）构建的 torch，需联网；需你同意 |
| 其他 | 本机 vGPU 32 GB，内存 755 GB | 够用 |

## 2. 必须修改（P0）

### 2.1 接入新数据（`train.py`）

- 新增参数 `--data_backend new` 与 `--new_data_root`，在该分支中用 `MoEDiffDataset` 构建 train / val / test / test_ood 的 DataLoader，绕过旧的 `StaticInterferenceDataset`、`ConcatDataset` 与增强流程。
- `data_scale` 计算沿用现有写法（读取数据集的 `.samples`），新加载器已兼容。
- 最终评估同时跑 test 与 test_ood，结果分别保存。
- 信号长度、采样率等从数据 `metadata.json` 读取，不再依赖命令行 `--signal_length`。

### 2.2 干扰类型名映射错位（`train.py:1620`）

`_INT_NAMES = ['NFM','RFM','LFM','MT','ST','PL','NB','WB']` 是旧顺序；新数据的类型顺序是 `['ST','MT','PL','LFM','NB','WB','NFM','RFM']`。不改的话分类型统计的标签会**静默错位**（不报错）。
→ 改为使用 `dataset.type_names`；并新增按调制（`dataset.mod_names`）的分档统计。

### 2.3 采样时幅度被截断（`models/physics_moe_pipeline.py:13,77`）— 新旧数据都有

`create_scheduler` 里 `clip_sample=True`，DDIM 每步把 x0 估计裁到 [-1, 1]。
- **实测**：让模型"想输出" x0=3.0，实际得到 1.0（cosine、linear 两种调度均如此）。
- **实测**：16QAM 干净信号按 `data_scale` 归一化后，36.8% 的采样点绝对值 > 1，最大 2.89。
→ 恢复出的波形峰值会被系统性压平，直接损害 MSE/BER。改为 `clip_sample=False`。
验证：用含大幅值的合成信号做一次"理想噪声预测"的采样，输出应还原而不被截断。

### 2.4 QPSK 星座约束损失在新数据下无效（`training/losses.py:223`）

当前做法：在固定采样偏移（`tau_s`）、固定步长（`samples_per_symbol=16`）处取点，拉向 4 个固定 QPSK 点，权重 0.05（写死在 `train.py:447`）。新数据下不成立：
- 相位、频偏、定时偏移每条样本随机，符号位置和星座旋转都不固定；
- 采样率变为 4 点/符号；
- 含 BPSK、16QAM；
- 未经匹配滤波的 RRC 波形在符号位置本身就有符号间干扰（旧数据同样如此）。

→ 本轮把该项权重置 0（先关闭）。可选的后续做法：用样本自带的 `demod` 参数对 x0 预测做匹配滤波、去旋转、增益归一后，再对该调制的理想星座点施加约束（需要把 `modulation`、`demod` 传入损失）。

### 2.5 写死的采样参数（`train.py:332-336`）

`samples_per_symbol=16, sps=16, rrc_beta=0.5, rrc_span=4, tau_s=8` 与新数据（4 点/符号，roll-off 0.25～0.5 随机，span 8）不符，会被损失函数和传统基线读到。→ 改为从数据元信息读取（roll-off 逐样本不同，不再取全局值）。

## 3. 评估（P1）

### 3.1 新评估脚本 `evaluate_new.py`

旧 `evaluate_with_all_metrics.py` 写死 16 点/符号、roll-off 0.5、固定偏移、无相位处理，且只认旧目录布局，**不适用**。新脚本：
- 载入 checkpoint，在 test、test_ood 上 DDIM 采样；
- 输出按「干扰类型 × 调制 × SIR」的 MSE、SINR、ΔSIR、BER；
- BER 用已知同步参数的匹配滤波（复用 `new/gen/signal_gen.demod_oracle`，参数由加载器的 `demod`、`symbols` 提供）；
- 同时给出两个参照：不处理直接解调、以及匹配滤波基线，用来看模型相对它们的提升。

### 3.2 最佳模型的选择指标（`train.py:~1315, ~1390`）

现在按验证集的全局 SQI = 10·log10(‖Y−X‖² / ‖X̂−X‖²)（batch 内聚合）选最佳。-20 dB 样本的干扰功率约为信号的 100 倍，会主导这个聚合值，使选择偏向低 SIR。
→ 建议改按验证集 BER（或各 SIR 档 SQI 的平均）选；同时仍打印全局 SQI。

## 4. 训练设置建议（P1，需观察后确认，非确定缺陷）

| 项 | 观察 | 建议 |
|---|---|---|
| EMA | 默认 0.9999，时间常数约 1 万步；新数据 6720 条、batch 32 ≈ 210 步/epoch，50 epoch 才约 1 万步，评估又使用 EMA 参数 | 改为 0.999 |
| 输入幅度 | 最低 SIR 时混合信号峰值最大 108，归一化后约 150，直接输入 DiT | 先观察训练是否稳定；备选：对输入做按样本 RMS 归一化 |
| STFT 参数 | `n_fft=32, hop=8` 是为 L=128 配的；L=2560、带内半宽约 0.17 时带内约 11 个频点 | 可试 `n_fft=64～128, hop=16～32`（非错误，属调参） |
| 增强 | 新加载器不含增强；旧增强的相位旋转/时移/频偏会使样本的 `demod` 参数失效，BER 评估不再成立 | 首轮关闭 |
| 训练输入 | 含噪声（`noise_mode='with'`） | 首轮用含噪声；无噪声作对照 |

## 5. 暂不处理（P2）

- 基线入口（CNN 路径与参数顺序、ICDM 缺文件、MF/LMMSE 缺 `traditional_trainer`）。核心流程跑通后再做，MF 基线用 3.1 中的匹配滤波参照代替。
- 软 MoE / 未使用的 diversity loss / 路由分析：属于方法层面，等你开始核心实验时再定。

## 6. 验证顺序

1. 重装 GPU 版 torch，确认 `torch.cuda.is_available()` 为 True。
2. 副本中接入新数据，1 epoch、小 batch 冒烟跑通（训练、验证、保存、最终评估）。
3. 单测：clip 修复后采样不截断；类型名映射与数据集一致。
4. 小规模过拟合：每格仅取少量样本训练，训练损失应明显下降，对训练样本能恢复。
5. 正式训练，用 `evaluate_new.py` 出结果表。

## 7. 需要你确认的决定

1. 采用"复制副本，不改原目录"的方式？（推荐：是）
2. 同意在 `moediff` 环境中重装 GPU 可用的 torch？（推荐：是）
3. 星座约束：本轮先关闭，还是同时实现匹配滤波域的星座约束？（推荐：先关闭）
4. 最佳模型按验证集 BER 选，还是保持全局 SQI？（推荐：BER）
5. EMA 调为 0.999？（推荐：是）

---

## 8. 执行记录（2026-10-10）

已按第 7 节的 5 项决定执行（副本方式、GPU 版 torch、星座约束关闭、按 BER 选模型、EMA=0.999）。

| 项 | 结果 |
|---|---|
| 环境 | `moediff` 中 torch 换为 2.14.1+cu126，GPU 可用（vGPU 32 GB，矩阵乘约 36 TFLOPS） |
| 副本 | `new/moe_diffic/`；改动：`train.py`、`training/losses.py`、`models/physics_moe_pipeline.py`，新增 `metrics_new.py`；原 `MoE_DiffIC/` 未动 |
| 数据接入 | `--data_backend new`（默认），读取 `new/data/v2`，训练结束同时评估 test 与 test_ood |
| clip 修复 | `new/loader/test_clip_fix.py` 通过：修复后理想噪声预测可还原峰值为 3 的信号，原设置误差 10.37 |
| 类型名映射 | 改用数据集自带的 `type_names`；冒烟测试输出的类型/调制标签正确 |
| 速度 | 每 epoch（210 步）约 26 秒（单任务），显存约 6 GB |

### 新增发现：x0 空间损失在 t→T 时爆炸（原代码即有）

`x0_pred = (x_t − √(1−ᾱ)·ε̂)/√ᾱ` 在 t 接近 1000 时被 1/√ᾱ 放大。实测重建损失：t=500 为 0.20，t=990 为 23，t=999 为 21848；2164 个训练步中 86 步 loss>5，最大 993。
新增可选开关 `--x0_loss_weighting alpha`（对 x0 空间的重建/频谱损失乘以 ᾱ_t），默认 `none` 保持原行为。

A/B（完整训练集，12 epoch，其余设置相同，val/test BER 越低越好）：

| | 训练 loss（末轮） | test BER | test_ood BER | 验证 SQI |
|---|---|---|---|---|
| none（原） | 约 2.5（含尖峰） | 0.296 | 0.300 | 11.5 dB |
| alpha | 约 0.23 | **0.225** | **0.231** | 12.9 dB |
| 不处理直接解调（匹配滤波基线） | – | 0.250 | 0.248 | – |

alpha 组 12 epoch 后，低 SIR（-20～-5 dB）已明显优于匹配滤波基线（如 -20 dB：0.330 对 0.421），但高 SIR（0～+10 dB）仍差于基线（+10 dB：0.098 对 0.022）。模型仍欠训练（仅约 2500 步），需要更长训练验证高 SIR 档能否追上。

---

## 9. 路由重设计与三组对比实验（2026-10-10）

### 路由问题（实测，12 epoch 的旧路由模型）
- 路由器无任何监督；均衡损失系数 0.005、实测约 3e-4，且其"负载"项恒为常数（softmax 权重恒 > 0），无梯度；多样性损失算出后未计入总损失；`moe_k=2` 未生效（4 个专家全部加权）。
- 门控权重主要随 SIR 变化（-20 dB 时 FD 专家 0.72，+10 dB 时 0.075），按干扰类型几乎无区分（类型间平均门控最大 L1 距离 0.47，满分 2），按调制完全相同。

### 新路由 `models/prior_router.py`（`--router_type prior`）
1. 输入先做按样本 RMS 归一化：路由只看信号形状，不受强度影响（实测强度放大 10 倍，logits 变化 8e-7）。
2. 时域 / 频谱 / 时频 三路特征。
3. 物理先验弱监督：干扰类型 → 擅长它的专家（表见 `TYPE_PRIOR`，可调），交叉熵，权重 `--route_loss_weight`（默认 0.1）。标签只在训练期使用，推理只看信号。
4. 稀疏 top-k 门控（`--moe_k`，默认 2），未选中专家权重为 0；训练期 logits 加小噪声（`--router_noise`）。
5. Switch 式可微均衡损失；多样性损失计入总损失。
独立验证（只训练路由器 800 步）：测试集上路由主专家与先验吻合率 88.7%（随机 25%）；弱项为 NFM（0.65，易与 LFM/RFM 混淆）与高 SIR（+10 dB：0.60，干扰本身很弱）。

### 三组实验（相同数据/种子/超参，100 epoch）
| 实验名 | 配置 |
|---|---|
| `exp_moe_legacy` | 原 MoE（无监督软路由） |
| `exp_nomoe` | `--disable_moe` |
| `exp_moe_prior` | `--router_type prior --balance_loss_coef 0.01` |

共同设置：`--epochs 100 --warmup_epochs 5 --lr 3e-4 --batch_size 32 --ddim_steps 25 --eval_freq 5 --save_every 20 --seed 42`；x0 损失加权默认 `alpha`；最佳模型按验证集 BER 选。结果文件：`runs/<实验名>/test_results.json`、`test_ood_results.json`（含 `routing` 段：各类型/SIR 的专家权重、与先验吻合率、类型间区分度）。
