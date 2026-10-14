# MoE-DiffIC：服务器迁移与项目速览

更新时间：2026-10-09。依据当前目录中的 Python 源码、实际 NPZ 文件、元数据和运行记录整理。

## 1. 先读这几条

- **目标**：在小样本、强干扰条件下，从受干扰的通信 I/Q 波形恢复目标波形，并改善解调误码率。
- **方法**：物理域特征 → 四个领域专家 → 动态软路由 → 条件 DiT 扩散恢复 → DDIM 采样。
- **现有数据**：仿真 QPSK；每条 2560 个复数采样点、160 个符号、320 个比特。
- **当前实际范围**：代码只识别 LFM、MT、PL 三类；八类数据目录中的其他五类会被加载器跳过。
- **当前状态**：研究原型。保留了训练启动记录，没有发现 `.pth`、`.pt`、`.ckpt` 权重或完整最终测试结果，迁移后需要重新训练。
- **优先启动路径**：静态 NPZ 数据 + `--model_type moe_diffmic`；部分基线和真实数据入口尚不完整。

## 2. 哪些文件需要迁移

建议迁移全部 **27 个 `.py` 源文件**并保留包结构，代码目录当前约 0.65 MiB（包含缓存和旧运行记录）。无需通过删减少量源码节省空间。

### 2.1 核心运行代码

| 文件／目录 | 用途 | 迁移要求 |
|---|---|---|
| `MoE_DiffIC/train.py` | 参数、配置、训练、验证、采样、模型保存 | 必须 |
| `MoE_DiffIC/data/__init__.py` | 数据包初始化 | 必须 |
| `MoE_DiffIC/data/static_interference_dataset.py` | NPZ 读取、I/Q 转换、样本筛选和增强 | 必须 |
| `MoE_DiffIC/data/interference_generator.py` | 类别列表与合成干扰生成器；加载器依赖类别列表 | 必须 |
| `MoE_DiffIC/data/augmentation.py` | 成对增强、C43 组合计划 | 必须，即使首次关闭增强 |
| `MoE_DiffIC/models/__init__.py` | 模型包初始化和导出 | 必须 |
| `MoE_DiffIC/models/physics_moe_model.py` | 主模型 | 必须 |
| `MoE_DiffIC/models/physics_moe_pipeline.py` | 扩散调度器、DDIM 采样器 | 必须 |
| `MoE_DiffIC/models/physics_moe.py` | 四专家组织、加权条件序列 | 必须 |
| `MoE_DiffIC/models/biased_router.py` | Gumbel-Softmax／Softmax 路由 | 必须 |
| `MoE_DiffIC/models/modal_projector.py` | FFT、STFT、多尺度投影 | 必须 |
| `MoE_DiffIC/models/domain_experts.py` | FD、MS、TF、TD 专家 | 必须 |
| `MoE_DiffIC/models/dit_blocks.py` | DiT、自注意力、交叉注意力 | 必须 |
| `MoE_DiffIC/models/attention_moe_fusion.py` | 保留的融合模块；模型包初始化会导入 | 必须保留 |
| `MoE_DiffIC/models/baseline/` | CNN、MF、LMMSE 和扩散对照源码 | 建议整目录迁移；其中 CNN、MF、LMMSE 文件被模型包初始化直接导入 |
| `MoE_DiffIC/training/__init__.py`、`losses.py` | 组合损失及训练包初始化 | 必须 |
| `MoE_DiffIC/evaluate_with_all_metrics.py` | 已训练模型的 MSE、MAE、SINR、ΔSIR、BER 评估 | 完整实验需迁移 |
| `MoE_DiffIC/evaluate_traditional_with_sir_breakdown.py` | MF／LMMSE 对照及 SIR 分档 | 对照实验需迁移 |

不能只复制 `train.py` 与 `physics_moe_model.py`：Python 包初始化还会导入其他模块。

### 2.2 数据：按实验范围选择

| 方案 | 需要的数据 | 适用目的 |
|---|---|---|
| 三类恢复实验 | `sim_2560/`；至少保留下述 balanced 训练／验证及 N100 测试 NPZ | 与当前代码类别一致，快速检查主模型流程 |
| 八类实验后续开发 | `8_sim_2560/` | 保留八类原始数据；训练前必须统一类别配置和加载器 |
| 完整科研迁移 | `sim_2560/` + `8_sim_2560/` | 保留当前所有数据版本，便于核对与后续实验 |

三类实验最小数据路径：

```text
sim_2560/train_2560/M005_balanced/{LFM,MT,PL}/train_random.npz
sim_2560/val_2560/M005_balanced/{LFM,MT,PL}/val_random.npz
sim_2560/test_2560/N100/{LFM,MT,PL}/sir_*.npz
```

建议同时保存对应 `metadata.json`、`manifest.csv` 和 `clean_bank_qpsk.npz`，用于追溯数据来源。静态训练不依赖 manifest 或 clean bank，而是递归扫描包含有效配对字段的 NPZ。**manifest 有过期条目，实际文件优先**：八类测试 manifest 有 560 条记录，实际仅 56 个组合文件；三类目录的旧 manifest 同样未完全匹配现状。

### 2.3 可选保留与可排除文件

| 内容 | 建议 |
|---|---|
| `data/generate_qpsk_trainset_by_m.py`、`generate_qpsk_seed_pairs_from_score_based.py` | 保留，用于追溯／重新生成数据；使用现有静态数据时不运行 |
| `data/freeze_static_splits.py` | 保留，用于离线增强与按种子划分数据 |
| `data/build_real_dataset.py` | 保留作为真实数据处理草稿；当前不能直接作为完整流水线使用 |
| `runs/moe_sim2560_3gpu/args.json`、`train.log` | 建议留档，记录已有参数和启动情况；不是模型权重 |
| 数据目录下 `viz/`、PNG、诊断 CSV／JSON | 可选，主要用于波形／星座／SIR 检查，训练不依赖 |
| `__pycache__/`、`*.pyc` | 可排除，由服务器 Python 重新生成 |
| 三个 `.rar` 压缩包 | 已复制解压目录时可排除；未核对压缩包内容与现有目录是否一致 |

当前目录容量：`sim_2560` 约 **124.71 MiB**，`8_sim_2560` 约 **264.94 MiB**。完整迁移源码和两套解压数据约 **390.30 MiB**，包括可选图片；无需再重复传输 RAR。

## 3. 服务器目录与依赖

建议保留以下相对位置，本说明放在项目根目录：

```text
MoEDiff/
├── SERVER_QUICKSTART.md
├── MoE_DiffIC/
│   ├── train.py
│   ├── evaluate_with_all_metrics.py
│   ├── evaluate_traditional_with_sir_breakdown.py
│   ├── data/                 # 全部 .py 与 __init__.py
│   ├── models/               # 含 baseline/，保留全部 .py
│   ├── training/
│   └── runs/                 # 旧记录可选，新运行会自动生成
├── sim_2560/
└── 8_sim_2560/                # 八类开发或完整迁移时保留
```

**运行工作目录固定为 `MoEDiff/MoE_DiffIC`**。这样 `../sim_2560`、`../8_sim_2560` 和包导入路径一致。

从实际导入推导的核心第三方依赖：

| 依赖 | 用途 |
|---|---|
| `torch` | 模型、训练、张量操作、CUDA |
| `numpy` | 数据读写、增强与指标 |
| `diffusers` | `DDIMScheduler` |
| `tensorboard` | `torch.utils.tensorboard.SummaryWriter` |
| `tqdm` | 训练与评估进度 |

仓库没有 requirements、环境文件或已验证的版本锁定记录。安装 PyTorch 时需与服务器 GPU、驱动及 CUDA 环境匹配；当前材料不能确定原训练环境的准确版本。现有静态 NPZ 训练不需要外部 `score_based_source_separation`／`rfcutils2`，这些属于数据生成脚本的可选来源。

服务器环境配置后先执行：

```bash
cd /path/to/MoEDiff/MoE_DiffIC
python -c "import torch, numpy, diffusers, tensorboard, tqdm; print('torch=', torch.__version__, 'numpy=', numpy.__version__, 'diffusers=', diffusers.__version__); print('CUDA=', torch.cuda.is_available(), 'GPU count=', torch.cuda.device_count())"
python -c "from models import PhysicsMoEModel; from data.static_interference_dataset import StaticInterferenceDataset; from training.losses import CombinedLoss; print('核心导入通过')"
python train.py --help
```

通过后保存实际环境，例如 `python -m pip freeze > requirements-server.lock.txt`。本说明中的服务器命令依据源码整理，尚未在目标服务器执行。

## 4. 数据与研究方法速览

### 数据规模与格式

| 位置 | 实际配对样本数 | 说明 |
|---|---:|---|
| `8_sim_2560/train_2560/M001～M005` | 分别 8、16、24、32、40 | 每类 M 条；合计 120 对，各组复用同一干净信号库 |
| `8_sim_2560/test_2560/N100` | 5600 | 8 类 × 7 档 SIR × 100 |
| `sim_2560/train_2560/M005` | 15 | 3 类 × 5 |
| `sim_2560/train_2560/M005_balanced` | 21 | 3 类 × 7 档 SIR × 1；目录名称不等于每类 5 条 |
| `sim_2560/val_2560/M005_balanced` | 3 | 每类 1 条，SIR=0 dB |
| `sim_2560/test_2560/N100` | 2100 | 3 类 × 7 档 SIR × 100 |

八类含义：NFM=窄带调频、RFM=随机调频、LFM=线性调频、MT=多音、ST=单音、PL=脉冲、NB=窄带噪声、WB=宽带噪声。SIR 档位为 `[-25,-20,-15,-10,-5,0,5] dB`。

实际来源记录为 `fallback_numpy_rrc`：NumPy 生成 QPSK，16 samples/symbol，RRC roll-off=0.5、span=4。每对样本保存 `clean_signal`、`mixture_signal`、`interference_signal`、`bit_stream`、`alpha`、目标／实际 SIR、类别和 `sample_index`。波形为 `complex64`，加载后转为 `[2,2560]` 的 `float32` I/Q。

旧版 M/N 文件中的 `interference_signal` 为缩放前干扰，混合关系为 `mixture = clean + alpha * interference`；balanced 文件中该字段为缩放后干扰，可直接与 `mixture-clean` 对应。不要混用两种字段含义。

### 模型与训练

```text
受干扰 I/Q
  ├─ FFT → FD 频域专家
  ├─ STFT → TF 时频专家
  ├─ 原始波形 → TD 时域专家
  └─ 多尺度池化 → MS 多尺度专家
            ↓ 由输入波形驱动的软路由加权
       四个专家条件 token
            ↓ 交叉注意力
    DiT（当前扩散状态 + 受干扰 I/Q + 时间步）
            ↓ DDIM 迭代
       恢复波形 → 指标／QPSK 解调
```

四个专家按表示域划分，不是一类干扰对应一个专家。训练对干净目标加入人为扩散噪声，监督模型预测扩散噪声，同时约束重建质量。通信干扰与扩散噪声是两个概念。

默认主干：hidden size=256、4 层 DiT、4 个注意力头、patch size=4。损失为噪声预测 MSE + 波形 SmoothL1 + 对数幅度谱 SmoothL1 + QPSK 星座约束 + MoE 均衡；默认权重依次为 1、0.5、0.05、0.05，均衡系数默认 0.005。训练包含 AdamW、warmup/cosine 学习率、EMA 和梯度裁剪。

评价关注 MSE、MAE、输出 SINR、SQI／ΔSIR 和 BER，并按干扰类别及 SIR 分档。所谓 N-way K-shot 主要是小样本监督恢复设置，当前未实现支持集／查询集式元学习。

## 5. 第一次运行：先检查三类主模型流程

先检查数据加载，不启动训练、不展开增强：

```bash
python - <<'PY'
from data.static_interference_dataset import StaticInterferenceDataset
for split in ('train_2560/M005_balanced', 'val_2560/M005_balanced', 'test_2560/N100'):
    ds = StaticInterferenceDataset('../sim_2560/' + split)
    print(split, 'samples=', len(ds), 'shape=', tuple(ds[0]['clean'].shape))
PY
```

预期数量为 **21、3、2100**，张量形状为 `(2,2560)`。这只是加载检查；数据中的目标波形重叠仍需处理后才能用于正式泛化实验。

下面用 1 epoch、小 batch、关闭增强检查训练／保存／最终测试流程。它会写入新的 `runs/server_smoke_3types/`，且末尾会对整个测试集采样，不是性能实验，也不是瞬间完成的检查：

```bash
python train.py \
  --model_type moe_diffmic \
  --exp_name server_smoke_3types \
  --use_static_dataset \
  --static_data_root ../sim_2560 \
  --static_train_split train_2560/M005_balanced \
  --static_val_split val_2560/M005_balanced \
  --static_test_split test_2560/N100 \
  --signal_length 2560 \
  --epochs 1 --warmup_epochs 0 \
  --batch_size 2 --num_workers 0 \
  --diffusion_steps 1000 --ddim_steps 5 \
  --device cuda --seed 42
```

流程检查通过后，再确定正式的数据划分、训练轮数、增强范围和采样步数。开启现有静态 C43 增强需要同时传入 `--use_geometric_aug --static_runtime_aug`；默认每条原始样本展开 **9768 个视图**，可用 `--static_c43_max_plan_per_seed` 限制规模。

旧记录 `runs/moe_sim2560_3gpu/args.json` 是参数留档，训练入口没有直接读取它或恢复训练的 CLI。记录使用 50 epochs、batch=32、lr=3e-4、1000 扩散步、50 DDIM 步、C43 增强、整个八类训练根目录、无验证集。由于类别过滤，实际仅加载三类共 45 条，增强后 439560 条视图；日志中参数量为 7,818,445。实验名中的 `3gpu` 不能单独证明实际使用了三张 GPU。

**每次使用新的 `--exp_name`**：`train.py` 启动时会删除同名运行目录中的旧 `train.log`、`args.json`、`best_model.pth`、`test_results.json` 和 TensorBoard events。当前无断点续训入口，不能将复制旧 runs 视为可直接继续训练。

## 6. 正式实验前必须理解的实现问题

| 问题 | 当前事实与影响 |
|---|---|
| 八类／三类不一致 | `data/interference_generator.py` 的 `INTERFERENCE_TYPES` 仅有 LFM、MT、PL；静态加载器跳过其余类型。配置中的八类列表不控制实际加载 |
| 训练／测试目标重叠 | 八类训练只有 5 条独立干净波形，全部存在于测试集；三类 balanced 训练有 21 条独立干净波形，其中 7 条存在于测试集。未发现完整配对完全重复，但对新目标内容的泛化评估受限 |
| 小样本控制 | 静态训练入口传入 `num_shots=None`，不会自动按 `--num_shots` 截断；必须显式选择对应 M 子目录或调整加载逻辑。整个训练根目录会合并所有 M 组 |
| 软 MoE／Top-k | 参数保留 `moe_k=2`，实际四专家全部计算，通过 Gumbel-Softmax／Softmax 加权，没有稀疏 Top-k 激活 |
| 多样性约束未接入 | MoE 中计算了 diversity loss，但主模型训练输出和最终总损失没有使用它 |
| 增强目标变化 | AWGN 同时加到输入和目标，增强目标含背景噪声；I/Q 单轴拉伸可使用大幅、负增益。需验证这些目标与固定 QPSK 星座约束的兼容性 |
| 非静态入口缺失 | `data/custom_interference_dataset.py` 不存在；现阶段使用静态模式 |
| CNN 基线入口 | `train.py` 导入 `models.cnn_baseline`，实际文件在 `models/baseline/`；训练调用 `(clean,noisy)` 与 CNN 的 `(noisy,clean)` 参数顺序也不一致。不能只修导入后就默认对照公平 |
| ICDM 系列基线 | 训练／评估导入路径不完整，baseline 文件还依赖缺失的 `models/old/icdm_baseline.py`，需补齐原实现与导出 |
| 传统训练分支 | 非 pure 的 MF／LMMSE 训练路径依赖缺失的 `traditional_trainer`；完整传统评估脚本自身有 MF／LMMSE 统计拟合逻辑，但数据路径需适配 |
| 真实数据流水线 | `build_real_dataset.py` 硬编码 Windows 原始数据路径，并依赖当前未提供的 `QPSKSounder`；未看到真实采集文件 |
| 对照公平性 | 若主模型有增强、基线无增强，差异同时包含模型与增强贡献；需要相同数据／增强条件的消融才能单独归因于 MoE |

另外，`evaluate_traditional_with_sir_breakdown.py` 的任务路径硬编码为 `dataset/sim_2560/...` 和 `dataset/realdata/...`，与本说明的同级数据布局不同；运行它之前需调整 `jobs` 或提供对应布局。`evaluate_with_all_metrics.py` 支持显式 `--test_root`，应指向 `../sim_2560/test_2560/N100` 这类包含各干扰类型目录的位置。

没有验证集时，训练代码会回退保存最后一轮为 `best_model.pth`，该名称不意味着按验证指标选出了最佳模型。`viz/` 中的 report／summary 是数据诊断，不能代替模型测试成绩。

## 7. 阅读入口与后续产物

- [训练参数与配置](MoE_DiffIC/train.py)：理解运行方式与默认超参数。
- [静态数据加载](MoE_DiffIC/data/static_interference_dataset.py)：理解类别过滤、样本选择和增强展开。
- [主模型](MoE_DiffIC/models/physics_moe_model.py)、[专家组织](MoE_DiffIC/models/physics_moe.py)、[路由器](MoE_DiffIC/models/biased_router.py)：理解实际 MoE 结构。
- [领域专家](MoE_DiffIC/models/domain_experts.py)、[DiT](MoE_DiffIC/models/dit_blocks.py)：理解特征与恢复网络。
- [损失函数](MoE_DiffIC/training/losses.py)、[增强](MoE_DiffIC/data/augmentation.py)：理解监督目标。
- [完整指标评估](MoE_DiffIC/evaluate_with_all_metrics.py)：理解波形指标、匹配滤波与 BER。
- [已有参数留档](MoE_DiffIC/runs/moe_sim2560_3gpu/args.json)、[启动记录](MoE_DiffIC/runs/moe_sim2560_3gpu/train.log)：仅当一并迁移旧 runs 时可读。

新训练成功后，重点保存 `runs/<exp_name>/args.json`、`train.log`、TensorBoard events、`best_model.pth`、`test_results.json`，以及设置 `--save_every` 后产生的周期 checkpoint。正式结果还应附上实际环境版本、独立的数据划分说明、使用的类别／SIR 和增强配置。
