# new/ —— 重构数据集

独立于 `MoE_DiffIC/`，不依赖也不修改其中任何代码。只需要 Python 3 + numpy（验收出图需要 matplotlib）。

```bash
cd new/gen
python make_dataset.py [输出目录]   # 默认 ../data/v2，约 1.5 分钟，1.3 GB
python validate.py [数据目录]       # 验收；输出 ../reports/<目录名>_report.json / _psd.png / _constellation.png
```

所有参数在 `gen/config.py`。

## 版本

| 目录 | 内容 | 说明 |
|---|---|---|
| `data/v2` | **当前版本**：3 种调制 × 8 类干扰，17640 条 | 当前代码可复现 |
| `data/v1` | 仅 QPSK × 6 类干扰，10920 条 | 早期版本，保留作对照；当前代码不能复现它（采样结构已改） |

## v2 设定

| 项 | 值 |
|---|---|
| 信号 | QPSK / BPSK / 16QAM（Gray），RRC（span 8），4 点/符号，2560 点 = 640 符号（1 / 2 / 4 bit 每符号） |
| 信号随机量 | 比特序列、roll-off ∈ [0.25, 0.5]、定时偏移 ∈ [0,1) 点、频偏 ±0.002 cycles/sample、初相 |
| 带内 | \|f\| ≤ fmax = (1+roll-off)/(2·4)，约 0.16～0.19 cycles/sample（占整个采样带宽的 31～38%） |
| 噪声 | 白高斯，带内 SNR ∈ [10, 30] dB 均匀 |
| 干扰 | ST、MT、PL、LFM、NB、WB、NFM、RFM，全部落在带内；带内能量占比 < 0.2 的参数会被重采样 |
| SIR | 带内 SIR（平均功率比），-20～+10 dB，步长 5（7 档） |
| 划分 | 每个「类型×调制×SIR」格子：train 40 / val 5 / test 40 / test_ood 20 条；共 6720 / 840 / 6720 / 3360 条 |
| 干净信号 | 每条样本独立 clean_id；各 split 的 clean_id 互不相交，split 内也无重复 |
| test_ood | 干扰参数取留出区间 B（见 `interference_gen.py` 各函数的 regime 分支）；用匹配滤波 BER 校准，与常规 test 逐格（类型×SIR）差异 ≤ 0.04 |
| 目录 | `<split>/<干扰类型>/<调制>/sir_<档位>.npz` |

所有类型和调制都出现在 train/val/test 中；没有留出整类干扰或整种调制。

## 样本关系与字段

```
mixture_signal = clean_signal + interference_signal + noise_signal      # 精确成立
```

- `clean_signal`：含频偏/初相/定时偏移，不含干扰与噪声（训练目标）。
- `interference_signal`：已按带内 SIR 缩放。`noise_signal`：已按带内 SNR 缩放。无噪声输入 = `clean_signal + interference_signal`。
- `bit_stream`（长度随调制变化）、`symbol_stream`（理想符号）、`modulation`、`bits_per_symbol`、`demod_gain`（匹配滤波相对符号的增益，16QAM 判决需要）。
- `sir_db_target/inband/fullband`、`snr_db_target/inband`、`alpha`、`interference_inband_frac`、`rolloff`、`timing_offset`、`cfo`、`phase`、`clean_id`、`interference_type`、`interference_params`（JSON 字符串）、`regime`（A/B）、`sample_index`。
- 同一文件内调制、干扰类型、SIR 档位相同；`manifest.csv` 列出全部文件，`metadata.json` 记录参数与定义。

## 评估解调

`signal_gen.demod_oracle`：已知 频偏/初相/定时/roll-off/增益 的匹配滤波硬判决，忽略首尾各 6 个符号。它隔离"干扰+噪声"对解调的影响，不含同步误差。

## 验收结论（v2，详见 reports/v2_report.json）

- 混合关系残差 5e-8；带内 SIR 重算偏差 0 dB；带内 SNR 均值偏差 0.00 dB。
- 干净信号解调 BER 三种调制均为 0；仅噪声时 BER：QPSK/BPSK 为 0，16QAM 为 0.0037（高阶调制本身对噪声更敏感）。
- 匹配滤波 BER 随 SIR 单调下降：除 PL 外，-20 dB 时约 0.46～0.48（接近随机），+10 dB 时 0.02 左右；调制难度 BPSK < QPSK < 16QAM。
- PL 的 BER 随 SIR 变化很小（0.07→0.02），原因是脉冲能量集中在少数时刻，平均 SIR 不反映其局部强度。
- NFM、RFM 的 MF 难度与 ST、LFM 基本一致。

## 尚未包含

同信道数字干扰、复合干扰、留出整类干扰/整种调制、多径衰落。小样本子集：不重新生成，计划之后写独立脚本，从完整训练集按固定种子、按 SIR 分层抽取嵌套样本清单。


## 难场景候选（2026-10-10，筛选用）

生成器：`gen/scenario_gen.py`（CCI 同速率同信道数字干扰、MIX 复合干扰 2~3 种叠加、CCIR 异速率同信道数字干扰），入口 `gen/make_scenarios.py`（→ `data/v3_hard`）与 `gen/make_scenarios_b.py`（→ `data/v3b_ccir`）。数据格式与 v2 相同，可直接用 MoEDiffDataset 读取。
注意：CCI 中干扰与目标统计性质完全相同，信号功率相近时原理上无法分辨谁是目标，仅作对照；CCIR 令干扰符号率与目标不同（0.6~1.7 倍，排除 0.9~1.1），目标可识别。
留出整类干扰的筛选使用 `train.py --new_types ...` 只在部分类型上训练，再用 `evaluate_new.py` 在完整 test 上评估。
