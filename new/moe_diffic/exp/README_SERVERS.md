# 三台服务器（A / B / C）运行手册

对应 `plan.md` 的全部实验。**无人值守**：把代码和数据克隆到三台机器后，每台机器只需执行一条启动命令，runner 会按依赖 DAG 自动跑完本机任务、
在 B 机内部自动完成两次“选择”，**不需要任何跨机通信**。结束后各自打包小文件，拷到一台机器合并出报告。

## 0. 分工（零跨机同步）

| 机器 | 卡数 | 负责的任务 |
|---|---|---|
| **A**（当前开发机，1 卡） | 1 | 逐类留出（E6）类型 LFM、RFM；P0 基线审计；E13 采样消融；E12 物理一致性（可选，低优先级）；锚点 |
| **B**（3 卡） | 3 | 依赖链：E1–E5（MoE 设计）、E7–E9（主干/局部 TF/条件注入）、E10–E11（残差扩散+门控）→ **自动选择胜出者** → 依赖胜出者的 E15/E16（J0–J5）→ 胜出者的 LOTO 复核；同机的 E15 基线（种子 42）；锚点 |
| **C**（2 卡） | 2 | 逐类留出类型 MT、PL、WB；E14 物理增强与重新混合；E15 基线行（CNN / 无专家 DiT / TF-only，3 种子）；锚点 |

要点：所有“需要上游结果”的任务都在 B 机内部，选择只用验证集（`exp/selection.py`）。三台机器各跑 2 个**锚点任务**（TF-only、CNN，种子 42，同一配置），
用来量化“同一配置在不同机器/环境上的差异”——此前实测同一种子在不同机器上结果可相差 0.02，**跨机器的比较必须参考锚点**。

## 1. 克隆后每台机器的准备（约 10 分钟）

```bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
export PY=/root/autodl-tmp/envs/moediff_bw/bin/python        # 能在本机 GPU 上运行的 python（Blackwell 需要 cu128+ 的 torch；moediff 环境不支持 RTX PRO 6000）
$PY exp/preflight.py --machine A        # 把 A 换成本机字母：检查 GPU 数/显存/MPS/磁盘/数据完整性/代码哈希
$PY exp/calibrate.py --machine A        # 5-8 分钟：实测本机速度与 MPS 收益，写入 exp_state/calib_A.json（launch 会自动读取）
```
- `preflight` 会打印**代码哈希**，三台必须相同；数据完整性与源机器的 `exp/data_manifest.json` 比对（源机器上已生成）。
- 若 GPU 型号不同（如 H800/A800），先看 `calibrate` 的推荐值；若实测速度显著低于参照（每卡 3.86 单位/小时），在启动时加大 `--deadline_h` 或接受 tier-2 任务被丢弃。
- 没有 MPS 或收益 <1.12 倍时，launch 自动退化为每卡 1 个任务（容量约 −20%，必做任务可能做不完，见第 5 节）。

## 2. 启动命令（每台机器一条）

```bash
exp/launch_A.sh real 11        # 在 A 机；real=正式，11=最长运行小时数（deadline）
exp/launch_B.sh real 11        # 在 B 机
exp/launch_C.sh real 11        # 在 C 机
```
先做冒烟（约 10-30 分钟，验证整条流水线，之后务必清理）：`exp/launch_A.sh smoke 1`，结束后 `python exp/clean_profile.py --profile smoke --yes`。

启动后做了什么：preflight → 为每张卡启动 MPS → `nohup setsid` 后台启动 runner（SSH 断开不影响）。

## 3. 监控（可选；不需要人工介入）

```bash
cat exp_state/B/progress.txt          # 状态计数 + 正在运行的任务
tail -f logs/runner_B.out
ls exp_state/B/jobs | wc -l ; grep -l '"failed"' exp_state/B/jobs/*.json     # 失败任务
tail -n 50 logs/jobs/<任务名>.log      # 单个任务日志
cat exp_state/B/winners.json ; cat exp_state/B/selection_log.txt             # B 机：自动选择结果
```
**中断/重启**：直接重新执行同一条 launch 命令即可续跑（已完成的任务自动跳过，运行中被打断的重新排队）；想重试失败任务加 `--retry_failed`：
`exp/launch_B.sh real 8 --retry_failed`。

## 4. 结束后

```bash
$PY exp/pack_results.py --machine B                 # 在每台机器上；生成 /root/autodl-tmp/exp_results_B_real.tar.gz（不含权重，通常几十 MB）
# 把三个 tar.gz 拷到同一台机器（任意方式），然后：
$PY exp/merge_results.py exp_results_A_real.tar.gz exp_results_B_real.tar.gz exp_results_C_real.tar.gz
$PY exp/final_report.py                             # -> ../reports/plan_report.md（所有实验的表格、配对对比、δ 交互量、锚点一致性）
$PY analysis/...  或  python ../analysis/build_master_table.py   # 旧的总表（可选）
```

## 5. 时间预算与降级

`python exp/make_plan.py` 会打印每台机器的“必做负载 / 容量”（按每卡 3.86 个代价单位/小时 × 11 小时 × 0.9）。默认计划：A≈76%、B≈86%、C≈74%（含 15% 缓冲），
实际会因 `skip_if`（胜出者与默认相同时跳过重复复核）而更低。runner 的降级机制：
- tier 0 环境测试/锚点 → tier 1 必做 → tier 2 填充（deadline 的 80% 之后不再启动）→ tier 3 收尾；
- 预计来不及的任务标记 `skipped_time`，不会占着 GPU 拖到最后；
- 到 deadline 时强制结束剩余任务，已完成的结果完整保留；
- bf16 必须可用（已验证 TF-only/CNN 与 fp32 一致，见 memory）；若 bf16 不可用，整体容量约减半，需增加 deadline 或砍 tier 2。

## 6. 这套代码做了哪些约束（审查要点）

- `exp_train.py` 是新的训练入口；`train.py` 的训练流程未改动，默认路径与旧检查点**逐位一致**（`/root/autodl-tmp/bench/golden.py check`）。
- 选择只用验证集；`oracle_sir`、`oracle` 门控、E4 的 oracle 路由仅作上界分析，不参与任何选择。
- 先验路由的类别先验只来自训练类型；`--holdout` 的类型不进入 train/val/属性统计。
- 单元测试：`cd tests && python run_all.py --fast`（张量形状、路由屏蔽、无泄漏、划分不相交、增强物理一致性、采样可复现、golden 回归）；32+ 种变体冒烟：`python tests/smoke_variants.py`。
- 已知局限（写进论文前请核对）：
  1. 自动选择只看验证集 BER（训练见过的类型），不能直接反映对未见类型的泛化；
  2. E15 基线行（C 机）与依赖行（B 机）跨机器，已在 B 机补了种子 42 的同机基线 `deb_*`；
  3. B 机依赖行（E15/E16 的 J1–J5）只有 2 个种子（容量限制），3 种子的关键对比需另行补；
  4. E12 只实现带限引导与“独立干扰估计器”引导；α-RGS 式联合分离未实现；
  5. 局部时频 token 的 STFT 是对 I、Q 两通道分别做的实数 STFT（沿用原专家），不是复基带 STFT。
