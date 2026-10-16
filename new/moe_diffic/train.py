#!/usr/bin/env python3
"""
train.py  —  MoE-DiffIC N-way K-shot 小样本抗干扰实验统一训练入口
================================================================

本文件是整个 MoE-DiffIC 项目的 **训练主入口**, 完成以下全部流程:
  1. 命令行参数解析  (parse_args)
  2. SimpleNamespace 配置树构建  (build_config)
  3. 数据集加载 + K-shot 截断 + 几何增强  (CustomInterferenceDataset)
  4. 模型创建  (ConcatDiffusionBaseline / PhysicsMoEDiffusion)
  5. AdamW + Warmup-Cosine 学习率调度
  6. EMA 指数移动平均
  7. 训练循环  (AMP-ready, 梯度裁剪, TensorBoard 实时监控)
  8. 定期 DDIM 采样评估  (MSE, SNR Improvement, SIR 分档, I-Q 星座图)
  9. 最佳模型保存 + Early Stopping
  10. 最终全测试集评估 → test_results.json

训练流程示意图
─────────────
  parse_args()
      │
      ▼
  build_config()  ──→  SimpleNamespace 树 (data / model / moe / diffusion / training ...)
      │
      ▼
  train(args)
      ├── 随机种子 → 设备 → 日志 → TensorBoard
      ├── CustomInterferenceDataset (train / val)
      ├── model = ConcatDiffusionBaseline | PhysicsMoEDiffusion
      ├── CombinedLoss + AdamW + LambdaLR (warmup+cosine)
      ├── data_scale = std(raw_clean)   ← 用于扩散输入归一化
      │
      ├── for epoch in range(epochs):
      │     ├── 训练: forward → loss → backward → clip → step → EMA
      │     ├── 验证: forward loss
      │     └── 定期评估:
      │           ├── DDIM 采样 → MSE / sqi / SIR 分档
      │           ├── I-Q 星座图 → TensorBoard
      │           └── 保存 best_model.pth (按 sqi)
      │
      └── 最终评估: 加载 best_model → 全测试集 → test_results.json

核心对比实验 (ICDM 无增强 vs MoE-DiffIC 有增强, 多组数据量梯度):
  ┌────────────┬────────────────────────┬──────────────────────────────┐
  │  Shot      │  ICDM (no aug)         │  MoE-DiffIC (geo aug)        │
  ├────────────┼────────────────────────┼──────────────────────────────┤
  │  1-Shot    │  --num_shots 1         │  --num_shots 1 --use_geo_aug │
  │  5-Shot    │  --num_shots 5         │  --num_shots 5 --use_geo_aug │
  │  10-Shot   │  --num_shots 10        │  --num_shots 10 --use_geo_aug│
  │  Full-Data │  --num_shots -1        │  --num_shots -1 --use_geo_aug│
  └────────────┴────────────────────────┴──────────────────────────────┘

K-shot 公平性保证:
  • 同一 num_shots + seed → icdm / moe_diffmic 取到物理相同的子集
  • 测试集始终完整 (不截断), 保证评估公平

功能:
  • argparse 完整参数控制 (model_type / num_shots / augmentation / lr / epochs ...)
  • TensorBoard 实时监控 (Train/Loss, Val/MSE, Val/SNR, I-Q 星座图)
  • 训练结束自动保存 test_results.json (供 evaluate_standard.py / plot 脚本读取)

用法示例:
  python train.py --model_type icdm --num_shots 5 --exp_name icdm_5shot
  python train.py --model_type moe_diffmic --num_shots 5 --use_geometric_aug --exp_name moe_5shot_aug

TensorBoard 查看:
  tensorboard --logdir=runs/
"""

# ============================================================
import os
import sys
import json
import math
import time
import logging
import argparse
import importlib
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader            # PyTorch 数据加载器
from torch.utils.tensorboard import SummaryWriter   # TensorBoard 日志写入
from tqdm import tqdm                                # 进度条

# [new-data] 新数据集加载器（new/loader）与 BER 计算
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'loader'))
from metrics_new import batch_ber                    # noqa: E402
import random
# ------ Windows 多线程兼容 ------
# Intel OpenMP 在 Windows 上可能因重复加载报错, 此行抑制该警告
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')




# ============================================================
#  命令行参数解析器
# ============================================================
#  参数分为 6 组:
#    ① 核心实验参数 — model_type, augmentation, exp_name
#    ② 训练超参数   — epochs, batch_size, lr, ema_rate ...
#    ③ 数据集参数   — samples_per_combo, num_shots, num_workers
#    ④ 扩散/采样    — diffusion_steps, ddim_steps, beta_schedule
#    ⑤ MoE 专用     — moe_k, balance_loss_coef
#    ⑥ 评估/日志    — eval_freq, early_stopping_patience, seed
# ============================================================

def build_parser():
    """
    构建命令行参数解析器（exp_train.py 在其上追加实验参数）；parse_args() 解析并返回 argparse.Namespace
    -----------------------------------------------
    使用 ArgumentDefaultsHelpFormatter 以便 --help 显示每个参数的默认值,
    便于实验复现和超参数记录。
    """
    p = argparse.ArgumentParser(
        description='MoE-DiffIC: N-way K-shot 小样本抗干扰实验统一训练入口',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ========== ① 核心实验参数 ==========
    # model_type 决定使用哪个模型类:
    #   'icdm'        → ICDMBaseline (ICDM扩散基线, 无 MoE)
    #   'icdm_fixed'  → ICDMBaselineFixed (ICDM修正版, 复数感知IQ处理)
    #   'cnn'         → PhysicsCNNBaseline (纯监督CNN, 无扩散过程)
    #   'mf'          → MatchedFilteringBaseline (匹配滤波传统方法)
    #   'lmmse'       → LMMSEBaseline (线性最小均方误差自适应滤波)
    #   'moe_diffmic' → PhysicsMoEModel (物理感知 MoE + 多模态投影 + DiT)
    p.add_argument('--model_type', type=str, required=True,
                   choices=['icdm', 'icdm_fixed', 'diffcom_1d', 'icdm_1d', 'cnn', 'mf', 'lmmse', 'moe_diffmic', 'dit_reg', 'mv_cnn', 'res_diff', 'res_cnn', 'restormer'],
                   help='模型类型: icdm=扩散基线, icdm_fixed=修正版(复数IQ), diffcom_1d=DiffCom迁移基线, icdm_1d=双模型ICDM迁移基线, cnn=监督基线, mf=匹配滤波, lmmse=自适应滤波, moe_diffmic=MoE主模型')
    # 几何数据增强开关:
    #   关闭 → ICDM 基线实验 (无增强, 公平对比)
    #   开启 → MoE-DiffIC 实验 (相位旋转 / IQ 交换 / 时移 / IQ 取反 / 频偏)
    p.add_argument('--use_geometric_aug', action='store_true', default=False,
                   help='启用几何数据增强 (相位旋转 / IQ交换 / 时移 等)')
    p.add_argument('--disable_moe', action='store_true', default=False,
                   help='关闭 MoE 分支，仅保留 MoE-DiffIC 主干')
    # 传统方法运行模式:
    #   False → 从训练集学习统计量，然后在验证集评估
    #   True  → 直接在测试集上应用理论算法（无需训练）
    p.add_argument('--pure_traditional', action='store_true', default=False,
                   help='传统方法：直接在测试集上运行（无需训练）')
    p.add_argument('--dps_step_size', type=float, default=0.10,
                   help='1D-DiffCom 采样时 measurement guidance 步长')
    p.add_argument('--icdm_dual_gamma', type=float, default=1.0,
                   help='1D-ICDM 双分数解耦中干扰分数权重')
    p.add_argument('--lmmse_window_length', type=int, default=256,
                   help='LMMSE 滑窗长度 (传统基线)')
    p.add_argument('--lmmse_overlap', type=int, default=128,
                   help='LMMSE 窗口重叠长度 (传统基线)')
    p.add_argument('--lmmse_update_rate', type=float, default=0.01,
                   help='LMMSE 协方差矩阵 EMA 更新率')
    p.add_argument('--lmmse_omega_scale', type=float, default=1.0,
                   help='LMMSE 正则项缩放: omega = scale * kappa^2')
    p.add_argument('--lmmse_kappa2', type=float, default=None,
                   help='LMMSE 固定 kappa^2 覆盖值 (None=从训练批次估计)')
    # 实验唯一标识: 用于 TensorBoard log 子目录 + checkpoint 保存路径
    p.add_argument('--exp_name', type=str, required=True,
                   help='实验唯一标识 (TensorBoard log_dir 子目录名)')

    # ========== ② 训练超参数 ==========
    p.add_argument('--epochs', type=int, default=100,
                   help='训练总 epoch 数')
    p.add_argument('--batch_size', type=int, default=64,
                   help='训练 batch size')
    # 学习率: AdamW 默认 2e-4, DiT 论文推荐范围 [1e-4, 3e-4]
    p.add_argument('--lr', type=float, default=1e-4,
                   help='初始学习率 (AdamW)')
    p.add_argument('--weight_decay', type=float, default=1e-4,
                   help='AdamW 权重衰减')
    # Warmup: 前 warmup_epochs 个 epoch 线性升温到 lr
    p.add_argument('--warmup_epochs', type=int, default=10,
                   help='学习率 warmup 周期数')
    # 梯度裁剪: 防止扩散训练初期梯度爆炸
    p.add_argument('--gradient_clip', type=float, default=1.0,
                   help='梯度裁剪范数上限')
    # EMA: 指数移动平均, 用于评估和最终推理, 0.9999 为扩散模型常用值
    p.add_argument('--ema_rate', type=float, default=0.999,
                   help='指数移动平均衰减率 [new-data] 默认由 0.9999 改为 0.999：新数据每 epoch 仅约 200 步，'
                        '0.9999 时训练结束 shadow 仍含约 35%% 的初始随机参数')

    # ========== ③ 数据集参数 ==========
    # samples_per_combo: Full-Data 池大小 = 8类 × 6SIR × 5SNR × N = 240N 条
    #   作为"数据池"上限使用, 实际训练样本数由 --num_shots 控制
    #   建议固定为 60 (每组合 60 条 = 总池 14400), 远大于最大 K=400 即可
    p.add_argument('--samples_per_combo', type=int, default=60,
                   help='每种 (type, SIR, SNR) 组合生成的数据池样本数; '
                        '实际训练样本数由 --num_shots 控制')
    p.add_argument('--test_samples_per_combo', type=int, default=15,
                   help='每种 (type, SIR, SNR) 组合测试样本数')
    # num_shots: K-shot 设定 — 控制每种干扰类型实际参与训练的样本数
    #   -1  = 不截断 (Full-Data), 使用数据池全部样本
    #   K>0 = 每类干扰截取 K 条, 总训练集 = 8 × K 条
    #         例: K=400 → 8×400=3200 条, 而非 8×6×5×400=96000 条
    #   截断在 CustomInterferenceDataset._truncate_kshot() 中实现
    p.add_argument('--num_shots', type=int, default=-1,
                   help='K-shot: 每类干扰截取 K 条训练样本 (总训练集=8×K). '
                        '-1=不截断(Full-Data)')
    p.add_argument('--num_workers', type=int, default=4,
                   help='DataLoader 并行加载进程数')
    p.add_argument('--signal_length', type=int, default=128,
                   help='信号长度 L。静态数据集模式下会自动按样本长度覆盖。')
    p.add_argument('--use_static_dataset', action='store_true', default=False,
                   help='启用静态离线数据集模式 (从 npz 直接加载)')
    p.add_argument('--static_data_root', type=str, default='dataset',
                   help='静态数据集根目录')
    p.add_argument('--static_train_split', type=str, default=None,
                   help='静态训练子目录名 (如 train_N20)')
    p.add_argument('--static_val_split', type=str, default=None,
                   help='静态验证子目录名 (如 val_N20)')
    p.add_argument('--disable_static_val', action='store_true', default=False,
                   help='静态模式下显式关闭验证集')
    p.add_argument('--static_test_split', type=str, default=None,
                   help='兼容保留参数: 静态测试子目录名')
    p.add_argument('--static_virtual_multiplier', type=int, default=1,
                   help='静态训练集虚拟扩增倍数 (通过重复数据集实现)')
    p.add_argument('--static_c43_max_plan_per_seed', type=int, default=0,
                   help='静态 C43 全枚举时每个种子最多保留的增强组合数; 0=不限制(全保留)')
    p.add_argument('--static_c43_include_stage3', action='store_true', default=False,
                   help='是否包含 C43 stage3 交叉全组合(约 1w/seed); 默认关闭')
    p.add_argument('--static_val_aug', action='store_true', default=False,
                   help='静态模式验证集是否启用增强 (默认关闭以控制规模)')
    p.add_argument('--static_val_seed_index', type=int, default=-1,
                   help='静态模式验证种子 sample_index; -1=按seed随机选择一个并从训练集中排除')
    p.add_argument('--static_runtime_aug', action='store_true', default=False,
                   help='静态数据集模式下是否启用 DataLoader 在线增强; 默认关闭(仅加载静态样本)')

    # ========== ④ 扩散 / 采样参数 ==========
    # diffusion_steps T: 前向扩散步数, 1000 为 DDPM 标准值
    p.add_argument('--diffusion_steps', type=int, default=1000,
                   help='扩散步数 T')
    # ddim_steps: DDIM 加速采样步数, 50 步 ≈ 原始 1000 步的 95%+ 质量
    p.add_argument('--ddim_steps', type=int, default=50,
                   help='DDIM 采样步数 (推理)')
    # beta_schedule: β 调度策略
    #   'cosine' → 更均匀的信噪比分布, 适合信号处理 (MoE-DiffIC 默认)
    #   'linear' → DDPM 原始策略, ICDM 基线强制使用此项
    p.add_argument('--beta_schedule', type=str, default='cosine',
                   choices=['cosine', 'linear', 'sigmoid'],
                   help='β 调度策略')

    # ========== ⑤ MoE 专用参数 (仅 moe_diffmic 有效) ==========
    # moe_k: top-k 路由, k=2 表示每个 token 激活 2/4 个领域专家
    p.add_argument('--moe_k', type=int, default=2,
                   help='MoE top-k 路由')
    # balance_loss_coef: MoE 负载平衡损失系数
    #   过大 → 强制均匀激活, 抑制专业化; 过小 → 负载不均, 部分专家退化
    p.add_argument('--balance_loss_coef', type=float, default=0.005,
                   help='MoE 负载平衡损失系数')

    # ========== ⑥ 评估 / 日志 / 设备 ==========
    # eval_freq: 每隔 N 个 epoch 做完整的 DDIM 采样评估 (含星座图)
    p.add_argument('--eval_freq', type=int, default=5,
                   help='每隔多少 epoch 做一次全量评估')
    p.add_argument('--val_every', type=int, default=5,
                   help='每隔多少 epoch 执行一次验证损失计算；1=每轮验证')
    # 兼容旧训练命令参数: --eval_every
    p.add_argument('--eval_every', type=int, default=None,
                   help=argparse.SUPPRESS)
    # 兼容旧训练命令参数: --save_every
    p.add_argument('--save_every', type=int, default=None,
                   help=argparse.SUPPRESS)
    # early_stopping_patience: 默认 9999 即不启用, 设小可节省训练时间
    p.add_argument('--early_stopping_patience', type=int, default=9999,
                   help='Early stopping patience (eval 周期为单位)')
    # seed: 全局随机种子, 保证 K-shot 子集、权重初始化、扩散噪声均可复现
    p.add_argument('--seed', type=int, default=42,
                   help='全局随机种子')

    p.add_argument('--device', type=str, default='cuda',
                   help='训练设备')
    p.add_argument('--multi_gpu', action='store_true', default=False,
                   help='单实验多卡并行训练 (DataParallel)')

    # ========== ④ [new-data] 新数据集 new/data/v2 与评估 ==========
    p.add_argument('--data_backend', choices=['new', 'legacy'], default='new',
                   help="new: 读取 new/data/v2（MoEDiffDataset）；legacy: 沿用原有 --use_static_dataset 流程")
    p.add_argument('--new_data_root', type=str, default='../data/v2',
                   help='新数据集根目录（含 metadata.json 与 train/val/test/test_ood）')
    p.add_argument('--new_types', nargs='+', default=None, help='只用这些干扰类型（默认全部）')
    p.add_argument('--new_mods', nargs='+', default=None, help='只用这些调制（默认全部）')
    p.add_argument('--max_per_cell', type=int, default=None,
                   help='每个(类型,调制,SIR)格子最多取多少条（固定种子；用于冒烟/过拟合测试）')
    p.add_argument('--train_per_cell', type=int, default=None,
                   help='只对训练集：每个(类型,调制,SIR)格子取 K 条（小样本；固定种子，K 小的子集包含于 K 大的）')
    p.add_argument('--val_per_cell', type=int, default=None, help='只对验证集：每格取多少条（小样本时避免用到过大的验证集）')
    p.add_argument('--augment', action='store_true', default=False,
                   help='训练集使用增强（旋转/循环时移/I-Q 微拉伸/输入侧高斯噪声，见 loader/moe_dataset.py）')
    p.add_argument('--aug_multiplier', type=int, default=1, help='增强时数据集长度放大倍数')
    p.add_argument('--noise_mode', choices=['with', 'without'], default='with',
                   help='输入是否含 AWGN：with=mixture_signal；without=clean+interference')
    p.add_argument('--select_metric', choices=['ber', 'sqi'], default='ber',
                   help='最佳模型的选择指标：验证集 BER（越低越好）或全局 SQI')
    p.add_argument('--min_best_epoch', type=int, default=0,
                   help='从第几个 epoch 起才允许保存最佳模型（原代码写死为 30）')
    p.add_argument('--val_max_batches', type=int, default=0,
                   help='周期评估最多用多少个验证 batch；0=全部（原代码写死只用前 16 个）')
    p.add_argument('--dit_hidden', type=int, default=256, help='DiT 隐层宽度（默认 256，约 7.0M 参数；272 约 7.9M，用于与 MoE-DiffIC 容量匹配的对照）')
    p.add_argument('--cnn_base_ch', type=int, default=32, help='CNN 基线的基础通道数（原设计 32，约 0.9M 参数）')
    p.add_argument('--amp', choices=['none', 'bf16'], default='none',
                   help='[new-data] 混合精度：bf16=前向用 bfloat16 自动混合精度（损失、权重、EMA 仍为 fp32）；默认 none 与原行为一致')
    p.add_argument('--experts', nargs='+', default=['FD', 'MS', 'TF', 'TD'], choices=['FD', 'MS', 'TF', 'TD', 'GN', 'GN2', 'GN3', 'GN4', 'CYC', 'CYX'],
                   help='[new-data] 使用的领域专家子集（专家消融）；默认全部四个')
    p.add_argument('--router_type', choices=['legacy', 'prior', 'dual'], default='legacy',
                   help='legacy=原无监督路由；prior=先验引导的稀疏 top-k 路由（models/prior_router.py）')
    p.add_argument('--router_noise', type=float, default=0.2, help='prior 路由训练期 logits 噪声标准差')
    p.add_argument('--route_loss_weight', type=float, default=0.1,
                   help='prior 路由的监督(交叉熵)损失权重；legacy 路由下无作用')
    p.add_argument('--x0_loss_weighting', choices=['none', 'alpha'], default='alpha',
                   help="x0 空间损失(重建/频谱)的时间步加权。alpha(默认)=乘以 ᾱ_t，抑制 t→T 时 1/√ᾱ 造成的损失爆炸；none=原行为")
    p.add_argument('--constellation_weight', type=float, default=0.0,
                   help='QPSK 星座约束权重；新数据（随机相位/频偏/定时、多种调制）下该项无效，默认 0 关闭')

    return p


def parse_args():
    return build_parser().parse_args()


# ============================================================
#  Config 构建: argparse → SimpleNamespace 树
# ============================================================
#  将扁平的 argparse 参数映射为层次化的 SimpleNamespace 树,
#  保持与原 YAML 配置完全兼容的 config.xxx.yyy 访问方式。
#
#  树结构:
#    config
#    ├── data        — 数据集 / 增强 / DataLoader
#    ├── model       — 模型类型 + condition_dim
#    ├── moe         — MoE 路由 / 专家数 / 负载平衡
#    ├── modal       — ModalProjector (STFT / FFT / 多尺度)
#    ├── channel     — 信道估计嵌入维度
#    ├── denoiser    — DiT 去噪器结构 (depth / heads / hidden)
#    ├── diffusion   — 扩散过程参数 (T / β / 方差类型)
#    ├── training    — 训练超参 (lr / epochs / ema / loss)
#    ├── sampling    — 推理采样 (DDIM steps / η)
#    └── logging     — 日志 / 保存路径
# ============================================================

def build_config(args):
    """
    将 argparse 参数映射为与 YAML config 结构一致的 SimpleNamespace 树
    -------------------------------------------------------------------
    Args:
        args: parse_args() 返回的 Namespace

    Returns:
        config: SimpleNamespace 树, 各子模块可通过 config.data.xxx 访问
    """
    from types import SimpleNamespace

    # 几何增强总开关: 由命令行 --use_geometric_aug 控制
    aug_enabled = args.use_geometric_aug

    config = SimpleNamespace(
        # ======================= 数据集配置 =======================
        data=SimpleNamespace(
            dataset='custom_interference',           # 数据集标识符
            rml2016_path='./dataset/RML2016.10a_dict.pkl',  # RML2016 干净信号池路径
            # 用于构建信号池的调制方式 (V5.4: 全覆盖 11 种)
            clean_modulations=[
                'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',  # 数字调制
                'CPFSK', 'GFSK', 'PAM4',                   # FSK/PAM
                'WBFM', 'AM-DSB', 'AM-SSB'                 # 模拟调制
            ],
            # 8 种干扰类型: 窄带FM / 宽带FM / 线性调频 / 多音 / 单音 / 脉冲 / 窄带 / 宽带
            interference_types=['NFM', 'RFM', 'LFM', 'MT', 'ST', 'PL', 'NB', 'WB'],
            sir_levels=[-10, -5, 0, 5, 10, 15],     # 信干比 SIR 梯度 (dB)
            snr_levels=[6, 10, 14, 18],             # 选取 RML 中这些 SNR 级别的信号
            samples_per_combo=args.samples_per_combo,     # Full-Data 每组合样本数
            test_samples_per_combo=args.test_samples_per_combo,  # 测试每组合样本数
            sir_jitter=0.5,                          # SIR 随机抖动范围 ±0.5dB (V5.4: 减小)
            # 注: V5.4 移除 snr_jitter，SNR 由 RML 原始标签决定，不额外加 AWGN
            signal_length=int(args.signal_length),   # IQ 信号长度 L
            num_channels=2,                          # I/Q 双通道
            samples_per_symbol=16,                   # QPSK 采样率 (samples/symbol)
            sps=16,                                  # 兼容别名
            rrc_beta=0.5,                            # RRC roll-off
            rrc_span=4,                              # 默认 RRC 跨度(符号数)
            tau_s=8,                                 # 默认符号起点
            train_batch_size=args.batch_size,        # 训练 batch size
            test_batch_size=args.batch_size * 2,     # 测试 batch size (无梯度, 可翻倍)
            num_workers=args.num_workers,             # DataLoader 并行进程数
            # ---------- 传统方法超参数 ----------
            lmmse_window_length=min(int(args.lmmse_window_length), int(args.signal_length)),
            lmmse_overlap=min(int(args.lmmse_overlap), max(0, min(int(args.lmmse_window_length), int(args.signal_length)) - 1)),
            lmmse_update_rate=float(args.lmmse_update_rate),
            lmmse_omega_scale=float(args.lmmse_omega_scale),
            lmmse_kappa2=None if args.lmmse_kappa2 is None else float(args.lmmse_kappa2),
            lmmse_use_kappa2_reg=True,
            # ---------- 几何数据增强子配置 ----------
            # 设计原则: 所有增强均为 **功率不变** 的几何变换,
            # 保证信号能量分布不变, 对扩散模型的噪声预测友好
            augmentation=SimpleNamespace(
                enabled=aug_enabled,                  # 增强总开关
                # 保底机制: 即使各子增强概率抽样都未触发,
                # 也强制至少执行 min_augmentations 种增强
                min_augmentations=1,
                # 虚拟 epoch 扩充:
                #   K-shot 时训练集极小 (如 8 条), 通过 virtual_multiplier
                #   让每个种子样本在每 epoch 被增强采样多次
                #   0 = 自动计算 (扩充到 ~2048 条/epoch), >0 = 手动指定
                virtual_multiplier=0,
                # ----- 5 种功率不变的几何增强 -----
                rotation=aug_enabled,                 # 相位旋转 e^{jθ}
                rotation_prob=0.8,                    # 触发概率 80%
                time_shift=aug_enabled,               # 循环时移
                time_shift_prob=0.5,                  # 触发概率 50%
                max_shift=8,                          # 最大移位 ±8 采样点
                iq_swap=aug_enabled,                  # I/Q 通道交换 (等效 90° 旋转)
                iq_swap_prob=0.5,                     # 触发概率 50%
                iq_negate=aug_enabled,                # I/Q 取反 (等效 180° 旋转)
                iq_negate_prob=0.5,                   # 触发概率 50%
                freq_offset=aug_enabled,              # 微小频偏 e^{j2πΔf·n}
                freq_offset_prob=0.5,                 # 触发概率 50%
                max_freq_offset=0.005,                # 归一化最大频偏 0.5%
            ),
        ),

        # ======================= 模型类型配置 =======================
        # type 字段决定 train() 中实例化哪个模型类
        model=SimpleNamespace(
            type='concat_diffusion' if args.model_type in ['icdm', 'icdm_fixed', 'diffcom_1d', 'icdm_1d'] else 'physics_moe',
            condition_dim=256,                        # 条件嵌入维度 (信道 + 时间步)
        ),

        # ======================= MoE 路由配置 =======================
        # 仅在 model_type='moe_diffmic' 时生效
        moe=SimpleNamespace(
            enabled=not args.disable_moe,             # 是否启用 MoE 分支
            num_experts=len(getattr(args, 'experts', None) or ['FD', 'MS', 'TF', 'TD']),   # 领域专家数（默认 4；--experts 可取子集做消融；旧 run 的 args 无此字段）
            active_experts=list(getattr(args, 'experts', None) or ['FD', 'MS', 'TF', 'TD']),   # 频域 FD / 多尺度 MS / 时频 TF / 时域 TD
            input_dim=256,                            # 路由器输入维度
            hidden_dim=256,                           # 专家隐层维度
            expert_output_dim=256,                    # 专家输出维度
            condition_dim=256,                        # 条件嵌入维度
            k=args.moe_k,                             # top-k 路由 (默认 2；router_type=prior 时真正生效)
            router_type=args.router_type,             # [new-data]
            router_noise=args.router_noise,           # [new-data]
            noisy_gating=True,                        # 训练时向门控添加噪声
            noise_epsilon=0.01,                       # 门控噪声缩放因子
            balance_loss_coef=args.balance_loss_coef, # 负载平衡损失系数
            use_router=True,                          # 启用可学习路由器
        ),

        # ======================= 多模态投影配置 =======================
        # ModalProjector: IQ → (FFT + STFT + 多尺度) 四路物理域特征
        modal=SimpleNamespace(
            n_fft=32,                                 # STFT 窗长
            hop_length=8,                             # STFT 跳步
            n_scales=3,                               # 多尺度 AvgPool 层数
        ),

        # ======================= 信道估计嵌入配置 =======================
        channel=SimpleNamespace(
            channel_dim=64,                           # 信道特征输入维度
            hidden_dim=128,                           # 信道 MLP 隐层维度
        ),

        # ======================= DiT 去噪器配置 =======================
        denoiser=SimpleNamespace(
            backbone='dit',                           # 骨干网络类型
            hidden_size=getattr(args, 'dit_hidden', 256),   # Transformer 隐层维度 [new-data] 可用 --dit_hidden 调大做容量匹配对照
            num_heads=4,                              # 多头注意力头数
            depth=4,                                  # DiT Block 堆叠层数
            mlp_ratio=4.0,                            # FFN 扩展比 (hidden_size × 4)
            patch_size=4,                             # Patch 嵌入步长
        ),

        # ======================= 扩散过程配置 =======================
        diffusion=SimpleNamespace(
            timesteps=args.diffusion_steps,            # 前向扩散步数 T=1000
            test_timesteps=args.ddim_steps,            # 推理 DDIM 步数
            beta_schedule=args.beta_schedule,          # β 调度: cosine / linear / sigmoid
            beta_start=0.0001,                         # β₁ (linear 起始)
            beta_end=0.02,                             # β_T (linear 终止)
            model_var_type='fixedsmall',               # 方差类型: 固定小方差 σ²_t = β̃_t
        ),

        # ======================= 训练超参数配置 =======================
        training=SimpleNamespace(
            epochs=args.epochs,                        # 训练总 epoch 数
            lr=args.lr,                                # 初始学习率
            weight_decay=args.weight_decay,            # AdamW 权重衰减
            warmup_epochs=args.warmup_epochs,          # Warmup 周期
            ema=True,                                  # 启用 EMA
            ema_rate=args.ema_rate,                    # EMA 衰减率
            gradient_clip=args.gradient_clip,          # 梯度裁剪上限
            loss_type='mse',                           # 扩散损失类型: MSE
            diffusion_loss_weight=1.0,                 # 扩散损失权重
            reconstruction_loss_weight=0.5,            # clean 重建权重
            spectral_loss_weight=0.05,                 # 频域一致性权重
            constellation_loss_weight=args.constellation_weight,  # [new-data] 默认 0（关闭）
            route_loss_weight=args.route_loss_weight if args.router_type == 'prior' else 0.0,  # [new-data]
            seed=args.seed,                            # 随机种子
            min_eval_epoch=args.min_best_epoch,        # 最早保存模型的 epoch
            early_stopping_patience=args.early_stopping_patience,  # ES patience
        ),

        # ======================= 推理采样配置 =======================
        sampling=SimpleNamespace(
            method='ddim',                             # 采样方法: DDIM
            steps=args.ddim_steps,                     # DDIM 采样步数
            eta=0.0,                                   # DDIM η=0 → 确定性采样
        ),

        # ======================= 日志 / 保存配置 =======================
        logging=SimpleNamespace(
            log_dir=os.path.join('runs', args.exp_name),  # TensorBoard 日志目录
            save_freq=10,                              # 定期保存频率 (未实际使用)
            vis_freq=args.eval_freq,                   # 可视化频率 = 评估频率
        ),
    )
    from exp_config import extend_config      # [exp] 实验扩展字段（均有默认值，旧 checkpoint 的 args 缺字段时行为不变）
    config = extend_config(config, args)
    return config


# ============================================================
#  EMA (Exponential Moving Average) 指数移动平均辅助类
# ============================================================
#  EMA 是扩散模型训练的标准技巧 (DDPM / DiT 均使用):
#    shadow_θ = decay × shadow_θ + (1 - decay) × θ
#
#  效果: shadow 参数比在线参数更平滑, 减少训练末期的振荡,
#        生成/采样质量更稳定。
#
#  使用模式 (在训练循环中):
#    1. 每步训练后: ema.update()          ← 更新 shadow 参数
#    2. 评估前:     ema.apply_shadow()    ← 用 shadow 替换模型参数
#    3. 评估后:     ema.restore()         ← 恢复在线参数继续训练
# ============================================================

class EMA:
    """
    指数移动平均 (EMA) 参数管理器
    ----------------------------
    维护模型可训练参数的影子副本 (shadow),
    通过指数衰减平均实现参数平滑。

    公式:
      θ_shadow ← decay · θ_shadow + (1 - decay) · θ_online

    典型 decay = 0.9999, 即 shadow 每步仅吸收 0.01% 的在线更新,
    约需 10000 步才能完全 "忘记" 初始值。

    Attributes:
        model:  被管理的 nn.Module
        decay:  衰减率 (默认 0.9999)
        shadow: Dict[str, Tensor] — 影子参数 {name: tensor}
        backup: Dict[str, Tensor] — apply_shadow 时暂存的在线参数
    """

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}    # 影子参数存储
        self.backup = {}    # 在线参数临时备份
        # 初始化: 将当前模型参数复制为 shadow 初始值
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()
        # 【V5.1 加固】同步 buffers (如 BatchNorm running_mean/var 等)
        # 当前模型以 InstanceNorm/LayerNorm 为主, 无 running stats buffers,
        # 但预留此逻辑以兼容未来可能加入 BatchNorm 或其他有状态层。
        self.shadow_buffers = {}
        for name, buf in model.named_buffers():
            self.shadow_buffers[name] = buf.data.clone()

    def update(self):
        """
        每步训练后调用: 用在线参数更新 shadow
        公式: shadow ← decay × shadow + (1 - decay) × online
        """
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = (1 - self.decay) * p.data + self.decay * self.shadow[name]
        # 【V5.1 加固】buffers 直接拷贝 (不做 EMA 平滑, 因为 running stats 本身已是移动平均)
        for name, buf in self.model.named_buffers():
            if name in self.shadow_buffers:
                self.shadow_buffers[name] = buf.data.clone()

    def apply_shadow(self):
        """
        评估前调用: 将 shadow 参数写入模型 (同时备份在线参数)
        调用后模型使用平滑后的 EMA 参数进行推理
        """
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data        # 备份当前在线参数
                p.data = self.shadow[name]         # 加载 shadow 参数
        # 【V5.1 加固】同步 buffers
        self.backup_buffers = {}
        for name, buf in self.model.named_buffers():
            if name in self.shadow_buffers:
                self.backup_buffers[name] = buf.data.clone()
                buf.data = self.shadow_buffers[name]

    def restore(self):
        """
        评估后调用: 从 backup 恢复在线参数, 继续正常训练
        必须在 apply_shadow() 之后成对调用
        """
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[name]         # 恢复在线参数
        self.backup = {}                           # 清空备份, 释放内存
        # 【V5.1 加固】恢复 buffers
        for name, buf in self.model.named_buffers():
            if hasattr(self, 'backup_buffers') and name in self.backup_buffers:
                buf.data = self.backup_buffers[name]
        self.backup_buffers = {}


# ============================================================
#  核心训练循环
# ============================================================
#  train() 是整个训练流程的主函数, 按 4 个阶段执行:
#    [1/4] 数据集加载 (CustomInterferenceDataset, K-shot 截断)
#    [2/4] 模型创建 (ConcatDiffusionBaseline / PhysicsMoEDiffusion)
#    [3/4] 训练循环 (forward → loss → backward → EMA → 定期评估)
#    [4/4] 最终测试 (加载 best_model → 全测试集 → JSON)
# ============================================================

def train(args):
    """
    完整训练流程
    -----------
    Args:
        args: parse_args() 返回的 Namespace, 包含全部命令行参数

    流程:
        1. 设定随机种子 → 选择设备 → 构建 config → 初始化日志
        2. 创建 train/val 数据集 + DataLoader
        3. 实例化模型 + CombinedLoss + AdamW + LR scheduler + EMA
        4. 计算 data_scale (原始信号标准差, 用于扩散归一化)
        5. 训练循环: 每 epoch 训练 + 验证 + 定期 DDIM 评估
        6. 加载最佳模型, 在全测试集上做最终评估, 保存 JSON
    """

    # 兼容旧版参数名: --eval_every 等价于 --eval_freq
    if getattr(args, 'eval_every', None) is not None:
        args.eval_freq = args.eval_every

    # [new-data] 原 ICDMBaseline(models/old) 缺失；其定义为"纯 DiT 条件扩散、无 MoE"，即 moe_diffmic --disable_moe
    if args.model_type in ('icdm', 'icdm_fixed'):
        args.model_type, args.disable_moe = 'moe_diffmic', True

    # ================================================================
    #  Step 0: 随机种子 — 保证 K-shot 子集 + 权重初始化 + 扩散噪声可复现
    # ================================================================

    random.seed(args.seed)                          # Python 标准库随机
    np.random.seed(args.seed)                       # NumPy 随机
    torch.manual_seed(args.seed)                    # PyTorch CPU 随机
    torch.cuda.manual_seed_all(args.seed)           # PyTorch 所有 GPU 随机
    torch.backends.cudnn.deterministic = True       # cuDNN 确定性模式
    torch.backends.cudnn.benchmark = False           # 禁用自动寻优 (保证可复现)

    # ================================================================
    #  Step 1: 设备选择
    # ================================================================
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA unavailable, fallback to CPU')
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # ================================================================
    #  Step 2: 配置构建 + 日志目录初始化
    # ================================================================
    config = build_config(args)
    log_dir = config.logging.log_dir

    # 重新训练时清理旧文件, 避免日志/TensorBoard 事件叠加
    if os.path.exists(log_dir):
        for fname in ['train.log', 'args.json', 'best_model.pth',
                      'test_results.json']:
            fpath = os.path.join(log_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        # 清理 TensorBoard events 文件 (防止旧数据与新训练混淆)
        for fname in os.listdir(log_dir):
            if fname.startswith('events.out.tfevents'):
                os.remove(os.path.join(log_dir, fname))
    os.makedirs(log_dir, exist_ok=True)

    # ================================================================
    #  Step 3: 日志系统 — 同时输出到文件和控制台
    # ================================================================
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout),       # 控制台实时显示
        ],
    )
    logger = logging.getLogger('train')

    # 保存本次实验完整参数 → args.json (便于事后检查实验配置)
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger.info('=' * 60)
    logger.info(f'  Experiment: {args.exp_name}')
    logger.info(f'  Model type: {args.model_type}')
    logger.info(f'  Geometric augmentation: {args.use_geometric_aug}')
    logger.info(f'  Num shots: {args.num_shots} (per class, -1=Full-Data)')
    logger.info(f'  Device: {device}')
    logger.info('=' * 60)

    # ---- TensorBoard 日志写入器 ----
    writer = SummaryWriter(log_dir=log_dir)

    def _log_moe_gate_stats(gates, tag_prefix, step):
        """将 MoE 门控统计写入 TensorBoard, 便于远程监控专家利用率。"""
        if gates is None or (not torch.is_tensor(gates)) or gates.ndim != 2:
            return

        g = gates.detach()
        probs = g.clamp_min(1e-8)
        importance = g.mean(dim=0)                      # 各专家平均权重
        load = (g > 0).float().mean(dim=0)             # 各专家激活比例
        entropy = (-(probs * torch.log(probs)).sum(dim=-1)).mean()
        top1_mean = g.max(dim=-1).values.mean()
        active_mean = (g > 0).float().sum(dim=-1).mean()

        n_exp = g.shape[1]
        for i in range(n_exp):
            name = expert_names[i] if i < len(expert_names) else f'E{i}'
            writer.add_scalar(f'{tag_prefix}/GateImportance/{name}', importance[i].item(), step)
            writer.add_scalar(f'{tag_prefix}/GateLoad/{name}', load[i].item(), step)

        writer.add_scalar(f'{tag_prefix}/GateEntropy', entropy.item(), step)
        writer.add_scalar(f'{tag_prefix}/GateTop1Mean', top1_mean.item(), step)
        writer.add_scalar(f'{tag_prefix}/GateActiveExpertsMean', active_mean.item(), step)

    # ================================================================
    #  [1/4] 数据集加载
    # ================================================================
    #  创建 train / val 两个 CustomInterferenceDataset 实例:
    #    - train: 可截断 (K-shot), 可增强 (几何变换), seed=args.seed
    #    - val:   始终完整 (num_shots=-1), 无增强, seed+1 保证与 train 不重叠
    #  K-shot 公平性: 同一 seed + num_shots 保证不同 model_type 取相同子集
    # ================================================================
    logger.info('[1/4] Loading dataset …')
    new_meta = None                      # [new-data] 数据集元信息；None 表示 legacy 后端
    mod_names = []
    int_names = ['NFM', 'RFM', 'LFM', 'MT', 'ST', 'PL', 'NB', 'WB']   # legacy 顺序
    ood_dataset = None
    if args.data_backend == 'new':
        from moe_dataset import MoEDiffDataset
        root = os.path.abspath(args.new_data_root)
        kw = dict(types=args.new_types, mods=args.new_mods, noise_mode=args.noise_mode,
                  max_per_cell=args.max_per_cell, seed=args.seed)
        kw_tr, kw_va = dict(kw), dict(kw)
        if args.train_per_cell is not None:
            kw_tr['max_per_cell'] = args.train_per_cell
        if args.val_per_cell is not None:
            kw_va['max_per_cell'] = args.val_per_cell
        train_dataset = MoEDiffDataset(root, 'train', augment=args.augment, virtual_multiplier=args.aug_multiplier, **kw_tr)
        val_dataset = MoEDiffDataset(root, 'val', **kw_va)
        test_dataset = MoEDiffDataset(root, 'test', **kw)
        ood_dataset = MoEDiffDataset(root, 'test_ood', **kw)
        new_meta = train_dataset.meta
        mod_names = list(train_dataset.mod_names)
        int_names = list(train_dataset.type_names)
        logger.info(f'  [New] root={root}  noise_mode={args.noise_mode}  types={args.new_types or "all"}  '
                    f'mods={args.new_mods or "all"}  max_per_cell={args.max_per_cell}')
        logger.info(f'  [New] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} '
                    f'test_ood={len(ood_dataset)}')
        # 采样参数以数据集为准（原 config 中写死 16 点/符号）
        config.data.signal_length = int(new_meta['SIG_LEN'])
        config.data.samples_per_symbol = config.data.sps = int(new_meta['SPS'])
        config.data.rrc_span = int(new_meta['RRC_SPAN'])
    elif args.use_static_dataset:
        from torch.utils.data import ConcatDataset
        from data.static_interference_dataset import StaticInterferenceDataset
        from types import SimpleNamespace

        def _resolve_split(root_dir, explicit_split, candidates):
            if explicit_split:
                explicit_path = os.path.join(root_dir, explicit_split)
                if not os.path.isdir(explicit_path):
                    raise FileNotFoundError(f'指定 split 不存在: {explicit_path}')
                return explicit_split
            for c in candidates:
                if os.path.isdir(os.path.join(root_dir, c)):
                    return c
            raise FileNotFoundError(
                f'未找到可用静态 split，root={root_dir}, candidates={candidates}'
            )

        if args.num_shots > 0:
            n3 = f"{int(args.num_shots):03d}"
            train_candidates = [f'train_N{args.num_shots}', f'train/M{n3}', f'M{n3}']
            # 默认不把 test split 当作 val，避免验证泄漏测试集。
            val_candidates = [f'val_N{args.num_shots}', f'val/N{n3}']
            test_candidates = [f'test/N{n3}', f'N{n3}']
        else:
            train_candidates = ['train_N50', 'train/M050', 'M050']
            val_candidates = ['val_N50', 'val/N050']
            test_candidates = ['test/N050', 'N050']

        train_split = _resolve_split(args.static_data_root, args.static_train_split, train_candidates)
        try:
            val_split = _resolve_split(args.static_data_root, args.static_val_split, val_candidates)
        except FileNotFoundError:
            val_split = None
        train_root = os.path.join(args.static_data_root, train_split)
        val_root = os.path.join(args.static_data_root, val_split) if val_split is not None else None
        try:
            test_split = _resolve_split(args.static_data_root, args.static_test_split, test_candidates)
            test_root = os.path.join(args.static_data_root, test_split)
        except FileNotFoundError:
            test_split = None
            test_root = None

        # Static 模式下, 仅对 MoE-DiffIC 在开启几何增强时启用 C43 全枚举增强。
        static_aug_enabled = bool(args.use_geometric_aug and args.static_runtime_aug)
        static_aug_cfg = None
        if static_aug_enabled:
            base_aug = config.data.augmentation
            static_aug_cfg = SimpleNamespace(
                rotation=True,
                awgn=True,
                time_shift=False,
                stretching=True,
                # 严格 C43 全枚举策略（固定网格 + 分阶段组合）
                c43_enabled=True,
                c43_enumerate_all=True,
                force_all_enabled=False,
                # Stage 0/1/2/3 固定参数
                c43_awgn_snr_grid=list(range(-6, 10, 2)),       # 8 档: -6, -4, ..., 8
                c43_rotation_deg_set=list(range(-180, 180, 10)),   # 36 档
                c43_sto_shifts=[],                              # 已删除时移
                c43_stretch_axes=['I', 'Q'],                    # 2 个轴
                c43_stretch_gains=[-4.0, -3.6, -3.2, -2.8, -2.4, -2.0, -1.6, -1.2, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0],  # 16 档
                # 全枚举口径下不使用随机 phase2 混合开关
                c43_phase2_mix_enabled=False,
                c43_phase2_mix_ratio=0.0,
                c43_enforce_stretch_rotation_ratio=False,
                c43_stretch_to_rotation_ratio=1.0,
                c43_max_plan_per_seed=max(0, int(args.static_c43_max_plan_per_seed)),
                c43_include_stage3=bool(args.static_c43_include_stage3),
                # 兼容基础增强调参
                min_augmentations=getattr(base_aug, 'min_augmentations', 1),
                max_augmentations=getattr(base_aug, 'max_augmentations', 4),
                awgn_snr_db_min=-6.0,
                awgn_snr_db_max=10.0,
                stretching_max_epsilon=getattr(base_aug, 'stretching_max_epsilon', 0.05),
            )

        # C43 全枚举已对每个种子产生固定大量组合, 不再叠加虚拟扩增倍数。
        virtual_multiplier = int(args.static_virtual_multiplier)
        if static_aug_enabled:
            virtual_multiplier = 1

        if args.disable_static_val:
            val_root = None
        same_train_val_root = (val_root is not None) and (os.path.normpath(train_root) == os.path.normpath(val_root))
        val_aug_enabled = static_aug_enabled and same_train_val_root
        static_val_seed_index = None
        train_exclude_seed_index = None

        if same_train_val_root:
            # 验证集从 train root 中选一个 seed，并从训练集中排除，避免数据泄漏。
            probe_ds = StaticInterferenceDataset(
                train_root,
                num_shots=None,
                seed=args.seed,
                apply_augmentation=False,
            )
            candidate_seed_ids = sorted({int(s['sample_index']) for s in probe_ds.samples})
            if not candidate_seed_ids:
                raise ValueError('No sample_index found in static train split for val seed selection.')

            if int(args.static_val_seed_index) >= 0:
                if int(args.static_val_seed_index) not in candidate_seed_ids:
                    raise ValueError(
                        'Requested static_val_seed_index={} not in candidates={}'.format(
                            args.static_val_seed_index, candidate_seed_ids
                        )
                    )
                static_val_seed_index = int(args.static_val_seed_index)
            else:
                rng = np.random.RandomState(args.seed)
                static_val_seed_index = int(rng.choice(candidate_seed_ids))

            train_exclude_seed_index = static_val_seed_index
            logger.info(
                f'  [Static][ValSplit] train/val share root; selected val sample_index={static_val_seed_index}, excluded from train.'
            )

        if virtual_multiplier > 1:
            train_views = []
            for rep in range(virtual_multiplier):
                train_views.append(
                    StaticInterferenceDataset(
                        train_root,
                        num_shots=None,
                        seed=args.seed + rep,
                        exclude_sample_index=train_exclude_seed_index,
                        apply_augmentation=static_aug_enabled,
                        augment_config=static_aug_cfg,
                    )
                )
            train_dataset = ConcatDataset(train_views)
        else:
            train_dataset = StaticInterferenceDataset(
                train_root,
                num_shots=None,
                seed=args.seed,
                exclude_sample_index=train_exclude_seed_index,
                apply_augmentation=static_aug_enabled,
                augment_config=static_aug_cfg,
            )

        val_dataset = None
        if val_root is not None:
            # 验证集: 当 val 与 train 同源时自动同步增强；否则默认关闭增强。
            val_dataset = StaticInterferenceDataset(
                val_root,
                num_shots=None,
                seed=args.seed + 1,
                include_sample_index=static_val_seed_index if same_train_val_root else (0 if val_aug_enabled else None),
                apply_augmentation=val_aug_enabled,
                augment_config=static_aug_cfg,
            )
        test_dataset = None
        if test_root is not None:
            test_dataset = StaticInterferenceDataset(test_root, num_shots=None, seed=args.seed + 2)
            logger.info(f'  Static test enabled: test={test_root}')
        logger.info(f'  Static mode enabled: train={train_root}, val={val_root}')
        if val_dataset is None:
            logger.warning('  Static val split not found; validation will be skipped (no test leakage).')
        if static_aug_enabled:
            # 统计 C43 分阶段数量并与理论值对齐
            train_ref_ds = train_dataset.datasets[0] if hasattr(train_dataset, 'datasets') else train_dataset
            c43_plan = getattr(train_ref_ds, 'c43_plan', None)
            if c43_plan is None:
                raise RuntimeError('C43 full enumeration expected but c43_plan is missing in static dataset.')

            stage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for p in c43_plan:
                st = int(p.get('stage', -1))
                if st in stage_counts:
                    stage_counts[st] += 1
            expected_stage_counts = {0: 8, 1: 544, 2: 9216, 3: 0}
            if stage_counts != expected_stage_counts:
                raise ValueError(
                    f'C43 stage counts mismatch: got={stage_counts}, expected={expected_stage_counts}'
                )

            logger.info('  [Static][Aug] 使用 C43 严格全枚举策略 (AWGN/Rotation/Stretch，已删除时移)')
            logger.info(
                '  [Static][Aug] 分阶段数量(每种子): '
                f'stage0={stage_counts[0]}, stage1={stage_counts[1]}, '
                f'stage2={stage_counts[2]}, stage3={stage_counts[3]}'
            )
            logger.info(f'  [Static][Aug] include_stage3={bool(args.static_c43_include_stage3)}')
            logger.info(f'  [Static][Aug] 每种子总增强计划数={len(c43_plan)}')
            if args.static_c43_max_plan_per_seed > 0:
                logger.info(f'  [Static][Aug] C43 每种子上限: {args.static_c43_max_plan_per_seed}')
            if val_aug_enabled:
                logger.info(f'  [Static][Val] 选择单种子 sample_index={static_val_seed_index if static_val_seed_index is not None else 0}，并与训练共享相同 C43 增强计划')
            else:
                logger.info('  [Static][Val] 默认关闭增强（可用 --static_val_aug 开启）')
            logger.info('  [Static][Aug] virtual_multiplier=1 (C43 全枚举模式下固定)')

        # Static npz 数据可为任意长度（如 2560），自动覆盖配置长度。
        ref_ds = train_dataset.datasets[0] if hasattr(train_dataset, 'datasets') else train_dataset
        sample0 = ref_ds[0]
        inferred_len = int(sample0['clean'].shape[-1])
        if inferred_len != int(config.data.signal_length):
            logger.info(f'  [Static] signal_length override: {config.data.signal_length} -> {inferred_len}')
            config.data.signal_length = inferred_len
    else:
        try:
            dataset_mod = importlib.import_module('data.custom_interference_dataset')
            CustomInterferenceDataset = getattr(dataset_mod, 'CustomInterferenceDataset')
        except (ModuleNotFoundError, AttributeError) as e:
            raise ModuleNotFoundError(
                '当前仓库缺少 data.custom_interference_dataset，非静态模式不可用。'
                '请改用 --use_static_dataset 并设置 --static_data_root（例如 dataset）。'
            ) from e

        # 训练集: 根据 num_shots 决定是否 K-shot 截断
        train_dataset = CustomInterferenceDataset(
            config, mode='train', seed=args.seed, num_shots=args.num_shots
        )
        # 验证/测试集: 始终使用全部样本, seed+1 保证与训练集统计独立
        val_dataset = CustomInterferenceDataset(
            config, mode='test', seed=args.seed + 1, num_shots=-1  # 测试集永远不截断
        )
        test_dataset = None

    # ---- K-shot 自适应: batch_size 不能超过训练集大小 ----
    # 例如 1-Shot 时训练集仅 8 条, batch_size=64 会导致 DataLoader 异常
    effective_bs = min(config.data.train_batch_size, len(train_dataset))
    if effective_bs < config.data.train_batch_size:
        logger.info(f'  [K-shot] batch_size capped: {config.data.train_batch_size} -> {effective_bs}')

    # ---- 构建 DataLoader ----
    # pin_memory: 当 num_workers>0 时启用页锁内存, 加速 GPU 传输
    # drop_last=False: K-shot 场景样本珍贵, 不丢弃最后不完整 batch
    train_loader = DataLoader(
        train_dataset,                          # 数据集对象
        batch_size=effective_bs,                # 批大小 (可能被 K-shot 截断)
        shuffle=True,                           # 每 epoch 打乱顺序 (训练必须)
        num_workers=args.num_workers,           # 并行加载进程数 (默认 4)
        pin_memory=(args.num_workers > 0),      # 页锁内存 (加速 GPU 传输)
        drop_last=False,                        # 不丢弃最后不完整的 batch
    )
    # ========== 验证数据加载器 ==========
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.test_batch_size, # 测试 batch 更大 (无梯度, 可翻倍)
            shuffle=False,                          # 验证不打乱 (保证可复现)
            num_workers=args.num_workers,
            pin_memory=(args.num_workers > 0),
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.num_workers > 0),
        )

    ood_loader = None
    if ood_dataset is not None:
        ood_loader = DataLoader(
            ood_dataset,
            batch_size=config.data.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.num_workers > 0),
        )

    # [new-data] 路由先验表 (num_types, num_experts)：干扰类型 -> 专家先验分布
    route_prior = None
    if args.router_type == 'prior' and new_meta is not None and not args.disable_moe:
        from models.prior_router import build_prior_table
        route_prior = build_prior_table(int_names, list(config.moe.active_experts)).to(device)
        logger.info('  [Router] prior-guided sparse top-%d router; prior table built for types %s' % (args.moe_k, int_names))

    steps_per_epoch = len(train_loader)    # 每 epoch 的 batch 数
    actual_epochs = args.epochs            # 实际训练 epoch 数

    logger.info(f'  Train: {len(train_dataset)} samples, '
                f'{steps_per_epoch} batches/epoch, '
                f'{actual_epochs} epochs → ~{actual_epochs * steps_per_epoch} total steps')
    logger.info(f'  Val:   {len(val_dataset) if val_dataset is not None else 0} samples')

    # ================================================================
    #  [2/4] 模型创建 - 非侵入式ICDM基线支持
    # ================================================================
    #  V5.4 更新: 根据 model_type 选择不同的模型类
    #    • model_type='icdm'        → ICDMBaseline (独立ICDM基线)
    #    • model_type='moe_diffmic' → PhysicsMoEModel (MoE-DiffIC主架构)
    #
    #  设计理念: 非侵入式集成，保持原有MoE代码完全不变
    # ================================================================
    logger.info('[2/4] Creating model …')

    if args.model_type == 'icdm':
        # ICDM 基线模型（直接导入避免pipeline依赖）
        from models.icdm_baseline import ICDMBaseline
        model = ICDMBaseline(config)
        logger.info('  Model: ICDMBaseline (ICDM扩散基线)')
    elif args.model_type == 'icdm_fixed':
        # ICDM 修正版基线模型（正确处理IQ信号复数特性）
        try:
            fixed_mod = importlib.import_module('models.icdm_baseline_fixed')
            ICDMBaselineFixed = getattr(fixed_mod, 'ICDMBaselineFixed')
            model = ICDMBaselineFixed(config)
            logger.info('  Model: ICDMBaselineFixed (ICDM修正版 - 复数感知IQ处理)')
        except (ModuleNotFoundError, AttributeError):
            from models.icdm_baseline import ICDMBaseline
            model = ICDMBaseline(config)
            logger.warning('  icdm_baseline_fixed 不存在，已回退为 ICDMBaseline')
    elif args.model_type == 'diffcom_1d':
        from models.baseline_new import DiffCom1DBaseline   # [new-data] 重写版，见该文件说明
        model = DiffCom1DBaseline(config, dps_step_size=args.dps_step_size)
        logger.info(f'  Model: DiffCom1DBaseline (DPS), dps_step_size={args.dps_step_size}')
    elif args.model_type == 'icdm_1d':
        from models.baseline_new import ICDMDual1DBaseline   # [new-data] 重写版，见该文件说明
        model = ICDMDual1DBaseline(config, dual_gamma=args.icdm_dual_gamma)
        logger.info(f'  Model: ICDMDual1DBaseline (Dual-Score), gamma={args.icdm_dual_gamma}')
    elif args.model_type == 'mv_cnn':
        # [new-data] 多分支特征 + CNN 对照（瓶颈层交叉注意力注入 MoE 专家特征）
        from models.baseline_new import MultiViewCNNBaseline
        model = MultiViewCNNBaseline(config, base_ch=args.cnn_base_ch, balance_coef=args.balance_loss_coef)
        logger.info('  Model: MultiViewCNNBaseline')
    elif args.model_type == 'dit_reg':
        # [new-data] 同骨干 DiT 的直接回归对照（不做扩散）；--disable_moe 为纯 DiT 回归，否则带 MoE 条件
        from models.baseline_new import DiTRegressionBaseline
        model = DiTRegressionBaseline(config, balance_coef=args.balance_loss_coef)
        logger.info('  Model: DiTRegressionBaseline (moe=%s)' % (not args.disable_moe))
    elif args.model_type == 'cnn':
        # CNN 基线模型（纯监督学习，无扩散过程）
        from models.baseline.cnn_baseline import PhysicsCNNBaseline   # [new-data] 修正导入路径
        model = PhysicsCNNBaseline(
            in_channels=config.data.num_channels,      # 2 (I/Q)
            base_ch=args.cnn_base_ch,                   # 基础通道数 (默认 32)
            signal_length=config.data.signal_length,   # 128
            freq_loss_weight=0.1                       # 频域损失权重
        )
        logger.info('  Model: PhysicsCNNBaseline (CNN监督基线)')
    elif args.model_type == 'mf':
        # 匹配滤波基线（传统信号处理方法）
        from models.baseline.matched_filtering_baseline import MatchedFilteringBaseline
        model = MatchedFilteringBaseline(config)
        logger.info('  Model: MatchedFilteringBaseline (匹配滤波传统基线)')
    elif args.model_type == 'lmmse':
        # LMMSE 基线（自适应滤波方法）
        from models.baseline.lmmse_baseline import LMMSEBaseline
        model = LMMSEBaseline(config)
        logger.info('  Model: LMMSEBaseline (线性最小均方误差基线)')
    else:
        # MoE-DiffIC 主模型 (默认)
        from models import PhysicsMoEModel
        model = PhysicsMoEModel(config)
        logger.info('  Model: PhysicsMoEModel (MoE-DiffIC主模型)')

    model = model.to(device)
    if args.multi_gpu and device.type == 'cuda' and torch.cuda.device_count() > 1 and args.model_type not in ['mf', 'lmmse']:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        logger.info(f'  Multi-GPU: enabled DataParallel with {torch.cuda.device_count()} GPUs')

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    total_params = sum(p.numel() for p in base_model.parameters())
    logger.info(f'  Params: {total_params:,}')              # 打印总参数量

    # 专家名称用于门控日志可读性 (ICDM基线时为空)
    expert_names = list(getattr(base_model, 'active_expert_names', []))

    # ================================================================
    #  损失函数
    # ================================================================
    #  CombinedLoss 组合:
    #    total = w_diff × DiffusionLoss + MoEBalanceLoss
    #  详见 training/losses.py
    # ================================================================
    from training.losses import CombinedLoss
    criterion = CombinedLoss(config)

    # ================================================================
    #  优化器 + 学习率调度 (传统方法无需优化器)
    # ================================================================
    #  AdamW: 权重衰减解耦, 适合 Transformer 架构
    #  LR 调度: Warmup + Cosine Annealing
    #    - 前 warmup_steps 步: 线性升温 0 → lr
    #    - 之后: 余弦退火 lr → 0
    #    lr(step) = lr × 0.5 × (1 + cos(π × progress))
    # ================================================================

    # 传统方法（无可训练参数）跳过优化器创建
    if args.model_type in ['mf', 'lmmse']:
        logger.info('  传统方法，跳过优化器创建（无可训练参数）')
        optimizer = None
        lr_scheduler = None
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )
        # 计算总步数和 warmup 步数 (基于 actual_epochs)
        total_steps = actual_epochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            """
            Warmup + Cosine 学习率调度函数
            --------------------------------
            step < warmup_steps:  lr_ratio = step / warmup_steps  (线性升温)
            step >= warmup_steps: lr_ratio = 0.5 × (1 + cos(π × progress))  (余弦退火)
            """
            if step < warmup_steps:
                # 线性 warmup: 从 ~0 线性增长到 1.0
                return max(step / max(warmup_steps, 1), 1e-6)
            # 余弦退火: progress ∈ [0, 1] → lr_ratio ∈ [1, 0]
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(min(progress, 1.0) * math.pi))

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- EMA 初始化 ----
    ema = EMA(base_model, decay=args.ema_rate)

    def _sample_from_model(noisy_norm, num_steps):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):     # [new-data] 采样可选 bf16
            if isinstance(model, nn.DataParallel):
                out = model.module.sample(noisy_norm, num_steps=num_steps)
            else:
                out = model.sample(noisy_norm, num_steps=num_steps)
        return out.float()

    # ================================================================
    #  data_scale 计算 — 扩散模型输入归一化
    # ================================================================
    #  扩散模型假设输入数据 ~ N(0, 1), 但原始 IQ 信号幅度不确定。
    #  data_scale = std(原始干净信号) 用于将信号归一化到单位方差:
    #    x_norm = x / data_scale
    #  采样后反归一化:
    #    x_recovered = x_norm_pred × data_scale
    #
    #  注意: 从原始未增强的真实样本计算, 避免增强引入的随机偏差
    # ================================================================
    logger.info('  Computing data_scale from raw samples ...')
    data_scale_source = train_dataset
    if hasattr(train_dataset, 'datasets') and len(train_dataset.datasets) > 0:
        # ConcatDataset 场景下只取第一个基础子数据集计算原始尺度，避免重复统计。
        data_scale_source = train_dataset.datasets[0]

    if not hasattr(data_scale_source, 'samples'):
        raise AttributeError('Unable to compute data_scale: dataset has no samples attribute')

    raw_clean = np.concatenate(
        [s['clean'] for s in data_scale_source.samples], axis=0
    )
    data_scale = float(np.std(raw_clean))   # 全局标准差 (标量)
    if data_scale < 1e-6:
        data_scale = 1.0                     # 防护: 避免除以零
    del raw_clean                            # 释放内存
    if args.model_type == 'icdm_1d':         # [new-data] 干扰先验按自身尺度归一化
        _b = np.concatenate([(s_['noisy'] - s_['clean']).ravel() for s_ in data_scale_source.samples])
        base_model.set_interference_scale(float(np.std(_b)) / data_scale)
        logger.info(f'  [ICDM-dual] interference scale (relative to data_scale) = {float(np.std(_b)) / data_scale:.3f}')
        del _b
    logger.info(f'  data_scale = {data_scale:.6f}')

    # ================================================================
    #  [3/4] 训练主循环 - 分支处理
    # ================================================================
    logger.info('[3/4] Training …')

    # ---- 传统方法：两种运行模式 ----
    if args.model_type in ['mf', 'lmmse']:
        if args.pure_traditional:
            # 模式1：纯传统方法，直接在 N100 测试集上运行
            try:
                pure_eval_mod = importlib.import_module('pure_traditional_evaluation')
                evaluate_traditional_method_only = getattr(pure_eval_mod, 'evaluate_traditional_method_only')
            except (ModuleNotFoundError, AttributeError) as e:
                raise ModuleNotFoundError(
                    '缺少 pure_traditional_evaluation.py，无法运行 --pure_traditional。'
                ) from e

            logger.info("纯传统方法模式：直接在测试集上运行（与深度学习评估对齐）")
            logger.info("特点：无需训练，无需学习统计量")

            if test_loader is not None:
                eval_loader = test_loader
                logger.info("使用 static_test_split 对应测试集进行评估")
            elif val_loader is not None:
                eval_loader = val_loader
                logger.warning("未找到 test_loader，回退到 val_loader 评估")
            else:
                raise ValueError('pure_traditional 模式下 test_loader/val_loader 均不可用')

            eval_result = evaluate_traditional_method_only(
                model=model,
                test_loader=eval_loader,
                device=device,
                method_name=args.model_type.upper()
            )

            out_json = os.path.join(log_dir, 'test_results.json')
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(eval_result, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved test results -> {out_json}")

            logger.info("=============================================================")
            logger.info("  纯传统方法评估完成!")
            logger.info(f"  方法: {eval_result['method']}")
            logger.info(f"  测试样本: {eval_result['total_samples']}")
            logger.info(f"  最终SQI: {eval_result['global']['sqi_mean']:+.2f} dB")
            logger.info(f"  最终MSE: {eval_result['global']['mse_mean']:.6f}")
            logger.info("=============================================================")
            return  # 完成，直接返回
        else:
            # 模式2：数据驱动传统方法，从训练集学习统计量
            from traditional_trainer import train_traditional_method

            if val_loader is None:
                raise ValueError('传统方法训练需要验证集，请提供 --static_val_split 或可用 val split。')

            logger.info("传统方法训练：一次性统计学习（无epoch概念）")
            train_result = train_traditional_method(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                method_name=args.model_type.upper(),
                exp_dir=log_dir
            )

            logger.info("============================================================")
            logger.info("  传统方法训练完成!")
            logger.info(f"  方法: {train_result['method']}")
            logger.info(f"  训练样本: {train_result['training_samples']}")
            logger.info(f"  最终SQI: {train_result['performance']['global']['sqi_mean']:+.2f} dB")
            logger.info(f"  最终MSE: {train_result['performance']['global']['mse_mean']:.6f}")
            logger.info("============================================================")
            return  # 传统方法训练完成，直接返回

    # ---- 深度学习方法：多epoch训练循环 ----
    logger.info("深度学习方法训练：多epoch梯度优化")
    global_step = 0                          # 全局训练步数
    best_snr = float('-inf')                 # 历史最优 SNR Improvement (dB)
    best_epoch = -1                          # 最优 epoch 号
    best_metrics = {}                        # 最优时的完整指标
    no_improve_count = 0                     # 无改善计数 (Early Stopping)
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'sqi': []}

    for epoch in range(actual_epochs):
        # ============================================================
        #  训练阶段
        # ============================================================
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{actual_epochs}', leave=False)

        for batch in pbar:
            # ---- 数据准备: 归一化到单位方差 ----
            clean   = batch['clean'].to(device) / data_scale     # (B, 2, 128)
            noisy   = batch['noisy'].to(device) / data_scale     # (B, 2, 128)

            # ---- 前向传播 ----
            # model.forward() 返回 dict:
            #   - noise_pred:    (B, 2, 128) 预测噪声 ε_θ
            #   - noise_target:  (B, 2, 128) 真实噪声 ε
            #   - gates:         (B, num_experts) MoE 门控权重 (仅 MoE 模型)
            #   - balance_loss:  标量 MoE 负载平衡损失 (仅 MoE 模型)
            #   - moe_alpha:     残差门控 α 值 (仅 MoE 模型)
            # 传统方法 (MF/LMMSE) 返回空字典 {}
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(args.amp == 'bf16')):   # [new-data] 可选 bf16 混合精度（只包前向）
                result = model(noisy, clean) if args.model_type == 'cnn' else model(clean, noisy)   # [new-data] CNN 签名为 (noisy, clean)
            if args.amp == 'bf16':                                                           # 损失一律在 fp32 下计算
                result = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in result.items()}

            # ---- 传统方法特殊处理 ----
            if args.model_type in ['mf', 'lmmse']:
                # 传统方法：只更新统计量，无损失计算
                # 设置虚拟损失为0，用于记录和显示
                loss = torch.tensor(0.0, device=device, requires_grad=False)
                loss_dict = {'training_progress': model.samples_seen.item() if hasattr(model, 'samples_seen') else global_step}
            else:
                # 深度学习方法：正常损失计算
                # ---- 计算组合损失 ----
                # total = w_diff × L_diff + L_balance
                loss, loss_dict = criterion(
                    noise_pred=result['noise_pred'],
                    noise_target=result['noise_target'],
                    x0_pred=result.get('x0_pred'),
                    x0_target=result.get('x0_target'),
                    gates=result.get('gates'),                    # ICDM 无此项
                    balance_loss=result.get('balance_loss'),      # ICDM 无此项
                    x0_weight=(base_model.scheduler.alphas_cumprod.to(device)[result['t']]
                               if args.x0_loss_weighting == 'alpha' and 't' in result else None),
                    route_logits=result.get('route_logits') if args.router_type == 'prior' else None,
                    route_target=(route_prior[batch['interference_type'].to(device)]
                                  if args.router_type == 'prior' and route_prior is not None else None),
                    diversity_loss=result.get('diversity_loss') if args.router_type == 'prior' else None,
                )

            if args.model_type in ('cnn', 'dit_reg', 'mv_cnn'):      # [new-data] 直接回归类模型使用自带损失 = MSE + 0.1·频域 MSE（+ MoE 均衡）
                loss = result['loss']
                loss_dict = {'cnn_mse': result['mse_loss'].detach(), 'cnn_freq': result['freq_loss'].detach()}

            # ---- 反向传播 + 梯度裁剪 + 参数更新 (传统方法跳过) ----
            if args.model_type in ['mf', 'lmmse']:
                # 传统方法：无梯度更新，直接记录损失
                pass
            else:
                optimizer.zero_grad()
                loss.backward()
                if args.gradient_clip > 0:
                    # 梯度范数裁剪: 防止扩散训练初期的梯度爆炸
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                lr_scheduler.step()     # 每步更新学习率 (非每 epoch)
            ema.update()         # 每步更新 EMA shadow 参数

            epoch_loss += loss.item()
            global_step += 1

            # ---- TensorBoard 标量记录 (每 10 步) ----
            if global_step % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                # 传统方法无学习率调度器
                if args.model_type not in ['mf', 'lmmse']:
                    writer.add_scalar('Train/LR', lr_scheduler.get_last_lr()[0], global_step)
                # 记录各子损失分量
                for k, v in loss_dict.items():
                    val = v.item() if torch.is_tensor(v) else v
                    writer.add_scalar(f'Train/{k}', val, global_step)
                # MoE 残差门控 α 监控: α 从 0 开始增长, 反映 MoE 逐步介入
                if 'moe_alpha' in result:
                    alpha_val = result['moe_alpha'].detach().mean().item()
                    writer.add_scalar('Train/MoE_Alpha', alpha_val, global_step)
                # MoE 路由利用率监控
                _log_moe_gate_stats(result.get('gates'), 'Train_MoE', global_step)

            # 进度条显示当前 loss 和学习率（传统方法无学习率）
            if args.model_type in ['mf', 'lmmse']:
                pbar.set_postfix(loss=f'{loss.item():.4f}', method='traditional')
            else:
                pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{lr_scheduler.get_last_lr()[0]:.2e}')

        avg_train_loss = epoch_loss / len(train_loader)

        # ============================================================
        #  验证阶段 — 每 epoch 执行, 使用 EMA 参数
        # ============================================================
        model.eval()
        ema.apply_shadow()      # 切换到 EMA 平滑参数

        should_run_val = (val_loader is not None) and ((epoch + 1) % max(1, int(args.val_every)) == 0)

        if args.model_type in ['mf', 'lmmse'] and should_run_val:
            # 传统方法：直接评估性能指标
            try:
                trad_eval_mod = importlib.import_module('traditional_evaluation')
                evaluate_traditional_method = getattr(trad_eval_mod, 'evaluate_traditional_method')
            except (ModuleNotFoundError, AttributeError) as e:
                raise ModuleNotFoundError(
                    '缺少 traditional_evaluation.py，无法评估传统方法。'
                ) from e

            # 创建小批量验证加载器（避免内存过大）
            val_subset = torch.utils.data.Subset(val_loader.dataset, range(0, min(500, len(val_loader.dataset))))
            val_subset_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)

            val_results = evaluate_traditional_method(model, val_subset_loader, device, args.model_type.upper())
            avg_val_loss = val_results['global']['mse_mean']  # 使用MSE作为"验证损失"
            avg_sqi = val_results['global']['sqi_mean']       # SQI指标

            writer.add_scalar('Val/MSE', avg_val_loss, epoch)
            writer.add_scalar('Val/SQI', avg_sqi, epoch)
            logger.info(f'Epoch {epoch}: train_progress={epoch}  val_mse={avg_val_loss:.6f}  val_sqi={avg_sqi:+.2f}dB')
        elif should_run_val:
            # 深度学习方法：正常损失计算
            val_loss_sum = 0.0
            val_gate_sum = None
            val_load_sum = None
            val_gate_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    clean_v   = batch['clean'].to(device) / data_scale
                    noisy_v   = batch['noisy'].to(device) / data_scale
                    result_v  = model(noisy_v, clean_v) if args.model_type == 'cnn' else model(clean_v, noisy_v)

                    gates_v = result_v.get('gates')
                    if gates_v is not None and torch.is_tensor(gates_v) and gates_v.ndim == 2:
                        g = gates_v.detach()
                        if val_gate_sum is None:
                            n_exp = g.shape[1]
                            val_gate_sum = torch.zeros(n_exp, dtype=torch.float64)
                            val_load_sum = torch.zeros(n_exp, dtype=torch.float64)
                        val_gate_sum += g.sum(dim=0).double().cpu()
                        val_load_sum += (g > 0).float().sum(dim=0).double().cpu()
                        val_gate_count += g.shape[0]

                    vl, _ = criterion(
                        noise_pred=result_v['noise_pred'],
                        noise_target=result_v['noise_target'],
                        gates=result_v.get('gates'),
                        balance_loss=result_v.get('balance_loss'),
                    )
                    val_loss_sum += vl.item()
            avg_val_loss = val_loss_sum / len(val_loader)

            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            if val_gate_count > 0:
                val_importance = val_gate_sum / max(val_gate_count, 1)
                val_load = val_load_sum / max(val_gate_count, 1)
                for i in range(val_importance.shape[0]):
                    name = expert_names[i] if i < len(expert_names) else f'E{i}'
                    writer.add_scalar(f'Val_MoE/GateImportance/{name}', float(val_importance[i]), epoch)
                    writer.add_scalar(f'Val_MoE/GateLoad/{name}', float(val_load[i]), epoch)
            logger.info(f'Epoch {epoch}: train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}')
        else:
            avg_val_loss = avg_train_loss
            logger.info(f'Epoch {epoch}: train_loss={avg_train_loss:.6f}  val=SKIPPED')

        # 为后续代码统一avg_val_loss变量（传统方法使用MSE，深度学习方法使用loss）

        # 兼容旧版参数名: 每 save_every 个 epoch 保留一个周期性 checkpoint
        if args.save_every is not None and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            periodic_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            periodic_checkpoint = {
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
            'ema_shadow': ema.shadow,
                'data_scale': data_scale,
                'args': vars(args),
            }
            # 传统方法无优化器状态
            if args.model_type not in ['mf', 'lmmse']:
                periodic_checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(periodic_checkpoint, periodic_path)
            logger.info(f'  Saved periodic checkpoint → {periodic_path}')

        # ============================================================
        #  定期全量评估 + I-Q 星座图 (每 eval_freq 个 epoch)
        # ============================================================
        #  评估内容:
        #    1. DDIM 采样恢复信号 → 计算全局 MSE 和 SNR Improvement
        #    2. 按 SIR 等级分档统计 SNR Improvement (分析不同干扰强度的效果)
        #    3. 记录历史 + 保存最佳模型 (按 sqi)
        #
        #  Signal Quality Improvement (SQI) 定义:
        #    SQI = 10 × log₁₀( ||Y - X||² / ||X̂ - X||² )
        #    其中 Y=noisy, X=clean, X̂=recovered
        #    正值表示恢复信号比原始受损信号更接近真值
        #
        #  注意: 由于 RML2016 clean 信号含 AWGN 噪声, 该指标主要
        #       衡量干扰抑制效果, 而非真实 SNR 提升
        # ============================================================
        if val_loader is not None and (epoch + 1) % args.eval_freq == 0:
            sir_metrics = {}   # {sir_db: {'sqi': []}} — SIR 分档统计
            ber_list, ber_noisy_list, sir_ber = [], [], {}   # [new-data] BER（恢复后 / 不处理）
            mse_list    = []   # 每 batch 的全局 MSE
            sqi_list    = []   # 每 batch 的全局 SQI (Signal Quality Improvement)

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    clean_e   = batch['clean'].to(device)       # (B, 2, 128) 原始尺度
                    noisy_e   = batch['noisy'].to(device)       # (B, 2, 128) 原始尺度
                    sir_vals  = batch.get('sir_db', [0.0] * clean_e.shape[0])

                    # ---- DDIM 采样恢复信号 ----
                    # 输入归一化 → DDIM 采样 → 反归一化回原始尺度
                    noisy_norm = noisy_e / data_scale
                    recovered_norm = _sample_from_model(noisy_norm, num_steps=args.ddim_steps)
                    recovered = recovered_norm * data_scale     # (B, 2, 128) 原始尺度

                    # ---- 全局指标计算 ----
                    # MSE: ||X̂ - X||² 的 batch 均值
                    mse = ((recovered - clean_e) ** 2).mean().item()
                    # noise_before: ||Y - X||² = 输入干扰+噪声功率
                    noise_before = ((noisy_e - clean_e) ** 2).mean().item()
                    # Signal Quality Improvement (SQI): 10 × log₁₀( ||Y-X||² / ||X̂-X||² )
                    sqi = 10 * np.log10(noise_before / (mse + 1e-10))
                    mse_list.append(mse)
                    sqi_list.append(sqi)

                    # ---- 逐样本计算 SQI (按 SIR 分档) ----
                    for b in range(clean_e.shape[0]):
                        sir_key  = round(float(sir_vals[b]))   # 四舍五入到整数 dB
                        s_clean  = clean_e[b]                  # (2, L)
                        s_rec    = recovered[b]                # (2, L)
                        s_noisy  = noisy_e[b]                  # (2, L)

                        s_mse    = ((s_rec - s_clean) ** 2).mean().item()         # ||X̂-X||²
                        s_nb     = ((s_noisy - s_clean) ** 2).mean().item()       # ||Y-X||²
                        s_sqi    = 10 * np.log10(s_nb / (s_mse + 1e-10))         # SQI

                        sir_metrics.setdefault(sir_key, {'sqi': []})
                        sir_metrics[sir_key]['sqi'].append(s_sqi)

                    if new_meta is not None:
                        b_rec = batch_ber(recovered, batch, new_meta, mod_names)
                        b_noi = batch_ber(noisy_e, batch, new_meta, mod_names)
                        ber_list.extend(b_rec.tolist())
                        ber_noisy_list.extend(b_noi.tolist())
                        for b in range(len(b_rec)):
                            sir_ber.setdefault(round(float(sir_vals[b])), []).append(float(b_rec[b]))

                    if args.val_max_batches > 0 and i + 1 >= args.val_max_batches:
                        break

            # ---- 汇总全局指标 ----
            avg_mse     = float(np.mean(mse_list))
            avg_sqi     = float(np.mean(sqi_list))
            writer.add_scalar('Val/MSE',    avg_mse, epoch)
            writer.add_scalar('Val/SQI_dB', avg_sqi, epoch)
            # [new-data] 最佳模型的选择分数（越大越好）：-BER 或 SQI
            avg_ber = float(np.mean(ber_list)) if ber_list else None
            avg_ber_noisy = float(np.mean(ber_noisy_list)) if ber_noisy_list else None
            if args.select_metric == 'ber' and avg_ber is not None:
                cur_score = -avg_ber
            else:
                cur_score = avg_sqi
            if avg_ber is not None:
                writer.add_scalar('Val/BER', avg_ber, epoch)
                writer.add_scalar('Val/BER_noisy_input', avg_ber_noisy, epoch)

            # ---- SIR 分档写入 TensorBoard ----
            sir_avg = {
                k: {'sqi': float(np.mean(v['sqi']))}
                for k, v in sorted(sir_metrics.items())
            }
            for sir_key, sir_vals_d in sir_avg.items():
                writer.add_scalar(f'Val_SIR/SIR_{sir_key}dB_SQI', sir_vals_d['sqi'], epoch)

            logger.info(
                f'Epoch {epoch}: [EVAL] MSE={avg_mse:.6f}, '
                f'SQI={avg_sqi:.4f} dB'
                + (f', BER={avg_ber:.4f} (不处理直接解调: {avg_ber_noisy:.4f})' if avg_ber is not None else '')
            )
            if sir_ber:
                logger.info('  SIR BER: ' + ', '.join(f'{k}dB→{np.mean(v):.3f}' for k, v in sorted(sir_ber.items())))
            sir_str = ', '.join(
                f'{k}dB→SQI:{v["sqi"]:.2f}'
                for k, v in sir_avg.items()
            )
            logger.info(f'  SIR breakdown: {sir_str}')

            # ---- 记录训练历史 (用于后续绘图) ----
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['sqi'].append(avg_sqi)

            # ============================================================
            #  最佳模型保存策略
            # ============================================================
            #  条件: epoch >= 30 (min_eval_epoch, 避免保存训练初期不稳定的模型)
            #        且 sqi > 历史最优
            #  保存内容: model_state_dict, optimizer, ema_shadow, data_scale, args
            #  用途: 训练结束后加载 best_model.pth 做最终评估
            # ============================================================
            if epoch >= args.min_best_epoch and cur_score > best_snr:
                best_snr = cur_score          # [new-data] 现为选择分数：-BER 或 SQI
                best_epoch = epoch
                best_metrics = {
                    'mse': avg_mse,
                    'sqi': avg_sqi,
                    'ber': avg_ber,
                    'select_metric': args.select_metric,
                    'sir_metrics': sir_avg,
                    'epoch': epoch,
                }
                save_path = os.path.join(log_dir, 'best_model.pth')
                best_checkpoint = {
                    'epoch': epoch,                              # 最优 epoch
                    'model_state_dict': base_model.state_dict(),      # 在线参数
                    'ema_shadow': ema.shadow,                    # EMA 影子参数
                    'data_scale': data_scale,                    # 归一化因子
                    'best_snr': best_snr,                        # 最优 sqi
                    'best_metrics': best_metrics,                # 完整指标
                    'args': vars(args),                          # 实验参数 (复现用)
                }
                # 传统方法无优化器状态
                if args.model_type not in ['mf', 'lmmse']:
                    best_checkpoint['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(best_checkpoint, save_path)
                logger.info(f'  ★ Saved best model → {save_path}')
                no_improve_count = 0                             # 重置 ES 计数器
            elif epoch >= args.min_best_epoch:
                # 未改善: 递增 Early Stopping 计数
                no_improve_count += 1
                if args.early_stopping_patience < 9999:
                    logger.info(
                        f'  No improve {no_improve_count}/{args.early_stopping_patience} '
                        f'(best score={best_snr:.4f} [{args.select_metric}] @ ep{best_epoch})'
                    )

            # ---- Early Stopping: 连续多轮无改善则终止训练 ----
            if no_improve_count >= args.early_stopping_patience:
                logger.info(f'  ★ Early stopping at epoch {epoch}!')
                break

        # 评估结束后恢复在线参数, 继续下一 epoch 训练
        ema.restore()

    # ================================================================
    #  训练结束: 关闭 TensorBoard, 打印摘要
    # ================================================================
    writer.close()
    logger.info('=' * 60)
    logger.info('  Training Completed!')
    logger.info(f'  Best Epoch: {best_epoch}')
    logger.info(f'  Best score: {best_snr:.4f}  (select_metric={args.select_metric}; ber 模式下为 -BER)')
    logger.info('=' * 60)

    # 无验证集场景: 回退保存最后一轮参数为 best_model，保证最终测试可执行。
    ckpt_path = os.path.join(log_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        fallback_metrics = {
            'mse': None,
            'sqi': None,
            'sir_metrics': {},
            'epoch': actual_epochs - 1,
            'source': 'fallback_last_epoch_no_val',
        }
        fallback_checkpoint = {
            'epoch': actual_epochs - 1,
            'model_state_dict': base_model.state_dict(),
            'ema_shadow': ema.shadow,
            'data_scale': data_scale,
            'best_snr': best_snr,
            'best_metrics': fallback_metrics,
            'args': vars(args),
        }
        if args.model_type not in ['mf', 'lmmse'] and optimizer is not None:
            fallback_checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(fallback_checkpoint, ckpt_path)
        logger.info('  [Fallback] No best checkpoint from validation; saved last-epoch model as best_model.pth')

    # ================================================================
    #  [4/4] 最终评估: 加载最佳模型 → 全测试集 → test_results.json
    # ================================================================
    #  最终评估与训练中的定期评估不同:
    #    - 遍历 **全部** val_loader batch (非仅 16 个)
    #    - 同时统计 SIR 分档 和 干扰类型分档
    #    - 结果保存为 JSON, 供 evaluate_standard.py / plot 脚本读取
    # ================================================================
    logger.info('[4/4] Final evaluation on full test set …')
    if not os.path.exists(ckpt_path):
        logger.warning('  No best_model.pth found, skipping final eval.')
        return

    # ---- 加载最佳模型 checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt['model_state_dict'])     # 加载在线参数

    # 用 EMA shadow 参数替换模型参数 (推理时使用更平滑的 EMA 参数)
    for name, p in base_model.named_parameters():
        if name in ckpt.get('ema_shadow', {}):
            p.data = ckpt['ema_shadow'][name].to(device)
    model.eval()

    eval_sets = [('test', test_loader if test_loader is not None else val_loader)]
    if ood_loader is not None:
        eval_sets.append(('test_ood', ood_loader))   # [new-data] 未见干扰参数
    for eval_tag, eval_loader in eval_sets:
        logger.info(f'  === 最终评估: {eval_tag} ===')
        # ---- 收集结果的字典 ----
        sir_results  = {}   # {sir_db: {'mse': [], 'sqi': []}} — 按 SIR 分档
        type_results = {}   # {interf_type: 同上} — 按干扰类型分档
        mod_results = {}    # [new-data] 按调制分档
        gate_rows = []      # [new-data] 路由分析：(类型下标, SIR, 门控向量)
        global_ber, global_ber_noisy = [], []

        if eval_loader is None:
            logger.warning(f'  No eval loader for {eval_tag}, skipping.')
            continue
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc='Final eval'):
                clean_e   = batch['clean'].to(device)       # (B, 2, 128)
                noisy_e   = batch['noisy'].to(device)       # (B, 2, 128)
                sir_vals  = batch.get('sir_db', [0.0] * clean_e.shape[0])
                type_vals = batch.get('interference_type', [0] * clean_e.shape[0])

                # DDIM 采样: 归一化 → 采样 → 反归一化
                noisy_norm = noisy_e / data_scale
                recovered = _sample_from_model(noisy_norm, num_steps=args.ddim_steps) * data_scale

                if new_meta is not None and not args.disable_moe and hasattr(base_model, 'compute_moe_condition'):
                    _, _minfo = base_model.compute_moe_condition(noisy_norm, training=False)
                    if _minfo['gates'] is not None:
                        for _b in range(noisy_e.shape[0]):
                            gate_rows.append((int(batch['interference_type'][_b]), round(float(batch['sir_db'][_b])),
                                              _minfo['gates'][_b].detach().cpu().numpy()))

                if new_meta is not None:
                    ber_rec = batch_ber(recovered, batch, new_meta, mod_names)
                    ber_noi = batch_ber(noisy_e, batch, new_meta, mod_names)
                    mod_vals = batch['modulation']

                # ---- 逐样本统计 ----
                for b in range(clean_e.shape[0]):
                    s_clean = clean_e[b]
                    s_rec   = recovered[b]
                    s_noisy = noisy_e[b]

                    # MSE / SNR Improvement
                    s_mse  = float(((s_rec - s_clean) ** 2).mean().item())
                    s_nb   = float(((s_noisy - s_clean) ** 2).mean().item())
                    s_snr  = 10 * np.log10(s_nb / (s_mse + 1e-10))

                    # ---- 按 SIR 分档 ----
                    sir_key = round(float(sir_vals[b]))
                    sir_results.setdefault(sir_key, {'mse': [], 'sqi': []})
                    sir_results[sir_key]['mse'].append(s_mse)
                    sir_results[sir_key]['sqi'].append(s_snr)

                    # ---- 按干扰类型分档 ----
                    # type_vals 可能是整数索引或字符串, 统一转为字符串名称
                    _INT_NAMES = int_names   # [new-data] 与数据集一致（原为写死的旧顺序）
                    tid = int(type_vals[b]) if not isinstance(type_vals[b], str) else -1
                    tkey = _INT_NAMES[tid] if 0 <= tid < len(_INT_NAMES) else str(type_vals[b])
                    type_results.setdefault(tkey, {'mse': [], 'sqi': []})
                    type_results[tkey]['mse'].append(s_mse)
                    type_results[tkey]['sqi'].append(s_snr)

                    # ---- [new-data] BER 与按调制分档 ----
                    if new_meta is not None:
                        mkey = mod_names[int(mod_vals[b])]
                        mod_results.setdefault(mkey, {'mse': [], 'sqi': [], 'ber': [], 'ber_noisy': []})
                        mod_results[mkey]['mse'].append(s_mse)
                        mod_results[mkey]['sqi'].append(s_snr)
                        for d_ in (sir_results[sir_key], type_results[tkey], mod_results[mkey]):
                            d_.setdefault('ber', []).append(float(ber_rec[b]))
                            d_.setdefault('ber_noisy', []).append(float(ber_noi[b]))
                        global_ber.append(float(ber_rec[b]))
                        global_ber_noisy.append(float(ber_noi[b]))

        # ================================================================
        #  汇总并保存 test_results.json
        # ================================================================
        #  JSON 结构:
        #    {
        #      "exp_name": "moe_5shot_aug",
        #      "model_type": "moe_diffmic",
        #      "sir_results": {
        #        "-10": {
        #          "mse_mean": ..., "sqi_mean": ..., "sqi_std": ...,
        #          "count": ...
        #        }, ...
        #      },
        #      ...
        #    }
        # ================================================================
        test_results = {
            'exp_name': args.exp_name,
            'model_type': args.model_type,
            'use_geometric_aug': args.use_geometric_aug,
            'num_shots': args.num_shots,
            'best_epoch': best_epoch,
            'best_snr': best_snr,
            # SIR 分档结果: 按 SIR 值排序
            'sir_results': {
                str(k): {
                    'mse_mean':    float(np.mean(v['mse'])),      # 平均 MSE
                    'sqi_mean':    float(np.mean(v['sqi'])),      # 平均 SQI (dB)
                    'sqi_std':     float(np.std(v['sqi'])),       # SQI 标准差
                    'count':       len(v['sqi']),                  # 样本数
                }
                for k, v in sorted(sir_results.items())
            },
            # 干扰类型分档结果: 按类型名排序
            'type_results': {
                k: {
                    'mse_mean':    float(np.mean(v['mse'])),      # 平均 MSE
                    'sqi_mean':    float(np.mean(v['sqi'])),      # 平均 SQI (dB)
                    'count':       len(v['sqi']),                  # 样本数
                }
                for k, v in sorted(type_results.items())
            },
        }

        test_results['eval_set'] = eval_tag
        if gate_rows:
            _G = np.stack([r[2] for r in gate_rows]); _T = np.array([r[0] for r in gate_rows]); _S = np.array([r[1] for r in gate_rows])
            _en = list(config.moe.active_experts)
            _prior = None
            try:
                from models.prior_router import build_prior_table
                _prior = build_prior_table(int_names, _en).numpy()
            except Exception:
                pass
            test_results['routing'] = {
                'experts': _en,
                'router_type': args.router_type,
                'gate_by_type': {int_names[t]: _G[_T == t].mean(0).round(4).tolist() for t in sorted(set(_T.tolist()))},
                'gate_by_sir': {str(s_): _G[_S == s_].mean(0).round(4).tolist() for s_ in sorted(set(_S.tolist()))},
                'gate_entropy_mean': float((-(np.clip(_G, 1e-8, 1) * np.log(np.clip(_G, 1e-8, 1))).sum(1)).mean()),
                'active_experts_mean': float((_G > 1e-6).sum(1).mean()),
            }
            if _prior is not None:
                _hit = (_G.argmax(1) == _prior.argmax(1)[_T])
                test_results['routing']['top1_matches_prior'] = float(_hit.mean())
                test_results['routing']['top1_matches_prior_by_type'] = {int_names[t]: float(_hit[_T == t].mean()) for t in sorted(set(_T.tolist()))}
                _typ = np.stack([_G[_T == t].mean(0) for t in sorted(set(_T.tolist()))])
                test_results['routing']['type_gate_max_l1'] = float(np.abs(_typ[:, None] - _typ[None]).sum(-1).max())
            logger.info('  [%s] 路由: top1 与先验吻合 %s | 非零专家数 %.2f | 类型间门控最大L1距离 %s' % (
                eval_tag, ('%.3f' % test_results['routing'].get('top1_matches_prior')) if 'top1_matches_prior' in test_results['routing'] else 'n/a',
                test_results['routing']['active_experts_mean'], ('%.3f' % test_results['routing'].get('type_gate_max_l1')) if 'type_gate_max_l1' in test_results['routing'] else 'n/a'))
        if new_meta is not None:
            def _agg(d):
                return {str(k): {'mse_mean': float(np.mean(v['mse'])), 'sqi_mean': float(np.mean(v['sqi'])),
                                 'ber_mean': float(np.mean(v['ber'])), 'ber_noisy_input_mean': float(np.mean(v['ber_noisy'])),
                                 'count': len(v['sqi'])} for k, v in sorted(d.items())}
            test_results['sir_results'] = _agg(sir_results)
            test_results['type_results'] = _agg(type_results)
            test_results['mod_results'] = _agg(mod_results)
            test_results['global'] = {'ber_mean': float(np.mean(global_ber)), 'ber_noisy_input_mean': float(np.mean(global_ber_noisy)),
                                      'count': len(global_ber)}
            test_results['select_metric'] = args.select_metric
            test_results['data_backend'] = args.data_backend
            test_results['noise_mode'] = args.noise_mode
        out_json = os.path.join(log_dir, 'test_results.json' if eval_tag == 'test' else f'{eval_tag}_results.json')
        with open(out_json, 'w') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        logger.info(f'  Saved test results → {out_json}')

        # ---- 在日志中打印 SIR 分档概要 ----
        logger.info(f'  [{eval_tag}] 汇总')
        for sir_key in sorted(sir_results.keys()):
            v = sir_results[sir_key]
            logger.info(
                f'  SIR_in={sir_key:+3d} dB: '
                f'MSE={np.mean(v["mse"]):.6f}, '
                f'SQI={np.mean(v["sqi"]):+.2f} dB'
                + (f', BER={np.mean(v["ber"]):.4f} (不处理: {np.mean(v["ber_noisy"]):.4f})' if v.get('ber') else '')
            )


# ============================================================
#  程序入口
# ============================================================
#  使用方法:
#    # ICDM 基线 (无增强)
#    python train.py --model_type icdm --num_shots 5 --exp_name icdm_5shot
#
#    # MoE-DiffIC (几何增强)
#    python train.py --model_type moe_diffmic --num_shots 5 \
#                    --use_geometric_aug --exp_name moe_5shot_aug
#
#    # Full-Data 训练
#    python train.py --model_type moe_diffmic --num_shots -1 \
#                    --use_geometric_aug --exp_name moe_full_aug
# ============================================================

if __name__ == '__main__':
    args = parse_args()   # 解析命令行参数
    train(args)           # 执行完整训练流程











