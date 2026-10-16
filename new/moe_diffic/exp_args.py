"""实验扩展参数（在 train.build_parser() 之上追加）。所有默认值都对应“原行为”。"""


def add_exp_args(p):
    g = p.add_argument_group('experiment extensions (plan.md)')
    # --- 去噪主干 / 条件注入（实验 7、9）
    g.add_argument('--backbone', choices=['dit', 'unet1d', 'diffwave'], default='dit', help='扩散去噪主干')
    g.add_argument('--unet_base', type=int, default=64, help='UNet1D 基础通道数（64≈11.3M 参数，48≈6.4M）')
    g.add_argument('--diffwave_res', type=int, default=128, help='DiffWave 残差通道数（128≈3.7M，192≈8M）')
    g.add_argument('--diffwave_layers', type=int, default=20)
    g.add_argument('--cond_mode', choices=['xattn', 'film', 'adaln', 'none'], default='xattn', help='MoE 条件注入方式')
    g.add_argument('--true_uncond', action='store_true', help='关闭 MoE 时不传伪零 token（真正的无条件分支）')
    # --- 专家 / 路由 / 融合（实验 1-5、8）
    g.add_argument('--shared_expert', action='store_true', help='实验 2：加入始终激活的共享专家')
    g.add_argument('--tf_tokens', type=int, default=1, help='实验 8：TF 专家输出的局部 token 数（1=全局，原行为）')
    g.add_argument('--fusion', choices=['xattn', 'uniform', 'sum_single', 'topk_mask', 'anchor'], default='xattn', help='实验 5：专家条件融合方式；anchor=锚定残差融合(TF 为锚，其余专家以零初始化的可学习标量/动态门控累加，起点等价于 TF 单专家)')
    g.add_argument('--dual_variant', choices=['dual', 'dual_noint', 'morph', 'strength', 'oracle_sir'], default='dual', help='实验 4：双因素路由变体（router_type=dual 时）')
    g.add_argument('--dual_tau', type=float, default=0.5)
    # --- 物理属性辅助监督（实验 3）与 SIR 估计辅助监督（实验 4）
    g.add_argument('--attr_mode', choices=['none', 'joint', 'pretrain_freeze', 'pretrain_ft'], default='none', help='实验 3：物理属性辅助监督方式')
    g.add_argument('--attr_weight', type=float, default=0.1)
    g.add_argument('--attr_pretrain_epochs', type=int, default=20, help='pretrain_* 模式下仅用属性损失预训练专家的轮数')
    g.add_argument('--sir_weight', type=float, default=0.1, help='双因素路由 SIR 估计 NLL 权重')
    g.add_argument('--morph_attr_weight', type=float, default=0.1, help='双因素路由形态分支属性预测权重')
    # --- 数据：逐类留出 / 物理增强（实验 6、14）
    g.add_argument('--holdout', nargs='+', default=None, help='实验 6：留出的干扰类型（训练与验证都不含它们；评估时单独报告）')
    g.add_argument('--phys_aug', type=str, default='none', help="实验 14：物理增强算子，如 'rot,remix'（none 关闭）")
    # --- 残差扩散（实验 10、11）
    g.add_argument('--base_ckpt', type=str, default=None, help='实验 10：冻结的基础恢复器 CNN 的 run 名（runs/<名>/best_model.pth）')
    g.add_argument('--base_oof', type=str, default=None, help='实验 10：样本外基础估计缓存 npz（make_oof_base.py 生成）；缺省则在线计算（偏乐观）')
    g.add_argument('--oof_fold', type=int, default=-1, help='K 折样本外预测：训练时排除 clean_id%%oof_folds==oof_fold 的训练样本，结束后对它们预测并保存 oof_fold.npz')
    g.add_argument('--oof_folds', type=int, default=5)
    g.add_argument('--target', choices=['clean', 'interf'], default='clean', help='回归类模型的学习目标：clean=干净信号（默认）；interf=干扰信号（实验 12 的独立干扰估计器）')
    g.add_argument('--restormer_dim', type=int, default=48, help='Restormer 基线的基础通道数（官方默认 48，约 22.8M 参数）')
    g.add_argument('--dps_warm_start', action='store_true', help='DiffCom 基线：官方式热启动（从接收信号 y 加噪起步，起始步由训练集平均 SINR 按官方公式给出）')
    g.add_argument('--loss_variant', choices=['default', 'eps_only', 'mrstft_mag', 'mrstft_cplx'], default='default',
                   help='实验 D2：default=现有损失(ε + 0.5·ᾱ·SmoothL1(x̂0) + 0.05·ᾱ·单分辨率对数幅度谱)；eps_only=只有 ε 损失；'
                        'mrstft_mag=把单分辨率谱项换成多分辨率(谱收敛+对数幅度)；mrstft_cplx=在 mrstft_mag 上再加多分辨率复数谱 L1')
    g.add_argument('--stft_weight', type=float, default=0.1, help='实验 D2：多分辨率 STFT 项的权重（逐样本乘 ᾱ_t）')
    g.add_argument('--oracle_cond', choices=['none', 'type', 'attrs', 'type_attrs'], default='none',
                   help='诊断实验(oracle，仅作信息上界，不可部署)：把真实干扰类型 one-hot 和/或 12 维干扰物理属性经 MLP 变成条件 token 送入去噪网络')
    g.add_argument('--pretrain_pool', type=str, default=None, help='干净信号预训练：干净波形池 npz 路径（gen/make_clean_pool.py 生成）或 train420（只用本次 10% 训练对里的干净信号，作对照）')
    g.add_argument('--pretrain_steps', type=int, default=40000, help='干净信号预训练的更新步数')
    g.add_argument('--init_from', type=str, default=None, help='微调：用 runs/<名>/best_model.pth 里预训练的 EMA 权重初始化去噪网络（denoiser.*，逐张量核对，不允许缺失/形状不符）')
    g.add_argument('--extra_local', choices=['none', 'oracle_mask', 'e1'], default='none',
                   help='实验 E1：额外的局部输入通道(16 个干扰时频子带功率，随时间变化)。oracle_mask=真实干扰(上界诊断)；e1=由 y 预测(训练时用真实干扰做稠密辅助监督)')
    g.add_argument('--e1_weight', type=float, default=0.1, help='E1 辅助监督(MSE)权重')
    # --- 训练/验证细节
    g.add_argument('--val_stride', type=int, default=1, help='验证集每隔多少条取一条（加速）')
    g.add_argument('--skip_final_eval', action='store_true', help='训练结束不做最终的快速测试评估')
    g.add_argument('--abort_chance_frac', type=float, default=0.5, help='训练进度超过该比例时验证 BER 仍≥0.45（机会水平）则中止')
    g.add_argument('--keep_ckpt', action='store_true', help='保留 best_model.pth（默认由任务清单在评估后按需删除）')
    return p
