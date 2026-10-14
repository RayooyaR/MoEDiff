"""试点数据集的全部参数。改这里即可，生成与验收脚本都读这个文件。

约定（全部按归一化采样率 fs=1 的复基带信号）：
  - 信号带宽半宽 fmax = (1 + rolloff) / (2 * SPS)  [cycles/sample]
  - "带内" = |f| <= fmax
  - SIR / SNR 全部按带内功率定义（见 README.md）
"""

SEED = 20261010

# ---- 信号 ----
SIG_LEN = 2560           # 采样点数
SPS = 4                  # 过采样率（点/符号）
N_SYM = SIG_LEN // SPS   # 640 个符号（bit 数 = 640 * 每符号比特数）
ROLLOFF = (0.25, 0.50)   # RRC roll-off 均匀采样范围
RRC_SPAN = 8             # RRC 滤波器跨度（符号数）
CFO_MAX = 0.002          # 载波频偏范围 ±(cycles/sample)，约为符号率的 ±0.8%
EDGE_SKIP = 6            # 评估 BER 时忽略首尾符号数（滤波器边缘效应）

# ---- 调制 ----
MODS = ['QPSK', 'BPSK', '16QAM']
BITS_PER_SYM = {'BPSK': 1, 'QPSK': 2, '16QAM': 4}

# ---- 噪声 ----
SNR_DB_RANGE = (10.0, 30.0)   # 带内 SNR，均匀采样

# ---- 干扰 ----
TYPES = ['ST', 'MT', 'PL', 'LFM', 'NB', 'WB', 'NFM', 'RFM']
SIR_DB = [-20, -15, -10, -5, 0, 5, 10]   # 带内 SIR 标准档位
MIN_INBAND_FRAC = 0.2    # 干扰带内能量占比下限，不足则重采样参数

# ---- 划分（每个 类型×调制×SIR 格子内的样本数）----
# 每条样本都使用独立的干净信号；clean_id 区段互不重叠，因此各 split 的干净信号天然不相交。
SPLITS = {
    # name:      (id_base,  n_per_cell, interference parameter regime)
    'train':    (0,        40, 'A'),
    'val':      (100000,    5, 'A'),
    'test':     (200000,   40, 'A'),
    'test_ood': (300000,   20, 'B'),   # 干扰参数取留出区间
}
