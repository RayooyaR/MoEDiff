import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, '..', 'loader'))
import torch
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_cfg(argv=()):
    """用 exp_train 的解析器构造 config（只取默认值 + 给定覆盖）。"""
    import train as T
    from exp_args import add_exp_args
    p = T.build_parser(); add_exp_args(p)
    args = p.parse_args(['--model_type', 'moe_diffmic', '--exp_name', 'unit'] + list(argv))
    cfg = T.build_config(args); cfg.data.signal_length = 2560; cfg.data.samples_per_symbol = cfg.data.sps = 4; cfg.data.rrc_span = 8
    return args, cfg


def randomize(m, std=0.05, seed=0):
    """DiT/UNet 的若干层是零初始化（训练开始时条件无影响）。单元测试需要“条件真有影响”的网络，这里给零参数加小随机值。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            if p.numel() and p.abs().max() == 0:
                p.copy_((torch.randn(p.shape, generator=g) * std).to(p.device))
    return m
