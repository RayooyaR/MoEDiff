"""把实验扩展参数写入 config（build_config 末尾调用）。所有字段用 getattr 取默认值：旧 checkpoint 的 args 缺这些字段时，
重建出的模型与原来逐位一致。参数定义见 exp_args.py。"""


def extend_config(config, args):
    g = lambda k, d: getattr(args, k, d)
    m, d = config.moe, config.denoiser
    m.cond_mode = g('cond_mode', 'xattn')
    m.true_uncond = bool(g('true_uncond', False))
    m.shared_expert = bool(g('shared_expert', False))
    m.tf_tokens = int(g('tf_tokens', 1))
    m.fusion = g('fusion', 'xattn')
    m.oracle_cond = g('oracle_cond', 'none')
    m.extra_local = g('extra_local', 'none')
    m.attr_heads = g('attr_mode', 'none') != 'none'
    m.dual_variant = g('dual_variant', 'dual')
    m.dual_tau = float(g('dual_tau', 0.5))
    d.backbone = g('backbone', 'dit')
    if g('extra_local', 'none') != 'none':
        d.in_channels = 4 + 16                     # [x_t, y] + 16 个局部干扰定位通道
    d.unet_base = int(g('unet_base', 64))
    d.diffwave_res = int(g('diffwave_res', 128))
    d.diffwave_layers = int(g('diffwave_layers', 20))
    return config


def build_oracle(batch, mode, device, n_types=8):
    """诊断用 oracle 条件向量：真实干扰类型 one-hot（8 维）和/或 12 维干扰物理属性（已按训练集统计量标准化）。"""
    import torch
    parts = []
    if 'type' in mode:
        parts.append(torch.nn.functional.one_hot(batch['interference_type'].long(), n_types).float())
    if 'attrs' in mode:
        parts.append(batch['attrs'].float())
    return torch.cat(parts, dim=1).to(device)
