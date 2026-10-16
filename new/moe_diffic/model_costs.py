"""各模型配置的参数量 / 单次去噪网络前向 FLOPs / 25 步采样的单样本延迟（随机初始化即可，不需要训练）。
对应 plan.md 的“报告总参数与去噪器参数、FLOPs/GPU 小时”。输出 CSV，final_report 的 E7 表引用。
用法: python model_costs.py --out ../reports/model_costs.csv [--steps 25] [--bs 16]"""
import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'loader')); sys.path.insert(0, os.path.join(HERE, 'exp'))
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import exp_train as ET  # noqa: E402
import registry as REG  # noqa: E402
import train as T  # noqa: E402


def build(args_str, model_type='moe_diffmic'):
    a = ET.get_args(['--model_type', model_type, '--exp_name', 'cost'] + args_str.split())
    cfg = T.build_config(a); cfg.data.signal_length = 2560; cfg.data.samples_per_symbol = cfg.data.sps = 4; cfg.data.rrc_span = 8
    return a, ET.build_model_for_training(a, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--steps', type=int, default=25)
    ap.add_argument('--bs', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    specs = dict(REG.CFG)
    specs['cnn'] = ('cnn', ''); specs['res_diff_tf'] = ('res_diff', '--experts TF --base_ckpt x')
    rows = []
    for name, v in specs.items():
        mt, args_str = v if isinstance(v, tuple) else ('moe_diffmic', v)
        try:
            if mt == 'res_diff':
                continue
            _, m = build(args_str, mt)
            m = m.to(dev).eval()
            n_par = sum(p.numel() for p in m.parameters())
            n_den = sum(p.numel() for p in m.denoiser.parameters()) if hasattr(m, 'denoiser') else 0
            y = torch.randn(a.bs, 2, 2560, device=dev)
            flops = ms = float('nan')
            if mt == 'moe_diffmic':
                x = torch.randn(1, 2, 2560, device=dev); t = torch.zeros(1, dtype=torch.long, device=dev)
                with torch.no_grad():
                    cond, info = m.compute_moe_condition(y[:1], training=False)
                    with FlopCounterMode(display=False) as fc:
                        m.denoise(x, y[:1], t, cond, info.get('cond_mask'))
                    flops = fc.get_total_flops() / 1e9
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                        m.sample(y, num_steps=2)
                        torch.cuda.synchronize(); t0 = time.time()
                        m.sample(y, num_steps=a.steps)
                        torch.cuda.synchronize(); ms = (time.time() - t0) / a.bs * 1000
            rows.append(dict(config=name, params_M=round(n_par / 1e6, 3), denoiser_params_M=round(n_den / 1e6, 3), denoiser_gflops_per_step=round(flops, 2), sample_ms_per_sample=round(ms, 2)))
            print(rows[-1], flush=True)
        except Exception as e:
            print(name, 'FAILED', repr(e)[:100])
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
