"""冒烟：对 exp_train.py 的每个新选项各跑 2 个 epoch（极小数据），确认能训练、验证、保存并被 evaluate 路径重建。
用法: python tests/smoke_variants.py [过滤关键字]"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PY = sys.executable
BASE = ['--lr', '3e-4', '--batch_size', '16', '--num_workers', '0', '--ddim_steps', '3', '--eval_freq', '1', '--val_every', '1', '--warmup_epochs', '0',
        '--epochs', '2', '--seed', '1', '--new_types', 'ST', 'LFM', 'PL', '--train_per_cell', '1', '--val_per_cell', '1', '--val_stride', '2', '--amp', 'bf16', '--abort_chance_frac', '9']
V = {
    'moe_default':   ['--model_type', 'moe_diffmic'],
    'tf_only':       ['--model_type', 'moe_diffmic', '--experts', 'TF'],
    'true_uncond':   ['--model_type', 'moe_diffmic', '--disable_moe', '--true_uncond'],
    'unet1d':        ['--model_type', 'moe_diffmic', '--experts', 'TF', '--backbone', 'unet1d', '--unet_base', '32'],
    'unet1d_noMoE':  ['--model_type', 'moe_diffmic', '--disable_moe', '--true_uncond', '--backbone', 'unet1d', '--unet_base', '32'],
    'diffwave':      ['--model_type', 'moe_diffmic', '--experts', 'TF', '--backbone', 'diffwave', '--diffwave_res', '64', '--diffwave_layers', '8'],
    'film':          ['--model_type', 'moe_diffmic', '--experts', 'TF', '--cond_mode', 'film'],
    'adaln':         ['--model_type', 'moe_diffmic', '--experts', 'TF', '--cond_mode', 'adaln'],
    'unet_film':     ['--model_type', 'moe_diffmic', '--experts', 'TF', '--backbone', 'unet1d', '--unet_base', '32', '--cond_mode', 'film'],
    'tf_tokens16':   ['--model_type', 'moe_diffmic', '--experts', 'TF', '--tf_tokens', '16'],
    'tf_tok_unet':   ['--model_type', 'moe_diffmic', '--experts', 'TF', '--tf_tokens', '16', '--backbone', 'unet1d', '--unet_base', '32'],
    'shared_tf':     ['--model_type', 'moe_diffmic', '--experts', 'TF', '--shared_expert'],
    'shared_all':    ['--model_type', 'moe_diffmic', '--shared_expert'],
    'attr_joint':    ['--model_type', 'moe_diffmic', '--attr_mode', 'joint'],
    'attr_pre_frz':  ['--model_type', 'moe_diffmic', '--attr_mode', 'pretrain_freeze', '--attr_pretrain_epochs', '1'],
    'attr_pre_ft':   ['--model_type', 'moe_diffmic', '--attr_mode', 'pretrain_ft', '--attr_pretrain_epochs', '1'],
    'dual':          ['--model_type', 'moe_diffmic', '--router_type', 'dual', '--dual_variant', 'dual', '--attr_mode', 'joint'],
    'dual_morph':    ['--model_type', 'moe_diffmic', '--router_type', 'dual', '--dual_variant', 'morph'],
    'dual_strength': ['--model_type', 'moe_diffmic', '--router_type', 'dual', '--dual_variant', 'strength'],
    'dual_noint':    ['--model_type', 'moe_diffmic', '--router_type', 'dual', '--dual_variant', 'dual_noint'],
    'dual_oracle':   ['--model_type', 'moe_diffmic', '--router_type', 'dual', '--dual_variant', 'oracle_sir'],
    'fus_uniform':   ['--model_type', 'moe_diffmic', '--fusion', 'uniform'],
    'fus_sum':       ['--model_type', 'moe_diffmic', '--fusion', 'sum_single'],
    'fus_topk':      ['--model_type', 'moe_diffmic', '--fusion', 'topk_mask', '--router_type', 'prior'],
    'fus_topk_dual': ['--model_type', 'moe_diffmic', '--fusion', 'topk_mask', '--router_type', 'dual', '--shared_expert'],
    'holdout':       ['--model_type', 'moe_diffmic', '--experts', 'TF', '--holdout', 'LFM'],
    'phys_aug':      ['--model_type', 'moe_diffmic', '--experts', 'TF', '--phys_aug', 'rot,remix,awgn', '--aug_multiplier', '2'],
    'cnn':           ['--model_type', 'cnn'],
    'mv_cnn':        ['--model_type', 'mv_cnn'],
    'dit_reg':       ['--model_type', 'dit_reg', '--disable_moe'],
    'mv_cnn_new':    ['--model_type', 'mv_cnn', '--shared_expert', '--router_type', 'dual'],
    'generic4':      ['--model_type', 'moe_diffmic', '--experts', 'GN', 'GN2', 'GN3', 'GN4'],
}
if __name__ == '__main__':
    flt = sys.argv[1] if len(sys.argv) > 1 else ''
    bad = []
    for name, extra in V.items():
        if flt and flt not in name:
            continue
        t0 = time.time()
        cmd = [PY, '-W', 'ignore', os.path.join(ROOT, 'exp_train.py'), '--exp_name', 'smk_' + name] + extra + BASE
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        ok = p.returncode == 0 and os.path.exists(os.path.join(ROOT, 'runs', 'smk_' + name, 'best_model.pth'))
        tail = ' '.join(l for l in p.stdout.splitlines() if 'Params' in l or 'Done' in l)[-160:] if ok else (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or ['?'])[-1][:200]
        print('%-14s %s %5.1fs %s' % (name, 'OK  ' if ok else 'FAIL', time.time() - t0, tail), flush=True)
        if not ok:
            bad.append(name)
    print('SMOKE', 'FAIL: ' + ','.join(bad) if bad else 'ALL OK')
