"""实验 10：生成残差扩散所需的基础恢复器与样本外（out-of-fold）基础估计。

流程（均调用 exp_train.py，CNN 直接监督）：
  1) 对 fold=0..F-1：排除 clean_id%F==fold 的训练样本训练一个 CNN，并对被排除的样本预测 -> runs/<name>_f<k>/oof_fold.npz
  2) 合并为 runs/<name>_oof.npz（clean_id, pred），按 clean_id 查表
  3) 在全部训练样本上训练最终基础恢复器 -> runs/<name>/best_model.pth（推理时 s_base=f(y) 对任何输入都是样本外的）
用法: python make_oof_base.py --name base_zs2 --types ST MT NB WB PL [--epochs 60 --folds 5 --amp bf16 --data_root ../data/v2] [其余参数透传给 exp_train.py]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print('+', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--types', nargs='+', default=None)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--amp', default='bf16')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data_root', default='../data/v2')
    ap.add_argument('--train_per_cell', type=int, default=None)
    a, extra = ap.parse_known_args()         # 其余参数原样传给 exp_train.py（如 --val_per_cell 1）
    py = sys.executable
    base = [py, '-W', 'ignore', os.path.join(HERE, 'exp_train.py'), '--model_type', 'cnn', '--lr', '3e-4', '--batch_size', '32', '--num_workers', '0',
            '--eval_freq', '5', '--val_every', '5', '--warmup_epochs', '3', '--epochs', str(a.epochs), '--seed', str(a.seed), '--amp', a.amp,
            '--new_data_root', a.data_root, '--abort_chance_frac', '9', '--skip_final_eval'] + extra
    if a.types:
        base += ['--new_types'] + a.types
    if a.train_per_cell:
        base += ['--train_per_cell', str(a.train_per_cell)]
    ids, preds = [], []
    for k in range(a.folds):
        name = '%s_f%d' % (a.name, k)
        if not os.path.exists(os.path.join(HERE, 'runs', name, 'oof_fold.npz')):
            run(base + ['--exp_name', name, '--oof_fold', str(k), '--oof_folds', str(a.folds)])
        z = np.load(os.path.join(HERE, 'runs', name, 'oof_fold.npz'))
        ids.append(z['clean_id']); preds.append(z['pred'])
    np.savez(os.path.join(HERE, 'runs', a.name + '_oof.npz'), clean_id=np.concatenate(ids), pred=np.concatenate(preds))
    print('合并样本外预测：%d 条 -> runs/%s_oof.npz' % (sum(len(i) for i in ids), a.name))
    if not os.path.exists(os.path.join(HERE, 'runs', a.name, 'best_model.pth')):
        run(base + ['--exp_name', a.name])
    for k in range(a.folds):                        # 折模型只为生成样本外预测，删除权重节省空间
        p = os.path.join(HERE, 'runs', '%s_f%d' % (a.name, k), 'best_model.pth')
        if os.path.exists(p):
            os.remove(p)


if __name__ == '__main__':
    main()
