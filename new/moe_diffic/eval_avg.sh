#!/bin/bash
# 用法: eval_avg.sh <run名> <kind: dit|moe|diffcom|icdm> [额外 evaluate_new 参数...]
# 随机采样取平均（K=8），使用测试集每隔一条的一半样本；diffcom/icdm 读取 runs/<run>/chosen.json 里验证集选定的 ζ / γ
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
PY="/root/miniconda3/envs/moediff/bin/python -W ignore"
n=$1; kind=$2; shift 2
extra=""
if [ "$kind" = diffcom ]; then extra="--dps_step $($PY -c "import json;print(json.load(open('runs/$n/chosen.json'))['chosen'])")"; fi
if [ "$kind" = icdm ]; then extra="--dual_gamma $($PY -c "import json;print(json.load(open('runs/$n/chosen.json'))['chosen'])")"; fi
$PY evaluate_new.py --runs $n --splits test --stride 2 --avg_k 8 --tag _avg8 $extra "$@" > logs/avg8_$n.log 2>&1
