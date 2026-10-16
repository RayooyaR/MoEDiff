#!/bin/bash
# 用法: exp/launch.sh <A|B|C> [real|smoke] [deadline_h] [额外 runner 参数...]
# 做的事：检查 -> 启动每张卡的 MPS 守护进程 -> 后台启动 runner（nohup+setsid，SSH 断开不影响）。
M=${1:?用法: launch.sh <A|B|C> [real|smoke] [deadline_h]}; PROFILE=${2:-real}; DEADLINE=${3:-11}; shift $(( $# < 3 ? $# : 3 ))
cd "$(dirname "$0")/.."
export PY=${PY:-/root/autodl-tmp/envs/moediff_bw/bin/python}
[ -x "$PY" ] || { echo "找不到 python: $PY（请设置环境变量 PY 指向可在 Blackwell/本机 GPU 上运行的 python）"; exit 1; }
case $M in A) NG=1;; B) NG=3;; C) NG=2;; esac
DET=$(nvidia-smi -L | wc -l); [ "$DET" -ne "$NG" ] && echo "WARN: 检测到 $DET 张 GPU，计划 $NG 张；将使用 $DET 张" && NG=$DET
echo "== preflight =="; $PY exp/preflight.py --machine $M || { echo "preflight 失败，已中止（修复后重试，或确认可忽略后设置 SKIP_PREFLIGHT=1）"; [ -z "$SKIP_PREFLIGHT" ] && exit 1; }
SLOTS=3; NOMPS=""
for g in $(seq 0 $((NG-1))); do
  mkdir -p /tmp/mps_pipe_g$g /tmp/mps_log_g$g
  if ! pgrep -f "mps_pipe_g$g" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=$g CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_g$g CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_g$g nvidia-cuda-mps-control -d 2>/dev/null
  fi
done
sleep 1
for g in $(seq 0 $((NG-1))); do
  if ! (echo get_server_list | CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_g$g timeout 5 nvidia-cuda-mps-control >/dev/null 2>&1); then SLOTS=1; NOMPS="--no_mps"; echo "WARN: GPU$g 的 MPS 不可用，退化为每卡 1 个任务"; fi
done
UNIT=""; [ -f exp_state/calib_$M.json ] && UNIT="--unit_seconds $($PY -c "import json;print(json.load(open('exp_state/calib_$M.json'))['unit_seconds'])")" && echo "使用校准结果 $UNIT"
export EXP_PROFILE=$PROFILE
mkdir -p logs exp_state
setsid nohup $PY -W ignore exp/runner.py --machine $M --profile $PROFILE --gpus $NG --slots $SLOTS --deadline_h $DEADLINE $NOMPS $UNIT "$@" > logs/runner_$M.out 2>&1 &
echo "runner 已在后台启动 (PID $!)，机器 $M，profile $PROFILE，$NG 卡 × $SLOTS 并发，deadline ${DEADLINE}h"
echo "监控:  cat exp_state/$M/progress.txt ;  tail -f logs/runner_$M.out ;  ls exp_state/$M/jobs | wc -l"
echo "结束后: $PY exp/pack_results.py --machine $M   (打包小文件结果，拷到汇总机器)"
