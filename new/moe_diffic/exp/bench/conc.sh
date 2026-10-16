#!/bin/bash
# 用法: conc.sh <run> <mode> <N> [steps]   同时起 N 个进程，对齐开始时刻，打印每个进程的 ms/步 与聚合吞吐(步/秒)
PY="${PY:-python} -W ignore"
run=$1; mode=$2; N=$3; steps=${4:-200}
START=$(python3 -c "import time;print(time.time()+25)")
for i in $(seq 1 $N); do START_AT=$START $PY $(dirname $0)/step.py $run $mode $steps 2>&1 | tail -1 > /tmp/conc_$i.txt & done; wait
cat /tmp/conc_*.txt | awk -v N=$N '{v=$3; s+=1000/v; printf "%s ", v} END {printf "\n  => N=%d 聚合吞吐 %.1f 步/秒\n", N, s}'
rm -f /tmp/conc_*.txt
