#!/bin/bash
# 专家子集消融：每当同时在训的 train.py 少于 3 个，就启动下一个任务（不与队列10争抢显存）
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
while IFS= read -r job; do
  while [ "$(ps -eo args | grep -c '^/root/miniconda3/envs/moediff/bin/python -W ignore train.py')" -ge 3 ]; do sleep 60; done
  nohup bash -c "$job" > /dev/null 2>&1 &
  sleep 120
done < logs/queue12_jobs.txt
wait
