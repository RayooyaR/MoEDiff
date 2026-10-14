#!/bin/bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
while pgrep -f "queue2_job[s]" > /dev/null; do sleep 60; done
cat logs/queue3_jobs.txt | xargs -P 3 -I{} bash -c '{}'
/root/miniconda3/envs/moediff/bin/python -W ignore lmmse_fit.py --exp_name v4_lmmse --data_root ../data/v4_digital > logs/v4_lmmse.log 2>&1
/root/miniconda3/envs/moediff/bin/python -W ignore evaluate_new.py --runs v4_lmmse --splits test --data_root ../data/v4_digital > logs/ev_v4_lmmse.log 2>&1
