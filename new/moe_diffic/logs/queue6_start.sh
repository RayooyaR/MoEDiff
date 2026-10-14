#!/bin/bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
while pgrep -f "queue4_job[s]" > /dev/null || pgrep -f "queue5_job[s]" > /dev/null || pgrep -f "queue5_star[t]" > /dev/null; do sleep 60; done
cat logs/queue6_jobs.txt | xargs -P 3 -I{} bash -c '{}'
