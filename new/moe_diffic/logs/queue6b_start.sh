#!/bin/bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
while pgrep -f "queue[456]_job[s]" > /dev/null || pgrep -f "queue[56]_star[t]" > /dev/null; do sleep 60; done
cat logs/queue6b_jobs.txt | xargs -P 3 -I{} bash -c '{}'
