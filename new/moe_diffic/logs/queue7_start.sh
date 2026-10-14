#!/bin/bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
while pgrep -f "queue[4568]_job[s]" > /dev/null || pgrep -f "queue[56]_star[t]" > /dev/null || pgrep -f "queue6b_star[t]" > /dev/null; do sleep 60; done
cat logs/queue7_jobs.txt | xargs -P 3 -I{} bash -c '{}'
