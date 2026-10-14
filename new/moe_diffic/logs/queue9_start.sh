#!/bin/bash
cd /root/autodl-tmp/MoEDiff/new/moe_diffic
while pgrep -f "queue8_job[s]" > /dev/null; do sleep 60; done
cat logs/queue9_jobs.txt | xargs -P 3 -I{} bash -c '{}'
