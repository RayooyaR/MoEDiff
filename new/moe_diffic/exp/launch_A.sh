#!/bin/bash
# 服务器 A：1 卡机（现在这台）负责：逐类留出类型 LFM/RFM、P0 审计、E13 采样消融、E12 物理一致性(可选)
# 用法: exp/launch_A.sh [real|smoke] [deadline_h] [额外 runner 参数，如 --retry_failed]     默认 real、11 小时
# 建议顺序: (1) python exp/preflight.py --machine A  (2) python exp/calibrate.py --machine A  (3) 本脚本
exec "$(dirname "$0")/launch.sh" A "${1:-real}" "${2:-11}" "${@:3}"
