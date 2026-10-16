#!/bin/bash
# 服务器 C：2 卡机负责：逐类留出类型 MT/PL/WB、E14 增强、E15 基线行
# 用法: exp/launch_C.sh [real|smoke] [deadline_h] [额外 runner 参数，如 --retry_failed]     默认 real、11 小时
# 建议顺序: (1) python exp/preflight.py --machine C  (2) python exp/calibrate.py --machine C  (3) 本脚本
exec "$(dirname "$0")/launch.sh" C "${1:-real}" "${2:-11}" "${@:3}"
