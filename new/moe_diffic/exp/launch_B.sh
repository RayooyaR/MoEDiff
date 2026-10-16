#!/bin/bash
# 服务器 B：3 卡机负责：E1–E5、E7–E11 依赖链、依赖胜出者的 E15/E16、胜出者的 LOTO（含 2 次自动选择）
# 用法: exp/launch_B.sh [real|smoke] [deadline_h] [额外 runner 参数，如 --retry_failed]     默认 real、11 小时
# 建议顺序: (1) python exp/preflight.py --machine B  (2) python exp/calibrate.py --machine B  (3) 本脚本
exec "$(dirname "$0")/launch.sh" B "${1:-real}" "${2:-11}" "${@:3}"
