"""更难场景 v5：标准 8 类干扰 × 3 种调制，干扰上叠加随机信号影响（7 种效应各 50% 概率独立出现）。
用法：  python make_scenarios_d.py [输出目录，默认 ../data/v5_effects]
划分与 v2 相同（train 40 / val 5 / test 40 / test_ood 20 每格），test_ood 使用基础干扰的留出参数区间。
"""
import os
import sys

import make_dataset as MD
import scenario_gen

scenario_gen.register_effects()
if len(sys.argv) < 2:
    sys.argv.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'v5_effects')))
MD.main()
