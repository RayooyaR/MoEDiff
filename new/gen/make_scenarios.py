"""生成候选难场景的试点数据：  python make_scenarios.py [输出目录，默认 ../data/v3_hard]

数据格式与 v2 完全相同（可直接用 MoEDiffDataset 读取），干扰类型为 CCI、MIX。
"""
import os
import sys

import config as C
import make_dataset as MD
import scenario_gen

scenario_gen.register()
C.TYPES = ['CCI', 'MIX']
C.SPLITS = {
    'train':    (0,        100, 'A'),
    'val':      (100000,    10, 'A'),
    'test':     (200000,    40, 'A'),
    'test_ood': (300000,     1, 'A'),   # 占位（加载器要求该目录存在）；本场景不使用留出参数
}
if len(sys.argv) < 2:
    sys.argv.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'v3_hard')))
MD.main()
