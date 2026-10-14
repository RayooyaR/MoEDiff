"""异速率同信道数字干扰场景：  python make_scenarios_b.py [输出目录，默认 ../data/v3b_ccir]"""
import os
import sys

import config as C
import make_dataset as MD
import scenario_gen

scenario_gen.register()
C.TYPES = ['CCIR']
C.SPLITS = {
    'train':    (0,        200, 'A'),
    'val':      (100000,    20, 'A'),
    'test':     (200000,    80, 'A'),
    'test_ood': (300000,     1, 'A'),
}
if len(sys.argv) < 2:
    sys.argv.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'v3b_ccir')))
MD.main()
