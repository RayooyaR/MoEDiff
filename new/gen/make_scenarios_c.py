"""同信道数字干扰家族数据集：  python make_scenarios_c.py [输出目录，默认 ../data/v4_digital]
子类（作为 interference_type）：CCIR、FSK、OFDM、BURST、ACI。数据格式与 v2 相同。
"""
import os
import sys

import config as C
import make_dataset as MD
import scenario_gen

scenario_gen.register_digital()
C.TYPES = ['CCIR', 'FSK', 'OFDM', 'BURST', 'ACI']
C.SPLITS = {
    'train':    (0,        60, 'A'),
    'val':      (100000,    8, 'A'),
    'test':     (200000,   30, 'A'),
    'test_ood': (300000,    1, 'A'),   # 占位
}
if len(sys.argv) < 2:
    sys.argv.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'v4_digital')))
MD.main()
