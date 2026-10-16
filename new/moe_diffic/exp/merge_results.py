"""在汇总机器上合并各台打包的结果（只补充缺失文件，不覆盖已存在的文件，除非 --overwrite）。
用法: python exp/merge_results.py exp_results_A_real.tar.gz exp_results_B_real.tar.gz exp_results_C_real.tar.gz"""
import argparse
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.abspath(os.path.join(HERE, '..', '..'))        # .../new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tars', nargs='+')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    n = skipped = 0
    for t in a.tars:
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                dest = os.path.join(NEW, m.name)
                if os.path.exists(dest) and not a.overwrite:
                    skipped += 1; continue
                tf.extract(m, NEW); n += 1
    print('合并完成：新增 %d 个文件，跳过已存在 %d 个' % (n, skipped))


if __name__ == '__main__':
    main()
