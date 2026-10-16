"""启动前检查（每台服务器）：环境 / GPU / 数据完整性（克隆是否完整）/ 磁盘 / 代码一致性哈希 / 依赖的既有检查点。
用法: python exp/preflight.py --machine B [--write_manifest]   （--write_manifest 只在“源机器”上执行一次，生成 exp/data_manifest.json 供另外两台核对）
退出码: 0=可以启动（可能有 WARN）；1=有 FAIL。"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.abspath(os.path.join(ROOT, '..', 'data', 'v2'))
sys.path.insert(0, HERE)
import registry as REG  # noqa: E402

FAIL = WARN = 0


def say(level, msg):
    global FAIL, WARN
    FAIL += level == 'FAIL'
    WARN += level == 'WARN'
    print('[%s] %s' % (level, msg))


def data_manifest():
    items = []
    for f in sorted(glob.glob(os.path.join(DATA, '**', '*'), recursive=True)):
        if os.path.isfile(f):
            items.append((os.path.relpath(f, DATA), os.path.getsize(f)))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest(), len(items), sum(s for _, s in items)


def code_hash():
    h = hashlib.sha1()
    roots = [ROOT, os.path.abspath(os.path.join(ROOT, '..', 'loader')), os.path.abspath(os.path.join(ROOT, '..', 'analysis'))]
    n = 0
    for r in roots:
        for f in sorted(glob.glob(os.path.join(r, '**', '*.py'), recursive=True)):
            if '/runs/' in f or '/logs/' in f or '__pycache__' in f or '/exp_state/' in f or '/plans/' in f:
                continue
            h.update(os.path.relpath(f, os.path.dirname(ROOT)).encode()); h.update(open(f, 'rb').read()); n += 1
    return h.hexdigest()[:12], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', required=True, choices=['A', 'B', 'C'])
    ap.add_argument('--write_manifest', action='store_true')
    a = ap.parse_args()
    import torch
    print('python %s | torch %s (cuda %s)' % (sys.version.split()[0], torch.__version__, torch.version.cuda))
    exp_g = REG.MACHINE_GPUS[a.machine]
    n = torch.cuda.device_count()
    if n == 0:
        say('FAIL', '没有可见 GPU')
    elif n != exp_g:
        say('WARN', '机器 %s 计划使用 %d 张 GPU，实际检测到 %d 张（runner 会按 --gpus 使用；负载与时间需重估）' % (a.machine, exp_g, n))
    else:
        say('OK', '检测到 %d 张 GPU' % n)
    for i in range(n):
        try:
            x = torch.randn(256, 256, device='cuda:%d' % i); (x @ x).sum().item()
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            say('OK' if free > 40e9 else 'WARN', 'GPU%d %s sm_%d%d 显存 %.0f/%.0f GB 空闲%s' % (i, props.name, props.major, props.minor, free / 1e9, total / 1e9, '' if free > 40e9 else '（<40GB，3 并发可能不足）'))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                (x @ x).sum().item()
        except Exception as e:
            say('FAIL', 'GPU%d 无法执行计算: %s' % (i, str(e).splitlines()[0]))
    say('OK' if shutil.which('nvidia-cuda-mps-control') else 'WARN', 'MPS %s' % ('可用' if shutil.which('nvidia-cuda-mps-control') else '不可用（将退化为每卡 1 个任务，容量约下降 20%）'))
    free = shutil.disk_usage(ROOT).free / 1e9
    say('OK' if free > 15 else ('WARN' if free > 5 else 'FAIL'), '磁盘剩余 %.1f GB（每个 run 约 100MB，评估后删除权重）' % free)
    if not os.path.isdir(DATA):
        say('FAIL', '缺少数据 %s' % DATA)
    else:
        h, nf, sz = data_manifest()
        mf = os.path.join(HERE, 'data_manifest.json')
        if a.write_manifest:
            json.dump(dict(sha1=h, files=nf, bytes=sz), open(mf, 'w')); print('写入 %s: %s files %d bytes %d' % (mf, h[:12], nf, sz))
        elif os.path.exists(mf):
            m = json.load(open(mf))
            say('OK' if m['sha1'] == h else 'FAIL', '数据 v2 完整性：%s（文件 %d/%d，字节 %d/%d）' % ('与源机器一致' if m['sha1'] == h else '与源机器不一致，克隆可能不完整', nf, m['files'], sz, m['bytes']))
        else:
            say('WARN', '没有 data_manifest.json，未做数据完整性核对')
    ch, nc = code_hash()
    print('代码哈希 %s（%d 个 py 文件）——三台服务器必须相同' % (ch, nc))
    for r in ('zs2_ex_TF_s44', 'zs2_dit'):
        say('OK' if os.path.exists(os.path.join(ROOT, 'runs', r, 'best_model.pth')) else 'WARN', '既有检查点 %s（E12/E13 需要）' % r)
    py = os.environ.get('PY') or sys.executable
    say('OK', '使用 python: %s' % py)
    print('PREFLIGHT', 'FAIL' if FAIL else ('OK（%d 个警告）' % WARN if WARN else 'OK'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
