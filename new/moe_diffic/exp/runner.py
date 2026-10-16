"""单台服务器的任务执行器（无人值守）。读取 exp/plans/<profile>_<机器>.json，按依赖与优先级把任务派给 (GPU × 每卡并发槽)，
不需要任何跨机通信。关键行为：
  * 依赖 DAG：任务的 deps 全部 done 才启动；上游失败 -> 本任务标记 dep_failed 并跳过（不拖垮无关任务）。kind=select 的任务(mode=any_terminal)
    在其 deps 全部到达终态后运行，读取 runs/ 里的验证结果，写 exp_state/<机器>/winners.json；命令里的 {CAND:键} 启动前替换（缺失用 DEFAULTS）。
  * 优先级：tier 0(环境/锚点) > 1(必做) > 2(填充，deadline 的 80% 之后不再启动) > 3(收尾)；同级按代价从大到小（避免拖尾）。
  * 不会启动“预计来不及”的任务：预计耗时 = cost × unit_seconds，剩余时间不够则标记 skipped_time。
  * 超时（默认预计耗时的 6 倍，至少 20 分钟）强杀；失败自动重试一次；退出码 4(早期中止：训练过半仍是机会水平) 不重试；到 deadline 强杀剩余任务。
  * 断点续跑：重新运行同一命令会跳过已 done/skipped 的任务；running/interrupted 的重新排队。
用法: python exp/runner.py --machine B --profile real --gpus 3 --slots 3 --deadline_h 11
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import registry as REG  # noqa: E402

TERMINAL = {'done', 'failed', 'aborted', 'skipped', 'dep_failed', 'timeout', 'skipped_time'}
GOOD = {'done', 'skipped'}


def now():
    return time.time()


class Runner:
    def __init__(self, a):
        self.a = a
        self.state_dir = os.path.join(ROOT, 'exp_state', a.machine)
        os.makedirs(os.path.join(self.state_dir, 'jobs'), exist_ok=True)
        os.makedirs(os.path.join(ROOT, 'logs', 'jobs'), exist_ok=True)
        plan_f = os.path.join(HERE, 'plans', '%s_%s.json' % (a.profile, a.machine))
        if not os.path.exists(plan_f):
            subprocess.check_call([sys.executable, os.path.join(HERE, 'make_plan.py'), '--profile', a.profile])
        self.plan = json.load(open(plan_f))
        self.jobs = {j['name']: j for j in self.plan['jobs']}
        self.status = {}
        for n in self.jobs:
            f = os.path.join(self.state_dir, 'jobs', n + '.json')
            st = json.load(open(f)) if os.path.exists(f) else {}
            if st.get('status') in ('running', 'interrupted') or (a.retry_failed and st.get('status') in ('failed', 'aborted', 'dep_failed', 'timeout', 'skipped_time')):
                st = {}
            self.status[n] = st
        self.running = {}          # name -> dict(proc, gpu, t0, timeout)
        self.t_start = now()
        self.deadline = self.t_start + a.deadline_h * 3600
        self.py = os.environ.get('PY') or a.python
        self.stop = False
        signal.signal(signal.SIGINT, self._sig)
        signal.signal(signal.SIGTERM, self._sig)

    # ------------------------------------------------------------ 状态
    def st(self, n):
        return self.status[n].get('status', 'pending')

    def set(self, n, **kw):
        self.status[n].update(kw)
        self.status[n]['name'] = n
        json.dump(self.status[n], open(os.path.join(self.state_dir, 'jobs', n + '.json'), 'w'))

    def _sig(self, *_):
        self.stop = True

    # ------------------------------------------------------------ winners
    def winners(self):
        f = os.path.join(self.state_dir, 'winners.json')
        w = dict(REG.DEFAULTS)
        if os.path.exists(f):
            try:
                w.update(json.load(open(f)))
            except Exception:
                pass
        return w

    def resolve(self, cmd):
        w = self.winners()
        return re.sub(r'\{CAND:(\w+)\}', lambda m: str(w.get(m.group(1), REG.DEFAULTS.get(m.group(1), ''))), cmd)

    def skip_cond(self, j):
        c = j.get('skip_if')
        if not c:
            return False
        return str(self.winners().get(c, REG.DEFAULTS.get(c, False))).lower() in ('true', '1')

    # ------------------------------------------------------------ 就绪判定
    def classify(self, n):
        """返回 'ready' | 'wait' | 'dep_failed' | 'skip'"""
        j = self.jobs[n]
        ds = [self.st(d) if d in self.jobs else 'done' for d in j['deps']]       # 不在本机计划里的依赖视为已满足（不应出现）
        if j['deps_mode'] == 'any_terminal':
            return 'ready' if all(s in TERMINAL for s in ds) else 'wait'
        if any(s in TERMINAL and s not in GOOD for s in ds):
            return 'dep_failed'
        if all(s in GOOD for s in ds):
            return 'skip' if self.skip_cond(j) else 'ready'
        return 'wait'

    def est_s(self, j):
        return max(30.0, j['cost'] * self.a.unit_seconds)

    # ------------------------------------------------------------ 启动
    def env_for(self, gpu):
        e = dict(os.environ)
        e['PY'] = self.py + ' -W ignore'
        e['PYTHONUNBUFFERED'] = '1'
        e['CUDA_VISIBLE_DEVICES'] = str(gpu)
        e['OMP_NUM_THREADS'] = '2'
        e['MKL_NUM_THREADS'] = '2'
        if not self.a.no_mps:
            e['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/mps_pipe_g%d' % gpu
            e['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/mps_log_g%d' % gpu
        return e

    def launch(self, n, gpu):
        j = self.jobs[n]
        cmd = self.resolve(j['cmd'])
        log = open(os.path.join(ROOT, 'logs', 'jobs', n + '.log'), 'a')
        log.write('\n===== %s attempt %d gpu %d @ %s =====\n%s\n' % (n, self.status[n].get('attempts', 0) + 1, gpu, time.strftime('%F %T'), cmd))
        log.flush()
        p = subprocess.Popen(['bash', '-c', 'set -o pipefail; ' + cmd], cwd=ROOT, env=self.env_for(gpu), stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        to = max(1200.0, self.est_s(j) * self.a.timeout_scale)
        self.running[n] = dict(proc=p, gpu=gpu, t0=now(), timeout=to, log=log)
        self.set(n, status='running', gpu=gpu, start=now(), attempts=self.status[n].get('attempts', 0) + 1)

    def run_inline(self, n):
        """select / 内建任务：在调度线程里直接跑（CPU 轻量）。"""
        j = self.jobs[n]
        cmd = self.resolve(j['cmd'])
        self.set(n, status='running', start=now(), attempts=1)
        if cmd.startswith('BUILTIN:cleanup'):
            for m in self.plan['jobs']:
                if m['meta'].get('kind') == 'train' or m['cmd'].startswith(REG.PYV + ' exp_train.py') or 'exp_train.py' in m['cmd']:
                    mm = re.search(r'--exp_name (\S+)', m['cmd']) or re.search(r'--runs (\S+)', m['cmd'])
                    if mm:
                        p = os.path.join(ROOT, 'runs', mm.group(1), 'best_model.pth')
                        if os.path.exists(p):
                            os.remove(p)
            self.set(n, status='done', end=now(), rc=0)
            return
        with open(os.path.join(ROOT, 'logs', 'jobs', n + '.log'), 'a') as log:
            env = self.env_for(0); env['CUDA_VISIBLE_DEVICES'] = ''
            env['EXP_STATE_DIR'] = self.state_dir
            rc = subprocess.call(['bash', '-c', cmd], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.set(n, status='done' if rc == 0 else 'failed', end=now(), rc=rc)

    # ------------------------------------------------------------ 主循环
    def poll(self):
        for n in list(self.running):
            r = self.running[n]
            rc = r['proc'].poll()
            el = now() - r['t0']
            if rc is None and el > r['timeout']:
                os.killpg(os.getpgid(r['proc'].pid), signal.SIGKILL)
                r['log'].close(); del self.running[n]
                self.set(n, status='timeout', end=now(), seconds=round(el, 1))
                continue
            if rc is None:
                continue
            r['log'].close(); del self.running[n]
            if rc == 0:
                self.set(n, status='done', end=now(), rc=0, seconds=round(el, 1))
            elif rc == 4:
                self.set(n, status='aborted', end=now(), rc=rc, seconds=round(el, 1))
            elif self.status[n].get('attempts', 1) < 2 and now() + self.est_s(self.jobs[n]) < self.deadline and not self.stop:
                self.set(n, status='pending', rc=rc, seconds=round(el, 1))        # 重试一次
            else:
                self.set(n, status='failed', end=now(), rc=rc, seconds=round(el, 1))

    def write_progress(self):
        c = {}
        for n in self.jobs:
            c[self.st(n)] = c.get(self.st(n), 0) + 1
        el = now() - self.t_start
        lines = ['机器 %s  profile %s  已运行 %.1f h / deadline %.1f h' % (self.a.machine, self.a.profile, el / 3600, self.a.deadline_h),
                 '状态计数: ' + ', '.join('%s=%d' % kv for kv in sorted(c.items())),
                 '运行中: ' + ', '.join('%s@gpu%d(%.0fm)' % (n, r['gpu'], (now() - r['t0']) / 60) for n, r in self.running.items())]
        open(os.path.join(self.state_dir, 'progress.txt'), 'w').write('\n'.join(lines) + '\n')

    def loop(self):
        n_slots = self.a.gpus * self.a.slots
        last_prog = 0
        while True:
            self.poll()
            elapsed = now() - self.t_start
            if self.stop or now() > self.deadline:
                for n, r in list(self.running.items()):
                    try:
                        os.killpg(os.getpgid(r['proc'].pid), signal.SIGKILL)
                    except Exception:
                        pass
                    self.set(n, status='interrupted' if self.stop else 'timeout', end=now())
                break
            # 终态传播与内联任务
            progressed = False
            for n in self.jobs:
                if self.st(n) != 'pending':
                    continue
                c = self.classify(n)
                j = self.jobs[n]
                if c == 'dep_failed':
                    self.set(n, status='dep_failed'); progressed = True
                elif c == 'skip':
                    self.set(n, status='skipped', note='skip_if %s' % j['skip_if']); progressed = True
                elif c == 'ready' and (j['meta'].get('kind') == 'select' or j['cmd'].startswith('BUILTIN:')):
                    if j['tier'] >= 3 and any(self.st(m) == 'pending' or m in self.running for m in self.jobs if self.jobs[m]['tier'] < 3):
                        continue
                    self.run_inline(n); progressed = True
            # 填槽
            if len(self.running) < n_slots:
                ready = [n for n in self.jobs if self.st(n) == 'pending' and self.classify(n) == 'ready' and self.jobs[n]['meta'].get('kind') != 'select'
                         and not self.jobs[n]['cmd'].startswith('BUILTIN:')]
                ready.sort(key=lambda n: (self.jobs[n]['tier'], -self.jobs[n]['cost']))
                for n in ready:
                    if len(self.running) >= n_slots:
                        break
                    j = self.jobs[n]
                    left = self.deadline - now()
                    if j['tier'] >= 2 and elapsed > 0.8 * self.a.deadline_h * 3600:
                        self.set(n, status='skipped_time', note='tier2 after 80%% of deadline'); continue
                    if left < self.est_s(j) * 0.8 and j['tier'] > 0:
                        self.set(n, status='skipped_time', note='remaining %.0fs < est %.0fs' % (left, self.est_s(j))); continue
                    load = [sum(1 for r in self.running.values() if r['gpu'] == g) for g in range(self.a.gpus)]
                    self.launch(n, int(min(range(self.a.gpus), key=lambda g: load[g])))
            if now() - last_prog > 20:
                self.write_progress(); last_prog = now()
            pending = [n for n in self.jobs if self.st(n) in ('pending', 'running')]
            if not pending and not self.running:
                break
            if not self.running and not progressed and not any(self.classify(n) in ('ready', 'dep_failed', 'skip') for n in self.jobs if self.st(n) == 'pending'):
                break          # 剩余任务全部被阻塞（上游被跳过/失败）
            time.sleep(self.a.poll if self.running else 0.5)
        self.finish()

    def finish(self):
        self.write_progress()
        c = {}
        for n in self.jobs:
            c.setdefault(self.st(n), []).append(n)
        json.dump({'machine': self.a.machine, 'profile': self.a.profile, 'hours': round((now() - self.t_start) / 3600, 2), 'counts': {k: len(v) for k, v in c.items()},
                   'failed': c.get('failed', []) + c.get('timeout', []), 'aborted': c.get('aborted', []), 'dep_failed': c.get('dep_failed', []), 'skipped_time': c.get('skipped_time', [])},
                  open(os.path.join(self.state_dir, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
        print(open(os.path.join(self.state_dir, 'summary.json')).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', required=True, choices=['A', 'B', 'C'])
    ap.add_argument('--profile', default='real', choices=['real', 'smoke', 'fill', 'm10', 'd2', 'aug', 'aug2', 'aug3', 'aug4', 'loc', 'fast', 'fast2', 'v4m', 'v4a', 'pre'])
    ap.add_argument('--gpus', type=int, default=None)
    ap.add_argument('--slots', type=int, default=3, help='每张卡同时运行的任务数（开 MPS 时 3；无 MPS 时 1）')
    ap.add_argument('--deadline_h', type=float, default=11.0)
    ap.add_argument('--unit_seconds', type=float, default=None, help='1 个代价单位的预计秒数（由 calibrate 给出；默认取 profile 值）')
    ap.add_argument('--timeout_scale', type=float, default=6.0)
    ap.add_argument('--poll', type=float, default=2.0)
    ap.add_argument('--no_mps', action='store_true')
    ap.add_argument('--retry_failed', action='store_true')
    ap.add_argument('--python', default='/root/autodl-tmp/envs/moediff_bw/bin/python')
    a = ap.parse_args()
    a.gpus = a.gpus or REG.MACHINE_GPUS[a.machine]
    a.unit_seconds = a.unit_seconds or REG.profile(a.profile)['unit_seconds']
    Runner(a).loop()


if __name__ == '__main__':
    main()
