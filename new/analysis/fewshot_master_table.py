"""少样本/监督/留出 全场景汇总表：所有方法 × 所有场景 × 全部指标（BER/ΔBER/负增益/EVM/MSE/MAE/带内SINR）。

用法（在 new/moe_diffic 下）： python ../analysis/fewshot_master_table.py
输出： ../reports/fewshot_all_baselines.md 、 ../reports/fewshot_all_baselines.csv
数值 = 各随机种子分别计算后取均值（BER 另给种子间标准差）；扩散类 = 平均 8 次采样；测试集每隔一条（与 *_avg8 同一批样本）。
"""
import csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from aggregate_results import load, type_mask, summarize
RUNS = os.path.join(HERE, '..', 'moe_diffic', 'runs'); OUT = os.path.join(HERE, '..', 'reports')
UN = ['LFM', 'NFM', 'RFM']

def get(run, split, suf):
    d = load(run, split, suf, RUNS)
    if d is None and suf == '_avg8':      # 非扩散旧机 run 没有 avg8 文件：取完整测试集每隔一条
        d = load(run, split, '', RUNS)
        if d is not None and len(d['type']) > 0:
            n = len(d['type']); d = {k: (v[::2] if getattr(v, 'ndim', 0) >= 1 and len(v) == n else v) for k, v in d.items()}
    return d

def params(run):
    try: return json.load(open(os.path.join(RUNS, run, 'run_info.json')))['params'] / 1e6
    except Exception: return None

def agg(runs, split, suf, subset):
    """runs: 同一方法的各种子 run 名；subset: ('all',) | ('seen',) | ('unseen',) | ('type',[...]) | ('exclude',[...])"""
    vs = []
    for r in runs:
        d = get(r, split, suf)
        if d is None: continue
        if subset[0] == 'all': m = np.ones(len(d['type']), bool)
        elif subset[0] == 'seen': m = type_mask(d, exclude=UN)
        elif subset[0] == 'unseen': m = type_mask(d, types=UN)
        elif subset[0] == 'type': m = type_mask(d, types=subset[1])
        else: m = type_mask(d, exclude=subset[1])
        s = summarize(d, m); s['neg'] = s['neg_gain_ratio']; vs.append(s)
    if not vs: return None
    o = {k: float(np.mean([v[k] for v in vs])) for k in ('ber', 'ber_input', 'delta_ber', 'neg', 'evm', 'mse', 'sinr_inband')}
    o['mae'] = None
    maes = []
    for r in runs:
        d = get(r, split, suf)
        if d is None: continue
        if subset[0] == 'all': m = np.ones(len(d['type']), bool)
        elif subset[0] == 'seen': m = type_mask(d, exclude=UN)
        elif subset[0] == 'unseen': m = type_mask(d, types=UN)
        elif subset[0] == 'type': m = type_mask(d, types=subset[1])
        else: m = type_mask(d, exclude=subset[1])
        maes.append(float(d['mae_model'][m].mean()))
    o['mae'] = float(np.mean(maes)); o['n_seed'] = len(vs); o['n'] = vs[0]['n']
    o['ber_sd'] = float(np.std([v['ber'] for v in vs], ddof=1)) if len(vs) > 1 else None
    return o

def S(base, seeds, suffix=''): return [f'{base}_s{s}' for s in seeds]
HDR = ['方法', '种子数', '参数量(M)', 'BER↓', 'BER种子sd', 'ΔBER↑', '负增益比例↓', 'EVM↓', 'MSE↓', 'MAE↓', '带内SINR(dB)↑']
rows_csv = []
md = []

def fmt(x, f): return '—' if x is None else f % x

def table(title, scen, split, subset, methods, suf_default='_avg8'):
    """methods: [(显示名, [run...], suffix或None, 参数量run)]"""
    md.append(f'\n**{title}**（样本数 {{N}}）\n')
    md.append('| ' + ' | '.join(HDR) + ' |'); md.append('|' + '---|' * len(HDR))
    N = None; mf_row = None
    body = []
    for name, runs, suf, prun in methods:
        a = agg(runs, split, suf or suf_default, subset)
        if a is None: continue
        N = a['n']
        if mf_row is None:       # 输入(MF)：同一批样本的 *_mf 字段
            d = get(runs[0], split, suf or suf_default)
            if subset[0] == 'all': m = np.ones(len(d['type']), bool)
            elif subset[0] == 'seen': m = type_mask(d, exclude=UN)
            elif subset[0] == 'unseen': m = type_mask(d, types=UN)
            elif subset[0] == 'type': m = type_mask(d, types=subset[1])
            else: m = type_mask(d, exclude=subset[1])
            mf_row = ['不处理(匹配滤波)', '—', '—', '%.4f' % d['ber_mf'][m].mean(), '—', '—', '—', '%.3f' % d['evm_mf'][m].mean(), '%.3f' % d['mse_mf'][m].mean(), '%.3f' % d['mae_mf'][m].mean(), '%.2f' % d['sinr_inband_mf'][m].mean()]
        p = params(prun or runs[0])
        body.append([name, str(a['n_seed']), fmt(p, '%.2f'), '%.4f' % a['ber'], fmt(a['ber_sd'], '%.4f'), '%+.4f' % a['delta_ber'], '%.3f' % a['neg'], '%.3f' % a['evm'], '%.3f' % a['mse'], '%.3f' % a['mae'], '%.2f' % a['sinr_inband']])
        rows_csv.append([scen, split, str(subset)] + body[-1])
    if mf_row: body.insert(0, mf_row); rows_csv.append([scen, split, str(subset)] + mf_row)
    best = min((float(r[3]) for r in body[1:]), default=None)
    for r in body:
        if best is not None and r[0] != mf_row[0] and abs(float(r[3]) - best) < 1e-12: r = [r[0]] + r[1:3] + ['**' + r[3] + '**'] + r[4:]
        md.append('| ' + ' | '.join(r) + ' |')
    md[[i for i, x in enumerate(md) if '{N}' in x][-1]] = md[[i for i, x in enumerate(md) if '{N}' in x][-1]].replace('{N}', str(N))

# ---------- 方法定义（新管线：训练类型 ST/MT/NB/WB/PL；未见 = LFM/NFM/RFM）----------
def meth(P):
    m = [('CNN（1.59M，直接回归）', S('de_cnn_p%d' % P, (42, 43, 44)), None, None),
         ('DiT 扩散，无专家（真无条件）', S('de_dit_p%d' % P, (42, 43, 44)), None, None),
         ('DiT 扩散 + 时频专家', S('de_tf_p%d' % P, (42, 43, 44)), None, None)]
    if P in (10, 100): m.append(('**U-Net 扩散 + 时频专家（J0）**', S('j0_p%d' % P, (42, 43)), None, None))
    m += [('U-Net 扩散 + 共享+四专家（J1）', S('de_moe_p%d' % P, (42, 43)), None, None),
          ('CNN 打底 + 残差扩散（时频，λ=1）', S('de_res_p%d' % P, (42, 43)), None, None),
          ('CNN 打底 + 残差扩散 + 门控（J2）', S('de_res_p%d' % P, (42, 43)), '_avg8g', 'de_res_p%d_s42' % P),
          ('CNN 打底 + 新MoE + 残差扩散 + 门控（J3）', S('de_joint_p%d' % P, (42, 43)), '_avg8g', 'de_joint_p%d_s42' % P)]
    if P in (10, 100):
        m += [('静态多分支 + 残差扩散 + 门控（J4）', S('j4_p%d' % P, (42, 43)), '_avg8g', 'j4_p%d_s42' % P),
              ('新MoE + 直接回归 CNN，不用扩散（J5）', S('j5_p%d' % P, (42, 43)), None, None)]
    return m

SUBS = [('见过类型（ST/MT/NB/WB/PL）', ('seen',)), ('未见类型（LFM/NFM/RFM，零样本）', ('unseen',)), ('全部类型', ('all',))]
md.append('# 少样本 / 监督 / 留出：全场景 × 全部方法 × 全部指标\n')
md.append('> 由 `new/analysis/fewshot_master_table.py` 生成。数值=各种子分别算后取均值；扩散类=平均 8 次采样、25 步 DDIM；测试集每隔一条。粗体=该表 BER 最优。\n> 指标：BER 误码率、ΔBER=不处理 BER − 方法 BER、负增益比例=处理后反而更差的样本占比、EVM 误差向量幅度、MSE 均方误差、MAE 平均绝对误差、带内SINR=恢复后带内信号与残余干扰加噪声功率比（dB）。\n')
md.append('## 一、新管线：按训练数据量（10% / 30% / 100%，步数对齐）× 测试集\n')
md.append('训练只含 5 类干扰（ST/MT/NB/WB/PL）；LFM/NFM/RFM 三类整组不参与训练（零样本）。10%/30%/100% = 每个（类型×调制×信干比）格取 4 / 12 / 40 条。\n')
for P in (10, 30, 100):
    md.append(f'\n### 训练数据 {P}%\n')
    for sn, sub in SUBS:
        table(f'{P}% 数据 · 测试集A（同参数区间） · {sn}', f'新管线{P}%', 'test', sub, meth(P))
    table(f'{P}% 数据 · 测试集B（未见参数区间，OOD） · 全部类型', f'新管线{P}%', 'test_ood', ('all',), meth(P))
# ---------- E7 主干对照 (100%, zs2) ----------
md.append('\n## 二、去噪主干对照（E7，100% 数据，单种子 42）\n')
E7 = [('DiT（无专家）', ['e7_dit0_s42'], None, None), ('DiT + 时频', ['e1_tf_s42'], None, None), ('U-Net(48) 无专家', ['e7_u0_s42'], None, None),
      ('U-Net(48) + 时频', ['e7_uTF48_s42'], None, None), ('U-Net(64) + 时频', ['e7_uTF_s42'], None, None),
      ('DiffWave + 时频', ['e7_dwTF_s42'], None, None), ('DiffWave(192) + 时频', ['e7_dwTF192_s42'], None, None), ('CNN（1.59M）', ['de_cnn_p100_s42'], None, None)]
for sn, sub in SUBS: table(f'E7 · {sn}', 'E7主干', 'test', sub, E7)
# ---------- 旧机同一测试集的传统/对照基线（100% 数据） ----------
md.append('\n## 三、同一测试集上的传统方法与前沿对照（旧机，fp32，100% 数据，训练类型同上，种子 42）\n')
OLD = [('LMMSE（传统滑窗维纳）', ['zs2_lmmse'], None, None), ('CNN（旧机）', ['zs2_cnn'], None, None), ('加宽 CNN（对照）', ['zs2_cnnL'], None, None),
       ('DiffCom（无条件先验+测量一致性引导）', ['zs2_diffcom'], None, None), ('ICDM 双模型', ['zs2_icdm'], None, None),
       ('DiT 扩散（无专家）', ['zs2_dit'], None, None), ('DiT 扩散-272（容量匹配）', ['zs2_ditL'], None, None), ('MoE 四专家（旧路由）', ['zs2_moe'], None, None),
       ('仅时频专家', ['zs2_ex_TF'], None, None), ('DiT 直接回归（不扩散）', ['zs2_dreg'], None, None), ('MoE 直接回归（不扩散）', ['zs2_mreg'], None, None), ('CNN + 专家特征', ['zs2_mvcnn'], None, None)]
for sn, sub in SUBS: table(f'旧机 100% · {sn}', '旧机zs2', 'test', sub, OLD)
# ---------- 逐类留出 LOTO ----------
md.append('\n## 四、逐类留出（LOTO：只拿走一个类型，其余 7 类参与训练；表内 = 被留出类型上的结果）\n')
for T in ('LFM', 'RFM', 'MT', 'PL', 'WB'):
    M = [('CNN（3 种子）', S(f'lo_{T}_cnn', (42, 43, 44)), None, None), ('DiT 无专家', [f'lo_{T}_dit0_s42'], None, None), ('时频专家（3 种子）', S(f'lo_{T}_tf', (42, 43, 44)), None, None),
         ('四专家·旧路由（3 种子）', S(f'lo_{T}_legacy4', (42, 43, 44)), None, None), ('四专家·静态均匀', [f'lo_{T}_static4_s42'], None, None),
         ('双因素路由', [f'lo_{T}_dual_s42'], None, None), ('双因素+共享专家', [f'lo_{T}_dual_shared_s42'], None, None), ('双因素+属性监督（3 种子）', S(f'lo_{T}_dual_attr', (42, 43, 44)), None, None),
         ('U-Net 扩散 + 时频（lof）', [f'lof_{T}_tf_s42'], None, None), ('U-Net 扩散 + 共享四专家（lof）', [f'lof_{T}_moe_s42'], None, None), ('CNN（lof，单种子）', [f'lof_{T}_cnn_s42'], None, None)]
    table(f'留出 {T} · 被留出类型', f'LOTO-{T}', 'test', ('type', [T]), M)
open(os.path.join(OUT, 'fewshot_all_baselines.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
with open(os.path.join(OUT, 'fewshot_all_baselines.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['场景', '测试集', '子集'] + HDR); w.writerows(rows_csv)
print('rows', len(rows_csv))
