"""自动选择脚本：按验证集 BER 在候选中选胜出者，写 exp_state/<机器>/winners.json（由 runner 在启动后续任务前读取）。
所有选择只使用 train/val 产生的 val_summary.json（最佳验证 BER），**不看任何 test 结果**（除 e12 的权重在 val 上选）。
规则（预先登记）：候选按（多种子均值的）验证 BER 升序；BER 差 < 0.001 视为并列，取参数更少者；oracle 变体与对照不参选。
用法: python exp/selection.py s1|s2|aug|e12      （环境变量 EXP_STATE_DIR 指定输出目录；缺省 exp_state/B）"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import registry as REG  # noqa: E402

STATE = os.environ.get('EXP_STATE_DIR') or os.path.join(ROOT, 'exp_state', 'B')
TIE = 0.001


def summ(name):
    f = os.path.join(ROOT, 'runs', name, 'val_summary.json')
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d if d.get('status') == 0 and d.get('best_val_ber') is not None else None


def score(cfg, seeds=(42,)):
    vals, params = [], None
    for s in seeds:
        d = summ('%s_s%d' % (cfg, s))
        if d:
            vals.append(d['best_val_ber']); params = d['params']
    return (sum(vals) / len(vals), params, len(vals)) if vals else None


def rank(cfgs, seeds=(42,), scorer=None):
    sc = [(c, (scorer or score)(c, seeds)) for c in cfgs]
    sc = [(c, s) for c, s in sc if s]
    sc.sort(key=lambda x: x[1][0])
    out = []
    for c, s in sc:                                    # 并列者取参数少
        out.append((c, s))
    i = 0
    while i + 1 < len(out):
        if abs(out[i][1][0] - out[i + 1][1][0]) < TIE and (out[i + 1][1][1] or 0) < (out[i][1][1] or 0):
            out[i], out[i + 1] = out[i + 1], out[i]
        i += 1
    return out


def load():
    f = os.path.join(STATE, 'winners.json')
    return json.load(open(f)) if os.path.exists(f) else {}


def save(w, log):
    os.makedirs(STATE, exist_ok=True)
    tmp = os.path.join(STATE, 'winners.json.tmp')
    json.dump(w, open(tmp, 'w'), indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(STATE, 'winners.json'))
    open(os.path.join(STATE, 'selection_log.txt'), 'a').write(log + '\n')
    print(log)


def norm(args):
    return ' '.join(sorted(args.split())) if False else ' '.join(args.split())


def s1():
    w, log = load(), ['[s1]']

    def fmt(r):
        return ['%s=%.4f' % (c, s[0]) for c, s in r]
    r = rank(list(REG.BACKBONE_ARGS))
    if r:
        w['bb_top1'] = REG.BACKBONE_ARGS[r[0][0]]; w['bb_top1_is_dit'] = r[0][0] == 'e1_tf'
        w['bb_top2'] = REG.BACKBONE_ARGS[r[1][0]] if len(r) > 1 else ''; w['bb_top2_is_dit'] = (r[1][0] == 'e1_tf') if len(r) > 1 else True
        log.append('  backbone: %s' % fmt(r))
    r = rank(list(REG.TOK_ARGS))
    if r:
        w['tok_top1'] = REG.TOK_ARGS[r[0][0]]; w['tok_default'] = r[0][0] == 'e1_tf'
        log.append('  tf_tokens: %s' % fmt(r))
    r = rank(list(REG.COND_ARGS))
    if r:
        w['cond_top1'] = REG.COND_ARGS[r[0][0]]; w['cond_default'] = r[0][0] == 'e1_tf'
        log.append('  cond_mode: %s' % fmt(r))
    r = rank(REG.MOE_POOL)
    if r:
        w['moe_top1'] = REG.CFG[r[0][0]]; w['moe_top1_cfg'] = r[0][0]; w['top1_dup'] = r[0][0] in ('e1_tf', 'e1_four')
        w['moe_top2'] = REG.CFG[r[1][0]] if len(r) > 1 else w.get('moe_top2', ''); w['moe_top2_cfg'] = r[1][0] if len(r) > 1 else ''
        w['top2_dup'] = (r[1][0] in ('e1_tf', 'e1_four')) if len(r) > 1 else True
        log.append('  moe_pool: %s' % fmt(r))
        sc = {c: s[0] for c, s in r}
        base = sc.get('e1_four', 9.0)
        # 组合 = 专家集(含共享) + 属性监督 + 路由 + 融合：每一项只在“优于四专家基线”时才加入
        ex = [c for c in ('e1_tf', 'e1_four', 'e2_shared4', 'e2_sharedTF') if c in sc]
        s_cfg = min(ex, key=lambda c: sc[c]) if ex else 'e1_four'
        att = [c for c in ('e3_joint4', 'e3_prefrz4', 'e3_preft4') if c in sc]
        a_cfg = min(att, key=lambda c: sc[c]) if att and min(sc[c] for c in att) < base - 1e-4 else None
        rt = [c for c in ('e4_dual', 'e4_noint', 'e4_morph', 'e4_strength', 'e4_prior') if c in sc]
        r_cfg = min(rt, key=lambda c: sc[c]) if rt and min(sc[c] for c in rt) < base - 1e-4 else None
        parts = [REG.CFG[s_cfg]]
        if a_cfg:
            parts.append('--attr_mode ' + {'e3_joint4': 'joint', 'e3_prefrz4': 'pretrain_freeze', 'e3_preft4': 'pretrain_ft', 'e3_jointTF': 'joint'}[a_cfg])
            if a_cfg in ('e3_prefrz4', 'e3_preft4'):
                parts.append('--attr_pretrain_epochs %d' % (15 if profile_is_real() else 1))
        if r_cfg:
            parts.append(REG.CFG[r_cfg])
        if 'e5_topk' in sc and r_cfg and r_cfg != 'e4_prior' and sc['e5_topk'] < sc.get('e4_dual', 9.0) - 1e-4:
            parts.append('--fusion topk_mask')
        elif 'e5_sum' in sc and sc['e5_sum'] < base - 1e-4:
            parts.append('--fusion sum_single')
        comp = norm(' '.join(p for p in parts if p))
        w['moe_compose'] = comp
        w['compose_dup'] = comp in (norm(w['moe_top1']), norm(w['moe_top2']))
        log.append('  compose = "%s" (dup=%s)' % (comp, w['compose_dup']))
    # 门控（E11）：在 rdTF 上比较各变体的交叉验证 BER，只有明显优于 λ≡1 才启用
    best_g, best_cv, lam1 = None, None, None
    for g in REG.GATE_VARIANTS:
        f = os.path.join(ROOT, 'runs', 'e10_rdTF_s42', 'gate_%s.json' % g)
        if os.path.exists(f) and g != 'oracle':
            d = json.load(open(f)); lam1 = d['ber_lambda1']
            if best_cv is None or d['cv_ber'] < best_cv:
                best_g, best_cv = g, d['cv_ber']
    if best_g is not None:
        use = best_cv < lam1 - 5e-4
        v, loc = REG.GATE_VARIANTS[best_g] if use else ('none', False)
        w['gate_variant_base'] = v; w['gate_local_flag'] = '--local' if loc else ''; w['gate_variant'] = best_g if use else 'none'
        log.append('  gate: best=%s cv_ber=%.4f vs λ≡1 %.4f -> 使用 %s' % (best_g, best_cv, lam1, w['gate_variant']))
    save(w, '\n'.join(log))


def profile_is_real():
    return os.environ.get('EXP_PROFILE', 'real') == 'real'


def s2():
    w, log = load(), ['[s2]']
    seeds3 = (42, 43, 44)

    def val(name):
        d = summ(name)
        return d['best_val_ber'] if d else None

    def mean(names):
        v = [val(n) for n in names]; v = [x for x in v if x is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)
    default, _ = mean(['e1_tf_s%d' % s for s in seeds3])
    log.append('  TF-only 默认(3 种子)=%s' % default)
    # 主干 / 局部TF / 条件注入：先看各单项在 3 种子上是否优于默认，再看组合（2 种子）
    best_single = (default, '', '')
    for kind, key, amap, pool in (('backbone', 'bb_top1', REG.BACKBONE_ARGS, REG.BACKBONE_ARGS), ('tf_tokens', 'tok_top1', REG.TOK_ARGS, REG.TOK_ARGS), ('cond_mode', 'cond_top1', REG.COND_ARGS, REG.COND_ARGS)):
        r = rank(list(pool))
        if not r or r[0][0] == 'e1_tf':
            continue
        vals = [val('%s_s42' % r[0][0])] + [val('c_%s_s%d' % (key, s)) for s in (43, 44)]
        vals = [v for v in vals if v is not None]
        if vals and default is not None and sum(vals) / len(vals) < best_single[0] - 5e-4:
            best_single = (sum(vals) / len(vals), amap[r[0][0]], kind)
        log.append('  %s 单项 %s: %s' % (kind, r[0][0], [round(v, 4) for v in vals]))
    combo, _ = mean(['c_basecombo_s42', 'c_basecombo_s43'])
    d2, _ = mean(['e1_tf_s42', 'e1_tf_s43'])
    combo_args = norm('%s %s %s' % (w.get('bb_top1', ''), w.get('tok_top1', ''), w.get('cond_top1', '')))
    if combo is not None and d2 is not None and combo < d2 - 5e-4 and combo_args:
        fb, why = combo_args, '组合 %.4f < 默认 %.4f' % (combo, d2)
    elif best_single[2]:
        fb, why = best_single[1], '单项 %s %.4f' % (best_single[2], best_single[0])
    else:
        fb, why = '', '默认 DiT + 全局 TF + 交叉注意力'
    w['final_base'] = fb; w['backbone_final'] = fb
    log.append('  final_base = "%s"  (%s)' % (fb, why))
    # MoE 最终：候选在 3 种子均值上比较
    cands = {}
    for nm, names, args in (('e1_tf', ['e1_tf_s%d' % s for s in seeds3], REG.CFG['e1_tf']), ('e1_four', ['e1_four_s%d' % s for s in seeds3], REG.CFG['e1_four']),
                            ('compose', ['c_moe_compose_s%d' % s for s in seeds3], w.get('moe_compose', '')),
                            ('top1', ['%s_s42' % w.get('moe_top1_cfg', 'e1_tf')] + ['c_moe_top1_s%d' % s for s in (43, 44)], w.get('moe_top1', '')),
                            ('top2', ['%s_s42' % w.get('moe_top2_cfg', 'e1_tf')] + ['c_moe_top2_s%d' % s for s in (43, 44)], w.get('moe_top2', ''))):
        m, n = mean(names)
        if m is not None:
            cands[nm] = (m, args, n)
    if w.get('compose_dup'):
        cands.pop('compose', None)
    if cands:
        order = sorted(cands.items(), key=lambda kv: kv[1][0])
        w['moe_final'] = order[0][1][1]; w['moe_final_name'] = order[0][0]
        log.append('  moe_final: %s -> "%s"' % ([(k, round(v[0], 4), v[2]) for k, v in order], w['moe_final']))
    save(w, '\n'.join(log))


def aug():
    w, log = load(), ['[aug]']
    singles = [a for a in REG.AUG_SINGLES]
    r = rank(['e14_%s' % a for a in singles])
    if r:
        order = [c.replace('e14_', '') for c, _ in r]
        w['aug_top1'] = order[0]; w['aug_top2'] = order[1] if len(order) > 1 else order[0]
        w['aug_combo1'] = ','.join(order[:2]) if len(order) > 1 else order[0]
        w['aug_combo2'] = ','.join([order[0], order[2]]) if len(order) > 2 else w['aug_combo1']
        log.append('  singles: %s -> combo1=%s combo2=%s' % (['%s=%.4f' % (c, s[0]) for c, s in r], w['aug_combo1'], w['aug_combo2']))
    save(w, '\n'.join(log))


def e12():
    w, log = load(), ['[e12]']
    f = os.path.join(ROOT, '..', 'reports', 'e12_val.json')
    if os.path.exists(f):
        rows = json.load(open(f))
        rows.sort(key=lambda r: (r['ber'], r['w_obs'] + r['w_band']))
        w['e12_w_obs'], w['e12_w_band'] = str(rows[0]['w_obs']), str(rows[0]['w_band'])
        log.append('  best val: w_obs=%s w_band=%s ber=%.4f' % (rows[0]['w_obs'], rows[0]['w_band'], rows[0]['ber']))
    save(w, '\n'.join(log))


if __name__ == '__main__':
    {'s1': s1, 's2': s2, 'aug': aug, 'e12': e12}[sys.argv[1]]()
