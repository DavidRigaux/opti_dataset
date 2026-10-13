#!/usr/bin/env python3
"""
optimise6.py — instance 6 uniquement, ne garde que la meilleure solution.

solveur.py — « IA responsables » : solveur complet en un seul fichier (solutions légales).

Prérequis : pip install numpy scipy sortedcontainers
Placer ce fichier dans le dossier des instances (.json), puis :

    python solveur.py                       # instance 6, tous les réglages par défaut, tourne jusqu'à Ctrl+C
    python solveur.py --instance 7_big.json --procs 2 --decalage 300
    python solveur.py --procs 4 --mip 1200 --runs 8
    python solveur.py --instance 5_energy.json --procs 2 --mip 600 --lns 14400   # MIP puis 4 h de LNS par run
                                                                                 # (pip install highspy)

Chaque run (une graine) : LP -> MIP restreint aux modèles marginaux -> affectation des datasets
-> MIP résiduel -> remplissage glouton -> recherche locale. Plusieurs runs tournent en parallèle
(un processus chacun) et de nouveaux runs sont relancés avec de nouvelles graines tant qu'on ne
fait pas Ctrl+C (ou jusqu'à --runs).
Toutes les solutions valides vont dans out/ ; la meilleure est copiée dans <instance>_MEILLEUR.json
(revalidée, prête à soumettre). Logs détaillés dans logs/.
"""
import json, sys, time, bisect, os, contextlib, random, re, glob, shutil, subprocess, argparse, gc
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from sortedcontainers import SortedList

# ===== LP =====

T = 'ntic'

@contextlib.contextmanager
def silence():
    """coupe stdout/stderr au niveau système (messages C++ de HiGHS)"""
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    nul = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(nul, 1); os.dup2(nul, 2)
        yield
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in saved + [nul]: os.close(fd)
SMAX = 10000

LP_METHOD = os.environ.get('LP_METHOD', 'highs-ds')  # simplexe dual : le plus fiable sur ce modèle

def build_and_solve(inst, datasets=None, energy=None, exclude=(), src_avail=None,
                    K=6, clones=False, clone_exclude=(), waste=None, verbose=True, SMALL=2500, KQ=10, mip_time=None, mip_gap=1e-6, mip_seed=0, FLUIDMIN=None, FDELTA=0.02, fix=None, WADD=0.0, WSMALL=1000, BUCKET=0, NB=40, return_model=False, SRCFLUID=False, BCOUNT=0, SRCW=None, WSRC=0.0, WSRCBIG=0.0, SRCBK=0, NMAX=4, PAIRS=0, PAIRMIN=1000, PAIRMAXW=200, NOFLUIDSRC=False, QPAIRS=0, QPAIRMIN=50, BIGL=0, NBF=12, BIGMIN=10000):
    M = inst['models']
    D = inst['datasets'] if datasets is None else datasets
    cap = inst['energyCap']
    en = cap if energy is None else energy
    F = np.zeros(4); C = np.zeros(4)
    by = {(t, cr): [] for t in T for cr in (0, 1)}
    for d in D:
        k = T.index(d['dataType'])
        if d['isCopyrighted']: C[k] += d['size']
        else: F[k] += d['size']
        by[(d['dataType'], d['isCopyrighted'])].append((d['size'], d['id']))
    for v in by.values(): v.sort()
    dsize = {d['id']: d['size'] for d in D}
    dcr = {d['id']: d['isCopyrighted'] for d in D}
    wf = waste or {t: 0.0 for t in T}
    cols = []; obj = []; rows = []
    R_FREE, R_TOT, R_E, R_SRC = 0, 4, 8, 9
    b = [*F, *(F + C), 1.0, 0, 0, 0, 0]
    if src_avail: b[9:13] = [src_avail[t] for t in T]
    nrow = 13
    R_LGT = {}; R_LGF = {}
    if BIGL:
        for k_, t_ in enumerate(T):
            R_LGT[t_] = nrow; b.append(float(sum(sz for cr_ in (0, 1) for sz, _ in by[(t_, cr_)] if sz >= BIGL))); nrow += 1
            R_LGF[t_] = nrow; b.append(float(sum(sz for sz, _ in by[(t_, 0)] if sz >= BIGL))); nrow += 1
    drow = {}
    eqrows = []; neq = 0
    W = BUCKET
    Q = BCOUNT
    bstat = {}; binfo = {}; dbk = {}
    if Q:
        for (t_, cr_), lst in by.items():
            info = []
            for k_ in range(0, len(lst), Q):
                ch = lst[k_:k_ + Q]
                info.append((ch[0][0], ch[-1][0], len(ch), sum(x[0] for x in ch) / len(ch)))
                for x in ch: dbk[x[1]] = k_ // Q
            binfo[(t_, cr_)] = info
    elif W:
        for (t_, cr_), lst in by.items():
            for sz, did in lst:
                k_ = (t_, cr_, sz // W); c_, s_ = bstat.get(k_, (0, 0)); bstat[k_] = (c_ + 1, s_ + sz)
    def cands_for(t, lb, ub, crs, KK):
        out = []
        for cr in crs:
            lst = by[(t, cr)]
            j = bisect.bisect_left(lst, (lb, -1))
            if Q:
                kj = j // Q; lim_pos = (kj + 1) * Q
                for q in range(j, min(j + KK, lim_pos, len(lst))):
                    if lst[q][0] > ub: break
                    out.append((lst[q][0], lst[q][1], cr))
                info = binfo[(t, cr)]; n = 0; k = kj + 1
                while k < len(info) and info[k][1] <= ub and n < NB:
                    bmin, bmax, cnt, mean = info[k]
                    out.append((mean, ('b', cr, k, bmin, bmax), cr)); n += 1; k += 1
                continue
            lim = ((lb // W) + 1) * W if W else None
            for q in range(j, min(j + KK, len(lst))):
                if lst[q][0] > ub or (W and lst[q][0] >= lim): break
                out.append((lst[q][0], lst[q][1], cr))
            if W:
                bb = lb // W + 1; n = 0
                while (bb + 1) * W - 1 <= ub and n < NB:
                    st = bstat.get((t, cr, bb))
                    if st: out.append((st[1] / st[0], ('b', cr, bb), cr)); n += 1
                    bb += 1
                    if bb * W > SMAX: break
        return out
    brow = {}
    def brow_of(t, cr, bb):
        nonlocal nrow
        key = (t, cr, bb)
        if key not in brow:
            cap_ = binfo[(t, cr)][bb][2] if Q else bstat[key][0]
            brow[key] = nrow; nrow += 1; b.append(float(cap_))
        return brow[key]
    def attach(jv, did, t, cr):
        """lignes de capacité du dataset (ou du compartiment de tailles)"""
        if isinstance(did, tuple):
            out = [(brow_of(t, cr, did[2]), jv, 1)]
            if BIGL and W and did[2] * W >= BIGL:
                mean_ = bstat[(t, cr, did[2])][1] / bstat[(t, cr, did[2])][0]
                out.append((R_LGT[t], jv, mean_))
                if not cr: out.append((R_LGF[t], jv, mean_))
            return out
        out = [(drow_of(did), jv, 1)]
        if Q: out.append((brow_of(t, cr, dbk[did]), jv, 1))
        elif W: out.append((brow_of(t, cr, dsize[did] // W), jv, 1))
        if BIGL and dsize[did] >= BIGL:
            out.append((R_LGT[t], jv, dsize[did]))
            if not cr: out.append((R_LGF[t], jv, dsize[did]))
        return out
    def gen_pairs(t, lb, ub, crs, nmax, maxw):
        allt = sorted(x_ for cr in crs for x_ in by[(t, cr)])
        sizes_ = [x_[0] for x_ in allt]; out = []
        lo_i = bisect.bisect_left(sizes_, (lb + 1) // 2); hi_i = bisect.bisect_left(sizes_, lb)
        span = max(1, (hi_i - lo_i) // (3 * nmax) + 1)
        for ia in range(lo_i, hi_i, span):
            a_sz, a_id = allt[ia]
            jb = bisect.bisect_left(sizes_, lb - a_sz)
            while jb < len(allt) and allt[jb][1] == a_id: jb += 1
            if jb < len(allt) and a_sz + allt[jb][0] <= min(ub, lb + maxw):
                out.append((a_sz + allt[jb][0] - lb, (a_id, allt[jb][1])))
        out.sort(); seen = set(); res_ = []
        for wst, pr_ in out:
            k2 = tuple(sorted(pr_))
            if k2 in seen: continue
            seen.add(k2); res_.append(pr_)
            if len(res_) >= nmax: break
        return res_
    _bb_cache = {}
    def finish_opts(t, lb, crs):
        """(spec, taille moyenne, cr) : None = pas de pièce ; sinon un compartiment < BIGL réparti régulièrement"""
        out = [(None, 0.0, None)]
        for cr in crs:
            if (t, cr) not in _bb_cache:
                _bb_cache[(t, cr)] = sorted(bb for (t_, cr_, bb) in bstat if t_ == t and cr_ == cr)
            lim_ = min(BIGL, lb)
            bbs = [bb for bb in _bb_cache[(t, cr)] if (bb + 1) * W <= lim_]
            if not bbs: continue
            stp = max(1, len(bbs) // NBF)
            for bb in bbs[::stp]:
                c_, s_ = bstat[(t, cr, bb)]; out.append((('b', cr, bb), s_ / c_, cr))
        return out
    def drow_of(did):
        nonlocal nrow
        if did not in drow:
            drow[did] = nrow; nrow += 1; b.append(1.0)
        return drow[did]
    ubv = []
    def addcol(key, c, ub=1.0):
        cols.append(key); obj.append(-c); ubv.append(ub); return len(cols) - 1
    for m in M:
        i = m['id']
        if i in exclude and (not clones or i in clone_exclude): continue
        reqs = m['dataRequirements']
        neg = any(r['upperBound'] < r['lowerBound'] for r in reqs.values())
        E = m['energyCost'] / en
        if clones and i not in clone_exclude:
            jc = addcol(('c', i, None), m['value'] // 2)
            rows.append((R_E, jc, E))
            for t in reqs: rows.append((R_SRC + T.index(t), jc, 1))
        if i in exclude: continue
        ex = nrow; nrow += 1; b.append(1.0)
        jx = None
        if not neg:
            jx = addcol(('x', i, None), m['value'])
            rows += [(ex, jx, 1), (R_E, jx, E)]
        jz = addcol(('z', i, None), m['value'] // 2)
        rows += [(ex, jz, 1), (R_E, jz, E)]
        for t, r in reqs.items():
            k = T.index(t); l = r['lowerBound'] * (1 + wf[t])
            lb, ub = r['lowerBound'], r['upperBound']
            if lb < WSMALL: l += WADD
            jy = addcol(('y', i, t), 0.0)
            rows.append((R_SRC + k, jy, 1))
            if ub < lb:
                eqrows.append((neq, jz, 1)); eqrows.append((neq, jy, -1)); neq += 1
                continue
            small = lb < SMALL
            if small:
                if jx is not None:
                    eqrows.append((neq, jx, -1))
                    if QPAIRS and lb >= QPAIRMIN:
                        for pr_ in gen_pairs(t, lb, ub, (0,), QPAIRS, PAIRMAXW):
                            jq = addcol(('qx', i, (t, ('pair',) + tuple(pr_))), 0.0)
                            rows += [(R_TOT + k, jq, sum(dsize[d_] for d_ in pr_)), (R_FREE + k, jq, sum(dsize[d_] for d_ in pr_))]
                            for d_ in pr_: rows += attach(jq, d_, t, dcr[d_])
                            eqrows.append((neq, jq, 1))
                    for sz, did, cr in cands_for(t, lb, ub, (0,), KQ):
                        jq = addcol(('qx', i, (t, did)), 0.0)
                        rows += [(R_TOT + k, jq, sz), (R_FREE + k, jq, sz)] + attach(jq, did, t, cr)
                        eqrows.append((neq, jq, 1))
                    if FLUIDMIN is not None and lb >= FLUIDMIN:
                        jf = addcol(('fx', i, t), 0.0); lf = lb * (1 + FDELTA)
                        rows += [(R_TOT + k, jf, lf), (R_FREE + k, jf, lf)]; eqrows.append((neq, jf, 1))
                    neq += 1
                eqrows.append((neq, jz, -1)); eqrows.append((neq, jy, 1))
                if FLUIDMIN is not None and lb >= FLUIDMIN:
                    jf = addcol(('fz', i, t), 0.0)
                    rows.append((R_TOT + k, jf, lb * (1 + FDELTA))); eqrows.append((neq, jf, 1))
                if QPAIRS and lb >= QPAIRMIN:
                    for pr_ in gen_pairs(t, lb, ub, (1, 0), QPAIRS, PAIRMAXW):
                        jq = addcol(('qz', i, (t, ('pair',) + tuple(pr_))), 0.0)
                        rows += [(R_TOT + k, jq, sum(dsize[d_] for d_ in pr_))]
                        for d_ in pr_:
                            rows += attach(jq, d_, t, dcr[d_])
                            if not dcr[d_]: rows.append((R_FREE + k, jq, dsize[d_]))
                        eqrows.append((neq, jq, 1))
                for sz, did, cr in cands_for(t, lb, ub, (1, 0), KQ):
                    jq = addcol(('qz', i, (t, did)), 0.0)
                    rows += [(R_TOT + k, jq, sz)] + attach(jq, did, t, cr)
                    if not cr: rows.append((R_FREE + k, jq, sz))
                    eqrows.append((neq, jq, 1))
                neq += 1
            elif BIGL and W and lb >= BIGMIN:
                # gros besoin = gros oeuvre (datasets >= BIGL, en vrac) + pièce de finition explicite
                if jx is not None:
                    eqrows.append((neq, jx, -1))
                    for spec, ms, cr in finish_opts(t, lb, (0,)):
                        jg = addcol(('gx', i, (t, spec)), 0.0); bulk = lb - ms
                        rows += [(R_FREE + k, jg, lb), (R_TOT + k, jg, lb), (R_LGT[t], jg, bulk), (R_LGF[t], jg, bulk)]
                        if spec is not None: rows += attach(jg, spec, t, cr)
                        eqrows.append((neq, jg, 1))
                    neq += 1
                eqrows.append((neq, jz, -1)); eqrows.append((neq, jy, 1))
                for spec, ms, cr in finish_opts(t, lb, (1, 0)):
                    jg = addcol(('gz', i, (t, spec)), 0.0); bulk = lb - ms
                    rows += [(R_TOT + k, jg, lb), (R_LGT[t], jg, bulk)]
                    if spec is not None:
                        rows += attach(jg, spec, t, cr)
                        if not cr: rows.append((R_FREE + k, jg, ms))
                    eqrows.append((neq, jg, 1))
                neq += 1
            else:
                if jx is not None:
                    rows += [(R_FREE + k, jx, l), (R_TOT + k, jx, l)]
                rows.append((R_TOT + k, jz, l))
                rows.append((R_TOT + k, jy, -l))
                rr = nrow; nrow += 1; b.append(0.0); rows += [(rr, jy, 1), (rr, jz, -1)]
        if len(reqs) == 1 and not neg:
            t, r = next(iter(reqs.items())); k = T.index(t)
            lb, ub = r['lowerBound'], r['upperBound']
            if SRCBK and Q and lb >= SRCBK:
                # source remplie par plusieurs datasets : nombre de datasets pris dans chaque tranche
                jv = addcol(('v', i, t), 0.0)
                rows += [(ex, jv, 1), (R_E, jv, E), (R_SRC + k, jv, -1)]
                rlo = nrow; rhi = nrow + 1; rcn = nrow + 2; nrow += 3; b.extend([0.0, 0.0, 0.0])
                rows += [(rlo, jv, lb), (rhi, jv, -ub), (rcn, jv, -NMAX)]
                for cr in (1, 0):
                    for kk, (bmin, bmax, cnt, mean) in enumerate(binfo[(t, cr)]):
                        if bmin > ub: break
                        if mean < lb / NMAX: continue
                        u_ = min(cnt, NMAX, int(ub // max(bmin, 1)))
                        if u_ < 1: continue
                        jn = addcol(('n', i, (t, ('b', cr, kk, bmin, bmax))), 0.0, ub=float(u_))
                        rows += [(rlo, jn, -mean), (rhi, jn, mean), (rcn, jn, 1), (R_TOT + k, jn, mean),
                                 (brow_of(t, cr, kk), jn, 1)]
                        if not cr: rows.append((R_FREE + k, jn, mean))
                continue
            combos = []
            if PAIRS and PAIRMIN <= lb <= SMAX * 3:
                allt = sorted(by[(t, 0)] + by[(t, 1)])
                sizes_ = [x_[0] for x_ in allt]
                def pairs_for(lo_, hi_, excl, nmax):
                    out = []
                    lo_i = bisect.bisect_left(sizes_, (lo_ + 1) // 2); hi_i = bisect.bisect_left(sizes_, lo_)
                    span = max(1, (hi_i - lo_i) // (3 * nmax) + 1)
                    for ia in range(lo_i, hi_i, span):
                        a_sz, a_id = allt[ia]
                        if a_id in excl: continue
                        jb = bisect.bisect_left(sizes_, lo_ - a_sz)
                        while jb < len(allt) and (allt[jb][1] == a_id or allt[jb][1] in excl): jb += 1
                        if jb < len(allt) and a_sz + allt[jb][0] <= hi_:
                            out.append((a_sz + allt[jb][0] - lo_, (a_id, allt[jb][1])))
                    return out
                hi_c = min(ub, lb + PAIRMAXW)
                if lb <= SMAX * 2:
                    combos = pairs_for(lb, hi_c, (), PAIRS)
                if lb > SMAX:  # triplets : un gros dataset + une paire exacte pour le reste
                    lo3 = bisect.bisect_left(sizes_, lb // 3); hi3 = bisect.bisect_left(sizes_, min(lb, SMAX + 1))
                    stp = max(1, (hi3 - lo3) // (2 * PAIRS))
                    big = allt[lo3:hi3:stp]
                    for c_sz, c_id in big:
                        for w_, pr_ in sorted(pairs_for(lb - c_sz, hi_c - c_sz, (c_id,), 3))[:2]:
                            combos.append((w_, (c_id,) + pr_))
                combos.sort()
            seen = set(); ncomb = 0
            for wst, ids_ in combos:
                if ncomb >= PAIRS: break
                key2 = tuple(sorted(ids_))
                if key2 in seen or len(set(ids_)) < len(ids_): continue
                seen.add(key2); ncomb += 1
                jp = addcol(('p', i, (t,) + tuple(ids_)), 0.0)
                rows += [(ex, jp, 1), (R_E, jp, E), (R_SRC + k, jp, -1), (R_TOT + k, jp, sum(dsize[d_] for d_ in ids_))]
                for did_ in ids_:
                    cr_ = dcr[did_]
                    rows += attach(jp, did_, t, cr_)
                    if not cr_: rows.append((R_FREE + k, jp, dsize[did_]))
            if ncomb and NOFLUIDSRC and lb > SMAX:
                continue
            cands = [] if SRCFLUID else cands_for(t, lb, ub, (1, 0), K)
            for sz, did, cr in cands:
                ju = addcol(('u', i, (t, did)), 0.0)
                rows += [(ex, ju, 1), (R_E, ju, E), (R_TOT + k, ju, sz), (R_SRC + k, ju, -1)] + attach(ju, did, t, cr)
                if not cr: rows.append((R_FREE + k, ju, sz))
            if BIGL and W and lb >= BIGMIN:
                for spec, ms, cr in finish_opts(t, lb, (1, 0)):
                    jg = addcol(('g', i, (t, spec)), 0.0); bulk = lb - ms
                    rows += [(ex, jg, 1), (R_E, jg, E), (R_SRC + k, jg, -1), (R_TOT + k, jg, lb), (R_LGT[t], jg, bulk)]
                    if spec is not None:
                        rows += attach(jg, spec, t, cr)
                        if not cr: rows.append((R_FREE + k, jg, ms))
            elif lb > (SMAX if SRCW is None else SRCW) or SRCFLUID:
                jw = addcol(('w', i, t), 0.0)
                lw = lb * (1 + wf[t]) + WSRCBIG if lb > SMAX else lb * (1 + FDELTA) + WSRC
                rows += [(ex, jw, 1), (R_E, jw, E), (R_TOT + k, jw, lw), (R_SRC + k, jw, -1)]
    r_, c_, v_ = zip(*rows)
    A = sp.csr_matrix((v_, (r_, c_)), shape=(nrow, len(cols)))
    t0 = time.time()
    if eqrows:
        r_, c_, v_ = zip(*eqrows)
        Aeq = sp.csr_matrix((v_, (r_, c_)), shape=(neq, len(cols))); beq = np.zeros(neq)
    else:
        Aeq = None; beq = None
    if return_model:
        return np.array(obj), A, np.array(b, dtype=float), Aeq, beq, cols, np.array(ubv)
    if mip_time:
        from scipy.optimize import milp, LinearConstraint, Bounds
        cons = [LinearConstraint(A, -np.inf, np.array(b))]
        if Aeq is not None: cons.append(LinearConstraint(Aeq, beq, beq))
        c = np.array(obj)
        if mip_seed:  # perturbation infime pour diversifier la recherche selon la graine
            rng = np.random.default_rng(mip_seed)
            c = c * (1 + 1e-6 * rng.random(len(c)))
        lo = np.zeros(len(cols)); hi = np.array(ubv, dtype=float)
        if fix:
            for j, key in enumerate(cols):
                if key in fix: lo[j] = hi[j] = fix[key]
        with silence():
            integ = np.array([0 if key[0] == 'n' else 1 for key in cols])  # comptes par tranche : continus
            r = milp(c, integrality=integ, bounds=Bounds(lo, hi), constraints=cons,
                     options={'time_limit': mip_time, 'mip_rel_gap': mip_gap, 'disp': False})
        if verbose: print(f'MIP st={r.status} ({"optimal" if r.status == 0 else "limite de temps"}) val={(-np.array(obj) @ r.x) if r.x is not None else float('nan'):.0f} {time.time()-t0:.1f}s', flush=True)
        class R: pass
        res = R(); res.x = r.x if r.x is not None else np.zeros(len(cols)); res.status = r.status
        res.fun = r.fun if r.x is not None else 0; res.prices = None
        return res, cols
    if verbose:
        print(f'  LP construit : {len(cols)} variables, {nrow} lignes ({time.time()-t0:.0f}s depuis la fin de la construction) -> résolution ({LP_METHOD})...', flush=True)
    res = linprog(np.array(obj), A_ub=A, b_ub=np.array(b), A_eq=Aeq, b_eq=beq,
                  bounds=np.column_stack([np.zeros(len(ubv)), np.array(ubv)]) if any(u != 1 for u in ubv) else (0, 1), method=LP_METHOD)
    if verbose:
        print(f'LP st={res.status} val={-res.fun:.1f} vars={len(cols)} rows={nrow} {time.time()-t0:.1f}s', flush=True)
    du = res.ineqlin.marginals
    res.prices = {t: (-du[R_FREE + k] - du[R_TOT + k], -du[R_TOT + k]) for k, t in enumerate(T)}
    return res, cols

# ===== AFFECTATION =====

class Pools:
    def __init__(self, D):
        self.p = {t: [SortedList(), SortedList()] for t in T}
        self.D = D
        self.pr = {t: {0: 1.0, 1: 1.0} for t in T}
        self.left = {(t, cr): 0 for t in T for cr in (0, 1)}
        for d in D:
            self.p[d['dataType']][d['isCopyrighted']].add((d['size'], d['id'])); self.left[(d['dataType'], d['isCopyrighted'])] += d['size']
    def put_back(self, ids):
        for i in ids:
            d = self.D[i]; self.p[d['dataType']][d['isCopyrighted']].add((d['size'], d['id'])); self.left[(d['dataType'], d['isCopyrighted'])] += d['size']
    def take_id(self, i):
        d = self.D[i]; it = (d['size'], i); p = self.p[d['dataType']][d['isCopyrighted']]
        if it in p: p.remove(it); self.left[(d['dataType'], d['isCopyrighted'])] -= d['size']; return True
        return False
    def take_spec(self, t, lb, ub, spec, W):
        """spec = id de dataset, ou ('b', cr, compartiment) : prend le plus petit dataset du compartiment dans [lb,ub]"""
        if not isinstance(spec, tuple):
            return spec if self.take_id(spec) else None
        if spec[0] == 'pair':
            got = []
            for d in spec[1:]:
                if self.take_id(d): got.append(d)
                else:
                    self.put_back(got); return None
            return got
        W = W or 1
        if len(spec) == 5:
            _, cr, bb, bmin, bmax = spec; lo = max(lb, bmin); hi = min(ub, bmax)
        else:
            _, cr, bb = spec; lo = max(lb, bb * W); hi = min(ub, (bb + 1) * W - 1)
        p = self.p[t][cr]; i = p.bisect_left((lo, -1))
        if i < len(p) and p[i][0] <= hi:
            it = p[i]; p.remove(it); self.left[(t, cr)] -= it[0]; return it[1]
        return None
    def remaining(self):
        return [self.D[i] for t in T for cr in (0, 1) for _, i in self.p[t][cr]]
    def _pair(self, pools, pr, lo, hi, pair_iter, excl=None, best=None):
        for ia, (cra, pa) in enumerate(pools):
            for crb, pb in pools[ia:]:
                i = pa.bisect_left(((lo + 1) // 2, -1)); cnt = 0
                while i < len(pa) and pa[i][0] < lo and cnt < pair_iter:
                    a = pa[i]; cnt += 1; i += 1
                    if a == excl: continue
                    j = pb.bisect_left((lo - a[0], -1))
                    while j < len(pb) and (pb[j] == a or pb[j] == excl): j += 1
                    if j < len(pb) and pb[j][0] <= hi - a[0]:
                        c = pr[cra] * a[0] + pr[crb] * pb[j][0]
                        if best is None or c < best[0] - 1e-12:
                            best = (c, [(cra, a), (crb, pb[j])], a[0] + pb[j][0] - lo)
                            if best[2] == 0: break
        return best
    def _finish(self, pools, pr, lo, hi, pair_iter):
        best = None
        for cr, p in pools:
            i = p.bisect_left((lo, -1))
            if i < len(p) and p[i][0] <= hi:
                c = pr[cr] * p[i][0]
                if best is None or c < best[0]: best = (c, [(cr, p[i])], p[i][0] - lo)
        exact_cost = min(pr[cr] for cr, _ in pools) * lo
        if best is None or best[0] > exact_cost * 1.0005 + 1e-12:
            best = self._pair(pools, pr, lo, hi, pair_iter, best=best)
        if TRIPLE_ITER and best is not None and best[2] > FIN_TOL:
            for cr, p in pools:
                i = p.bisect_left((lo, -1)) - 1; cnt = 0
                while i >= 0 and cnt < TRIPLE_ITER:
                    cc = p[i]; i -= 1; cnt += 1
                    r = lo - cc[0]
                    if r < 2: continue
                    q = self._pair(pools, pr, r, hi - cc[0], 60, excl=cc)
                    if q is not None and pr[cr] * cc[0] + q[0] < best[0] - 1e-12:
                        best = (pr[cr] * cc[0] + q[0], [(cr, cc)] + q[1], q[2])
        return best
    def fill_dp(self, t, lb, ub, crs, pr):
        """remplissage EXACT par programmation dynamique sur tout le stock restant :
        coût minimal (prix x taille) parmi tous les sous-ensembles dont la somme tombe dans [lb, ub]"""
        items = [(cr, it) for cr in crs for it in self.p[t][cr] if it[0] <= ub]
        if not items or ub < lb: return None
        ub = int(ub); dp = np.full(ub + 1, np.inf); dp[0] = 0.0
        take = np.zeros((len(items), ub + 1), dtype=bool)
        for k, (cr, it) in enumerate(items):
            sz = it[0]; c = pr[cr] * (sz + DPK)
            new = dp[:-sz] + c; seg = dp[sz:]; m = new < seg - 1e-12
            seg[m] = new[m]; take[k, sz:][m] = True
        seg = dp[lb:ub + 1]
        if not np.isfinite(seg).any(): return None
        S = lb + int(np.argmin(seg)); out = []
        for k in range(len(items) - 1, -1, -1):
            if S == 0: break
            if take[k, S]:
                cr, it = items[k]; out.append((cr, it)); S -= it[0]
        for cr, it in out: self.p[t][cr].remove(it); self.left[(t, cr)] -= it[0]
        return [it[1] for _, it in out]
    def fill(self, t, lb, ub, crs, pair_iter=150, pr=None):
        if DPFILL and lb >= DPMIN:
            return self.fill_dp(t, lb, ub, crs, pr or self.pr[t])
        """pr: price per unit for cr 0/1. crs: allowed pools"""
        pr = pr or self.pr[t]
        pools = sorted([(cr, self.p[t][cr]) for cr in crs], key=lambda x: pr[x[0]])
        taken = []
        lo, hi = lb, ub
        def take(cr, item):
            self.p[t][cr].remove(item); taken.append((cr, item)); self.left[(t, cr)] -= item[0]
        if hi < lo: return None
        while True:
            if lo <= 2 * SMAX:
                best = self._finish(pools, pr, lo, hi, pair_iter)
                if best is not None:
                    for cr, it in best[1]: take(cr, it)
                    return [it[1] for _, it in taken]
            took = False
            if lo > 2 * SMAX:
                for cr, p in pools:
                    if len(p): take(cr, p[-1]); took = True; break
            else:
                tgt = max(lo - SMAX // 2, 1) if lo > SMAX // 2 else max(lo // 2, 1)
                for cr, p in pools:
                    i = p.bisect_left((tgt, -1))
                    if i < len(p) and p[i][0] < lo: take(cr, p[i]); took = True; break
                    if i - 1 >= 0: take(cr, p[i - 1]); took = True; break
            if not took:
                for cr, it in taken: self.p[t][cr].add(it); self.left[(t, cr)] += it[0]
                return None
            lo -= taken[-1][1][0]; hi -= taken[-1][1][0]

class Sol:
    def __init__(self, inst):
        self.inst = inst; self.M = inst['models']; self.N = len(self.M)
        self.pools = Pools(inst['datasets'])
        self.energy = 0; self.data = {}; self.mm = []; self.status = {}
        self.src_free = {t: [] for t in T}
        self.clones = set()
    def eleft(self): return self.inst['energyCap'] - self.energy
    def try_src(self, m, did=None):
        mod = self.M[m]
        if m in self.status or mod['energyCost'] > self.eleft(): return False
        t, r = next(iter(mod['dataRequirements'].items()))
        if isinstance(did, tuple) and did and did[0] == 'pair':
            got_ = []
            for d_ in did[1:]:
                if self.pools.take_id(d_): got_.append(d_)
                else: break
            if len(got_) == len(did) - 1:
                self.energy += mod['energyCost']; self.data[m] = got_; self.status[m] = 'src'
                self.src_free[t].append(m); return True
            self.pools.put_back(got_); did = None
        if isinstance(did, tuple) and did and did[0] == 'multi':
            ids = self.take_multi(t, r['lowerBound'], r['upperBound'], did[1])
            if ids is None: return False
            self.energy += mod['energyCost']; self.data[m] = ids; self.status[m] = 'src'
            self.src_free[t].append(m); return True
        got = self.pools.take_spec(t, r['lowerBound'], r['upperBound'], did, BUCKET_W) if did is not None else None
        if got is not None: ids = [got]
        else:
            ids = self.pools.fill(t, r['lowerBound'], r['upperBound'], (1,)) if SRC_CR_FIRST else None
            if ids is None: ids = self.pools.fill(t, r['lowerBound'], r['upperBound'], (1, 0), pr=SRC_PR)
            if ids is None: return False
        self.energy += mod['energyCost']; self.data[m] = ids; self.status[m] = 'src'
        self.src_free[t].append(m); return True
    def try_model(self, m, mode, src_types=(), clone=False, fixed=None):
        mod = self.M[m]
        if mod['energyCost'] > self.eleft(): return False
        if clone:
            if m in self.clones: return False
            src_types = list(mod['dataRequirements'])
        elif m in self.status: return False
        reqs = mod['dataRequirements']
        c = Counter(src_types)
        for t, k in c.items():
            if len(self.src_free[t]) < k: return False
        got = []
        crs = (0,) if mode == 'full' else (1, 0)
        for t, r in reqs.items():
            if t in c: continue
            if fixed and t in fixed:
                got.append(list(fixed[t]) if isinstance(fixed[t], list) else [fixed[t]]); continue
            ids = self.pools.fill(t, r['lowerBound'], r['upperBound'], crs)
            if ids is None:
                fx = set(d for v in fixed.values() for d in (v if isinstance(v, list) else [v])) if fixed else set()
                self.pools.put_back([i for g in got for i in g if i not in fx]); return False
            got.append(ids)
        self.energy += mod['energyCost']
        if clone:
            key = m - self.N; self.clones.add(m)
        else:
            key = m; self.status[m] = mode; self.data[m] = [i for g in got for i in g]
        for t in src_types:
            self.mm.append((self.src_free[t].pop(), key))
        return True
    def take_multi(self, t, lb, ub, plan):
        """prend les datasets prévus par tranche, puis ajuste pour tomber dans [lb, ub]"""
        ids = []; S = 0
        for cnt, spec in sorted(plan, key=lambda p: -p[1][3]):
            for _ in range(cnt):
                d = self.pools.take_spec(t, 1, 10 ** 9, spec, BUCKET_W)
                if d is None: break
                ids.append(d); S += self.pools.D[d]['size']
        # retirer le superflu : le plus gros dataset dont on peut se passer
        while True:
            sz = {d: self.pools.D[d]['size'] for d in ids}
            cand = [d for d in ids if S - sz[d] >= lb]
            if not cand: break
            d = max(cand, key=lambda d: sz[d]); ids.remove(d); self.pools.put_back([d]); S -= sz[d]
        if lb <= S <= ub: return ids
        if S < lb:
            more = self.pools.fill(t, lb - S, ub - S, (1, 0))
            if more is not None: return ids + more
        # échec de l'ajustement : on rend tout et on remplit librement
        self.pools.put_back(ids)
        return self.pools.fill(t, lb, ub, (1, 0))
    def remove_unused_sources(self):
        for t in T:
            for s in self.src_free[t]:
                self.energy -= self.M[s]['energyCost']; self.pools.put_back(self.data.pop(s)); del self.status[s]
            self.src_free[t] = []
    def solution(self):
        return {"dataMappings": [[d, m] for m, ds in self.data.items() for d in ds],
                "modelMappings": [[a, b] for a, b in self.mm]}
    def value(self):
        tg = set(self.tg) if self.mm is None else {b for a, b in self.mm}
        v = 0
        for m, st in self.status.items():
            if st == 'src': continue
            mod = self.M[m]
            if st == 'full' and m not in tg: v += mod['value']
            else: v += mod['value'] // 2
        for m in self.clones: v += self.M[m]['value'] // 2
        return v

TIGHT = 5000
DPFILL = False  # remplissage exact par programmation dynamique pour les besoins >= DPMIN
DPMIN = 2500
DPK = 200  # pénalité par dataset (en unités) : préférer peu de datasets, garder les petits
DIAG = Counter()
TINY = 0
BUCKET_W = 0
SRC_CR_FIRST = False
FIN_TOL = 2      # gaspillage toléré sans chercher plus loin
WSRC_FIRST = False  # sources « en vrac » avant les modèles pleins
WSRC_DESC = False   # ... par minimum décroissant
SRC_PR = None  # prix utilisés pour remplir les sources (None = prix duaux)
TRIPLE_ITER = 0  # >0 : essaie aussi des triplets quand la fin gaspille
def plan_int(lst):
    """comptes continus par tranche -> comptes entiers (arrondi par excès des plus gros d'abord)"""
    lst = sorted(lst, key=lambda p: -p[0])
    out = [(int(v), sp_) for v, sp_ in lst if int(v) > 0]
    rest = sum(v - int(v) for v, _ in lst)
    for v, sp_ in lst:
        if rest <= 0.25: break
        if v - int(v) > 1e-6:
            out.append((1, sp_)); rest -= 1
    return out

def realize(sol, res, cols, verbose=True, frac=1.0):
    M = sol.M
    agg = defaultdict(float); ys = defaultdict(float); us = defaultdict(list); qs = defaultdict(list); ns = defaultdict(list)
    for key, v in zip(cols, res.x):
        if v < 1e-9: continue
        k, i, t = key
        if k in ('x', 'z', 'c'): agg[(k, i)] = v
        elif k == 'y': ys[(i, t)] = v
        elif k == 'u': agg[('s', i)] += v; us[i].append((v, t[1]))
        elif k == 'w': agg[('s', i)] += v
        elif k == 'v': agg[('s', i)] += v
        elif k == 'p': agg[('s', i)] += v; us[i].append((v, ('pair',) + tuple(t[1:])))
        elif k == 'n': ns[i].append((v, t[1]))
        elif k in ('qx', 'qz'): qs[(k, i, t[0])].append((v, t[1]))
    full, half, srcs, clones = [], [], [], []
    for m in M:
        i = m['id']
        xv, zv, sv = agg.get(('x', i), 0), agg.get(('z', i), 0), agg.get(('s', i), 0)
        best = max(xv, zv, sv)
        if best >= 0.5:
            if best == xv: full.append(i)
            elif best == zv:
                half.append((i, [t for t in m['dataRequirements'] if ys.get((i, t), 0) > 0.5 * zv]))
            else: srcs.append((i, ('multi', plan_int(ns[i])) if ns[i] else (max(us[i], key=lambda p: p[0])[1] if us[i] else None)))
        if agg.get(('c', i), 0) >= 0.5: clones.append(i)
    if verbose: print(f'  rounded full={len(full)} half={len(half)} src={len(srcs)} clone={len(clones)}')
    if frac < 1.0:
        dens = lambda i: M[i]['value'] / (M[i]['energyCost'] + 1e9)
        full = sorted(full, key=lambda i: -dens(i))[:max(1, int(len(full) * frac))]
        half = sorted(half, key=lambda p: -dens(p[0]))[:max(1, int(len(half) * frac))]
    fails = 0
    # reserve designated small-requirement datasets
    fixed = {}; reserved = set()
    for i in full:
        for t in M[i]['dataRequirements']:
            if qs.get(('qx', i, t)):
                r = M[i]['dataRequirements'][t]
                spec_ = max(qs[('qx', i, t)], key=lambda p: p[0])[1]
                did = sol.pools.take_spec(t, r['lowerBound'], r['upperBound'], spec_, BUCKET_W)
                DIAG['res_x', type(spec_).__name__ if not isinstance(spec_, tuple) else spec_[0], did is not None] += 1
                if did is not None: fixed.setdefault(i, {})[t] = did; reserved.update(did if isinstance(did, list) else [did])
    for i, st in half:
        for t in M[i]['dataRequirements']:
            if t in st: continue
            if qs.get(('qz', i, t)):
                r = M[i]['dataRequirements'][t]
                spec_ = max(qs[('qz', i, t)], key=lambda p: p[0])[1]
                did = sol.pools.take_spec(t, r['lowerBound'], r['upperBound'], spec_, BUCKET_W)
                DIAG['res_z', type(spec_).__name__ if not isinstance(spec_, tuple) else spec_[0], did is not None] += 1
                if did is not None: fixed.setdefault(i, {})[t] = did; reserved.update(did if isinstance(did, list) else [did])
    for i, did in srcs:
        if did is not None and not sol.try_src(i, did): fails += 1
    mub = lambda i: min(r['upperBound'] for r in M[i]['dataRequirements'].values())
    mlb = lambda i: min(r['lowerBound'] for r in M[i]['dataRequirements'].values())
    def okey(i):
        if TINY and mlb(i) < TINY: return (0, mlb(i))
        if mub(i) < TIGHT: return (1, mub(i))
        return (2, -M[i]['value'])
    _lbs = lambda p: next(iter(M[p[0]]['dataRequirements'].values()))['lowerBound']
    if WSRC_FIRST:
        for i, did in sorted(srcs, key=lambda p: -_lbs(p)):
            if did is None and not sol.try_src(i, did): fails += 1
    for i in sorted(full, key=okey):
        if not sol.try_model(i, 'full', fixed=fixed.get(i)): fails += 1
    if not WSRC_FIRST:
        for i, did in sorted(srcs, key=lambda p: -_lbs(p) if WSRC_DESC else _lbs(p)):
            if did is None and not sol.try_src(i, did): fails += 1
    for i, st in sorted(half, key=lambda p: (-len(p[1]), -M[p[0]]['value'])):
        if st and not sol.try_model(i, 'half', st, fixed=fixed.get(i)): fails += 1
    for i in sorted(clones, key=lambda i: -M[i]['value']):
        if not sol.try_model(i, 'half', clone=True): fails += 1
    for i, st in half:
        if not st and not sol.try_model(i, 'half', fixed=fixed.get(i)): fails += 1
    used = {d for ds in sol.data.values() for d in ds}
    sol.pools.put_back([d for d in reserved if d not in used])
    if verbose: print(f'  fails={fails} energy={sol.energy/sol.inst["energyCap"]:.5f} value={sol.value()}')

def greedy_fill(sol, clones=False):
    M = sol.M
    cand = [m for m in M if m['id'] not in sol.status]
    cand.sort(key=lambda m: -m['value'] / (m['energyCost'] + 1))
    added = 0
    for m in cand:
        if m['energyCost'] > sol.eleft() or m['value'] == 0: continue
        i = m['id']
        neg = [t for t, r in m['dataRequirements'].items() if r['upperBound'] < r['lowerBound']]
        if not neg and sol.try_model(i, 'full'): added += 1; continue
        if sol.try_model(i, 'half', neg): added += 1; continue
    if clones:
        for m in sorted(M, key=lambda m: -m['value'] / (m['energyCost'] + 1)):
            if m['id'] not in sol.clones and m['energyCost'] <= sol.eleft():
                if sol.try_model(m['id'], 'half', clone=True): added += 1
    return added

def run(f, iters=4, clones=False, K=20, out=None):
    txt = open(f).read(); inst = json.loads(txt)
    sol = Sol(inst)
    for it in range(iters):
        if it == 0:
            res, cols = build_and_solve(inst, K=K, clones=clones)
            for t in T:
                pf, pc = res.prices[t]; eps = max(pf, pc, 1e-9) * 1e-3
                sol.pools.pr[t] = {0: pf + eps, 1: pc + eps}
            print('prices', sol.pools.pr)
        else:
            res, cols = build_and_solve(inst, datasets=sol.pools.remaining(), energy=max(sol.eleft(), 1),
                                        exclude=set(sol.status), src_avail={t: len(sol.src_free[t]) for t in T},
                                        K=K, clones=clones, clone_exclude=sol.clones)
        v0 = sol.value()
        realize(sol, res, cols)
        if sol.value() == v0 and it > 0: break
    sol.remove_unused_sources()
    a = greedy_fill(sol, clones)
    print('greedy added', a, sol.value())
    s = sol.solution()
    sc = getSolutionScore(json.dumps(s), txt, show_details=False)
    print('FINAL', f, sc, flush=True)
    if out and sc[1]: json.dump(s, open(out, 'w'))
    return sol, sc

# ===== RECHERCHE LOCALE =====

def to_struct(sol):
    tg = defaultdict(list)
    for s, t in sol.mm: tg[t].append(s)
    sol.tg = tg
    sol.mm = None

def from_struct(sol):
    sol.mm = [(s, t) for t, ss in sol.tg.items() for s in ss]

def srctype(sol, s): return next(iter(sol.M[s]['dataRequirements']))

def remove(sol, key):
    m = key if key >= 0 else key + sol.N
    if key >= 0 and sol.status.get(m) == 'src':
        t = srctype(sol, m); sol.src_free[t].remove(m)
        ds = sol.data.pop(m); sol.pools.put_back(ds); del sol.status[m]
        sol.energy -= sol.M[m]['energyCost']
        return (key, 'src', ds, [])
    mod = sol.M[m]
    srcs = sol.tg.pop(key, [])
    for s in srcs: sol.src_free[srctype(sol, s)].append(s)
    if key >= 0:
        ds = sol.data.pop(m); sol.pools.put_back(ds); mode = sol.status.pop(m)
    else:
        ds = []; mode = 'clone'; sol.clones.discard(m)
    sol.energy -= mod['energyCost']
    return (key, mode, ds, srcs)

def readd(sol, rec):
    key, mode, ds, srcs = rec
    m = key if key >= 0 else key + sol.N
    if mode == 'src':
        for d in ds: assert sol.pools.take_id(d)
        sol.data[m] = list(ds); sol.status[m] = 'src'; sol.src_free[srctype(sol, m)].append(m)
        sol.energy += sol.M[m]['energyCost']; return
    for d in ds: assert sol.pools.take_id(d)
    for s in srcs: sol.src_free[srctype(sol, s)].remove(s)
    if srcs: sol.tg[key] = list(srcs)
    sol.energy += sol.M[m]['energyCost']
    if key >= 0: sol.data[m] = list(ds); sol.status[m] = mode
    else: sol.clones.add(m)

def add(sol, m, mode, src_types=(), clone=False):
    """wraps try_model with tg struct"""
    sol.mm = []
    ok = sol.try_model(m, mode, src_types, clone)
    if ok:
        key = m if not clone else m - sol.N
        if sol.mm: sol.tg[key] = [s for s, _ in sol.mm]
    sol.mm = None
    return ok

def val_of(sol, key):
    m = key if key >= 0 else key + sol.N
    v = sol.M[m]['value']
    if key >= 0 and sol.status[m] == 'full' and key not in sol.tg: return v
    return v // 2

def snapshot(sol):
    return ({k: list(v) for k, v in sol.data.items()}, dict(sol.status), {k: list(v) for k, v in sol.tg.items()},
            set(sol.clones), sol.energy, {t: list(v) for t, v in sol.src_free.items()})

def restore(sol, snap):
    data, status, tg, clones, energy, sf = snap
    pr = sol.pools.pr; sol.pools = Pools(sol.inst['datasets']); sol.pools.pr = pr
    for ds in data.values():
        for d in ds: assert sol.pools.take_id(d)
    sol.data = {k: list(v) for k, v in data.items()}; sol.status = dict(status)
    sol.tg = {k: list(v) for k, v in tg.items()}; sol.clones = set(clones); sol.energy = energy
    sol.src_free = {t: list(v) for t, v in sf.items()}

def src_cost(sol, m, lam):
    t, r = next(iter(m['dataRequirements'].items()))
    return lam * m['energyCost'] + sol.pools.pr[t][1] * r['lowerBound']

def local_search(sol, lam, tlimit=120, clones=False, seed=0, verbose=True, kmax=3, ncand=400, srcmode=True, alpha=0.0, save_cb=None, save_every=60):
    rnd = random.Random(seed)
    M = sol.M; pr = sol.pools.pr
    monos = {t: [] for t in T}
    for m in M:
        reqs = m['dataRequirements']
        if len(reqs) == 1:
            t, r = next(iter(reqs.items()))
            if r['upperBound'] >= r['lowerBound']: monos[t].append((src_cost(sol, m, lam), m['id']))
    for t in T: monos[t].sort()
    # typical source cost per type = 300th cheapest-ish
    scost = {t: (monos[t][min(len(monos[t]) - 1, max(0, sum(1 for st in sol.status.values() if st == 'src') // 4))][0]
                 if monos[t] else 1e18) for t in T}
    def plan(m):
        """returns (score_full, score_half, src_types_half)"""
        reqs = m['dataRequirements']
        neg = any(r['upperBound'] < r['lowerBound'] for r in reqs.values())
        base = lam * m['energyCost']
        sf = -1e18 if neg else m['value'] - base - sum(pr[t][0] * r['lowerBound'] for t, r in reqs.items())
        sh = m['value'] // 2 - base; st = []
        for t, r in reqs.items():
            dc = 1e18 if r['upperBound'] < r['lowerBound'] else pr[t][1] * r['lowerBound']
            sc = scost[t] if srcmode or r['upperBound'] < r['lowerBound'] else 1e18
            if sc < dc: st.append(t); sh -= sc
            else: sh -= dc
        return sf, sh, st
    plans = {m['id']: plan(m) for m in M}
    scores = sorted(((max(p[0], p[1]), i) for i, p in plans.items()), reverse=True)
    cscores = []
    if clones:
        cscores = sorted(((m['value'] // 2 - lam * m['energyCost'] - sum(scost[t] for t in m['dataRequirements']), m['id'])
                          for m in M), reverse=True)[:ncand]
    def get_sources(types, log):
        need = Counter(types)
        for t, k in need.items():
            k -= len(sol.src_free[t])
            if k <= 0: continue
            for c, s in monos[t]:
                if s in sol.status: continue
                if sol.try_src(s):
                    log.append(s); k -= 1
                    if k == 0: break
            if k > 0: return False
        return True
    def try_add(i, added):
        sf, sh, st = plans[i]
        m = M[i]
        if sf >= sh and add(sol, i, 'full'): added.append(i); return True
        log = []
        if st and not get_sources(st, log):
            for s in log: remove(sol, s)
            return False
        if add(sol, i, 'half', st): added.extend(log); added.append(i); return True
        for s in log: remove(sol, s)
        if sf < sh and plans[i][0] > -1e17 and add(sol, i, 'full'): added.append(i); return True
        return False
    def obj():
        v = sol.value()
        if alpha:
            v += alpha * (lam * sol.eleft() + sum(pr[t][cr] * sol.pools.left[(t, cr)] for t in T for cr in (0, 1)))
        return v
    t0 = time.time(); cur = obj(); it = 0; acc = 0; bestv = sol.value(); snap = snapshot(sol); last_save = time.time(); saved_v = bestv
    while time.time() - t0 < tlimit:
        it += 1
        keys = [k for k, st in sol.status.items() if st != 'src'] + [c - sol.N for c in sol.clones]
        if not keys: break
        k = rnd.randint(1, min(kmax, len(keys)))
        rem = rnd.sample(keys, k)
        recs = [remove(sol, key) for key in rem]
        for t in T:
            for s in list(sol.src_free[t]): recs.append(remove(sol, s))
        removed_ids = {r[0] if r[0] >= 0 else r[0] + sol.N for r in recs}
        added = []
        cands = []
        for sc, i in scores:
            if i not in sol.status: cands.append((sc * (1 + 0.05 * rnd.random()), i))
            if len(cands) >= ncand: break
        cands.sort(reverse=True)
        for sc, i in cands:
            if M[i]['energyCost'] > sol.eleft() or i in sol.status: continue
            if i in removed_ids and rnd.random() < 0.5: continue
            try_add(i, added)
        if clones:
            for sc, i in cscores:
                if i not in sol.clones and M[i]['energyCost'] <= sol.eleft():
                    log = []
                    ts = list(M[i]['dataRequirements'])
                    if get_sources(ts, log) and add(sol, i, 'half', clone=True):
                        added.extend(log); added.append(i - sol.N)
                    else:
                        for s in log: remove(sol, s)
        # drop unused sources
        for t in T:
            for s in list(sol.src_free[t]):
                if s in added: added.remove(s); remove(sol, s)
                else: recs.append(remove(sol, s))
        if save_cb and bestv > saved_v and time.time() - last_save > save_every and sol.value() == bestv:
            save_cb(sol); last_save = time.time(); saved_v = bestv
        new = obj()
        if new >= cur - 1e-9:
            cur = new; acc += 1
            v = sol.value()
            if v > bestv:
                bestv = v; snap = snapshot(sol)

                if verbose: print(f'  it {it} value {v} obj {new:.1f} ({time.time()-t0:.0f}s)', flush=True)
        else:
            for key in reversed(added): remove(sol, key)
            for r in reversed(recs): readd(sol, r)
    if verbose: print(f'  LS done it={it} acc={acc} value={sol.value()} best={bestv}')
    if sol.value() < bestv: restore(sol, snap)
    return cur

# ===== MIP RESTREINT =====
def core_fix(res, cols, ncore):
    x = res.x
    rc = np.abs(res.lower.marginals) + np.abs(res.upper.marginals)
    score = defaultdict(lambda: 1e18)
    for j, key in enumerate(cols):
        s = 0.0 if 1e-6 < x[j] < 1 - 1e-6 else rc[j]
        if key[0] in ('x', 'z', 'u', 'w', 'c'):
            score[key[1]] = min(score[key[1]], s)
    core = set(sorted(score, key=lambda i: score[i])[:ncore])
    return {key: float(round(x[j])) for j, key in enumerate(cols) if key[1] not in core}, core

def child(path, kw, ncore, mip_time, seed, resid):
    """exécuté dans un sous-processus ; renvoie seulement les variables non nulles"""
    inst = json.load(open(path))
    if resid:
        ds_ids, energy, exclude, src_avail = resid
        res, cols = build_and_solve(inst, datasets=[inst['datasets'][i] for i in ds_ids], energy=energy,
                                    exclude=set(exclude), src_avail=src_avail, **kw)
        out = dict(prices=None, lam=None)
    else:
        res, cols = build_and_solve(inst, **kw)
        out = dict(prices=res.prices, lam=-res.ineqlin.marginals[8] / inst['energyCap'])
        if mip_time:
            fix, core = core_fix(res, cols, ncore)
            print(f'coeur : {len(core)} modèles libres, {len(fix)} variables fixées', flush=True)
            del res
            res, cols = build_and_solve(inst, mip_time=mip_time, mip_seed=seed, fix=fix, **kw)
    nz = [(cols[j], float(v)) for j, v in enumerate(res.x) if v > 1e-9]
    out['keys'] = [k for k, _ in nz]; out['vals'] = [v for _, v in nz]
    return out

def in_child(*args):
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        return ex.submit(child, *args).result()

class _R: pass
RESID_MIP = 60  # MIP (au lieu d'un arrondi de LP) sur le problème résiduel après affectation

def run(f, ncore, mip_time, K, KQ, SMALL, tl, seed=1, lpx=None):
    global BUCKET_W
    if lpx and lpx.get('BUCKET'): BUCKET_W = lpx['BUCKET']
    txt = open(f).read(); inst = json.loads(txt)
    kw = dict(K=K, KQ=KQ, SMALL=SMALL, **(lpx or {}))
    sol = Sol(inst)
    if mip_time: print(f'MIP en cours (limite {mip_time:.0f}s)...', flush=True)
    out = in_child(f, kw, ncore, mip_time, seed, None)
    for t in T:
        pf, pc = out['prices'][t]; eps = max(pf, pc, 1e-9) * 1e-3
        sol.pools.pr[t] = {0: pf + eps, 1: pc + eps}
    lam = out['lam']
    r = _R(); r.x = out['vals']; realize(sol, r, out['keys'])
    for it in range(3):
        v0 = sol.value()
        import gc
        gc.collect()
        try:
            res, cols = build_and_solve(inst, datasets=sol.pools.remaining(), energy=max(sol.eleft(), 1),
                                        exclude=set(sol.status), src_avail={t: len(sol.src_free[t]) for t in T},
                                        **kw, **({'mip_time': RESID_MIP, 'mip_seed': seed} if RESID_MIP and mip_time else {}))
        except MemoryError:
            print('  (LP de complément abandonné : mémoire insuffisante, on passe au remplissage glouton)', flush=True)
            break
        realize(sol, res, cols); del res, cols; gc.collect()
        if sol.value() == v0: break
    sol.remove_unused_sources(); greedy_fill(sol)
    print('after greedy', sol.value(), flush=True)
    name = os.path.basename(f).replace('.json', '')
    def save_cb(sol):
        s = {"dataMappings": [[d, m] for m, ds in sol.data.items() for d in ds],
             "modelMappings": [[a, b] for b, ss in sol.tg.items() for a in ss]}
        st = json.dumps(s); sc = getSolutionScore(st, txt, show_details=False)
        if sc[1]:
            os.makedirs('out', exist_ok=True)
            open(f'out/{name}_legit_{sc[0]}.json', 'w').write(st)
            print(f'  SAUVEGARDE {sc[0]} -> out/{name}_legit_{sc[0]}.json', flush=True)
        return sc
    to_struct(sol); save_cb(sol)
    if tl > 0:
        local_search(sol, lam, tlimit=tl, seed=seed, srcmode=False, save_cb=save_cb)
    sc = save_cb(sol)
    print('FINAL', f, 'legit', sc, flush=True)

# ===== VALIDATEUR (copie de test_solution.py) =====
DATA_TYPE_LIST = 'ntic'
def getSolutionScore(solution_txt,dataset_txt, show_details = True):
	"""Evaluate the solution and return a tuple (score, is_valid, message).

	Arguments:
	solution_txt -- the solution to be evaluated (as a string)
	dataset_txt -- the dataset for which the solution is made (as a string)

	show_details -- an optional parameter to show some details on the submitted solution
	"""

	try:
		dataset = json.loads(dataset_txt)
	except:
		return 0, False, 'Error while processing the dataset. Please contact the contest organizer.'
	
	try:
		solution = json.loads(solution_txt)
	except:
		return 0, False, 'Submission is not a valid JSON object'

	for key in ('dataMappings','modelMappings'):
		if key not in solution:
			return 0, False, f'Submission does not contain the "{key}" key'
		if not isinstance(solution[key], list):
			return 0, False, f'"{key}" is not a list'
		for mapping in solution[key]:
			if not isinstance(mapping,list) or len(mapping)!=2 or not isinstance(mapping[0],int) or not isinstance(mapping[1],int):
				return 0, False, f'One item of "{key}" does not have the right format'

	model_used = set()
	model_used.update(model for _,model in solution["dataMappings"])
	model_used.update(model for models in solution["modelMappings"] for model in models)

	final_value = final_energy = 0

	model_fills = {i:[0]*4 for i in model_used}
	model_defiled ={i:0 for i in model_used}

	data_passed = set()

	for db,model in solution["dataMappings"]:
		if not 0<=db<len(dataset["datasets"]):
			return 0, False, f"Data {db} does not exist"
		if not 0<=model<len(dataset["models"]):
			return 0, False, f"Model {model} does not exist"

		if db in data_passed:
			return 0, False, f"Data {db} has been assigned more than once"
		data_passed.add(db)

		data_type = dataset["datasets"][db]["dataType"]

		if data_type not in dataset["models"][model]["dataRequirements"]:
			return 0, False, f"Model {model} does not accept the type of dataset {db} ({data_type})"

		data_type_to_number = DATA_TYPE_LIST.find(dataset["datasets"][db]["dataType"])
		model_fills[model][data_type_to_number]+=dataset["datasets"][db]["size"]
		model_defiled[model]|=dataset["datasets"][db]["isCopyrighted"]

	model_used_as_source = set()

	for model_source, model_target in solution["modelMappings"]:
		if model_source in model_used_as_source:
			return 0, False, f"Model {model_source} has already been used as a data source"
		model_used_as_source.add(model_source)
		if len(dataset["models"][model_source]["dataRequirements"])!=1:
			return 0, False, f"Model {model_source}, used as a data source, needs more than one data type"
		data_type = next(iter(dataset["models"][model_source]["dataRequirements"]))
		data_type_to_number = DATA_TYPE_LIST.find(data_type)

		data_req = dataset["models"][model_source]["dataRequirements"][data_type]
		if not data_req["lowerBound"]<=model_fills[model_source][data_type_to_number]<=data_req["upperBound"]:
			return 0, False, f"The data ({data_type}) requirements of model {model_source} used as data source are not fulfilled"

		model_defiled[model_target]=1
		model_fills[model_target][data_type_to_number]=-1

		final_energy += dataset["models"][model_source]["energyCost"]


	for model_id in model_used:
		if model_id in model_used_as_source:
			continue
		model=dataset["models"][model_id]
		for data_type,data_req in model["dataRequirements"].items():
			val = model_fills[model_id][DATA_TYPE_LIST.find(data_type)]
			if val == -1:
				continue
			if val<data_req["lowerBound"]:
				return 0, False, f"Model {model_id} has too few data of type {data_type}"
			if val>data_req["upperBound"]:
				return 0, False, f"Model {model_id} has too many data of type {data_type}"

		if model_defiled[model_id]:
			final_value += model["value"]//2
		else:
			final_value += model["value"]

		final_energy += model["energyCost"]

	if final_energy>dataset["energyCap"]:
		return 0, False, f"The total energy is too high ({final_energy}/{dataset['energyCap']})"

	return final_value, True, ""

# ===== LNS PAR MIP AVEC DÉMARRAGE À CHAUD (highspy) =====
class _R: pass
def make_highs(c, A, b, Aeq, beq):
    import highspy
    M = sp.vstack([A, Aeq]).tocsc() if Aeq is not None else A.tocsc()
    rl = np.concatenate([np.full(A.shape[0], -highspy.kHighsInf), beq if Aeq is not None else []])
    ru = np.concatenate([b, beq if Aeq is not None else []])
    lp = highspy.HighsLp()
    n = M.shape[1]
    lp.num_col_ = n; lp.num_row_ = M.shape[0]
    lp.col_cost_ = c; lp.col_lower_ = np.zeros(n); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = rl; lp.row_upper_ = ru
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr; lp.a_matrix_.index_ = M.indices; lp.a_matrix_.value_ = M.data
    h = highspy.Highs(); h.setOptionValue('output_flag', False)
    h.setOptionValue('mip_rel_gap', 0.0); h.setOptionValue('mip_abs_gap', 0.99)
    h.passModel(lp)
    return h, A.shape[0]

def set_bounds(h, lo, hi):
    import highspy
    n = len(lo); idx = np.arange(n, dtype=np.int32)
    h.changeColsBounds(n, idx, lo, hi)

def realize_x(inst, prices, x, cols, kw, seed):
    sol = Sol(inst)
    for t in T:
        pf, pc = prices[t]; eps = max(pf, pc, 1e-9) * 1e-3
        sol.pools.pr[t] = {0: pf + eps, 1: pc + eps}
    r = _R(); r.x = x; realize(sol, r, cols, verbose=False)
    for it in range(3):
        v0 = sol.value()
        try:
            res, cc = build_and_solve(inst, datasets=sol.pools.remaining(), energy=max(sol.eleft(), 1),
                                      exclude=set(sol.status), src_avail={t: len(sol.src_free[t]) for t in T},
                                      mip_time=60, mip_seed=seed, verbose=False, **kw)
        except MemoryError:
            break
        realize(sol, res, cc, verbose=False); del res, cc; gc.collect()
        if sol.value() == v0: break
    sol.remove_unused_sources(); greedy_fill(sol)
    return sol

def lns_run(f, kw, ncore0, t0_mip, ncore, t_mip, duree, seed):
    import highspy
    rnd = random.Random(seed)
    global BUCKET_W
    if kw.get('BUCKET'): BUCKET_W = kw['BUCKET']
    txt = open(f).read(); inst = json.loads(txt); name = os.path.basename(f).replace('.json', '')
    c, A, b, Aeq, beq, cols, _ubv = build_and_solve(inst, return_model=True, **kw)
    n = len(cols); nA = A.shape[0]
    h, nA = make_highs(c, A, b, Aeq, beq); del A, Aeq; gc.collect()
    # 1) relaxation LP : prix duaux + coûts réduits
    integ = [highspy.HighsVarType.kContinuous] * n
    t = time.time(); h.run(); s = h.getSolution()
    xl = np.array(s.col_value); du = np.array(s.row_dual); rc = np.abs(np.array(s.col_dual))
    print(f'LP {-h.getInfo().objective_function_value:.1f} ({time.time()-t:.0f}s)', flush=True)
    prices = {tt: (-du[k] - du[4 + k], -du[4 + k]) for k, tt in enumerate(T)}
    owner = np.array([key[1] for key in cols]); kinds = [key[0] for key in cols]
    score = defaultdict(lambda: 1e18)
    for j in range(n):
        if kinds[j] in ('x', 'z', 'u', 'w'):
            sc = 0.0 if 1e-6 < xl[j] < 1 - 1e-6 else rc[j]
            if sc < score[owner[j]]: score[owner[j]] = sc
    order = sorted(score, key=lambda i: score[i])
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * n))
    def solve_core(core, xstar, tl, warm=True):
        free = np.isin(owner, np.fromiter(core, dtype=owner.dtype))
        lo = np.where(free, 0.0, xstar); hi = np.where(free, 1.0, xstar)
        set_bounds(h, lo, hi)
        h.setOptionValue('time_limit', float(h.getRunTime() + tl))
        if xstar is not None and warm:
            hs = highspy.HighsSolution(); hs.col_value = list(xstar); hs.value_valid = True
            h.setSolution(hs)
        with silence():
            h.run()
        x = np.round(np.array(h.getSolution().col_value))
        if False: print('   ', h.modelStatusToString(h.getModelStatus()), f'{h.getInfo().mip_gap:.2e}', f'{-c @ x:.0f}', flush=True)
        if h.getInfo().primal_solution_status != 2:
            return (xstar, float(-c @ xstar)) if warm else (None, None)
        return x, float(-c @ x)
    # 2) solution de départ : coeur marginal
    x0 = np.round(xl)
    print(f'MIP en cours (limite {t0_mip:.0f}s)...', flush=True)
    xstar = None
    for essai in range(4):  # le MIP de départ doit trouver une solution entière (sinon on prolonge)
        xstar, v = solve_core(set(order[:ncore0]), x0, t0_mip * (essai + 1), warm=False)
        if xstar is not None: break
        print(f'  pas encore de solution entière, prolongation du MIP de départ...', flush=True)
    if xstar is None:
        print('ÉCHEC : le MIP de départ n\'a trouvé aucune solution (augmenter --mip)', flush=True); return
    # vérif faisabilité de base
    print(f'MIP st=1 départ val={v:.0f}', flush=True)
    best = 0; bestfile = None
    def eval_and_save(x):
        nonlocal best
        sol = realize_x(inst, prices, x, cols, kw, seed)
        st = json.dumps(sol.solution()); sc = getSolutionScore(st, txt, show_details=False)
        if sc[1] and sc[0] > best:
            best = sc[0]; os.makedirs('out', exist_ok=True)
            open(f'out/{name}_legit_{sc[0]}.json', 'w').write(st)
            print(f'  SAUVEGARDE {sc[0]}', flush=True)
        return sc[0]
    r = eval_and_save(xstar); print(f'after greedy {r}', flush=True)
    vstar = v; tstart = time.time(); it = 0
    marg = order[:4 * ncore]
    while time.time() - tstart < duree:
        it += 1
        core = set(rnd.sample(marg, ncore // 2)) | set(rnd.sample(range(len(inst['models'])), ncore // 2))
        x, v = solve_core(core, xstar, t_mip)
        if v > vstar + 0.5:
            xstar, vstar = x, v
            r = eval_and_save(x)
            print(f'  it {it} value {best} (modèle {v:.0f}, réalisé {r})', flush=True)
        elif it % 10 == 0:
            print(f'LNS {it}: pas d\'amélioration (modèle {vstar:.0f})', flush=True)
    print(f"FINAL {f} legit ({best}, True, '')", flush=True)

# ===== EXPORT / IMPORT POUR UN SOLVEUR EXTERNE (Gurobi, CPLEX, ...) =====
def exporter_mps(f, kw):
    import highspy
    inst = json.load(open(f)); name = nom(f)
    c, A, b, Aeq, beq, cols, _ubv = build_and_solve(inst, return_model=True, **kw)
    h, _ = make_highs(c, A, b, Aeq, beq); n = len(cols)
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * n))
    h.writeModel(f'{name}.mps')
    print(f'Modèle écrit : {name}.mps ({n} variables binaires, minimisation de -valeur).', flush=True)
    print(f'Exemple Gurobi : gurobi_cl TimeLimit=3600 ResultFile={name}.sol {name}.mps', flush=True)
    print(f'Puis : python solveur.py --instance {f} --importer {name}.sol', flush=True)

def importer_sol(f, kw, solfile):
    global BUCKET_W
    if kw.get('BUCKET'): BUCKET_W = kw['BUCKET']
    txt = open(f).read(); inst = json.loads(txt); name = nom(f)
    res, cols = build_and_solve(inst, verbose=False, **kw)
    prices = res.prices; del res
    x = np.zeros(len(cols))
    for line in open(solfile):
        p = line.split()
        if len(p) >= 2 and p[0].startswith('c') and p[0][1:].isdigit():
            try: x[int(p[0][1:])] = float(p[1])
            except ValueError: pass
    print(f'solution importée : valeur modèle {sum(cols_val(cols, inst, x)):.0f}', flush=True)
    sol = realize_x(inst, prices, np.round(x), cols, kw, 1)
    st = json.dumps(sol.solution()); sc = getSolutionScore(st, txt, show_details=False)
    print('score réel', sc, flush=True)
    if sc[1]:
        os.makedirs('out', exist_ok=True); open(os.path.join('out', f'{name}_legit_{sc[0]}.json'), 'w').write(st)
        print(f'-> out/{name}_legit_{sc[0]}.json', flush=True)

def cols_val(cols, inst, x):
    M = inst['models']
    for (k, i, _), v in zip(cols, x):
        if v > 0.5 and k == 'x': yield M[i]['value']
        elif v > 0.5 and k == 'z': yield M[i]['value'] // 2

# ===== RÉGLAGES PAR INSTANCE =====
REGLAGES = {
    '1_example':  dict(ncore=10**9, K=20, KQ=20, SMALL=0, wadd=0, lpx={}),
    '2_medium':   dict(ncore=10**9, K=20, KQ=40, SMALL=2500, wadd=0, lpx={}),
    '3_free':     dict(ncore=10**9, K=20, KQ=20, SMALL=0, wadd=0, lpx={}),
    '4_precise':  dict(ncore=1500, K=20, KQ=20, SMALL=0, wadd=0, lpx={}),
    '5_energy':   dict(ncore=3000, K=20, KQ=20, SMALL=10000, wadd=0,
                       lpx=dict(BUCKET=20, NB=50, FLUIDMIN=2500, FDELTA=0.005)),
    '6_shortage': dict(ncore=2000, K=20, KQ=30, SMALL=10000, wadd=0,
                       lpx=dict(FLUIDMIN=2500, FDELTA=0.001, BUCKET=20, NB=60)),
    '7_big':      dict(ncore=3000, K=8, KQ=8, SMALL=0, wadd=70, lpx={}),
}
DEFAUT = dict(ncore=2000, K=20, KQ=20, SMALL=0, wadd=0, lpx={})

def nom(f): return os.path.basename(f).replace('.json', '')

def worker(a):
    R = dict(REGLAGES.get(nom(a.instance), DEFAUT))
    if a.ncore: R['ncore'] = a.ncore
    lpx = dict(R['lpx'])
    if R['wadd']: lpx['WADD'] = R['wadd']
    if a.lns:
        try:
            import highspy  # noqa
        except ImportError:
            print('ÉCHEC : --lns demande highspy (pip install highspy)', flush=True); return
        kw = dict(K=R['K'], KQ=R['KQ'], SMALL=R['SMALL'], **lpx)
        lns_run(a.instance, kw, R['ncore'], a.mip, a.lns_core or max(300, R['ncore'] // 2), a.lns_mip, a.lns, a.seed)
    else:
        run(a.instance, R['ncore'], a.mip, R['K'], R['KQ'], R['SMALL'], a.rl, seed=a.seed, lpx=lpx or None)

def avancement(path):
    best = 0; phase = 'LP'
    try:
        for l in open(path, encoding='utf-8', errors='replace'):
            m = None
            if l.startswith('  it'): m = re.search(r'value (\d+)', l)
            elif 'SAUVEGARDE' in l: m = re.search(r'SAUVEGARDE (\d+)', l)
            elif l.startswith('FINAL'): m = re.search(r'\((\d+), True', l); phase = 'fini'
            elif l.startswith('after greedy'): m = re.search(r'after greedy (\d+)', l); phase = 'RL'
            elif l.startswith('MIP en cours'): phase = 'MIP'
            elif l.startswith('MIP st') and phase == 'MIP': phase = 'affectation'
            if m: best = max(best, int(m.group(1)))
    except FileNotFoundError:
        pass
    return best, phase

def meilleur_fichier(name):
    sc = [(int(re.findall(r'_(\d+)\.json$', p)[0]), p) for p in glob.glob(os.path.join('out', f'{name}_legit_*.json'))]
    return max(sc) if sc else (0, None)

def lanceur(a):
    if not os.path.exists(a.instance):
        print(f'Instance introuvable : {a.instance} (lancer le script dans le dossier des instances)'); sys.exit(1)
    name = nom(a.instance); txt = open(a.instance).read()
    os.makedirs('logs', exist_ok=True); os.makedirs('out', exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    print(f'{name} : {a.procs} processus en parallèle, MIP {a.mip:.0f}s, ' + (f'LNS {a.lns:.0f}s, ' if a.lns else f'recherche locale {a.rl:.0f}s, ') +
          f'{"runs illimités (Ctrl+C pour arrêter)" if not a.runs else f"{a.runs} runs"}', flush=True)
    seed = a.premiere_graine; lances = 0; running = []; t0 = time.time(); last = 0; last_launch = -1e9
    best_publie = 0
    def publier():
        nonlocal best_publie
        v, p = meilleur_fichier(name)
        if v > best_publie and p:
            st = open(p).read(); sc = getSolutionScore(st, txt, show_details=False)
            neg = any(x < 0 or y < 0 for x, y in json.loads(st)['modelMappings'])
            if sc[1] and not neg:
                shutil.copy(p, f'{name}_MEILLEUR.json'); best_publie = sc[0]
                print(f'[{time.time()-t0:7.0f}s] *** NOUVEAU MEILLEUR {sc[0]} -> {name}_MEILLEUR.json', flush=True)
    try:
        while True:
            while len(running) < a.procs and (not a.runs or lances < a.runs) and time.time() - last_launch >= a.decalage:
                cmd = [sys.executable, '-u', os.path.abspath(__file__), '--worker', '--instance', a.instance,
                       '--mip', str(a.mip), '--rl', str(a.rl), '--seed', str(seed),
                       '--lns', str(a.lns), '--lns-mip', str(a.lns_mip), '--lns-core', str(a.lns_core)] + (['--ncore', str(a.ncore)] if a.ncore else [])
                log = open(os.path.join('logs', f'{name}_graine{seed}.txt'), 'w')
                p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append((seed, p, log)); last_launch = time.time(); lances += 1
                print(f'[{time.time()-t0:7.0f}s] lancé graine {seed} ({len(running)} en cours)', flush=True)
                seed += 1
            for r in running[:]:
                if r[1].poll() is not None:
                    r[2].close(); running.remove(r)
                    lines = [l.strip() for l in open(os.path.join('logs', f'{name}_graine{r[0]}.txt'), encoding='utf-8', errors='replace') if l.strip()]
                    fin = [l for l in lines if l.startswith('FINAL')]
                    msg = fin[-1] if fin else 'ÉCHEC : ' + (lines[-1] if lines else 'log vide')
                    if not fin and lines and 'emory' in lines[-1]: msg += '  -> mémoire insuffisante : réduire --procs ou augmenter --decalage'
                    print(f'[{time.time()-t0:7.0f}s] fini graine {r[0]} : {msg}', flush=True)
                    publier()
            if time.time() - last > 60:
                last = time.time(); publier()
                det = ' '.join(f'g{s}:{b}[{ph}]' for s, (b, ph) in ((s, avancement(os.path.join('logs', f'{name}_graine{s}.txt'))) for s, _, _ in running))
                print(f'[{time.time()-t0:7.0f}s] MEILLEUR : {best_publie} | en cours : {det}', flush=True)
            if a.runs and lances >= a.runs and not running: break
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nArrêt demandé : on termine les processus en cours...', flush=True)
        for _, p, _ in running: p.terminate()
    publier()
    print(f'Meilleur score : {best_publie}  (fichier {name}_MEILLEUR.json)', flush=True)

def main():
    ap = argparse.ArgumentParser(description='Solveur IA responsables (solutions légales)')
    ap.add_argument('--instance', default='6_shortage.json')
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) // 2), help='runs en parallèle')
    ap.add_argument('--mip', type=float, default=600, help='limite du MIP restreint (s)')
    ap.add_argument('--rl', type=float, default=60, help='durée de la recherche locale finale (s)')
    ap.add_argument('--runs', type=int, default=0, help='nombre total de runs (0 = jusqu\'à Ctrl+C)')
    ap.add_argument('--decalage', type=float, default=60, help='secondes entre deux lancements (pic de RAM du LP)')
    ap.add_argument('--ncore', type=int, default=0, help='modèles laissés libres dans le MIP (0 = réglage par instance)')
    ap.add_argument('--lns', type=float, default=0, help='durée (s) de la boucle LNS après le MIP de départ (demande highspy)')
    ap.add_argument('--lns-mip', dest='lns_mip', type=float, default=60, help='limite de chaque MIP de la boucle LNS (s)')
    ap.add_argument('--lns-core', dest='lns_core', type=int, default=0, help='modèles libérés à chaque tour LNS (0 = auto)')
    ap.add_argument('--exporter-mps', dest='exporter_mps', action='store_true', help='écrit le modèle .mps pour un solveur externe')
    ap.add_argument('--importer', default='', help='fichier .sol (Gurobi/HiGHS) à transformer en solution')
    ap.add_argument('--premiere-graine', dest='premiere_graine', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.exporter_mps or a.importer:
        R = dict(REGLAGES.get(nom(a.instance), DEFAUT)); kw = dict(K=R['K'], KQ=R['KQ'], SMALL=R['SMALL'], **R['lpx'])
        if R['wadd']: kw['WADD'] = R['wadd']
        if a.exporter_mps: exporter_mps(a.instance, kw)
        if a.importer: importer_sol(a.instance, kw, a.importer)
    elif a.worker: worker(a)
    else: lanceur(a)

KW = dict(K=20, KQ=30, SMALL=10000, FLUIDMIN=2500, FDELTA=0.005, BUCKET=50, NB=40, QPAIRS=20, QPAIRMIN=50, PAIRS=30)

def mip_gurobi(c, A, b, Aeq, beq, ub, tl, threads):
    import gurobipy as gp
    m = gp.Model(); m.Params.OutputFlag = 1; m.Params.TimeLimit = tl; m.Params.MIPGap = 0.0
    if threads: m.Params.Threads = threads
    x = m.addMVar(len(c), lb=0.0, ub=ub, vtype=gp.GRB.BINARY)
    m.addMConstr(A.tocsr(), x, '<', b)
    if Aeq is not None: m.addMConstr(Aeq.tocsr(), x, '=', beq)
    m.setObjective(c @ x, gp.GRB.MINIMIZE); m.optimize()
    if m.SolCount == 0: return None
    return np.array(x.X)

def mip_highs(c, A, b, Aeq, beq, ub, tl):
    from scipy.optimize import milp, LinearConstraint, Bounds
    cons = [LinearConstraint(A, -np.inf, b)]
    if Aeq is not None: cons.append(LinearConstraint(Aeq, beq, beq))
    r = milp(c, integrality=np.ones(len(c)), bounds=Bounds(np.zeros(len(c)), ub), constraints=cons,
             options={'time_limit': tl, 'mip_rel_gap': 1e-6, 'disp': True})
    return r.x

def main_six():
    global BUCKET_W, DPFILL
    sys.argv = [sys.argv[0]] + [x for x in sys.argv[1:] if x != '--six']
    ap = argparse.ArgumentParser(description='Solveur clé en main, instance 6')
    ap.add_argument('--instance', default='6_shortage.json')
    ap.add_argument('--temps', type=float, default=1800, help='limite du MIP principal (s)')
    ap.add_argument('--solveur', default='auto', choices=['auto', 'gurobi', 'highs'])
    ap.add_argument('--rc', type=float, default=0.5, help='seuil de coût réduit pour garder une colonne (plus grand = MIP plus gros)')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--depart', default='', help='solution existante : on n\'écrit que si on fait mieux')
    a = ap.parse_args()
    t0 = time.time(); BUCKET_W = KW['BUCKET']; DPFILL = True
    txt = open(a.instance).read(); inst = json.loads(txt); name = os.path.basename(a.instance).replace('.json', '')
    best0 = 0
    if a.depart:
        best0 = getSolutionScore(open(a.depart).read(), txt, show_details=False)[0]; print('solution de départ :', best0, flush=True)
    solv = a.solveur
    if solv == 'auto':
        try:
            import gurobipy as gp; gp.Model().dispose(); solv = 'gurobi'
        except Exception:
            solv = 'highs'
    import scipy, platform
    print(f'solveur MIP : {solv} | Python {platform.python_version()} | scipy {scipy.__version__}', flush=True)
    # 1) LP complet -> colonnes utiles (calculé une seule fois, puis relu depuis le cache)
    import pickle
    cache = f'cache_lp_{name}.pkl'; lock = cache + '.verrou'
    data_ = None
    while data_ is None:
        if os.path.exists(cache + '.gz') and not os.path.exists(cache):
            import gzip
            with gzip.open(cache + '.gz', 'rb') as fh: data_ = pickle.load(fh)
            print(f'LP relu depuis le cache fourni {cache}.gz ({time.time()-t0:.0f}s)', flush=True); break
        if os.path.exists(cache):
            with open(cache, 'rb') as fh: data_ = pickle.load(fh)
            print(f'LP relu depuis le cache {cache} ({time.time()-t0:.0f}s)', flush=True); break
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError:
            if time.time() - os.path.getmtime(lock) > 4 * 3600: os.remove(lock); continue
            print(f'  un autre calcul prépare le LP, attente... ({time.time()-t0:.0f}s)', flush=True); time.sleep(30); continue
        try:
            print(f'construction et résolution du LP (une seule fois pour tous les calculs)...', flush=True)
            res, cols = build_and_solve(inst, **KW)
            data_ = dict(cols=cols, x=np.array(res.x), rc=np.abs(res.lower.marginals) + np.abs(res.upper.marginals),
                         prices=res.prices, fun=float(res.fun))
            del res
            with open(cache + '.tmp', 'wb') as fh: pickle.dump(data_, fh, protocol=4)
            os.replace(cache + '.tmp', cache)
        finally:
            try: os.remove(lock)
            except OSError: pass
    class _Res: pass
    res = _Res(); res.fun = data_['fun']; cols = data_['cols']
    x = data_['x']; rc = data_['rc']; prices = data_['prices']
    keep_model = {key[1] for j, key in enumerate(cols) if x[j] > 1e-6 or rc[j] <= a.rc}
    fix = {}
    for j, key in enumerate(cols):
        if key[1] not in keep_model: fix[key] = 0.0
        elif key[0] not in ('x', 'z', 'y') and not (x[j] > 1e-6 or rc[j] <= a.rc): fix[key] = 0.0
    print(f'borne LP {-res.fun:.0f}, colonnes libres {len(cols) - len(fix)} ({time.time()-t0:.0f}s)', flush=True)
    del res, x, rc, data_; gc.collect()
    # 2) MIP
    c, A, b, Aeq, beq, cols, ubv = build_and_solve(inst, return_model=True, verbose=False, **KW)
    ubv = ubv.astype(float).copy()
    for j, key in enumerate(cols):
        if key in fix: ubv[j] = 0.0
    xm = None
    if solv == 'gurobi':
        try: xm = mip_gurobi(c, A, b, Aeq, beq, ubv, a.temps, a.threads)
        except Exception as e:
            print('Gurobi indisponible pour ce modèle (licence ?) :', str(e)[:120], '-> HiGHS', flush=True); solv = 'highs'
    if solv == 'highs': xm = mip_highs(c, A, b, Aeq, beq, ubv, a.temps)
    if xm is None: print('ÉCHEC : aucune solution entière (augmenter --temps)'); return
    print(f'MIP : valeur du plan {float(-c @ np.round(xm)):.0f} ({time.time()-t0:.0f}s)', flush=True)
    del A, Aeq; gc.collect()
    # 3) affectation exacte
    sol = Sol(inst)
    for t in T:
        pf, pc = prices[t]; eps = max(pf, pc, 1e-9) * 1e-3
        sol.pools.pr[t] = {0: pf + eps, 1: pc + eps}
    class R: pass
    r = R(); r.x = np.round(xm); realize(sol, r, cols)
    for it in range(3):
        vb = sol.value()
        r3, c3 = build_and_solve(inst, datasets=sol.pools.remaining(), energy=max(sol.eleft(), 1), exclude=set(sol.status),
                                 src_avail={t: len(sol.src_free[t]) for t in T}, mip_time=120, **KW)
        realize(sol, r3, c3); del r3, c3
        if sol.value() == vb: break
    sol.remove_unused_sources(); greedy_fill(sol)
    s = sol.solution(); st = json.dumps(s); sc = getSolutionScore(st, txt, show_details=False)
    print(f'RÉSULTAT {sc} ({time.time()-t0:.0f}s)', flush=True)
    if sc[1]:
        os.makedirs('out', exist_ok=True); open(os.path.join('out', f'{name}_legit_{sc[0]}.json'), 'w').write(st)
    print(f"FINAL {a.instance} legit {sc}", flush=True)

# ===== RECHERCHE PAR COLONNES EXACTES DEPUIS LA MEILLEURE SOLUTION (cg6c) =====
CG_SRC = '"""Génération de colonnes sur mesure (instance 6).\nChaque colonne « motif » = liste EXACTE de datasets pour un besoin (modèle, type, mode) ou pour une source.\nMaître : choix des modèles (plein/demi), types alimentés par source, motifs, sources ; chaque dataset au plus une fois.\nDémarre de la meilleure solution connue (ses motifs sont dans le maître) : le MIP ne peut pas faire pire,\net la valeur du MIP = valeur réelle (aucune étape d\'affectation)."""\nimport json, sys, os, time, random\nimport numpy as np, scipy.sparse as sp, highspy\nfrom collections import defaultdict\n\nT = \'ntic\'\nf, solf = CG_ARGS[0], CG_ARGS[1]\nITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6\nMIPT = float(sys.argv[4]) if len(sys.argv) > 4 else 120\ntxt = open(f).read(); inst = json.loads(txt); M = inst[\'models\']; D = inst[\'datasets\']; cap = inst[\'energyCap\']\nN = len(M); name = os.path.basename(f).replace(\'.json\', \'\')\nsize = np.array([d[\'size\'] for d in D]); dtype = [d[\'dataType\'] for d in D]; dcr = np.array([d[\'isCopyrighted\'] for d in D])\nbyt = {t: [d[\'id\'] for d in D if d[\'dataType\'] == t] for t in T}\ndef neg(r): return r[\'upperBound\'] < r[\'lowerBound\']\n\n# ---------- lignes ----------\nrows_lo, rows_hi = [], []\ndef newrow(lo, hi): rows_lo.append(lo); rows_hi.append(hi); return len(rows_lo) - 1\nR_E = newrow(-np.inf, 1.0)\nR_D = {d: newrow(-np.inf, 1.0) for d in range(len(D))}\nR_TOK = {t: newrow(-np.inf, 0.0) for t in T}\nR_EX = {}; R_COV = {}\nfor m in M:\n    i = m[\'id\']; R_EX[i] = newrow(-np.inf, 1.0)\n    for t, r in m[\'dataRequirements\'].items():\n        if not neg(r): R_COV[(i, t, \'f\')] = newrow(0.0, 0.0)\n        R_COV[(i, t, \'h\')] = newrow(0.0, 0.0)\n# ---------- colonnes ----------\ncols = []  # (clé, coût, [(ligne, coef)])\ncolset = set()\ndef addcol(key, cost, entries):\n    if key in colset: return None\n    colset.add(key); cols.append((key, cost, entries)); return len(cols) - 1\nfor m in M:\n    i = m[\'id\']; E = m[\'energyCost\'] / cap; reqs = m[\'dataRequirements\']\n    if not any(neg(r) for r in reqs.values()):\n        addcol((\'x\', i), -m[\'value\'], [(R_E, E), (R_EX[i], 1)] + [(R_COV[(i, t, \'f\')], -1) for t in reqs])\n    addcol((\'z\', i), -(m[\'value\'] // 2), [(R_E, E), (R_EX[i], 1)] + [(R_COV[(i, t, \'h\')], -1) for t in reqs])\n    for t in reqs: addcol((\'y\', i, t), 0.0, [(R_COV[(i, t, \'h\')], 1), (R_TOK[t], 1)])\ndef add_pattern(i, t, mode, ds):\n    ds = tuple(sorted(ds))\n    return addcol((\'p\', i, t, mode, ds), 0.0, [(R_COV[(i, t, mode)], 1)] + [(R_D[d], 1) for d in ds])\ndef add_source(s, ds):\n    ds = tuple(sorted(ds)); t = next(iter(M[s][\'dataRequirements\']))\n    return addcol((\'s\', s, ds), 0.0, [(R_E, M[s][\'energyCost\'] / cap), (R_EX[s], 1), (R_TOK[t], -1)] + [(R_D[d], 1) for d in ds])\n\n# ---------- solution de départ ----------\nsol0 = json.load(open(solf))\nsrcs = {a for a, b in sol0[\'modelMappings\']}; tg = defaultdict(set)\nfor a, b in sol0[\'modelMappings\']: tg[b].add(next(iter(M[a][\'dataRequirements\'])))\ndata = defaultdict(list)\nfor d, m in sol0[\'dataMappings\']: data[m].append(d)\nstart = {}\nfor m in set(data) | set(tg) | srcs:\n    if m in srcs:\n        j = add_source(m, data[m]); start[cols[j][0]] = 1; continue\n    defiled = bool(tg.get(m)) or any(dcr[d] for d in data[m]); mode = \'h\' if defiled else \'f\'\n    start[(\'z\', m) if defiled else (\'x\', m)] = 1\n    for t in M[m][\'dataRequirements\']:\n        if t in tg.get(m, ()): start[(\'y\', m, t)] = 1\n        else:\n            j = add_pattern(m, t, mode, [d for d in data[m] if dtype[d] == t]); start[cols[j][0]] = 1\nv_start = -sum(c for k, c, _ in cols if k in start)\nprint(f\'départ {v_start:.0f}, colonnes {len(cols)}\', flush=True)\n\n# ---------- génération de motifs (heuristique) ----------\ndef best_pattern(t, lb, ub, w, freeonly, rnd):\n    cand = [d for d in byt[t] if not (freeonly and dcr[d])]\n    if not cand: return None, None\n    best = (None, np.inf)\n    # 1 dataset\n    for d in cand:\n        if lb <= size[d] <= ub and w[d] < best[1]: best = ([d], w[d])\n    # glouton par coût/taille (+ bruit), puis ajustement exact au plus juste\n    for rep in range(3):\n        noise = 1 + 0.05 * rep\n        order = sorted(cand, key=lambda d: (w[d] + 1e-9) / size[d] * (1 + (noise - 1) * rnd.random()))\n        S = 0; ch = []; cost = 0.0\n        for d in order:\n            if S >= lb: break\n            if S + size[d] <= ub: ch.append(d); S += size[d]; cost += w[d]\n        if S < lb: continue\n        # retirer le superflu le plus coûteux\n        for d in sorted(ch, key=lambda d: -w[d]):\n            if S - size[d] >= lb: ch.remove(d); S -= size[d]; cost -= w[d]\n        if cost < best[1]: best = (ch, cost)\n    return best\n\ndef build_lp():\n    n = len(cols); r_, c_, v_ = [], [], []\n    for j, (k, c, ent) in enumerate(cols):\n        for r, a in ent: r_.append(r); c_.append(j); v_.append(a)\n    A = sp.csc_matrix((v_, (r_, c_)), shape=(len(rows_lo), n))\n    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = len(rows_lo)\n    lp.col_cost_ = np.array([c for _, c, _ in cols]); lp.col_lower_ = np.zeros(n); lp.col_upper_ = np.ones(n)\n    lp.row_lower_ = np.array(rows_lo); lp.row_upper_ = np.array(rows_hi)\n    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise\n    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data\n    h = highspy.Highs(); h.setOptionValue(\'output_flag\', False); h.passModel(lp); return h\n\nclass DP:\n    """tarification exacte : coût minimal (somme des duaux des datasets) pour chaque somme de tailles exacte"""\n    def __init__(self, t, freeonly, w, smax):\n        self.c = [d for d in byt[t] if not (freeonly and dcr[d])]\n        n = len(self.c); self.smax = smax\n        dp = np.full(smax + 1, np.inf); dp[0] = 0.0\n        self.take = np.zeros((n, smax + 1), dtype=bool)\n        for k, d in enumerate(self.c):\n            sz = int(size[d])\n            if sz > smax: continue\n            new = dp[:-sz] + w[d]; seg = dp[sz:]; m = new < seg - 1e-12\n            seg[m] = new[m]; self.take[k, sz:][m] = True\n        self.dp = dp\n    def best(self, lb, ub):\n        ub = min(ub, self.smax)\n        if lb > ub: return None, np.inf\n        seg = self.dp[lb:ub + 1]; j = int(np.argmin(seg))\n        if not np.isfinite(seg[j]): return None, np.inf\n        S = lb + j; cost = seg[j]; ds = []\n        for k in range(len(self.c) - 1, -1, -1):\n            if S == 0: break\n            if self.take[k, S]: ds.append(self.c[k]); S -= int(size[self.c[k]])\n        return ds, cost\n\n\nubmax = {t: max(r[\'upperBound\'] for m in M for tt, r in m[\'dataRequirements\'].items() if tt == t) for t in T}\nNDRAW = int(os.environ.get(\'NDRAW\', 3)); NCAND = int(os.environ.get(\'NCAND\', 6000))\nNCORE = int(os.environ.get(\'NCORE\', 300)); LNST = float(os.environ.get(\'LNST\', 30)); DUREE = float(os.environ.get(\'DUREE\', 900))\nt0 = time.time()\n# ---- prix stables (LP agrégé) et modèles candidats ----\nres, lcols = build_and_solve(inst, K=6, KQ=6, SMALL=0, verbose=False)\nprices = res.prices\nrc = np.abs(res.lower.marginals) + np.abs(res.upper.marginals); score = defaultdict(lambda: 1e18)\nfor j, key in enumerate(lcols):\n    if key[0] in (\'x\', \'z\', \'u\', \'w\'):\n        sc_ = 0.0 if res.x[j] > 1e-6 else rc[j]\n        score[key[1]] = min(score[key[1]], sc_)\ndel res, lcols\ncand = set(sorted(score, key=lambda i: score[i])[:NCAND]) | set(i for k in start for i in [k[1]])\nprint(f\'modèles candidats {len(cand)}\', flush=True)\nrnd = np.random.default_rng(1)\nfor draw in range(NDRAW):\n    w = np.array([prices[dtype[d]][dcr[d]] * size[d] for d in range(len(D))])\n    if draw: w = w * (1 + 0.15 * rnd.random(len(D)))\n    added = 0\n    for t in T:\n        for freeonly in (True, False):\n            dp = DP(t, freeonly, w, min(ubmax[t], 120000)); mode = \'f\' if freeonly else \'h\'\n            for i in cand:\n                q = M[i][\'dataRequirements\'].get(t)\n                if q is None or neg(q) or (i, t, mode) not in R_COV: continue\n                ds, cst = dp.best(q[\'lowerBound\'], q[\'upperBound\'])\n                if ds is not None and add_pattern(i, t, mode, ds) is not None: added += 1\n                if not freeonly and len(M[i][\'dataRequirements\']) == 1 and ds is not None:\n                    if add_source(i, ds) is not None: added += 1\n            del dp\n    print(f\'tirage {draw}: colonnes ajoutées {added}, total {len(cols)} ({time.time()-t0:.0f}s)\', flush=True)\n\n# ---- MIP par morceaux (LNS), démarrage à chaud ----\nh = build_lp(); n = len(cols)\nh.setOptionValue(\'mip_rel_gap\', 0.0); h.setOptionValue(\'mip_abs_gap\', 0.5)\nh.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * n))\nowner = np.array([k[1] for k, _, _ in cols]); cost = np.array([c for _, c, _ in cols])\nxs = np.array([1.0 if k in start else 0.0 for k, _, _ in cols]); vbest = float(-cost @ xs)\nidx = np.arange(n, dtype=np.int32); candl = sorted(cand); marg = sorted(score, key=lambda i: score[i])[:3000]\ndef reconstruct(x):\n    dm = []; mm = []; toks = defaultdict(list); need = []\n    for (k, c, ent), xv in zip(cols, x):\n        if xv < 0.5: continue\n        if k[0] == \'p\':\n            for d in k[4]: dm.append([d, k[1]])\n        elif k[0] == \'s\':\n            for d in k[2]: dm.append([d, k[1]])\n            toks[next(iter(M[k[1]][\'dataRequirements\']))].append(k[1])\n        elif k[0] == \'y\': need.append((k[2], k[1]))\n    for t, i in need: mm.append([toks[t].pop(), i])\n    return {"dataMappings": dm, "modelMappings": mm}\ndef add_col_highs(j):\n    k, c, ent = cols[j]\n    rr = np.array([r for r, _ in ent], dtype=np.int32); vv = np.array([v for _, v in ent], dtype=float)\n    h.addCol(float(c), 0.0, 1.0, len(rr), rr, vv)\n    h.changeColIntegrality(j, highspy.HighsVarType.kInteger)\nNUNCH = int(os.environ.get(\'NUNCH\', 800))\nT0 = float(os.environ.get(\'T0\', 0)); KICK = int(os.environ.get(\'KICK\', 40)); KFRAC = float(os.environ.get(\'KFRAC\', 0.03))\nMAXCOLS = int(os.environ.get(\'MAXCOLS\', 400000))\nSEQ = os.environ.get(\'SEQ\', \'1\') == \'1\'\nPREGION = float(os.environ.get(\'PREGION\', 0.7)); RMAX = int(os.environ.get(\'RMAX\', 5000)); RWIN = int(os.environ.get(\'RWIN\', 1500))\ntabu = {}; TABU = int(os.environ.get(\'TABU\', 0)); recent_out = set(); WPOW = float(os.environ.get(\'WPOW\', 1.0))\nxbest = xs.copy(); last_imp = 0; last_kick = 0; last_cur = 0; base_cols = len(cols)\ndef rebuild(keep):\n    global cols, colset, h, owner, cost, xs, xbest\n    keep = [j for j in keep if j < len(cols)]\n    xs_k = xs[keep]; xb_k = xbest[keep] if len(xbest) >= len(cols) else np.concatenate([xbest, np.zeros(len(cols) - len(xbest))])[keep]\n    cols = [cols[j] for j in keep]; colset = set(k for k, _, _ in cols)\n    h = build_lp(); n_ = len(cols)\n    h.setOptionValue(\'mip_rel_gap\', 0.0); h.setOptionValue(\'mip_abs_gap\', 0.5)\n    h.changeColsIntegrality(n_, np.arange(n_, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * n_))\n    owner = np.array([k[1] for k, _, _ in cols]); cost = np.array([c for _, c, _ in cols]); xs = xs_k; xbest = xb_k\n    print(f\'  élagage : {n_} colonnes conservées\', flush=True)\nit = 0; tl0 = time.time()\nchosen_models = lambda: sorted({owner[j] for j in np.nonzero(xs > 0.5)[0] if cols[j][0][0] != \'y\'})\nunch_cand = list(range(N)) if os.environ.get(\'ALLCAND\', \'1\') == \'1\' else [i for i in candl]\nwhile time.time() - tl0 < DUREE:\n    it += 1\n    ch = chosen_models()\n    temp = T0 * max(0.0, 1 - (time.time() - tl0) / DUREE)\n    if rnd.random() < PREGION:\n        # coeur « régional » : modèles qui détiennent des datasets d\'un type dans une plage de tailles\n        t_r = T[int(rnd.integers(4))]; a_r = int(rnd.integers(0, RMAX)); b_r = a_r + RWIN\n        holders = set()\n        for j in np.nonzero(xs > 0.5)[0]:\n            k = cols[j][0]\n            dl = k[4] if k[0] == \'p\' else (k[2] if k[0] == \'s\' else ())\n            if any(dtype[d] == t_r and a_r <= size[d] <= b_r for d in dl): holders.add(int(k[1]))\n        holders = list(holders)\n        if len(holders) > 2 * NCORE: holders = [int(i) for i in rnd.choice(holders, 2 * NCORE, replace=False)]\n        core = set(holders) | set(int(i) for i in rnd.choice(ch, max(1, NCORE // 3), replace=False))\n    else:\n        core = set(int(i) for i in rnd.choice(ch, NCORE, replace=False))\n    chs = set(ch); pool_un = np.array([i for i in unch_cand if i not in chs])\n    wv = np.array([(M[i][\'value\'] + 1) / (M[i][\'energyCost\'] + 1e10) for i in pool_un]) ** WPOW; wv /= wv.sum()\n    sel_un = set(int(i) for i in rnd.choice(pool_un, NUNCH, replace=False, p=wv))\n    if not TABU: sel_un |= {i for i in recent_out if i not in chs}\n    sel_un = {i for i in sel_un if tabu.get(i, -1) < it}\n    # datasets disponibles : non utilisés + ceux des modèles du coeur\n    used = set()\n    for j in np.nonzero(xs > 0.5)[0]:\n        k = cols[j][0]\n        if k[0] == \'p\' and k[1] not in core: used.update(k[4])\n        elif k[0] == \'s\' and k[1] not in core: used.update(k[2])\n    avail = [d for d in range(len(D)) if d not in used]\n    w = np.array([prices[dtype[d]][dcr[d]] * size[d] for d in range(len(D))]) * (1 + 0.1 * rnd.random(len(D)))\n    nbefore = len(cols)\n    for t in T:\n        for freeonly in (True, False):\n            sub = [d for d in avail if dtype[d] == t and not (freeonly and dcr[d])]\n            if not sub: continue\n            saved = byt[t]; byt[t] = sub\n            dp = DP(t, freeonly, w, min(ubmax[t], 120000)); byt[t] = saved\n            mode = \'f\' if freeonly else \'h\'\n            for i in core | sel_un:\n                q = M[i][\'dataRequirements\'].get(t)\n                if q is None or neg(q) or (i, t, mode) not in R_COV: continue\n                ds, cst = dp.best(q[\'lowerBound\'], q[\'upperBound\'])\n                if ds is not None: add_pattern(i, t, mode, ds)\n                if not freeonly and len(M[i][\'dataRequirements\']) == 1 and ds is not None: add_source(i, ds)\n            del dp\n    # passe séquentielle : on construit des motifs DISJOINTS (chaque besoin prend des datasets que les\n    # précédents n\'ont pas pris), dans un ordre aléatoire -> au moins une affectation cohérente du coeur\n    if SEQ:\n        rem = set(avail)\n        items = [(i, t) for i in core for t in M[i][\'dataRequirements\'] if not neg(M[i][\'dataRequirements\'][t])]\n        rnd.shuffle(items)\n        for i, t in items:\n            q = M[i][\'dataRequirements\'][t]\n            for freeonly in ((True, False) if (i, t, \'f\') in R_COV else (False,)):\n                sub = [d for d in rem if dtype[d] == t and not (freeonly and dcr[d])]\n                if not sub: continue\n                saved = byt[t]; byt[t] = sub\n                dp = DP(t, freeonly, w, min(q[\'upperBound\'], 120000)); byt[t] = saved\n                ds, cst = dp.best(q[\'lowerBound\'], q[\'upperBound\'])\n                if ds is None: continue\n                add_pattern(i, t, \'f\' if freeonly else \'h\', ds)\n                if len(M[i][\'dataRequirements\']) == 1 and not freeonly: add_source(i, ds)\n                rem -= set(ds); break\n    for j in range(nbefore, len(cols)): add_col_highs(j)\n    n = len(cols); idx = np.arange(n, dtype=np.int32)\n    owner = np.concatenate([owner, np.array([cols[j][0][1] for j in range(nbefore, n)])]) if n > len(owner) else owner\n    cost = np.array([c for _, c, _ in cols]); xs = np.concatenate([xs, np.zeros(n - len(xs))]); xbest = np.concatenate([xbest, np.zeros(n - len(xbest))])\n    free_ = np.isin(owner, np.fromiter(core | sel_un, dtype=owner.dtype))\n    hi_ = np.where(free_, 1.0, xs)\n    if TABU:\n        tb = np.fromiter((i for i, u in tabu.items() if u >= it), dtype=owner.dtype)\n        if len(tb): hi_ = np.where(np.isin(owner, tb) & (xs < 0.5), 0.0, hi_)\n    h.changeColsBounds(n, idx, np.where(free_, 0.0, xs), hi_)\n    hs = highspy.HighsSolution(); hs.col_value = list(xs); hs.value_valid = True; h.setSolution(hs)\n    h.setOptionValue(\'time_limit\', float(h.getRunTime() + LNST))\n    with silence(): h.run()\n    if h.getInfo().primal_solution_status != 2: continue\n    x = np.round(np.array(h.getSolution().col_value)); v = float(-cost @ x)\n    if v > vbest + 0.5:\n        s_ = reconstruct(x); st = json.dumps(s_); sc = getSolutionScore(st, txt, show_details=False)\n        if sc[1]:\n            xbest, vbest = x.copy(), v; last_imp = it\n            open(f\'out/{name}_legit_{sc[0]}.json\', \'w\').write(st)\n            print(f\'  LNS {it}: NOUVEAU MEILLEUR {sc[0]}  (colonnes {n}, {time.time()-t0:.0f}s)\', flush=True)\n        else:\n            print(\'  solution invalide :\', sc[2], flush=True); x = xs\n    if v > float(-cost @ xs) + 0.5: last_cur = it\n    xs = x; vcur = v\n    # perturbation (recuit) : sans progrès depuis KICK tours, on retire au hasard une part des modèles\n    if it - max(last_cur, last_kick) >= KICK:\n        last_kick = it\n        chm = chosen_models(); out_ = set(int(i) for i in rnd.choice(chm, max(1, int(len(chm) * KFRAC)), replace=False)); recent_out.clear(); recent_out.update(out_)\n        for i in out_: tabu[i] = it + TABU\n        for j in np.nonzero(xs > 0.5)[0]:\n            if owner[j] in out_: xs[j] = 0.0\n        print(f\'  LNS {it}: perturbation, {len(out_)} modèles retirés (valeur {float(-cost @ xs):.0f})\', flush=True)\n    # élagage : on ne garde que les colonnes de la solution courante et du meilleur\n    if len(cols) > MAXCOLS:\n        keep = sorted(set(np.nonzero(xs > 0.5)[0]) | set(np.nonzero(xbest[:len(xs)] > 0.5)[0]) | set(range(base_cols)))\n        rebuild(keep)\n    if it % 10 == 0:\n        print(f\'  LNS {it}: courant {float(-cost @ xs):.0f}, meilleur {vbest:.0f} (colonnes {len(cols)}, {time.time()-t0:.0f}s)\', flush=True)\nprint(\'FINAL\', vbest, flush=True)\n'

def main_cg():
    ap = argparse.ArgumentParser(); ap.add_argument('--cg', action='store_true')
    ap.add_argument('--instance'); ap.add_argument('--depart'); ap.add_argument('--duree', type=float, default=3600)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--kfrac', type=float, default=0.02); ap.add_argument('--tabu', type=int, default=0)
    a = ap.parse_args()
    os.environ.update(dict(NDRAW='1', NCORE='150', NUNCH='600', LNST='10', DUREE=str(a.duree), KICK='40', KFRAC=str(a.kfrac),
                           TABU=str(a.tabu), MAXCOLS='250000', SEED=str(a.seed), PREGION='0.5'))
    ns = {'__name__': 'cg', 'silence': silence, 'getSolutionScore': getSolutionScore, 'build_and_solve': build_and_solve,
          'CG_ARGS': [a.instance, a.depart]}
    sys.argv = [sys.argv[0], a.instance, a.depart]
    exec(compile(CG_SRC, 'cg6c', 'exec'), ns)

# ===== ORCHESTRATEUR : tout sur l'instance 6, on ne garde que la meilleure solution =====
def score_fichier(p, txt):
    try:
        st = open(p).read(); sol = json.loads(st)
        if any(x < 0 or y < 0 for x, y in sol.get('modelMappings', [])): return 0
        sc = getSolutionScore(st, txt, show_details=False)
        return sc[0] if sc[1] else 0
    except Exception:
        return 0


def etat_calcul(path):
    """(phase, meilleure valeur vue, dernière ligne utile) d'après le journal d'un calcul"""
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, 2); n = fh.tell(); fh.seek(max(0, n - 60000))
            lines = fh.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return 'démarrage', 0, ''
    phase = 'préparation (LP)'; best = 0; der = ''
    for l in lines:
        ls = l.strip()
        if not ls: continue
        der = ls
        m = None
        if ls.startswith('borne LP'): phase = 'MIP (paires exactes)'
        elif ls.startswith('MIP en cours'): phase = 'MIP'
        elif ls.startswith('MIP :') or ls.startswith('MIP st'): phase = 'affectation'
        elif ls.startswith('after greedy'): phase = 'recherche locale'; m = re.search(r'after greedy (\d+)', ls)
        elif ls.startswith('tirage'): phase = 'colonnes'
        elif ls.startswith('LNS') or ls.startswith('  LNS') or ls.startswith('it '):
            phase = 'LNS'; m = re.search(r'meilleur (\d+)', ls) or re.search(r'value (\d+)', ls)
        elif 'NOUVEAU MEILLEUR' in ls: m = re.search(r'MEILLEUR (\d+)', ls)
        elif 'SAUVEGARDE' in ls: m = re.search(r'SAUVEGARDE (\d+)', ls)
        elif re.match(r'd\W{0,2}part ', ls): m = re.search(r'part (\d+)', ls)
        elif ls.startswith('RÉSULTAT') or ls.startswith('FINAL'):
            phase = 'terminé'; m = re.search(r'\((\d+), True', ls)
        elif 'ÉCHEC' in ls or 'Error' in ls or 'Traceback' in ls: phase = 'ERREUR'
        else:
            mm = re.search(r'(-?\d+\.\d+)\s+(-?\d+\.\d+|inf)\s+(\d+\.\d+%|inf|Large)', ls)
            if mm and phase.startswith('MIP'):
                try:
                    bb = -float(mm.group(1)); bs = mm.group(2)
                    der = f'plan entier: {(-float(bs)):.0f}' if bs != 'inf' else 'pas encore de plan entier'
                    der += f' | borne {bb:.0f} | écart {mm.group(3)}'
                    continue
                except ValueError:
                    pass
        if m:
            try: best = max(best, int(float(m.group(1))))
            except ValueError: pass
    return phase, best, der[:90]

def ram_systeme():
    try:
        import psutil
        vm = psutil.virtual_memory(); return f'RAM utilisée {vm.used / 2**30:.0f}/{vm.total / 2**30:.0f} Go, CPU {psutil.cpu_percent():.0f} %'
    except Exception:
        return '(pip install psutil pour voir la RAM et le CPU réels)'

def orchestrer():
    ap = argparse.ArgumentParser(description='Optimise l\'instance 6 en parallèle (solutions légales), garde uniquement la meilleure')
    ap.add_argument('--heures', type=float, default=10.0)
    ap.add_argument('--ram', type=float, default=38.0, help='Go de RAM à utiliser au maximum')
    ap.add_argument('--cpu', type=int, default=24)
    ap.add_argument('--instance', default='6_shortage.json')
    ap.add_argument('--depart', default='', help='ta meilleure solution actuelle (fichier .json)')
    ap.add_argument('--copies', type=int, default=1, help='exemplaires de chaque réglage en parallèle (graines différentes)')
    a = ap.parse_args()
    if not os.path.exists(a.instance): print('Instance introuvable :', a.instance); return
    name = os.path.basename(a.instance).replace('.json', ''); BEST = f'{name}_MEILLEUR.json'
    txt = open(a.instance).read(); fin = time.time() + a.heures * 3600
    os.makedirs('out', exist_ok=True); os.makedirs('logs', exist_ok=True)
    best = score_fichier(BEST, txt) if os.path.exists(BEST) else 0
    if a.depart and os.path.exists(a.depart):
        v = score_fichier(a.depart, txt)
        if v > best: shutil.copy(a.depart, BEST); best = v
    print(f'meilleure solution de départ : {best}  ({BEST})', flush=True)
    print(f'Python utilisé : {sys.executable}', flush=True)
    try:
        import highspy  # noqa
        lns_ok = True
        print(f'highspy trouvé : {getattr(highspy, "__file__", "?")} -> recherche par colonnes et boucle LNS activées', flush=True)
    except Exception as e:
        lns_ok = False
        print(f'highspy NON disponible pour ce Python ({type(e).__name__}: {e})', flush=True)
        print('  -> installe-le dans CE Python :  "' + sys.executable + '" -m pip install highspy', flush=True)
    # gabarits : type, RAM (Go), paramètres
    gab = [dict(kind='six', ram=6.0, rc=0.3), dict(kind='six', ram=6.0, rc=0.6), dict(kind='six', ram=6.5, rc=1.2)]
    if lns_ok:
        gab += [dict(kind='cg', ram=3.5, kfrac=0.02, tabu=0), dict(kind='cg', ram=3.5, kfrac=0.03, tabu=80),
                dict(kind='cg', ram=3.5, kfrac=0.05, tabu=120), dict(kind='std', ram=4.0)]
    gab = [dict(g) for g in gab for _ in range(max(1, a.copies))]   # chaque réglage lancé en plusieurs exemplaires
    prio = {'cg': 0, 'std': 1, 'six': 2}   # priorité de lancement : colonnes d'abord (les plus prometteuses)
    gab.sort(key=lambda g: prio[g['kind']])
    print('calculs : ' + ', '.join(f"{k} x{sum(1 for g in gab if g['kind'] == k)}" for k in ('cg', 'std', 'six') if any(g['kind'] == k for g in gab)), flush=True)
    print(f'{len(gab)} calculs simultanés prévus (RAM estimée {sum(g["ram"] for g in gab):.0f} Go, plafond {a.ram:.0f} Go)', flush=True)
    running = []; seed = {i: 1 + 100 * i for i in range(len(gab))}; t0 = time.time(); last = 0; infos = {}; histo = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    me = os.path.abspath(__file__)
    def ram_used(): return sum(gab[g]['ram'] for g, _, _ in running)
    def publier():
        nonlocal best
        for p in glob.glob(os.path.join('out', '*.json')):
            v = score_fichier(p, txt)
            if v > best:
                shutil.copy(p, BEST); best = v
                print(f'[{(time.time()-t0)/3600:5.2f} h] *** NOUVEAU MEILLEUR {v} -> {BEST}', flush=True)
            try: os.remove(p)   # on ne garde que le meilleur
            except OSError: pass
    try:
        while time.time() < fin:
            reste = fin - time.time()
            if reste > 600:
                for gi, g in enumerate(gab):
                    if any(r[0] == gi for r in running): continue
                    if ram_used() + g['ram'] > a.ram or len(running) >= a.cpu: continue
                    if g['kind'] == 'cg' and not os.path.exists(BEST): continue
                    s_ = seed[gi]; seed[gi] += 1
                    if g['kind'] == 'six':
                        cmd = [sys.executable, '-u', me, '--six', '--instance', a.instance, '--rc', str(g['rc'] * (1 + 0.15 * ((s_ - 1) % 100))),
                               '--temps', str(min(3 * 3600, reste * 0.6)), '--threads', '8']
                    elif g['kind'] == 'cg':
                        cmd = [sys.executable, '-u', me, '--cg', '--instance', a.instance, '--depart', BEST,
                               '--duree', str(min(2 * 3600, reste - 300)), '--seed', str(s_ + 10 * gi), '--kfrac', str(g['kfrac']), '--tabu', str(g['tabu'])]
                    else:
                        mip = min(1800, reste * 0.3); lns = max(0, min(4 * 3600, reste * 0.6 - mip))
                        cmd = [sys.executable, '-u', me, '--worker', '--instance', a.instance, '--mip', str(mip), '--rl', '120',
                               '--seed', str(s_), '--lns', str(lns), '--lns-mip', '120', '--lns-core', '0']
                    log = open(os.path.join('logs', f'{g["kind"]}{gi}.txt'), 'w')   # un seul journal par type de calcul
                    pr = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                    running.append((gi, pr, log)); infos[gi] = (s_, time.time())
                    print(f'[{(time.time()-t0)/3600:5.2f} h] lancé {g["kind"]}{gi} graine {s_} (RAM prévue {ram_used():.0f} Go)', flush=True)
            for r in running[:]:
                if r[1].poll() is not None: r[2].close(); running.remove(r)
            if time.time() - last > 60:
                last = time.time(); avant = best; publier()
                if best > avant: histo.append((time.time() - t0, best))
                reste_h = max(0.0, (fin - time.time()) / 3600)
                print('', flush=True)
                print(f'=== {(time.time()-t0)/3600:5.2f} h écoulées, {reste_h:5.2f} h restantes | MEILLEUR {best} (écart à 669 800 : {best - 669800:+d}) | '
                      f'{len(running)} calculs | {ram_systeme()}', flush=True)
                for gi, pr, _ in sorted(running, key=lambda r: r[0]):
                    g = gab[gi]; s_, tl = infos.get(gi, (0, time.time()))
                    ph, bv, der = etat_calcul(os.path.join('logs', f'{g["kind"]}{gi}.txt'))
                    lib = {'six': 'paires exactes', 'cg': 'colonnes', 'std': 'MIP+LNS'}[g['kind']]
                    el = time.time() - tl
                    print(f'   {g["kind"]}{gi:<3} {lib:<15} graine {s_:<5} depuis {int(el // 3600)}h{int(el % 3600 // 60):02d}  '
                          f'phase: {ph:<22} meilleur vu: {bv if bv else "-":<8} | {der}', flush=True)
                if histo:
                    print('   progrès : ' + ', '.join(f'{v} à {t/3600:.2f} h' for t, v in histo[-6:]), flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print('Arrêt demandé...', flush=True)
    for r in running: r[1].terminate()
    time.sleep(3); publier()
    print(f'Terminé. Meilleur score : {best}  -> {BEST}', flush=True)

if __name__ == '__main__':
    if '--six' in sys.argv: main_six()
    elif '--cg' in sys.argv: main_cg()
    elif '--worker' in sys.argv or '--exporter-mps' in sys.argv or '--importer' in sys.argv: main()
    else: orchestrer()
