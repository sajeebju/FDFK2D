#!/usr/bin/env python3
"""
wv_invert.py
============
2-D cross-convolution waveform inversion of the Wabash SsPmp dataset, with
FDFK2D as the forward engine.

Read this before running anything
---------------------------------
One forward evaluation is 7 FDFK2D simulations, roughly 25 minutes each for a
585 km model, so ~25 min wall clock if you run the events in parallel on 7+
cores.  That cost dictates the whole design:

  * you get O(100) forward evaluations, not O(100000), so this is a
    derivative-free local refinement, NOT an FWI and NOT a Bayesian sampling
  * the model vector is therefore ~9-15 numbers (Moho depth nodes), not a grid
  * you must start from your existing hierarchical Bayesian result and treat
    this as a 2-D refinement plus validation of it

If you want uncertainties, they come from the 1-D Bayesian inversion you already
have.  Do not present a covariance from this.

Objective
---------
    Phi(m) = sum_i w_i * E_i(m)  +  lambda_s * roughness  +  lambda_p * prior

E_i is the cross-convolution shape misfit of trace i,
||c1/||c1|| - c2/||c2||||^2 with c1 = R_obs * Z_syn, c2 = R_syn * Z_obs.

A useful property of that misfit: shifting BOTH synthetic components by the same
tau shifts c1 and c2 equally, so Phi is insensitive to the absolute alignment of
the synthetic.  What it sees is the SsPmp-minus-S lag and the R/Z amplitude
partitioning -- exactly the two things crustal thickness controls.  This is why
the automatic S pick does not need to be perfect.

Data selection (the honest count)
---------------------------------
Per event: stations in t8.dat, keeping only shift.dat column 3 == 1, intersected
with tpmp.xy.  That is ~26 traces for 20150323045138 and ~175 event-station pairs
over all seven events.  Grid directories are NOT used here: the grid binning
exists so jinv2024 can assign a datum to a 1-D column, but a 2-D simulation
propagates through the whole model at once, so each trace is counted exactly
once.

Gradient
--------
Finite differences would cost N+1 forward evaluations per gradient.  Instead,
because Moho node j only affects traces whose bounce point is near x_j, the odd
nodes are perturbed together in one run and the even nodes in another, and each
partial derivative is formed from that node's own traces only.  Full gradient in
2 forward evaluations regardless of N.  It relies on node spacing being wider
than the bounce-point footprint; the script checks that and warns.

Typical iteration: 1 base + 2 gradient + ~2 line search = 5 evaluations ~ 2 h.

Usage
-----
  # 0. ALWAYS start here: 1-D sweep of a uniform Moho shift, ~9 evaluations
  python3 wv_invert.py --config wv.json --method scan --scan -6 6 1.5

  # 1. then the local refinement
  python3 wv_invert.py --config wv.json --method gradient --iters 15
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdfk2d_common import (ProfileGeom, read_t8, read_shift, read_tpmp, read_su,
                           bandpass, cos_taper, pick_max_envelope, ensure_dir,
                           read_kv, apply_kv, explicit_dests, require_paths,
                           prep_window)
from wv_model2d import Model2D, Background, load_ref_from_file
from su2sac import convert_run
from sac_crossconv import crossconv_pair, load_sac

SETTINGS_TEMPLATE = """# wv.paths -- settings for the Wabash 2-D SsPmp pipeline.
# One "key = value" per line, # starts a comment, an empty value means unset.
# Anything here can be overridden by the matching --flag on the command line.

# ---- paths (no defaults exist; the code has none baked in) ----
events_root  =                 # holds <event>/t8.dat, shift.dat, <sta>.r/.z
tpmp         =                 # .../tpmp.xy
work_root    =                 # every evaluation is written under here
fdfk2d_bin   =                 # the FDFK2D executable
grids_root   =                 # optional, only to copy the final synthetics
ref_1d       =                 # optional: depth_km Vp Vs reference column

# ---- profile ----
prf          = -C-88.0047/38.4484 -A120.5   # empty -> fitted from tpmp.xy

# ---- events ----
events = 20140824232145 20140825143137 20140925175117 20150323045138 20150529070009 20150624223221 20150729023559
nproc  = 7

# ---- model ----
h0_km                   = 52.0
node_spacing_km         = 40.0
structure_halfwidth_km  = 152.0
x0_km = -340.0
x1_km =  245.0
dx    =  200.0
dz    =  200.0
Zkm   =   80.0
npml  =   20
norder =   8
f0     =  0.8

# ---- processing ----
freq        = 0.05 0.5
freq_stages = 0.05 0.2  0.05 0.35  0.05 0.5    # low high pairs, run in order
corners     = 2
win         = -25.0 5.0
taper       = 0.10
dt_common   = 0.05
pick_band   = 0.05 1.0
sign_r      = -1.0
sign_z      = -1.0

# ---- inversion ----
lambda_smooth_gn = 0.5
mu_damp          = 0.05
h_min = 35.0
h_max = 65.0
gn_iters = 5
fd_delta_km = 1.0
# prior_moho = 52 52 53 54 53 52 52 51
# lambda_prior = 1.0
"""


ALL_EVENTS = ['20140824232145', '20140825143137', '20140925175117',
              '20150323045138', '20150529070009', '20150624223221',
              '20150729023559']


# ===========================================================================
# data
# ===========================================================================
def rms_noise(data, sampling_rate, lead_s=20.0):
    """tw_script.py's pre-signal RMS, reproduced so obs.tw is not required."""
    i = int(np.argmax(np.abs(data)))
    j = i - int(lead_s * sampling_rate)
    if j <= 0:
        return np.nan
    seg = data[:j]
    if not np.isfinite(seg).all():
        return np.nan
    return float(np.sqrt(np.mean(seg ** 2)))


def preprocess(x, dt, b, t_arr, win, band, corners, taper, dt_out):
    """Kept as a name; the implementation lives in fdfk2d_common.prep_window
    so the inversion and the standalone tool cut and taper identically."""
    return prep_window(x, dt, b, t_arr, win, dt_out, band, corners, taper)[0]


class Dataset:
    """Every usable event-station pair, with the observed side preprocessed once."""

    def __init__(self, cfg, geom, log):
        from obspy import read
        self.cfg = cfg
        self.geom = geom
        self.traces = []
        tpmp = read_tpmp(cfg['tpmp'])
        bp = {(r['event'], r['station']): r['x'] for r in tpmp}
        band = tuple(cfg['freq']) if cfg.get('freq') else None

        for ev in cfg['events']:
            ed = os.path.join(cfg['events_root'], ev)
            t8 = read_t8(os.path.join(ed, 't8.dat'))
            sh = read_shift(os.path.join(ed, 'shift.dat'))
            n_f = n_t = n_ok = 0
            for sta, rec in sorted(t8.items()):
                ts, flag = sh.get(sta, (None, 0))
                if flag != 1 or ts is None:
                    n_f += 1
                    continue
                if (ev, sta) not in bp:
                    n_t += 1
                    continue
                pr, pz = os.path.join(ed, sta + '.r'), os.path.join(ed, sta + '.z')
                if not (os.path.isfile(pr) and os.path.isfile(pz)):
                    continue
                tr_r = read(pr, format='SAC')[0]
                tr_z = read(pz, format='SAC')[0]
                sr, sz = tr_r.stats.sac, tr_z.stats.sac
                b_r, b_z = float(sr.get('b', 0.0)), float(sz.get('b', 0.0))
                t_end = b_z + (tr_z.stats.npts - 1) * tr_z.stats.delta
                t_obs = ts
                if not (b_z <= t_obs <= t_end):
                    o = float(sz['o']) if 'o' in sz and sz['o'] is not None else None
                    if o is not None and b_z <= o + ts <= t_end:
                        t_obs = o + ts
                    else:
                        log(f"    {ev} {sta}: S={ts:g} outside the trace, skipped")
                        continue
                rms = rms_noise(tr_z.data.astype(float), tr_z.stats.sampling_rate)
                dt_out = cfg['dt_common']
                R = preprocess(tr_r.data, tr_r.stats.delta, b_r, t_obs, cfg['win'],
                               band, cfg['corners'], cfg['taper'], dt_out)
                Z = preprocess(tr_z.data, tr_z.stats.delta, b_z, t_obs, cfg['win'],
                               band, cfg['corners'], cfg['taper'], dt_out)
                s = max(np.abs(R).max(), np.abs(Z).max()) or 1.0
                w = (s / rms) if (np.isfinite(rms) and rms > 0) else 1.0
                self.traces.append(dict(event=ev, station=sta, R=R / s, Z=Z / s,
                                        weight=float(w), rms=rms, t_obs=t_obs,
                                        bounce_x=bp[(ev, sta)],
                                        stla=float(sr['stla']), stlo=float(sr['stlo']),
                                        p_skm=rec['p_skm'], baz=rec['baz'],
                                        syn_pick=None))
                n_ok += 1
            log(f"  {ev}: {n_ok} traces used  (flag!=1 dropped {n_f}, "
                f"not in tpmp.xy {n_t})")

        w = np.array([t['weight'] for t in self.traces])
        if len(w):
            w /= w.mean()
            for t, ww in zip(self.traces, w):
                t['weight'] = float(ww)
        log(f"  total {len(self.traces)} event-station pairs")

    def by_event(self, ev):
        return [t for t in self.traces if t['event'] == ev]


# ===========================================================================
# forward
# ===========================================================================
class Forward:
    def __init__(self, cfg, model, dataset, geom, log):
        self.cfg = cfg
        self.model = model
        self.data = dataset
        self.geom = geom
        self.log = log
        self.n_eval = 0
        self.picks = {}
        self.here = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------------------------------------------
    def _setup(self, work, model_dir):
        cmd = [sys.executable, os.path.join(self.here, 'fdfk2d_setup.py'),
               '--events-root', self.cfg['events_root'],
               '--tpmp', self.cfg['tpmp'],
               '--model-dir', model_dir,
               '--out-root', os.path.join(work, 'runs'),
               '--events'] + list(self.cfg['events']) + [
               '--x0-km', str(self.model.x0_km),
               '--dx', str(self.model.dx), '--dz', str(self.model.dz),
               '--Zkm', str(self.model.Zkm), '--npml', str(self.model.npml),
               '--norder', str(self.cfg['norder']), '--f0', str(self.cfg['f0']),
               '--dense-dx-km', '0', '--shift-flag', '1', '--link-models']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError('fdfk2d_setup failed:\n' + r.stdout[-3000:] + r.stderr[-2000:])
        return r.stdout

    def _run_one(self, run_dir):
        ensure_dir(os.path.join(run_dir, 'seismograms'))
        with open(os.path.join(run_dir, 'run.log'), 'w') as lg:
            p = subprocess.run(f'yes Y | "{self.cfg["fdfk2d_bin"]}" ./input inpar.dat '
                               f'./seismograms seis',
                               shell=True, cwd=run_dir, stdout=lg,
                               stderr=subprocess.STDOUT)
        return run_dir, p.returncode

    # ---------------------------------------------------------------
    def __call__(self, m, tag, keep=False):
        t0 = time.time()
        self.n_eval += 1
        work = ensure_dir(os.path.join(self.cfg['work_root'], tag))
        model_dir = ensure_dir(os.path.join(work, 'model'))
        self.model.write(model_dir, m, self.cfg['fit_vp'], self.cfg['fit_vpvs'],
                         plot=self.cfg.get('plot_models', False))
        self._setup(work, model_dir)

        runs = [os.path.join(work, 'runs', ev) for ev in self.cfg['events']]
        runs = [r for r in runs if os.path.isdir(r)]
        with ThreadPoolExecutor(max_workers=self.cfg['nproc']) as ex:
            for rd, rc in ex.map(self._run_one, runs):
                if rc:
                    raise RuntimeError(f'FDFK2D failed in {rd}, see run.log')

        recs = self._misfit(work)
        phi_d = float(sum(r['weight'] * r['E'] for r in recs))
        phi_r, phi_p = self._regularisation(m)
        phi = phi_d + phi_r + phi_p
        dtm = (time.time() - t0) / 60.0
        self.log(f"  [{tag}] Phi={phi:.5f}  (data {phi_d:.5f} + rough {phi_r:.5f} "
                 f"+ prior {phi_p:.5f})  n={len(recs)}  {dtm:.1f} min")
        with open(os.path.join(work, 'misfit.json'), 'w') as fh:
            json.dump(dict(m=np.asarray(m).tolist(), phi=phi, phi_d=phi_d,
                           phi_rough=phi_r, phi_prior=phi_p,
                           traces=[{k: v for k, v in r.items()
                                    if k not in ('res', 'SR', 'SZ', 'oR', 'oZ',
                                                 'c1', 'c2', 'lag_axis')}
                                   for r in recs]), fh, indent=2)
        if not keep and not self.cfg.get('keep_all'):
            # the .su files are large; the SAC folders are small and are what
            # you would actually want to look at, so only the runs are removed
            shutil.rmtree(os.path.join(work, 'runs'), ignore_errors=True)
            if not self.cfg.get('keep_sac'):
                shutil.rmtree(os.path.join(work, 'sac'), ignore_errors=True)
        return phi, recs

    # ---------------------------------------------------------------
    def _misfit(self, work):
        """
        One iteration's data misfit, taking exactly the same route as the
        standalone tools:

            FDFK2D .su
              -> convert_run()      immediately written as SAC .r/.z, one folder
                                    per event, polarity applied, S pick in `a`
              -> load_sac()         read back as SAC, like the observed data
              -> prep_window()      cut + taper about each record's own S
              -> crossconv_pair()   normalise, convolve, second cut + taper
                                    on the cross-convolutions, compare

        There is no in-memory shortcut, so what the inversion scores is byte
        for byte what you can open in SAC and re-run through sac_crossconv.py.
        """
        cfg = self.cfg
        band = tuple(cfg['freq']) if cfg.get('freq') else None
        cc_win = None if cfg.get('cc_win') is None else tuple(cfg['cc_win'])
        sac_root = ensure_dir(os.path.join(work, 'sac'))
        recs = []
        for ev in cfg['events']:
            rd = os.path.join(work, 'runs', ev)
            if not os.path.isdir(rd):
                continue
            prev = self.picks.get(ev) if not cfg['repick'] else None
            rows, dt, meta = convert_run(
                rd, os.path.join(sac_root, ev),
                name_prefix=cfg.get('syn_prefix', 'fdfk2d_'),
                sign_r=cfg['sign_r'], sign_z=cfg['sign_z'],
                pick_band=tuple(cfg['pick_band']), prev_picks=prev,
                log=self.log)
            self.picks[ev] = {r['station']: r['a'] for r in rows}
            byname = {r['station']: r for r in rows}

            for tr in self.data.by_event(ev):
                row = byname.get(tr['station'])
                if row is None:
                    continue
                SRt = load_sac(row['path_r'])
                SZt = load_sac(row['path_z'])
                ta = SZt['a'] if SZt['a'] is not None else row['a']
                SR, _ = prep_window(SRt['data'], SRt['dt'], SRt['b'], ta,
                                    cfg['win'], cfg['dt_common'], band,
                                    cfg['corners'], cfg['taper'])
                SZ, _ = prep_window(SZt['data'], SZt['dt'], SZt['b'], ta,
                                    cfg['win'], cfg['dt_common'], band,
                                    cfg['corners'], cfg['taper'])
                m = crossconv_pair(tr['R'], tr['Z'], SR, SZ, cfg['dt_common'],
                                   tuple(cfg['win']), cfg['norm'], cc_win,
                                   cfg['cc_taper'])
                dec = max(1, int(round(cfg.get('res_dt', 0.1) / cfg['dt_common'])))
                rec = dict(event=ev, station=tr['station'],
                           E=float(m['E']), weight=tr['weight'],
                           bounce_x=tr['bounce_x'], syn_pick=float(ta),
                           lag_s=float(m['lag_s']), alpha=float(m['alpha']),
                           res=m['residual'][::dec],
                           sac_r=row['path_r'], sac_z=row['path_z'])
                if cfg.get('keep_windows'):
                    rec.update(SR=m['sR'], SZ=m['sZ'], oR=m['oR'], oZ=m['oZ'],
                               c1=m['c1'], c2=m['c2'], lag_axis=m['lag_axis'])
                recs.append(rec)
        return recs

    def _regularisation(self, m):
        h, _, _ = self.model.unpack(m, self.cfg['fit_vp'], self.cfg['fit_vpvs'])
        rough = 0.0
        if len(h) > 2:
            d2 = h[2:] - 2 * h[1:-1] + h[:-2]
            rough = self.cfg['lambda_smooth'] * float(np.sum(d2 ** 2))
        prior = 0.0
        if self.cfg.get('prior_moho') is not None:
            p = np.asarray(self.cfg['prior_moho'], float)
            sd = self.cfg.get('prior_sigma_km', 3.0)
            prior = self.cfg['lambda_prior'] * float(np.sum(((h - p) / sd) ** 2))
        return rough, prior


# ===========================================================================
# optimisers
# ===========================================================================
def node_traces(model, recs, radius=None):
    """
    Traces influenced by each Moho node.  A PCHIP node's support is one node
    spacing either side, so a trace contributes to the two nodes bracketing its
    bounce point.  With the even/odd staggering below, perturbed nodes are two
    spacings apart, so at most ONE perturbed node lies inside any trace's
    support -- which is exactly what makes the 2-run Jacobian valid.
    """
    xn = model.x_nodes
    if radius is None:
        radius = float(np.median(np.diff(xn))) if len(xn) > 1 else 40.0
    out = {j: [] for j in range(len(xn))}
    for k, r in enumerate(recs):
        for j, x in enumerate(xn):
            if abs(r['bounce_x'] - x) < radius:
                out[j].append(k)
    return out


def local_phi(recs, idx):
    return float(sum(recs[k]['weight'] * recs[k]['E'] for k in idx))


def staggered_gradient(fwd, model, m, recs0, delta, it, log):
    """
    Gradient of the data term with respect to the Moho nodes in 2 forward
    evaluations: perturb even nodes together, then odd nodes together, and take
    each partial from that node's own traces.
    """
    n = model.n_nodes
    g = np.zeros(len(m))
    groups = node_traces(model, recs0)
    base = {j: local_phi(recs0, idx) for j, idx in groups.items()}
    for par in (0, 1):
        mp = np.array(m, float)
        js = [j for j in range(n) if j % 2 == par]
        mp[js] += delta
        _, recs = fwd(mp, f'it{it:02d}_grad{par}')
        gp = node_traces(model, recs)
        for j in js:
            idx = gp[j]
            if not idx:
                continue
            g[j] = (local_phi(recs, idx) - base[j]) / delta
    empty = [j for j in range(n) if not groups[j]]
    if empty:
        log(f"    nodes with no traces (left at prior): "
            f"{np.round(model.x_nodes[empty]).astype(int).tolist()} km")
    return g


# ---------------------------------------------------------------------------
# damped linearized least squares (Gauss-Newton)
# ---------------------------------------------------------------------------
def _key(r):
    return (r['event'], r['station'])


def assemble(recs, keys):
    """Weighted residual vector, ordered by `keys`, with row slices per trace."""
    d = {_key(r): r for r in recs}
    seg, rows, n = {}, [], 0
    for k in keys:
        r = d[k]
        w = np.sqrt(r['weight'])
        rows.append(w * r['res'])
        seg[k] = slice(n, n + len(r['res']))
        n += len(r['res'])
    return np.concatenate(rows) if rows else np.zeros(0), seg


def staggered_jacobian(fwd, model, cfg, m, recs0, keys, seg, ndat, it, log):
    """
    Frechet derivatives of the weighted cross-convolution residual with respect
    to the Moho nodes, in TWO forward evaluations instead of N+1.

    Even nodes are perturbed together, then odd nodes.  Column j is filled only
    for the rows of traces inside node j's support; every other row is exactly
    zero, which is the locality assumption stated explicitly rather than hidden.
    """
    n = model.n_nodes
    J = np.zeros((ndat, len(m)))
    sup = node_traces(model, recs0)
    d0 = {_key(r): r for r in recs0}
    delta = cfg['fd_delta_km']
    for par in (0, 1):
        js = [j for j in range(n) if j % 2 == par]
        mp = np.array(m, float)
        mp[js] += delta
        _, recs = fwd(mp, f'it{it:02d}_jac{par}')
        dp = {_key(r): r for r in recs}
        for j in js:
            for kk in sup[j]:
                key = _key(recs0[kk])
                if key not in dp or key not in seg:
                    continue
                w = np.sqrt(d0[key]['weight'])
                J[seg[key], j] = w * (dp[key]['res'] - d0[key]['res']) / delta
    # optional scalar parameters: one extra evaluation each, all rows
    k = n
    for name, on in (('dvp', cfg['fit_vp']), ('dvpvs', cfg['fit_vpvs'])):
        if not on:
            continue
        step = cfg['fd_delta_vp'] if name == 'dvp' else cfg['fd_delta_vpvs']
        mp = np.array(m, float)
        mp[k] += step
        _, recs = fwd(mp, f'it{it:02d}_jac_{name}')
        rp, _ = assemble(recs, keys)
        r0, _ = assemble(recs0, keys)
        if len(rp) == len(r0):
            J[:, k] = (rp - r0) / step
        k += 1
    return J


def smoothing_operator(n_nodes, n_par):
    """Second-difference operator, zero-padded over any scalar parameters."""
    if n_nodes < 3:
        return np.zeros((0, n_par))
    L = np.zeros((n_nodes - 2, n_par))
    for i in range(n_nodes - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def solve_damped(J, r, L, lam, mu):
    """(J'J + lam^2 L'L + mu^2 I) dm = -J'r"""
    A = J.T @ J + lam ** 2 * (L.T @ L) + mu ** 2 * np.eye(J.shape[1])
    b = -J.T @ r
    dm = np.linalg.solve(A, b)
    return dm, A


def run_gauss_newton(fwd, model, cfg, m0, log):
    m = np.array(m0, float)
    keys = None
    hist = []
    lam = cfg['lambda_smooth_gn']
    mu = cfg['mu_damp']
    stages = cfg.get('freq_stages') or [cfg['freq']]

    for si, band in enumerate(stages):
        cfg['freq'] = list(band)
        log(f"\n########## stage {si + 1}/{len(stages)}: "
            f"{band[0]:g}-{band[1]:g} Hz ##########")
        if si > 0:
            log("  (observed side is re-filtered for the new band)")
            fwd.data = Dataset(cfg, fwd.geom, log)
        phi, recs = fwd(m, f's{si}_it00', keep=True)
        keys = [_key(r) for r in recs]

        for it in range(1, cfg['gn_iters'] + 1):
            r0, seg = assemble(recs, keys)
            log(f"\n--- stage {si + 1} iteration {it} --- Phi = {phi:.5f}, "
                f"{len(r0)} data, {len(m)} parameters")
            J = staggered_jacobian(fwd, model, cfg, m, recs, keys, seg,
                                   len(r0), it, log)
            L = smoothing_operator(model.n_nodes, len(m))
            dm, A = solve_damped(J, r0, L, lam, mu)
            log(f"  proposed dm [km]: {np.round(dm[:model.n_nodes], 2).tolist()}")

            accepted = False
            for scale in cfg['gn_step_scales']:
                trial = m + scale * dm
                lo, hi = model.param_bounds
                trial[:model.n_nodes] = np.clip(trial[:model.n_nodes], lo, hi)
                p2, r2 = fwd(trial, f's{si}_it{it:02d}_step{scale:g}'.replace('.', 'p'))
                if p2 < phi:
                    m, phi, recs, accepted = trial, p2, r2, True
                    log(f"  accepted step scale {scale:g}")
                    break
            if not accepted:
                log("  no step scale reduced the misfit -- linearization has "
                    "reached its limit for this band")
                break

            h, _, _ = model.unpack(m, cfg['fit_vp'], cfg['fit_vpvs'])
            log(f"  Moho [km]: {np.round(h, 2).tolist()}")
            hist.append(dict(stage=si, it=it, phi=phi, m=m.tolist()))
            with open(os.path.join(cfg['work_root'], 'gn_history.json'), 'w') as fh:
                json.dump(hist, fh, indent=2)
            if np.max(np.abs(dm[:model.n_nodes])) < cfg['gn_tol_km']:
                log("  parameter change below tolerance, converged")
                break

    # ---- posterior statistics, valid ONLY within this parameterisation -------
    r0, seg = assemble(recs, keys)
    J = staggered_jacobian(fwd, model, cfg, m, recs, keys, seg, len(r0), 99, log)
    L = smoothing_operator(model.n_nodes, len(m))
    A = J.T @ J + lam ** 2 * (L.T @ L) + mu ** 2 * np.eye(len(m))
    Ainv = np.linalg.inv(A)
    dof = max(1, len(r0) - len(m))
    s2 = float(r0 @ r0) / dof
    Cm = s2 * Ainv @ (J.T @ J) @ Ainv
    sd = np.sqrt(np.clip(np.diag(Cm), 0, None))
    R = Ainv @ (J.T @ J)
    log("\nposterior standard deviations [km] (conditional on this "
        "parameterisation, not a full uncertainty):")
    for j, x in enumerate(model.x_nodes):
        log(f"  x={x:+7.1f} km   h = {m[j]:6.2f} +/- {sd[j]:4.2f}   "
            f"resolution diag = {R[j, j]:.3f}")
    np.savez(os.path.join(cfg['work_root'], 'gn_posterior.npz'),
             m=m, Cm=Cm, R=R, sd=sd, x_nodes=model.x_nodes, sigma2=s2)
    log(f"saved {cfg['work_root']}/gn_posterior.npz")
    return m, phi, hist


def run_gradient(fwd, model, cfg, m0, log):
    m = np.array(m0, float)
    phi, recs = fwd(m, 'it00_base', keep=True)
    hist = [dict(it=0, phi=phi, m=m.tolist())]
    step = cfg['step0_km']
    for it in range(1, cfg['iters'] + 1):
        log(f"\n--- iteration {it} --- Phi = {phi:.5f}")
        g = staggered_gradient(fwd, model, m, recs, cfg['fd_delta_km'], it, log)
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            log('  zero gradient, stopping')
            break
        d = -g / gn
        ok = False
        for k in range(cfg['max_backtrack']):
            trial = m + step * d
            lo, hi = model.param_bounds
            trial[:model.n_nodes] = np.clip(trial[:model.n_nodes], lo, hi)
            p2, r2 = fwd(trial, f'it{it:02d}_ls{k}')
            if p2 < phi:
                m, phi, recs, ok = trial, p2, r2, True
                step *= 1.6
                break
            step *= 0.5
        if not ok:
            log(f'  line search failed at step {step:.3f} km, stopping')
            break
        h, _, _ = model.unpack(m, cfg['fit_vp'], cfg['fit_vpvs'])
        log(f"  Moho now {np.round(h, 1).tolist()}")
        hist.append(dict(it=it, phi=phi, m=m.tolist(), step_km=step))
        with open(os.path.join(cfg['work_root'], 'history.json'), 'w') as fh:
            json.dump(hist, fh, indent=2)
        if step < cfg['step_min_km']:
            log('  step below tolerance, converged')
            break
    return m, phi, hist


def run_scan(fwd, model, cfg, m0, lo, hi, dstep, log):
    """Uniform Moho shift sweep -- the cheapest possible reality check."""
    out = []
    for d in np.arange(lo, hi + 1e-9, dstep):
        m = np.array(m0, float)
        m[:model.n_nodes] += d
        phi, _ = fwd(m, f'scan_{d:+.2f}'.replace('.', 'p'))
        out.append((float(d), phi))
        log(f"  shift {d:+6.2f} km -> Phi {phi:.5f}")
    d, p = min(out, key=lambda t: t[1])
    log(f"\nbest uniform shift {d:+.2f} km, Phi {p:.5f}")
    if out[0][1] == p or out[-1][1] == p:
        log("  [WARN] the minimum is at an end of the sweep -- widen --scan")
    with open(os.path.join(cfg['work_root'], 'scan.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    m = np.array(m0, float)
    m[:model.n_nodes] += d
    return m, p, out


# ===========================================================================
# final synthetics + data-fit report
# ===========================================================================
def save_final(fwd, model, cfg, m, log, tag='final'):
    """
    Re-run the accepted model with the seismograms KEPT, convert every event to
    SAC through su2sac.py, and write the plots you need to judge the fit by eye.

    Outputs under <work_root>/<tag>/
        model/                    vp_model.su, vs_model.su, model.png
        synthetics/<event>/       fdfk2d_<event>_<station>.r/.z  + record section
        fit/<event>_<station>.png per-trace obs vs syn and the two cross-convolutions
        fit_summary.png           misfit vs bounce point, under the Moho profile
        record_section.png        all traces, observed black, synthetic red
        fit.csv                   per-trace numbers
    """
    import csv
    cfg = dict(cfg)
    cfg['keep_windows'] = True
    cfg['keep_all'] = True
    cfg['keep_sac'] = True
    fwd.cfg = cfg
    log(f"\nre-running the final model with seismograms kept ({tag}) ...")
    phi, recs = fwd(m, tag, keep=True)
    work = os.path.join(cfg['work_root'], tag)
    syn_root = ensure_dir(os.path.join(work, 'synthetics'))
    here = os.path.dirname(os.path.abspath(__file__))

    for ev in cfg['events']:
        rd = os.path.join(work, 'runs', ev)
        if not os.path.isdir(rd):
            continue
        cmd = [sys.executable, os.path.join(here, 'su2sac.py'),
               '--run-dir', rd, '--out-dir', os.path.join(syn_root, ev),
               '--sign-r', str(cfg['sign_r']), '--sign-z', str(cfg['sign_z']),
               '--plot']
        if cfg.get('grids_root'):
            cmd += ['--grids-root', cfg['grids_root']]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            log(f"  [warn] su2sac failed for {ev}:\n{r.stdout[-1500:]}{r.stderr[-800:]}")
        else:
            n = len([f for f in os.listdir(os.path.join(syn_root, ev))
                     if f.endswith('.r')])
            log(f"  {ev}: {n} SAC pairs -> {os.path.join(syn_root, ev)}")

    obs = {(t['event'], t['station']): t for t in fwd.data.traces}
    dt = cfg['dt_common']
    trel = np.arange(cfg['win'][0], cfg['win'][1] + 0.5 * dt, dt)
    recs = sorted(recs, key=lambda r: r['bounce_x'])

    with open(os.path.join(work, 'fit.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['event', 'station', 'bounce_x_km', 'E', 'weight',
                    'alpha', 'syn_pick_s'])
        for r in recs:
            w.writerow([r['event'], r['station'], f"{r['bounce_x']:.1f}",
                        f"{r['E']:.5f}", f"{r['weight']:.3f}",
                        f"{r.get('alpha', float('nan')):.4f}",
                        f"{r['syn_pick']:.3f}"])

    try:
        _plot_fit(work, recs, obs, trel, dt, model, cfg, m, phi, log)
    except Exception as exc:
        log(f"  [warn] plotting failed: {exc}")

    log(f"\nfinal synthetics and fit report in {work}")
    log(f"  worst-fitting traces:")
    for r in sorted(recs, key=lambda r: -r['E'])[:5]:
        log(f"    {r['event']}_{r['station']:6s}  x={r['bounce_x']:+7.1f} km  E={r['E']:.4f}")
    return phi, recs


def _plot_fit(work, recs, obs, trel, dt, model, cfg, m, phi, log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fitdir = ensure_dir(os.path.join(work, 'fit'))
    for r in recs:
        o = obs.get((r['event'], r['station']))
        if o is None or 'SR' not in r:
            continue
        lag = r.get('lag_axis')
        if lag is None:
            lag = (np.arange(len(r['c1'])) - (len(trel) - 1)) * dt
        fig, ax = plt.subplots(3, 1, figsize=(9, 8))
        oR = r.get('oR', o['R'])
        oZ = r.get('oZ', o['Z'])
        ax[0].plot(trel, oR, 'k', lw=1.1, label='obs R')
        ax[0].plot(trel, r['SR'], 'r', lw=1.1, label='syn R')
        ax[1].plot(trel, oZ, 'k', lw=1.1, label='obs Z')
        ax[1].plot(trel, r['SZ'], 'r', lw=1.1, label='syn Z')
        for a in ax[:2]:
            a.axvline(0, color='b', ls=':', lw=1)
            a.legend(fontsize=8, loc='upper left')
            a.set_xlabel('time relative to S [s]')
        ax[2].plot(lag, r['c1'], 'k', lw=1.1, label='obs R * syn Z')
        ax[2].plot(lag, r['c2'], 'r', lw=1.1, label='syn R * obs Z')
        ax[2].set_xlabel('cross-convolution lag [s]')
        ax[2].legend(fontsize=8, loc='upper left')
        ax[2].set_title(f"E = {r['E']:.4f}   lag = {r.get('lag_s', 0):+.2f} s   "
                        f"alpha = {r.get('alpha', float('nan')):.3f}")
        fig.suptitle(f"{r['event']}_{r['station']}   bounce x = {r['bounce_x']:+.1f} km")
        fig.tight_layout()
        fig.savefig(os.path.join(fitdir, f"{r['event']}_{r['station']}.png"), dpi=115)
        plt.close(fig)

    # record section, observed black over synthetic red, ordered by bounce point
    have = [r for r in recs if 'SR' in r and (r['event'], r['station']) in obs]
    if have:
        fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.28 * len(have))), sharey=True)
        for comp, ax, lab in (('R', axes[0], 'radial'), ('Z', axes[1], 'vertical')):
            for k, r in enumerate(have):
                o = obs[(r['event'], r['station'])]
                sy = r['SR'] if comp == 'R' else r['SZ']
                ax.plot(trel, 1.2 * o[comp] + k, 'k', lw=0.8)
                ax.plot(trel, 1.2 * sy + k, 'r', lw=0.8)
            ax.axvline(0, color='b', ls=':', lw=1)
            ax.set_title(f'{lab}  (black observed, red synthetic)')
            ax.set_xlabel('time relative to S [s]')
        axes[0].set_yticks(range(len(have)))
        axes[0].set_yticklabels([f"{r['station']} {r['bounce_x']:+.0f}km E={r['E']:.2f}"
                                 for r in have], fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(work, 'record_section.png'), dpi=130)
        plt.close(fig)

    # misfit against the model it came from
    h, _, _ = model.unpack(m, cfg['fit_vp'], cfg['fit_vpvs'])
    x, hx = model.moho_profile(h)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1.2]))
    ax[0].plot(x, hx, 'b-', lw=1.5)
    ax[0].plot(model.x_nodes, h, 'bo', ms=5)
    ax[0].axvline(-model.hw, color='0.6', ls='--')
    ax[0].axvline(model.hw, color='0.6', ls='--')
    ax[0].invert_yaxis()
    ax[0].set_ylabel('Moho depth [km]')
    ax[0].set_title(f'final model, Phi = {phi:.4f}, n = {len(recs)}')
    sc = ax[1].scatter([r['bounce_x'] for r in recs], [r['E'] for r in recs],
                       c=[r['weight'] for r in recs], cmap='viridis', s=28)
    ax[1].set_xlabel('SsPmp bounce point x [km]')
    ax[1].set_ylabel('cross-convolution misfit E')
    fig.colorbar(sc, ax=ax[1], label='weight')
    fig.tight_layout()
    fig.savefig(os.path.join(work, 'fit_summary.png'), dpi=130)
    plt.close(fig)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', default=None,
                    help='settings file with "key = value" lines holding the paths and '
                         'any tuning values you do not want to retype; every flag below '
                         'overrides it')
    ap.add_argument('--write-template', default=None, metavar='FILE',
                    help='write a commented settings template and exit')

    # ---- paths: no defaults, nothing baked into the source ----
    for name, helptext in [
            ('events-root', 'directory holding <event>/t8.dat, shift.dat, <sta>.r/.z'),
            ('tpmp', 'tpmp.xy'),
            ('work-root', 'where every evaluation is written'),
            ('fdfk2d-bin', 'the FDFK2D executable'),
            ('grids-root', 'optional: grid.jinv2024, only used to copy final synthetics'),
            ('ref-1d', 'optional: depth_km Vp Vs reference column'),
            ('background-dir', 'vp_model.su/vs_model.su from build_2d_model.py; '
                               'parameters perturb ITS Moho and m=0 reproduces it')]:
        ap.add_argument('--' + name, default=None, help=helptext)

    ap.add_argument('--prf', default=None,
                    help="GMT project string, e.g. '-C-88.0047/38.4484 -A120.5'; "
                         "omitted means fit it from tpmp.xy")
    ap.add_argument('--events', nargs='*', default=list(ALL_EVENTS))
    ap.add_argument('--nproc', type=int, default=1)

    ap.add_argument('--background-x0-km', type=float, default=None)
    ap.add_argument('--background-dx', type=float, default=None)
    ap.add_argument('--background-dz', type=float, default=None)
    ap.add_argument('--background-npml', type=int, default=None)
    ap.add_argument('--background-ns', type=int, default=None)
    ap.add_argument('--moho-vp', type=float, default=7500.0)
    ap.add_argument('--moho-method', default='gradient',
                    choices=['gradient', 'contour'])
    ap.add_argument('--no-edge-repair', action='store_true')
    ap.add_argument('--edge-frac', type=float, default=0.85)
    ap.add_argument('--dh-max-km', type=float, default=8.0)
    ap.add_argument('--h0-km', type=float, default=52.0)
    ap.add_argument('--node-spacing-km', type=float, default=40.0)
    ap.add_argument('--structure-halfwidth-km', type=float, default=152.0)
    ap.add_argument('--x0-km', type=float, default=-340.0)
    ap.add_argument('--x1-km', type=float, default=245.0)
    ap.add_argument('--dx', type=float, default=200.0)
    ap.add_argument('--dz', type=float, default=200.0)
    ap.add_argument('--Zkm', type=float, default=80.0)
    ap.add_argument('--npml', type=int, default=20)
    ap.add_argument('--norder', type=int, default=8)
    ap.add_argument('--f0', type=float, default=0.8)

    ap.add_argument('--freq', nargs=2, type=float, default=[0.05, 0.5])
    ap.add_argument('--freq-stages', nargs='*', type=float, default=[0.05, 0.2, 0.05, 0.35, 0.05, 0.5],
                    help='flattened low/high pairs for the multiscale schedule')
    ap.add_argument('--corners', type=int, default=2)
    ap.add_argument('--win', nargs=2, type=float, default=[-5.0, 15.0],
                    help='SsPmp follows S by 8.7-14.4 s here, so the window is after S')
    ap.add_argument('--taper', type=float, default=0.10)
    ap.add_argument('--dt-common', type=float, default=0.05)
    ap.add_argument('--res-dt', type=float, default=0.1)
    ap.add_argument('--pick-band', nargs=2, type=float, default=[0.05, 1.0])
    ap.add_argument('--repick', action='store_true',
                    help='re-pick the synthetic S every iteration instead of '
                         'holding the first pick (moving the Moho does not move '
                         'the direct S, so the default is the smoother choice)')
    ap.add_argument('--norm', default='peak',
                    choices=['peak', 'peakz', 'energy', 'none'])
    ap.add_argument('--cc-win', nargs=2, type=float, default=[-5.0, 15.0],
                    help='second cut on the cross-convolutions, tau = 0 at the S arrival')
    ap.add_argument('--no-cc-win', action='store_true')
    ap.add_argument('--cc-taper', type=float, default=0.10)
    ap.add_argument('--syn-prefix', default='fdfk2d_')
    ap.add_argument('--keep-sac', action='store_true',
                    help='keep every iteration\'s SAC folders, not just the final one')
    ap.add_argument('--sign-r', type=float, default=-1.0)
    ap.add_argument('--sign-z', type=float, default=-1.0)

    ap.add_argument('--fit-vp', action='store_true')
    ap.add_argument('--fit-vpvs', action='store_true')
    ap.add_argument('--prior-moho', nargs='*', type=float, default=None)
    ap.add_argument('--prior-sigma-km', type=float, default=3.0)
    ap.add_argument('--lambda-smooth', type=float, default=0.002)
    ap.add_argument('--lambda-prior', type=float, default=0.0)
    ap.add_argument('--lambda-smooth-gn', type=float, default=0.5)
    ap.add_argument('--mu-damp', type=float, default=0.05)
    ap.add_argument('--h-min', type=float, default=35.0)
    ap.add_argument('--h-max', type=float, default=65.0)

    ap.add_argument('--iters', type=int, default=15)
    ap.add_argument('--step0-km', type=float, default=1.5)
    ap.add_argument('--step-min-km', type=float, default=0.15)
    ap.add_argument('--max-backtrack', type=int, default=4)
    ap.add_argument('--fd-delta-km', type=float, default=1.0)
    ap.add_argument('--fd-delta-vp', type=float, default=100.0)
    ap.add_argument('--fd-delta-vpvs', type=float, default=0.02)
    ap.add_argument('--gn-iters', type=int, default=5)
    ap.add_argument('--gn-step-scales', nargs='*', type=float, default=[1.0, 0.5, 0.25])
    ap.add_argument('--gn-tol-km', type=float, default=0.1)
    ap.add_argument('--keep-all', action='store_true')
    ap.add_argument('--no-plot-models', action='store_true')

    ap.add_argument('--method', default='gn', choices=['forward', 'scan', 'gradient', 'gn'])
    ap.add_argument('--scan', nargs=3, type=float, default=[-6.0, 6.0, 1.5],
                    metavar=('LO', 'HI', 'STEP'))
    ap.add_argument('--save-final', action='store_true',
                    help='after the search, re-run the accepted model with the '
                         'seismograms kept, convert to SAC and write the fit report')
    ap.add_argument('--m-from', default=None,
                    help='JSON with an "m" list (misfit.json or gn_history.json); use '
                         'with --method forward --save-final to regenerate synthetics')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the data count and the cost estimate, then stop')
    args = ap.parse_args()

    if args.write_template:
        with open(args.write_template, 'w') as fh:
            fh.write(SETTINGS_TEMPLATE)
        print(f'wrote {args.write_template} -- fill in the paths, then:\n'
              f'  python3 wv_invert.py --paths {args.write_template} '
              f'--method gn --dry-run')
        return

    if args.paths:
        apply_kv(args, read_kv(args.paths), parser=ap,
                 explicit=explicit_dests(ap))
    require_paths(args, ['events_root', 'tpmp', 'work_root', 'fdfk2d_bin'])

    stages = None
    if args.freq_stages:
        f = list(args.freq_stages)
        if len(f) % 2:
            raise SystemExit('--freq-stages needs an even number of values (low high pairs)')
        stages = [f[i:i + 2] for i in range(0, len(f), 2)]

    cfg = dict(
        events_root=args.events_root, tpmp=args.tpmp, work_root=args.work_root,
        fdfk2d_bin=args.fdfk2d_bin, grids_root=args.grids_root, ref_1d=args.ref_1d,
        background_dir=args.background_dir, background_x0_km=args.background_x0_km,
        background_dx=args.background_dx, background_dz=args.background_dz,
        background_npml=args.background_npml, background_ns=args.background_ns,
        moho_vp=args.moho_vp, dh_max_km=args.dh_max_km,
        prf=args.prf, events=list(args.events), nproc=args.nproc,
        h0_km=args.h0_km, node_spacing_km=args.node_spacing_km,
        structure_halfwidth_km=args.structure_halfwidth_km,
        x0_km=args.x0_km, x1_km=args.x1_km, dx=args.dx, dz=args.dz,
        Zkm=args.Zkm, npml=args.npml, norder=args.norder, f0=args.f0,
        freq=list(args.freq), freq_stages=stages, corners=args.corners,
        win=list(args.win), taper=args.taper, dt_common=args.dt_common,
        res_dt=args.res_dt, pick_band=list(args.pick_band), repick=args.repick,
        norm=args.norm, cc_win=(None if args.no_cc_win else list(args.cc_win)),
        cc_taper=args.cc_taper, syn_prefix=args.syn_prefix,
        keep_sac=args.keep_sac, sign_r=args.sign_r, sign_z=args.sign_z,
        fit_vp=args.fit_vp, fit_vpvs=args.fit_vpvs,
        prior_moho=args.prior_moho, prior_sigma_km=args.prior_sigma_km,
        lambda_smooth=args.lambda_smooth, lambda_prior=args.lambda_prior,
        lambda_smooth_gn=args.lambda_smooth_gn, mu_damp=args.mu_damp,
        h_min=args.h_min, h_max=args.h_max,
        iters=args.iters, step0_km=args.step0_km, step_min_km=args.step_min_km,
        max_backtrack=args.max_backtrack, fd_delta_km=args.fd_delta_km,
        fd_delta_vp=args.fd_delta_vp, fd_delta_vpvs=args.fd_delta_vpvs,
        gn_iters=args.gn_iters, gn_step_scales=list(args.gn_step_scales),
        gn_tol_km=args.gn_tol_km, keep_all=args.keep_all,
        plot_models=not args.no_plot_models, keep_windows=False)

    ensure_dir(cfg['work_root'])
    logf = open(os.path.join(cfg['work_root'], 'invert.log'), 'a')

    def log(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    geom = (ProfileGeom.from_prf(cfg['prf']) if cfg.get('prf')
            else ProfileGeom.from_tpmp(cfg['tpmp']))
    log(f"\n===== {time.strftime('%Y-%m-%d %H:%M')} =====")
    log('command: ' + ' '.join(sys.argv))
    log(str(geom))

    ref = load_ref_from_file(cfg['ref_1d']) if cfg.get('ref_1d') else None
    bg = None
    if cfg.get('background_dir'):
        bg = Background(cfg['background_dir'],
                        cfg['background_x0_km'] if cfg.get('background_x0_km')
                        is not None else cfg['x0_km'],
                        cfg.get('background_dx') or cfg['dx'],
                        cfg.get('background_dz') or cfg['dz'],
                        cfg['background_npml'] if cfg.get('background_npml')
                        is not None else cfg['npml'],
                        ns=cfg.get('background_ns'), vp_thresh=cfg['moho_vp'],
                        moho_method=cfg.get('moho_method', 'gradient'),
                        repair=cfg.get('edge_repair', True),
                        edge_frac=cfg.get('edge_frac', 0.85), log=log)
        log(str(bg))
    nodes = np.arange(-cfg['structure_halfwidth_km'],
                      cfg['structure_halfwidth_km'] + 1e-6, cfg['node_spacing_km'])
    model = Model2D(ref, cfg['h0_km'], nodes, cfg['dx'], cfg['dz'],
                    cfg['x0_km'], cfg['x1_km'], cfg['Zkm'], cfg['npml'],
                    cfg['structure_halfwidth_km'], bg, cfg['dh_max_km'])
    log(f"model {model.nx} x {model.nz} columns, {model.n_nodes} Moho nodes at "
        f"{np.round(model.x_nodes).astype(int).tolist()} km")
    if model.is_background:
        log("  parameters are Moho PERTURBATIONS dh; m = 0 is your interpolated model")
    else:
        log("  [WARN] no --background-dir: starting from a FLAT "
            f"{cfg['h0_km']:g} km Moho on a generic 1-D column. Point "
            "--background-dir at the build_2d_model.py output instead.")

    log('\nbuilding the dataset:')
    data = Dataset(cfg, geom, log)
    if not data.traces:
        raise SystemExit('no usable traces')

    groups = node_traces(model, [dict(bounce_x=t['bounce_x']) for t in data.traces])
    log('  traces per node: ' +
        ', '.join(f"{int(x)}km:{len(groups[j])}" for j, x in enumerate(model.x_nodes)))
    thin = [int(model.x_nodes[j]) for j in groups if len(groups[j]) < 3]
    if thin:
        log(f"  [WARN] nodes with <3 traces are effectively unconstrained: {thin} km. "
            f"Widen --node-spacing-km, or set --prior-moho with --lambda-prior.")

    per_eval = 25.0 * max(1, np.ceil(len(cfg['events']) / cfg['nproc']))
    n_stage = len(cfg.get('freq_stages') or [cfg['freq']])
    n_eval = {'forward': 1,
              'scan': int(np.floor((args.scan[1] - args.scan[0]) / args.scan[2])) + 1,
              'gradient': 1 + cfg['iters'] * 4,
              'gn': n_stage * (1 + cfg['gn_iters'] * 3) + 2}[args.method]
    if args.save_final:
        n_eval += 1
    log(f"\ncost estimate: {n_eval} forward evaluations x ~{per_eval:.0f} min "
        f"= ~{n_eval * per_eval / 60:.1f} h wall clock on {cfg['nproc']} cores")
    if args.dry_run:
        log('dry run, stopping here')
        return

    fwd = Forward(cfg, model, data, geom, log)
    m0 = model.start_vector(cfg['fit_vp'], cfg['fit_vpvs'])
    if cfg.get('prior_moho'):
        m0[:model.n_nodes] = np.asarray(cfg['prior_moho'], float)
    if args.m_from:
        d = json.load(open(args.m_from))
        m0 = np.array((d[-1] if isinstance(d, list) else d)['m'], float)
        log(f"starting model taken from {args.m_from}: "
            f"{np.round(m0[:model.n_nodes], 2).tolist()}")

    if args.method == 'forward':
        if args.save_final:
            save_final(fwd, model, cfg, m0, log)
        else:
            fwd(m0, 'forward', keep=True)
    elif args.method == 'scan':
        m, phi, _ = run_scan(fwd, model, cfg, m0, *args.scan, log=log)
        if args.save_final:
            save_final(fwd, model, cfg, m, log)
    else:
        runner = run_gauss_newton if args.method == 'gn' else run_gradient
        m, phi, hist = runner(fwd, model, cfg, m0, log)
        h, dvp, dvpvs = model.unpack(m, cfg['fit_vp'], cfg['fit_vpvs'])
        log(f"\nfinal Phi {phi:.5f}")
        log(f"final Moho [km]: {np.round(h, 2).tolist()}")
        if cfg['fit_vp']:
            log(f"crustal dVp {dvp:+.1f} m/s")
        model.write(os.path.join(cfg['work_root'], 'final_model'), m,
                    cfg['fit_vp'], cfg['fit_vpvs'], plot=True)
        log(f"final model written to {cfg['work_root']}/final_model")
        if args.save_final:
            save_final(fwd, model, cfg, m, log)


if __name__ == '__main__':
    main()
