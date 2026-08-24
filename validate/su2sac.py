#!/usr/bin/env python3
"""
su2sac.py
=========
Convert the SU seismograms written by FDFK2D into SAC files that carry the same
metadata as the observed data, so that everything downstream (cross-convolution,
plotting, jinv2024 bookkeeping) can treat synthetic and observed identically.

This step does nothing but format conversion, polarity, headers and naming.
No filtering, no windowing, no normalisation -- those belong in sac_crossconv.py
so that the cross-convolution code stays usable with ANY pair of SAC files
(FDFK2D, CPS `syn.*`, or anything else).

Output naming follows the convention already in use:

    observed        <event>_<station>.r / .z
    CPS synthetic   syn.<event>_<station>.r / .z
    FDFK2D          fdfk2d_<event>_<station>.r / .z

Polarity
--------
Two sign conventions have to be reconciled:

  * z:  the FD grid has z increasing DOWNWARD (Z0 = 0 at the free surface,
        Zn at depth), so seisz is positive down.  SAC/CPS vertical is positive
        UP  ->  default --sign-z -1.
  * x:  FDFK2D's +x axis is the profile direction; before writing it multiplies
        the horizontal by cos(baz - az_org).  The SAC radial points FROM the
        event TO the station, i.e. opposite to the horizontal projection of the
        propagation direction  ->  default --sign-r -1.

Both are exposed as flags because they depend on the FDFK2D revision you built.
Verify them once with --qc against a CPS synthetic for the same station and then
leave them alone; the QC printout reports the sign of the direct-S peak on both.

Usage
-----
  python3 su2sac.py \
      --run-dir    /home/yaoj/C++/FDFK2D/runs/20150323045138 \
      --out-dir    /home/yaoj/C++/FDFK2D/runs/20150323045138/sac \
      --grids-root /home/yaoj/C++/grid.jinv2024 \
      --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdfk2d_common import (read_su, read_obs_tw, pick_max_envelope, bandpass,
                           ensure_dir, read_kv, apply_kv, explicit_dests, require_paths)


# ---------------------------------------------------------------------------
def write_sac(path, data, dt, hdr):
    from obspy.io.sac import SACTrace
    sac = SACTrace(data=np.asarray(data, dtype=np.float32), delta=float(dt),
                   b=float(hdr.get('b', 0.0)))
    for k, v in hdr.items():
        if k == 'b' or v is None:
            continue
        try:
            setattr(sac, k, v)
        except Exception:
            pass
    sac.lcalda = False
    sac.write(path)


def pick_arrival(r, z, dt, mode, band, window, guard):
    """
    Pick the direct-S arrival in the synthetic.

    'env'  : maximum of the analytic envelope of the radial (default).  For an
             S-wave source the direct S dominates the radial, and SsPmp -- which
             arrives ~10 s EARLIER -- is a secondary phase, so the envelope
             maximum is the direct S.
    'envz' : same on the vertical.
    'first': first sample exceeding a fraction of the peak (earlier, noisier).
    """
    x = r if mode in ('env', 'first') else z
    xf = bandpass(x, dt, *band) if band else x
    t, i = pick_max_envelope(xf, dt, b=0.0, t_search=window, smooth_s=1.0)
    if mode == 'first':
        env = np.abs(xf)
        thr = 0.15 * env[i]
        j = i
        while j > 0 and env[j] > thr:
            j -= 1
        t, i = j * dt, j
    n = len(x)
    trunc = (t < guard) or (t > (n - 1) * dt - guard)
    return float(t), int(i), bool(trunc)


# ---------------------------------------------------------------------------
def convert_run(run_dir, out_dir, seis_dir=None, prefix='seis', name_prefix='fdfk2d_',
                sign_r=-1.0, sign_z=-1.0, pick='env', pick_band=(0.05, 1.0),
                pick_window=None, edge_guard=8.0, prev_picks=None, log=print):
    """
    SU -> SAC for one event run directory.  Returns (rows, dt, meta).

    This is the function the inversion calls after every iteration, so the
    synthetics it cross-convolves are the same SAC files you can open in SAC
    yourself -- there is no separate in-memory shortcut that could drift away
    from what the standalone tool does.

    `prev_picks` is a {station: t} dict; supplying it reuses the direct-S pick
    from an earlier iteration instead of re-picking.  Moving the Moho does not
    move the direct S, and the cross-convolution is blind to a shift common to
    both synthetic components anyway, so holding the pick fixed keeps the
    objective function smooth.
    """
    seis_dir = seis_dir or os.path.join(run_dir, 'seismograms')
    out_dir = ensure_dir(out_dir)
    with open(os.path.join(run_dir, 'meta.json')) as fh:
        meta = json.load(fh)
    ev = meta['event']
    stations = meta['stations']
    n_sta = meta['n_station_receivers']

    ux, dt = read_su(os.path.join(seis_dir, prefix + 'x.su'))
    uz, dt_z = read_su(os.path.join(seis_dir, prefix + 'z.su'))
    if abs(dt - dt_z) > 1e-9:
        raise SystemExit(f'dt mismatch between {prefix}x.su and {prefix}z.su')
    if ux.shape[0] < n_sta:
        raise SystemExit(f'{ux.shape[0]} traces but meta.json expects {n_sta}')

    rows, trunc_n = [], 0
    for i, st in enumerate(stations[:n_sta]):
        r = sign_r * ux[i]
        z = sign_z * uz[i]
        sta = st['station']
        if prev_picks and sta in prev_picks:
            t_a, trunc = float(prev_picks[sta]), False
        elif pick == 'none':
            t_a, trunc = np.nan, False
        else:
            t_a, _, trunc = pick_arrival(r, z, dt, pick, pick_band,
                                         pick_window, edge_guard)
        trunc_n += int(trunc)

        base = dict(kstnm=sta[:8], kevnm=ev[:16],
                    stla=st['stla'], stlo=st['stlo'], stel=st['stel'],
                    evla=meta['evla'], evlo=meta['evlo'],
                    evdp=meta['evdp'] if meta['evdp'] is not None else 0.0,
                    baz=st['baz'], gcarc=st['gcarc'],
                    user0=st['p_skm'], kuser0='p_skm',
                    user1=meta['theta_true'], kuser1='theta', b=0.0, o=None)
        if np.isfinite(t_a):
            base.update(a=t_a, ka='S')

        tag = f'{name_prefix}{ev}_{sta}'
        pr = os.path.join(out_dir, tag + '.r')
        pz = os.path.join(out_dir, tag + '.z')
        write_sac(pr, r, dt, dict(base, kcmpnm='R'))
        write_sac(pz, z, dt, dict(base, kcmpnm='Z'))
        rows.append(dict(tag=f'{ev}_{sta}', station=sta, a=t_a, p_skm=st['p_skm'],
                         x_km=st['x_km'], rx_m=st['rx_m'], trunc=trunc,
                         peak_r=float(r[int(np.argmax(np.abs(r)))]),
                         peak_z=float(z[int(np.argmax(np.abs(z)))]),
                         path_r=pr, path_z=pz))
    if trunc_n and log:
        log(f'  [WARN] {trunc_n} picks within {edge_guard:g} s of a trace end in {ev}')

    with open(os.path.join(out_dir, 'syn.tw'), 'w') as fh:
        for r in rows:
            fh.write(f"{r['tag']} {r['p_skm']:.6f} {r['a']:.4f} 1.000\n")
    return rows, dt, meta


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', default=None,
                    help='settings file with "key = value" lines; command-line flags win')
    ap.add_argument('--run-dir', default=None,
                    help='event run directory produced by fdfk2d_setup.py')
    ap.add_argument('--seis-dir', default=None, help='default <run-dir>/seismograms')
    ap.add_argument('--prefix', default='seis', help='FDFK2D output prefix (default seis)')
    ap.add_argument('--out-dir', default=None, help='default <run-dir>/sac')
    ap.add_argument('--name-prefix', default='fdfk2d_')
    ap.add_argument('--grids-root', default=None,
                    help='if given, also copy each trace into every grid directory '
                         'whose obs.tw lists it')
    ap.add_argument('--sign-r', type=float, default=-1.0)
    ap.add_argument('--sign-z', type=float, default=-1.0)
    ap.add_argument('--no-flip', action='store_true',
                    help='shorthand for --sign-r 1 --sign-z 1')
    ap.add_argument('--pick', default='env', choices=['env', 'envz', 'first', 'none'])
    ap.add_argument('--pick-band', nargs=2, type=float, default=[0.05, 1.0],
                    help='band used ONLY for picking, not for the written data')
    ap.add_argument('--pick-window', nargs=2, type=float, default=None,
                    help='restrict the pick search to this absolute time window [s]')
    ap.add_argument('--edge-guard', type=float, default=8.0,
                    help='warn if the pick lands within this many seconds of a trace end')
    ap.add_argument('--dense', action='store_true',
                    help='also export the dense receiver line as dense_XXXX.r/.z')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--qc', default=None,
                    help='directory holding CPS synthetics (syn.<ev>_<sta>.z) to '
                         'cross-check polarity against')
    args = ap.parse_args()
    if args.paths:
        apply_kv(args, read_kv(args.paths), parser=ap,
                 explicit=explicit_dests(ap))
    require_paths(args, ['run_dir'])

    if args.no_flip:
        args.sign_r = args.sign_z = 1.0

    run_dir = args.run_dir
    out_dir = ensure_dir(args.out_dir or os.path.join(run_dir, 'sac'))
    rows, dt, meta = convert_run(
        run_dir, out_dir, args.seis_dir, args.prefix, args.name_prefix,
        args.sign_r, args.sign_z, args.pick, tuple(args.pick_band),
        args.pick_window, args.edge_guard)
    ev = meta['event']
    n_sta = meta['n_station_receivers']
    seis_dir = args.seis_dir or os.path.join(run_dir, 'seismograms')
    ux, _ = read_su(os.path.join(seis_dir, args.prefix + 'x.su'))
    uz, _ = read_su(os.path.join(seis_dir, args.prefix + 'z.su'))
    ntr, ns = ux.shape
    print(f"{ev}: {ntr} traces x {ns} samples, dt = {dt:g} s "
          f"(meta says tstp = {meta['tstp']:g} s)")
    for r in rows:
        print(f"  {r['station']:6s} x={r['x_km']:8.2f} km  rx={r['rx_m']/1000:7.2f} km  "
              f"S={r['a']:8.3f} s  peakR={r['peak_r']:+.3e} peakZ={r['peak_z']:+.3e}"
              f"{'  <-- pick near trace end' if r['trunc'] else ''}")
    print(f"  wrote {len(rows)} trace pairs and syn.tw to {out_dir}")

    if args.dense and ntr > n_sta:
        dd = ensure_dir(os.path.join(out_dir, 'dense'))
        x0 = meta['X0']
        step = meta['dense_dx_km'] * 1000.0
        for j in range(n_sta, ntr):
            k = j - n_sta
            rx = x0 + k * step
            write_sac(os.path.join(dd, f'dense_{k:05d}.r'), args.sign_r * ux[j], dt,
                      dict(kstnm=f'D{k:05d}', kcmpnm='R', b=0.0, dist=rx / 1000.0))
            write_sac(os.path.join(dd, f'dense_{k:05d}.z'), args.sign_z * uz[j], dt,
                      dict(kstnm=f'D{k:05d}', kcmpnm='Z', b=0.0, dist=rx / 1000.0))
        print(f"  wrote {ntr - n_sta} dense-line trace pairs to {dd}")

    # --- optional polarity cross-check against CPS --------------------------
    if args.qc:
        from obspy import read
        print("  polarity cross-check against CPS synthetics:")
        for r in rows[:10]:
            q = os.path.join(args.qc, f"syn.{r['tag']}.z")
            if not os.path.isfile(q):
                continue
            tr = read(q, format='SAC')[0]
            s = tr.data[int(np.argmax(np.abs(tr.data)))]
            print(f"    {r['station']:6s} CPS peakZ={s:+.3e}   FDFK2D peakZ={r['peak_z']:+.3e}"
                  f"   {'SAME' if s * r['peak_z'] > 0 else 'OPPOSITE'} sign")

    # --- distribute into grid directories -----------------------------------
    if args.grids_root:
        import shutil
        n_cp = 0
        for g in sorted(os.listdir(args.grids_root)):
            gd = os.path.join(args.grids_root, g)
            otw = os.path.join(gd, 'obs.tw')
            if not os.path.isdir(gd) or not os.path.isfile(otw):
                continue
            want = {o['tag'] for o in read_obs_tw(otw) if o['event'] == ev}
            for r in rows:
                if r['tag'] in want:
                    shutil.copy(r['path_r'], os.path.join(gd, os.path.basename(r['path_r'])))
                    shutil.copy(r['path_z'], os.path.join(gd, os.path.basename(r['path_z'])))
                    n_cp += 1
        print(f"  copied {n_cp} trace pairs into grid directories under {args.grids_root}")

    # --- record section -----------------------------------------------------
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = np.arange(ns) * dt
        fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
        order = np.argsort([r['x_km'] for r in rows])
        for comp, ax, sign, lab in ((ux, axes[0], args.sign_r, 'radial'),
                                    (uz, axes[1], args.sign_z, 'vertical')):
            for k, idx in enumerate(order):
                d = sign * comp[idx]
                m = np.abs(d).max() or 1.0
                ax.plot(t, 1.4 * d / m + k, lw=0.6, color='k')
                if np.isfinite(rows[idx]['a']):
                    ax.plot(rows[idx]['a'], k, 'r|', ms=10)
            ax.set_title(f"{ev}  {lab}  (red = picked S)")
            ax.set_xlabel('time since simulation start [s]')
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels([f"{rows[i]['station']} {rows[i]['x_km']:.0f}km"
                                 for i in order], fontsize=6)
        fig.tight_layout()
        png = os.path.join(out_dir, f'{ev}_record_section.png')
        fig.savefig(png, dpi=130)
        print(f"  wrote {png}")


if __name__ == '__main__':
    main()
