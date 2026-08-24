#!/usr/bin/env python3
"""
sac_crossconv.py
================
Cross-convolution misfit between observed and synthetic three-component-derived
R/Z SAC pairs.  Completely independent of how the synthetics were made: point it
at any directory holding `<prefix><event>_<station>.r/.z` and it works.  Used for
both the CPS synthetics (`--syn-prefix syn.`) and the FDFK2D 2-D synthetics
(`--syn-prefix fdfk2d_`).

Method
------
The cross-convolution misfit (Menke & Levin, 2003; used for SsPmp by Liu & Zhu,
2021) removes the unknown source time function without deconvolution:

    c1(t) = R_obs(t) * Z_syn(t)
    c2(t) = R_syn(t) * Z_obs(t)

If the synthetic structure is correct, c1 and c2 are identical up to a scalar,
because both equal  S(t) * S(t) * [structure response]  with the same source
term.  The misfit is therefore a comparison of c1 against c2, not of waveforms
against waveforms, and it is blind to any amplitude scaling common to a station
pair -- which is why the normalisation below only ever divides R and Z by the
SAME number, preserving the R/Z ratio that actually carries the structural
information.

Processing chain (per trace pair)
---------------------------------
  demean -> detrend -> 5% cosine taper -> zero-phase Butterworth bandpass
  -> resample to a common dt -> cut a window referenced to each record's OWN
  S arrival -> cosine taper the window -> normalise -> convolve -> compare.

S arrival sources
-----------------
  observed  : column 3 of obs.tw, i.e. the hand-refined pick in shift.dat.
              (t8.dat column 2 is deliberately NOT used.)
  synthetic : SAC header `a` (written by su2sac.py), or column 3 of a syn.tw.

Reported per trace
------------------
  alpha  optimal scalar minimising ||c1 - alpha*c2||
  VR     variance reduction, 1 - ||c1-alpha*c2||^2 / ||c1||^2   (1 = perfect)
  E      shape-only misfit ||c1/||c1|| - c2/||c2||||^2          (0 = perfect, 2 = orthogonal)
  ccmax  peak normalised cross-correlation between c1 and c2
  lag    the lag at which ccmax occurs; a systematic non-zero lag means the
         synthetic Moho is at the wrong depth (or the picks are inconsistent)

Usage
-----
  # FDFK2D synthetics already copied into the grid directory
  python3 sac_crossconv.py --grid-dir /home/yaoj/C++/grid.jinv2024/grid-1 \
      --syn-prefix fdfk2d_ --freq 0.05 0.5 --win -25 5 --plot

  # same grid, CPS synthetics, for a like-for-like comparison
  python3 sac_crossconv.py --grid-dir /home/yaoj/C++/grid.jinv2024/grid-1 \
      --syn-prefix syn. --freq 0.05 0.5 --win -25 5 --plot

  # every grid at once
  python3 sac_crossconv.py --grids-root /home/yaoj/C++/grid.jinv2024 \
      --syn-prefix fdfk2d_ --freq 0.05 0.5 --win -25 5 --csv all_grids.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdfk2d_common import (bandpass, cos_taper, read_obs_tw, ensure_dir,
                           read_kv, apply_kv, explicit_dests, prep_window)


# ---------------------------------------------------------------------------
def load_sac(path):
    from obspy import read
    tr = read(path, format='SAC')[0]
    s = tr.stats.sac
    return dict(data=tr.data.astype(np.float64),
                dt=float(tr.stats.delta),
                b=float(s.get('b', 0.0)),
                o=(float(s['o']) if 'o' in s and s['o'] is not None else None),
                a=(float(s['a']) if 'a' in s and s['a'] is not None else None),
                npts=int(tr.stats.npts))


def resolve_arrival(tr, t_pick, label, path):
    """
    The picks in shift.dat / t8.dat are plain numbers; decide whether they live
    on the trace's own time axis (b .. b+(npts-1)*dt) or are measured from the
    origin time.  Fail loudly rather than silently mis-aligning by ~1000 s.
    """
    t0, t1 = tr['b'], tr['b'] + (tr['npts'] - 1) * tr['dt']
    if t0 <= t_pick <= t1:
        return t_pick, 'trace'
    if tr['o'] is not None and t0 <= tr['o'] + t_pick <= t1:
        return tr['o'] + t_pick, 'origin'
    raise ValueError(f"{label} arrival {t_pick:g} s does not fall inside {os.path.basename(path)} "
                     f"[{t0:.2f}, {t1:.2f}] s, with or without o={tr['o']}")


def prep(tr, t_arr, dt_out, win, band, corners, taper_frac):
    """Thin wrapper so the file-based and array-based paths share prep_window."""
    w, t_rel = prep_window(tr['data'], tr['dt'], tr['b'], t_arr, win, dt_out,
                           band, corners, taper_frac)
    if not np.any(np.abs(w) > 0):
        raise ValueError('window falls entirely outside the trace')
    return w, t_rel


def normalise_pair(r, z, mode):
    if mode == 'none':
        return r, z, 1.0
    if mode == 'peak':
        s = max(np.abs(r).max(), np.abs(z).max())
    elif mode == 'energy':
        s = np.sqrt(np.sum(r ** 2 + z ** 2))
    elif mode == 'peakz':
        s = np.abs(z).max()
    else:
        raise ValueError(mode)
    s = s if s > 0 else 1.0
    return r / s, z / s, float(s)


def cc_time_axis(n_c, dt, win):
    """
    Time axis of a cross-convolution of two windows that each run from win[0]
    to win[1] relative to their own S arrival.

    Convolution index j sums contributions with t1 + t2 = 2*win[0] + j*dt, so
    the natural axis is

        tau = 2*win[0] + j*dt,     j = 0 .. 2N-2

    and tau = 0 is where the two direct-S arrivals coincide.  A term pairing S
    in one trace with SsPmp in the other lands at tau = the SsPmp-S delay,
    which is what the misfit is actually comparing.

    Centring on j = N-1 instead -- the usual "lag" convention -- only puts S*S
    at zero when the window is symmetric.  For win = (-25, 5) it displaces the
    axis by win[0] + win[1] = -20 s.
    """
    return 2.0 * win[0] + np.arange(n_c) * dt


def cut_taper_cc(c, dt, win, cc_win, taper_frac):
    """
    Second cut + taper, applied to the cross-convolution on the tau axis above.

    Linear convolution of two N-sample windows gives 2N-1 samples spanning
    2*win[0] .. 2*win[1], and the outer ends are built from tapered edges
    overlapping almost nothing.  Cutting back to cc_win keeps only the delays
    that carry signal, with tau = 0 pinned to the S arrival.

    Returns (windowed cross-convolution, tau axis).
    """
    tau = cc_time_axis(len(c), dt, win)
    if cc_win is None:
        return c, tau
    m = (tau >= cc_win[0]) & (tau <= cc_win[1])
    if not m.any():
        raise ValueError(f'cross-convolution window {cc_win} s lies outside the '
                         f'available range {tau[0]:.1f}..{tau[-1]:.1f} s')
    return c[m] * cos_taper(int(m.sum()), taper_frac), tau[m]


def crossconv_pair(oR, oZ, sR, sZ, dt, win=(-5.0, 15.0), norm='peak',
                   cc_win=(-5.0, 15.0), cc_taper=0.10):
    """
    Cross-convolve one already-windowed observed/synthetic pair.

        c1 = R_obs * Z_syn      c2 = R_syn * Z_obs

    R and Z of a pair are divided by the SAME scalar, so the R/Z ratio that
    carries the structure survives; then the second cut+taper above; then the
    shape comparison.  Used by both the standalone tool and the inversion, so
    there is exactly one definition of the misfit.
    """
    oR, oZ, sc_o = normalise_pair(oR, oZ, norm)
    sR, sZ, sc_s = normalise_pair(sR, sZ, norm)
    c1, lag = cut_taper_cc(np.convolve(oR, sZ), dt, win, cc_win, cc_taper)
    c2, _ = cut_taper_cc(np.convolve(sR, oZ), dt, win, cc_win, cc_taper)
    m = compare(c1, c2)
    m['lag_s'] = m['lag_idx'] * dt
    n1 = np.linalg.norm(c1) or 1.0
    n2 = np.linalg.norm(c2) or 1.0
    m.update(c1=c1 / n1, c2=c2 / n2, lag_axis=lag, tau=lag,
             oR=oR, oZ=oZ, sR=sR, sZ=sZ, obs_scale=sc_o, syn_scale=sc_s,
             residual=c1 / n1 - c2 / n2)
    return m


def compare(c1, c2):
    n1 = np.linalg.norm(c1)
    n2 = np.linalg.norm(c2)
    if n1 == 0 or n2 == 0:
        return dict(alpha=np.nan, vr=np.nan, E=np.nan, ccmax=np.nan, lag_idx=0)
    alpha = float(np.dot(c1, c2) / np.dot(c2, c2))
    vr = float(1.0 - np.sum((c1 - alpha * c2) ** 2) / np.sum(c1 ** 2))
    E = float(np.sum((c1 / n1 - c2 / n2) ** 2))
    xc = np.correlate(c1 / n1, c2 / n2, mode='full')
    k = int(np.argmax(np.abs(xc)))
    return dict(alpha=alpha, vr=vr, E=E, ccmax=float(xc[k]),
                lag_idx=k - (len(c2) - 1))


# ---------------------------------------------------------------------------
def process_grid(grid_dir, args, writer, log):
    otw = os.path.join(grid_dir, 'obs.tw')
    if not os.path.isfile(otw):
        log(f"  no obs.tw in {grid_dir}, skipped")
        return []
    obs_rows = read_obs_tw(otw)

    syn_dir = args.syn_dir or grid_dir
    syn_tw = {}
    if args.syn_tw:
        for r in read_obs_tw(args.syn_tw):
            syn_tw[r['tag']] = r['t_s']

    results = []
    for o in obs_rows:
        if args.events and o['event'] not in args.events:
            continue
        if args.stations and o['station'] not in args.stations:
            continue
        po_r = os.path.join(grid_dir, f"{o['tag']}.r")
        po_z = os.path.join(grid_dir, f"{o['tag']}.z")
        ps_r = os.path.join(syn_dir, f"{args.syn_prefix}{o['tag']}.r")
        ps_z = os.path.join(syn_dir, f"{args.syn_prefix}{o['tag']}.z")
        missing = [p for p in (po_r, po_z, ps_r, ps_z) if not os.path.isfile(p)]
        if missing:
            log(f"  {o['tag']}: missing {', '.join(os.path.basename(m) for m in missing)}")
            continue

        try:
            OR, OZ = load_sac(po_r), load_sac(po_z)
            SR, SZ = load_sac(ps_r), load_sac(ps_z)

            t_obs, base = resolve_arrival(OZ, o['t_s'], 'observed S', po_z)
            if o['tag'] in syn_tw:
                t_syn, _ = resolve_arrival(SZ, syn_tw[o['tag']], 'synthetic S', ps_z)
            elif SZ['a'] is not None:
                t_syn = SZ['a']
            else:
                raise ValueError('no synthetic S arrival: set SAC header `a` '
                                 '(su2sac.py does this) or pass --syn-tw')

            dt_out = args.dt or max(OZ['dt'], SZ['dt'])
            band = tuple(args.freq) if args.freq else None
            or_, _ = prep(OR, t_obs, dt_out, args.win, band, args.corners, args.taper)
            oz_, _ = prep(OZ, t_obs, dt_out, args.win, band, args.corners, args.taper)
            sr_, _ = prep(SR, t_syn, dt_out, args.win, band, args.corners, args.taper)
            sz_, trel = prep(SZ, t_syn, dt_out, args.win, band, args.corners, args.taper)

            cc_win = None if args.cc_win is None else tuple(args.cc_win)
            m = crossconv_pair(or_, oz_, sr_, sz_, dt_out, tuple(args.win),
                               args.norm, cc_win, args.cc_taper)
            or_, oz_, sr_, sz_ = m['oR'], m['oZ'], m['sR'], m['sZ']
            sc_o, sc_s = m['obs_scale'], m['syn_scale']
            c1, c2 = m['c1'], m['c2']

            # weight from the pre-signal RMS noise measured by tw_script.py,
            # expressed in the same normalised units as the data
            sig = o['rms'] / sc_o if (np.isfinite(o['rms']) and o['rms'] > 0 and sc_o) else np.nan
            w = 1.0 / sig if np.isfinite(sig) and sig > 0 else 1.0

            rec = dict(grid=os.path.basename(grid_dir), tag=o['tag'],
                       event=o['event'], station=o['station'],
                       t_obs=t_obs, t_syn=t_syn, time_base=base,
                       dt=dt_out, npts_win=len(or_),
                       obs_scale=sc_o, syn_scale=sc_s, rms=o['rms'], weight=w,
                       **{k: m[k] for k in ('alpha', 'vr', 'E', 'ccmax', 'lag_s')})
            results.append(rec)
            if writer:
                writer.writerow(rec)
            log(f"  {o['tag']:28s} E={m['E']:.4f}  VR={m['vr']:+.3f}  "
                f"cc={m['ccmax']:+.3f}  lag={m['lag_s']:+.2f}s  alpha={m['alpha']:.3f}")

            if args.plot:
                plot_pair(args, grid_dir, o, trel, or_, oz_, sr_, sz_,
                          c1, c2, m, dt_out, m['lag_axis'])
        except Exception as exc:
            log(f"  {o['tag']}: {exc}")
            continue

    if results:
        E = np.array([r['E'] for r in results])
        W = np.array([r['weight'] for r in results])
        lag = np.array([r['lag_s'] for r in results])
        log(f"  --> {os.path.basename(grid_dir)}: n={len(results)}  "
            f"mean E={E.mean():.4f}  weighted E={np.sum(W*E)/np.sum(W):.4f}  "
            f"median lag={np.median(lag):+.2f} s")
        if abs(np.median(lag)) > 3 * results[0]['dt']:
            log(f"      (a systematic lag means the synthetic Moho depth or the S picks "
                f"are offset, not that the waveforms disagree in shape)")
    return results


def plot_pair(args, grid_dir, o, trel, or_, oz_, sr_, sz_, c1, c2, m, dt, lag=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = ensure_dir(args.plot_dir or os.path.join(grid_dir, 'crossconv'))
    if lag is None:
        lag = (np.arange(len(c1)) - (len(or_) - 1)) * dt
    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    ax[0].plot(trel, or_, 'k', lw=1, label='obs R')
    ax[0].plot(trel, sr_, 'r', lw=1, label='syn R')
    ax[1].plot(trel, oz_, 'k', lw=1, label='obs Z')
    ax[1].plot(trel, sz_, 'r', lw=1, label='syn Z')
    for a in ax[:2]:
        a.axvline(0, color='b', ls=':', lw=1)
        a.legend(fontsize=8, loc='upper left')
        a.set_xlabel('time relative to S [s]')
    ax[2].plot(lag, c1, 'k', lw=1, label='obs R * syn Z')
    ax[2].plot(lag, c2, 'r', lw=1, label='syn R * obs Z')
    ax[2].axvline(0, color='b', ls=':', lw=1)
    ax[2].set_xlabel('cross-convolution tau [s]   (0 = S arrival)')
    ax[2].legend(fontsize=8, loc='upper left')
    ax[2].set_title(f"E={m['E']:.4f}  VR={m['vr']:+.3f}  cc={m['ccmax']:+.3f}  "
                    f"lag={m['lag_s']:+.2f} s")
    fig.suptitle(f"{os.path.basename(grid_dir)}  {o['tag']}  {args.syn_prefix}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{args.syn_prefix.strip('._')}_{o['tag']}.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', default=None,
                    help='settings file with "key = value" lines; command-line flags win')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--grid-dir', default=None)
    g.add_argument('--grids-root', default=None)
    ap.add_argument('--syn-dir', default=None,
                    help='where the synthetics live (default: the grid directory)')
    ap.add_argument('--syn-prefix', default='fdfk2d_')
    ap.add_argument('--syn-tw', default=None,
                    help='syn.tw with synthetic S arrivals; otherwise SAC header `a` is used')
    ap.add_argument('--events', nargs='*', default=None)
    ap.add_argument('--stations', nargs='*', default=None)
    ap.add_argument('--freq', nargs=2, type=float, default=[0.05, 0.5],
                    help='bandpass corners in Hz, applied identically to obs and syn')
    ap.add_argument('--corners', type=int, default=2)
    ap.add_argument('--win', nargs=2, type=float, default=[-5.0, 15.0],
                    help='window relative to the S arrival. SsPmp is a free-surface '
                         'multiple so it FOLLOWS S, by 8.7-14.4 s in this dataset '
                         '(tpmp.xy column 5), which the default brackets')
    ap.add_argument('--taper', type=float, default=0.10)
    ap.add_argument('--dt', type=float, default=None,
                    help='common sample interval (default: the coarser of the two)')
    ap.add_argument('--norm', default='peak', choices=['peak', 'peakz', 'energy', 'none'],
                    help='R and Z of a pair are always divided by the SAME scalar')
    ap.add_argument('--cc-win', nargs=2, type=float, default=[-5.0, 15.0],
                    help='second cut applied to the cross-convolutions, on the tau '
                         'axis where tau = 0 is the S arrival; use --no-cc-win to '
                         'keep all 2N-1 samples')
    ap.add_argument('--no-cc-win', action='store_true')
    ap.add_argument('--cc-taper', type=float, default=0.10)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--plot-dir', default=None)
    args = ap.parse_args()
    if args.paths:
        apply_kv(args, read_kv(args.paths), parser=ap,
                 explicit=explicit_dests(ap))
    if args.no_cc_win:
        args.cc_win = None
    if not (args.grid_dir or args.grids_root):
        raise SystemExit('give --grid-dir or --grids-root (or set one in --paths)')

    def log(*a):
        print(*a, flush=True)

    fields = ['grid', 'tag', 'event', 'station', 't_obs', 't_syn', 'time_base', 'dt',
              'npts_win', 'obs_scale', 'syn_scale', 'rms', 'weight',
              'alpha', 'vr', 'E', 'ccmax', 'lag_s']
    fh = writer = None
    if args.csv:
        fh = open(args.csv, 'w', newline='')
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

    grids = ([args.grid_dir] if args.grid_dir else
             sorted(os.path.join(args.grids_root, d) for d in os.listdir(args.grids_root)
                    if os.path.isdir(os.path.join(args.grids_root, d))))
    allr = []
    for gd in grids:
        log(f"\n=== {gd} ===")
        allr += process_grid(gd, args, writer, log)

    if fh:
        fh.close()
        log(f"\nwrote {args.csv}")
    if allr:
        E = np.array([r['E'] for r in allr])
        W = np.array([r['weight'] for r in allr])
        log(f"\ntotal: n={len(allr)}  mean E={E.mean():.4f}  "
            f"weighted E={np.sum(W*E)/np.sum(W):.4f}  "
            f"median VR={np.median([r['vr'] for r in allr]):+.3f}")


if __name__ == '__main__':
    main()
