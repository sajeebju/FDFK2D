#!/usr/bin/env python3
"""
model_extent.py
===============
Compute the profile extent the FD grid actually needs, from the real station
coordinates rather than from an estimate.

For every usable trace it projects the station onto the profile, computes the
SsPmp free-surface reflection point at

    x_reflect = x_station + 2*h*tan(i_P) * cos(baz - profile azimuth)

and reports the union of stations and reflection points, plus the margin you
should add.  Also reports where tpmp.xy column 1 sits relative to the station,
which is what tells you whether that column is the Moho bounce point (about
half the offset) or the free-surface reflection point (the full offset).

  python3 model_extent.py --paths wv.paths --moho-km 52 --vp-crust 6800
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdfk2d_common import (ProfileGeom, read_t8, read_shift, read_tpmp,
                           read_kv, apply_kv, explicit_dests, require_paths)
from wv_model2d import DEFAULT_REF, load_ref_from_file

ALL_EVENTS = ['20140824232145', '20140825143137', '20140925175117',
              '20150323045138', '20150529070009', '20150624223221',
              '20150729023559']


def surface_offset_km(p_skm, moho_km, layers, dz_km=0.1):
    """
    Horizontal distance from the station to the SsPmp free-surface reflection
    point, by integrating the converted P leg through the layered crust:

        offset = 2 * integral_0^h  p*v / sqrt(1 - (p*v)^2)  dz

    A single mean velocity overestimates this badly, because the ray spends its
    first kilometres in slow sediment where tan(i) is small: for p = 0.1156
    s/km and a 4.2 / 6.0 / 6.8 km/s column, 2h*tan(i) with v = 6.8 gives 132 km
    while the integral gives 117 km.  That 12% matters when you are deciding
    how wide to build a 700 km grid.
    """
    z = np.arange(0.0, moho_km, dz_km) + 0.5 * dz_km
    tops = np.array([l[0] for l in layers], float)
    vps = np.array([l[1] for l in layers], float) / 1000.0
    v = vps[np.clip(np.searchsorted(tops, z, side='right') - 1, 0, len(vps) - 1)]
    eta = np.clip(p_skm * v, 0.0, 0.999)
    return float(2.0 * np.sum(eta / np.sqrt(1.0 - eta ** 2)) * dz_km)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', default=None)
    ap.add_argument('--events-root', default=None)
    ap.add_argument('--tpmp', default=None)
    ap.add_argument('--prf', default=None)
    ap.add_argument('--events', nargs='*', default=list(ALL_EVENTS))
    ap.add_argument('--moho-km', type=float, default=52.0)
    ap.add_argument('--vp-crust', type=float, default=6800.0,
                    help='single crustal Vp [m/s]; ignored when --ref-1d is given')
    ap.add_argument('--ref-1d', default=None,
                    help='depth_km Vp_m/s Vs_m/s column; the P leg is integrated '
                         'through it instead of using one mean velocity. Use the '
                         'column your grid inversion actually produced.')
    ap.add_argument('--use-builtin-column', action='store_true',
                    help='integrate through the built-in generic WVSZ column')
    ap.add_argument('--margin-km', type=float, default=40.0,
                    help='clearance to keep between any raypath and the FK/PML edge')
    ap.add_argument('--shift-flag', default='1', choices=['1', '0', 'any'])
    args = ap.parse_args()
    if args.paths:
        apply_kv(args, read_kv(args.paths), parser=ap,
                 explicit=explicit_dests(ap))
    require_paths(args, ['events_root', 'tpmp'])

    from obspy import read
    geom = (ProfileGeom.from_prf(args.prf) if args.prf
            else ProfileGeom.from_tpmp(args.tpmp))
    print(geom)

    layers = None
    if args.ref_1d:
        layers = load_ref_from_file(args.ref_1d)
        print(f"integrating the P leg through {args.ref_1d} ({len(layers)} layers)")
    elif args.use_builtin_column:
        layers = DEFAULT_REF
        print("integrating the P leg through the built-in generic WVSZ column")
    else:
        print(f"using a single crustal Vp = {args.vp_crust:g} m/s "
              f"(pass --ref-1d for the layered integral, which is more accurate)")

    tp = {(r['event'], r['station']): r['x'] for r in read_tpmp(args.tpmp)}
    sx_all, rx_all, off_all = [], [], []

    print(f"\n{'event':16s} {'n':>3s} {'x_sta [km]':>18s} {'x_reflect [km]':>18s} "
          f"{'cos(th)':>8s}")
    for ev in args.events:
        ed = os.path.join(args.events_root, ev)
        t8p = os.path.join(ed, 't8.dat')
        if not os.path.isfile(t8p):
            print(f"{ev:16s}  no t8.dat, skipped")
            continue
        t8 = read_t8(t8p)
        sh = read_shift(os.path.join(ed, 'shift.dat')) \
            if os.path.isfile(os.path.join(ed, 'shift.dat')) else {}
        xs, xr, cs = [], [], []
        for sta, rec in sorted(t8.items()):
            if args.shift_flag != 'any':
                if sh.get(sta, (None, 0))[1] != int(args.shift_flag):
                    continue
            zf = os.path.join(ed, sta + '.z')
            if not os.path.isfile(zf):
                continue
            try:
                s = read(zf, format='SAC', headonly=True)[0].stats.sac
                stla, stlo = float(s['stla']), float(s['stlo'])
            except Exception:
                continue
            xk, _ = geom.ll2xy(stla, stlo)
            theta = np.radians((rec['baz'] - geom.azimuth + 180.0) % 360.0 - 180.0)
            if layers is not None:
                d = surface_offset_km(rec['p_skm'], args.moho_km, layers)
            else:
                eta = rec['p_skm'] / 1000.0 * args.vp_crust
                if eta >= 1.0:
                    continue
                d = 2.0 * args.moho_km * eta / np.sqrt(1.0 - eta ** 2)
            xs.append(float(xk))
            xr.append(float(xk) + d * np.cos(theta))
            cs.append(float(np.cos(theta)))
            if (ev, sta) in tp:
                # the ALONG-PROFILE offset to the surface reflection point is
                # d*cos(theta); dividing by d alone mixes the sign of cos(theta)
                # into the ratio and makes the two event groups cancel
                den = d * np.cos(theta)
                if abs(den) > 1e-6:
                    off_all.append((tp[(ev, sta)] - float(xk)) / den)
        if not xs:
            continue
        sx_all += xs
        rx_all += xr
        print(f"{ev:16s} {len(xs):3d} {min(xs):8.1f} .. {max(xs):6.1f} "
              f"{min(xr):8.1f} .. {max(xr):6.1f} {np.mean(cs):+8.3f}")

    if not sx_all:
        raise SystemExit('no usable stations found')

    lo = min(min(sx_all), min(rx_all))
    hi = max(max(sx_all), max(rx_all))
    print(f"\nstations            {min(sx_all):8.1f} .. {max(sx_all):6.1f} km "
          f"({max(sx_all)-min(sx_all):.0f} km array)")
    print(f"SsPmp reflection    {min(rx_all):8.1f} .. {max(rx_all):6.1f} km")
    print(f"raypaths span       {lo:8.1f} .. {hi:6.1f} km")
    print(f"\n  ==> build the model from {np.floor((lo-args.margin_km)/5)*5:.0f} "
          f"to {np.ceil((hi+args.margin_km)/5)*5:.0f} km "
          f"({(np.ceil((hi+args.margin_km)/5)-np.floor((lo-args.margin_km)/5))*5:.0f} km, "
          f"margin {args.margin_km:g} km)")
    print(f"  ==> set x0_km = {np.floor((lo-args.margin_km)/5)*5:.0f} in wv.paths")

    # ---- what tpmp.xy column 1 actually is, and the extent it implies -------
    tx = np.array([v for v in tp.values()])
    print(f"\ntpmp.xy column 1 spans     {tx.min():8.1f} .. {tx.max():6.1f} km")
    if off_all:
        f = float(np.median(off_all))
        sc = 0.5 * float(np.percentile(off_all, 84) - np.percentile(off_all, 16))
        print(f"  it sits at {f:.2f} +/- {sc:.2f} of the modelled along-profile "
              f"station-to-surface offset ({len(off_all)} traces)")

        # Rather than a threshold, report what each hypothesis would REQUIRE.
        # Only one of them is usually physically possible.
        pmed = 0.1156
        print("\n  what each interpretation would require:")
        for name, frac in (('free-surface reflection point', 1.0),
                           ('Moho bounce point', 0.5)):
            scale = f / frac                     # true offset / modelled offset
            tan_req = scale * (np.tan(np.arcsin(np.clip(pmed * args.vp_crust / 1000.,
                                                        0, .999))))
            eta_req = tan_req / np.sqrt(1.0 + tan_req ** 2)
            vp_req = eta_req / pmed
            h_req = args.moho_km * scale
            ok = 5.9 <= vp_req <= 7.1
            print(f"    {name:32s} crustal Vp = {vp_req:4.2f} km/s at h = "
                  f"{args.moho_km:.0f} km, or h = {h_req:4.1f} km at the given Vp"
                  f"   {'plausible' if ok else 'NOT plausible'}")

        lo1 = min(min(sx_all), float(tx.min()))
        hi1 = max(max(sx_all), float(tx.max()))
        c = float(np.median(sx_all))
        lo2 = min(min(sx_all), 2 * (float(tx.min()) - c) + c)
        hi2 = max(max(sx_all), 2 * (float(tx.max()) - c) + c)
        for name, lo_, hi_ in (('tpmp IS the reflection point', lo1, hi1),
                               ('tpmp is the Moho midpoint (x2)', lo2, hi2)):
            a = np.floor((lo_ - args.margin_km) / 5) * 5
            b = np.ceil((hi_ + args.margin_km) / 5) * 5
            print(f"\n  {name:34s} model {a:7.0f} .. {b:6.0f} km "
                  f"({b - a:.0f} km, x0_km = {a:.0f})")
        print("\n  Pick the interpretation whose required Vp is physical. If both "
              "look plausible, take the wider model.")


if __name__ == '__main__':
    main()
