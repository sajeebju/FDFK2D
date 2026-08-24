#!/usr/bin/env python3
"""
fdfk2d_setup.py
===============
Build one complete, self-consistent FDFK2D input deck per event for the Wabash
Valley SsPmp experiment, deriving every parameter from the real metadata
(t8.dat, shift.dat, tpmp.xy, SAC headers) instead of hand-entering them.

One simulation per EVENT (not per grid).  The 2-D model spans the whole profile,
so a single run gives synthetics at every station of that event; the grid
assignment is applied afterwards, when the SAC files are distributed
(su2sac.py) and when the misfit is formed (sac_crossconv.py).

What it derives / checks
------------------------
  * profile origin + azimuth, recovered from tpmp.xy
  * station positions projected onto the profile (from SAC stla/stlo)
  * event location (SAC evla/evlo, cross-checked against t8.dat gcarc/baz)
  * ray parameter in s/deg from t8.dat (which stores s/km)
  * az_org solved so that FDFK2D's internal (baz - az_org) equals the true
    in-plane angle  -- see README, this is the parameter that is easiest to get
    silently wrong
  * |baz - profile azimuth| against FDFK2D's own 30 deg validity limit
  * whether the SsPmp free-surface reflection point still lies inside the model
  * dt from the CFL limit, tmax from the plane-wave moveout across the model,
    tstp against the INTEGER*2 header limit, f0 against points-per-wavelength

Usage
-----
  python3 fdfk2d_setup.py \
      --events-root /home/yaoj/data/SSPMP/wabash \
      --tpmp        /home/yaoj/paper/others/LiuYC/pmp/tpmp.xy \
      --model-dir   /home/yaoj/C++/FDFK2D/input_2d \
      --out-root    /home/yaoj/C++/FDFK2D/runs \
      --x0-km -318 --x1-km 222 --dx 200 --dz 200 --Zkm 80 --npml 20

  # then
  bash /home/yaoj/C++/FDFK2D/runs/run_all.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdfk2d_common import (ProfileGeom, backaz, read_su, read_t8, read_shift,
                           read_tpmp, fdfk2d_theta, solve_az_org, ensure_dir,
                           read_kv, apply_kv, explicit_dests, require_paths, DEG2KM, eprint)

ALL_EVENTS = ['20140824232145', '20140825143137', '20140925175117',
              '20150323045138', '20150529070009', '20150624223221',
              '20150729023559']


# ---------------------------------------------------------------------------
def read_sac_headers(path):
    from obspy import read
    tr = read(path, format='SAC', headonly=True)[0]
    s = tr.stats.sac
    return dict(delta=float(tr.stats.delta),
                npts=int(tr.stats.npts),
                b=float(s.get('b', 0.0)),
                o=float(s['o']) if 'o' in s else None,
                stla=float(s['stla']) if 'stla' in s else None,
                stlo=float(s['stlo']) if 'stlo' in s else None,
                stel=float(s.get('stel', 0.0) or 0.0),
                evla=float(s['evla']) if 'evla' in s else None,
                evlo=float(s['evlo']) if 'evlo' in s else None,
                evdp=float(s['evdp']) if 'evdp' in s else None,
                user0=float(s['user0']) if 'user0' in s else None,
                baz=float(s['baz']) if 'baz' in s else None,
                gcarc=float(s['gcarc']) if 'gcarc' in s else None)


def source_from_baz_delta(stla, stlo, baz_deg, delta_deg):
    """Great-circle back-projection: station + (delta, baz) -> event lat/lon.
    Only used when the SAC headers carry no evla/evlo."""
    la = np.radians(stla)
    lo = np.radians(stlo)
    d = np.radians(delta_deg)
    b = np.radians(baz_deg)
    lat2 = np.arcsin(np.sin(la) * np.cos(d) + np.cos(la) * np.sin(d) * np.cos(b))
    lon2 = lo + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la),
                           np.cos(d) - np.sin(la) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 540.0) % 360.0 - 180.0


def compress_to_layers(depth_m, vp, vs, tol_vp=60.0, min_thick_m=1000.0):
    """Turn a sampled 1-D column into an FDFK2D FK layer stack (vp, vs, ztop)."""
    layers = [(float(vp[0]), float(vs[0]), 0.0)]
    for k in range(1, len(depth_m)):
        if (abs(vp[k] - layers[-1][0]) > tol_vp and
                depth_m[k] - layers[-1][2] >= min_thick_m):
            layers.append((float(vp[k]), float(vs[k]), float(depth_m[k])))
    return layers


def write_fk_model(path, layers, title):
    """
    FDFK2D reads `nlayer` and then exactly `nlayer` lines, so nlayer must equal
    the number of (Vp, Vs, ztop) rows -- the deepest row is the half-space.
    Density is not read; the code derives it as rho = (Vp + 980)/2.760.
    """
    with open(path, 'w') as fh:
        fh.write('nlayer\n')
        fh.write('  %d\n' % len(layers))
        fh.write('Vp(m/s)   Vs(m/s)   z+(m) <----- top boundary of the layers  ! %s\n' % title)
        for vp, vs, z in layers:
            fh.write('%9.1f %9.1f %12.1f\n' % (vp, vs, z))


# ---------------------------------------------------------------------------
def build_event(ev, args, geom, model, tpmp_rows, log):
    ev_dir = os.path.join(args.events_root, ev)
    t8_path = os.path.join(ev_dir, 't8.dat')
    sh_path = os.path.join(ev_dir, 'shift.dat')
    if not os.path.isfile(t8_path):
        log(f"  [skip] no t8.dat in {ev_dir}")
        return None
    t8 = read_t8(t8_path)
    shift = read_shift(sh_path) if os.path.isfile(sh_path) else {}

    in_tpmp = {r['station'] for r in tpmp_rows if r['event'] == ev}
    n_flag_drop = [0]

    stations = []
    evla = evlo = evdp = None
    for sta, rec in sorted(t8.items()):
        zf = os.path.join(ev_dir, sta + '.z')
        if not os.path.isfile(zf):
            continue
        try:
            h = read_sac_headers(zf)
        except Exception as exc:
            log(f"  [warn] cannot read {zf}: {exc}")
            continue
        if h['stla'] is None or h['stlo'] is None:
            log(f"  [warn] {sta}: no stla/stlo in SAC header, dropped")
            continue
        flag = shift.get(sta, (np.nan, 1))[1]
        if args.shift_flag != 'any' and flag != int(args.shift_flag):
            n_flag_drop[0] += 1
            continue
        if evla is None and h['evla'] is not None:
            evla, evlo, evdp = h['evla'], h['evlo'], (h['evdp'] or 0.0)
        xk, yk = geom.ll2xy(h['stla'], h['stlo'])
        stations.append(dict(station=sta, stla=h['stla'], stlo=h['stlo'],
                             stel=h['stel'], x_km=float(xk), y_km=float(yk),
                             gcarc=rec['gcarc'], baz=rec['baz'],
                             p_skm=rec['p_skm'],
                             t_s_shift=shift.get(sta, (np.nan, 1))[0],
                             shift_flag=shift.get(sta, (np.nan, 1))[1],
                             t_s_t8=rec['t_s_t8'],
                             delta_sac=h['delta'], b_sac=h['b'], o_sac=h['o'],
                             user0=h['user0'],
                             used_in_tpmp=sta in in_tpmp))
    if n_flag_drop[0]:
        log(f"  shift.dat flag filter (={args.shift_flag}) dropped {n_flag_drop[0]} of "
            f"{n_flag_drop[0] + len(stations)} traces")

    if args.only_stations:
        keep_names = set(args.only_stations)
        dropped = [s['station'] for s in stations if s['station'] not in keep_names]
        stations = [s for s in stations if s['station'] in keep_names]
        missing = keep_names - {s['station'] for s in stations}
        if missing:
            log(f"  [warn] --only-stations asked for {sorted(missing)}, not present in {ev}")
        if dropped:
            log(f"  --only-stations: keeping {[s['station'] for s in stations]}, "
                f"dropped {len(dropped)} others")

    if not stations:
        log(f"  [skip] no readable SAC files for {ev}")
        return None

    # ---- event location -----------------------------------------------------
    if evla is None:
        s0 = stations[0]
        evla, evlo = source_from_baz_delta(s0['stla'], s0['stlo'], s0['baz'], s0['gcarc'])
        evdp = 0.0
        log(f"  [note] no evla/evlo in SAC headers; back-projected from "
            f"({s0['station']}, baz={s0['baz']:.2f}, gcarc={s0['gcarc']:.2f})"
            f" -> {evla:.4f}, {evlo:.4f}")
    else:
        s0 = stations[len(stations) // 2]
        d_chk, _, b_chk = backaz(evla, evlo, s0['stla'], s0['stlo'])
        log(f"  event {evla:.4f}N {evlo:.4f}E depth {evdp:.1f}; "
            f"check at {s0['station']}: gcarc {d_chk:.2f} vs t8 {s0['gcarc']:.2f}, "
            f"baz {b_chk:.2f} vs t8 {s0['baz']:.2f}")
        if abs(b_chk - s0['baz']) > 2.0 or abs(d_chk - s0['gcarc']) > 2.0:
            log("  [WARN] SAC evla/evlo disagrees with t8.dat by >2 deg -- check which is right")

    # ---- representative ray parameter & back-azimuth ------------------------
    sel = [s for s in stations if s['used_in_tpmp']] or stations
    p_skm = float(np.median([s['p_skm'] for s in sel]))
    p_sdeg = p_skm * DEG2KM
    baz_true = float(np.median([s['baz'] for s in sel]))
    p_spread = (max(s['p_skm'] for s in sel) - min(s['p_skm'] for s in sel)) * DEG2KM

    theta_true = (baz_true - geom.azimuth + 180.0) % 360.0 - 180.0
    cos_t = np.cos(np.radians(theta_true))
    side = 'right (+x)' if cos_t > 0 else 'left (-x)'
    dev = min(abs(theta_true), abs(180.0 - abs(theta_true)))
    log(f"  p = {p_skm:.5f} s/km = {p_sdeg:.4f} s/deg  (spread over stations "
        f"{p_spread:.3f} s/deg)")
    log(f"  baz = {baz_true:.2f}, profile az = {geom.azimuth:.2f}  ->  "
        f"theta = {theta_true:+.2f} deg, cos(theta) = {cos_t:+.4f}, "
        f"wave enters from the {side}")
    if dev > 30.0:
        log(f"  [WARN] |baz - profile azimuth| = {dev:.1f} deg exceeds FDFK2D's own 30 deg "
            f"validity limit; the binary will stop and ask 'Do you want to continue? [Y/N]'. "
            f"The run script answers Y, but treat this event's amplitudes with caution.")

    # ---- az_org that makes FDFK2D's internal angle correct ------------------
    x_center_m = 0.5 * (model['X0'] + model['Xn'])
    lat_org, lon_org = geom.xy2ll(model['x0_km'], 0.0)
    az_org, th_got, resid = solve_az_org(lat_org, lon_org, evla, evlo,
                                         x_center_m, theta_true,
                                         variant=args.geom_variant)
    th_edges = fdfk2d_theta(lat_org, lon_org, az_org, evla, evlo,
                            np.array([model['X0'], x_center_m, model['Xn']]),
                            variant=args.geom_variant)
    log(f"  az_org written to FD_model.dat = {az_org:.4f} deg  "
        f"(variant '{args.geom_variant}')")
    log(f"  FDFK2D-internal theta at x = X0 / centre / Xn : "
        f"{th_edges[0]:+.2f} / {th_edges[1]:+.2f} / {th_edges[2]:+.2f} deg "
        f"(target {theta_true:+.2f}, residual {resid:.4f})")
    if resid > 0.5:
        log("  [WARN] could not match the target in-plane angle to better than 0.5 deg")

    # ---- receivers ----------------------------------------------------------
    rx_sta, keep = [], []
    for s in stations:
        rx = (s['x_km'] - model['x0_km']) * 1000.0
        if rx < model['X0'] - 1e-6 or rx > model['Xn'] + 1e-6:
            log(f"  [warn] {s['station']} projects to x = {s['x_km']:.1f} km "
                f"(model x = {rx/1000:.1f} km), OUTSIDE the model -- dropped")
            continue
        s['rx_m'] = float(rx)
        rx_sta.append(rx)
        keep.append(s)
    stations = keep
    if not stations:
        log(f"  [skip] every station of {ev} falls outside the model")
        return None

    # SsPmp free-surface reflection point must still be inside the model
    h_moho = model['moho_m']
    vp_crust = model['vp_crust']
    eta = p_skm / 1000.0 * vp_crust
    if eta >= 1.0:
        log("  [WARN] p*Vp >= 1 in the crust: the converted P leg is evanescent, check p")
        d_off = np.nan
    else:
        d_off = 2.0 * h_moho * eta / np.sqrt(1.0 - eta ** 2)      # metres, station -> surface bounce
        xs_ref = np.array([s['rx_m'] for s in stations]) + d_off * cos_t
        n_out = int(np.sum((xs_ref < model['X0'] + args.edge_margin_km * 1000) |
                           (xs_ref > model['Xn'] - args.edge_margin_km * 1000)))
        log(f"  SsPmp surface reflection point sits {d_off/1000:.1f} km from the station "
            f"toward the source")
        if n_out:
            log(f"  [WARN] {n_out}/{len(stations)} stations have their SsPmp reflection point "
                f"within {args.edge_margin_km:g} km of the model edge -- widen the model or "
                f"those synthetics will be contaminated by the FK/PML boundary")

    rx_dense = (np.arange(model['X0'], model['Xn'] + 1e-6, args.dense_dx_km * 1000.0)
                if args.dense_dx_km > 0 else np.empty(0))
    rx_all = np.concatenate([np.array(rx_sta), rx_dense])
    n_sta = len(rx_sta)

    # ---- time stepping ------------------------------------------------------
    vmax = model['vp_max']
    dt = args.dt if args.dt else 0.3 * model['dx'] / vmax
    dt = float(np.floor(dt * 1e4) / 1e4)                       # 0.1 ms granularity
    moveout = abs(p_skm * cos_t) * (model['Xn'] - model['X0']) / 1000.0
    tmax = args.tmax if args.tmax else max(60.0, 1.6 * moveout + args.tmax_margin)
    tmax = float(np.ceil(tmax))

    tstp_want = min(0.030, 1.0 / (20.0 * 3.0 * args.f0))
    nstp = max(1, int(np.floor(tstp_want / dt)))
    tstp = nstp * dt
    nsamp = int(tmax / tstp) + 1
    if nsamp >= 32767:
        raise SystemExit(f"{ev}: {nsamp} output samples exceeds FDFK2D's INTEGER*2 header "
                         f"limit; increase tstp or reduce tmax")
    if int(round(tstp * 1e6)) >= 32768:
        raise SystemExit(f"{ev}: tstp={tstp}s does not fit in the INTEGER*2 SU header")

    # source frequency vs grid dispersion (S source -> Vs controls it)
    f0_max = model['vs_min'] / (args.ppw * model['dx'] * 2.5)
    log(f"  dt = {dt:g} s (CFL, vmax={vmax:.0f} m/s), tmax = {tmax:g} s "
        f"(moveout {moveout:.1f} s), tstp = {tstp:g} s, {nsamp} samples/trace")
    log(f"  f0 = {args.f0:g} Hz; grid supports up to {f0_max:.2f} Hz at "
        f"{args.ppw:g} pts/wavelength (Vs_min = {model['vs_min']:.0f} m/s)")
    if args.f0 > f0_max:
        log(f"  [WARN] f0 = {args.f0:g} Hz is above the dispersion-safe limit "
            f"{f0_max:.2f} Hz -- lower f0 or refine dx")

    # ---- write the deck -----------------------------------------------------
    run_dir = ensure_dir(os.path.join(args.out_root, ev))
    inp = ensure_dir(os.path.join(run_dir, 'input'))
    ensure_dir(os.path.join(run_dir, 'seismograms'))

    with open(os.path.join(inp, 'FD_model.dat'), 'w') as fh:
        fh.write('order of FD(2m>=2)\n')
        fh.write('  %d\n' % args.norder)
        fh.write('X0(m)        Xn(m)        Z0(m)     Zn(m)       Dx(m)      Dz(m)       Dt(s)\n')
        fh.write('  %-10.1f %-12.1f %-9.1f %-11.1f %-10.1f %-11.1f %g\n' %
                 (model['X0'], model['Xn'], model['Z0'], model['Zn'],
                  model['dx'], model['dz'], dt))
        fh.write('tmax(s)      tstep        nPML(number of grid)  is_PML_top\n')
        fh.write('  %-12g %-12g %-21d %s\n' % (tmax, tstp, model['npml'], '.false.'))
        fh.write('latitude of (X0,Z0)       longitude of (X0,Z0)  azimuthe of profile\n')
        fh.write('  %-25.8f %-21.8f %.8f\n' % (lat_org, lon_org, az_org))
        fh.write('output snapshot?(.true./.false.)  nstep\n')
        fh.write('  %s                          %d\n' %
                 ('.true.' if args.snapshots else '.false.', args.snap_every))

    with open(os.path.join(inp, 'Source.dat'), 'w') as fh:
        fh.write('f0(Hz)   strength(m)   source type(P:.true.;S:.false.)\n')
        fh.write('   %-8g %-13s %s\n' % (args.f0, '1.d-3', '.false.'))
        fh.write('ray parameter(s/deg)   src_lat(deg)   src_lon(deg)\n')
        fh.write('  %-22.5f %-14.5f %.5f    ! event %s, p=%.6f s/km, baz=%.2f\n'
                 % (p_sdeg, evla, evlo, ev, p_skm, baz_true))

    with open(os.path.join(inp, 'Receiver.dat'), 'w') as fh:
        fh.write('number of receivers:\n')
        fh.write('     %d\n' % len(rx_all))
        fh.write('rx(m)   rz(m)\n')
        for i, rx in enumerate(rx_all):
            tag = stations[i]['station'] if i < n_sta else 'dense'
            fh.write('%.3f\t    0.0\t\t! %s\n' % (rx, tag))

    write_fk_model(os.path.join(inp, 'FK_model_left.dat'), model['fk_left'], 'left edge')
    write_fk_model(os.path.join(inp, 'FK_model_right.dat'), model['fk_right'], 'right edge')

    for f in (model['vp_name'], model['vs_name']):
        src = os.path.join(args.model_dir, f)
        dst = os.path.join(inp, f)
        if not os.path.exists(dst):
            (os.symlink if args.link_models else shutil.copy)(os.path.abspath(src), dst)

    with open(os.path.join(inp, 'inpar.dat'), 'w') as fh:
        fh.write('FD_model.dat\nSource.dat\nReceiver.dat\n'
                 'FK_model_left.dat\nFK_model_right.dat\n%s\n%s\n'
                 % (model['vp_name'], model['vs_name']))

    meta = dict(event=ev, evla=evla, evlo=evlo, evdp=evdp,
                p_skm=p_skm, p_sdeg=p_sdeg, baz_true=baz_true,
                profile_lat0=geom.lat0, profile_lon0=geom.lon0,
                profile_azimuth=geom.azimuth,
                lat_org=lat_org, lon_org=lon_org, az_org=az_org,
                geom_variant=args.geom_variant,
                theta_true=theta_true, cos_theta=float(cos_t),
                incidence_side=side, deviation_deg=dev,
                X0=model['X0'], Xn=model['Xn'], Z0=model['Z0'], Zn=model['Zn'],
                dx=model['dx'], dz=model['dz'], npml=model['npml'],
                x0_km=model['x0_km'], dt=dt, tmax=tmax, tstp=tstp,
                nsamp=nsamp, f0=args.f0, norder=args.norder,
                src_type='S', n_station_receivers=n_sta,
                dense_dx_km=args.dense_dx_km,
                sspmp_offset_km=(float(d_off / 1000.0) if np.isfinite(d_off) else None),
                stations=stations)
    with open(os.path.join(run_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)

    log(f"  wrote {run_dir}  ({n_sta} station receivers + {len(rx_dense)} dense)")
    return run_dir


# ---------------------------------------------------------------------------
def load_model(args, log):
    """Read vp/vs .su, recover grid geometry, min/max velocities, FK edge models."""
    vp_name = args.vp_su
    vs_name = args.vs_su
    ns_hint = args.model_ns
    vp, _ = read_su(os.path.join(args.model_dir, vp_name), ns=ns_hint)
    vs, _ = read_su(os.path.join(args.model_dir, vs_name), ns=ns_hint)
    if vp.shape != vs.shape:
        raise SystemExit(f"vp {vp.shape} and vs {vs.shape} models differ in size")

    npml = args.npml
    ntr, nzs = vp.shape
    nx = ntr - 2 * npml
    nz = nzs - npml                      # is_PML_top = .false.  ->  izt = 1
    X0 = 0.0
    Xn = (nx - 1) * args.dx
    Z0 = 0.0
    Zn = (nz - 1) * args.dz
    log(f"model: {vp_name} {vp.shape} -> nx={nx}, nz={nz}, "
        f"X0..Xn = 0..{Xn/1000:.1f} km, Z0..Zn = 0..{Zn/1000:.1f} km, npml={npml}")
    if args.Zkm and abs(Zn / 1000.0 - args.Zkm) > 0.5 * args.dz / 1000.0:
        log(f"  [WARN] model depth from the .su file is {Zn/1000:.1f} km but --Zkm says "
            f"{args.Zkm:g} km; --npml is the usual culprit")

    interior = vp[npml:npml + nx, :nz]
    vs_int = vs[npml:npml + nx, :nz]
    if not args.no_edge_repair:
        from wv_model2d import repair_edges
        interior, vs_int, nl, nr = repair_edges(interior, vs_int,
                                                args.edge_frac, log)
        if nl or nr:
            log("  (FK_model_left/right are extracted from these columns, so an "
                "un-repaired taper would sit directly on the injection boundary)")
    depth = np.arange(nz) * args.dz

    # Moho proxy: shallowest depth where Vp crosses 7.5 km/s, per column
    moho = []
    for ix in range(nx):
        k = np.argmax(interior[ix] >= 7500.0)
        moho.append(depth[k] if interior[ix].max() >= 7500.0 else np.nan)
    moho = np.array(moho, float)
    moho_m = float(np.nanmedian(moho)) if np.isfinite(moho).any() else 45000.0
    vp_crust = float(np.median(interior[:, :max(1, int(moho_m / args.dz))]))

    # ---- FK 1-D models -----------------------------------------------------
    # FDFK2D requires the two edge models to share an identical bottom
    # half-space, so find the shallowest depth below which the left and right
    # columns already agree, and make that the half-space top.
    dif = np.abs(interior[0] - interior[-1])
    agree = dif <= args.layer_tol
    k_half = nz - 1
    for k in range(nz):
        if agree[k:].all():
            k_half = k
            break
    else:
        log(f"  [WARN] the two model edges never agree to within {args.layer_tol:g} m/s; "
            f"the shared FK half-space is an approximation")
    z_half = float(depth[k_half])
    half = (float(0.5 * (interior[0, k_half:] + interior[-1, k_half:]).mean()),
            float(0.5 * (vs_int[0, k_half:] + vs_int[-1, k_half:]).mean()),
            z_half)
    def _stack(col_vp, col_vs):
        lay = [l for l in compress_to_layers(depth[:k_half + 1], col_vp[:k_half + 1],
                                             col_vs[:k_half + 1],
                                             args.layer_tol, args.layer_min_thick)
               if l[2] < z_half]
        # drop a trailing layer that merely duplicates the half-space
        while lay and abs(lay[-1][0] - half[0]) <= args.layer_tol:
            lay.pop()
        return lay + [half]

    left = _stack(interior[0], vs_int[0])
    right = _stack(interior[-1], vs_int[-1])
    if len(left) < 2 or len(right) < 2:
        log("  [WARN] an FK edge model collapsed to a half-space; loosen --layer-tol")

    log(f"  Moho proxy (Vp=7.5 km/s) median {moho_m/1000:.1f} km, crustal Vp {vp_crust:.0f} m/s")
    log(f"  FK left model {len(left)} layers, right {len(right)} layers; "
        f"shared half-space Vp={half[0]:.0f} Vs={half[1]:.0f} from {half[2]/1000:.1f} km down")

    return dict(X0=X0, Xn=Xn, Z0=Z0, Zn=Zn, dx=args.dx, dz=args.dz, npml=npml,
                nx=nx, nz=nz, x0_km=args.x0_km,
                vp_max=float(vp.max()), vs_min=float(vs_int[vs_int > 0].min()),
                moho_m=moho_m, vp_crust=vp_crust,
                fk_left=left, fk_right=right,
                vp_name=vp_name, vs_name=vs_name)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', default=None,
                    help='settings file with "key = value" lines; any flag given on '
                         'the command line overrides it')
    ap.add_argument('--events-root', default=None)
    ap.add_argument('--tpmp', default=None)
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--out-root', default=None)
    ap.add_argument('--events', nargs='*', default=ALL_EVENTS)
    ap.add_argument('--vp-su', default='vp_model.su')
    ap.add_argument('--vs-su', default='vs_model.su')
    ap.add_argument('--model-ns', type=int, default=None,
                    help='override the sample count in the model .su trace header')
    ap.add_argument('--x0-km', type=float, default=None,
                    help='profile coordinate (tpmp.xy x, km) of model X0')
    ap.add_argument('--dx', type=float, default=200.0)
    ap.add_argument('--dz', type=float, default=200.0)
    ap.add_argument('--Zkm', type=float, default=None)
    ap.add_argument('--npml', type=int, default=20)
    ap.add_argument('--norder', type=int, default=8)
    ap.add_argument('--f0', type=float, default=0.8)
    ap.add_argument('--ppw', type=float, default=5.0,
                    help='minimum points per shortest wavelength (default 5)')
    ap.add_argument('--dt', type=float, default=None, help='override the CFL estimate')
    ap.add_argument('--tmax', type=float, default=None)
    ap.add_argument('--tmax-margin', type=float, default=60.0)
    ap.add_argument('--shift-flag', default='1', choices=['1', '0', 'any'],
                    help="keep only traces with this shift.dat column-3 flag. 1 = the "
                         "traces that survived the minCC clustering in the Makefile and "
                         "are the ones present in tpmp.xy (default)")
    ap.add_argument('--dense-dx-km', type=float, default=0.0,
                    help='spacing of the extra dense receiver line, 0 to disable')
    ap.add_argument('--edge-margin-km', type=float, default=20.0)
    ap.add_argument('--only-stations', nargs='*', default=None,
                    help='restrict the receiver list to these stations (single-station tests); '
                         'the FD cost is unchanged, only the seismogram arrays shrink')
    ap.add_argument('--no-edge-repair', action='store_true')
    ap.add_argument('--edge-frac', type=float, default=0.85)
    ap.add_argument('--layer-tol', type=float, default=60.0, help='Vp step [m/s] for FK layers')
    ap.add_argument('--layer-min-thick', type=float, default=1000.0)
    ap.add_argument('--geom-variant', default='as_coded',
                    choices=['as_coded', 'transpose', 'corrected'],
                    help="which define_rotation_matrix.f90 your binary was built from")
    ap.add_argument('--profile-json', default=None,
                    help='reuse a previously fitted profile instead of re-fitting tpmp.xy')
    ap.add_argument('--snapshots', action='store_true')
    ap.add_argument('--snap-every', type=int, default=200)
    ap.add_argument('--link-models', action='store_true',
                    help='symlink vp/vs .su into each run instead of copying')
    ap.add_argument('--fdfk2d-bin', default='FDFK2D')
    args = ap.parse_args()
    if args.paths:
        apply_kv(args, read_kv(args.paths), parser=ap,
                 explicit=explicit_dests(ap))
    require_paths(args, ['events_root', 'tpmp', 'model_dir', 'out_root'])
    if args.x0_km is None:
        raise SystemExit('--x0-km is required (profile coordinate of model X0, km)')

    def log(*a):
        print(*a, flush=True)

    geom = (ProfileGeom.from_json(args.profile_json) if args.profile_json
            else ProfileGeom.from_tpmp(args.tpmp))
    log(str(geom))
    if geom.rms_km and geom.rms_km > 3.0:
        log(f"[WARN] profile fit rms {geom.rms_km:.1f} km is large; tpmp.xy may not be a "
            f"simple straight-line projection")
    geom.to_json(os.path.join(ensure_dir(args.out_root), 'profile.json'))

    model = load_model(args, log)
    tpmp_rows = read_tpmp(args.tpmp)

    made = []
    for ev in args.events:
        log(f"\n=== {ev} ===")
        try:
            d = build_event(ev, args, geom, model, tpmp_rows, log)
        except Exception as exc:
            log(f"  [ERROR] {ev}: {exc}")
            continue
        if d:
            made.append((ev, d))

    runner = os.path.join(args.out_root, 'run_all.sh')
    with open(runner, 'w') as fh:
        fh.write('#!/usr/bin/env bash\n')
        fh.write('# Generated by fdfk2d_setup.py.  "yes Y |" answers FDFK2D\'s interactive\n')
        fh.write('# prompt when |baz - profile azimuth| > 30 deg, which would otherwise\n')
        fh.write('# hang a batch run.\n')
        fh.write('set -u\n')
        fh.write('BIN=%s\n' % args.fdfk2d_bin)
        for ev, d in made:
            fh.write(f'\necho "=== {ev} ==="\n')
            fh.write(f'cd {d}\n')
            fh.write('mkdir -p seismograms snapshots\n')
            fh.write(f'yes Y | "$BIN" ./input inpar.dat ./seismograms seis '
                     f'2>&1 | tee run.log\n')
        fh.write('\necho "done"\n')
    os.chmod(runner, 0o755)
    log(f"\nwrote {runner}  ({len(made)} events)")


if __name__ == '__main__':
    main()
