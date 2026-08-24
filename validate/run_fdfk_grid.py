#!/usr/bin/env python3
"""
run_fdfk_grid.py
================
End-to-end driver: take a jinv2024-style grid directory (start.mod, obs.tw,
observed SAC .r/.z), run FDFK2D in 1D mode, and write synthetic .r/.z back into
the SAME grid directory, in the SAME SAC format as your reference syn files
(delta = 0.1 s, npts = 400).

Everything is path-configurable. Output goes to <grid_dir>/fdfk_out/ by default.

Typical use on your Dell:
    python3 run_fdfk_grid.py \
        --grid_dir /home/yaoj/C++/grid.jinv2024/grid-1 \
        --fdfk_bin FDFK2D \
        --event_station 20150323045138_WB12 \
        --ray_param 6.51 \
        --src_lat -18.353 --src_lon -69.166

Stages (each can be skipped with --skip):
    1 build   : start.mod -> FK files + SU grid + FD/Source/Receiver/inpar
    2 run     : call FDFK2D (writes seisx.su/seisz.su)
    3 post    : SU -> SAC .r/.z at 0.1 s / 400 pts, matching the syn spec
    4 validate: overlay vs obs (and vs syn if present)

NOTE on the .df file: that is a derivative/sensitivity matrix (400 x Nparam)
from your CPS joint inversion. FDFK2D does NOT produce it -- an equivalent
would need adjoint/finite-difference gradients, which is the later 2D-inversion
project, not this forward step. This script only reproduces the .r/.z waveforms.
"""

import numpy as np
import argparse
import subprocess
import os
import sys
from scipy.signal import butter, filtfilt


# ======================================================================
# SAC I/O
# ======================================================================
def read_sac(f):
    raw = open(f, 'rb').read()
    F = np.frombuffer(raw[:280], dtype='<f4').copy()
    I = np.frombuffer(raw[280:440], dtype='<i4').copy()
    delta = float(F[0]); b = float(F[5]); npts = int(I[9])
    data = np.frombuffer(raw[632:632 + npts * 4], dtype='<f4').copy()
    return dict(delta=delta, b=b, npts=npts, data=data, raw=raw[:632])


def write_sac(fname, delta, b, data, template_hdr=None):
    npts = len(data)
    if template_hdr is not None:
        raw = bytearray(template_hdr[:632])
    else:
        raw = bytearray(632)
        F = np.full(70, -12345.0, dtype='<f4'); I = np.full(40, -12345, dtype='<i4')
        raw[:280] = F.tobytes(); raw[280:440] = I.tobytes()
    F = np.frombuffer(bytes(raw[:280]), dtype='<f4').copy()
    I = np.frombuffer(bytes(raw[280:440]), dtype='<i4').copy()
    F[0] = delta; F[5] = b; F[6] = b + (npts - 1) * delta
    I[9] = npts; I[6] = 6; I[15] = 1; I[35] = 1
    raw[:280] = F.astype('<f4').tobytes(); raw[280:440] = I.astype('<i4').tobytes()
    with open(fname, 'wb') as fh:
        fh.write(bytes(raw[:632])); fh.write(data.astype('<f4').tobytes())


# ======================================================================
# SU I/O
# ======================================================================
def read_su(fname):
    raw = open(fname, 'rb').read()
    h = np.frombuffer(raw[:240], dtype='<i2')
    ns = int(h[57]); dt_us = int(h[58])
    dt = dt_us / 1e6 if dt_us > 0 else 0.01
    tb = 240 + ns * 4; ntr = len(raw) // tb
    d = np.zeros((ns, ntr), dtype=np.float32)
    for i in range(ntr):
        o = i * tb + 240
        d[:, i] = np.frombuffer(raw[o:o + ns * 4], dtype='<f4')
    return d, ns, dt, ntr


def write_su_model(fname, arr2d, dz_m):
    nz, nx = arr2d.shape
    with open(fname, 'wb') as f:
        for ix in range(nx):
            head = np.zeros(120, dtype='<u2')
            head[57] = nz; head[58] = int(dz_m * 1e3) & 0xffff
            f.write(head.tobytes())
            f.write(arr2d[:, ix].astype('<f4').tobytes())


# ======================================================================
# CPS model
# ======================================================================
def read_cps_mod(path):
    lines = open(path).read().splitlines()
    start = next((i + 1 for i, ln in enumerate(lines)
                  if ln.strip().startswith('H(KM)')), None)
    if start is None:
        raise ValueError("no 'H(KM)' header in .mod")
    H, VP, VS, RHO = [], [], [], []
    for ln in lines[start:]:
        p = ln.split()
        if len(p) < 4:
            continue
        H.append(float(p[0])); VP.append(float(p[1]))
        VS.append(float(p[2])); RHO.append(float(p[3]))
    return map(np.array, (H, VP, VS, RHO))


def write_fk(path, H, VP, VS):
    ztop = np.zeros(len(H)); ztop[1:] = np.cumsum(H[:-1])
    with open(path, 'w') as f:
        f.write("nlayer\n"); f.write(f"  {len(H)}\n")
        f.write("Vp(m/s)   Vs(m/s)   z+(m)\n")
        for vp, vs, zt in zip(VP, VS, ztop):
            f.write(f"{vp*1000:.1f}    {vs*1000:.1f}    {zt*1000:.3f}\n")


def profile_on_grid(H, VP, VS, RHO, dz_km, zmax_km, npml):
    """Sample the 1D profile onto the FDFK2D depth grid.
    FDFK2D reader (Input.f90) expects, per trace:
        iz = izt .. nz+npml     with izt=1 (is_PML_top=.false.), nz=nint(Z/dz)+1
    i.e. nz+npml rows = the model grid PLUS npml bottom-PML padding rows.
    The padding repeats the mantle halfspace (matches create_one_layer_model.m)."""
    nz = int(round(zmax_km / dz_km)) + 1          # <-- +1 : matches nint()+1
    zc = np.arange(nz) * dz_km                     # node depths 0..Z
    ztop = np.zeros(len(H)); ztop[1:] = np.cumsum(H[:-1])
    zbot = np.append(ztop[1:], zmax_km + 1e6)
    vp = np.zeros(nz); vs = np.zeros(nz); rho = np.zeros(nz)
    for i in range(nz):
        idx = min(np.searchsorted(zbot, zc[i], side='right'), len(H) - 1)
        vp[i], vs[i], rho[i] = VP[idx], VS[idx], RHO[idx]
    # append npml bottom-PML rows (repeat deepest = mantle halfspace)
    vp = np.concatenate([vp,  np.full(npml, vp[-1])])
    vs = np.concatenate([vs,  np.full(npml, vs[-1])])
    rho = np.concatenate([rho, np.full(npml, rho[-1])])
    return vp, vs, rho     # length now nz+npml, exactly what the reader wants


# ======================================================================
# Stage 1: build inputs
# ======================================================================
def stage_build(a, indir):
    os.makedirs(indir, exist_ok=True)
    H, VP, VS, RHO = read_cps_mod(a.mod)
    total = float(np.cumsum(H[:-1])[-1])
    print(f"  model: {len(H)} layers, ~{total:.1f} km, mantle Vp={VP[-1]:.2f}")

    write_fk(f"{indir}/FK_model_left.dat",  H, VP, VS)
    write_fk(f"{indir}/FK_model_right.dat", H, VP, VS)

    zmax_km = a.Z / 1000.0
    vp1d, vs1d, rho1d = profile_on_grid(H, VP, VS, RHO, a.dz / 1000.0,
                                        zmax_km, a.npml)
    nx = int(round(a.X / a.dx)) + 1 + 2 * a.npml
    vp = np.tile(vp1d[:, None] * 1000, (1, nx))
    vs = np.tile(vs1d[:, None] * 1000, (1, nx))
    rho = np.tile(rho1d[:, None] * 1000, (1, nx))
    write_su_model(f"{indir}/vp_model.su",  vp,  a.dz)
    write_su_model(f"{indir}/vs_model.su",  vs,  a.dz)
    write_su_model(f"{indir}/rho_model.su", rho, a.dz)
    print(f"  grid nx={nx} nz={len(vp1d)}")

    # FD_model.dat
    with open(f"{indir}/FD_model.dat", 'w') as f:
        f.write("order of FD(2m>=2)\n  16\n")
        f.write("X0(m) Xn(m) Z0(m) Zn(m) Dx(m) Dz(m) Dt(s)\n")
        f.write(f"  0  {int(a.X)}  0  {int(a.Z)}  {int(a.dx)}  {int(a.dz)}  {a.dt}\n")
        f.write("tmax(s) tstep nPML is_PML_top\n")
        f.write(f"  {a.tmax}  {a.dt}  {a.npml}  .false.\n")
        f.write("lat(X0,Z0) lon(X0,Z0) azimuth\n")
        f.write(f"  {a.prof_lat}  {a.prof_lon}  {a.prof_az}\n")
        f.write("snapshot? nstep\n  .false.  100\n")

    # Source.dat  (verified format: ray_param s/deg, src_lat, src_lon)
    ptype = ".true." if a.phase.upper() == 'P' else ".false."
    with open(f"{indir}/Source.dat", 'w') as f:
        f.write("f0(Hz)   strength(m)   source type(P:.true.;S:.false.)\n")
        f.write(f"   {a.f0}     1.d-3        {ptype}\n")
        f.write("ray parameter(s/deg)   src_lat(deg)   src_lon(deg)\n")
        f.write(f"  {a.ray_param}   {a.src_lat}   {a.src_lon}\n")

    # Receiver.dat (dense array; target station at mid-array)
    rx = np.linspace(0, a.X, a.nrec)
    with open(f"{indir}/Receiver.dat", 'w') as f:
        f.write("number of receivers:\n")
        f.write(f"     {a.nrec}\n")
        f.write("rx(m)   rz(m)\n")
        for x in rx:
            f.write(f"{x:.0f}\t    0\n")

    with open(f"{indir}/inpar.dat", 'w') as f:
        f.write("FD_model.dat\nSource.dat\nReceiver.dat\n"
                "FK_model_left.dat\nFK_model_right.dat\n"
                "vp_model.su\nvs_model.su\n")
    print(f"  inputs written to {indir}")


# ======================================================================
# Stage 2: run FDFK2D
# ======================================================================
def stage_run(a, indir, seisdir):
    os.makedirs(seisdir, exist_ok=True)
    # FDFK2D <input_dir> <inpar> <output_dir> <prefix>
    cmd = [a.fdfk_bin, indir, "inpar.dat", seisdir, "seis"]
    print(f"  running: {' '.join(cmd)}")
    # FDFK2D reads inpar-listed files relative to input_dir; run from indir's parent
    r = subprocess.run(cmd, cwd=os.path.dirname(indir) or '.',
                       capture_output=False)
    if r.returncode != 0:
        print("  FDFK2D returned non-zero; check output above.")
        sys.exit(1)


# ======================================================================
# Stage 3: post-process SU -> SAC matching syn
# ======================================================================
def decimate_to(tr, dt_in, dt_out):
    fac = int(round(dt_out / dt_in))
    b, a_ = butter(6, (0.5 / dt_out) / (0.5 / dt_in), btype='low')
    return filtfilt(b, a_, tr)[::fac], fac


def detect_arrival(zr, dtz, search_after=0.0, mode='peak'):
    """Find the arrival sample in the FDFK trace.

    The FDFK P/S arrival does NOT sit at t=0 -- for S it lands ~30 s into the
    40 s simulation. CPS rcvFn aligns t=0 to the direct-arrival PEAK, so 'peak'
    (the default) matches that convention. 'onset' finds the wavetrain start
    instead (useful if you want the leading edge).
    """
    env = np.abs(zr)
    i_after = int(round(search_after / dtz))
    e = env.copy(); e[:i_after] = 0.0
    if mode == 'peak':
        return int(np.argmax(e))          # global peak after the transient
    # onset mode: first sustained rise above baseline, refined to nearby peak
    base = np.median(env[:max(i_after, 50)]) + 1e-12
    thr = max(5.0 * base, 0.15 * np.max(e))
    above = np.where(e > thr)[0]
    if len(above) == 0:
        return int(np.argmax(e))
    onset = above[0]
    w = int(round(3.0 / dtz))
    seg = e[onset:onset + w]
    return onset + int(np.argmax(seg)) if len(seg) else onset


def stage_post(a, seisdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    X, nsx, dtx, ntrx = read_su(f"{seisdir}/seisx.su")
    Z, nsz, dtz, ntrz = read_su(f"{seisdir}/seisz.su")
    ri = a.rec_index if a.rec_index >= 0 else ntrx // 2

    # target spec: from syn if present, else default 0.1s/400/b=-12.44
    ref = None
    for cand in (os.path.join(a.grid_dir, f"syn_{a.event_station}.r"),
                 os.path.join(a.grid_dir, f"syn.{a.event_station}.r")):
        if os.path.exists(cand):
            ref = read_sac(cand); print(f"  matching syn spec from {cand}")
            break
    if ref is None:
        ref = dict(delta=0.1, b=-12.439126, npts=400, raw=None,
                   data=np.zeros(400, dtype=np.float32))
        print("  no syn found; using default 0.1 s / 400 pts / b=-12.44")
    d_delta, d_b, d_npts = ref['delta'], ref['b'], ref['npts']
    tmpl = ref['raw']

    xr, fac = decimate_to(X[:, ri].astype(float), dtx, d_delta)
    zr, _ = decimate_to(Z[:, ri].astype(float), dtz, d_delta)

    # ---- locate the arrival in the FDFK trace ----
    if a.fdfk_arr >= 0.0:
        i_arr = int(round(a.fdfk_arr / d_delta))          # user-specified time (s)
        src = "user --fdfk_arr"
    else:
        i_arr = detect_arrival(zr, d_delta, a.search_after, a.detect)  # auto
        src = f"auto ({a.detect})"
    print(f"  FDFK arrival {src} at t={i_arr * d_delta:.2f}s "
          f"(sample {i_arr} in decimated trace)")

    # window so the arrival sits where the syn's arrival sits: syn b<0 means the
    # window starts |b| seconds BEFORE the arrival. Match that pre-arrival lead.
    pre_samples = int(round((0.0 - d_b) / d_delta))       # samples before arrival
    start = i_arr - pre_samples

    def cut(v):
        out = np.zeros(d_npts, dtype=np.float32)
        s = max(start, 0); e = min(start + d_npts, len(v)); off = s - start
        out[off:off + (e - s)] = v[s:e]
        return out

    sgn_z = -1.0 if a.flip_z else 1.0                     # Z polarity convention
    sgn_r = -1.0 if a.flip_r else 1.0
    out_r = os.path.join(outdir, f"fdfk_{a.event_station}.r")
    out_z = os.path.join(outdir, f"fdfk_{a.event_station}.z")
    write_sac(out_r, d_delta, d_b, sgn_r * cut(xr), tmpl)
    write_sac(out_z, d_delta, d_b, sgn_z * cut(zr), tmpl)
    print(f"  wrote {out_r}")
    print(f"  wrote {out_z}   (npts={d_npts}, matches syn)")


# ======================================================================
# Stage 4: validate
# ======================================================================
def stage_validate(a, outdir):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt

    def m(x, y):
        x = x / (np.max(np.abs(x)) + 1e-30); y = y / (np.max(np.abs(y)) + 1e-30)
        return np.corrcoef(x, y)[0, 1], np.sqrt(np.mean((x - y) ** 2))

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for row, c in enumerate(['z', 'r']):
        fd = read_sac(os.path.join(outdir, f"fdfk_{a.event_station}.{c}"))
        tf = fd['b'] + np.arange(fd['npts']) * fd['delta']
        # vs syn
        syn = None
        for cand in (f"syn_{a.event_station}.{c}", f"syn.{a.event_station}.{c}"):
            p = os.path.join(a.grid_dir, cand)
            if os.path.exists(p):
                syn = read_sac(p); break
        if syn:
            ts = syn['b'] + np.arange(syn['npts']) * syn['delta']
            cc, nr = m(fd['data'], syn['data'][:fd['npts']] if syn['npts'] >= fd['npts'] else np.pad(syn['data'], (0, fd['npts'] - syn['npts'])))
            ax[row, 0].plot(ts, syn['data'] / (np.max(np.abs(syn['data'])) + 1e-30), 'k', lw=1.2, label='syn')
            ax[row, 0].plot(tf, fd['data'] / (np.max(np.abs(fd['data'])) + 1e-30), 'r--', lw=1.0, label='FDFK2D')
            ax[row, 0].set_title(f"{c.upper()} FDFK vs syn  CC={cc:.3f} nRMS={nr:.3f}")
        else:
            ax[row, 0].plot(tf, fd['data'], 'r'); ax[row, 0].set_title(f"{c.upper()} FDFK (no syn)")
        ax[row, 0].legend(fontsize=8); ax[row, 0].grid(alpha=0.3)
        # vs obs
        obs = None
        for cand in (f"{a.event_station}.{c}",):
            p = os.path.join(a.grid_dir, cand)
            if os.path.exists(p):
                obs = read_sac(p); break
        if obs:
            to = obs['b'] + np.arange(obs['npts']) * obs['delta']
            ax[row, 1].plot(to, obs['data'] / (np.max(np.abs(obs['data'])) + 1e-30), 'b', lw=0.7, alpha=0.7, label='obs')
            ax[row, 1].plot(tf, fd['data'] / (np.max(np.abs(fd['data'])) + 1e-30), 'r', lw=1.0, label='FDFK2D')
            ax[row, 1].set_title(f"{c.upper()} FDFK vs observed"); ax[row, 1].legend(fontsize=8)
        ax[row, 1].grid(alpha=0.3)
    ax[1, 0].set_xlabel('Time (s)'); ax[1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    out = os.path.join(outdir, f"validation_{a.event_station}.png")
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print(f"  wrote {out}")
    print("  GATE: CC(FDFK vs syn) > ~0.95 on both -> 1D benchmark PASSES.")


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid_dir', required=True,
                    help='e.g. /home/yaoj/C++/grid.jinv2024/grid-1')
    ap.add_argument('--event_station', required=True,
                    help='e.g. 20150323045138_WB12')
    ap.add_argument('--fdfk_bin', default='FDFK2D')
    ap.add_argument('--mod', default=None, help='default <grid_dir>/start.mod')
    ap.add_argument('--outsub', default='fdfk_out',
                    help='output subdir inside grid_dir')
    # geometry
    ap.add_argument('--ray_param', type=float, default=6.51, help='s/deg')
    ap.add_argument('--src_lat', type=float, default=-18.353)
    ap.add_argument('--src_lon', type=float, default=-69.166)
    ap.add_argument('--phase', default='P')
    ap.add_argument('--f0', type=float, default=1.0)
    ap.add_argument('--prof_lat', type=float, default=38.6915)
    ap.add_argument('--prof_lon', type=float, default=-88.5652)
    ap.add_argument('--prof_az', type=float, default=158.62)
    # grid
    ap.add_argument('--X', type=float, default=200000.0)
    ap.add_argument('--Z', type=float, default=80000.0)
    ap.add_argument('--dx', type=float, default=200.0)
    ap.add_argument('--dz', type=float, default=200.0)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--tmax', type=float, default=40.0)
    ap.add_argument('--npml', type=int, default=20)
    ap.add_argument('--nrec', type=int, default=401)
    ap.add_argument('--rec_index', type=int, default=-1,
                    help='receiver col for the station; -1 = mid-array')
    # post
    ap.add_argument('--fdfk_arr', type=float, default=-1.0,
                    help='arrival time (s) in the FDFK trace to window around. '
                         '-1 = auto-detect (recommended; S lands ~30 s in, not 0)')
    ap.add_argument('--search_after', type=float, default=2.0,
                    help='ignore the first N s when auto-detecting the arrival '
                         '(skips the FK injection transient at t~0)')
    ap.add_argument('--detect', choices=['peak', 'onset'], default='peak',
                    help="'peak' matches CPS (aligns t=0 to the arrival peak); "
                         "'onset' finds the wavetrain leading edge")
    ap.add_argument('--flip_z', action='store_true',
                    help='flip Z polarity to match CPS convention '
                         '(needed: FDFK seisz sign is opposite CPS)')
    ap.add_argument('--flip_r', action='store_true',
                    help='flip R polarity if needed')
    ap.add_argument('--skip', default='', help='comma list: build,run,post,validate')
    args = ap.parse_args()

    args.grid_dir = os.path.abspath(args.grid_dir)
    if args.mod is None:
        args.mod = os.path.join(args.grid_dir, 'start.mod')
    indir = os.path.join(args.grid_dir, args.outsub, 'input')
    seisdir = os.path.join(args.grid_dir, args.outsub, 'seismograms')
    outdir = os.path.join(args.grid_dir, args.outsub)
    skip = set(s.strip() for s in args.skip.split(',') if s.strip())

    print(f"grid_dir = {args.grid_dir}")
    print(f"output   = {outdir}\n")

    if 'build' not in skip:
        print("[1/4] build inputs"); stage_build(args, indir)
    if 'run' not in skip:
        print("[2/4] run FDFK2D"); stage_run(args, indir, seisdir)
    if 'post' not in skip:
        print("[3/4] post-process to SAC"); stage_post(args, seisdir, outdir)
    if 'validate' not in skip:
        print("[4/4] validate"); stage_validate(args, outdir)
    print("\nDONE.")


if __name__ == '__main__':
    main()
