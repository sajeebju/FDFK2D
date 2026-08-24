#!/usr/bin/env python3
"""
build_2d_model.py
=================
Build a CONTINUOUS 2D velocity model for FDFK2D from your 152 grid 1D models
(grid-151, grid-149, ..., grid-1, grid1, ..., grid151), each containing a CPS
start.mod.

WHY NOT RAW STITCHING: the 152 columns have DIFFERENT layer structures (different
thicknesses, Moho depths, layer counts). You cannot place them side by side and
interpolate directly -- the layering doesn't line up. So we:

  1. Parse each grid's start.mod.
  2. Resample EACH column onto a COMMON fine depth grid (dz).  -> Vp(z),Vs(z),rho(z)
  3. Place each column at its x-position (grid number -> km along array).
  4. Interpolate laterally onto FDFK2D's fine x-grid, with smoothing so interfaces
     DIP smoothly between grids (this is what makes it a true 2D model, not
     stacked-1D -- the dip is the physics FDFK2D adds over rcvFn).
  5. Emit FDFK2D SU files (vp/vs/rho) + FK left/right end columns.

USAGE:
  python3 build_2d_model.py \
      --grids_root /home/yaoj/C++/grid.jinv2024 \
      --out input_2d \
      --x0 -151 --x1 151 --km_per_grid 1.0 \
      --dx 200 --dz 200 --Zkm 80 --npml 20 \
      --lat_smooth_km 5

  Then point run_fdfk_grid.py at this model (skip its build stage; use these files).

NOTE on x-mapping: grid number N -> x_km = N * km_per_grid. With grids -151..151
(step 2) over ~300 km, km_per_grid ~1.0 gives a 302 km span. Adjust to your real
station coordinates if the grid numbers are not exactly km.
"""
import numpy as np
import argparse
import os
import glob
import re


# ----------------------------------------------------------------------
def read_cps_mod(path):
    lines = open(path).read().splitlines()
    start = next((i + 1 for i, l in enumerate(lines)
                  if l.strip().startswith('H(KM)')), None)
    if start is None:
        raise ValueError(f"no H(KM) header in {path}")
    H, VP, VS, RHO = [], [], [], []
    for l in lines[start:]:
        p = l.split()
        if len(p) < 4:
            continue
        H.append(float(p[0])); VP.append(float(p[1]))
        VS.append(float(p[2])); RHO.append(float(p[3]))
    return np.array(H), np.array(VP), np.array(VS), np.array(RHO)


def column_on_depth_grid(H, VP, VS, RHO, dz_km, zmax_km):
    """Resample a layered model onto regular depth nodes 0..zmax (step dz).
    Last layer (H=0) is the halfspace filling everything below."""
    nz = int(round(zmax_km / dz_km)) + 1
    zc = np.arange(nz) * dz_km
    ztop = np.zeros(len(H)); ztop[1:] = np.cumsum(H[:-1])
    zbot = np.append(ztop[1:], zmax_km + 1e6)
    vp = np.zeros(nz); vs = np.zeros(nz); rho = np.zeros(nz)
    for i in range(nz):
        idx = min(np.searchsorted(zbot, zc[i], side='right'), len(H) - 1)
        vp[i], vs[i], rho[i] = VP[idx], VS[idx], RHO[idx]
    return vp, vs, rho


def find_grid_dirs(root):
    """Return sorted list of (grid_number, path_to_start.mod)."""
    out = []
    for d in glob.glob(os.path.join(root, 'grid*')):
        base = os.path.basename(d)
        m = re.match(r'grid(-?\d+)$', base)
        if not m:
            continue
        mod = os.path.join(d, 'start.mod')
        if os.path.exists(mod):
            out.append((int(m.group(1)), mod))
    out.sort(key=lambda t: t[0])
    return out


def gaussian_smooth_lat(field, dx_km, smooth_km):
    """Smooth each depth row laterally with a Gaussian (std=smooth_km)."""
    if smooth_km <= 0:
        return field
    sig = smooth_km / dx_km
    half = int(np.ceil(3 * sig))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= k.sum()
    out = np.empty_like(field)
    for iz in range(field.shape[0]):
        out[iz] = np.convolve(field[iz], k, mode='same')
    return out


def write_su(path, arr2d, dz_m):
    nz, nx = arr2d.shape
    with open(path, 'wb') as f:
        for ix in range(nx):
            head = np.zeros(120, dtype='<u2')
            head[57] = nz
            head[58] = int(dz_m * 1e3) & 0xffff
            f.write(head.tobytes())
            f.write(arr2d[:, ix].astype('<f4').tobytes())
    print(f"  wrote {path}  (nz={nz}, nx={nx})")


def write_fk_from_column(path, vp_col, vs_col, dz_km):
    """Write an FK layered file from a resampled column (one value per dz node).
    Collapse consecutive identical samples into layers for compactness."""
    nz = len(vp_col)
    # build layers where velocity changes
    tops = [0]
    for i in range(1, nz):
        if abs(vp_col[i] - vp_col[i-1]) > 1e-3 or abs(vs_col[i] - vs_col[i-1]) > 1e-3:
            tops.append(i)
    with open(path, 'w') as f:
        f.write("nlayer\n")
        f.write(f"  {len(tops)}\n")
        f.write("Vp(m/s)   Vs(m/s)   z+(m)\n")
        for t in tops:
            f.write(f"{vp_col[t]*1000:.1f}    {vs_col[t]*1000:.1f}    {t*dz_km*1000:.3f}\n")
    print(f"  wrote {path}  ({len(tops)} layers)")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grids_root', required=True,
                    help='dir containing grid-151 ... grid151 subdirs')
    ap.add_argument('--out', default='input_2d')
    ap.add_argument('--km_per_grid', type=float, default=1.0,
                    help='x_km = grid_number * km_per_grid')
    ap.add_argument('--dx', type=float, default=200.0, help='FD grid dx (m)')
    ap.add_argument('--dz', type=float, default=200.0, help='FD grid dz (m)')
    ap.add_argument('--Zkm', type=float, default=80.0, help='model depth (km)')
    ap.add_argument('--npml', type=int, default=20)
    ap.add_argument('--lat_smooth_km', type=float, default=5.0,
                    help='lateral Gaussian smoothing std (km); 0 = none')
    ap.add_argument('--x_pad_km', type=float, default=20.0,
                    help='extra model width beyond the grid span, each side (km)')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dz_km = args.dz / 1000.0

    # 1. find and read all grid models
    grids = find_grid_dirs(args.grids_root)
    if not grids:
        raise SystemExit(f"No grid*/start.mod found under {args.grids_root}")
    print(f"found {len(grids)} grid models: "
          f"{grids[0][0]} .. {grids[-1][0]}")

    # 2. resample each onto common depth grid
    xs = []          # x position (km) of each column
    cols_vp = []; cols_vs = []; cols_rho = []
    for gnum, path in grids:
        H, VP, VS, RHO = read_cps_mod(path)
        vp, vs, rho = column_on_depth_grid(H, VP, VS, RHO, dz_km, args.Zkm)
        xs.append(gnum * args.km_per_grid)
        cols_vp.append(vp); cols_vs.append(vs); cols_rho.append(rho)
    xs = np.array(xs)
    nz = len(cols_vp[0])
    Vp = np.array(cols_vp).T   # (nz, ncol)
    Vs = np.array(cols_vs).T
    Rho = np.array(cols_rho).T
    print(f"  column depth samples nz={nz}, grid x range {xs.min():.1f}..{xs.max():.1f} km")

    # 3. define FDFK2D fine x-grid (with padding + PML handled by nx convention)
    x_lo = xs.min() - args.x_pad_km
    x_hi = xs.max() + args.x_pad_km
    Xspan_m = (x_hi - x_lo) * 1000.0
    nx_model = int(round(Xspan_m / args.dx)) + 1
    x_fine = np.linspace(x_lo, x_hi, nx_model)   # km

    # 4. lateral interpolation onto the fine grid, per depth row
    Vp_f = np.empty((nz, nx_model)); Vs_f = np.empty((nz, nx_model)); Rho_f = np.empty((nz, nx_model))
    order = np.argsort(xs)
    for iz in range(nz):
        Vp_f[iz]  = np.interp(x_fine, xs[order], Vp[iz][order])
        Vs_f[iz]  = np.interp(x_fine, xs[order], Vs[iz][order])
        Rho_f[iz] = np.interp(x_fine, xs[order], Rho[iz][order])
    # lateral smoothing so interfaces dip smoothly
    Vp_f  = gaussian_smooth_lat(Vp_f,  args.dx/1000.0, args.lat_smooth_km)
    Vs_f  = gaussian_smooth_lat(Vs_f,  args.dx/1000.0, args.lat_smooth_km)
    Rho_f = gaussian_smooth_lat(Rho_f, args.dx/1000.0, args.lat_smooth_km)

    # 5. pad columns with npml on each side (edge replicate) + bottom PML rows
    def pad(field):
        left  = np.repeat(field[:, :1],  args.npml, axis=1)
        right = np.repeat(field[:, -1:], args.npml, axis=1)
        f = np.hstack([left, field, right])          # x-PML
        bot = np.repeat(f[-1:, :], args.npml, axis=0) # z bottom PML
        return np.vstack([f, bot])
    Vp_g, Vs_g, Rho_g = pad(Vp_f), pad(Vs_f), pad(Rho_f)

    # write SU (convert km/s -> m/s, g/cc -> kg/m3)
    write_su(f"{args.out}/vp_model.su",  Vp_g*1000,  args.dz)
    write_su(f"{args.out}/vs_model.su",  Vs_g*1000,  args.dz)
    write_su(f"{args.out}/rho_model.su", Rho_g*1000, args.dz)

    # FK left/right = the actual leftmost/rightmost MODEL columns (not PML)
    write_fk_from_column(f"{args.out}/FK_model_left.dat",  Vp_f[:, 0],  Vs_f[:, 0],  dz_km)
    write_fk_from_column(f"{args.out}/FK_model_right.dat", Vp_f[:, -1], Vs_f[:, -1], dz_km)

    # FD_model.dat geometry to match
    Xn = round(Xspan_m)
    Zn = int(args.Zkm * 1000)
    with open(f"{args.out}/FD_model.dat", 'w') as f:
        f.write("order of FD(2m>=2)\n  16\n")
        f.write("X0(m) Xn(m) Z0(m) Zn(m) Dx(m) Dz(m) Dt(s)\n")
        f.write(f"  0  {Xn}  0  {Zn}  {int(args.dx)}  {int(args.dz)}  0.01\n")
        f.write("tmax(s) tstep nPML is_PML_top\n")
        f.write(f"  40.0  0.01  {args.npml}  .false.\n")
        f.write("lat(X0,Z0) lon(X0,Z0) azimuth\n")
        f.write("  38.6915  -88.5652  158.62\n")
        f.write("snapshot? nstep\n  .false.  100\n")

    print(f"\nDONE. 2D model in {args.out}/")
    print(f"  model width: {x_lo:.1f} .. {x_hi:.1f} km  ({nx_model} cols + {2*args.npml} PML)")
    print(f"  station x-positions map via grid_number * {args.km_per_grid} km")
    print(f"  IMPORTANT: verify km_per_grid and the station->x mapping match your")
    print(f"             real coordinates, and set receiver x-positions accordingly.")

    if args.plot:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        z = np.arange(nz) * dz_km
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(Vs_f, aspect='auto', cmap='viridis_r',
                       extent=[x_lo, x_hi, z[-1], 0])
        ax.set_xlabel('x along array (km)'); ax.set_ylabel('depth (km)')
        ax.set_title('2D Vs model from 152 grids (lateral interp + smooth)')
        plt.colorbar(im, label='Vs (km/s)')
        plt.tight_layout(); plt.savefig(f"{args.out}/model_2d.png", dpi=120)
        print(f"  wrote {args.out}/model_2d.png")


if __name__ == '__main__':
    main()
