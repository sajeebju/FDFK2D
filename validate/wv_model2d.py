#!/usr/bin/env python3
"""
wv_model2d.py
=============
The model side of the 2-D SsPmp inversion: turn a short parameter vector into
`vp_model.su` / `vs_model.su` for FDFK2D.

Two background modes
--------------------
**Background mode (use this).**  Point `--background-dir` at the model your
`build_2d_model.py` produced by laterally interpolating the `start.mod` files
out of the `grid*` directories.  That model carries everything the grid
inversion knows -- sediment thickness variations, lateral velocity changes,
the Moho shape itself -- and the parameters then become a *perturbation* to its
Moho:

    h(x) = h_background(x) + dh(x)

so m = 0 means "exactly your interpolated model" and the inversion starts from
your own result instead of from a generic column.  Background columns narrower
than the FD grid are padded with their edge column, which is how you use a
+/-171 km interpolation inside a -340..+245 km grid.

**Reference-column mode (fallback).**  With no background, a single 1-D column
is draped everywhere and the parameters are absolute Moho depths.  Only sensible
for synthetic tests.

Parameterisation
----------------
SsPmp is one post-critical Moho reflection.  What it actually constrains is
(a) the two-way P time through the crust, i.e. crustal thickness divided by
crustal Vp, and (b) the impedance contrast at the Moho through the post-critical
phase shift.  Anything more elaborate is not resolved by this phase, so the
model vector is deliberately small:

    m = [ h_1 ... h_N ,  (dvp) ,  (dvpvs) ]

    h_j     Moho depth [km] at control node x_j along the profile
    dvp     uniform crustal Vp perturbation [m/s]      (optional, --fit-vp)
    dvpvs   uniform crustal Vp/Vs perturbation         (optional, --fit-vpvs)

Between nodes h(x) is a monotone cubic (PCHIP) interpolant, so the Moho stays
smooth without the overshoot a plain cubic spline would give.

Crust is stretched, mantle is not
---------------------------------
A reference 1-D column (Vp, Vs vs depth, with a reference Moho h0) is mapped
into each output column by

    z < h(x):   z_ref = z * h0 / h(x)          (crust stretched/squeezed)
    z >= h(x):  z_ref = z - h(x) + h0          (mantle translated)

so crustal layer *proportions* and the mantle gradient are preserved while the
Moho moves.  This keeps the number of free parameters equal to the number of
things SsPmp can see.

Wings
-----
The station array spans about +/-152 km but SsPmp reflection points reach
-318 to +222 km, so the FD grid has to be much wider than the region your grid
inversion resolves.  Outside `--structure-halfwidth`, h(x) is held at its edge
value, i.e. the wings are 1-D.  That is the same assumption FK_model_left/right
already make at the boundaries, so it costs run time rather than credibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdfk2d_common import (write_su, read_su, ensure_dir, read_kv, apply_kv, explicit_dests,
                           require_paths)


# ---------------------------------------------------------------------------
def pchip(xk, yk, x):
    """Monotone cubic Hermite interpolation (Fritsch-Carlson), no SciPy needed."""
    xk = np.asarray(xk, float)
    yk = np.asarray(yk, float)
    x = np.asarray(x, float)
    n = len(xk)
    if n == 1:
        return np.full_like(x, yk[0])
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.zeros(n)
    m[0] = d[0]
    m[-1] = d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    xc = np.clip(x, xk[0], xk[-1])
    idx = np.clip(np.searchsorted(xk, xc) - 1, 0, n - 2)
    s = xc - xk[idx]
    hh = h[idx]
    t = s / hh
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)
    return (h00 * yk[idx] + h10 * hh * m[idx]
            + h01 * yk[idx + 1] + h11 * hh * m[idx + 1])


# ---------------------------------------------------------------------------
DEFAULT_REF = [
    # depth_top_km, Vp m/s, Vs m/s     -- generic WVSZ / Illinois basin starting column.
    (0.0,   4200.0, 2350.0),   # Paleozoic sediments
    (2.0,   6000.0, 3450.0),   # upper crust
    (12.0,  6350.0, 3650.0),
    (25.0,  6800.0, 3880.0),   # lower crust
    (52.0,  8050.0, 4600.0),   # mantle  <- reference Moho
    (80.0,  8200.0, 4680.0),
]


def detect_moho(vp, dz, vp_thresh=7500.0, zmin_m=25000.0, zmax_m=70000.0,
                method='gradient', win_m=2000.0):
    """
    Moho depth per column.

    method='gradient' (default): depth of the steepest Vp increase over a
        `win_m` window inside the search band.  This is the right choice for
        models interpolated from 1-D inversion results, which have a gradational
        lower crust rather than a sharp step.

    method='contour': first crossing of `vp_thresh`, refined to sub-grid depth.
        Looks safer but is not: in a jinv2024 start.mod column the LOWER CRUST
        already reaches 7.5 km/s, so the contour lands ~10 km above the real
        discontinuity.  On the Wabash model it gives a median of 46.3 km where
        the actual velocity jump -- and the Bayesian posterior -- is at 52.8 km.
    """
    nx, nz = vp.shape
    kmin = int(zmin_m / dz)
    kmax = min(nz - 1, int(zmax_m / dz))
    if method == 'gradient':
        w = max(1, int(win_m / dz))
        out = np.empty(nx)
        for i in range(nx):
            hi = min(kmax, nz - 1)
            lo = max(1, kmin)
            if hi - lo <= w:
                out[i] = 0.5 * (zmin_m + zmax_m) / 1000.0
                continue
            g = vp[i, lo + w:hi + 1] - vp[i, lo:hi + 1 - w]
            out[i] = (lo + int(np.argmax(g)) + 0.5 * w) * dz / 1000.0
        return out

    h = np.empty(nx)
    for i in range(nx):
        col = vp[i]
        k = None
        for j in range(max(1, kmin), kmax + 1):
            if col[j - 1] < vp_thresh <= col[j]:
                k = j
                break
        if k is not None:
            f = ((vp_thresh - col[k - 1]) / (col[k] - col[k - 1])
                 if col[k] != col[k - 1] else 0.0)
            h[i] = (k - 1 + f) * dz
        else:
            g = np.diff(col[kmin:kmax + 1])
            h[i] = (kmin + int(np.argmax(g))) * dz if len(g) else 0.5 * (zmin_m + zmax_m)
    return h / 1000.0


def repair_edges(vp, vs, frac=0.85, log=None):
    """
    Repair columns whose velocities have been faded toward zero at the model
    edges.

    Lateral smoothing implemented with zero padding (rather than edge
    replication) halves the amplitude at the outermost column and tapers back
    to normal over roughly the kernel width.  On the Wabash model the outer
    ~6 km on each side comes out at Vp 1430-7400 m/s where the mantle should be
    8200-8500 -- and those are exactly the columns that get copied into the PML
    and extracted as FK_model_left/right, so the error lands on the FK/PML
    boundary where it does the most damage.

    Any column whose peak Vp falls below `frac` of the interior median is
    replaced by the nearest healthy column.  Returns (vp, vs, n_left, n_right).
    """
    nx = vp.shape[0]
    peak = vp.max(1)
    ref = float(np.median(peak[nx // 4:3 * nx // 4]))
    good = np.where(peak >= frac * ref)[0]
    if len(good) == 0:
        raise ValueError('every column looks faded; check the model')
    lo, hi = int(good[0]), int(good[-1])
    vp = vp.copy()
    vs = vs.copy()
    if lo > 0:
        vp[:lo] = vp[lo]
        vs[:lo] = vs[lo]
    if hi < nx - 1:
        vp[hi + 1:] = vp[hi]
        vs[hi + 1:] = vs[hi]
    if log and (lo or hi < nx - 1):
        log(f"  [repair] {lo} columns on the left and {nx - 1 - hi} on the right "
            f"had peak Vp below {frac:.0%} of the interior median "
            f"({ref:.0f} m/s); replaced with the nearest healthy column")
    return vp, vs, lo, nx - 1 - hi


class Background:
    """
    The laterally interpolated model from build_2d_model.py, resampled onto the
    FD grid and with its own Moho detected so the parameters can perturb it.
    """

    def __init__(self, model_dir, x0_km, dx_bg, dz_bg, npml_bg,
                 vp_name='vp_model.su', vs_name='vs_model.su', ns=None,
                 vp_thresh=7500.0, moho_method='gradient', repair=True,
                 edge_frac=0.85, log=None):
        vp, _ = read_su(os.path.join(model_dir, vp_name), ns=ns)
        vs, _ = read_su(os.path.join(model_dir, vs_name), ns=ns)
        if vp.shape != vs.shape:
            raise ValueError(f'vp {vp.shape} and vs {vs.shape} differ')
        ntr, nzs = vp.shape
        nx = ntr - 2 * npml_bg
        nz = nzs - npml_bg
        self.vp = vp[npml_bg:npml_bg + nx, :nz]
        self.vs = vs[npml_bg:npml_bg + nx, :nz]
        self.n_repaired = (0, 0)
        if repair:
            self.vp, self.vs, nl, nr = repair_edges(self.vp, self.vs,
                                                    edge_frac, log)
            self.n_repaired = (nl, nr)
        self.dx = float(dx_bg)
        self.dz = float(dz_bg)
        self.x = float(x0_km) + np.arange(nx) * self.dx / 1000.0
        self.z = np.arange(nz) * self.dz / 1000.0
        self.moho = detect_moho(self.vp, self.dz, vp_thresh,
                                method=moho_method)
        self.dir = model_dir

    def column(self, x_km):
        """Nearest background column, clamped -> edge columns pad the wings."""
        return int(np.clip(round((x_km - self.x[0]) / (self.dx / 1000.0)),
                           0, len(self.x) - 1))

    def sample(self, i, z_km):
        """Background Vp, Vs at reference depths z_km in column i."""
        return (np.interp(z_km, self.z, self.vp[i]),
                np.interp(z_km, self.z, self.vs[i]))

    def __repr__(self):
        r = (f", repaired {self.n_repaired[0]}L/{self.n_repaired[1]}R cols"
             if any(self.n_repaired) else "")
        return (f"Background({os.path.basename(self.dir)}: {self.vp.shape}, "
                f"x {self.x[0]:.0f}..{self.x[-1]:.0f} km, "
                f"Moho {self.moho.min():.1f}..{self.moho.max():.1f} km"
                f" median {np.median(self.moho):.1f}{r})")


class Model2D:
    def __init__(self, ref_layers=None, h0_km=None, x_nodes_km=None,
                 dx=200.0, dz=200.0, x0_km=-340.0, x1_km=245.0, Zkm=80.0,
                 npml=20, structure_halfwidth_km=152.0, background=None,
                 dh_max_km=8.0):
        self.bg = background
        self.dh_max = float(dh_max_km)
        self.ref = np.array(ref_layers if ref_layers is not None else DEFAULT_REF, float)
        self.h0 = float(h0_km) if h0_km else self._detect_moho()
        self.dx = float(dx)
        self.dz = float(dz)
        self.x0_km = float(x0_km)
        self.x1_km = float(x1_km)
        self.Zkm = float(Zkm)
        self.npml = int(npml)
        self.hw = float(structure_halfwidth_km)
        self.x_nodes = (np.asarray(x_nodes_km, float) if x_nodes_km is not None
                        else np.arange(-self.hw, self.hw + 1e-6, 40.0))
        self.nx = int(round((self.x1_km - self.x0_km) * 1000.0 / self.dx)) + 1
        self.nz = int(round(self.Zkm * 1000.0 / self.dz)) + 1

    def _detect_moho(self):
        v = self.ref[:, 1]
        k = int(np.argmax(v >= 7500.0))
        return float(self.ref[k, 0]) if v.max() >= 7500.0 else 45.0

    # ---------------- reference column sampling ----------------
    def _ref_at(self, z_km):
        """Piecewise-constant reference column at reference depth z_km."""
        idx = np.clip(np.searchsorted(self.ref[:, 0], z_km, side='right') - 1,
                      0, len(self.ref) - 1)
        return self.ref[idx, 1], self.ref[idx, 2]

    # ---------------- parameter vector ----------------
    @property
    def n_nodes(self):
        return len(self.x_nodes)

    @property
    def is_background(self):
        return self.bg is not None

    @property
    def param_bounds(self):
        """(lo, hi) for the node parameters: dh in background mode, absolute h
        otherwise.  The optimisers clip against this."""
        if self.is_background:
            return -self.dh_max, self.dh_max
        return 30.0, 70.0

    def start_vector(self, fit_vp=False, fit_vpvs=False):
        # background mode: m = 0 means exactly the interpolated model
        m = list(np.zeros(self.n_nodes) if self.is_background
                 else np.full(self.n_nodes, self.h0))
        if fit_vp:
            m.append(0.0)
        if fit_vpvs:
            m.append(0.0)
        return np.array(m, float)

    def unpack(self, m, fit_vp=False, fit_vpvs=False):
        m = np.asarray(m, float)
        h = m[:self.n_nodes]
        k = self.n_nodes
        dvp = m[k] if fit_vp else 0.0
        k += int(fit_vp)
        dvpvs = m[k] if fit_vpvs else 0.0
        return h, float(dvp), float(dvpvs)

    def moho_profile(self, h_nodes):
        """
        h(x) on the FD column grid, held constant outside the structure width.
        In background mode the nodes carry dh and this returns
        h_background(x) + dh(x), so the interpolated Moho shape is preserved and
        only perturbed.
        """
        x = self.x0_km + np.arange(self.nx) * self.dx / 1000.0
        xc = np.clip(x, self.x_nodes[0], self.x_nodes[-1])
        dh = pchip(self.x_nodes, h_nodes, xc)
        if not self.is_background:
            return x, dh
        hbg = np.array([self.bg.moho[self.bg.column(xx)] for xx in x])
        return x, hbg + dh

    def background_moho(self):
        x = self.x0_km + np.arange(self.nx) * self.dx / 1000.0
        if not self.is_background:
            return x, np.full(self.nx, self.h0)
        return x, np.array([self.bg.moho[self.bg.column(xx)] for xx in x])

    # ---------------- grids ----------------
    def build(self, m, fit_vp=False, fit_vpvs=False):
        h_nodes, dvp, dvpvs = self.unpack(m, fit_vp, fit_vpvs)
        x, h = self.moho_profile(h_nodes)
        npml, nx, nz = self.npml, self.nx, self.nz
        z = np.arange(nz + npml) * self.dz / 1000.0        # includes bottom PML

        VP = np.empty((nx + 2 * npml, nz + npml), np.float32)
        VS = np.empty_like(VP)
        _, hbg = self.background_moho()
        for i in range(nx + 2 * npml):
            ii = min(max(i - npml, 0), nx - 1)             # side PMLs copy the edge column
            hi = h[ii]
            if self.is_background:
                # stretch this column's own crust from its background Moho to hi
                h_ref = hbg[ii]
                j = self.bg.column(x[ii])
                zref = np.where(z < hi, z * h_ref / hi, z - hi + h_ref)
                vp, vs = self.bg.sample(j, zref)
            else:
                h_ref = self.h0
                zref = np.where(z < hi, z * h_ref / hi, z - hi + h_ref)
                vp, vs = self._ref_at(zref)
            crust = z < hi
            vp = vp + np.where(crust, dvp, 0.0)
            vs = np.where(crust, vp / ((vp - dvp) / vs + dvpvs), vs)
            VP[i] = vp
            VS[i] = vs
        return VP, VS, x, h

    def write(self, out_dir, m, fit_vp=False, fit_vpvs=False, plot=False):
        ensure_dir(out_dir)
        VP, VS, x, h = self.build(m, fit_vp, fit_vpvs)
        write_su(os.path.join(out_dir, 'vp_model.su'), VP, 0.001)
        write_su(os.path.join(out_dir, 'vs_model.su'), VS, 0.001)
        meta = dict(x0_km=self.x0_km, x1_km=self.x1_km, dx=self.dx, dz=self.dz,
                    Zkm=self.Zkm, npml=self.npml, nx=self.nx, nz=self.nz,
                    h0_km=self.h0, x_nodes_km=self.x_nodes.tolist(),
                    m=np.asarray(m, float).tolist(),
                    fit_vp=bool(fit_vp), fit_vpvs=bool(fit_vpvs),
                    structure_halfwidth_km=self.hw,
                    ref_layers=None if self.is_background else self.ref.tolist(),
                    background_dir=(self.bg.dir if self.is_background else None),
                    background_moho_km=(self.background_moho()[1].tolist()
                                        if self.is_background else None),
                    moho_km=h.tolist())
        with open(os.path.join(out_dir, 'model_meta.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)
        if plot:
            self._plot(out_dir, VP, x, h)
        return meta

    def _plot(self, out_dir, VP, x, h):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        npml = self.npml
        interior = VP[npml:npml + self.nx, :self.nz].T
        z = np.arange(self.nz) * self.dz / 1000.0
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.pcolormesh(x, z, interior / 1000.0, shading='auto', cmap='viridis')
        if self.is_background:
            ax.plot(x, self.background_moho()[1], 'w--', lw=1.2,
                    label='background Moho (grid* start.mod)')
        ax.plot(x, h, 'r-', lw=1.5, label='Moho')
        ax.axvline(-self.hw, color='w', ls='--', lw=1)
        ax.axvline(self.hw, color='w', ls='--', lw=1, label='1-D wings outside')
        ax.invert_yaxis()
        ax.set_xlabel('profile x [km]')
        ax.set_ylabel('depth [km]')
        ax.legend(loc='lower right', fontsize=8)
        fig.colorbar(im, ax=ax, label='Vp [km/s]')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'model.png'), dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------------
def load_ref_from_file(path):
    """Three columns: depth_top_km  Vp_m/s  Vs_m/s   (comments with # ignored)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split('#')[0].split()
            if len(line) >= 3:
                rows.append(tuple(float(v) for v in line[:3]))
    if not rows:
        raise ValueError(f'no layers read from {path}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', default=None,
                    help='settings file with "key = value" lines; command-line flags win')
    ap.add_argument('--out', default=None)
    ap.add_argument('--background-dir', default=None,
                    help='directory holding the vp_model.su / vs_model.su produced by '
                         'build_2d_model.py (lateral interpolation of the grid* '
                         'start.mod files). Parameters then perturb ITS Moho, and '
                         'm = 0 reproduces it exactly.')
    ap.add_argument('--background-dx', type=float, default=None,
                    help='dx of the background model; default = --dx')
    ap.add_argument('--background-dz', type=float, default=None)
    ap.add_argument('--background-npml', type=int, default=None)
    ap.add_argument('--background-x0-km', type=float, default=None,
                    help='profile coordinate of the background model left edge; '
                         'default = --x0-km')
    ap.add_argument('--background-ns', type=int, default=None)
    ap.add_argument('--moho-vp', type=float, default=7500.0,
                    help='Vp contour, only used with --moho-method contour')
    ap.add_argument('--moho-method', default='gradient',
                    choices=['gradient', 'contour'],
                    help='how to pick the background Moho (default gradient)')
    ap.add_argument('--no-edge-repair', action='store_true')
    ap.add_argument('--edge-frac', type=float, default=0.85)
    ap.add_argument('--dh-max-km', type=float, default=8.0,
                    help='bound on the Moho perturbation in background mode')
    ap.add_argument('--ref-1d', default=None, help='depth_km Vp Vs table; default is built in')
    ap.add_argument('--h0', type=float, default=None, help='reference Moho depth [km]')
    ap.add_argument('--moho', nargs='*', type=float, default=None,
                    help='Moho depth at each node [km]; default = flat at h0')
    ap.add_argument('--node-spacing-km', type=float, default=40.0)
    ap.add_argument('--structure-halfwidth', type=float, default=152.0)
    ap.add_argument('--x0-km', type=float, default=-340.0)
    ap.add_argument('--x1-km', type=float, default=245.0)
    ap.add_argument('--dx', type=float, default=200.0)
    ap.add_argument('--dz', type=float, default=200.0)
    ap.add_argument('--Zkm', type=float, default=80.0)
    ap.add_argument('--npml', type=int, default=20)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    if args.paths:
        apply_kv(args, read_kv(args.paths), parser=ap,
                 explicit=explicit_dests(ap))
    require_paths(args, ['out'])

    ref = load_ref_from_file(args.ref_1d) if args.ref_1d else None
    bg = None
    if args.background_dir:
        bg = Background(args.background_dir,
                        args.background_x0_km if args.background_x0_km is not None
                        else args.x0_km,
                        args.background_dx or args.dx,
                        args.background_dz or args.dz,
                        args.background_npml if args.background_npml is not None
                        else args.npml,
                        ns=args.background_ns, vp_thresh=args.moho_vp,
                        moho_method=args.moho_method,
                        repair=not args.no_edge_repair,
                        edge_frac=args.edge_frac, log=print)
        print(bg)
    nodes = np.arange(-args.structure_halfwidth, args.structure_halfwidth + 1e-6,
                      args.node_spacing_km)
    mod = Model2D(ref, args.h0, nodes, args.dx, args.dz, args.x0_km, args.x1_km,
                  args.Zkm, args.npml, args.structure_halfwidth, bg, args.dh_max_km)
    m = np.array(args.moho, float) if args.moho else mod.start_vector()
    if len(m) != mod.n_nodes:
        raise SystemExit(f'--moho needs {mod.n_nodes} values (nodes at '
                         f'{np.round(mod.x_nodes, 1).tolist()})')
    meta = mod.write(args.out, m, plot=args.plot)
    print(f"model {mod.nx} x {mod.nz} (+{mod.npml} PML) -> {args.out}")
    print(f"  x {mod.x0_km:g} .. {mod.x1_km:g} km, z 0 .. {mod.Zkm:g} km, "
          f"dx={mod.dx:g} dz={mod.dz:g} m")
    print(f"  {mod.n_nodes} Moho nodes at {np.round(mod.x_nodes, 0).astype(int).tolist()} km")
    if mod.is_background:
        bgm = meta['background_moho_km']
        print(f"  background Moho {min(bgm):.1f} .. {max(bgm):.1f} km "
              f"(from {mod.bg.dir})")
    print(f"  model Moho      {min(meta['moho_km']):.1f} .. {max(meta['moho_km']):.1f} km")


if __name__ == '__main__':
    main()
