#!/usr/bin/env python3
"""
fdfk2d_common.py
================
Shared utilities for the Wabash Valley 2-D teleseismic (SsPmp) forward-modelling
pipeline built on FDFK2D (Liu et al., 2024, SRL, doi:10.1785/0220240231).

Contents
--------
  * readers for t8.dat / shift.dat / tpmp.xy / obs.tw
  * ProfileGeom      : least-squares recovery of the 2-D profile (origin + azimuth)
                       from the tpmp.xy (x, y) <-> (lon, lat) pairs
  * backaz()         : verbatim re-implementation of FDFK2D's src/calc_backazimuth.f90
  * fdfk2d_x_to_ll() : verbatim re-implementation of FDFK2D's coordinate mapping,
                       in BOTH the "as released" and the "corrected" variant
  * solve_az_org()   : chooses the az_org that makes FDFK2D's internal
                       (baz - az_org) equal the physically correct in-plane angle
  * read_su/write_su : the exact flavour of SU that FDFK2D writes/reads
  * signal helpers   : bandpass, cosine taper, envelope, S-arrival picking

Author: written for A. Habib's jinv2024 / WVSZ project.
"""

from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
R_EARTH_M = 6371.0e3          # FDFK2D hard-codes this (calc_backazimuth.f90)
R_EARTH_KM = 6371.0
FLATTENING = 1.0 / 298.257    # FDFK2D hard-codes this too
DEG2KM = np.pi * R_EARTH_KM / 180.0     # 111.1949...  -> matches FDFK2D's s/deg -> s/m
SEC_PER_DEG_TO_SEC_PER_KM = 1.0 / DEG2KM


# ============================================================================
# 1.  ASCII readers
# ============================================================================
def read_t8(path):
    """
    t8.dat  ->  {station: dict}

    Column layout (verified against the 20150323045138 Chile event):
        1  trace name, e.g. 'WB12.z'
        2  S arrival time in the trace's own time base [s]   (SUPERSEDED by shift.dat)
        3  epicentral distance GCARC [deg]
        4  back-azimuth station->event [deg]
        5  ray parameter [s/km]
    """
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            sta = p[0].split('.')[0]
            out[sta] = dict(station=sta,
                            t_s_t8=float(p[1]),
                            gcarc=float(p[2]),
                            baz=float(p[3]),
                            p_skm=float(p[4]))
    return out


def read_shift(path):
    """
    shift.dat -> {station: (t_S [s], flag)}

    Column 2 is the S arrival that MUST be used (hand-refined).
    Column 3 is a 0/1 flag whose meaning is not documented in the files you gave
    me -- it is carried through untouched and exposed via --shift-flag so you can
    filter on it once you confirm what it means.  Nothing in this pipeline
    depends on it by default.
    """
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 2:
                continue
            sta = p[0].split('.')[0]
            flag = int(float(p[2])) if len(p) > 2 else 1
            out[sta] = (float(p[1]), flag)
    return out


def read_tpmp(path):
    """
    tpmp.xy -> list of dicts.

        1  x  along-profile coordinate of the SsPmp bounce point [km]  (= grid id)
        2  y  profile-perpendicular coordinate of the bounce point [km]
        3  event id
        4  station
        5  observed SsPmp-S differential time [s]
        6  its uncertainty [s]
        7  bounce point longitude [deg east, 0-360]
        8  bounce point latitude  [deg]
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 8:
                continue
            lon = float(p[6])
            if lon > 180.0:
                lon -= 360.0
            rows.append(dict(x=float(p[0]), y=float(p[1]), event=p[2], station=p[3],
                             dt=float(p[4]), sigma=float(p[5]),
                             bp_lon=lon, bp_lat=float(p[7])))
    return rows


def read_obs_tw(path):
    """
    obs.tw (produced by tw_script.py) -> list of dicts

        1  <event>_<station>
        2  SAC user0 of the .z file
        3  S arrival taken from shift.dat
        4  pre-signal RMS noise
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            tag = p[0]
            ev, _, sta = tag.partition('_')

            def _f(v):
                try:
                    return float(v)
                except ValueError:
                    return np.nan
            rows.append(dict(tag=tag, event=ev, station=sta,
                             user0=_f(p[1]), t_s=_f(p[2]), rms=_f(p[3])))
    return rows


# ============================================================================
# 2.  Profile geometry
# ============================================================================
class ProfileGeom:
    """
    The profile frame used by GMT `project -C<clon>/<clat> -A<az>`, which is what
    built tpmp.xy: column 1 is `p` (great-circle distance along the profile from
    the -C centre) and column 2 is `q` (perpendicular distance).  x = 0 is the
    CENTRE of the profile, not an end.

    Preferred construction is from the actual PRF line in your para.mk:

        ProfileGeom.from_prf("-C-88.0/38.44 -A120.5")

    `from_tpmp` recovers it by least squares when you do not have that line;
    for the Wabash table it returns -C-88.000/38.440 -A120.50 with 1.02 km rms
    and 1.08 km maximum residual over a 540 km profile.
    """

    def __init__(self, lat0, lon0, azimuth, rms_km=None, n=None):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.azimuth = float(azimuth) % 360.0
        self.rms_km = rms_km
        self.n = n
        self._basis()

    def _basis(self):
        la = np.radians(self.lat0)
        lo = np.radians(self.lon0)
        a = np.radians(self.azimuth)
        self._c = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        east = np.array([-np.sin(lo), np.cos(lo), 0.0])
        north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
        self._x = np.sin(a) * east + np.cos(a) * north      # +p direction
        self._y = np.cross(self._c, self._x)                # +q direction

    # ---------------- construction ----------------
    @classmethod
    def from_prf(cls, prf):
        """
        Parse a GMT project option string, e.g. '-C-88.0/38.44 -A120.5'.
        Accepts either the whole string or an already-split token list.
        """
        if isinstance(prf, (list, tuple)):
            prf = ' '.join(str(t) for t in prf)
        clon = clat = az = None
        for tok in str(prf).replace(',', ' ').split():
            if tok.startswith('-C'):
                a, _, b = tok[2:].partition('/')
                clon, clat = float(a), float(b)
            elif tok.startswith('-A'):
                az = float(tok[2:])
        if clon is None or az is None:
            raise ValueError(f"could not parse -C and -A out of {prf!r}")
        return cls(clat, clon, az)

    @classmethod
    def from_tpmp(cls, tpmp_path, refine=True):
        rows = read_tpmp(tpmp_path)
        lat = np.array([r['bp_lat'] for r in rows])
        lon = np.array([r['bp_lon'] for r in rows])
        X = np.array([r['x'] for r in rows])
        Y = np.array([r['y'] for r in rows])

        def misfit(clat, clon, az):
            g = cls(clat, clon, az)
            p, q = g.ll2xy(lat, lon)
            return float(np.sqrt(np.mean((p - X) ** 2 + (q - Y) ** 2)))

        best = (np.inf, 38.5, -88.0, 120.0)
        for az in np.arange(110.0, 131.0, 0.5):
            for clat in np.arange(37.8, 39.2, 0.1):
                for clon in np.arange(-89.0, -87.0, 0.1):
                    r = misfit(clat, clon, az)
                    if r < best[0]:
                        best = (r, clat, clon, az)
        if refine:
            step = np.array([0.05, 0.05, 0.25])
            cur = np.array(best[1:], float)
            r = best[0]
            for _ in range(60):
                improved = False
                for k in range(3):
                    for sgn in (+1, -1):
                        t = cur.copy()
                        t[k] += sgn * step[k]
                        rt = misfit(*t)
                        if rt < r:
                            cur, r, improved = t, rt, True
                if not improved:
                    step *= 0.5
                    if step.max() < 1e-4:
                        break
            best = (r, cur[0], cur[1], cur[2])
        return cls(best[1], best[2], best[3], rms_km=best[0], n=len(X))

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(d['lat0'], d['lon0'], d['azimuth'], d.get('rms_km'), d.get('n'))

    def to_json(self, path):
        with open(path, 'w') as fh:
            json.dump(dict(lat0=self.lat0, lon0=self.lon0, azimuth=self.azimuth,
                           rms_km=self.rms_km, n=self.n), fh, indent=2)

    # ---------------- projection ----------------
    def _vec(self, lat, lon):
        la = np.radians(np.atleast_1d(np.asarray(lat, float)))
        lo = np.radians(np.atleast_1d(np.asarray(lon, float)))
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

    def ll2xy(self, lat, lon):
        """geographic -> (p along profile [km], q perpendicular [km]), as GMT project"""
        P = self._vec(lat, lon)
        p = np.degrees(np.arctan2(P.T @ self._x, P.T @ self._c)) * (np.pi * R_EARTH_KM / 180.0)
        q = np.degrees(np.arcsin(np.clip(P.T @ self._y, -1.0, 1.0))) * (np.pi * R_EARTH_KM / 180.0)
        return (float(p[0]), float(q[0])) if p.size == 1 else (p, q)

    def xy2ll(self, x, y=0.0):
        """(p [km], q [km]) -> geographic"""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float)) * np.ones_like(x)
        ap = np.radians(x / (np.pi * R_EARTH_KM / 180.0))
        aq = np.radians(y / (np.pi * R_EARTH_KM / 180.0))
        V = (np.cos(aq)[:, None] * (np.cos(ap)[:, None] * self._c[None, :]
                                    + np.sin(ap)[:, None] * self._x[None, :])
             + np.sin(aq)[:, None] * self._y[None, :])
        lat = np.degrees(np.arcsin(np.clip(V[:, 2], -1.0, 1.0)))
        lon = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
        return (float(lat[0]), float(lon[0])) if lat.size == 1 else (lat, lon)

    def __repr__(self):
        s = (f"ProfileGeom(-C{self.lon0:.4f}/{self.lat0:.4f} -A{self.azimuth:.3f}"
             f"  [x=0 at profile centre]")
        if self.rms_km is not None:
            s += f", fit rms={self.rms_km:.3f} km on n={self.n}"
        return s + ")"


# ============================================================================
# 3.  FDFK2D internal geometry, re-implemented verbatim
# ============================================================================
def backaz(evla, evlo, stla, stlo):
    """
    Verbatim port of subroutine backaz() in FDFK2D/src/calc_backazimuth.f90
    (T. Owens' Bullen formulation).  Returns (delta_deg, az_deg, baz_deg).
    """
    d2r = np.pi / 180.0
    coeff = (1.0 - FLATTENING) ** 2
    scola = np.pi / 2 - np.arctan(coeff * np.tan(stla * d2r))
    ecola = np.pi / 2 - np.arctan(coeff * np.tan(evla * d2r))
    slon = stlo * d2r
    elon = evlo * d2r

    a = np.sin(scola) * np.cos(slon)
    b = np.sin(scola) * np.sin(slon)
    c = np.cos(scola)
    d = np.sin(slon)
    e = -np.cos(slon)
    g = -c * e
    h = c * d
    k = -np.sin(scola)

    aa = np.sin(ecola) * np.cos(elon)
    bb = np.sin(ecola) * np.sin(elon)
    cc = np.cos(ecola)
    dd = np.sin(elon)
    ee = -np.cos(elon)
    gg = -cc * ee
    hh = cc * dd
    kk = -np.sin(ecola)

    predel = np.clip(a * aa + b * bb + c * cc, -1.0, 1.0)
    delta = np.degrees(np.arccos(predel))

    rhs1 = (aa - d) ** 2 + (bb - e) ** 2 + cc ** 2 - 2.0
    rhs2 = (aa - g) ** 2 + (bb - h) ** 2 + (cc - k) ** 2 - 2.0
    baz = np.degrees(np.arctan2(rhs1, rhs2)) % 360.0

    rhs1 = (a - dd) ** 2 + (b - ee) ** 2 + c ** 2 - 2.0
    rhs2 = (a - gg) ** 2 + (b - hh) ** 2 + (c - kk) ** 2 - 2.0
    az = np.degrees(np.arctan2(rhs1, rhs2)) % 360.0
    return delta, az, baz


def _fdfk2d_xhat(lat_org, lon_org, az_org, variant='as_coded'):
    """
    First column of FDFK2D's `inv_rotate_matrix`, i.e. the ECEF direction in which
    the code believes the model's +x axis points.

    variant='as_coded'  reproduces src/define_rotation_matrix.f90 exactly.
    variant='corrected' uses the mathematically consistent inverse (= transpose of
                        rotate_matrix) together with the conventional
                        x_hat = sin(az)*east + cos(az)*north.
    """
    la, lo, a = np.radians([lat_org, lon_org, az_org])
    st, ct = np.sin(la), np.cos(la)
    sp, cp = np.sin(lo), np.cos(lo)
    sa, ca = np.sin(a), np.cos(a)
    f1 = st * cp
    f2 = st * sp
    if variant == 'as_coded':
        # inv_rotate_matrix(:,1) as literally written in the release
        return np.array([-sa * sp + ca * f1,
                         sa * cp + ca * f1,
                         -ca * ct])
    elif variant == 'transpose':
        # what the released rotate_matrix's first ROW implies
        # (x_hat = sin(az)*east - cos(az)*north)
        return np.array([-sa * sp + ca * f1,
                         sa * cp + ca * f2,
                         -ca * ct])
    elif variant == 'corrected':
        # conventional: x_hat = sin(az)*east + cos(az)*north
        east = np.array([-sp, cp, 0.0])
        north = np.array([-st * cp, -st * sp, ct])
        return sa * east + ca * north
    raise ValueError(variant)


def fdfk2d_x_to_ll(lat_org, lon_org, az_org, x_m, variant='as_coded'):
    """
    Reproduce FDFK2D/src/coord_transform.f90::compute_backazimuth_grids, i.e. the
    mapping from model x [m] to the (stla, stlo) the code will actually use.
    """
    x_m = np.atleast_1d(np.asarray(x_m, float))
    xhat = _fdfk2d_xhat(lat_org, lon_org, az_org, variant)
    colat = np.radians(90.0 - lat_org)
    lon = np.radians(lon_org)
    o = R_EARTH_M * np.array([np.sin(colat) * np.cos(lon),
                              np.sin(colat) * np.sin(lon),
                              np.cos(colat)])
    P = o[None, :] + x_m[:, None] * xhat[None, :]
    r = np.hypot(P[:, 0], P[:, 1])
    lat = np.degrees(np.arctan2(P[:, 2], r))
    lo = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    return lat, lo


def fdfk2d_theta(lat_org, lon_org, az_org, evla, evlo, x_m, variant='as_coded'):
    """
    The angle FDFK2D will internally use, i.e. `bazs - az_org` [deg], at model
    positions x_m [m].  cos(theta) is the in-plane slowness projection factor and
    also the factor applied to the horizontal seismogram before it is written.
    """
    stla, stlo = fdfk2d_x_to_ll(lat_org, lon_org, az_org, x_m, variant)
    baz = np.array([backaz(evla, evlo, la, lo)[2] for la, lo in zip(stla, stlo)])
    th = (baz - az_org + 180.0) % 360.0 - 180.0
    return th


def solve_az_org(lat_org, lon_org, evla, evlo, x_center_m, theta_target_deg,
                 variant='as_coded', bracket=(-360.0, 360.0), n_scan=2881):
    """
    Choose az_org so that FDFK2D's own (baz - az_org) at the centre of the model
    equals the physically correct in-plane angle theta_target = baz_true - azimuth_profile.

    Why this is needed
    ------------------
    az_org enters the code twice: once inside the rotation that maps model x to
    geographic coordinates, and once as the constant that is subtracted from the
    resulting back-azimuths.  In the released source those two uses are not
    mutually consistent (see notes in the README), so simply writing the true
    profile azimuth into FD_model.dat does not, in general, give the correct
    in-plane angle.  Solving the 1-D root problem below makes cos(theta) correct
    regardless of which convention the compiled binary follows.

    Returns (az_org, theta_achieved, residual).
    """
    xs = np.atleast_1d(x_center_m)
    scan = np.linspace(bracket[0], bracket[1], n_scan)

    def f(a):
        th = fdfk2d_theta(lat_org, lon_org, a, evla, evlo, xs, variant).mean()
        return (th - theta_target_deg + 180.0) % 360.0 - 180.0

    vals = np.array([f(a) for a in scan])
    # pick the root with the smallest |az| jump across a sign change
    best = None
    for i in range(len(scan) - 1):
        if vals[i] == 0.0:
            cand = scan[i]
        elif vals[i] * vals[i + 1] < 0 and abs(vals[i] - vals[i + 1]) < 90.0:
            lo, hi = scan[i], scan[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            cand = 0.5 * (lo + hi)
        else:
            continue
        r = abs(f(cand))
        if best is None or r < best[1]:
            best = (cand, r)
    if best is None:
        i = int(np.argmin(np.abs(vals)))
        best = (scan[i], abs(vals[i]))
    az = best[0] % 360.0
    return az, fdfk2d_theta(lat_org, lon_org, az, evla, evlo, xs, variant).mean(), best[1]


# ============================================================================
# 4.  SU I/O in FDFK2D's flavour
# ============================================================================
#   240-byte trace header written as `integer(2) head(1:120)` = all zeros except
#     head(58) = number of samples          (bytes 115-116, SU 'ns')
#     head(59) = sample interval in micro-s (bytes 117-118, SU 'dt')
#   followed by `ns` native-endian real(4) samples.
#   NOTE: head is INTEGER*2, so ns < 32768 and dt < 32768 us (0.032768 s).

_SU_HEADER_BYTES = 240
_NS_WORD = 57            # 0-based index of head(58)
_DT_WORD = 58            # 0-based index of head(59)


def read_su(path, endian='<', ns=None):
    """
    Read an FDFK2D-style SU file.  Returns (data[ntr, ns], dt_seconds).

    `ns` overrides the value in the trace header, for files written by other
    tools that leave head(58) empty (some model builders do).
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < _SU_HEADER_BYTES:
        raise IOError(f"{path}: too short to be an SU file")
    h = np.frombuffer(raw[:_SU_HEADER_BYTES].tobytes(), dtype=endian + 'i2')
    ns_hdr = int(h[_NS_WORD])
    dt_us = int(h[_DT_WORD])
    if ns is None:
        ns = ns_hdr
    if ns <= 0:
        raise IOError(f"{path}: ns={ns} in trace header -- wrong endianness, or pass ns=")
    trace_bytes = _SU_HEADER_BYTES + 4 * ns
    if raw.size % trace_bytes:
        raise IOError(f"{path}: file size {raw.size} not a multiple of trace size {trace_bytes} "
                      f"(ns={ns}); the file is truncated or ns is wrong")
    ntr = raw.size // trace_bytes
    blk = raw.reshape(ntr, trace_bytes)
    data = np.frombuffer(np.ascontiguousarray(blk[:, _SU_HEADER_BYTES:]).tobytes(),
                         dtype=endian + 'f4').reshape(ntr, ns).astype(np.float64)
    dt = dt_us * 1e-6
    if dt <= 0:
        dt = np.nan
    return data, dt


def write_su(path, data, dt, endian='<'):
    """Write an FDFK2D-style SU file (used for vp_model.su / vs_model.su)."""
    data = np.atleast_2d(np.asarray(data, dtype=endian + 'f4'))
    ntr, ns = data.shape
    if ns >= 32768:
        raise ValueError(f"ns={ns} does not fit in INTEGER*2 (FDFK2D limit 32767)")
    dt_us = int(round(dt * 1e6))
    if not (0 < dt_us < 32768):
        raise ValueError(f"dt={dt}s -> {dt_us} us does not fit in INTEGER*2 "
                         f"(FDFK2D limit 0.032767 s)")
    h = np.zeros(120, dtype=endian + 'i2')
    h[_NS_WORD] = ns
    h[_DT_WORD] = dt_us
    hb = h.tobytes()
    with open(path, 'wb') as fh:
        for i in range(ntr):
            fh.write(hb)
            fh.write(data[i].tobytes())


# ============================================================================
# 5.  Signal helpers
# ============================================================================
def cos_taper(n, frac=0.05):
    """Symmetric cosine taper of total length n, frac at each end."""
    w = np.ones(n)
    m = int(max(1, round(frac * n)))
    if 2 * m > n:
        m = n // 2
    if m < 1:
        return w
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / m))
    w[:m] = ramp
    w[n - m:] = ramp[::-1]
    return w


def bandpass(x, dt, fmin, fmax, corners=2, zerophase=True):
    """Butterworth bandpass; matches CPS `hp c fmin np 2 p 2 / lp c fmax np 2 p 2`."""
    from scipy.signal import butter, sosfilt, sosfiltfilt
    nyq = 0.5 / dt
    lo = max(1e-6, fmin / nyq)
    hi = min(0.999999, fmax / nyq)
    if lo >= hi:
        raise ValueError(f"bad band {fmin}-{fmax} Hz for dt={dt}")
    sos = butter(corners, [lo, hi], btype='band', output='sos')
    return sosfiltfilt(sos, x) if zerophase else sosfilt(sos, x)


def envelope(x):
    from scipy.signal import hilbert
    return np.abs(hilbert(x))


def pick_max_envelope(x, dt, b=0.0, t_search=None, smooth_s=0.0):
    """
    Time of the maximum of the analytic envelope.  `t_search=(t1, t2)` restricts
    the search to that absolute-time window.  Returns time in the trace's own base.
    """
    e = envelope(x)
    if smooth_s > 0:
        w = max(1, int(round(smooth_s / dt)))
        e = np.convolve(e, np.ones(w) / w, mode='same')
    t = b + np.arange(len(x)) * dt
    m = np.ones(len(x), bool)
    if t_search is not None:
        m = (t >= t_search[0]) & (t <= t_search[1])
        if not m.any():
            raise ValueError(f"empty search window {t_search} for trace spanning "
                             f"{t[0]:.2f}-{t[-1]:.2f} s")
    i = int(np.argmax(np.where(m, e, -np.inf)))
    return t[i], i


def prep_window(x, dt, b, t_ref, win, dt_out, band=None, corners=2,
                taper_frac=0.10, pre_taper=0.05):
    """
    The single cut+taper implementation used everywhere: standalone
    cross-convolution, the inversion loop, and the plots.

        demean -> detrend -> pre-taper -> zero-phase bandpass
        -> resample onto dt_out while cutting `win` relative to t_ref
        -> cosine taper the window

    `win` is (t1, t2) in seconds relative to t_ref, which is the S arrival of
    that record: the observed pick from shift.dat, or the synthetic pick in
    SAC header `a`.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n > 2:
        t = np.arange(n)
        c, *_ = np.linalg.lstsq(np.vstack([t, np.ones(n)]).T, x, rcond=None)
        x = x - (c[0] * t + c[1])
    x = x * cos_taper(n, pre_taper)
    if band:
        x = bandpass(x, dt, band[0], band[1], corners=corners)
    t_in = b + np.arange(n) * dt
    t_rel = np.arange(win[0], win[1] + 0.5 * dt_out, dt_out)
    w = np.interp(t_ref + t_rel, t_in, x, left=0.0, right=0.0)
    return w * cos_taper(len(w), taper_frac), t_rel


def resample_to(x, dt_in, dt_out, b_in=0.0, b_out=None, n_out=None):
    """Band-limited-safe linear resampling onto a new uniform grid."""
    n_in = len(x)
    t_in = b_in + np.arange(n_in) * dt_in
    if b_out is None:
        b_out = b_in
    if n_out is None:
        n_out = int(np.floor((t_in[-1] - b_out) / dt_out)) + 1
    t_out = b_out + np.arange(n_out) * dt_out
    return np.interp(t_out, t_in, x, left=0.0, right=0.0), b_out


# ============================================================================
# 5b.  key = value settings files
# ============================================================================
def read_kv(path):
    """
    Parse a plain-text settings file.

        # comments after a hash are ignored
        events_root = /data/SSPMP/wabash
        events      = 20140824232145 20150323045138
        freq        = 0.05 0.5
        nproc       = 7
        repick      = false
        ref_1d      =                      # empty value -> None

    Values are coerced to int, float, bool, or a list of those when there is
    more than one token; anything else stays a string.  Paths are never
    interpreted, so spaces in a path are preserved as long as it is the only
    token on the line.
    """
    def coerce_one(tok):
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in '"\'':
            return tok[1:-1]          # quoted -> always a string
        low = tok.lower()
        if low in ('true', '.true.', 'yes'):
            return True
        if low in ('false', '.false.', 'no'):
            return False
        if low in ('none', 'null'):
            return None
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            pass
        return tok

    out = {}
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.split('#')[0].rstrip()
            if not line.strip():
                continue
            if '=' not in line:
                raise ValueError(f"{path}:{ln}: expected 'key = value', got {line!r}")
            k, _, v = line.partition('=')
            k = k.strip()
            v = v.strip()
            if not k:
                raise ValueError(f"{path}:{ln}: empty key")
            if v == '':
                out[k] = None
                continue
            toks = v.split()
            out[k] = coerce_one(toks[0]) if len(toks) == 1 else [coerce_one(t) for t in toks]
    return out


def explicit_dests(parser, argv=None):
    """
    Which options the user actually typed, read off sys.argv.

    Comparing the parsed value against the default cannot tell "not given" from
    "given a value that happens to equal the default" -- so `--win -5 15` on a
    parser whose default is already [-5, 15] would look untyped and lose to the
    settings file.  Scanning argv removes the guesswork.
    """
    argv = sys.argv[1:] if argv is None else argv
    opt2dest = {}
    for a in parser._actions:
        for o in a.option_strings:
            opt2dest[o] = a.dest
    out = set()
    for tok in argv:
        if not tok.startswith('-') or tok == '-':
            continue
        key = tok.split('=', 1)[0]
        if key in opt2dest:
            out.add(opt2dest[key])
            continue
        hits = {d for o, d in opt2dest.items() if o.startswith(key)}
        if len(hits) == 1:                      # unambiguous abbreviation
            out |= hits
    return out


def apply_kv(args, kv, keys=None, override_only_defaults=True, parser=None,
             explicit=None):
    """
    Fold a settings dict into an argparse Namespace.  An explicit command-line
    flag always wins: the file only fills values the user did not give, checked
    against the parser's defaults rather than guessed.  Values are cast with the
    argument's declared `type`, so an event id like 20140824232145 stays the
    string argparse expects instead of turning into an integer.
    """
    if kv is None:
        return args
    if explicit is None and parser is not None:
        explicit = explicit_dests(parser)
    explicit = explicit or set()
    defaults, actions = {}, {}
    if parser is not None:
        for a in parser._actions:
            defaults[a.dest] = a.default
            actions[a.dest] = a
    for k, v in kv.items():
        dest = k.replace('-', '_')
        if keys is not None and dest not in keys:
            continue
        if not hasattr(args, dest):
            continue
        if dest in explicit:
            continue              # the user typed it; the command line wins
        if override_only_defaults and dest in defaults:
            cur = getattr(args, dest)
            if cur is not None and cur != defaults[dest]:
                continue
        a = actions.get(dest)
        if a is not None and v is not None:
            cast = a.type or (type(a.default) if isinstance(a.default, (int, float))
                              else str)
            want_list = a.nargs in ('*', '+') or isinstance(a.nargs, int)
            try:
                if want_list:
                    v = [cast(x) for x in (v if isinstance(v, list) else [v])]
                elif isinstance(v, list):
                    # a scalar option given several tokens: only meaningful for a
                    # string like the GMT -C/-A pair, so rejoin rather than mangle
                    v = ' '.join(str(x) for x in v) if cast is str else v
                elif not isinstance(a.const, bool) and not isinstance(v, bool):
                    v = cast(v)
            except (TypeError, ValueError):
                pass
        setattr(args, dest, v)
    return args


def require_paths(args, names):
    """Fail with one clear message listing everything that is still unset."""
    missing = [n for n in names if getattr(args, n, None) in (None, '')]
    if missing:
        raise SystemExit(
            'missing required path(s): ' + ', '.join('--' + m.replace('_', '-')
                                                     for m in missing) +
            '\ngive them on the command line, or put them in a settings file '
            'passed with --paths, e.g.\n\n' +
            '\n'.join(f'  {m} = /your/path/here' for m in missing))


# ============================================================================
# 6.  misc
# ============================================================================
def eprint(*a):
    print(*a, file=sys.stderr)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d
