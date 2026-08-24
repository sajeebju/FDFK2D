# Wabash SsPmp 2-D forward modelling: FDFK2D → SAC → cross-convolution

## Paths

No path is baked into any script. Give them as flags, or put the ones you retype
into a settings file and pass `--paths`:

```bash
python3 wv_invert.py --write-template wv.paths   # commented, all values blank
```

```
# wv.paths -- "key = value", # comments, empty value = unset
events_root = /home/yaoj/data/SSPMP/wabash
tpmp        = /home/yaoj/paper/others/LiuYC/pmp/tpmp.xy
model_dir   = /home/yaoj/C++/FDFK2D/input_2d
out_root    = /home/yaoj/C++/FDFK2D/runs
work_root   = /home/yaoj/C++/FDFK2D/inv
grids_root  = /home/yaoj/C++/grid.jinv2024
fdfk2d_bin  = /home/yaoj/bin/FDFK2D
prf         = -C-88.0047/38.4484 -A120.5
events      = 20140824232145 20140825143137 20140925175117 20150323045138 20150529070009 20150624223221 20150729023559
x0_km       = -318
nproc       = 7
```

Every script takes `--paths`, and **a flag on the command line always wins** —
so one file plus `--f0 0.7` or `--gn-iters 3` covers the usual case. Any tuning
key can live in the file too; values are cast through the flag's declared type,
so `20140824232145` stays a string rather than becoming an integer. A missing
path fails immediately with a message naming exactly what to add.

Three separate stages, deliberately decoupled so the cross-convolution code has
no idea FDFK2D exists and works on any SAC data:

| stage | script | in → out |
|---|---|---|
| 1 | `fdfk2d_setup.py` | metadata → one FDFK2D input deck per event + `run_all.sh` |
| 2 | `su2sac.py` | `seisx.su`/`seisz.su` → `fdfk2d_<ev>_<sta>.r/.z` SAC + `syn.tw` |
| 3 | `sac_crossconv.py` | any obs/syn SAC pair → cross-convolution misfit + plots |

`fdfk2d_common.py` holds the shared readers, geometry and SU I/O.

---

## 0. What was probably wrong before

Ranked by how badly each one bites, from your posted commands:

1. **`FD_model.dat` line 8 — `latitude / longitude / azimuth of (X0,Z0)`.**
   FDFK2D does not take the incidence angle from you. It maps every model
   `x` back to a lat/lon using that line, calls `backaz()` against
   `src_lat/src_lon`, subtracts `az_org`, and uses `cos(baz − az_org)` both as
   the in-plane slowness projection **and** as the sign that decides whether the
   wave enters from the left or the right boundary. For this profile the correct
   value is `cos θ = +0.78`. If that line is left at a template default:

   | line 8 contents | cos θ at model centre |
   |---|---|
   | `45.0  0.0  0.0` (test1 template) | **−0.44** |
   | `0 0 0` | **−0.32** |
   | right origin, azimuth left at 0 | **−0.91** |
   | correct | **+0.78** |

   A negative value means the wave is injected from the wrong side of the model.
   Nothing in the output announces this — you just get a plausible-looking
   seismogram with the wrong moveout sign. `build_2d_model.py` had no reason to
   know the geographic reference, so this is the first thing to check.

2. **`tmax`.** A 342 km model with in-plane slowness ≈0.09 s/km has ~31 s of
   plane-wave moveout across it, on top of the traveltime to the first receiver.
   The template `tmax = 40 s` truncates the far half of the array. The setup
   script computes `tmax = 1.6 × moveout + 60 s` (139 s for the full 540 km
   profile) and `su2sac.py` warns when a pick lands near a trace end.

3. **Model too narrow.** The SsPmp free-surface reflection point sits
   `2h·tan(i_P) ≈ 139 km` from the station toward the source (h = 50 km,
   p = 0.1156 s/km). With `--model_x0_km -171 --model_x1_km 171`, stations in the
   +x half have that reflection point outside the model, so their SsPmp is formed
   in the FK/PML boundary rather than in your 2-D structure. The setup script
   checks this per station and tells you which ones fail.

4. **`--station_x_km 60` for WB12.** `tpmp.xy` column 1 is the *bounce point*,
   not the station: for WB12 in this event it is +4.4 km, and the station is one
   full offset back up-profile from it, i.e. at negative x. Stage 1 now projects
   the station from its SAC `stla/stlo` onto the fitted profile, so the number is
   never typed by hand.

5. **Ray parameter.** `12.79 s/deg` vs `12.86 s/deg` from t8.dat column 5
   (`0.115621 s/km × 111.195`). Small, but free to get right.

6. **`dt = 0.01 s` with `order 16`, `dx = 200 m`, `Vp_max ≈ 8.1 km/s`** gives
   Courant ≈ 0.41, which is marginal for a 16th-order operator. Default is now
   order 8 with `dt = 0.3·dx/Vp_max`.

7. **`f0 = 1.0 Hz`** needs `Vs_min ≥ 2.5 km/s` to keep 5 points per shortest
   wavelength at `dx = 200 m`. Default is now `0.8 Hz`, and the script prints the
   dispersion-safe ceiling for your actual model.

---

## 1. Build the input decks

```bash
cd /home/yaoj/C++/FDFK2D/validate     # or wherever you keep these scripts

python3 fdfk2d_setup.py \
    --events-root /home/yaoj/data/SSPMP/wabash \
    --tpmp        /home/yaoj/paper/others/LiuYC/pmp/tpmp.xy \
    --model-dir   /home/yaoj/C++/FDFK2D/input_2d \
    --out-root    /home/yaoj/C++/FDFK2D/runs \
    --x0-km -318 --dx 200 --dz 200 --Zkm 80 --npml 20 \
    --norder 8 --f0 0.8 --dense-dx-km 2
```

`--x0-km` is the **profile coordinate of model X0**, in the same km units as
`tpmp.xy` column 1. Everything else is derived:

* profile origin and azimuth, least-squares fitted from `tpmp.xy`'s
  (x, y) ↔ (lon, lat) pairs — for your table that is
  **x = 0 at 38.486 N, 87.986 W, azimuth 120.16°, 1.0 km rms over 540 km**
* station positions from SAC `stla/stlo`, projected onto that profile
* event location from SAC `evla/evlo`, cross-checked against `t8.dat`
  `gcarc`/`baz` (it warns if they disagree by more than 2°)
* ray parameter = median of `t8.dat` column 5 over the stations that appear in
  `tpmp.xy` for that event, converted s/km → s/deg
* `FK_model_left/right.dat` extracted from the outermost interior columns of
  `vp_model.su`/`vs_model.su`, with a shared bottom half-space starting at the
  shallowest depth where the two edges already agree (FDFK2D refuses to run
  otherwise)
* `dt`, `tmax`, `tstep`, and a check that the output sample count and `tstep`
  still fit in FDFK2D's `INTEGER*2` SU trace header (ns < 32768,
  tstep < 0.032768 s — silent corruption if you exceed either)

Receivers are written **stations first, in `meta.json` order**, then a dense
line for plotting. Stage 2 relies on that ordering.

Run it:

```bash
bash /home/yaoj/C++/FDFK2D/runs/run_all.sh
```

The runner pipes `yes Y` into the binary. FDFK2D stops and asks
`Do you want to continue? [Y/N]` whenever `|baz − az_org| > 30°`, which would
otherwise hang a batch job.

**Read that warning rather than only suppressing it.** Your seven events split
into two groups relative to the 120.16° profile:

* South American (`20140824232145`, `20140825143137`, `20150323045138`),
  baz ≈ 156–160°, deviation ≈ **35–40°** — past FDFK2D's own validity limit
* the northwestern group (`20140925175117`, `20150529070009`, `20150624223221`,
  `20150729023559`), baz ≈ 320–325°, arriving from the −x side, deviation from
  the anti-profile direction ≈ **20–25°** — inside the limit

The deviation is real physics, not a code problem: at 38° off-plane the 2-D
model cannot represent the out-of-plane part of the path. The in-plane slowness
projection is still handled correctly, so the *timing* stays good; it is the
amplitudes and any 3-D structure off the line that degrade. If a grid's fit is
carried mainly by the South American events, that is worth a sentence in the
paper.

---

## 2. SU → SAC

```bash
for ev in 20140824232145 20140825143137 20140925175117 20150323045138 \
          20150529070009 20150624223221 20150729023559; do
  python3 su2sac.py \
      --run-dir    /home/yaoj/C++/FDFK2D/runs/$ev \
      --grids-root /home/yaoj/C++/grid.jinv2024 \
      --plot
done
```

Writes `fdfk2d_<event>_<station>.r/.z` into `<run-dir>/sac`, copies each pair
into every grid directory whose `obs.tw` lists it (same matching rule as
`tw_script.py`), and writes `syn.tw` alongside.

Headers set: `delta`, `b=0`, `kstnm`, `kevnm`, `stla/stlo/stel`,
`evla/evlo/evdp`, `baz`, `gcarc`, `user0 = p [s/km]`, and **`a` = the picked
direct-S arrival**, which is what stage 3 aligns on. The pick is the maximum of
the analytic envelope of the radial, band-limited for picking only — SsPmp
arrives 9–14 s *before* S in this dataset and is the weaker phase, so the
envelope maximum is the direct S.

### Polarity

Two conventions have to be reconciled and both are exposed as flags:

* `--sign-z -1` — the FD grid has z positive **down** (`Z0 = 0` at the free
  surface, `Zn` at depth); SAC/CPS vertical is positive **up**.
* `--sign-r -1` — FDFK2D writes `seisx = u_x · cos(baz − az_org)`, i.e. the
  along-profile horizontal projected onto the propagation direction; the SAC
  radial points *from the event to the station*, which is the opposite sense.

These match the `--flip_z --flip_r` you were already using. Confirm them once
rather than trusting me:

```bash
python3 su2sac.py --run-dir /home/yaoj/C++/FDFK2D/runs/20150323045138 \
    --qc /home/yaoj/C++/grid.jinv2024/grid-1
```

which prints the sign of the direct-S peak on the CPS `syn.*` and the FDFK2D
trace side by side. If they come out `OPPOSITE`, flip the corresponding
`--sign-*`. Do not let stage 3 "discover" the polarity — a sign error there is
absorbed into `alpha` and quietly degrades the misfit instead of failing.

---

## 3. Cross-convolution

Works on any obs/syn SAC pair; the only thing that changes between CPS and
FDFK2D is `--syn-prefix`.

```bash
# FDFK2D
python3 sac_crossconv.py \
    --grids-root /home/yaoj/C++/grid.jinv2024 \
    --syn-prefix fdfk2d_ \
    --freq 0.05 0.5 --win -25 5 --norm peak \
    --csv fdfk2d_crossconv.csv --plot

# CPS, same settings, for a like-for-like baseline
python3 sac_crossconv.py \
    --grids-root /home/yaoj/C++/grid.jinv2024 \
    --syn-prefix syn. \
    --freq 0.05 0.5 --win -25 5 --norm peak \
    --csv cps_crossconv.csv
```

Per trace pair:

```
demean → detrend → 5% taper → zero-phase Butterworth bandpass
→ resample to a common dt → cut a window referenced to each record's OWN
S arrival → 10% cosine taper → normalise → convolve
```

* **observed S** comes from `obs.tw` column 3, i.e. your hand-refined
  `shift.dat` pick. `t8.dat` column 2 is never used. The script auto-detects
  whether the number lives on the trace's own axis or is measured from `o`, and
  raises rather than mis-aligning by ~1000 s.
* **synthetic S** comes from SAC header `a`, or `--syn-tw`.

### Scaling / normalisation

`c1 = R_obs ∗ Z_syn` and `c2 = R_syn ∗ Z_obs` are both bilinear, so any scalar
common to a station pair cancels — but the **R/Z ratio does not**, and that
ratio is where the structural information lives. So R and Z of a pair are always
divided by the *same* scalar:

* `--norm peak` (default) — divide by `max(|R|, |Z|)` over the window, the CPS
  convention
* `--norm peakz` — divide by `max(|Z|)`
* `--norm energy` — divide by `sqrt(Σ R² + Z²)`
* `--norm none`

Never normalise R and Z independently; that destroys exactly the quantity being
inverted.

### What the numbers mean

| column | meaning |
|---|---|
| `alpha` | optimal scalar minimising ‖c1 − α·c2‖ |
| `VR` | variance reduction, 1 = perfect |
| `E` | shape-only misfit ‖ĉ1 − ĉ2‖², 0 = perfect, 2 = orthogonal |
| `ccmax` | peak normalised correlation between c1 and c2 |
| `lag_s` | lag at which `ccmax` occurs |

`lag_s` is the diagnostic worth watching: a **systematic** non-zero lag across a
grid means the synthetic Moho depth is off (or the two S picks are on different
references), not that the waveform shapes disagree. Shape mismatch shows up in
`E` with `lag_s ≈ 0`. Weights come from the pre-signal RMS in `obs.tw` column 4,
rescaled by the same normalisation applied to the data.

---

## Two things in FDFK2D worth knowing about

Both found by reading the released source, both handled automatically.

**`src/define_rotation_matrix.f90`.** `inv_rotate_matrix` is not the transpose of
`rotate_matrix` — five of nine entries differ — and `rotate_matrix`'s first row
works out to `x̂ = sin(az)·ê − cos(az)·n̂`, so the code's `az_org` is not simply
the true profile azimuth. `compute_backazimuth_grids` uses that inverse to turn
model `x` into lat/lon, so the geographic positions it derives are wrong: for
your profile the far end lands in Michigan instead of Kentucky.

The **practical** effect is small, because back-azimuth to a source 60° away
varies slowly: the induced error in `baz` is under 1°, i.e. under 1% in
`cos θ`. So this is worth reporting upstream but is not what is breaking your
runs. `fdfk2d_setup.py` removes even that residual by numerically solving for the
`az_org` that makes the code's own `baz − az_org` equal the true in-plane angle
at the model centre, reproducing the code's arithmetic exactly
(`--geom-variant as_coded`, the default). If you patch the source, rerun with
`--geom-variant corrected`; you will get `az_org ≈ 120.16` and essentially the
same result.

**`INTEGER*2` SU headers.** `head(58) = ns` and `head(59) = tstep·10⁶`, so
`ns < 32768` and `tstep < 0.032768 s`. Exceeding either wraps silently. Stage 1
checks both.

---

## Where the numbers come from

| file | columns |
|---|---|
| `t8.dat` | `<sta>.z`, S arrival (superseded), GCARC °, back-azimuth °, ray parameter **s/km** |
| `shift.dat` | `<sta>.z`, **S arrival to use**, 0/1 flag |
| `tpmp.xy` | bounce-point x km, y km, event, station, SsPmp−S s, σ s, bounce lon °E, bounce lat ° |
| `obs.tw` | `<event>_<station>`, SAC `user0`, S from `shift.dat`, pre-signal RMS |

The 0/1 flag in `shift.dat` column 3 is carried through untouched — nothing in
this pipeline depends on it, because I could not determine its meaning from the
files you sent. It does not correlate with whether the pick was refined
(`WB03` has flag 1 with a value identical to `t8.dat`, `WB14` has flag 0 with the
same property). If it is a quality or polarity flag, say so and I will wire it
into the trace selection.

---

# 2-D inversion

Two more files: `wv_model2d.py` (parameters → model) and `wv_invert.py` (driver).

```bash
python3 wv_invert.py --config wv.json --write-config   # then edit paths
python3 wv_invert.py --config wv.json --method gradient --dry-run
python3 wv_invert.py --config wv.json --method scan --scan -6 6 1.5
python3 wv_invert.py --config wv.json --method gradient --iters 15
```

**Read the cost line before starting.** One forward evaluation = 7 FDFK2D runs
≈ 25 min wall clock on 7 cores. That budget is what forces every design choice
below, and it means this is a local refinement of your Bayesian result, not an
FWI and not a sampler. Uncertainties come from the 1-D inversion you already
have; do not quote a covariance from this.

**Model vector** = Moho depth at ~8 nodes (40 km spacing), PCHIP-interpolated,
optionally a uniform crustal `dVp` and `dVp/Vs`. Crust is stretched to move the
Moho, mantle is translated, so layer proportions are preserved. Outside
±152 km the wings are 1-D — the FD grid must reach −340…+245 km to contain the
reflection points, but only the array width is resolved.

**Data** = every event-station pair with `shift.dat` flag 1, intersected with
`tpmp.xy`: 26 for `20150323045138`, ~175 over seven events. Grid directories are
not used — the grid binning exists so jinv2024 can assign a datum to a 1-D
column, but a 2-D simulation propagates through the whole model at once, so each
trace is counted exactly once.

**Gradient in 2 evaluations, not N+1**: node *j* only affects traces whose
bounce point is near *x_j*, so even nodes are perturbed together in one run and
odd nodes in another, and each partial comes from that node's own traces. An
iteration is 1 base + 2 gradient + ~2 backtracking = 5 evaluations ≈ 2 h.

**Why the automatic S pick doesn't have to be perfect**: shifting both synthetic
components by the same τ shifts `c1` and `c2` equally, so the cross-convolution
misfit is blind to the absolute alignment. What it sees is the SsPmp−S lag and
the R/Z partitioning — the two things crustal thickness controls.

Watch the `traces per node` line. Nodes with fewer than 3 traces are
unconstrained and will drift unless you raise `lambda_prior` or widen
`node_spacing_km`.
