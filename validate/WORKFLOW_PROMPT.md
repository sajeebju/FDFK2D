# Wabash Valley 2-D SsPmp forward modelling — complete workflow

Paste this whole file into a new conversation as context. It contains the project
setup, the four stages with every command, what to check at each step, and the
known problems with their fixes.

---

## 0. Context

I am modelling teleseismic **SsPmp** (the post-critical Moho reflection of the
free-surface S→P conversion) along a 2-D profile across the Wabash Valley
Seismic Zone, using **FDFK2D** (Liu et al. 2024, SRL; `YoushanLiu/FDFK2D`) as
the forward engine. The goal at this stage is **forward modelling and
validation only** — generate 2-D synthetics for all events and stations, and
cross-convolve them against the observed data. No inversion yet.

### Dataset

| item | value |
|---|---|
| events | 7 (see list below) |
| stations | 47, array spans −152.1 … +152.7 km along profile (305 km) |
| usable traces | 157 event-station pairs (`shift.dat` flag = 1, present in `tpmp.xy`) |
| profile | GMT `project -C-88.0047/38.4484 -A120.5`, x = 0 at profile centre |

```
20140824232145  20140825143137  20140925175117  20150323045138
20150529070009  20150624223221  20150729023559
```

Two back-azimuth groups: three "S" events from South America (baz ≈ 156–161°,
cos θ = +0.74 … +0.78, i.e. **38–42° off profile**, past FDFK2D's own 30° limit)
and four "N" events (baz ≈ 320–325°, cos θ = −0.92 … −0.97, i.e. 15–23° off,
comfortably inside it).

### File formats

| file | columns |
|---|---|
| `<event>/t8.dat` | `<sta>.z`, S arrival (unused), GCARC °, back-azimuth °, ray parameter **s/km** |
| `<event>/shift.dat` | `<sta>.z`, **S arrival to use**, 0/1 flag (**1 = keep**) |
| `tpmp.xy` | bounce x km, y km, event, station, SsPmp−S s, σ s, lon °E, lat ° |
| `grid*/obs.tw` | `<event>_<station>`, SAC `user0`, S from `shift.dat`, pre-signal RMS |

Naming: observed `<event>_<station>.r/.z`, CPS synthetic
`syn.<event>_<station>.r/.z`, FDFK2D `fdfk2d_<event>_<station>.r/.z`.

### Physics that drives the processing choices

**SsPmp is a free-surface multiple, so it arrives AFTER the direct S**, by
8.7–14.4 s here (`tpmp.xy` column 5). Windows therefore run **−5 to +15 s**
about S, not before it.

The free-surface reflection point sits **~139 km up-dip of the station**, so the
FD grid has to be far wider than the array: −360 … +265 km. Outside ±152 km
nothing is resolved, so the wings are held at the edge column.

### Scripts (all in `/home/yaoj/C++/FDFK2D/validate`)

```
fdfk2d_common.py    shared: readers, profile geometry, SU I/O, prep_window
wv_model2d.py       parameters + background model -> vp_model.su / vs_model.su
model_extent.py     how wide does the grid need to be, from real station coords
fdfk2d_setup.py     one FDFK2D input deck per event + run_all.sh
su2sac.py           seisx/seisz.su -> SAC .r/.z, one folder per event
sac_crossconv.py    cross-convolution misfit, works on any obs/syn SAC pair
wv_invert.py        inversion driver (NOT used in this workflow)
wv.paths            settings; every key can be overridden by the matching flag
```

`--paths wv.paths` supplies defaults; a flag typed on the command line always
wins.

---

## `wv.paths`

```
# ---- paths ----
events_root      = /home/yaoj/data/SSPMP/wabash
tpmp             = /home/yaoj/paper/others/LiuYC/pmp/tpmp.xy
grids_root       = /home/yaoj/C++/grid.jinv2024
fdfk2d_bin       = /home/yaoj/bin/FDFK2D
model_dir        = /home/yaoj/C++/FDFK2D/input_2d_fd
out_root         = /home/yaoj/C++/FDFK2D/runs
work_root        = /home/yaoj/C++/FDFK2D/inv

# ---- background model: build_2d_model.py output, +/-171 km ----
background_dir   = /home/yaoj/C++/FDFK2D/input_2d_wide
background_x0_km = -171.0
background_dx    =  200.0
background_dz    =  200.0
background_npml  =   20
moho_method      = gradient

# ---- profile ----
prf = -C-88.0047/38.4484 -A120.5

# ---- events ----
events = 20140824232145 20140825143137 20140925175117 20150323045138 20150529070009 20150624223221 20150729023559
nproc  = 2

# ---- FD grid ----
x0_km  = -360.0
x1_km  =  265.0
dx     =  200.0
dz     =  200.0
Zkm    =   80.0
npml   =   20
norder =    8
f0     =    0.5

structure_halfwidth_km = 152.0
node_spacing_km        =  40.0
h0_km                  =  52.0

# ---- processing ----
freq        = 0.05 0.5
corners     = 2
win         = -5.0 15.0
taper       = 0.10
dt_common   = 0.05
pick_band   = 0.05 1.0
cc_win      = -5.0 15.0
cc_taper    = 0.10
sign_r      = -1.0
sign_z      = -1.0
```

---

## Stage 1 — build the 2-D input model

`build_2d_model.py` laterally interpolates the `grid*/start.mod` 1-D columns and
produces a **±171 km** model. `wv_model2d.py` then re-registers that onto the
wide FD grid, padding the wings with the edge column.

```bash
cd /home/yaoj/C++/FDFK2D/validate

# 1a. how wide does the grid actually need to be?
python3 model_extent.py --paths wv.paths --moho-km 52 --use-builtin-column

# 1b. the interpolated background (skip if input_2d_wide already exists)
python3 build_2d_model.py \
    --grids_root /home/yaoj/C++/grid.jinv2024 \
    --out /home/yaoj/C++/FDFK2D/input_2d_wide \
    --km_per_grid 1.0 --dx 200 --dz 200 --Zkm 80 --npml 20 \
    --lat_smooth_km 5 --plot

eog /home/yaoj/C++/FDFK2D/input_2d_wide/model_2d.png

# 1c. the FD input model on the wide grid
python3 wv_model2d.py --paths wv.paths \
    --out /home/yaoj/C++/FDFK2D/input_2d_fd --plot

eog /home/yaoj/C++/FDFK2D/input_2d_fd/model.png
```

**Check:**

```
[repair] 26 columns on the left and 25 on the right had peak Vp below 85% ...
Background(input_2d_wide: (1711, 401), x -171..171 km, Moho 36.8..60.2 km median 52.8, ...)
model 3126 x 401 (+20 PML)
  x -360 .. 265 km
  background Moho 36.8 .. 60.2 km
  model Moho      36.8 .. 60.2 km      <- must MATCH
```

* `x -171..171` — if it says anything else, `background_x0_km` is wrong.
* the two Moho lines must be identical (m = 0 reproduces the background).
* median Moho ≈ 52.8 km, matching the 1-D Bayesian result.
* in the plot: white dashed under red across ±171, flat wings, no bright edge
  stripes.

**Two known problems, both handled automatically:**

`build_2d_model.py` fades the outer ~26 columns toward zero (lateral smoothing
with zero padding instead of edge replication) — mantle Vp 4248 where it should
be 8400. Those columns feed the PML and the FK models, so `repair_edges`
replaces them with the nearest healthy column.

Picking the Moho by the 7.5 km/s contour gives 46.4 km, ~6 km too shallow,
because a `start.mod` column has a gradational lower crust that reaches 7.5
above the discontinuity. `moho_method = gradient` finds the steepest Vp increase
instead and gives 52.8 km.

---

## Stage 2 — forward modelling

```bash
cd /home/yaoj/C++/FDFK2D/validate
python3 fdfk2d_setup.py --paths wv.paths

sed -n '5,6p' /home/yaoj/C++/FDFK2D/runs/20140825143137/input/FD_model.dat
cat /home/yaoj/C++/FDFK2D/runs/20150323045138/input/Source.dat
```

**Check the per-event printout:**

* `gcarc`/`baz` from SAC agree with `t8.dat` to <2°
* `cos(theta)` positive for S events, negative for N events
* `SsPmp surface reflection point` **not** flagged as near the model edge
* `shift.dat flag filter (=1) dropped N of M`
* no `OUTSIDE the model -- dropped`

The 38–42° warning for the three S events is expected and correct.

```bash
# one event first, alone, with the machine to itself
export OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores
cd /home/yaoj/C++/FDFK2D/runs/20140825143137
mkdir -p seismograms
time (yes Y | /home/yaoj/bin/FDFK2D ./input inpar.dat ./seismograms seis 2>&1 | tee run.log)
```

`yes Y` answers FDFK2D's interactive prompt when |baz − az| > 30°, which
otherwise hangs a batch job.

```bash
# the rest, TWO at a time
export OMP_NUM_THREADS=12
for ev in 20140825143137 20140925175117 \
          20150529070009 20150624223221 20150729023559; do
  ( cd /home/yaoj/C++/FDFK2D/runs/$ev && mkdir -p seismograms &&
    yes Y | /home/yaoj/bin/FDFK2D ./input inpar.dat ./seismograms seis > run.log 2>&1 ) &
  while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 60; done
done; wait

# monitor from another terminal
watch -n 30 'tail -n 1 /home/yaoj/C++/FDFK2D/runs/*/run.log'
```

**Memory:** running all 7 concurrently OOM-killed 6 of them on a 24-core box.
FDFK2D stores the FK boundary field for every timestep, several GB per process.
Two at a time with 12 OpenMP threads each is the working configuration. Measure
the real peak with:

```bash
/usr/bin/time -v /home/yaoj/bin/FDFK2D ./input inpar.dat ./seismograms seis 2>&1 \
  | grep -E "Maximum resident|Elapsed"
```

**Runtime:** ~1–3 h per event on a 3126×401 grid. `tmax` dominates: check what
`fdfk2d_setup.py` chose and, once the record section confirms nothing is
truncated, pass `--tmax 110` to cut it.

Output is **two files per event**, not one per receiver: `seisx.su` and
`seisz.su`, each holding one trace per receiver in `Receiver.dat` order.

---

## Stage 3 — SU to SAC

```bash
cd /home/yaoj/C++/FDFK2D/validate

# one event, with the polarity cross-check
python3 su2sac.py --paths wv.paths \
    --run-dir /home/yaoj/C++/FDFK2D/runs/20140824232145 \
    --grids-root /home/yaoj/C++/grid.jinv2024 \
    --qc /home/yaoj/C++/grid.jinv2024/grid-1 --plot

eog /home/yaoj/C++/FDFK2D/runs/20140824232145/sac/20140824232145_record_section.png

#-----------------------------------------

# then all seven
for ev in 20140824232145 20140825143137 20140925175117 20150323045138 \
          20150529070009 20150624223221 20150729023559; do
  python3 su2sac.py --paths wv.paths \
      --run-dir /home/yaoj/C++/FDFK2D/runs/$ev \
      --grids-root /home/yaoj/C++/grid.jinv2024 --plot
done
```

Writes `fdfk2d_<event>_<station>.r/.z` into `<run-dir>/sac/`, copies each pair
into every grid directory whose `obs.tw` lists it, and writes `syn.tw`. SAC
headers carry `delta`, `b=0`, `kstnm`, `stla/stlo`, `evla/evlo`, `baz`, `gcarc`,
`user0 = p [s/km]`, and **`a` = the picked direct-S arrival**, which stage 4
aligns on.

**Polarity — do this once, properly.** The `--qc` block prints, per station,
the sign of the direct-S peak on the CPS synthetic beside the FDFK2D one:

```
WB12   CPS peakZ=-1.23e-05   FDFK2D peakZ=-9.87e-06   SAME sign
```

`SAME` → keep `sign_r = -1, sign_z = -1`. `OPPOSITE` → flip that one in
`wv.paths` and rerun. A sign error is absorbed into the cross-convolution
scalar α, so it never raises an error — it just quietly worsens every misfit.

The defaults come from two conventions: the FD grid has z positive **down**
while SAC is positive **up**; and FDFK2D writes `seisx = u_x·cos(baz − az_org)`,
whose sense is opposite to the SAC radial.

**Record section check:** picked S (red ticks) should follow a straight moveout
line, tilting the correct way for the incidence side, with no trace clipped at
either end. `[WARN] N picks fall within 8 s of a trace end` means `tmax` is too
small.

---

## Stage 4 — cross-convolution against the observed

```bash
cd /home/yaoj/C++/FDFK2D/validate

# FDFK2D
python3 sac_crossconv.py --paths wv.paths \
    --grids-root /home/yaoj/C++/grid.jinv2024 \
    --syn-prefix fdfk2d_ \
    --win -5 15 --cc-win -5 15 --freq 0.05 0.5 --norm peak \
    --csv fdfk2d_all.csv --plot

# CPS, identical settings = the control
python3 sac_crossconv.py --paths wv.paths \
    --grids-root /home/yaoj/C++/grid.jinv2024 \
    --syn-prefix syn. \
    --win -5 15 --cc-win -5 15 --freq 0.05 0.5 --norm peak \
    --csv cps_all.csv

python3 -c "
import csv, numpy as np
for f in ('fdfk2d_all.csv','cps_all.csv'):
    r=list(csv.DictReader(open(f)))
    E=[float(x['E']) for x in r]; L=[float(x['lag_s']) for x in r]
    print('%-16s n=%3d  mean E=%.4f  median lag=%+.2f s'%(f,len(E),np.mean(E),np.median(L)))"

eog /home/yaoj/C++/grid.jinv2024/grid97/crossconv/fdfk2d_20150323045138_WB30.png
```

### The processing chain

```
demean -> detrend -> 5% taper -> zero-phase Butterworth 0.05-0.5 Hz
-> resample to dt_common -> cut [-5, +15] s about EACH record's OWN S arrival
-> 10% cosine taper -> normalise -> convolve
-> SECOND cut to [-5, +15] s on the tau axis -> compare
```

`c1 = R_obs ∗ Z_syn`, `c2 = R_syn ∗ Z_obs`. The source wavelet appears
identically in both and cancels, so this fits the **R/Z transfer function**
without deconvolution.

**The tau axis:** for windows running `w0 … w1`, convolution index j has
`tau = 2·w0 + j·dt`, so tau = 0 is where the two direct-S arrivals coincide,
S⊗SsPmp cross-terms land at tau = the SsPmp−S delay, and the full axis spans
`2·w0 … 2·w1`. Centring on `j = N−1` (the usual "lag" convention) only puts S⊗S
at zero for a symmetric window.

**Normalisation:** R and Z of a pair are always divided by the *same* scalar
(`--norm peak` = `max(|R|,|Z|)`), because the cross-convolution is bilinear and
any common scalar cancels — but the R/Z ratio does not, and that ratio carries
the structure.

**S arrivals:** observed from `obs.tw` column 3 (i.e. `shift.dat`), never
`t8.dat`; synthetic from SAC header `a`.

### Reading the output

| column | meaning |
|---|---|
| `E` | shape misfit ‖ĉ1 − ĉ2‖², 0 = perfect, 2 = orthogonal |
| `VR` | variance reduction, 1 = perfect |
| `alpha` | optimal scalar minimising ‖c1 − α·c2‖ |
| `ccmax` | peak normalised correlation |
| `lag_s` | lag of `ccmax`; a **systematic** offset means wrong Moho depth, not a shape mismatch |

* FDFK2D ≈ CPS → the 2-D pipeline is validated.
* FDFK2D ≫ CPS worse → the deck, not the model. Check `cos θ` in `meta.json`
  and the polarity from stage 3.
* both bad → window, band, or the S picks.

A trace appears in several grid directories because `tw_script.py` bins with
±10 km overlap; expect WB11 scored more than once.

**If S⊗S at tau = 0 dominates and dilutes the SsPmp signal**, use
`--cc-win 4 15` to drop it and score only the S–SsPmp cross-terms. With
`--win -5 15` the direct S sits 5 s from the left edge, so a 10% taper eats 2 s
of it — `--taper 0.05` if that matters.

---

## Troubleshooting

| symptom | cause |
|---|---|
| a flag seems ignored | you have an old `fdfk2d_common.py`; `apply_kv` used to lose to the settings file when the typed value equalled the default |
| `x -340..2 km` in the Background line | `background_x0_km` wrong; must be the left edge of `input_2d_wide` (−171) |
| 0 station receivers / `OUTSIDE the model` | `x0_km` disagrees with the model in `model_dir` |
| `Killed` in `run.log` | OOM — reduce concurrency to 2 |
| FDFK2D hangs with no output | the `[Y/N]` prompt; use `yes Y \|` |
| all `E ≈ 1`, α negative | polarity: redo the stage-3 `--qc` check |
| systematic non-zero `lag_s` | Moho depth offset, not a shape problem |
| picks near a trace end | `tmax` too small; rebuild the decks |
| Moho detected ~46 km | contour method; use `moho_method = gradient` |
| bright stripes at the model edges | faded columns; `repair_edges` handles it, don't disable |
| garbage SU, wrong `ns` | `tstep > 0.032768 s` or `ns > 32767` — INTEGER*2 headers wrap silently |

## Open questions

1. Is `tpmp.xy` column 1 the Moho bounce point or the free-surface reflection
   point? `model_extent.py` reports which is physically possible; it changes the
   required grid width.
2. A 2-column spike to 36.8 km Moho at x = −31 km looks like a false gradient
   pick — worth checking that grid's `start.mod`.
3. Early synthetic energy ~10 s **before** the direct S appeared in one panel.
   Nothing causal should produce that; if it moves out linearly across stations
   it is the injected plane wave and the picked `a` is on the wrong pulse.
