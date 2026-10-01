# Dynamic mass and end of ablation

**`wmpl.Utils.DynamicMassFit`** estimates the mass of a fireball near the end of its luminous path from
its deceleration, then follows that body with a single-body ablation simulation down to the speed where
ablation stops. The point, direction, speed and mass it ends with are the starting conditions of a dark
flight computation. It can propagate the trajectory solver's uncertainty with Monte Carlo, take the
atmosphere and winds from the same profile file a dark flight code uses, and save everything in a file
a dark flight code can read.

## Table of Contents

- [Quick start](#quick-start)
- [What it reads](#what-it-reads)
- [Command line reference](#command-line-reference)
- [How the dynamic mass is computed](#how-the-dynamic-mass-is-computed)
- [The end of ablation](#the-end-of-ablation)
- [Uncertainties: the analytic interval and the Monte Carlo](#uncertainties-the-analytic-interval-and-the-monte-carlo)
- [The atmosphere](#the-atmosphere)
- [Winds](#winds)
- [What it writes](#what-it-writes)
- [Worked examples](#worked-examples)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```
python -m wmpl.Utils.DynamicMassFit /path/to/20210228_215416_trajectory.pickle 35 27.5
```

That fits the deceleration between 35 and 27.5 km, computes the dynamic mass halfway through that window,
simulates the body down to 3 km/s, and prints and plots the result. A run with the full uncertainty
propagation, a measured atmosphere with its winds, and the file for a dark flight code:

```
python -m wmpl.Utils.DynamicMassFit traj.pickle 35 27.5 --mc --dens_sigma 300 --vkill_sigma 0.5 \
    --atm_profile ModelWindProfile_ERA5.csv --save_pickle
```

---

## What it reads

- **The trajectory pickle** (`*_trajectory.pickle`) written by the trajectory solver. The point velocities,
  heights and times of every station are taken from it.
- **Optionally, the solver's Monte Carlo pickle** (`Monte Carlo/*_mc_uncertainties.pickle`, next to the
  trajectory pickle), with `--mc`. The trajectory has to have been solved with Monte Carlo error
  estimation. The main trajectory pickle is not enough: the solver strips the individual realizations
  from it when saving, and only this file keeps them.
- **Optionally, an atmosphere profile** with `--atm_profile`, in one of the formats OpenDarkflight reads
  (see [The atmosphere](#the-atmosphere)).

---

## Command line reference

| Option | Default | What it does |
|---|---|---|
| `TRAJ_PATH` | | Trajectory pickle. |
| `HT_MAX`, `HT_MIN` | | Height window of the deceleration fit (km). A negative `HT_MAX`, e.g. `-3`, takes the last 3 km; `HT_MIN = -1` goes down to the last point. |
| `-d`, `--dens` | 3500 | Bulk density (kg/m³). |
| `-g`, `--ga` | 0.55 | Drag coefficient times shape factor, ΓA, for the dynamic mass. |
| `-e`, `--eval` | 0.5 | Where in the window the dynamic mass is evaluated (0 = bottom, 1 = top). |
| `--sigma_clip` | 3 | Outlier rejection of the velocity fit, in robust standard deviations. |
| `--maxvel`, `--maxmass` | 73 km/s, 50 kg | Discard faster points; clip runaway masses. |
| `--vkill` | 3 | Speed (km/s, relative to the air) where ablation is taken to stop. |
| `--mc` | off | Monte Carlo over the solver's realizations. |
| `--mc_samples` | all | Use at most this many realizations (random subset). |
| `--mc_no_final_sim` | off | Monte Carlo of the dynamic mass only, without the end-of-ablation simulation. |
| `--mc_seed` | random | Seed for the subsampling and every draw. |
| `--mc_cores` | all | Processes for the Monte Carlo. The results do not depend on it. |
| `--dens_sigma` | 0 | With `--mc`, each realization draws its bulk density from N(`--dens`, sigma), in kg/m³. |
| `--vkill_sigma` | 0 | With `--mc`, each realization draws its kill speed from N(`--vkill`, sigma), in km/s. |
| `--atm_profile` | MSIS | Take the air density and winds from this profile file. |
| `--atm_profile_type` | `wrf` | Its format: `wrf`, `wyoming` or `supracenter`. |
| `--no_winds` | off | Use the profile's density but not its winds. |
| `--save_pickle [PATH]` | off | Save the results for a dark flight code. |

The sigmas are one standard deviation. The intervals the tool reports are 95% intervals, about ±2
sigma: `--vkill_sigma 0.5` means about 2–4 km/s at 95%.

---

## How the dynamic mass is computed

A straight line is fitted to the point velocities against time inside the height window, with a robust
loss and outlier rejection; its slope is the deceleration *a*. At the evaluation point, with speed *v*
and air density ρ<sub>a</sub>, the drag equation gives the mass:

m = (ΓA · ρ<sub>a</sub> · v² / a)³ / ρ<sub>m</sub>²

The cubes matter: an error of a few percent in the air density, or in the speed relative to the air,
becomes three or six times larger in the mass. That is why the atmosphere and the winds below are worth
getting right.

## The end of ablation

From the evaluation point, a single-body MetSim simulation (`MetSimErosion`, with erosion and
fragmentation off) follows the body until its speed relative to the air drops below `--vkill`, or it
reaches 15 km. The simulation uses Γ = 0.7 and A = 1.21 regardless of `--ga`, on purpose: with the same ΓA
as the dynamic mass the simulated path would not depend on it at all.

The end point is placed on the solver's trajectory line, including its gravity drop. The direction given
is that of the apparent ground-fixed radiant at the end point, steepened by the gravity turn along the
path (MetSim does not put gravity in the velocity, so the turn is added afterwards). These are the inputs
a dark flight computation needs.

---

## Uncertainties: the analytic interval and the Monte Carlo

**Without `--mc`**, the report gives the dynamic mass and the end point for the deceleration ±2 sigma of
the velocity fit. That only reflects the scatter of the velocities inside the window, on the nominal
trajectory.

**With `--mc`**, every realization of the solver's Monte Carlo is processed again: its own velocity fit,
dynamic mass and end-of-ablation simulation. Two things are worth knowing about it:

- **The solver's realizations alone carry almost no deceleration uncertainty.** The solver adds noise to
  the lines of sight to get each realization's radiant and state vector, but then computes the point
  velocities from the original, un-noised observations (`Trajectory.run()`, `_mc_run`). Every
  realization therefore fits the same velocity scatter. On Winchcombe the realizations alone give a
  dynamic mass of 0.097–0.104 kg, against 0.071–0.190 kg from the deceleration fit. So each realization
  also draws its fit slope and intercept from the fit's covariance, and the Monte Carlo interval carries
  both: 0.065–0.149 kg on Winchcombe, with 24 solver realizations. The geometry-only interval is printed
  next to it.
- **The physical assumptions can vary too.** `--dens_sigma` and `--vkill_sigma` give every realization its
  own bulk density and kill speed. The kill speed is a convention more than a measurement, and on
  Winchcombe a sigma of 0.5 km/s moves the end height over 25.5–27.8 km instead of 26.5–27.3 km.

Realizations whose end of ablation cannot be simulated (already below the kill speed, or a failed
simulation) stay in the dynamic mass statistics with no end point, instead of being silently dropped.

The 95% interval edges need a few hundred realizations to be stable; with a few dozen they are close to
the sample minimum and maximum. The realizations run in parallel: 500 of them take about 12 s on 8 cores
instead of 36 s.

---

## The atmosphere

**By default** the air density comes from the MSIS model at the end of the trajectory. The end-of-ablation
simulation fits MetSim's density polynomial only over the heights it goes through, from 15 km up to its
start: a single polynomial fitted up to 180 km, as before, misses the stratosphere by 10–30%.

**With `--atm_profile`**, the density (and the winds, see below) come from a profile file instead. Two
reasons to do so:

- **The mass.** The real atmosphere of the night differs from the MSIS climatology. For the Pampeano
  fireball an ERA5 profile is 4–5% denser than MSIS-00 at 20–30 km, which is 11–16% in the dynamic mass.
- **Continuity with the dark flight.** The dark flight starts where this tool ends, and should continue in
  the same atmosphere.

The profile is read and evaluated exactly as OpenDarkflight does, without importing it
(`wmpl.Utils.AtmosphereProfile`):

- `wrf`: the CSV every model profile OpenDarkflight downloads is written in (ERA5, ECMWF, GEFS, HRRR…).
  Its density column is used as is.
- `wyoming`: a University of Wyoming radiosonde, in knots; `supracenter`: the older UWO format. The
  density comes from pressure and temperature with OpenDarkflight's gas constants.
- Between levels the logarithm of the density is interpolated linearly, and the winds component by
  component with a PCHIP interpolator, as OpenDarkflight does.

On OpenDarkflight's example files the density and the velocity relative to the air agree with
OpenDarkflight's to 1e-15 and 1e-12 m/s.

**The profile has to cover the heights needed**, from 15 km up to the top of the fit window. Outside its
range nothing is extrapolated: OpenDarkflight extrapolates above the top of a profile in its own way, and
doing it differently here would break the continuity the profile is for. A radiosonde that bursts below
the window is refused with a message saying which heights are missing. ERA5 on pressure levels reaches
about 48 km.

The pickle written with `--save_pickle` records the profile's path, format and sha256, so the dark flight
side can check it uses the same file.

---

## Winds

With a profile, its winds are used unless `--no_winds` is given. Drag, ablation and light depend on the
velocity relative to the air, not on the velocity over the ground that the cameras measure:

- **The dynamic mass** uses the speed relative to the air at the evaluation point. Since the mass goes as
  the sixth power of that speed, a 45 m/s head or tail wind changes it by about 3% on Winchcombe.
- **The end-of-ablation simulation** runs MetSim with the wind of each height. MetSimErosion has an
  optional wind for this (`Constants.wind_profile`, see its docstring): at every step it computes the
  velocity relative to the air with the wind at the fragment's current height, applies drag and ablation
  to it, and moves the fragment in 3D. Its end position and velocity relative to the ground give the end
  point and the final direction and speed.

On Winchcombe a 45 m/s wind moves the end point by only 10–20 m, but the final direction by 0.3° (head or
tail wind) to 0.6° (crosswind). The dark flight starts from that direction, and 0.3–0.6° is more than the
direction uncertainty usually given to it (0.1°).

**Why inside MetSim.** An earlier version ran MetSim in the frame of the wind at the start and corrected its
end afterwards. That is exact for a uniform wind, but the wind changes with height in speed and
direction, and a correction applied after the simulation cannot reproduce how drag and ablation respond
to it at each height. Running MetSim with the wind of each height is the physical model, and costs
little.

**Gravity** is not affected by a horizontal wind: it accelerates the body the same in the air's frame and
the ground's. What the wind changes is the drag, and with it how long the body takes to slow down, over
which the gravity turn builds up.

**Validation.** MetSim with winds was compared with an independent integration of the same equations, with
the wind evaluated at every step and a converged time step, using the ERA5 winds of the Pampeano event.
They agree to within what MetSim's 5 ms steps leave without winds: a few metres to a few tens of metres in
position, 0.002° in direction and 0.06% in mass, while ignoring the wind is off by 0.36° in direction.
Tests in `wmpl/MetSim/Tests/test_MetSimErosion.py` repeat that comparison with a wind that turns and grows
with height.

**Cost.** Without winds MetSimErosion is unchanged: bit-identical results on a single body, the default
erosion run and a Winchcombe fit with complex fragmentation, and run times within 1% of before (−0.2% and
+0.6%, inside the run-to-run scatter). With winds the wind is looked up for every fragment at every step:
the single-body simulation of this tool goes from 2.1 to 3.4 ms with a profile, and the Winchcombe
fragmentation fit takes 2.1 times as long.

---

## What it writes

Next to the trajectory pickle:

- `<name>_dyn_mass_fit.png`: the velocities in the window, the fit, and the simulated end of ablation.
- `<name>_dyn_mass_mc.npz` with `--mc`: every realization's values (dynamic mass, deceleration, density,
  kill speed, end point, direction, speed…).
- `<name>_dyn_mass_fit.pickle` with `--save_pickle`, or the path given: the results for a dark flight code.

The pickle is a single dict of built-in Python types only, so it loads with `pickle.load()` in any Python 3
environment without wmpl or numpy. It is designed to be read directly by OpenDarkflight: ejection states
use the units of the OpenDarkflight input file, and `ref_time` its time format. It holds the nominal and
±2 sigma end states and, with `--mc`, one joint end state per realization. The realizations must be used
as they are, not resampled parameter by parameter, because position, height, time and mass move together.
Speeds are relative to the ground; `v_kill` is relative to the air. The layout is described in
`wmpl/Utils/DynamicMassFitExport.py`. OpenDarkflight cannot read it yet; the reader is a separate piece of
work on that side.

---

## Worked examples

The last 7.5 km of Winchcombe, with the analytic interval only:

```
python -m wmpl.Utils.DynamicMassFit 20210228_215416_trajectory.pickle 35 27.5
```

The full Monte Carlo on 8 cores, reproducible, with uncertain density and kill speed:

```
python -m wmpl.Utils.DynamicMassFit traj.pickle 35 27.5 --mc --mc_cores 8 --mc_seed 1 \
    --dens_sigma 300 --vkill_sigma 0.5
```

With the ERA5 profile OpenDarkflight downloaded for the event, and the file for the dark flight:

```
python -m wmpl.Utils.DynamicMassFit traj.pickle 35 27.5 --mc \
    --atm_profile ModelWindProfile_ERA5_OpenDarkflight_event.csv --save_pickle
```

The same atmosphere without its winds, to see what they change:

```
python -m wmpl.Utils.DynamicMassFit traj.pickle 35 27.5 --atm_profile profile.csv --no_winds
```

---

## Known limitations

- **One body.** The end of ablation is a single body without erosion or fragmentation, with the fixed
  Γ = 0.7 and A = 1.21. The dark flight code uses its own drag model from there on, so the drag is not
  continuous at the hand-over even when the atmosphere is.
- **One column of wind.** The profile is a single vertical column, as in OpenDarkflight; horizontal changes
  of the wind are not modelled. The wind is applied in the east-north frame of the start of the
  simulation, which turns by less than 1° over 100 km of path. Vertical wind is ignored, as in
  OpenDarkflight.
- **No wind uncertainty.** The Monte Carlo does not vary the wind.
- **The gravity turn** is added analytically after the simulation, since MetSim does not include gravity
  in the velocity.

---

## Troubleshooting

**"No 'Monte Carlo/*_mc_uncertainties.pickle' file with Monte Carlo trajectory realizations was found".** The trajectory was solved without
Monte Carlo, or the `Monte Carlo` folder was not kept next to the trajectory pickle. Solve it again with
Monte Carlo error estimation.

**"The atmosphere profile … covers X to Y km, but A to B km are needed".** The profile does not reach the
top of the fit window or does not go down to 15 km. Use a deeper profile (ERA5 on pressure levels reaches
about 48 km), or lower the top of the window.

**Few or no Monte Carlo realizations simulated.** Realizations whose speed relative to the air is already
below `--vkill` at the evaluation point have no end of ablation. Evaluate the mass higher up, or lower
`--vkill`.
