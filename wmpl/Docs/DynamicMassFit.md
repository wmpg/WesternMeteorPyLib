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
| `--atm` | `00` | MSIS model when there is no profile: `00` (NRLMSISE-00), `2.0` or `2.1` (NRLMSIS 2.x). |
| `--atmtime` | trajectory | Evaluate MSIS at this UTC time instead, as `YYYYMMDD-HHMMSS`. |
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

**The speeds are taken relative to the ground.** The trajectory solver works in the inertial ECI frame
with moving stations, so its point velocities include the part of the Earth's rotation along the motion.
For Winchcombe, moving east at 52° N, that is 217 m/s: 6.30 km/s in ECI at 30 km is 6.09 km/s relative to
the ground. Drag acts on the speed relative to the air, which turns with the ground, and with m ∝ v⁶ the
ECI speed made the dynamic mass 24% too large there (0.112 instead of 0.090 kg). For a westward fireball
the error has the opposite sign. Every point speed is converted before the fit (`groundSpeed()`): relative
to the ground the body moves along the fixed direction of the apparent ground-fixed radiant, so the speed
along the fitted line is that speed projected on the line plus the rotation velocity projected on it. The
deceleration is barely affected, since the rotation adds an almost constant offset.

## The end of ablation

From the evaluation point, a single-body MetSim simulation (`MetSimErosion`, with erosion and
fragmentation off) follows the body until its speed relative to the air drops below `--vkill`, or it
reaches 15 km. The simulation uses Γ = 0.7 and A = 1.21 regardless of `--ga`, on purpose: with the same ΓA
as the dynamic mass the simulated path would not depend on it at all.

The simulation starts at the evaluation point of the solver's trajectory, with its gravity drop, and moves
relative to the ground in the direction of the apparent ground-fixed radiant at that point, steepened by
the gravity turn accumulated since the radiant was tangent to the path (`evalPointState()`). MetSim then
follows the body in 3D in the east-north-up frame of that point, which is fixed to the ground, with gravity
towards the Earth's centre and the Coriolis acceleration in its velocity (`Constants.gravity_3d` and
`Constants.latitude`), and with the wind of each height if there is a profile. The final position and
velocity are carried to the ECI frame and to the local horizon of the final point. The final point, the
direction of the apparent ground-fixed radiant there, and the speed relative to the ground are the inputs a
dark flight computation needs.

**Why gravity goes in the simulation.** MetSim's default path is a straight line with a negligible drop.
Earlier versions started it with the elevation of the radiant at the reference point of the trajectory,
placed the end on the solver's straight line with its drag-free gravity drop extrapolated, and added the
gravity turn to the final direction afterwards. The direction came out right to 0.001°, but the simulated
body flew above the real, curving path, through thinner air, and went too far. Compared with an
integration of the same body with gravity at every step, that put the end 2 m off for Winchcombe, but
278 m along the track for a 20 kg body entering at 8 km/s and 15°, whose last 4.7 s are simulated. The
elevation of the radiant also changes along the trajectory as the horizon turns, by 0.5° between the
beginning and the end of Winchcombe. With gravity in the simulation and the start taken at the evaluation
point, the turn, the drop and the drag come out of the same integration.

**Why Coriolis too.** The simulation runs in a frame fixed to the ground, because the air moves with it.
In that frame the Earth's rotation turns the path with the Coriolis acceleration, by up to 0.008°/s: 0.04°
over the 5 s of the test case below, where gravity turns the path by 0.08–0.13°.

**Validation.** The end state was compared with an integration of the same body in the inertial frame,
where the drag acts on v − ω × r and the rotation of the Earth needs no Coriolis term to be written, so it
shares nothing with MetSim's formulation or with DynamicMassFit's frame conversions. For a 1 kg and a
1000 kg body entering at 20 km/s, the final direction agrees to 0.0003° and 0.00004°, and the position
to 3–20 m, the distance travelled in one of MetSim's 5 ms steps; without the Coriolis term the direction
would be 0.04° off, and with its sign flipped 0.08°. `wmpl/Utils/Tests/test_DynamicMassFit.py` repeats
that comparison, and also checks the final point and direction with plain geodetic conversions, without
the sidereal time and Earth rotation the tool uses; a missing rotation would move the end by hundreds of
metres.

**What changed for Winchcombe** with the speeds relative to the ground and the 3D simulation: the dynamic
mass went from 0.112 to 0.090 kg and the final mass from 0.103 to 0.084 kg; the end of ablation is 200 m
higher and 50 m away horizontally, and its direction within 0.007°.

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
  dynamic mass of 0.078–0.084 kg, against 0.058–0.154 kg from the deceleration fit. So each realization
  also draws its fit slope and intercept from the fit's covariance, and the Monte Carlo interval carries
  both: 0.057–0.156 kg on Winchcombe, with 24 solver realizations and the nominal solution. The
  geometry-only interval is printed next to it.
- **The physical assumptions can vary too.** `--dens_sigma` and `--vkill_sigma` give every realization its
  own bulk density and kill speed. The kill speed is a convention more than a measurement, and on
  Winchcombe a sigma of 0.5 km/s moves the end height over 26.3–28.3 km instead of 26.9–27.8 km.

Realizations whose end of ablation cannot be simulated (already below the kill speed, or a failed
simulation) stay in the dynamic mass statistics with no end point, instead of being silently dropped.

The 95% interval edges need a few hundred realizations to be stable; with a few dozen they are close to
the sample minimum and maximum. The realizations run in parallel: 500 of them take about 12 s on 8 cores
instead of 36 s.

---

## The atmosphere

**By default** the air density comes from the MSIS model at the end of the trajectory: NRLMSISE-00, or
NRLMSIS 2.0 or 2.1 with `--atm`, evaluated at the time of the trajectory unless `--atmtime` gives another one.
The Monte Carlo realizations use the same model. The end-of-ablation simulation fits MetSim's density
polynomial only over the heights it goes through, from 15 km up to its start: a single polynomial fitted up
to 180 km, as before, misses the stratosphere by 10–30%.

**With `--atm_profile`**, the density (and the winds, see below) come from a profile file instead. Two
reasons to do so:

- **The mass.** The real atmosphere of the night differs from the MSIS climatology. For Winchcombe the
  ERA5 profile of the night is 9–10% denser than MSIS-00 at 20–30 km, which makes the dynamic mass 34%
  larger: 0.121 kg instead of 0.090 kg.
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
which the gravity turn builds up. Both are in the same integration, so this comes out by itself.

**Validation.** MetSim with winds was compared with an independent integration of the same equations, with
the wind evaluated at every step and a converged time step. Two winds were used: the ERA5 winds of the
Winchcombe night, weak at these heights (1–10 m/s), and a wind turning from 250° to 300° and growing from
10 to 70 m/s between 20 and 32 km. They agree to within what MetSim's 5 ms steps leave without winds: 1–12 m
in position, 0.001° in direction and 0.06% in mass. Ignoring the wind is off by 0.05° in direction with the
ERA5 winds and by 0.36° with the stronger ones.
Tests in `wmpl/MetSim/Tests/test_MetSimErosion.py` repeat that comparison with a wind that turns and grows
with height, with and without gravity and the Coriolis acceleration.

**Cost.** MetSimErosion without winds or `gravity_3d`, as every fit uses it, is unchanged: bit-identical
results on a single body, the default erosion run and a Winchcombe fit with complex fragmentation, and run
times within 1% of before, inside the run-to-run scatter. In 3D, the single-body simulation of this tool
takes 3.05 ms instead of 2.91 ms with gravity, and 4.8 ms with gravity and the winds of a profile, out of
about 50 ms for the whole end of ablation (most of it fitting the atmosphere). A Monte Carlo of Winchcombe
takes as long as before. A complex fragmentation fit with winds takes 2.1 times as long, because the wind
is looked up for every fragment at every step.

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
- **The gravity turn before the evaluation point** is estimated as g·cos(e)·t/v<sub>avg</sub>, with the
  average speed of the trajectory. For a body that slowed down a lot, ∫dt/v is larger; on Winchcombe the
  two differ by 0.005–0.016°, depending on where the radiant is tangent to the path.
- **The direction at the evaluation point** is the fitted radiant derotated there, which assumes a path
  straight in the inertial frame before that point, as the solver's line does; the Coriolis turn is only
  simulated after it. The difference is at most Ω·t, 0.03° for the 7 s before the evaluation point of Winchcombe.
- **A sphere for the heights.** MetSim measures heights over a sphere of the Earth's mean radius centred
  below the evaluation point; over 100 km of path that differs from the ellipsoid by about a metre.
- **Moving stations.** The conversion of the speeds and of the radiant to the ground assumes the solver used
  moving stations, as it does unless its line-of-sight solution fails.
- **No wind uncertainty.** The Monte Carlo does not vary the wind.

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
