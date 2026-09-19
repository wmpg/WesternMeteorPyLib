# REBOUND orbital integration

**`wmpl.Rebound.REBOUND`** integrates a meteoroid orbit backwards or forwards in time through the
real solar system, starting from a solved trajectory. It reports the orbital elements at the end of
the integration, finds close encounters and impacts, propagates the measurement uncertainty with
Monte Carlo clones, and can optionally include radiation forces and measure how chaotic the orbit is.

## Table of Contents

- [Quick start](#quick-start)
- [Requirements](#requirements)
- [What it reads](#what-it-reads)
- [Command line reference](#command-line-reference)
- [Choosing an integrator](#choosing-an-integrator)
- [Radiation forces](#radiation-forces)
- [Close encounters, impacts and ejections](#close-encounters-impacts-and-ejections)
- [Monte Carlo clones](#monte-carlo-clones)
- [Chaos: MEGNO](#chaos-megno)
- [What it writes](#what-it-writes)
- [Worked examples](#worked-examples)
- [Using it as a library](#using-it-as-a-library)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```
python -m wmpl.Rebound.REBOUND /path/to/20191023_091225_trajectory.pickle
```

That integrates the nominal orbit 60 days backwards with IAS15 and writes a report, a plot and a JSON
file next to the pickle. A more typical run, 100 years backwards with 100 Monte Carlo clones on 8
cores and a fixed seed:

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 36525 --outputs 2000 --mc 100 --cores 8 --seed 42
```

---

## Requirements

REBOUND is an optional dependency and is not installed by the standard wmpl setup:

```
conda install -y -c conda-forge astropy
conda install -y -c conda-forge rebound
pip install reboundx
```

Check that both imported:

```
python -c "import rebound, reboundx; print('ok')"
```

**Both are needed.** `rebound` alone is not enough — the general relativity correction, the Earth's
gravitational harmonics and the radiation forces all come from `reboundx`.

**Platform note:** this works on Linux and macOS. `reboundx` has no Windows wheel and does not compile
with MSVC (it uses C99 variable-length arrays, which MSVC does not support), so on Windows use WSL2:
install Ubuntu, set up the wmpl conda environment inside it, and run from that shell.

**Version note:** the TRACE integrator was added in REBOUND 4.4.0. On an older REBOUND,
`--integrator trace` fails immediately with a message naming the installed version, before any work
is done.

---

## What it reads

**The trajectory pickle** (the positional argument). See [the trajectory manual](Trajectory.md) for how
to produce one. The integration uses `traj.jdt_ref` as the starting epoch, `traj.state_vect_mini`,
`traj.v_init` and `traj.radiant_eci_mini` as the starting state, `traj.traj_id` as the object name,
and `traj.uncertainties` to draw the Monte Carlo clones.

**The planetary ephemeris.** By default the local DE430 kernel at `wmpl/share/de430.bsp`, which the
wmpl setup downloads. With `--horizons` the planet states come from the JPL Horizons web service
instead, which is slower and needs network access, but does not need the kernel.

The Sun, all eight planets and the Moon are included as massive bodies (the Moon separately, since a meteoroid can pass through the Earth-Moon system). The meteoroid is a
massless test particle, so it does not perturb them, which is what makes the Monte Carlo clones
independent and cheap to run in parallel.

---

## Command line reference

```
python -m wmpl.Rebound.REBOUND --help
```

| Flag | Description |
| :--- | :--- |
| `pickle_path` | **Required.** Path to the trajectory pickle. All output is written next to it. |
| `--days DAYS` | Length of the integration in days. Default: `60`. |
| `--forward [DAYS]` | Integrate forward in time instead of backward. With a value, that many days; without one, `--days` is used. |
| `--outputs N` | Number of times along the integration at which the state is saved. Default: `500`. Increase it for long runs — 500 samples over a century is one sample per 10 weeks. |
| `--mc N` | Number of Monte Carlo clones drawn from the trajectory covariance. Default: `0`. |
| `--seed SEED` | Random seed for the Monte Carlo sampling, making a run exactly reproducible. If not given, a seed is drawn and reported. |
| `--cores N` | Number of parallel processes used for the clones. Default: all but one core. |
| `--geocentric` | Report in the geocentric frame instead of the heliocentric one. Switches the semi-major axis and perihelion units from AU to km. |
| `--horizons` | Use the JPL Horizons web service for planet positions instead of the local DE430 kernel. |
| `--integrator {ias15,whfast,trace}` | Which integrator to use. Default: `ias15`. See [Choosing an integrator](#choosing-an-integrator). |
| `--dt DAYS` | Timestep in days for `whfast` and `trace`. Default: `0.5`. Must be positive. Ignored by `ias15`. |
| `--beta BETA` | Include solar radiation pressure and Poynting–Robertson drag with this beta. Mutually exclusive with `--radius`/`--density`. |
| `--radius METRES` | Object radius in metres. With `--density`, beta is computed from it. Purely gravitational if not given. |
| `--density KG_M3` | Bulk density in kg/m³, used with `--radius`. Default: `3000`. |
| `--compute_megno` | After the integration, measure the MEGNO chaos indicator of the nominal orbit. See [Chaos: MEGNO](#chaos-megno). |
| `--verbose` | Print the progress of the simulation. |

`--beta` and `--radius` are mutually exclusive: give the beta directly, or give the size and density
and let the code compute it, but not both.

---

## Choosing an integrator

| Integrator | Timestep | Resolves close encounters | Use it when |
| :--- | :--- | :--- | :--- |
| `ias15` (default) | Adaptive | Yes | Almost always. Accurate to machine precision, and the only safe choice if you are unsure. |
| `whfast` | Fixed (`--dt`) | **No** | Long integrations of an orbit that does not pass near a planet, where speed matters. |
| `trace` | Fixed (`--dt`) | Yes | Long integrations that do have close encounters. Needs REBOUND ≥ 4.4. |

Two things happen automatically with the fixed-step integrators, and it is worth knowing about both:

**The departure from the Earth is always integrated with IAS15.** The object starts on the Earth's
surface, which is as deep a close encounter as it gets. WHFast cannot resolve it at all, and TRACE
resolves the encounter but applies the Earth's J2/J4 harmonics without sub-stepping them, which leaves
the orbit about 1–2 % off in the semi-major axis. Both therefore start with IAS15 and take over at the
first output after the object has left the Earth's neighbourhood (3 Earth Hill radii), where the
harmonics are negligible. The report states when the handover happened, and the JSON records it as
`run.fixed_step_from_days`.

**A backward TRACE run is integrated in reversed time.** TRACE mishandles close encounters when the
timestep is negative — a REBOUND bug, present in 4.6.0 and 5.1.1, where a flyby integrated backward
comes out about 10 % off in the semi-major axis while the same flyby forward matches IAS15 to 1e-7.
The workaround is to integrate forward with every velocity reversed, which is exact for gravity, for
general relativity and for the Earth's harmonics. Poynting–Robertson drag is not time-reversible, so
it is handled by flipping the sign of `c` on the radiation force, which gives exactly `a(r, −v)`.
This is all internal; the times reported stay on the signed axis you asked for.

**On accuracy.** On the example trajectory `20191023_091225` integrated 100 years backward, WHFast and
TRACE at `--dt 0.5` both ended within about 1e-8 of IAS15 in the semi-major axis, at roughly 4× and 3×
the speed. **That number is not a guarantee.** On a synthetic Earth-departure test with a lunar
encounter, the same 0.5-day step gave differences of 2e-4 (WHFast) and 7e-5 (TRACE) over the same
span. An orbit with a small perihelion needs a shorter step, because the fixed step has to resolve the
fastest part of the orbit. **Before trusting a long fixed-step run, do one shorter run both ways and
compare** — the run is cheap relative to being wrong:

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 3652 --integrator ias15
python -m wmpl.Rebound.REBOUND traj.pickle --days 3652 --integrator trace --dt 0.5
```

WHFast in particular does not resolve close encounters at all: measured through flybys inside 0.1 Hill
radii, its error in the semi-major axis afterwards was 0.3 % to 200 %, where TRACE stayed within 4e-6.
The report therefore prints a warning whenever a close encounter happened while WHFast was
integrating, for the nominal solution or for any clone. **Take that warning seriously** — rerun with
`--integrator trace` or `ias15`.

REBOUNDx will warn that the `gr_full` general relativity correction is velocity-dependent when used
with WHFast. Its measured effect on the orbit was below 1e-8 in the semi-major axis over 100 years, so
the warning can be ignored here.

---

## Radiation forces

By default the integration is purely gravitational, since a trajectory solution does not constrain the
object's size or density. To include solar radiation pressure and Poynting–Robertson drag, give the
dimensionless **beta**, the ratio of radiation pressure to solar gravity:

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 36525 --beta 0.01
```

or let it be computed from a size and a density:

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 36525 --radius 1e-5 --density 3000
```

For a spherical grain, `beta = 5.7425e-4*Q_pr/(rho*s)` with the density in kg/m³ and the radius in m.
A 1 µm grain at 3000 kg/m³ gives beta = 0.19; a 1 cm meteoroid of the same density gives 1.9e-5. In
other words, radiation forces matter for small grains and are negligible for fireball-producing
bodies — but they are not negligible over long integrations, where they changed the semi-major axis by
25 % to 190 % in the cases measured here.

---

## Close encounters, impacts and ejections

Closest approaches are tracked at **every internal integrator timestep**, not at the output samples,
and then refined *within* the step. Both steps matter:

- Near the Earth and the Moon the object can cross a whole detection sphere between two output
  samples, so a scan of the sampled output would miss the encounter entirely or badly overestimate
  its distance.
- Away from the Earth, IAS15 takes steps of one to two days, so the object can pass a planet
  completely inside a single step. The distance at the step ends alone can then be far above the
  true minimum. The minimum is therefore found on the cubic Hermite interpolant of the relative
  position and velocity across the step, which reproduces straight-line motion exactly and so is
  accurate even when one step spans the whole flyby.

- **Close encounters** are reported for any body approached within **3 Hill radii**, listed in the
  order they happened. **A body can appear more than once**, if the object passed it more than once:
  each local minimum of the distance is refined and listed separately. A meteoroid in resonance with
  a planet commonly meets it many times.
- **Impacts** are detected against the physical radii of the bodies, using REBOUND's line-of-travel
  collision mode so a fast mover cannot tunnel through. Detection is armed only once the object has
  left the Earth's neighbourhood, so the trivial fact that it starts on the Earth is not reported as an
  impact. An impact ends the integration and is printed as a banner at the top of the report.
- **Ejections** are recorded if the object leaves the simulation volume (1000 AU by default), after
  which it stops being integrated.

On the plot, every passage is marked with a red star, but only the deepest one per body is labelled
with the body's name, so a long run against a repeatedly-met planet stays readable.

The first close encounter going backwards is almost always the Earth itself — that is the meteoroid
arriving. What matters is what came before it.

---

## Monte Carlo clones

`--mc N` draws N clones from the trajectory's measurement covariance and integrates each one, in
parallel, through an independent copy of the simulation. This is what turns a single number into a
number with an uncertainty:

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 36525 --mc 200 --cores 8 --seed 42
```

The report gives the final elements with 95 % confidence intervals from the clone ensemble, how many
clones impacted which body, how many escaped, and how many had a close encounter with each body. If
some clones did not survive, the confidence intervals are computed from the survivors only and the
report says so (`ci_uses_survivors_only` in the JSON).

The ensemble also gives a divergence estimate for free: the clones are trajectories started from
slightly different state vectors, so their spread over time is exactly what a Lyapunov analysis
measures. The report fits their root-mean-square separation both as exponential growth (the chaotic
case) and as linear growth (the regular case) and says which describes the data better. This is a
finite-time estimate driven by the real measurement uncertainty, so it is only meaningful while the
ensemble stays compact.

**Always pass `--seed`** if you intend to quote the result. Without it a seed is drawn and reported,
which is recoverable, but passing it is less work than reading it back out of the report.

---

## Chaos: MEGNO

`--compute_megno` runs a second, separate integration of the nominal solution over the same span, with
Newtonian gravity only and always with IAS15, and computes the MEGNO chaos indicator.

MEGNO (Mean Exponential Growth factor of Nearby Orbits) follows how fast a small deviation from the
orbit grows. Its running mean ⟨Y⟩ tends to **2** for a regular, quasi-periodic orbit, tends to **0**
for a stable periodic one (for example an orbit librating in a resonance), and grows without bound as
(λ/2)·t for a chaotic one, where λ is the Lyapunov exponent.

Only the object's own deviation is followed. REBOUND's built-in `init_megno()` perturbs every body in
the simulation, and the linearly growing deviations of the Moon and the planets then swamp the
object's exponential growth for centuries — an orbit with a 64-year Lyapunov time still read
⟨Y⟩ = 2.01 after 300 years that way, against 4.5 when only the object is followed.

The verdict is based on the final ⟨Y⟩ and its mean over the last quarter of the run:

| Verdict | Meaning |
| :--- | :--- |
| `regular` | Both within 0.1 of 2. Quasi-periodic motion over the integrated span. |
| `chaotic` | Both above 2.5. A Lyapunov time is estimated from the slope over the second half of the run. |
| `periodic` | Both below 1. The deviation stays bounded, e.g. libration in a mean-motion resonance. Regular, but not converging to 2. |
| `not_converged` | Anything else. Integrate for longer to decide. |
| `too_short` | The run covers fewer than 20 orbital periods. No verdict; the report says how long the run would need to be. |
| `unbound` | The orbit is hyperbolic, so MEGNO — which describes bounded motion — does not apply. |

**MEGNO needs tens of orbital periods to settle**, so set `--days` accordingly. For an orbit with
a = 3 AU (a period of about 5.2 years) that is at least 100 years, i.e. `--days 36525`. A run that is
too short gets no verdict rather than a wrong one.

Two caveats. Every verdict holds **over the integrated span only** — a sticky chaotic orbit can look
regular for a long time before it does not. And this is a gravity-only measurement: radiation forces
are dissipative, and MEGNO is not defined for them.

The extra integration costs roughly what a gravity-only run of the same span costs, which is small
next to a full run with the Monte Carlo clones. It is skipped, with a warning, if you call
`reboundSimulate` with `return_diagnostics=False`, since that is the only way the result comes back.

---

## What it writes

All output goes into the directory containing the input pickle, under **fixed names**. Two runs on
the same pickle overwrite each other, so copy the results elsewhere before rerunning with different
options.

| File | Contents |
| :--- | :--- |
| `rebound_simulation_results.txt` | The human-readable report: run settings, the integrator that actually ran, final elements with confidence intervals, close encounters, impacts, clone outcomes, energy drift, divergence, and the MEGNO verdict. |
| `rebound_simulation.png` | A 3×3 panel figure of the element evolution for the nominal solution and the clones. |
| `rebound_megno.png` | Only with `--compute_megno`: the evolution of ⟨Y⟩ against time in years, with the ⟨Y⟩ = 2 line marked. |
| `rebound_simulation_results.json` | The same results in machine-readable form, so downstream analysis does not have to parse the text. |

The JSON keys, at the top level:

| Key | Contents |
| :--- | :--- |
| `traj_id` | Event identifier. |
| `run` | Run settings: `integration_days`, `direction`, `reference_frame`, `ephemeris`, `n_outputs`, `beta`, `start_epoch_jd_tdb`, `final_epoch_jd_tdb`, `final_epoch_utc`, `mc_runs`, `random_seed`, `runtime_s`, `integrator`, `dt_days`, `fixed_step_from_days`. |
| `final_elements` | `a`, `q`, `e`, `incl_deg`, `peri_deg`, `node_deg`, `f_deg`, `tisserand_jupiter`, with `a_units`/`q_units` (AU heliocentric, km geocentric). |
| `encounters` | Every close encounter of the nominal solution, ordered by time, each with `body`, `min_dist_au`, `time_days`, `hill_radius_au` and `n_hill`. A body can appear more than once. |
| `closest_approaches_au`, `closest_approach_times_days` | Closest approach to every tracked body, whether or not it counted as an encounter. This is the deepest approach only, one value per body. |
| `impact`, `escaped` | The impact or ejection of the nominal solution, or `null`. |
| `clone_outcomes` | Clone statistics: counts and fractions of impacts and encounters per body, `ci_uses_survivors_only`, `n_hill_threshold`. Under `close_encounters`, `count` is how many clones met the body and `n_encounters` how many passages they made in total. `null` without `--mc`. |
| `clone_closest_approaches_au` | Per-clone closest approach to every body. |
| `clone_encounters` | Each clone's full list of encounters, in the same format as `encounters`. |
| `energy_rel_drift` | Relative energy drift of the massive subsystem, as an integrator-quality check. |
| `divergence` | The exponential and linear fits to the clone spread, with `r2_exponential` and `r2_linear`. |
| `whfast_encounter_warning` | The WHFast close-encounter warning string, or `null`. |
| `megno` | With `--compute_megno`: `times_days`, `megno`, `a_au`, `escaped` and the `verdict` dict. `null` otherwise. |
| `nominal`, `monte_carlo` | Full element series: `time_days`, `a_au`, `e`, `incl_deg`, `peri_deg`, `node_deg`, `f_deg`, `body_distances_au`. |

A quick sanity check on any run is `energy_rel_drift`. It measures the massive subsystem the object
moves through, not the object itself (which is massless), but a large drift means the integration is
not to be trusted.

---

## Worked examples

**1. The default run — where did it come from in the last two months?**

```
python -m wmpl.Rebound.REBOUND traj.pickle
```

60 days backwards, IAS15, nominal solution only. Fast, and enough to see the geocentric encounter and
the pre-encounter heliocentric orbit.

**2. A century backwards, with enough sampling to see it.**

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 36525 --outputs 5000
```

The default 500 outputs over a century is one sample every ten weeks, which is too coarse to see a
planetary encounter in the plots. The encounter *detection* is unaffected — that runs at every
internal timestep — but the plots and the JSON series are.

**3. Uncertainty propagation.**

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 36525 --outputs 5000 --mc 200 --cores 8 --seed 42
```

200 clones on 8 cores. The report now gives 95 % confidence intervals on every element, and says how
many clones impacted something, escaped, or passed near a planet.

**4. A fast long run — and checking that it was safe.**

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 365250 --outputs 5000 --integrator trace --dt 0.5
```

A thousand years with TRACE, which resolves close encounters. Use `--integrator whfast` instead only
if you know the orbit stays away from the planets; if it does not, the report will tell you so, and
the run has to be redone. Before quoting a fixed-step result, compare a shorter run against IAS15 as
shown in [Choosing an integrator](#choosing-an-integrator).

**5. A small grain, where radiation forces matter.**

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 36525 --radius 1e-5 --density 3000
```

A 10 µm grain. The report prints the beta that was computed. Compare against the same run without
`--radius` to see how much the radiation forces actually moved the orbit.

**6. Is the orbit chaotic?**

```
python -m wmpl.Rebound.REBOUND traj.pickle --days 365250 --outputs 5000 --compute_megno
```

A thousand years, which covers enough orbital periods for a verdict for most meteoroid orbits. The
report ends with a MEGNO block:

```
  MEGNO of the nominal orbit (Newtonian gravity only, IAS15):
    <Y> at the end: 0.512   mean over the last quarter: 0.498
    span: 91.2 orbital periods
    Converges to 2: NO (tends to 0). <Y> tends to 0 rather than 2: the deviation from the orbit
    stays bounded, the signature of a stable periodic orbit (e.g. libration in a mean-motion
    resonance). Regular, not chaotic.
```

If it says `too_short`, the message tells you how many years the run needs.

**7. Reading the results back in Python.**

```python
import json

with open("rebound_simulation_results.json") as f:
    res = json.load(f)

print("Integrator:", res["run"]["integrator"], "over", res["run"]["integration_days"], "days")
print("a = {:.4f} {:s}".format(res["final_elements"]["a"], res["final_elements"]["a_units"]))

for enc in res["encounters"]:
    print("{:s}: {:.5f} AU ({:.2f} Hill radii) at t = {:.1f} d".format(
        enc["body"], enc["min_dist_au"], enc["n_hill"], enc["time_days"]))

if res["megno"] is not None:
    print("MEGNO verdict:", res["megno"]["verdict"]["status"])

# The full element series of the nominal solution
t_days = res["nominal"]["time_days"]
a_au = res["nominal"]["a_au"]
```

---

## Using it as a library

The command line is a wrapper around `reboundSimulate`, which can be called directly:

```python
import os
from wmpl.Utils.Pickling import loadPickle
from wmpl.Rebound.REBOUND import reboundSimulate

traj = loadPickle(*os.path.split("/path/to/traj.pickle"))

outputs, outputs_mc, diagnostics = reboundSimulate(
    None, None, traj=traj,
    direction="backward", sim_days=36525, n_outputs=5000,
    mc_runs=100, random_seed=42,
    integrator="trace", dt_days=0.5,
    compute_megno=True, return_diagnostics=True
    )

# Orbital elements at the end of the nominal integration
final = outputs[traj.traj_id][-1][2]
print("a = {:.4f} AU, e = {:.4f}".format(final.a, final.e))

# The MEGNO verdict lives in the nominal solution's diagnostics
print(diagnostics[traj.traj_id]["megno"]["verdict"]["status"])
```

`compute_megno=True` needs `return_diagnostics=True`, since the diagnostics are the only place the
result is returned.

Other functions worth knowing about, all in `wmpl.Rebound.REBOUND`:

| Function | Purpose |
| :--- | :--- |
| `radiationPressureBeta(radius_m, density_kgm3)` | Beta for a spherical grain. |
| `encountersFromMinDistances(min_dist_au, min_time_days, n_hill=3.0)` | The close-encounter list from the closest-approach dictionaries, one entry per body. The per-particle `encounters` diagnostic lists every passage instead. |
| `cloneEncounterSummary(clone_diag)` | Per body, how many clones met it, how many passages they made, and the closest approach over all of them. |
| `hermiteClosestApproach(t0, r0, v0, t1, r1, v1)` | Closest approach of a relative trajectory inside one integrator step. |
| `whfastEncounterWarning(diagnostics, n_hill=3.0)` | The WHFast warning string, or `None`. |
| `computeMegno(task, seed=1)` / `classifyMegno(times_days, megno, a_au)` | MEGNO series and verdict. |
| `estimateLyapunovFromMC(sim_outputs, sim_outputs_mc)` | The divergence fit over the clone ensemble. |
| `checkIntegratorAvailable(integrator)` | Raises if the installed REBOUND does not have it. |
| `findEarthDepartureIndex(sim_outputs, n_hill=3.0)` | First output at which the object left the Earth's neighbourhood. |

---

## Troubleshooting

**`reboundx` will not install on Windows.** It has no Windows wheel and does not compile with MSVC.
Use WSL2. Installing only `rebound` is not enough.

**`The installed REBOUND (4.3.0) does not provide the TRACE integrator.`** TRACE was added in REBOUND
4.4.0. Upgrade, or use `--integrator ias15`.

**`--dt must be positive.`** A zero or negative step would silently collapse to a single step per
output interval, so it is refused rather than accepted.

**`WARNING: N close encounter(s) ... happened while WHFast was integrating.`** WHFast does not resolve
close encounters, so the orbit after them may be badly wrong. Rerun with `--integrator trace` or
`ias15`.

**`REBOUNDx: Passing a velocity-dependent force to WHFAST.`** Expected, and harmless here: the
measured effect of the general relativity correction on the orbit was below 1e-8 in the semi-major
axis over 100 years.

**The run hangs at the end.** It is not hung — the script ends with a blocking `plt.show()`. Close the
figure windows. The files are already written by that point.

**Results changed between two runs with the same arguments.** Pass `--seed`. Without it the Monte
Carlo seed is drawn at random (and reported, so an old run can be reproduced from its report).

**Two runs overwrote each other.** Output names are fixed and go next to the input pickle. Copy the
results out, or keep the pickles in separate directories.
