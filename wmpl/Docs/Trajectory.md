# Solving a trajectory

**The trajectory solver** turns multi-station meteor observations into a trajectory and a heliocentric
orbit, and saves the result as a pickle file. That pickle is the input to most of the rest of the
library, including [the REBOUND integrator](REBOUND.md), so this page is mainly about how to produce
one and what is inside it.

## Table of Contents

- [Running the solver](#running-the-solver)
- [The trajectory pickle](#the-trajectory-pickle)
- [Loading a pickle in Python](#loading-a-pickle-in-python)

---

## Running the solver

The solver is driven through the input format of your data, not through a single generic entry point.
Each format module under `wmpl/Formats/` has its own command line interface that reads the
observations, runs the solver and writes the results:

```
python -m wmpl.Formats.ECSV /path/to/event/*.ecsv
python -m wmpl.Formats.RMSJSON /path/to/event/*.json
python -m wmpl.Formats.CAMS /path/to/event/
python -m wmpl.Formats.Met /path/to/event.met
```

Run any of them with `--help` for the full list of options. The ones that matter most often:

| Flag | Description |
| :--- | :--- |
| `-s`, `--solver` | Which solver to use. `original` (default) is the Monte Carlo solver; `gural0`–`gural3` are the Gural multi-parameter fits with constant, linear, quadratic and exponential deceleration. |
| `-r`, `--mcruns` | Number of Monte Carlo runs used to estimate the uncertainties. |
| `-d`, `--disablemc` | Only compute the geometric solution, skipping the Monte Carlo step. Fast, but gives no uncertainties. |
| `-t`, `--maxtoffset` | Maximum timing offset allowed between stations, in seconds. |
| `-v`, `--vinitht` | Estimate the initial velocity as the average above this height, in km. |
| `--vinitdrag` | Estimate the initial velocity from a single-body drag and ablation fit to the points selected by `--vinitdragtime` and `--vinitdraght`, for meteors that already decelerate in their first part (e.g. fireballs first seen below 60-70 km), where the default straight-line fit underestimates it. Erosion of the body is absorbed by the fitted ablation coefficient (sigma + eta), but a significant wake of eroded grains shifts the measured centroids behind the body and can bias the velocity by hundreds of m/s; see `wmpl.Utils.DragInitialVelocity`. The straight line is kept if the fit fails, does not fit better or measures no deceleration (then it is only a straight line over its points). The fit estimates the time offsets of the stations again, as the solver's come from its own lag model, except those given as fixed (`--fixedtimes`), which it keeps. The report gives the dynamic pressure and the energy received over the fitted part, with a note when they cross typical fragmentation pressures (0.04-0.12 and 0.5-5 MPa, Borovicka et al. 2020) or erosion onset energies (1-2 MJ/m^2, Buccongello et al. 2024), a cue to check the light curve. Off by default; the solver then warns, in the output and the report, when a parabola over the straight line's points puts the initial velocity more than 2 sigma above it. |
| `--vinitdragtime` | Only points within this time from the first point, in seconds, are used in the `--vinitdrag` fit, 1 s by default, or no time limit if only `--vinitdraght` is given; with both, the fit ends at whichever is reached first. The fit does not model fragmentation, so the fitted part should end before the first one; but the longer the part that can be trusted to be free of fragmentation, the better the velocity (median uncertainty 97, 33 and 9 m/s fitting 0.5 s, 1 s and all points in synthetic tests). Extend it as far as the light curve shows no flare. |
| `--vinitdraght` | Only points above this height, in km, are used in the `--vinitdrag` fit, e.g. above a flare in the light curve or where the report notes typical fragmentation pressures. Given alone, it is the only limit; with `--vinitdragtime`, the fit ends at whichever is reached first. No limit by default. |
| `-l`, `--plotallspatial` | Save the full set of diagnostic plots. |

To see the solver working on a synthetic example without any data of your own:

```
python -m wmpl.Trajectory.Trajectory
```

## The trajectory pickle

A successful run writes `<event>_trajectory.pickle` (plus plots and a text report) into the output
directory. The pickled object is a `wmpl.Trajectory.Trajectory.Trajectory`. The attributes used by the
other tools are:

| Attribute | Meaning |
| :--- | :--- |
| `traj.jdt_ref` | Reference epoch of the trajectory, as a Julian date in TDB. |
| `traj.traj_id` | Identifier of the event, e.g. `20191023_091225`. Used as the object name downstream. |
| `traj.orbit` | The computed heliocentric orbit, including the geocentric radiant and velocity. |
| `traj.state_vect_mini`, `traj.v_init`, `traj.radiant_eci_mini` | Position, speed and radiant direction at the reference point. Together these are the state vector REBOUND starts from. |
| `traj.uncertainties` | One-sigma uncertainties from the Monte Carlo runs, used to draw clones. |
| `traj.observations` | The per-station observations that went into the fit. |

## Loading a pickle in Python

```python
import os
from wmpl.Utils.Pickling import loadPickle

pickle_path = "/path/to/20191023_091225_trajectory.pickle"
traj = loadPickle(*os.path.split(pickle_path))

print(traj.traj_id, traj.jdt_ref)
print("a = {:.4f} AU, e = {:.4f}".format(traj.orbit.a, traj.orbit.e))
```

Once you have the pickle, see [the REBOUND manual](REBOUND.md) for integrating the orbit backwards or
forwards in time.
