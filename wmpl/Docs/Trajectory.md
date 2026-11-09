# Solving a trajectory

**The trajectory solver** turns multi-station meteor observations into a trajectory and a heliocentric
orbit, and saves the result as a pickle file. That pickle is the input to most of the rest of the
library, including [the REBOUND integrator](REBOUND.md), so this page is mainly about how to produce
one and what is inside it.

## Table of Contents

- [Running the solver](#running-the-solver)
- [The trajectory pickle](#the-trajectory-pickle)
- [Time scale](#time-scale)
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
| `traj.jdt_ref` | Reference epoch of the trajectory, as a Julian date in UTC. |
| `traj.traj_id` | Identifier of the event, e.g. `20191023_091225`. Used as the object name downstream. |
| `traj.orbit` | The computed heliocentric orbit, including the geocentric radiant and velocity. |
| `traj.state_vect_mini`, `traj.v_init`, `traj.radiant_eci_mini` | Position, speed and radiant direction at the reference point. Together these are the state vector REBOUND starts from. |
| `traj.uncertainties` | One-sigma uncertainties from the Monte Carlo runs, used to draw clones. |
| `traj.observations` | The per-station observations that went into the fit. |

## Time scale

The times of the measurements are UTC, the time of the clocks, but the Earth turns by UT1, which differs
from UTC by up to 0.9 s. The solver takes the difference into account:

- **Where UT1 - UTC comes from.** `Trajectory(..., dut1=None)` reads it at the time of the event from the
  IERS tables, through astropy. Past events use the final values of Bulletin B that astropy ships. Recent
  ones use Bulletin A, which astropy downloads when its copy is more than 30 days old. Where the tables do
  not reach, it is 0, with a message. Give `dut1` (s), or set the environment variable `WMPL_UT1_UTC`, to
  set it. The report gives the value used.
- **The frames.** The ECI coordinates of the solution turn with the Earth by UTC, as everywhere in wmpl, so
  `cartesian2Geo` and `geo2Cartesian` with the UTC times convert them to and from positions on the Earth
  exactly. The directions measured against the stars are turned into these coordinates by the rotation of
  the Earth in UT1 - UTC: the R.A./Dec., and the azimuths and altitudes that RMS and CAMS reduce from them
  with UTC. The radiant and the orbit are turned back to the frame of the stars, and so is the state vector
  that REBOUND takes. `wmpl.Utils.EarthRotation.eciToInertial(vect, traj.dut1)` does the same for any
  vector of a solution.
- **How much it matters.** On the straight synthetic meteor of the tests (3 stations, 2.5 s at 30 km/s),
  with UT1 - UTC of 0.5 s and 0.9 s, the trajectory without the correction is 3.7 m and 6.6 m off the true
  line. Its radiant is 2.7 and 4.9 arcsec off in the Earth-fixed frame, and 0.16 and 0.28 arcsec in the
  frame of the stars. With the correction, for 0.5 s, it is 0.5 mm and 0.002 arcsec off, the precision of
  the fit.
- **When to give 0.** Observations computed with UTC as UT1, as synthetic ones made with the conversions of
  wmpl are, have to be solved with `dut1=0`. The tests of wmpl do so through `conftest.py`.

The Gural and MILIG solvers do not correct for UT1 - UTC.

Reference: Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010). IERS Technical Note 36,
chapter 5.

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
