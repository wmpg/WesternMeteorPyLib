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
| `-l`, `--plotallspatial` | Save the full set of diagnostic plots. |

To see the solver working on a synthetic example without any data of your own:

```
python -m wmpl.Trajectory.Trajectory
```

### Additional fragments in ECSV files

ECSV files can describe more fragments than the main one, as in Appendix 4 of the GFE standard: one row
per frame, with the columns of each additional fragment carrying its number as a suffix (`datetime1`,
`ra1`, `dec1`, `azimuth1`, `altitude1`, ...), empty on the frames where it was not measured. The main
fragment, fragment 0, is always solved from the columns without a suffix (or with the optional suffix 0),
leaving out its points with `trajectory_use` = False (e.g. saturated flares kept in the file).

Points of the main fragment flagged with `flare` = True are used in the trajectory like the others,
unless they are also flagged with `trajectory_use` = False. Either way, the solver reports the time and
height of every flare point in the terminal and at the end of the saved report. A point used in the
trajectory is projected on it. A point not used is only a time, as where it was clicked is not a
measurement (e.g. a saturated flare in which the meteor cannot be seen), and takes the height of the
trajectory at that time. The report gives the flares of every station (its consecutive flare points),
with the uncertainties of their times and heights, and how the flares of different stations compare, as
they need not be the same flare (two cameras can see a fireball flaring at different times). The
uncertainties come from the Monte Carlo runs and from the duration of the frame the flare was seen in, as
it happened at any time within it. The times are relative to the clock of the reference station, whose
time offset is not estimated, so the Monte Carlo runs give it no uncertainty in time, but the frame does.

```
python -m wmpl.Formats.ECSV /path/to/event/*.ecsv --fragments
```

| Flag | Description |
| :--- | :--- |
| `--fragments` | After the main fragment, solve every additional fragment seen from at least 2 stations with the same options (Monte Carlo included), save each one in a `fragment_k` folder of the output directory, and print how it differs from the main one: convergence angle of its stations, radiant, velocities, heights, distance from the main path and how far ahead of the main fragment it is at the same time. |
| `--fragtimefit` | Estimate the station time offsets for every fragment, instead of reusing those of the main trajectory, which its larger number of points constrains better. Without the timing estimation (`-e`), the fragments take the same fixed offsets as the main fragment. |
| `--fragnosys` | Leave out of the Monte Carlo uncertainties of the fragments the systematic errors of the stations, which are otherwise included when the main trajectory has 3 stations or more (see below). |

Two solutions differ slightly even for identical data, as each goes through its own iterations of the
timing and velocity estimation: in tests, by about 0.001 deg and tens of metres with well spread stations,
and up to about 0.1 deg and a few hundred metres with nearly parallel ones. With the Monte Carlo runs, the
solutions compared are still those with the original picks (not the best Monte Carlo runs, which differ by
about 1 sigma even for identical data), and the differences are also given in units of the combined
uncertainty. The time offsets reused by the fragments are also those of the main solution with the original
picks, the ones in its report.

The comparison notes a fragment more than 1.5 km off the main path at its first or last point. No fragment
of the main went further in fireballs simulated with MetSim (2 to 7 fragments, with the lateral speeds of
real fragmentations), while fragments of another body 2 km away did in 97 % of the cases. Such a fragment
may come from another body, or the stations may have measured different fragments under the same number.
Two stations measuring different fragments mostly give a line close to the main path anyway, which cannot be
told apart, and often no trajectory at all.

The comparison also gives the convergence angle of the stations of every fragment, against that of the main
trajectory, and notes one below 3 deg, the least the trajectory correlator of WMPL takes. Two stations that
hardly converge for a fragment, e.g. close to each other and looking the same way, still give a line, but one
which the small errors of their picks can turn far from the true path.

The pointing of a station can be off by the same amount at all its points (its calibration, or the centroid
of a saturated meteor), which does not average out with their number, so the Monte Carlo runs do not
include it. With 3 or more stations in the main trajectory, the others constrain its line, and the residuals
of each station across the line show its own error, as a constant and a slope along the trajectory. Their
typical size is added to the Monte Carlo runs of every fragment, as a random constant and slope of each
station in every run. The solution of the fragment is the one with the original picks, as without them, and
only its uncertainties change. On fireballs simulated with MetSim, with 2 to 7 fragments, noise and
saturation, the true direction of a fragment was within the 95 % region of its uncertainties in 93 % of the
cases with them, and in 75 % without. With 2 stations in the main trajectory there is nothing to measure,
and the solver says so.

A fragment seen with enough points (4 or more) from only one station cannot be solved, as one station gives a
plane and not a line, and the trajectory of a fragment seen from more can still fail, or say little: its
stations can converge by less than 3 deg, or it can be more than 1.5 km off the main path (see above). The
points of such a fragment are projected on the main trajectory, each station on its own, at the times of their
frames corrected by the time offset of the station, as the flares are, and the report says why. The terminal
and the end of the saved report give the height and time of every point, how far off the main path it is and
how fast it moves away from it, and how far ahead of the main fragment it is at the same time. This assumes
that the fragment is on the main path, which one station cannot test. Only the part of a separation across the
line of sight is seen, so the distances off the path and the speeds are lower limits. The rest moves the point
along the line of sight, which the report gives as the error of the height, and of how far ahead it is, per
100 m off the path. The uncertainties come from the Monte Carlo runs of the main trajectory, with the scatter
of the points of the fragment as their noise, and do not include a separation from the main path. A station
which is not in the main trajectory is skipped, as its time offset is unknown.

When several fragments are close together, two stations can measure them under crossed numbers, which their
trajectory hardly shows (see above). So the fragments of every two stations are also matched by where they
are along the main path at the same times, whatever their numbers in the files. The points of every station
are projected on the main trajectory on their own, as above, and the distance of two fragments is how far
apart along the path their projections are, as a constant difference in a joint fit of their positions in
time. Pairs more than 1.5 km apart are not matched, and the pairs are chosen by the least total distance
(Hungarian algorithm). The end of the saved report lists the matches of every two stations with the distance
of the next best candidate of either fragment, calls a match ambiguous when it is less than 0.5 km closer
than that one, and notes a match of different numbers, and a fragment which is not matched to the one with
its number at the other station. The numbers in the files are not changed. On fireballs simulated with
MetSim (2 or 3 stations, 2 to 7 fragments, noise and saturation, 100 events per case), 96 to 99 % of the
true pairs were matched. The 1 to 3 % of the matches which joined different fragments were all ambiguous,
and 0.5 % of the true pairs were noted as not matched. As the matching does not use the numbers, crossed
numbers are noted without ambiguity about two times out of three, and as ambiguous in most of the rest.

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
