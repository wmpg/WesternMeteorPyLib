# Solving a trajectory

**The trajectory solver** turns multi-station meteor observations into a trajectory and a heliocentric
orbit, and saves the result as a pickle file. That pickle is the input to most of the rest of the
library, including [the REBOUND integrator](REBOUND.md), so this page is mainly about how to produce
one and what is inside it.

## Table of Contents

- [Running the solver](#running-the-solver)
- [The trajectory pickle](#the-trajectory-pickle)
- [Loading a pickle in Python](#loading-a-pickle-in-python)
- [The gravity drop](#the-gravity-drop)

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

## The gravity drop

With the gravity correction on (the default, `-g` turns it off), the solver fits the line of the trajectory
together with the drop of the meteor below it caused by gravity, from the first point. The line is the
direction of motion at the first point.

Drag also slows down the velocity which gravity adds across the path, like the rest of the velocity, so a
decelerating meteor drops less than one at a constant velocity. Its path bends at `dtheta/dt = g_perp/v`, as
in [the drag fit](DragFit.md). This depends only on the velocity: the size, shape and density of the meteoroid
and the density of the air act on the drop only through the deceleration they cause. The solver scales the
drop at a constant velocity by the ratio of the two. In the pass with the drop, it first fits the line with
the drop at a constant velocity, fits the velocity to the lengths of the points on that line, and fits the
line again with the scaled drop. The lengths on the straight line of the first pass are not used, as they also
hold the speed gravity adds along the path, which does not bend it: without drag, the drop across the first
direction is `g_perp*t^2/2` whatever the speed along it.

The velocity is fitted without a model of the deceleration, of the atmosphere or of the heights:

- It is constant, or linear in time over up to 8 segments joined end to end, and the Bayesian information
  criterion (Schwarz 1978) chooses the number of segments. Points which show no change of velocity keep a
  constant velocity, the drop is not scaled, and the solution is the one with the drop at a constant velocity.
  A deceleration of any shape, such as that of a fireball which fragments, gets the segments it needs.
- The velocity does not increase faster than gravity can speed a meteor up (9.83 m/s^2), as drag only slows
  a meteor down. This keeps the fit from following the scatter of few or noisy points, while it follows a
  meteor which gravity alone speeds up or slows down.
- It is held above 3 km/s, where ablation stops.

For Winchcombe (8.2 s, 13.8 to 3.6 km/s), the drop at the last point is 189 m instead of 245 m, the same as
from the bending of the drag fit, and along the whole path the two drops differ by at most 0.9 m. The end
height moves up by 41 m and the radiant by 27 arcsec. On fireballs integrated in 3D with drag and gravity,
which slow down to 4-10 km/s, the radiant error of a noise-free solution falls from 5-65 arcsec to 0.2-0.8
arcsec, and the largest error of the positions across the path from 4-71 m to 0.1-1.3 m.

The meteor falls from the line along the whole path, while the vertical turns with the curvature of the
Earth, so dropping every point by `g*t^2/2` along its own vertical tilts the line, by about `0.08*t^2`
arcsec after `t` seconds. The solver drops the points by the free fall of the meteor from the line under the
gravity of the Earth instead, a point mass with its oblateness (J2 of WGS84, around the axis of the ECI frame
of date):

    displacement(t) = integral from 0 to t of g(L(q)) (l(t) - l(q))/v(q) dq

where `L(q)` is the point of the line at the time `q`, `l` the length travelled and `v` the velocity of drag
alone, the one fitted above. At a constant velocity this is the free fall from the line, and in a uniform
field it is `g` times the scaled drop. Gravity is interpolated by a cubic in the distance along the line, and
the velocity enters through integrals of the motion tabulated from the fit, so the displacement is exact where
the fitted velocity changes its slope. It follows fireballs integrated in 3D with drag and gravity to 0.5-2.4
cm, against 0.2-2.1 m for the scaled drop along the vertical. On the same fireballs, without noise, the
radiant error falls from 0.6-3.6 arcsec to 0.00-0.18 arcsec and the largest error across the path from
0.08-1.3 m to 0.01-0.26 m. The radiant of Winchcombe moves by 7.7 arcsec, where the two drops differ by up to
2.8 m across the 97 km of the path. The oblateness accounts for 1 arcsec of it: on a sphere the drop across
its path differs by up to 0.45 m.

- `Trajectory(..., gravity_decel=False)` keeps the drop at a constant velocity. Trajectories pickled before
  the option was added keep it too.
- `Trajectory(..., gravity_free_fall=False)` drops every point along its own vertical, as before.
  Trajectories pickled before the option was added keep it too. `traj.gravityFreeFall()` gives the table of
  the motion along the line for the free fall, and `traj.gravityDropPoint()` drops a point of the line as the
  solver does.
- `traj.gravity_drop_scale` holds the times from the first point and the scale at those times, and
  `traj.gravityDropFactor(t)` gives the factor of the drop at the time `t` from the first point. The drag
  fit and the dynamic mass fit apply the same drop as the solver with it.
- With few points, gaps between the stations and a large scatter of the lengths, the points may only support
  a constant deceleration, which then stands for a deceleration concentrated at the end. On such a fireball
  (3.4 s, 57 points, 400 m of scatter), the drop differs from that of the drag fit by up to 2.7 m out of 53 m.
- As in the drag fit, the velocity is that of the solver, not relative to the air.
