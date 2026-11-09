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
- [Drag against the rotating air](#drag-against-the-rotating-air)

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
gravity of the Earth instead:

    displacement(t) = integral from 0 to t of g(L(q)) (l(t) - l(q))/v(q) dq

where `L(q)` is the point of the line at the time `q`, `l` the length travelled and `v` the velocity of drag
alone, the one fitted above. At a constant velocity this is the free fall from the line, and in a uniform
field it is `g` times the scaled drop. Gravity is interpolated by a cubic in the distance along the line, and
the velocity enters through integrals of the motion tabulated from the fit, so the displacement is exact where
the fitted velocity changes its slope. It follows fireballs integrated in 3D with drag and gravity to 0.5-2.4
cm, against 0.2-2.1 m for the scaled drop along the vertical. On the same fireballs, without noise, the
radiant error falls from 0.6-3.6 arcsec to 0.00-0.18 arcsec and the largest error across the path from
0.08-1.3 m to 0.01-0.26 m. The radiant of Winchcombe moves by 7 arcsec, where the two drops differ by up to
2.8 m across the 97 km of the path.

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

## Drag against the rotating air

The air rotates with the Earth, eastwards at up to 465 m/s at the equator and about 330 m/s at 45 deg, in the
ECI frame the solver works in. Drag acts against the velocity of the meteor relative to the air, not against
its ECI velocity, so it pulls the velocity of a decelerating meteor across its path towards the velocity of
the air. In ECI coordinates a decelerating meteor then does not move along a straight line, even without
gravity: the fireballs below, which slow down to 4-10 km/s, are pulled by 80-600 m. A meteor at a constant
velocity is not pulled.

With the gravity correction on, the solver also moves the points of its line by this displacement. The
velocity across the line, `p`, follows `dp/dt = (a/w)*(u_perp - p)`, where `a` is the deceleration by drag,
`w` the speed relative to the air and `u_perp` the velocity of the air across the line. It is 0 at the first
point, so the line stays the direction of motion at the first point in ECI coordinates, which the orbit is
computed from. Like the drop, it depends only on the velocity, which comes from the same fit to the lengths
of the points, without a model of the deceleration. These are the lengths on the line of the pass with the
drop, which do not hold the speed gravity adds along the path, so a velocity changed by gravity alone does not
move the points. The velocity of the air is taken at the points along the line, which changes the
displacement by 0.9-13 m in the fireballs checked.

In the pass with the drop, after fitting the line with the scaled drop, the solver computes the displacement
from the lengths of the points on that line and fits the line again with it. The lengths on a line depend on
the displacement, as moving the line across also moves the points of the lines of sight closest to it along
the line, so this is repeated until the displacement changes by less than 0.1 m, at most 3 times. On the
fireballs below, the displacement from the first fit only gives radiant errors of up to 15 arcsec and errors
of the positions of up to 9.4 m, against 7.4 arcsec and 3.8 m. The median radiant error stays at 2.4 arcsec,
and the median error of the positions falls from 0.9 to 0.3 m.

- On 20 fireballs integrated in 3D with gravity and drag against the rotating air (5 fireballs of 14-25 km/s
  slowing down to 4-10 km/s in 3.3-9 s, each coming from 4 azimuths), the radiant error of a noise-free
  solution falls from 110-735 arcsec (median 360) to 0.5-7.4 arcsec (median 2.4), and the largest error of
  the positions across the path from 56-390 m to 0.1-3.8 m. The same fireballs integrated with drag against
  the ECI velocity are solved without the displacement to 0.5-3.5 arcsec and 0.05-1.2 m. The displacement
  matches the deviation of the integrated paths to 0.02-0.26 m out of 100-600 m, and the rest comes from the
  velocity fit and the drop.
- With 10 and 30 arcsec of noise (4 of the fireballs, 5 seeds each), the radiant errors are the same as those
  of the fireballs integrated without the rotating air and solved without the displacement, with medians of
  5-10 and 15-38 arcsec, against 170-540 and 220-570 arcsec without the displacement.
- For Winchcombe (8.2 s, 13.8 to 3.6 km/s), the points move by up to 237 m, the radiant by 154 arcsec and the
  end height up by 147 m, and the RMS of the angular residuals falls from 153 to 75 arcsec.

- `Trajectory(..., air_rotation=False)` keeps the dropped line. Trajectories pickled before the option was
  added keep it too, and so do solutions without the gravity correction.
- `traj.air_rotation_shift` holds the times from the first point and the displacements at those times, and
  `traj.airRotationShiftAt(t, radiant_eci)` gives the displacement at the time `t` from the first point,
  across the given line. The drag fit and the dynamic mass fit apply the same displacement as the solver.
- With the timing offsets estimated, the displacement also moves the lengths, so the offsets and the initial
  velocity can change too, when the stations see different parts of a fireball.
- A decelerating meteor needs two more fits of the line in each pass with the drop, and a better fit can
  reject more picks and solve the trajectory again: Winchcombe takes 4.2 s instead of 1.4 s.
- Winds, of tens of m/s at the heights of meteors, are not included. They would add to the velocity of the
  air.
