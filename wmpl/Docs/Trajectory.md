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
trajectory at that time. The report gives the flares of every station, with the uncertainties of their
times and heights, and how the flares of different stations compare, as they need not be the same flare
(two cameras can see a fireball flaring at different times). The flares of a station are its flare points
in consecutive rows of its file, among the points of the main fragment: to separate two flares, leave a
point between them that is not flagged. This works in the same way for any frame rate, and when only every
second frame or fewer were picked. Two flares on consecutive points cannot be told apart, but two cameras
could not tell them apart either, as the comparison below allows a frame between the same flare. The
uncertainties come from the Monte Carlo runs and from the duration of the frame the flare was seen in, as
it happened at any time within it. The duration of a frame comes from the frame numbers of the file, if it
gives them, so it is that of one frame also when not every frame was picked. The times are relative to the
clock of the reference station, whose time offset is not estimated, so the Monte Carlo runs give it no
uncertainty in time, but the frame does.

Every flare also gets the speed of the meteor relative to the ground at its time, from the drag fit of the
trajectory (`wmpl.Utils.DragFit`, with the Earth's rotation along the path taken off its inertial speeds),
and the dynamic pressure p = rho*v^2 with the NRLMSISE air density at its height, with uncertainties from
the Monte Carlo runs and the frame as for the height. The error of the air density model is not included.
On fireballs simulated with MetSim, the drag fit gave the dynamic pressure at the fragmentations within
0.7 % in 90 % of the cases (0.2 % in the median), and better than a robust alpha-beta fit late in the
flight, where that one gave it about 4 % too high.

The noise of the Monte Carlo runs only moves the radiant, so the runs alone do not change the speed: refitting
the drag model on every run gave 2 to 3 m/s of scatter in these fireballs, against a real error of about
30 m/s. As `Trajectory.sampleMCVelocities()` does with the initial velocity, every run therefore shifts the
speed within its uncertainty from the covariance of the drag fit (`DragFit.motionStdAtTime()`), and the
pressure of the run follows. Near the fragmentations of the simulated fireballs that uncertainty, about
30 m/s, held the real error within 1 sigma in 50 to 55 % of the cases and within 2 sigma in 77 to 82 %,
so it is somewhat small: the bias of the fit where the meteor fragments is not included. Its share of the
uncertainty of the pressure, about 0.3 %, is small next to that of the frame.

When the solver estimated the initial velocity from a drag fit of the first part of the trajectory
(`--vinitdrag`, `wmpl.Utils.DragInitialVelocity`), which models a single body without fragmentation, the
report warns if a flare point lies at or above the lowest fitted point, giving the highest flare point of
every station, and suggests a `--vinitdraght` that keeps only the points above the highest flare.

Two cameras can mark the same flare a frame apart, e.g. as it lasts longer than a frame or peaks between
two, so the comparison takes one frame of the slower camera off the time between two flares before it
weighs it against its uncertainty. In the simulated fireballs, with the frames of each camera at a random
phase and each marking the frame with the most light, 4.9 % of the same flares of a 25 and a 13 fps camera
were further apart than 3 sigma of the duration of their frames, and 2.7 % with that frame taken off (2.1 %
and 1.1 % with two 25 fps cameras). When the solver estimated the time offsets, the uncertainty of the time
between two stations also includes their real error, 7 ms (68 % of the errors in the same simulations, with
a median of 4.4 ms), against about 0.24 ms from the fit and the Monte Carlo runs, which do not predict it.
It also enters the heights compared at the same time. It changed the rates above by 0.2 to 0.3 points, with
cameras of 25 fps; with faster cameras the frames weigh less and it matters more.

The report then proposes the flares of the event. Two flares of different stations are taken as the same
flare when they are consistent and each one is the only consistent flare of its station for the other, and
the joined flares make one flare of the event. A flare consistent with two or more flares of another station
is left alone as ambiguous, and so are the flares of a group that would hold two flares of one station. Every
flare of the event gives the values of its first point in time among its stations, how many frames apart its
stations saw it begin and how far apart their pressures are, which says how much joining them matters.
When two of the flares an ambiguous flare could be are flares of one station with a single point not flagged
between them, the report says so: a frame left unflagged inside a flare splits it in two.
Without the Monte Carlo runs the flares are not joined. In the simulated fireballs, with every camera flagging
the frames in which the light of the dust of a fragmentation exceeds the rest by 0.25 mag, this joined no
flares of different fragmentations. Of the fragmentations seen by two or more stations whose flares held only
that fragmentation, 92 % were joined and 8 % left ambiguous; windows of 3 and 5 frames instead of 3 sigma left
14 and 26 % ambiguous. A window of one frame did better (99 %) when every frame of a flare is flagged, but
missed more of the same flares marked only on their brightest frame. A third of the fragmentations seen by two
or more stations had a flare that already held another fragmentation, a median 85 ms apart, which no
comparison of stations can split.

Every flare of the event takes the values of its first point in time among its stations. A camera flags a
flare once the light it integrates over a frame stands out, on average after the fragmentation, and the
earliest camera is the least late. In the simulated fireballs (threshold 0.25 mag, 54 957 joined flares), with
the times compared to that of the fragmentation and the pressures to the true pressure of the main at the time
taken:

| Time of the joined flare | Bias in time | Bias in pressure | Error of the pressure, median / 90th pct. |
|---|---|---|---|
| first point of any camera (used) | -2 ms | -0.3 % | 1.5 % / 4.3 % |
| mean of the beginnings of the stations | +10 ms | +1.7 % | 1.8 % / 5.1 % |
| one station | +10 ms | +1.7 % | 2.0 % / 5.8 % |
| mean of the centres of the flares | +35 ms | +6.4 % | 6.4 % / 31 % |

The centres of the flares come after the fragmentation, as its dust shines for several frames. With a
threshold of 0.5 mag the first point does better still (1.5 % / 4.6 % against 2.6 % / 6.6 % for the mean of
the beginnings). With 0.1 mag, which flags almost any brightening, the mean of the beginnings does slightly
better (1.4 % / 5.9 % against 1.7 % / 7.0 %), as the first point is then early. The simulated frames are timed
at the middle of their exposure, and RMS times the frames of FF files at their beginning, but the flares and
the trajectory share those times and the solver fits the time offsets of the stations, so the heights and
pressures follow the position of the meteor in the frame either way.

```
python -m wmpl.Formats.ECSV /path/to/event/*.ecsv --fragments
```

| Flag | Description |
| :--- | :--- |
| `--fragments` | After the main fragment, solve every additional fragment seen from at least 2 stations with the same options (Monte Carlo included), save each one in a `fragment_k` folder of the output directory, and print how it differs from the main one: convergence angle of its stations, radiant, velocities, heights, distance from the main path and how far ahead of the main fragment it is at the same time. |
| `--fragtimefit` | Estimate the station time offsets for every fragment, instead of reusing those of the main trajectory, which its larger number of points constrains better. Without the timing estimation (`-e`), the fragments take the same fixed offsets as the main fragment. |
| `--fragnosys` | Leave out of the Monte Carlo uncertainties of the fragments the systematic errors of the stations, which are otherwise included when the main trajectory has 3 stations or more (see below). |

With `--kml` or `--kmlmeas`, the 3D KML of the main trajectory also shows every solved fragment, from its
first to its last point, in its colour in SkyFit2 (1 cyan, 2 green, 3 magenta, 4 blue, 5 orange, 6 purple,
7 white, 8 pink).

Two solutions differ slightly even for identical data, as each goes through its own iterations of the
timing and velocity estimation: in tests, by about 0.001 deg and tens of metres with well spread stations,
and up to about 0.1 deg and a few hundred metres with nearly parallel ones. With the Monte Carlo runs, the
solutions compared are still those with the original picks (not the best Monte Carlo runs, which differ by
about 1 sigma even for identical data), and the differences are also given in units of the combined
uncertainty. The time offsets reused by the fragments are also those of the main solution with the original
picks, the ones in its report.

The comparison notes a fragment more than 1.5 km off the main path at its first or last point. No fragment
of the main went further in fireballs simulated with MetSim (2 to 7 fragments, with the lateral speeds of
real fragmentations, stations converging by 20 deg), while fragments of another body 2 km away did in 97 %
of the cases. Such a fragment may come from another body, the stations may have measured different fragments
under the same number, or its trajectory may be poorly constrained: with stations converging by 8 deg, 8 to
14 % of the fragments of the main went further. Two stations measuring different fragments mostly give a line
close to the main path anyway, and often no trajectory at all.

With the Monte Carlo runs, the distance off the main path is also given in units of the uncertainty of both
paths there, from the scatter of their runs across it (the Mahalanobis distance), and a fragment more than
1.5 km off but within 3 sigma is noted as poorly constrained instead. On 24 fireballs simulated with MetSim
per case, solved after merging master, this kept 99 % of the fragments of another body 2 km away noted, and
cut the fragments of the main noted from 6 to 2 % with stations converging by 20 deg, and from 27 to 8 % by
8 deg. A fragment of the main is often more than 3 sigma off, as it truly separates from it by up to 600 m,
and the runs are slightly optimistic: 80 % of the errors were within their 95 % region.

With the time offsets of the main trajectory, the comparison also gives how far apart along the path of the
fragment every two of its stations see it at the same times, from their points on its line, as a constant
difference in a joint fit of their positions in time, and notes more than 1 km. Two stations measuring
different fragments see them where the line passes closest to their lines of sight, at different places
along it. In the simulated fireballs, 0.2 to 0.3 % of the fragments of the main went further, against 70 %
of the fragments whose 2 stations measured different fragments, and 56 % with 3 stations, one of which did.
Time offsets estimated for the fragment (`--fragtimefit`) would absorb the difference, so it is then not
given.

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
of the closest other candidate of either fragment (which can be closer than the match, as the pairs are
chosen together), calls a match ambiguous when it is less than 0.5 km closer than that one, and notes a
match of different numbers, and a fragment more than 1.5 km from the one with its number at the other
station. The numbers in the files are not changed. On fireballs simulated with MetSim (2 or 3 stations, 2
to 7 fragments, noise and saturation, 100 events per case), 96 to 99 % of the true pairs were matched. The
1 to 3 % of the matches which joined different fragments were all ambiguous, and 0.2 % of the true pairs
were more than 1.5 km apart. As the matching does not use the numbers, crossed numbers are noted without
ambiguity about two times out of three, and as ambiguous in most of the rest.

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
