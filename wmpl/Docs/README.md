# WesternMeteorPyLib manuals

Task-oriented manuals for the larger tools in the library. Function-level documentation lives in the
docstrings; these pages cover how to actually run a tool end to end, what it reads, what it writes,
and what the options are for.

- [Solving a trajectory](Trajectory.md) — turning observations into a trajectory pickle, which is the
  input to most of the other tools.
- [Correlating RMS data](Correlator.md) — finding and solving trajectories automatically across a whole
  RMS archive, the operation modes, the databases, and distributed processing across several servers.
- [REBOUND orbital integration](REBOUND.md) — integrating a meteoroid orbit backwards or forwards,
  close encounters and impacts, Monte Carlo clones, radiation forces, and chaos indicators.
- [DynestyMetSim](../Dynesty/README.md) — nested-sampling fits of the erosion ablation model to a
  meteor light curve and dynamics.
- [Orbit dissimilarity criteria](OrbitSimilarity.md) — measuring how similar two orbits are, which
  criterion to pick, and why their thresholds are not interchangeable.
- [Classifying a single orbit](OrbitClassification.md) — Tisserand, the cometary tests, and the
  Tancredi scheme for asteroids in cometary orbits.
- [Choosing a D-criterion threshold](DThresholds.md) — the other half of a stream search, and why
  the traditional threshold is too permissive to use.

Something missing? The tools are all runnable as modules, so `python -m wmpl.<Package>.<Module> --help`
is usually the fastest answer.
