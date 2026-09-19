# WesternMeteorPyLib manuals

Task-oriented manuals for the larger tools in the library. Function-level documentation lives in the
docstrings; these pages cover how to actually run a tool end to end, what it reads, what it writes,
and what the options are for.

- [Solving a trajectory](Trajectory.md) — turning observations into a trajectory pickle, which is the
  input to most of the other tools.
- [REBOUND orbital integration](REBOUND.md) — integrating a meteoroid orbit backwards or forwards,
  close encounters and impacts, Monte Carlo clones, radiation forces, and chaos indicators.
- [DynestyMetSim](../Dynesty/README.md) — nested-sampling fits of the erosion ablation model to a
  meteor light curve and dynamics.

Something missing? The tools are all runnable as modules, so `python -m wmpl.<Package>.<Module> --help`
is usually the fastest answer.
