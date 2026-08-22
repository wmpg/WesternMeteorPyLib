""" MetSim test package. Kept importable so the scenario builders / regression tests can be reused
programmatically. Excluded from wmpl's import-time submodule walk (see wmpl/__init__.py) so that
`import wmpl` never triggers test-module side effects (e.g. matplotlib.use("Agg")). """
