""" Tests that reboundSimulate can leave out the nominal solution and integrate the same clones. """

import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dynesty",
    "examples", "20191023_091225")
EXAMPLE_FILE = "20191023_091225_trajectory.pickle"


@pytest.fixture(scope="module")
def reb():
    """ REBOUND.py under an isolated name, skipping if it cannot integrate. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_skip_nominal", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    try:
        module.checkReboundxUsable()
    except RuntimeError as e:
        pytest.skip(str(e).split(".")[0])

    if not os.path.isfile(module.config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not available")

    return module


@pytest.fixture(scope="module")
def traj():

    from wmpl.Utils.Pickling import loadPickle

    if not os.path.isfile(os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)):
        pytest.skip("example trajectory not available")

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)

    # The example has no Monte Carlo runs: a covariance that moves the clones by small amounts
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([1e2, 1e2, 1e2, 1.0, 1.0, 1.0])

    return traj


def _run(reb, traj, **kwargs):

    return reb.reboundSimulate(None, None, traj=traj, direction="backward", sim_days=2*365.25, n_outputs=5,
        mc_runs=3, random_seed=11, lean_outputs=True, return_diagnostics=True, show_progress=False,
        n_cpu=1, **kwargs)


def test_the_clones_do_not_change_without_the_nominal_solution(reb, traj):

    calls = []
    original = reb._integrateParticles

    def counting(task, **kwargs):
        calls.append(task["particle_names"])
        return original(task, **kwargs)

    reb._integrateParticles = counting

    try:
        nominal, outputs_mc, diagnostics = _run(reb, traj)
        n_with = len(calls)
        del calls[:]
        skipped, outputs_skip, diagnostics_skip = _run(reb, traj, skip_nominal=True)
        n_without = len(calls)

    finally:
        reb._integrateParticles = original

    assert nominal is not None and skipped is None
    assert (n_with, n_without) == (4, 3)
    assert "obj" in diagnostics and "obj" not in diagnostics_skip

    assert sorted(outputs_mc) == sorted(outputs_skip)
    for name in outputs_mc:
        for key in outputs_mc[name]:
            assert np.array_equal(outputs_mc[name][key], outputs_skip[name][key])


def test_the_megno_needs_the_nominal_solution(reb, traj):

    with pytest.raises(ValueError, match="skip_nominal"):
        _run(reb, traj, skip_nominal=True, compute_megno=True)
