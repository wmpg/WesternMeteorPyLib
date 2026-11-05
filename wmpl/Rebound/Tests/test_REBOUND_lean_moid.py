""" Tests that the lean outputs of the REBOUND integration keep the MOID of the full rows. They integrate for real,
with no ephemeris file (see test_REBOUND_lean_outputs), and are skipped when rebound/reboundx are not importable. """

import importlib.util
import os

import numpy as np
import pytest

from wmpl.Rebound.Tests.test_REBOUND_integrators import _departureTask


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_lean_moid", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    return module


def test_the_lean_outputs_keep_the_moid_of_the_rows(reb):

    task = dict(_departureTask("backward", days=2*365.25, n_outputs=20), compute_moid=True)

    full = reb._integrateParticles(dict(task))["diagnostics"]["obj"]["moid_au"]
    lean = reb._integrateParticles(dict(task, lean_outputs=True))["diagnostics"]["obj"]["moid_au"]

    assert full is not None and lean is not None
    assert sorted(full) == sorted(lean)

    for key in full:
        assert np.array_equal(np.array(full[key], dtype=float), np.array(lean[key], dtype=float), equal_nan=True)
