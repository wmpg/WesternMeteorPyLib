""" Tests for the lean output format of the REBOUND integration (lean_outputs).

The lean format keeps only the output times and the osculating elements, as arrays, so that an
ensemble of thousands of clones fits in memory. These tests integrate for real and check that it
carries exactly the values of the full rows, in both time directions and with both the adaptive and
a fixed-step integrator, and that it leaves the encounter diagnostics alone.

They need no ephemeris file (the task places the bodies on circular orbits) and are skipped when
rebound/reboundx are not importable.
"""

import os
import pickle
import importlib.util

import numpy as np
import pytest

from wmpl.Rebound.Tests.test_REBOUND_integrators import _departureTask


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name, skipping if REBOUND is unavailable. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_lean_outputs", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    return module


def _bothFormats(reb, task):
    """ The task's results in the full and in the lean format. """

    return reb._integrateParticles(dict(task)), reb._integrateParticles(dict(task, lean_outputs=True))


@pytest.mark.parametrize("integrator", ["ias15", "trace"])
@pytest.mark.parametrize("direction", ["forward", "backward"])
def testLeanOutputsHoldTheValuesOfTheRows(reb, integrator, direction):

    if integrator == "trace":
        try:
            reb.checkIntegratorAvailable("trace")
        except ValueError:
            pytest.skip("this REBOUND has no TRACE")

    # A backward TRACE run is integrated forward with the velocities reversed, so this also checks
    #   that the lean elements are read with the true velocities
    task = dict(_departureTask(direction, days=2*365.25, n_outputs=60), integrator=integrator)
    full, lean = _bothFormats(reb, task)

    rows = full["outputs"]["obj"]
    columns = lean["outputs"]["obj"]

    assert sorted(columns) == sorted(reb.LEAN_OUTPUT_KEYS)
    assert np.array_equal(columns["t"], [row[0] for row in rows])
    for key in reb.LEAN_OUTPUT_KEYS[1:]:
        assert np.array_equal(columns[key], [getattr(row[2], key) for row in rows]), key


def testLeanOutputsLeaveTheDiagnosticsUnchanged(reb):

    full, lean = _bothFormats(reb, _departureTask("backward", days=365.25, n_outputs=40))

    assert full["diagnostics"] == lean["diagnostics"]


def testLeanOutputsAreAFractionOfTheRows(reb):

    full, lean = _bothFormats(reb, _departureTask("forward", days=365.25, n_outputs=400))

    # What travels back from a worker process. With the four bodies of this task the rows pickle to
    #   ~250 bytes per output and the arrays to 56, the seven float64 values
    n_outputs = len(full["outputs"]["obj"])
    assert len(lean["outputs"]["obj"]["t"]) == n_outputs
    assert len(pickle.dumps(lean["outputs"]["obj"])) < 64*n_outputs < len(pickle.dumps(full["outputs"]["obj"]))
