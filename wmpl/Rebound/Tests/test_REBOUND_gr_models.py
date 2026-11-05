""" Tests for the general relativity models of the REBOUND integration (GR_MODELS).

An orbit around the Sun with nothing else to perturb it precesses only through general relativity,
at 6*pi*GM/(c^2*a*(1 - e^2)) per orbit. Each model is checked against that rate, and "none" against
a Keplerian orbit, whose perihelion does not move.

These integrate for real and are skipped when rebound/reboundx are not importable. They need no
ephemeris file.
"""

import os
import importlib.util

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

# Orbit of the test particle, well inside the Earth's orbit so that the Earth never comes near
A_AU = 0.4
ECC = 0.5
YEARS = 100.0


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name, skipping if REBOUND is unavailable. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_gr_models", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    return module


def _perihelionAdvance(reb, gr):
    """ Advance of the longitude of perihelion (rad) over YEARS, with the given GR model.

    The Earth is in the task only because the integration needs one, and the Sun's J2 is off, so
    general relativity is the only thing that can move the perihelion. The Earth's mass is 1e-15
    rather than zero, which turns its harmonics into nan; its pull moves the perihelion by ~1e-13
    rad over the span, against 2.5e-4 rad from general relativity.
    """

    q = A_AU*(1.0 - ECC)
    task = {
        "planet_names": ["Sun", "Earth"],
        "planet_states": [[0.0]*6, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        "planet_masses": [1.0, 1e-15],
        "particle_states": [[q, 0.0, 0.0, 0.0, np.sqrt((1.0 + ECC)/q), 0.0]],
        "particle_names": ["obj"],
        "times": [0.0, reb.daysToSimTime(YEARS*365.25)],
        "direction": "forward",
        "reference_frame": "heliocentric",
        "integrator": "ias15",
        "sun_j2": False,
        "gr": gr,
    }

    rows = reb._integrateParticles(task)["outputs"]["obj"]
    varpi = [row[2].Omega + row[2].omega for row in rows]

    return (varpi[-1] - varpi[0] + np.pi)%(2*np.pi) - np.pi


def testNoneModelIsInTheChoices(reb):

    assert "none" in reb.GR_MODELS


@pytest.mark.parametrize("gr", ["gr", "gr_full"])
def testRelativisticModelsGiveThePerihelionPrecession(reb, gr):

    period = 2*np.pi*A_AU**1.5
    n_orbits = reb.daysToSimTime(YEARS*365.25)/period
    expected = n_orbits*6*np.pi/(reb.rbxConstants.C**2*A_AU*(1.0 - ECC**2))

    # The analytic rate is the first post-Newtonian one, which both models reproduce to 0.02%
    assert abs(_perihelionAdvance(reb, gr) - expected) < 0.01*expected


def testNoneLeavesTheOrbitKeplerian(reb):

    # A thousandth of the relativistic advance: what is left is the integrator's own error
    expected_gr = abs(_perihelionAdvance(reb, "gr"))
    assert abs(_perihelionAdvance(reb, "none")) < 1e-3*expected_gr


def testUnknownModelIsRefused(reb):

    with pytest.raises(ValueError):
        _perihelionAdvance(reb, "newtonian")
