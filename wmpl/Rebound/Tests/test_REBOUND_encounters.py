""" Regression tests for the closest approaches measured during a real REBOUND integration.

Away from the Earth IAS15 takes steps of 1-2 days, so the distances at the step ends alone can miss
a planetary flyby's minimum by a large fraction of the step length. These tests integrate a
synthetic flyby with the integrator's own step choice and check the recorded minimum against a
dense re-integration of the same flyby. They are skipped when rebound/reboundx are not importable.
They need no ephemeris file: the massive bodies are placed on circular orbits.
"""

import os
import importlib.util

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

AU_KM = 149597870.7
DAY = 2*np.pi/365.25                   # REBOUND time units (G = 1, AU, M_sun) per day
KM_S = (86400.0/DAY)/AU_KM             # REBOUND velocity units per km/s


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name, skipping if REBOUND is unavailable. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_encounters", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    return module


def _circular(r_au, m, phase):
    """ Barycentric-ish state [x, y, z, vx, vy, vz] of a circular orbit around a unit-mass Sun. """

    v = np.sqrt((1.0 + m)/r_au)

    return [r_au*np.cos(phase), r_au*np.sin(phase), 0.0, -v*np.sin(phase), v*np.cos(phase), 0.0]


def _flybyTask(v_rel_kms, b_hill, t_before_days, direction):
    """ A Mercury flyby at b_hill Mercury Hill radii, closest approach ~t_before_days after the
    start (forward) or before it (backward). The object starts far from the Earth, so every body is
    tracked from the start.
    """

    m_mercury, m_earth = 1.66e-7, 3.0e-6
    planet_names = ["Sun", "Mercury", "Earth"]
    planet_states = [[0.0]*6, _circular(0.387, m_mercury, 0.3), _circular(1.0, m_earth, 2.5)]
    planet_masses = [1.0, m_mercury, m_earth]

    mercury = np.array(planet_states[1])
    u = np.array([0.6, -0.3, 0.74])
    u /= np.linalg.norm(u)
    n = np.cross(u, [0.0, 0.0, 1.0])
    n /= np.linalg.norm(n)

    b = b_hill*0.001475
    sign = 1.0 if direction == "forward" else -1.0
    r0 = mercury[:3] + b*n - sign*v_rel_kms*KM_S*u*t_before_days*DAY
    v0 = mercury[3:] + v_rel_kms*KM_S*u

    return {
        "planet_names": planet_names,
        "planet_states": planet_states,
        "planet_masses": planet_masses,
        "particle_states": [list(r0) + list(v0)],
        "particle_names": ["obj"],
        "direction": direction,
        "reference_frame": "heliocentric",
    }


def _closestApproach(reb, task, times):
    """ Mercury closest approach (AU, days) recorded by _integrateParticles for the given outputs. """

    task = dict(task, times=list(times))
    diag = reb._integrateParticles(task)["diagnostics"]["obj"]

    return diag["min_dist_au"]["Mercury"], diag["min_time_days"]["Mercury"]


@pytest.mark.parametrize("direction", ["forward", "backward"])
def testFlybyMinimumIsNotLimitedByTheIntegratorStep(reb, direction):
    """ A fast Mercury flyby at ~2 Hill radii, crossed in steps of millions of km. The step-end
    distances alone overestimate the minimum here (by 1% forward and 12% backward), so this fails
    without the refinement. The refined minimum must match a dense re-integration.
    """

    sign = 1.0 if direction == "forward" else -1.0
    task = _flybyTask(v_rel_kms=45.0, b_hill=2.0, t_before_days=2.0, direction=direction)

    # Natural IAS15 steps: a single output at the end of the span
    d_coarse, t_coarse = _closestApproach(reb, task, [0.0, sign*4.0*DAY])

    # Reference: outputs every ~2 minutes force short steps through the whole span
    d_dense, t_dense = _closestApproach(reb, task, sign*np.linspace(0.0, 4.0*DAY, 3001))

    assert d_coarse == pytest.approx(d_dense, rel=1e-6)
    assert abs(t_coarse - t_dense)*1440 < 0.5

    # The flyby really is an encounter well inside the 3 Hill-radii threshold
    assert d_dense/reb.HILL_RADII_AU["Mercury"] < 2.5
