""" Tests for the Monte Carlo realizations that reboundSimulate integrates: the ones it draws from the trajectory,
and the ones it is given, e.g. run back through the atmosphere first.

The integrations run on circular planetary orbits instead of the DE430 kernel, which is not in the repository.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

import wmpl.Rebound.REBOUND as reb
from wmpl.Utils.Pickling import loadPickle


EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"


class _CircularEphemeris(object):
    """ Stands in for an opened DE430 kernel: the planetary barycentres on circular orbits around the Sun, spread in
        longitude, with the planets at their barycentres and the Moon 384400 km from the Earth. """

    AU_KM = 149597870.7
    GM_SUN = 1.32712440018e11
    A_AU = {1: 0.387, 2: 0.723, 3: 1.0, 4: 1.524, 5: 5.203, 6: 9.537, 7: 19.19, 8: 30.07}

    def __getitem__(self, segment):
        center, target = segment
        pos, vel = np.zeros(3), np.zeros(3)

        if (center == 0) and (target in self.A_AU):
            a, lon = self.A_AU[target]*self.AU_KM, 0.7*target
            pos = a*np.array([np.cos(lon), np.sin(lon), 0.0])
            vel = np.sqrt(self.GM_SUN/a)*86400*np.array([-np.sin(lon), np.cos(lon), 0.0])

        elif (center, target) == (3, 301):
            pos, vel = np.array([384400.0, 0.0, 0.0]), np.array([0.0, 1.022*86400, 0.0])

        return SimpleNamespace(compute_and_differentiate=lambda jd: (pos, vel))


@pytest.fixture
def circularPlanets(monkeypatch):
    if not reb.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    monkeypatch.setattr(reb, "SPK", SimpleNamespace(open=lambda path: _CircularEphemeris()))


def _exampleTraj(sigma=None):
    """ The example trajectory, with a diagonal state vector covariance of the given sigma (m and m/s) if given. """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    if sigma is not None:
        traj.uncertainties = SimpleNamespace()
        traj.state_vect_cov = np.diag([sigma**2]*6)

    return traj, np.concatenate([traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini])


def test_sampled_realizations_are_the_draws_reboundSimulate_made_before():
    """ sampleStateVectors draws exactly what reboundSimulate drew inline, so seeded runs reproduce as before. """

    traj, state_vect = _exampleTraj(sigma=50.0)
    rng = np.random.default_rng(7)
    before = [rng.multivariate_normal(list(state_vect), traj.state_vect_cov) for _ in range(5)]

    assert np.array_equal(reb.sampleStateVectors(traj, 5, random_seed=7), before)
    assert reb.sampleStateVectors(traj, 1, random_seed=7) == []
    assert reb.sampleStateVectors(_exampleTraj()[0], 5, random_seed=7) == []


def test_given_realizations_are_integrated_as_clones(circularPlanets):
    """ Realizations given to reboundSimulate are integrated as its Monte Carlo clones, in that order: one equal to
        the nominal state ends where the nominal solution does, and one 1 km/s off does not. Without a trajectory or
        realizations, only the nominal solution is integrated. """

    traj, state_vect = _exampleTraj()
    offset = state_vect + np.array([0, 0, 0, 1000.0, 0, 0])
    kwargs = dict(direction="backward", sim_days=1, n_outputs=3, n_cpu=1, show_progress=False)

    outputs, outputs_mc = reb.reboundSimulate(traj.jdt_ref, state_vect,
        state_vect_realizations=[state_vect.copy(), offset], **kwargs)

    assert list(outputs_mc) == ["obj_MC_0", "obj_MC_1"]
    assert np.array_equal(outputs_mc["obj_MC_0"][-1][1], outputs[-1][1])
    assert not np.allclose(outputs_mc["obj_MC_1"][-1][1], outputs[-1][1])

    assert reb.reboundSimulate(traj.jdt_ref, state_vect, **kwargs)[1] == {}
