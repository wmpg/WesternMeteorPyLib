""" Tests for the photometric mass DynestyMetSim computes with a luminous efficiency model
(photometricMassLumEffModel()): the model is evaluated along the trajectory, at the speed of the trajectory's Jacchia
fit at each photometric point, with the mass iterated to self-consistency.

Run under pytest, or directly:

    python -m wmpl.Dynesty.Tests.test_DynestyPhotometricMass
"""

import os
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np

from wmpl.Dynesty.DynestyMetSim import ObservationData, photometricMassLumEffModel
from wmpl.Trajectory.Trajectory import jacchiaVelocityFunc
from wmpl.Utils.Physics import calcMass


# A synthetic decelerating meteor: 1 s of light peaking at -3 mag, 20 km/s at the start, losing 3 km/s by the end
TIME = np.linspace(0.0, 1.0, 51)
MAG_ABS = -3 + 5*(TIME - 0.6)**2
V_INIT = 20000.0
JACCHIA_FIT = np.array([-50.0, 3.0])

# A solved trajectory shipped with the repository, and a prior file to which the luminous efficiency lines are added
DYNESTY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_PICKLE = os.path.join(DYNESTY_DIR, "examples", "20191023_091225", "20191023_091225_trajectory.pickle")
BASE_PRIOR = os.path.join(DYNESTY_DIR, "priors", "stony_meteoroid_SingleBody.prior")


def _traj(jacchia_fit=JACCHIA_FIT):
    """ A trajectory stand-in with what photometricMassLumEffModel() reads. """

    v_avg = np.mean(jacchiaVelocityFunc(TIME, *JACCHIA_FIT, V_INIT))

    return SimpleNamespace(v_init=V_INIT, jacchia_fit=jacchia_fit, orbit=SimpleNamespace(v_init=V_INIT, v_avg=v_avg))


def test_model_is_evaluated_at_the_jacchia_speed_of_each_point():

    traj = _traj()
    vel_jacchia = jacchiaVelocityFunc(TIME, *JACCHIA_FIT, V_INIT)
    assert 2900 < V_INIT - vel_jacchia[-1] < 3100

    for lum_eff_type in (1, 4, 5, 7):
        mass, vel_used = photometricMassLumEffModel(TIME, MAG_ABS, traj, lum_eff_type, 1500.0)

        assert np.allclose(vel_used, vel_jacchia, rtol=0, atol=1e-9)
        assert np.isclose(mass, calcMass(TIME, MAG_ABS, vel_jacchia, tau=lum_eff_type, P_0m=1500.0), rtol=1e-12)


def test_mass_is_self_consistent():
    """ The mass the model is evaluated at is the mass it gives (the iteration of calcMass()). """

    traj = _traj()
    vel_jacchia = jacchiaVelocityFunc(TIME, *JACCHIA_FIT, V_INIT)

    for lum_eff_type in (1, 4, 7):
        mass, _ = photometricMassLumEffModel(TIME, MAG_ABS, traj, lum_eff_type, 1500.0)
        mass_at_mass = calcMass(TIME, MAG_ABS, vel_jacchia, tau=lum_eff_type, P_0m=1500.0, lum_eff_mass=mass)

        assert np.isclose(mass_at_mass, mass, rtol=1e-3), lum_eff_type


def test_without_a_usable_jacchia_fit_the_average_speed_is_used():
    """ No fit, or one with non-physical speeds, falls back on the average speed; a failed fit (zeros) is the
        constant initial speed. """

    for jacchia_fit in (None, np.array([-1e6, 50.0])):
        traj = _traj(jacchia_fit=jacchia_fit)
        mass, vel_used = photometricMassLumEffModel(TIME, MAG_ABS, traj, 1, 1500.0)

        assert np.ndim(vel_used) == 0 and vel_used == traj.orbit.v_avg
        assert np.isclose(mass, calcMass(TIME, MAG_ABS, traj.orbit.v_avg, tau=1, P_0m=1500.0), rtol=1e-12)

    _, vel_used = photometricMassLumEffModel(TIME, MAG_ABS, _traj(jacchia_fit=np.zeros(2)), 1, 1500.0)
    assert np.allclose(vel_used, V_INIT)


def test_numpy_model_codes_and_lists():
    """ The model code may come as a numpy integer and the light curve as lists, as loadPickleData() gives them. """

    traj = _traj()
    mass_int, _ = photometricMassLumEffModel(TIME, MAG_ABS, traj, 4, 1500.0)
    mass_np, _ = photometricMassLumEffModel(list(TIME), list(MAG_ABS), traj, np.int64(4), 1500.0)

    assert mass_np == mass_int


def _photometricMassFromPriors(lum_eff_lines):
    """ The m_init that ObservationData() estimates for the example trajectory with the given luminous efficiency
        lines in the prior file. """

    with open(BASE_PRIOR) as f:
        lines = [line for line in f if not line.startswith("lum_eff")]

    with tempfile.TemporaryDirectory() as tmp_dir:
        prior_path = os.path.join(tmp_dir, "test.prior")
        with open(prior_path, "w") as f:
            f.writelines(lines + [line + "\n" for line in lum_eff_lines])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ObservationData([EXAMPLE_PICKLE], prior_file_path=prior_path).m_init


def test_prior_file_luminous_efficiency():
    """ On the example trajectory: the constant model fixed without a lum_eff line takes the default 0.7 % (it used
        to fail dividing None), a constant 1.4 % halves that mass, and a model recomputes it. """

    mass_default = _photometricMassFromPriors(["lum_eff_type, 0, fix"])
    mass_double = _photometricMassFromPriors(["lum_eff_type, 0, fix", "lum_eff, 1.4, fix"])
    mass_rc2001 = _photometricMassFromPriors(["lum_eff_type, 1, fix"])

    assert np.isclose(mass_double, mass_default/2, rtol=1e-9)
    assert np.isfinite(mass_rc2001) and not np.isclose(mass_rc2001, mass_default, rtol=1e-2)


if __name__ == "__main__":
    test_model_is_evaluated_at_the_jacchia_speed_of_each_point()
    test_mass_is_self_consistent()
    test_without_a_usable_jacchia_fit_the_average_speed_is_used()
    test_numpy_model_codes_and_lists()
    test_prior_file_luminous_efficiency()
    print("All DynestyPhotometricMass checks passed.")
