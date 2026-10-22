""" Tests for the Borovicka et al. (2022) pressure factor Pf (wmpl.Utils.Physics.calcPf()) and what it takes from a
trajectory: the maximum dynamic pressure (maxDynamicPressure(), groundAlphaBetaFit()), the average zenith distance
(averageZenithDistance()) and the photometric mass (photometricMassBorovicka2022(), trajectoryLightCurve()).

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_PhysicsPf.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_PhysicsPf
"""

import copy
import os
import warnings
from types import SimpleNamespace

import numpy as np

from wmpl.Utils.AlphaBeta import alphaBetaVelocity, rescaleHeightToExponentialAtmosphere
from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect
from wmpl.Trajectory.AggregateAndPlot import computeMass
from wmpl.Utils.Physics import averageZenithDistance, calcMass, calcPf, groundAlphaBetaFit, maxDynamicPressure, \
    photometricMassBorovicka2022, trajectoryLightCurve
from wmpl.Utils.Pickling import loadPickle


# A solved trajectory shipped with the repository (2019-10-23, 67 km/s)
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"

LAT, LON, JD = np.radians(45.0), np.radians(15.0), 2459000.5


def _syntheticTrajectory(alpha, beta, v_ground, v_rotation, ht_beg=90e3, ht_end=40e3, ht_last=45e3, noise=50.0,
    seed=1):
    """ A trajectory stand-in whose two stations measure the speeds of an alpha-beta fireball, in the solver's
        inertial frame (v_rotation more than relative to the ground), with noise, down to ht_last, above the end
        height ht_end. Also returns the true maximum dynamic pressure over the whole trajectory (Pa). """

    rng = np.random.default_rng(seed)

    observations = []
    for k in range(2):
        hts = np.linspace(ht_beg - 500*k, ht_last, 60)
        lats, lons = np.full_like(hts, LAT), np.full_like(hts, LON)
        vels = alphaBetaVelocity(rescaleHeightToExponentialAtmosphere(lats, lons, hts, JD), alpha, beta, v_ground)
        observations.append(SimpleNamespace(ignore_station=False, ignore_list=np.zeros(len(hts), dtype=int),
            model_ht=hts, model_lat=lats, model_lon=lons, velocities=vels + v_rotation + rng.normal(0, noise,
            len(hts))))

    traj = SimpleNamespace(observations=observations, rbeg_ele=ht_beg, rend_ele=ht_end, rbeg_lat=LAT, rend_lat=LAT,
        rbeg_lon=LON, rend_lon=LON, jdt_ref=JD, orbit=SimpleNamespace(v_init=v_ground + v_rotation,
        v_init_norot=v_ground))

    hts = np.linspace(ht_beg, ht_end, 1000)
    lats, lons = np.full_like(hts, LAT), np.full_like(hts, LON)
    pressures = getAtmDensity_vect(lats, lons, hts, JD)*alphaBetaVelocity(
        rescaleHeightToExponentialAtmosphere(lats, lons, hts, JD), alpha, beta, v_ground)**2

    return traj, np.max(pressures)


def test_explicit_values():
    """ Borovicka et al. (2022, Paper II): Pf is one for 1 kg on a vertical trajectory at 21.5 km/s reaching
        21.5^1.5/100 MPa; the classes are bounded at 0.85, 0.27, 0.085 and 0.027. """

    pf, pf_class = calcPf(1e6*21.5**1.5/100, 0.0, 1.0, 21.5e3)
    assert np.isclose(pf, 1.0, rtol=1e-12) and pf_class == 1

    p_unit = 1e6*21.5**1.5/100
    for pf_value, pf_class in ((0.86, 1), (0.84, 2), (0.26, 3), (0.084, 4), (0.026, 5)):
        assert calcPf(pf_value*p_unit, 0.0, 1.0, 21.5e3)[1] == pf_class

    # The inputs are not changed in place
    p_max, v_0 = np.array(1e6), np.array(21.5e3)
    calcPf(p_max, 0.0, 1.0, v_0)
    assert p_max == 1e6 and v_0 == 21.5e3


def test_max_dynamic_pressure_of_synthetic_fireballs():
    """ The robust alpha-beta fit of noisy (50 m/s) inertial speeds, extrapolated 5 km below the last measurement,
        gives the true maximum pressure within 1%. Taking the inertial speeds as relative to the air would put it
        4-6% too high for 300 m/s of rotation. """

    for alpha, beta, v_ground in ((60.0, 1.5, 20000.0), (300.0, 0.5, 15000.0), (40.0, 0.8, 13500.0)):
        for v_rotation in (300.0, -300.0):

            traj, p_true = _syntheticTrajectory(alpha, beta, v_ground, v_rotation)
            p_max, ht_p_max, v_p_max = maxDynamicPressure(traj)

            assert abs(p_max/p_true - 1) < 0.01, (alpha, beta, v_ground, v_rotation)
            assert traj.rend_ele <= ht_p_max <= traj.rbeg_ele
            assert v_p_max <= v_ground

        # The rotation left in the speeds
        traj, p_true = _syntheticTrajectory(alpha, beta, v_ground, 300.0)
        traj.orbit.v_init_norot = traj.orbit.v_init
        assert maxDynamicPressure(traj)[0]/p_true - 1 > 0.02


def test_pf_from_the_example_trajectory():
    """ With a trajectory, calcPf() warns that p_max comes from a robust alpha-beta fit, and gives the Pf of the
        derived values. The maximum pressure of this fast, hardly decelerating meteor is at the end. """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf, pf_class = calcPf(mass=2e-7, traj=traj)

    messages = [str(w.message) for w in caught if "calcPf()" in str(w.message)]
    assert len(messages) == 1
    assert "robust alpha-beta fit" in messages[0] and "fitAlphaBeta(method='robust')" in messages[0]

    p_max, ht_p_max, _ = maxDynamicPressure(traj)
    zangle = averageZenithDistance(traj)
    assert (pf, pf_class) == calcPf(p_max, zangle, 2e-7, traj.orbit.v_init_norot)

    assert np.isclose(ht_p_max, traj.rend_ele, rtol=0, atol=100.0)

    # The average zenith distance is a fraction of a degree from the one at the reference point
    assert abs(np.degrees(zangle - (np.pi/2 - traj.orbit.elevation_apparent_norot))) < 0.5


def test_light_curve_is_aggregate_and_plot_s():
    """ The light curve is the one AggregateAndPlot.computeMass() takes, which gives the same mass as before it was
        taken out of it. """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    time_arr, mag_arr, ht_arr = trajectoryLightCurve(traj)

    assert np.all(np.diff(time_arr) >= 0) and np.all(mag_arr < 8)
    assert np.all((ht_arr <= traj.rbeg_ele + 1000) & (ht_arr >= traj.rend_ele - 1000))

    assert computeMass(traj, 1210) == 4.509403215639926e-06
    assert computeMass(traj, 1210) == calcMass(time_arr, mag_arr, traj.orbit.v_avg_norot, P_0m=1210)


def test_pf_computes_the_photometric_mass():
    """ Without a mass, calcPf() computes Paper I's photometric mass from the trajectory's light curve, at the speeds
        of the same robust alpha-beta fit, and says so; without magnitudes it needs the mass. """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    time_arr, mag_arr, ht_arr = trajectoryLightCurve(traj)
    velocity_at = groundAlphaBetaFit(traj)
    lats = np.interp(ht_arr, [traj.rend_ele, traj.rbeg_ele], [traj.rend_lat, traj.rbeg_lat])
    lons = np.interp(ht_arr, [traj.rend_ele, traj.rbeg_ele], [traj.rend_lon, traj.rbeg_lon])
    mass = calcMass(time_arr, mag_arr, velocity_at(ht_arr, lats, lons), tau='rc2001', P_0m=1500.0,
        lum_eff_mass=1.0)

    assert np.isclose(photometricMassBorovicka2022(traj), mass, rtol=1e-9)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf = calcPf(traj=traj)

    messages = [str(w.message) for w in caught if "calcPf()" in str(w.message)]
    assert len(messages) == 1 and "initial photometric mass" in messages[0] and "brightest" in messages[0]
    assert np.isclose(pf[0], calcPf(mass=mass, traj=traj)[0], rtol=1e-9)

    # Without magnitudes
    traj_dark = copy.deepcopy(traj)
    for obs in traj_dark.observations:
        obs.absolute_magnitudes = None
    assert photometricMassBorovicka2022(traj_dark) is None
    try:
        calcPf(traj=traj_dark)
    except ValueError:
        pass
    else:
        raise AssertionError("No ValueError without magnitudes nor mass")
    assert np.isclose(calcPf(mass=mass, traj=traj_dark)[0], pf[0], rtol=1e-9)


def test_inputs_are_checked():

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    for kwargs in (dict(p_max=1e5, mass=1.0, traj=traj), dict(zangle=0.5, mass=1.0, traj=traj),
            dict(p_max=1e5, zangle=0.5, mass=1.0), dict(p_max=1e5, zangle=0.5, v_0=2e4)):
        try:
            calcPf(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("No ValueError for {}".format(sorted(kwargs)))


if __name__ == "__main__":

    test_functions = [
        test_explicit_values,
        test_max_dynamic_pressure_of_synthetic_fireballs,
        test_pf_from_the_example_trajectory,
        test_light_curve_is_aggregate_and_plot_s,
        test_pf_computes_the_photometric_mass,
        test_inputs_are_checked,
    ]

    failed = 0
    for test_func in test_functions:

        try:
            test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {!r}".format(test_func.__name__, e))

    print()
    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
