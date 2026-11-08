""" Tests of the Borovicka et al. (2022) pressure factor Pf (Physics.calcPf()) and of the values it derives
    from a trajectory.

With a trajectory, calcPf() takes the maximum dynamic pressure from maxDynamicPressure(), the zenith distance
from averageZenithDistance() and the initial speed relative to the ground from the orbit. The mass is always
given. photometricMassBorovicka2022() computes the mass of Paper I from the light curve of a trajectory.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_PhysicsPf.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import copy
import io
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from wmpl.Trajectory.AggregateAndPlot import computeMass
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, JDT_REF, observeMeteor
from wmpl.Utils.AlphaBeta import alphaBetaVelocity, rescaleHeightToExponentialAtmosphere
from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect
from wmpl.Utils.Physics import averageZenithDistance, calcMass, calcPf, groundDragFit, groundVelocityFit, \
    maxDynamicPressure, photometricMassBorovicka2022, trajectoryLatLon, trajectoryLightCurve
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import cartesian2Geo, eci2RaDec, raDec2AltAz


# Solved trajectory shipped with the repository (2019-10-23, 67 km/s)
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    'Dynesty', 'examples', '20191023_091225')
EXAMPLE_PICKLE = '20191023_091225_trajectory.pickle'

# Location and time of the synthetic alpha-beta fireballs
LAT, LON, JD = np.radians(45.0), np.radians(15.0), 2459000.5



def _alphaBetaTrajectory(alpha, beta, v_ground, v_rotation, ht_beg=90e3, ht_end=40e3, ht_last=45e3, \
    noise=50.0, seed=1):
    """ Make a stand-in trajectory whose two stations measure the speeds of an alpha-beta fireball in the
        inertial frame of the solver, with noise, down to ht_last above the end height.

    Arguments:
        alpha: [float] Alpha parameter.
        beta: [float] Beta parameter.
        v_ground: [float] Initial speed relative to the ground (m/s).
        v_rotation: [float] Speed of the Earth's rotation along the path, added to the measured speeds (m/s).

    Keyword arguments:
        ht_beg: [float] Beginning height (m). 90 km by default.
        ht_end: [float] End height (m). 40 km by default.
        ht_last: [float] Height of the last measured speed (m). 45 km by default.
        noise: [float] Standard deviation of the speeds (m/s). 50 m/s by default.
        seed: [int] Seed of the noise. 1 by default.

    Return:
        traj: [SimpleNamespace] Stand-in trajectory.
        p_true: [float] True maximum dynamic pressure over the whole trajectory (Pa).
    """

    rng = np.random.default_rng(seed)

    # Measure the speeds from two stations
    observations = []
    for k in range(2):

        hts = np.linspace(ht_beg - 500*k, ht_last, 60)
        lats, lons = np.full_like(hts, LAT), np.full_like(hts, LON)
        vels = alphaBetaVelocity(rescaleHeightToExponentialAtmosphere(lats, lons, hts, JD), alpha, beta, \
            v_ground)

        observations.append(SimpleNamespace(ignore_station=False, ignore_list=np.zeros(len(hts), dtype=int), \
            model_ht=hts, model_lat=lats, model_lon=lons, \
            velocities=(vels + v_rotation + rng.normal(0, noise, len(hts)))))

    traj = SimpleNamespace(observations=observations, rbeg_ele=ht_beg, rend_ele=ht_end, rbeg_lat=LAT, \
        rend_lat=LAT, rbeg_lon=LON, rend_lon=LON, jdt_ref=JD, \
        orbit=SimpleNamespace(v_init=(v_ground + v_rotation), v_init_norot=v_ground))

    # Compute the true maximum pressure over the whole trajectory
    hts = np.linspace(ht_beg, ht_end, 1000)
    lats, lons = np.full_like(hts, LAT), np.full_like(hts, LON)
    vels = alphaBetaVelocity(rescaleHeightToExponentialAtmosphere(lats, lons, hts, JD), alpha, beta, v_ground)
    p_true = np.max(getAtmDensity_vect(lats, lons, hts, JD)*vels**2)

    return traj, p_true



def _solvedDragMeteor(*args):
    """ Simulate, observe and solve a fireball decelerated by drag with a growing ballistic coefficient.

    Arguments:
        *args: [list] Arguments of SyntheticTrajectory.DragMeteor.

    Return:
        traj: [Trajectory object] Solved trajectory.
        p_true: [float] True maximum dynamic pressure between the beginning and end heights of the solution,
            with the speeds relative to the ground (Pa).
    """

    meteor = DragMeteor(*args)
    traj = observeMeteor(meteor, calc_orbit=True)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    # Compute the true pressures relative to the ground
    v_rotation = traj.orbit.v_init - traj.orbit.v_init_norot
    t = np.linspace(0, meteor.t_vis_end - meteor.t_vis_beg, 2000)
    llh = np.array([cartesian2Geo(JDT_REF + ti/86400, *eci) for ti, eci in zip(t, meteor.eciAtTime(t))])
    pressures = getAtmDensity_vect(llh[:, 0], llh[:, 1], llh[:, 2], traj.jdt_ref) \
        *(meteor.velocityAtTime(t) - v_rotation)**2

    # Take the maximum between the solved beginning and end heights
    inside = (llh[:, 2] <= traj.rbeg_ele) & (llh[:, 2] >= traj.rend_ele)

    return traj, np.max(pressures[inside])



def testPaperIIExampleAndClasses():
    """ Check that Pf is 1 for 1 kg on a vertical trajectory at 21.5 km/s reaching 21.5^1.5/100 MPa, the
        bounds of the classes of Paper II, and that the inputs are not changed in place.
    """

    p_unit = 1e6*21.5**1.5/100

    pf, pf_class = calcPf(p_unit, 0.0, 1.0, 21.5e3)
    assert np.isclose(pf, 1.0, rtol=1e-12)
    assert pf_class == 1

    # Check the bounds of the classes
    for pf_value, pf_class in ((0.86, 1), (0.84, 2), (0.26, 3), (0.084, 4), (0.026, 5)):
        assert calcPf(pf_value*p_unit, 0.0, 1.0, 21.5e3)[1] == pf_class

    # Check that the inputs are not changed in place
    p_max, v_0 = np.array(1e6), np.array(21.5e3)
    calcPf(p_max, 0.0, 1.0, v_0)

    assert p_max == 1e6
    assert v_0 == 21.5e3



def testTheAlphaBetaFitGivesTheMaxPressureOfAlphaBetaFireballs():
    """ Check that the robust alpha-beta fit of noisy inertial speeds of alpha-beta fireballs, extrapolated
        5 km below the last measurement, gives the true maximum pressure within 1 %, and that leaving the
        Earth's rotation in the speeds puts it more than 2 % too high.
    """

    for alpha, beta, v_ground in ((60.0, 1.5, 20000.0), (300.0, 0.5, 15000.0), (40.0, 0.8, 13500.0)):
        for v_rotation in (300.0, -300.0):

            traj, p_true = _alphaBetaTrajectory(alpha, beta, v_ground, v_rotation)
            p_max, ht_p_max, v_p_max = maxDynamicPressure(traj, velocity_fit='alphabeta')

            assert abs(p_max/p_true - 1) < 0.01, (alpha, beta, v_ground, v_rotation)
            assert traj.rend_ele <= ht_p_max <= traj.rbeg_ele
            assert v_p_max <= v_ground

        # Leave the rotation in the speeds
        traj, p_true = _alphaBetaTrajectory(alpha, beta, v_ground, 300.0)
        traj.orbit.v_init_norot = traj.orbit.v_init

        assert maxDynamicPressure(traj, velocity_fit='alphabeta')[0]/p_true - 1 > 0.02



def testTheDragFitGivesTheMaxPressureOfDeceleratingFireballs():
    """ Check that the drag fit gives the true maximum pressure of fireballs whose ballistic coefficient grows
        within 1 %, while the robust alpha-beta fit puts it more than 2 % too high.
    """

    for args in ((18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0), (25000.0, 60.0, 30.0, 0.005, 0.04, 3.0, 7.0)):

        traj, p_true = _solvedDragMeteor(*args)

        assert abs(maxDynamicPressure(traj)[0]/p_true - 1) < 0.01, args
        assert maxDynamicPressure(traj, velocity_fit='alphabeta')[0]/p_true - 1 > 0.02, args



def testPfFromTheExampleTrajectory():
    """ Check that calcPf() with a trajectory warns which fit gives p_max, and gives the Pf of the derived
        values. The maximum pressure of this fast meteor, which hardly decelerates, is at the end.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    # Check the warning of the default drag fit
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pf, pf_class = calcPf(mass=2e-7, traj=traj)

    messages = [str(w.message) for w in caught if 'calcPf()' in str(w.message)]
    assert len(messages) == 1
    assert 'the drag fit of the trajectory (wmpl.Utils.DragFit)' in messages[0]

    # Check the warning of the alpha-beta fit
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pf_ab = calcPf(mass=2e-7, traj=traj, velocity_fit='alphabeta')

    messages = [str(w.message) for w in caught if 'calcPf()' in str(w.message)]
    assert "fitAlphaBeta(method='robust')" in messages[0]
    p_max_ab = maxDynamicPressure(traj, velocity_fit='alphabeta')[0]
    assert pf_ab == calcPf(p_max_ab, averageZenithDistance(traj), 2e-7, traj.orbit.v_init_norot)

    # Check the Pf of the derived values
    p_max, ht_p_max, _ = maxDynamicPressure(traj)
    zangle = averageZenithDistance(traj)

    assert (pf, pf_class) == calcPf(p_max, zangle, 2e-7, traj.orbit.v_init_norot)
    assert np.isclose(ht_p_max, traj.rend_ele, rtol=0, atol=100.0)

    # Check that the average zenith distance is close to the one at the reference point
    assert abs(np.degrees(zangle - (np.pi/2 - traj.orbit.elevation_apparent_norot))) < 0.5



def testAnEarthGrazerIsRejected():
    """ Check that a trajectory whose lowest point is more than 1 km below its end height raises a ValueError,
        as the heights are sampled only down to the end.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    traj.htmin_ele = traj.rend_ele - 5000

    with pytest.raises(ValueError):
        maxDynamicPressure(traj)



def testTheRadiantOfFixedStationsIsNotDerotated():
    """ Check that the zenith distance of a solution with fixed stations is the one of its radiant, which is
        already fixed to the ground, and that the derotation changes it for a moving-station solution.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    zangle_moving = averageZenithDistance(traj)

    traj.orbit_los_success = False
    zangle_fixed = averageZenithDistance(traj)

    # Compute the zenith distance of the radiant at the average point
    lat = (traj.rbeg_lat + traj.rend_lat)/2
    lon = (traj.rbeg_lon + traj.rend_lon)/2
    ra, dec = eci2RaDec(traj.radiant_eci_mini)
    _, elev = raDec2AltAz(ra, dec, traj.jdt_ref, lat, lon)

    assert np.isclose(zangle_fixed, np.pi/2 - elev, rtol=0, atol=1e-12)
    assert abs(np.degrees(zangle_fixed - zangle_moving)) > 0.01



def testTheLightCurveIsTheOneOfAggregateAndPlot():
    """ Check that the light curve is sorted, brighter than +8 and within the trajectory, and that
        AggregateAndPlot.computeMass() takes its mass from it.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    time_arr, mag_arr, ht_arr = trajectoryLightCurve(traj)

    assert np.all(np.diff(time_arr) >= 0)
    assert np.all(mag_arr < 8)
    assert np.all((ht_arr <= traj.rbeg_ele + 1000) & (ht_arr >= traj.rend_ele - 1000))

    assert computeMass(traj, 1210) == calcMass(time_arr, mag_arr, traj.orbit.v_avg_norot, P_0m=1210)



def testThePhotometricMassOfPaperI():
    """ Check that the photometric mass is computed from the light curve with the luminous efficiency of
        Paper I and the speeds of the drag fit, that it is None without magnitudes, and that a velocity fit
        which does not cover the light curve raises a ValueError.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    # Compute the mass from the light curve independently
    time_arr, mag_arr, ht_arr = trajectoryLightCurve(traj)
    velocity_at = groundDragFit(traj)
    mass = calcMass(time_arr, mag_arr, velocity_at(ht_arr, *trajectoryLatLon(traj, ht_arr)), tau='rc2001', \
        P_0m=1500.0, lum_eff_mass=1.0)

    assert np.isclose(photometricMassBorovicka2022(traj), mass, rtol=1e-9, atol=0)

    # Remove the magnitudes
    traj_dark = copy.deepcopy(traj)
    for obs in traj_dark.observations:
        obs.absolute_magnitudes = None

    assert photometricMassBorovicka2022(traj_dark) is None

    # Leave the bottom of the light curve outside the velocity fit
    ht_cut = np.median(ht_arr)

    def partialFit(hts, lats, lons):
        return np.where(np.asarray(hts) > ht_cut, velocity_at(hts, lats, lons), np.nan)

    with pytest.raises(ValueError):
        photometricMassBorovicka2022(traj, velocity_at=partialFit)



def testTheInputsAreChecked():
    """ Check that a missing mass, a trajectory together with explicit values and missing values without a
        trajectory raise a ValueError.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    for kwargs in (dict(traj=traj), dict(p_max=1e5, mass=1.0, traj=traj), \
        dict(zangle=0.5, mass=1.0, traj=traj), dict(p_max=1e5, zangle=0.5, mass=1.0), \
        dict(p_max=1e5, zangle=0.5, v_0=2e4)):

        with pytest.raises(ValueError):
            calcPf(**kwargs)



def testTheVelocityFitChoice():
    """ Check that an unknown velocity fit raises a ValueError, and that the alpha-beta fit is made with a
        warning when the drag model cannot be fitted.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    with pytest.raises(ValueError):
        groundVelocityFit(traj, velocity_fit='jacchia')

    # Make the drag fit fail
    drag_module = sys.modules['wmpl.Utils.DragFit']
    drag_fit_class = drag_module.DragFit

    def failingFit(traj):
        raise ValueError("no fit")

    drag_module.DragFit = failingFit

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, velocity_fit = groundVelocityFit(traj)
            calcPf(mass=2e-7, traj=traj)

    finally:
        drag_module.DragFit = drag_fit_class

    messages = [str(w.message) for w in caught]
    assert velocity_fit == 'alphabeta'
    assert any('The drag model could not be fitted (no fit)' in message for message in messages)
    assert any(('calcPf()' in message) and ('robust alpha-beta fit' in message) for message in messages)
