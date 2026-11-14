""" Tests of the UT1 - UTC correction (wmpl.Utils.EarthRotation) of the trajectory solver and the orbit.

The observations are directions measured against the stars, while the ECI coordinates of wmpl turn with the
Earth by UTC instead of UT1.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_UT1Correction.py -v
"""

from __future__ import print_function, division, absolute_import

import datetime

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, STRAIGHT_V_INIT, \
    straightMeteorPoints
from wmpl.Utils.EarthRotation import DUT1_ENV, deltaUT1, eciToInertial, inertialToECI, utcRotationAngle
from wmpl.Utils.Math import vectNorm
from wmpl.Utils.TrajConversions import datetime2JD, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


# UT1 - UTC of the observations (s), a rotation of the Earth by 7.5 arcsec
DUT1 = 0.5



def angleArcsec(vect1, vect2):
    """ Compute the angle between two directions from their chord, precise for small angles.

    Arguments:
        vect1: [ndarray] First direction.
        vect2: [ndarray] Second direction.

    Return:
        [float] Angle (arcsec).
    """

    chord = np.linalg.norm(vectNorm(np.array(vect1)) - vectNorm(np.array(vect2)))

    return 3600*np.degrees(2*np.arcsin(chord/2))



def observeAgainstTheStars(dut1, meastype=1, **kwargs):
    """ Observe the straight meteor of SyntheticTrajectory against the stars, with the Earth turned by DUT1
        of UT1 - UTC, as R.A./Dec. or as the azimuths and altitudes reduced from them with UTC.

    Arguments:
        dut1: [float] UT1 - UTC given to the solver (s).

    Keyword arguments:
        meastype: [int] 1 for R.A./Dec., 2 for azimuths and altitudes. 1 by default.
        kwargs: [dict] Other arguments of the Trajectory object.

    Return:
        (traj, begin, radiant_eci):
            - traj: [Trajectory object] Trajectory object with the observations, not yet solved.
            - begin: [ndarray] ECI coordinates of the beginning of the meteor (m).
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    times, points, begin, radiant_eci = straightMeteorPoints()
    jds = STRAIGHT_JDT_REF + times/86400.0

    traj = Trajectory(STRAIGHT_JDT_REF, meastype=meastype, gravity_correction=False, show_plots=False, \
        save_results=False, verbose=False, filter_picks=False, dut1=dut1, **kwargs)

    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        lat, lon = np.radians(s_lat), np.radians(s_lon)
        stations = np.array([geo2Cartesian(lat, lon, ele, jd) for jd in jds])

        # Directions against the stars
        ra, dec = np.array([eci2RaDec(eciToInertial(point - station, DUT1)) \
            for point, station in zip(points, stations)]).T

        if meastype == 1:
            traj.infillTrajectory(ra, dec, times, lat, lon, ele, station_id=station_id, radec_jd=jds)

        else:
            azim, alt = raDec2AltAz_vect(ra, dec, jds, lat, lon)
            traj.infillTrajectory(azim, alt, times, lat, lon, ele, station_id=station_id)

    return traj, begin, radiant_eci



def distanceFromLine(point, begin, radiant_eci):
    """ Compute the distance of a point from the line through the beginning of the meteor.

    Arguments:
        point: [ndarray] ECI coordinates of the point (m).
        begin: [ndarray] ECI coordinates of the beginning of the meteor (m).
        radiant_eci: [ndarray] Radiant unit vector in ECI coordinates.

    Return:
        [float] Distance (m).
    """

    offset = point - begin

    return np.linalg.norm(offset - np.dot(offset, radiant_eci)*radiant_eci)



@pytest.mark.parametrize('meastype', [1, 2])
def testDirectionsAgainstTheStarsGiveTheTrueLine(meastype):
    """ Check that directions measured against the stars give the true line and radiant, and that the
        radiant of the orbit is the true one against the stars.

    Arguments:
        meastype: [int] 1 for R.A./Dec., 2 for azimuths and altitudes.
    """

    traj, begin, radiant_eci = observeAgainstTheStars(DUT1, meastype=meastype, monte_carlo=False)
    traj = traj.run()

    assert distanceFromLine(traj.state_vect_mini, begin, radiant_eci) < 0.01
    assert angleArcsec(traj.radiant_eci_mini, radiant_eci) < 0.01

    # The radiants of the orbit and of the line-of-sight fit are against the stars
    radiant_stars = eciToInertial(radiant_eci, DUT1)
    assert angleArcsec(raDec2ECI(traj.orbit.ra, traj.orbit.dec), radiant_stars) < 0.01
    assert angleArcsec(raDec2ECI(*traj.radiant_eq_mini), radiant_stars) < 0.01

    # The modelled directions are those of the measurements
    for obs in traj.observations:
        for azim, elev, model_azim, model_elev in zip(obs.azim_data, obs.elev_data, obs.model_azim, \
            obs.model_elev):

            assert angleArcsec(raDec2ECI(model_azim, model_elev), raDec2ECI(azim, elev)) < 0.01

    report = traj.saveReport(None, None, save_results=False, verbose=False)
    assert "UT1 - UTC: +0.5000 s" in report



def testWithoutTheCorrectionTheLineIsOff():
    """ Check that the same observations solved with UTC as UT1 are off by 3.7 m and 2.7 arcsec, as
        documented in Trajectory.md.
    """

    traj, begin, radiant_eci = observeAgainstTheStars(0.0, monte_carlo=False)
    traj = traj.run()

    assert 3.0 < distanceFromLine(traj.state_vect_mini, begin, radiant_eci) < 4.5
    assert 2.0 < angleArcsec(traj.radiant_eci_mini, radiant_eci) < 3.5



def testMonteCarloRunsKeepTheFrame():
    """ Check that the noisy directions of the Monte Carlo runs are turned as the measurements are, so that
        the solution stays on the true line.
    """

    traj, begin, radiant_eci = observeAgainstTheStars(DUT1, monte_carlo=True, mc_runs=4, mc_cores=1, \
        geometric_uncert=True)
    traj = traj.run()

    assert distanceFromLine(traj.state_vect_mini, begin, radiant_eci) < 0.05
    assert angleArcsec(traj.radiant_eci_mini, radiant_eci) < 0.05

    # The runs are not turned by the rotation of the Earth in UT1 - UTC
    un = traj.uncertainties
    assert 3600*np.degrees(np.hypot(un.ra*np.cos(traj.orbit.dec), un.dec)) < 0.05



def testOrbitIsComputedInTheInertialFrame():
    """ Check that the orbit turns the radiant by the rotation of the Earth in UT1 - UTC, and that the
        Earth-fixed values do not change.
    """

    _, _, begin, radiant_eci = straightMeteorPoints()

    orb_utc = calcOrbit(radiant_eci, STRAIGHT_V_INIT, STRAIGHT_V_INIT, begin, STRAIGHT_JDT_REF)
    orb = calcOrbit(radiant_eci, STRAIGHT_V_INIT, STRAIGHT_V_INIT, begin, STRAIGHT_JDT_REF, dut1=DUT1)

    # Earth-fixed values agree to the resolution of a float Julian date (3e-9 rad)
    for name in ['lat_ref', 'lon_ref', 'azimuth_apparent', 'elevation_apparent', 'zc']:
        assert np.isclose(getattr(orb, name), getattr(orb_utc, name), rtol=0, atol=1e-8), name

    assert np.isclose(orb.ht_ref, orb_utc.ht_ref, rtol=0, atol=1e-3)
    assert np.isclose(orb.v_g, orb_utc.v_g, rtol=0, atol=1e-6)

    # The radiant and the sidereal time turn by the rotation of the Earth
    rotation = utcRotationAngle(DUT1)
    assert np.isclose(orb.ra - orb_utc.ra, rotation, rtol=0, atol=1e-12)
    assert np.isclose(orb.dec, orb_utc.dec, rtol=0, atol=1e-12)
    assert np.isclose(orb.lst_ref - orb_utc.lst_ref, rotation, rtol=0, atol=1e-9)

    # The geocentric radiant (J2000) turns by the same angle
    geocentric = angleArcsec(raDec2ECI(orb.ra_g, orb.dec_g), raDec2ECI(orb_utc.ra_g, orb_utc.dec_g))
    expected = 3600*np.degrees(rotation)*np.cos(orb.dec)
    assert abs(geocentric - expected) < 0.05*expected



def testFramesAreTurnedAroundThePole():
    """ Check that eciToInertial turns towards increasing R.A. by the rotation of the Earth in UT1 - UTC, and
        that inertialToECI undoes it, for one vector and for several.
    """

    vect = np.array(raDec2ECI(1.0, 0.3))
    ra, dec = eci2RaDec(eciToInertial(vect, DUT1))

    assert np.isclose(ra - 1.0, utcRotationAngle(DUT1), rtol=0, atol=1e-12)
    assert np.isclose(dec, 0.3, rtol=0, atol=1e-12)
    assert np.isclose(3600*np.degrees(utcRotationAngle(1.0)), 15.041, rtol=0, atol=1e-3)

    vects = np.random.default_rng(1).normal(size=(5, 3))
    assert np.allclose(inertialToECI(eciToInertial(vects, DUT1), DUT1), vects, rtol=0, atol=1e-12)
    assert np.array_equal(eciToInertial(vects, 0.0), vects)



def testUT1MinusUTCFromTheIERSTables(monkeypatch):
    """ Check that UT1 - UTC is read from the IERS tables across the leap second at the end of 2016, that it
        is 0 where the tables do not reach, and that the environment variable gives it instead.

    Arguments:
        monkeypatch: [object] Pytest fixture which sets the environment variable.
    """

    monkeypatch.delenv(DUT1_ENV, raising=False)

    before = deltaUT1(datetime2JD(datetime.datetime(2016, 12, 31, 12)))
    after = deltaUT1(datetime2JD(datetime.datetime(2017, 1, 1, 12)))

    assert -0.9 < before < 0 < after < 0.9
    assert abs(after - before - 1.0) < 0.01

    # Before the tables (1858)
    assert deltaUT1(2400000.5) == 0.0

    monkeypatch.setenv(DUT1_ENV, '0.25')
    assert deltaUT1(datetime2JD(datetime.datetime(2016, 12, 31, 12))) == 0.25



def testTheSolverReadsTheTablesByDefault(monkeypatch):
    """ Check that the solver takes UT1 - UTC from the tables unless it is given, or set to 0 by the
        environment.

    Arguments:
        monkeypatch: [object] Pytest fixture which sets the environment variable.
    """

    # 2021-02-28, 21:54 UTC
    jd = 2459274.4125

    monkeypatch.delenv(DUT1_ENV, raising=False)
    assert Trajectory(jd).dut1 == deltaUT1(jd) != 0
    assert Trajectory(jd, dut1=0.3).dut1 == 0.3

    monkeypatch.setenv(DUT1_ENV, '0')
    assert Trajectory(jd).dut1 == 0.0



if __name__ == "__main__":

    # Run the tests which need no pytest fixtures
    testDirectionsAgainstTheStarsGiveTheTrueLine(1)
    testDirectionsAgainstTheStarsGiveTheTrueLine(2)
    testWithoutTheCorrectionTheLineIsOff()
    testMonteCarloRunsKeepTheFrame()
    testOrbitIsComputedInTheInertialFrame()
    testFramesAreTurnedAroundThePole()

    print("All tests passed.")
