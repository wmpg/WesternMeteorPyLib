""" Tests for the conversion of J2000 coordinates to the true equator and equinox of date.

WMPL's ECI frame is the true equator and equinox of date, as geo2Cartesian and raDec2AltAz use the apparent
sidereal time. J2000 R.A./Dec. used to be brought there with precession alone, which only reaches the mean
equator and equinox, up to about 17 arcsec away. The reference here is ERFA's IAU 1976/1980
precession-nutation matrix (pnm80), which ships with astropy.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Nutation.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import erfa

import matplotlib
matplotlib.use('Agg')

from wmpl.Formats.GenericFunctions import MeteorObservation, prepareObservations
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    straightMeteorPoints
from wmpl.Utils.Earth import calcApparentSiderealEarthRotation
from wmpl.Utils.Math import angleBetweenVectors
from wmpl.Utils.TrajConversions import J2000_JD, nutationMatrix, equatorialCoordJ2000ToDate, \
    equatorialCoordDateToJ2000, jd2DynamicalTimeJD, raDec2ECI, eci2RaDec, geo2Cartesian, raDec2AltAz_vect


# Julian date of the tests
JD = STRAIGHT_JDT_REF



def _angle(ra1, dec1, ra2, dec2):
    """ Compute the angle between two directions.

    Arguments:
        ra1: [float] Right ascension of the first direction (radians).
        dec1: [float] Declination of the first direction (radians).
        ra2: [float] Right ascension of the second direction (radians).
        dec2: [float] Declination of the second direction (radians).

    Return:
        [float] Angle between the directions (arcsec).
    """

    return 3600*np.degrees(angleBetweenVectors(np.array(raDec2ECI(ra1, dec1)), \
        np.array(raDec2ECI(ra2, dec2))))



def _iauJ2000ToDate(jd):
    """ Compute ERFA's IAU 1976/1980 precession-nutation matrix from J2000 to the true equator and equinox
        of date.

    Arguments:
        jd: [float] Julian date.

    Return:
        [3x3 ndarray] Rotation matrix, applied to column vectors.
    """

    return erfa.pnm80(2400000.5, jd2DynamicalTimeJD(jd) - 2400000.5)



def testTheNutationMatchesTheApparentSiderealTime():
    """ Check that the nutation moves the equinox in R.A. by the equation of the equinoxes, the apparent
        minus the mean sidereal time, as the stations are placed with the apparent sidereal time.
    """

    # Compute the mean sidereal time
    t = (JD - J2000_JD.days)/36525.0
    gmst = np.radians((280.46061837 + 360.98564736629*(JD - J2000_JD.days) + 0.000387933*t**2 \
        - t**3/38710000)%360)

    equation_of_equinoxes = calcApparentSiderealEarthRotation(JD) - gmst

    equinox = nutationMatrix(JD).dot(np.array([1.0, 0.0, 0.0]))

    assert abs(np.arctan2(equinox[1], equinox[0]) - equation_of_equinoxes) < np.radians(1e-6/3600)



def testJ2000ToDateAgreesWithTheIAU():
    """ Check that J2000 to date is within 0.2 arcsec of the IAU over three dates. Precession alone was up
        to 15.8 arcsec away.
    """

    rng = np.random.default_rng(3)

    for jd in (2455000.5, JD, 2465000.5):

        npb = _iauJ2000ToDate(jd)

        for _ in range(100):

            ra, dec = rng.uniform(0, 2*np.pi), np.arcsin(rng.uniform(-1, 1))
            ra_ref, dec_ref = eci2RaDec(npb.dot(np.array(raDec2ECI(ra, dec))))

            assert _angle(*equatorialCoordJ2000ToDate(jd, ra, dec), ra_ref, dec_ref) < 0.2



def testTheConversionInverts():
    """ Check that the round trip J2000 to date and back is within 0.01 arcsec, limited by
        equatorialCoordPrecession, whose own round trip is 0.004 arcsec.
    """

    rng = np.random.default_rng(4)

    for _ in range(100):

        ra, dec = rng.uniform(0, 2*np.pi), np.arcsin(rng.uniform(-1, 1))
        ra_back, dec_back = equatorialCoordDateToJ2000(JD, *equatorialCoordJ2000ToDate(JD, ra, dec))

        assert _angle(ra, dec, ra_back, dec_back) < 0.01



def testATrajectoryFromJ2000LinesOfSightInverts():
    """ Check that a noise-free straight meteor is recovered from its lines of sight converted to J2000 with
        ERFA and read through prepareObservations. With precession alone, the radiant was 9.80 arcsec off
        and the worst residual was 5.44 m.
    """

    times, points, _, radiant_eci = straightMeteorPoints()

    # Convert the lines of sight to J2000 with ERFA
    meteors = []
    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        meteor = MeteorObservation(JD, station_id, np.radians(s_lat), np.radians(s_lon), ele, 1.0)

        for point, t in zip(points, times):

            jd = JD + t/86400.0
            los = point - np.array(geo2Cartesian(np.radians(s_lat), np.radians(s_lon), ele, jd))
            ra_j2000, dec_j2000 = eci2RaDec(_iauJ2000ToDate(jd).T.dot(los/np.linalg.norm(los)))

            meteor.addPoint(t, 0, 0, 0, 0, np.degrees(ra_j2000), np.degrees(dec_j2000), 0)

        meteor.finish()
        meteors.append(meteor)

    jdt_ref, meteors = prepareObservations(meteors)

    traj = Trajectory(jdt_ref, meastype=2, estimate_timing_vel=False, gravity_correction=False, \
        monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, verbose=False)

    # Give the azimuth and elevation at the time of each point, so the epoch convention of meastype 1 does
    #   not enter
    for meteor in meteors:

        azim, elev = raDec2AltAz_vect(meteor.ra_data, meteor.dec_data, \
            meteor.jdt_ref + meteor.time_data/86400.0, meteor.latitude, meteor.longitude)

        traj.infillTrajectory(azim, elev, meteor.time_data, meteor.latitude, meteor.longitude, \
            meteor.height, station_id=meteor.station_id)

    traj = traj.run()

    radiant_error = 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci))
    worst_residual = max(np.max(np.hypot(obs.h_residuals, obs.v_residuals)) for obs in traj.observations)

    assert radiant_error < 0.2
    assert worst_residual < 0.5
