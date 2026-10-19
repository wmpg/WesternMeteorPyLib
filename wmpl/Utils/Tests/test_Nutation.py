""" Tests for the conversion of J2000 coordinates to the true equator and equinox of date.

WMPL's ECI frame is the true equator and equinox of date: geo2Cartesian and raDec2AltAz use the apparent
sidereal time. J2000 R.A./Dec. used to be brought there with precession alone, which only reaches the mean
equator and equinox, up to about 17 arcsec away. The reference here is ERFA's IAU 1976/1980
precession-nutation matrix (pnm80), which ships with astropy.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Nutation.py -v
"""

import numpy as np
import erfa

import matplotlib
matplotlib.use("Agg")

from wmpl.Utils.Earth import calcApparentSiderealEarthRotation
from wmpl.Utils.TrajConversions import J2000_JD, nutationMatrix, equatorialCoordJ2000ToDate, \
    equatorialCoordDateToJ2000, jd2DynamicalTimeJD, raDec2ECI, eci2RaDec, altAz2RADec, geo2Cartesian, \
    raDec2AltAz_vect


JD = 2461070.9343128


def _angle(ra1, dec1, ra2, dec2):
    """ Angle between two directions, in arcseconds. """

    return 3600*np.degrees(np.arccos(np.clip(np.dot(raDec2ECI(ra1, dec1), raDec2ECI(ra2, dec2)), -1, 1)))


def test_the_nutation_matches_the_apparent_sidereal_time():
    """ The equinox moves in R.A. by the equation of the equinoxes, the apparent minus the mean sidereal time.

    That is what makes the frame consistent: the stations are placed with the apparent sidereal time.
    """

    t = (JD - J2000_JD.days)/36525.0
    gmst = np.radians((280.46061837 + 360.98564736629*(JD - J2000_JD.days) + 0.000387933*t**2 \
        - t**3/38710000)%360)
    equation_of_equinoxes = calcApparentSiderealEarthRotation(JD) - gmst

    equinox = nutationMatrix(JD).dot(np.array([1.0, 0.0, 0.0]))

    assert abs(np.arctan2(equinox[1], equinox[0]) - equation_of_equinoxes) < np.radians(1e-6/3600)


def test_j2000_to_date_agrees_with_the_iau():
    """ Within 0.16 arcsec over three dates; precession alone was up to 15.8 arcsec away. """

    rng = np.random.default_rng(3)

    for jd in (2455000.5, JD, 2465000.5):

        npb = erfa.pnm80(2400000.5, jd2DynamicalTimeJD(jd) - 2400000.5)

        for _ in range(100):
            ra, dec = rng.uniform(0, 2*np.pi), np.arcsin(rng.uniform(-1, 1))
            ra_ref, dec_ref = eci2RaDec(npb.dot(np.array(raDec2ECI(ra, dec))))

            assert _angle(*equatorialCoordJ2000ToDate(jd, ra, dec), ra_ref, dec_ref) < 0.2


def test_the_conversion_inverts():
    """ Limited by equatorialCoordPrecession, whose own round trip is 0.004 arcsec. """

    rng = np.random.default_rng(4)

    for _ in range(100):
        ra, dec = rng.uniform(0, 2*np.pi), np.arcsin(rng.uniform(-1, 1))
        ra_back, dec_back = equatorialCoordDateToJ2000(JD, *equatorialCoordJ2000ToDate(JD, ra, dec))

        assert _angle(ra, dec, ra_back, dec_back) < 0.01


def test_a_trajectory_from_j2000_lines_of_sight_inverts():
    """ Noise-free straight trajectory, its lines of sight converted to J2000 with ERFA, through
        prepareObservations. Before: 9.80 arcsec of radiant error and 5.44 m of residual.
    """

    from wmpl.Formats.GenericFunctions import MeteorObservation, prepareObservations
    from wmpl.Trajectory.Trajectory import Trajectory

    stations = [("A", -43.60, 172.12, 120.0), ("B", -42.90, 173.20, 200.0), ("C", -43.30, 171.30, 350.0)]

    lat, lon = np.radians(-43.20), np.radians(172.30)
    begin = np.array(geo2Cartesian(lat, lon, 85000.0, JD))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(55.0), JD, lat, lon)
    radiant_eci = np.array(raDec2ECI(ra, dec))
    times = np.linspace(0, 2.5, 40)
    points = begin - np.outer(30000.0*times, radiant_eci)

    meteors = []
    for station_id, s_lat, s_lon, ele in stations:

        meteor = MeteorObservation(JD, station_id, np.radians(s_lat), np.radians(s_lon), ele, 1.0)

        for point, t in zip(points, times):
            jd = JD + t/86400.0
            los = point - np.array(geo2Cartesian(np.radians(s_lat), np.radians(s_lon), ele, jd))
            npb = erfa.pnm80(2400000.5, jd2DynamicalTimeJD(jd) - 2400000.5)
            ra_j2000, dec_j2000 = eci2RaDec(npb.T.dot(los/np.linalg.norm(los)))
            meteor.addPoint(t, 0, 0, 0, 0, np.degrees(ra_j2000), np.degrees(dec_j2000), 0)

        meteor.finish()
        meteors.append(meteor)

    jdt_ref, meteors = prepareObservations(meteors)

    # Azimuth/elevation at each point's own time, so the epoch convention of meastype 1 does not enter
    traj = Trajectory(jdt_ref, meastype=2, estimate_timing_vel=False, gravity_correction=False, monte_carlo=False,
        calc_orbit=False, show_plots=False, save_results=False, verbose=False)
    for meteor in meteors:
        azim, elev = raDec2AltAz_vect(meteor.ra_data, meteor.dec_data, meteor.jdt_ref + meteor.time_data/86400.0,
            meteor.latitude, meteor.longitude)
        traj.infillTrajectory(azim, elev, meteor.time_data, meteor.latitude, meteor.longitude, meteor.height,
            station_id=meteor.station_id)
    traj = traj.run()

    radiant_error = 3600*np.degrees(np.arccos(np.clip(np.dot(traj.radiant_eci_mini, radiant_eci), -1, 1)))
    worst = max(np.max(np.hypot(obs.h_residuals, obs.v_residuals)) for obs in traj.observations)

    assert radiant_error < 0.2
    assert worst < 0.5
