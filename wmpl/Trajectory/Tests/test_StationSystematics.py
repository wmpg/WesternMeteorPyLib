""" Tests of the systematic errors of the stations in wmpl.Trajectory.StationSystematics, measured from the
    residuals of a solved trajectory, and of the Monte Carlo runs in wmpl.Trajectory.Trajectory which include
    them. A straight constant-velocity meteor is observed with noise, and with a pointing error of a station.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_StationSystematics.py -v
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.StationSystematics import stationSystematicErrors, systematicErrorSize
from wmpl.Trajectory.Tests.SyntheticTrajectory import JDT_REF, stationAzimAlt
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Utils.TrajConversions import altAz2RADec, geo2Cartesian, raDec2ECI


# Duration (s), number of points per station and the velocity (m/s)
DURATION = 1.5
N_POINTS = 40
V_INIT = 20000.0

# Noise on the azimuth and elevation (arcsec)
NOISE = 10.0

# Station ID, latitude (deg), longitude (deg) and elevation (m)
STATIONS = [
    ('A', 45.6, 14.6, 200.0),
    ('B', 44.5, 15.3, 400.0),
    ('C', 45.1, 15.9, 150.0),
]



def _simulateMeteor(stations, elev_offsets=None, seed=1, **kwargs):
    """ Make a trajectory object filled with noisy observations of a straight, constant-velocity meteor.

    Arguments:
        stations: [list] Stations which observe it, as in STATIONS.

    Keyword arguments:
        elev_offsets: [dict] Pointing errors of the elevation of some stations, {station ID: offset
            (arcsec)}. None by default.
        seed: [int] Seed of the observation noise. 1 by default.
        kwargs: Passed to the Trajectory object.

    Return:
        traj: [Trajectory object] Trajectory object with the observations, not yet solved.
    """

    rng = np.random.default_rng(seed)

    # Compute the beginning point and the radiant
    lat, lon = np.radians(45.0), np.radians(15.0)
    begin = np.array(geo2Cartesian(lat, lon, 90000.0, JDT_REF))
    ra, dec = altAz2RADec(np.radians(120.0), np.radians(45.0), JDT_REF, lat, lon)
    radiant_eci = np.array(raDec2ECI(ra, dec))

    # Compute the points along the trajectory
    times = np.linspace(0, DURATION, N_POINTS)
    points = begin - np.outer(V_INIT*times, radiant_eci)
    jds = JDT_REF + times/86400.0

    traj_kwargs = dict(meastype=2, monte_carlo=False, gravity_correction=False, show_plots=False, \
        save_results=False, verbose=False)
    traj_kwargs.update(kwargs)
    traj = Trajectory(JDT_REF, **traj_kwargs)

    for station_id, s_lat, s_lon, ele in stations:

        # Compute the direction of every point as seen from the station
        azim, elev = stationAzimAlt(points, jds, s_lat, s_lon, ele)

        # Add the noise, and the pointing error of the station
        azim = azim + rng.normal(0, np.radians(NOISE/3600), N_POINTS)/np.cos(elev)
        elev = elev + rng.normal(0, np.radians(NOISE/3600), N_POINTS)
        if elev_offsets is not None:
            elev = elev + np.radians(elev_offsets.get(station_id, 0.0)/3600)

        traj.infillTrajectory(azim, elev, times, np.radians(s_lat), np.radians(s_lon), ele, \
            station_id=station_id)

    return traj



def _fragmentMeteors(traj):
    """ Make observations of a fragment from those of a solved trajectory, as given to the solver.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [list] Objects with the attributes of MeteorObservation objects that systematicErrorSize() uses.
    """

    return [SimpleNamespace(ra_data=obs.ra_data, dec_data=obs.dec_data, jdt_ref=traj.jdt_ref, \
        time_data=obs.time_data, latitude=obs.lat, longitude=obs.lon, height=obs.ele) \
        for obs in traj.observations]



def testThePointingErrorOfAStationIsMeasured():
    """ Check that the pointing error of a station shows in the size of the systematic errors, and that
        without it the size is that of the noise.
    """

    sizes = []
    for elev_offsets in (None, {'B': 60.0}):

        traj = _simulateMeteor(STATIONS, elev_offsets=elev_offsets).run()
        errors = stationSystematicErrors(traj)
        assert sorted(errors) == ['A', 'B', 'C']

        size = systematicErrorSize(traj, errors, _fragmentMeteors(traj))
        sizes.append(np.degrees(size['sigma_c'])*3600)

    # About 3 arcsec from the noise alone, and 17 arcsec with the pointing error
    assert sizes[0] < 6.0
    assert sizes[1] > 10.0



def testTwoStationsHaveNoSystematicErrors():
    """ Check that nothing is measured with two stations, also when one of them gave two files. """

    traj = _simulateMeteor(STATIONS[:2]).run()
    assert stationSystematicErrors(traj) == {}
    assert systematicErrorSize(traj, {}, _fragmentMeteors(traj)) is None

    traj = _simulateMeteor([STATIONS[0], ('A_2',) + STATIONS[0][1:], STATIONS[1]]).run()
    assert stationSystematicErrors(traj) == {}



def testMonteCarloIncludesTheSystematicErrors():
    """ Check that in the Monte Carlo runs a constant systematic error of the stations moves the trajectory
        and a slope turns it, and that the original solution is kept.
    """

    nominal = _simulateMeteor(STATIONS[:2]).run()

    cases = [
        ('none', None),
        ('constant', {'sigma_c': np.radians(30/3600), 'sigma_g': 0.0}),
        ('slope', {'sigma_c': 0.0, 'sigma_g': np.radians(2/3600)/1000}),
    ]

    spreads = {}
    for name, mc_systematics in cases:

        np.random.seed(2)
        traj = _simulateMeteor(STATIONS[:2], monte_carlo=True, mc_runs=10, mc_cores=1)
        if mc_systematics is not None:
            traj.mc_systematics = mc_systematics

        traj = traj.run()

        # Spread of the position (m) and of the radiant (arcsec)
        spreads[name] = (np.linalg.norm(traj.uncertainties.state_vect_mini), \
            np.degrees(np.linalg.norm(traj.uncertainties.radiant_eci_mini))*3600)

        if mc_systematics is not None:
            assert np.allclose(traj.radiant_eci_mini, nominal.radiant_eci_mini, rtol=0, atol=1e-12)

    # About 4 m and 19 arcsec without them, 24 m with the constant and 250 arcsec with the slope
    assert spreads['constant'][0] > 3*spreads['none'][0]
    assert spreads['slope'][1] > 3*spreads['none'][1]
