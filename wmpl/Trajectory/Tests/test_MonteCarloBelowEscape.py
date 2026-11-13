""" Tests of the Monte Carlo uncertainties in wmpl.Trajectory.Trajectory for a meteor slower than the escape
    velocity, which has no geocentric orbit, such as a fragment seen after the main body slowed down. A
    straight constant-velocity meteor is observed from two stations, with noise.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_MonteCarloBelowEscape.py -v
"""

from __future__ import print_function, division, absolute_import

import sys
from types import SimpleNamespace

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory, calcMCUncertainties, checkMCTrajectories, \
    monteCarloTrajectory
from wmpl.Utils.TrajConversions import geo2Cartesian


# The module itself, to replace its functions in the tests (importing wmpl sets an attribute of the same
#   name on the package)
TRAJECTORY = sys.modules['wmpl.Trajectory.Trajectory']

# Reference Julian date of the meteor
JDT_REF = 2461000.5

# Duration (s) and number of points per station
DURATION = 1.0
N_POINTS = 25

# Noise on the right ascension and declination (arcsec)
NOISE = 20.0

# Station ID, latitude (deg), longitude (deg) and elevation (m)
STATIONS = [
    ('A', 44.80, 9.70, 100.0),
    ('B', 45.30, 10.40, 200.0),
]



def _simulateStraightMeteor(v_init, monte_carlo=True, seed=1):
    """ Make a trajectory object filled with noisy observations of a straight constant-velocity meteor which
        starts at 40 km.

    Arguments:
        v_init: [float] Velocity of the meteor (m/s).

    Keyword arguments:
        monte_carlo: [bool] Run 6 Monte Carlo runs on one core. True by default.
        seed: [int] Seed of the observation noise. 1 by default.

    Return:
        traj: [Trajectory object] Trajectory object with the observations, not yet solved.
    """

    rng = np.random.default_rng(seed)

    # Compute the beginning point and the direction of motion, 37 deg below the horizontal towards the east
    begin = np.array(geo2Cartesian(np.radians(45.0), np.radians(10.0), 40000.0, JDT_REF))
    up = begin/np.linalg.norm(begin)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    direction = -0.6*up + 0.8*east

    times = np.linspace(0, DURATION, N_POINTS)

    traj = Trajectory(JDT_REF, meastype=1, monte_carlo=monte_carlo, mc_runs=6, mc_cores=1, \
        geometric_uncert=True, show_plots=False, save_results=False, verbose=False)

    for station_id, lat, lon, ele in STATIONS:

        # Compute the direction of every point as seen from the station
        vectors = []
        for t in times:
            station = np.array(geo2Cartesian(np.radians(lat), np.radians(lon), ele, JDT_REF + t/86400.0))
            vectors.append(begin + v_init*t*direction - station)

        vectors = np.array(vectors)
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        ra = np.arctan2(vectors[:, 1], vectors[:, 0])%(2*np.pi)
        dec = np.arcsin(vectors[:, 2])

        # Add noise to the measurements
        ra = ra + rng.normal(0, np.radians(NOISE/3600), N_POINTS)/np.cos(dec)
        dec = dec + rng.normal(0, np.radians(NOISE/3600), N_POINTS)

        traj.infillTrajectory(ra, dec, times, np.radians(lat), np.radians(lon), ele, station_id=station_id)

    return traj



def testUncertaintiesBelowEscapeVelocity(monkeypatch):
    """ Check that a meteor slower than the escape velocity gets the uncertainties of its trajectory, from
        the runs and the original solution, but none of a geocentric orbit.
    """

    traj = _simulateStraightMeteor(8000.0)

    # Record the trajectories the uncertainties are computed from
    traj_list_used = []

    def _calcMCUncertainties(traj_list, traj_best):

        traj_list_used.extend(traj_list)

        return calcMCUncertainties(traj_list, traj_best)

    monkeypatch.setattr(TRAJECTORY, 'calcMCUncertainties', _calcMCUncertainties)

    traj_best = traj.run()

    assert any(traj_used is traj for traj_used in traj_list_used)

    assert traj_best.orbit.ra_g is None
    assert traj_best.uncertainties is not None
    assert traj_best.uncertainties.v_init > 0
    assert np.all(traj_best.uncertainties.radiant_eci_mini > 0)

    assert traj_best.uncertainties.ra_g is None
    assert traj_best.uncertainties.v_g is None



def testUncertaintiesAboveEscapeVelocity():
    """ Check that a meteor faster than the escape velocity keeps the uncertainties of its orbit. """

    traj = _simulateStraightMeteor(25000.0).run()

    assert traj.orbit.ra_g is not None
    assert traj.uncertainties is not None
    assert traj.uncertainties.ra_g > 0
    assert traj.uncertainties.v_g > 0



def testPlotsBelowEscapeVelocity():
    """ Check that the plots of the Monte Carlo runs skip the orbit of a meteor slower than the escape
        velocity.
    """

    traj = _simulateStraightMeteor(8000.0, monte_carlo=False).run()

    _, uncertainties = monteCarloTrajectory(traj, mc_runs=3, geometric_uncert=True, plot_results=True, \
        mc_cores=1)

    assert uncertainties is not None



def testRunsWithoutOrbitRejectedOnlyIfRequired():
    """ Check that the Monte Carlo runs without an orbit are rejected unless no orbit is expected. """

    # Make one run with and one without a geocentric radiant
    runs = [SimpleNamespace(timing_res=0.0, los_mini_status=True, \
        orbit=SimpleNamespace(ra_g=ra_g, dec_g=ra_g)) for ra_g in (None, 1.0)]

    assert len(checkMCTrajectories(runs)) == 1
    assert len(checkMCTrajectories(runs, require_orbit=False)) == 2




if __name__ == "__main__":

    # Solve the slow meteor and print the uncertainties of its velocity and radiant
    traj = _simulateStraightMeteor(8000.0).run()

    print("v_init = {:.1f} +/- {:.1f} m/s, radiant ECI sigma {}".format(traj.v_init, \
        traj.uncertainties.v_init, traj.uncertainties.radiant_eci_mini))
