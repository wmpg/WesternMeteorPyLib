""" Tests for the Monte Carlo uncertainties when the timing offsets are not estimated.

The Monte Carlo runs filter their solutions by the timing residual, rank them by it, and add the nominal
solution to them. With estimate_timing_vel=False the nominal solution had no timing residual (None), so
the runs failed with a TypeError; and every run estimated the timing anyway.

The tests observe a straight constant-velocity trajectory from three stations, with noise, as
azimuth/elevation at each point's own time. The gravity correction is off, so they do not depend on it.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_MonteCarloWithoutTiming.py -v
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Utils.TrajConversions import altAz2RADec, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


JDT_REF = 2461070.9343128
DURATION = 2.5
N_POINTS = 40
V_INIT = 30000.0

# Noise on the azimuth and elevation, radians (about 20 arcsec)
NOISE = 1e-4

# (station_id, lat, lon, elevation) in degrees and metres
STATIONS = [
    ("A", -43.60, 172.12, 120.0),
    ("B", -42.90, 173.20, 200.0),
    ("C", -43.30, 171.30, 350.0),
]


def _trajectory(estimate_timing_vel, monte_carlo, seed=20261018):
    """ Trajectory object filled with noisy observations of a straight, constant-velocity meteor. """

    rng = np.random.default_rng(seed)

    lat, lon, ht = np.radians(-43.20), np.radians(172.30), 85000.0
    begin = np.array(geo2Cartesian(lat, lon, ht, JDT_REF))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(55.0), JDT_REF, lat, lon)
    radiant_eci = np.array(raDec2ECI(ra, dec))

    times = np.linspace(0, DURATION, N_POINTS)
    points = begin - np.outer(V_INIT*times, radiant_eci)

    traj = Trajectory(JDT_REF, meastype=2, max_toffset=1.0, estimate_timing_vel=estimate_timing_vel,
        gravity_correction=False, monte_carlo=monte_carlo, mc_runs=5, mc_cores=1, mc_noise_std=1.0,
        calc_orbit=True, show_plots=False, save_results=False, verbose=False)

    for station_id, s_lat, s_lon, ele in STATIONS:

        jds = JDT_REF + times/86400.0

        ra_list, dec_list = [], []
        for point, jd in zip(points, jds):
            station = np.array(geo2Cartesian(np.radians(s_lat), np.radians(s_lon), ele, jd))
            r, d = eci2RaDec(point - station)
            ra_list.append(r)
            dec_list.append(d)

        azim, elev = raDec2AltAz_vect(np.array(ra_list), np.array(dec_list), jds, np.radians(s_lat),
            np.radians(s_lon))

        azim = azim + rng.normal(0, NOISE, N_POINTS)/np.cos(elev)
        elev = elev + rng.normal(0, NOISE, N_POINTS)

        traj.infillTrajectory(azim, elev, times, np.radians(s_lat), np.radians(s_lon), ele,
            station_id=station_id)

    return traj, radiant_eci


def test_the_timing_residual_exists_without_timing_estimation():
    """ The residual of the clocks as given, which the Monte Carlo runs compare against. """

    traj, _ = _trajectory(estimate_timing_vel=False, monte_carlo=False)
    traj = traj.run()

    assert traj.timing_res is not None
    assert np.isfinite(traj.timing_res)


def test_monte_carlo_runs_without_timing_estimation():
    """ It raised a TypeError comparing the runs' timing residuals with None. """

    np.random.seed(1)
    traj, _ = _trajectory(estimate_timing_vel=False, monte_carlo=True)
    traj = traj.run()

    assert traj.uncertainties is not None
    assert np.isfinite(traj.uncertainties.ra_g) and (traj.uncertainties.ra_g > 0)


def test_monte_carlo_keeps_the_clocks_fixed():
    """ The runs estimate the timing only if the nominal solution does. """

    np.random.seed(1)
    traj, _ = _trajectory(estimate_timing_vel=False, monte_carlo=True)
    traj = traj.run()

    assert not traj.estimate_timing_vel
    assert np.all(np.array(traj.time_diffs_final) == 0)


if __name__ == "__main__":

    np.random.seed(1)
    traj, radiant_eci = _trajectory(estimate_timing_vel=False, monte_carlo=True)
    traj = traj.run()
    print("radiant error {:.2f} arcsec, timing offsets {}".format(
        3600*np.degrees(np.arccos(np.clip(np.dot(traj.radiant_eci_mini, radiant_eci), -1, 1))),
        traj.time_diffs_final))
