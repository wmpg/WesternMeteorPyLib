""" Tests for the ground-fixed radiant used by DynamicMassFit and SampleTrajectoryPositions, and for the
DynamicMassFit velocity fit.

The trajectory solver works in the ECI frame of the true equator and equinox of date, and the apparent
ground-fixed radiant it reports (orbit.azimuth_apparent_norot, orbit.elevation_apparent_norot) is
computed without precession. SampleTrajectoryPositions used to precess the derotated radiant from J2000 to
the date before converting it to horizontal coordinates, which applied precession twice and biased its
azimuth and elevation by 0.1-0.5 deg on real trajectories. These tests pin the shared construction against
the solver's own value and, when astropy is installed, against an independent TETE -> ITRS transformation.

Run under pytest, or directly:

    python -m wmpl.Utils.Tests.test_DynamicMassFit
"""

import io
import os
import contextlib

import numpy as np
import pytest
import scipy.optimize

from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import derotatedRadiantAltAz, cartesian2Geo, jd2LST, latLonAlt2ECEF, ecef2ENU
from wmpl.Utils.GeoidHeightEGM96 import mslToWGS84Height
from wmpl.Utils.SampleTrajectoryPositions import sampleTrajectory
from wmpl.Utils.DynamicMassFit import pointOnTrajectory, _robust_linear_fit, fitVelocity
from wmpl.Utils.Math import lineFunc
from wmpl.Utils.DynamicMassFit import groundSpeed, computeFragEndParams, interpolateHtVsTimeLen


# A solved trajectory shipped with the repository (2019-10-23, four stations, gravity correction on)
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"


@pytest.fixture(scope="module")
def traj():
    """ Load the example trajectory once. """

    return loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)


### The ground-fixed radiant ###

def testDerotatedRadiantMatchesTheSolverAtTheReferencePoint(traj):
    """ At the reference point, with the initial velocity, the helper reproduces the solver's own apparent
        ground-fixed radiant, which is computed by the same construction in Orbit.calcOrbit().
    """

    lat, lon, _ = cartesian2Geo(traj.jdt_ref, *traj.state_vect_mini)
    azim, elev, v_norot = derotatedRadiantAltAz(traj.v_init*traj.radiant_eci_mini, traj.state_vect_mini, \
        traj.jdt_ref, lat, lon)

    assert np.degrees(azim) == pytest.approx(np.degrees(traj.orbit.azimuth_apparent_norot), abs=1e-6)
    assert np.degrees(elev) == pytest.approx(np.degrees(traj.orbit.elevation_apparent_norot), abs=1e-6)
    assert v_norot == pytest.approx(traj.orbit.v_init_norot, rel=1e-9)


def _referenceGroundFixedRadiant(lat, lon, ht, v_eci, jd):
    """ Independent reference with astropy: the ECI (true of date) velocity rotated to ITRS, minus the
        rotation velocity omega x r, projected on the geodetic ENU basis. No RA/Dec, no derotation formula.
    """

    from astropy.time import Time
    from astropy.coordinates import TETE, ITRS, CartesianRepresentation, EarthLocation
    import astropy.units as u

    t = Time(jd, format="jd", scale="utc")
    loc = EarthLocation.from_geodetic(np.degrees(lon)*u.deg, np.degrees(lat)*u.deg, ht*u.m)
    r_ef = np.array([loc.x.to_value(u.m), loc.y.to_value(u.m), loc.z.to_value(u.m)])

    def toItrs(x):
        return TETE(CartesianRepresentation(x*u.m), obstime=t).transform_to(ITRS(obstime=t)).cartesian.xyz.to_value(u.m)

    r_eci = ITRS(CartesianRepresentation(r_ef*u.m), obstime=t).transform_to(TETE(obstime=t)).cartesian.xyz.to_value(u.m)
    v_ef = toItrs(r_eci + v_eci) - r_ef - np.cross([0.0, 0.0, 7.2921150e-5], r_ef)

    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

    radiant = -v_ef/np.linalg.norm(v_ef)

    return np.degrees(np.arctan2(radiant @ east, radiant @ north)) % 360, np.degrees(np.arcsin(radiant @ up))


def testDerotatedRadiantMatchesAnIndependentFrameTransformation(traj):
    """ Along the whole observed path the helper agrees with an astropy TETE -> ITRS reference to better than
        the double-precession error it replaces (0.1-0.5 deg) by two orders of magnitude.
    """

    pytest.importorskip("astropy")

    for length in np.linspace(0.0, 30000.0, 4):
        t = length/traj.v_avg
        eci = pointOnTrajectory(traj, length, t)
        jd = traj.jdt_ref + t/86400
        lat, lon, ht = cartesian2Geo(jd, *eci)

        azim, elev, _ = derotatedRadiantAltAz(traj.v_init*traj.radiant_eci_mini, eci, jd, lat, lon)
        azim_ref, elev_ref = _referenceGroundFixedRadiant(lat, lon, ht, -traj.v_init*traj.radiant_eci_mini, jd)

        assert (np.degrees(azim) - azim_ref + 180) % 360 - 180 == pytest.approx(0.0, abs=2e-3)
        assert np.degrees(elev) == pytest.approx(elev_ref, abs=1e-3)


def testSampledTrajectoryRadiantIsNotPrecessedTwice(traj):
    """ The sampled azimuth and elevation stay within the chord-versus-radiant difference of the solver's
        values along the path. With the removed J2000 -> date precession they were off by ~0.5 deg here.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        samples = sampleTrajectory(traj, traj.rbeg_ele, traj.rend_ele, (traj.rbeg_ele - traj.rend_ele)/5, \
            show_plots=False)

    azims = np.degrees(np.array(samples.azim_norot, dtype=float))
    elevs = np.degrees(np.array(samples.elev_norot, dtype=float))
    valid = np.isfinite(azims)

    assert valid.sum() >= 4

    # The local horizon rotates by ~0.009 deg/km of ground track, which moves the elevation by ~0.1 deg
    # over this 20 km path and the azimuth by far less (the path runs nearly along a meridian). The
    # removed precession step put the azimuth 0.54 deg off here, so the azimuth carries the regression
    assert np.max(np.abs(azims[valid] - np.degrees(traj.orbit.azimuth_apparent_norot))) < 0.05
    assert np.max(np.abs(elevs[valid] - np.degrees(traj.orbit.elevation_apparent_norot))) < 0.15


### Points on the fitted trajectory ###

def testPointOnTrajectoryStartsAtTheStateVectorAndDropsPerpendicularly(traj):
    """ At zero length and the first observation time the point is the state vector; later, the gravity
        drop is applied only perpendicular to the fitted line and points downwards.
    """

    t0 = min(obs.time_data[0] for obs in traj.observations)

    assert pointOnTrajectory(traj, 0.0, t0) == pytest.approx(traj.state_vect_mini, abs=1e-6)

    length = 20000.0
    t = t0 + length/traj.v_avg
    on_line = traj.state_vect_mini - length*traj.radiant_eci_mini
    drop = pointOnTrajectory(traj, length, t) - on_line

    assert abs(np.dot(drop, traj.radiant_eci_mini)) < 1e-6
    assert 0.1 < np.linalg.norm(drop) < 10.0, "a ~0.3 s flight drops by a fraction of a metre to metres"
    assert np.dot(drop, on_line) < 0, "the drop points towards the Earth"


### Speeds relative to the ground ###

def testGroundSpeedRemovesTheEarthRotationFromTheSolverSpeeds(traj):
    """ The solver's point speeds are measured in the ECI frame. The speeds of its model points in a frame fixed
        to the ground, by finite differences after rotating them with the Earth, are the ground speeds. """

    for obs in traj.observations:
        t = obs.time_data
        eci = np.asarray(obs.model_eci)
        gst = np.radians([jd2LST(traj.jdt_ref + ti/86400, 0)[0] for ti in t])
        ecef = np.column_stack([np.cos(gst)*eci[:, 0] + np.sin(gst)*eci[:, 1], \
            -np.sin(gst)*eci[:, 0] + np.cos(gst)*eci[:, 1], eci[:, 2]])

        v_eci = np.linalg.norm(np.diff(eci, axis=0), axis=1)/np.diff(t)
        v_ground = np.linalg.norm(np.diff(ecef, axis=0), axis=1)/np.diff(t)

        assert groundSpeed(traj, eci[1:], v_eci) == pytest.approx(v_ground, rel=1e-4)
        assert np.max(np.abs(v_eci - v_ground)) > 40, "the rotation is a sizeable part of the speed"


def _ecef(lat, lon, ele_msl):
    """ ECEF position (m) of a point given by geodetic latitude and longitude (radians) and height above the
        sea level (m), independent of the ECI frame. """

    return np.array(latLonAlt2ECEF(lat, lon, mslToWGS84Height(lat, lon, ele_msl)))


def testEndOfAblationMovesWithTheGround(traj):
    """ The simulated length is relative to the ground, so seen from the ground the final point is that length
        away from the evaluation point along the ground-fixed direction of motion, with no sideways shift. The
        gravity drop is vertical and across the track only. Placing the length along the ECI line instead
        misses the rotation of the ground over the simulated time, here over a kilometre. """

    height = 100000.0
    with contextlib.redirect_stdout(io.StringIO()):
        sr, _, lat, lon, ele, _, _ = computeFragEndParams(traj, 1.0, 3500, height, 20000.0, 0.55)

    # The evaluation point, as computeFragEndParams() finds it
    ht_vs_time_interp, _ = interpolateHtVsTimeLen(traj, sample_step=0.1, show_plots=False)
    meas_time = ht_vs_time_interp(height)
    meas_len = scipy.optimize.brentq(lambda l: cartesian2Geo(traj.jdt_ref + meas_time/86400, \
        *pointOnTrajectory(traj, l, meas_time))[2] - height, 0, np.dot(traj.state_vect_mini, traj.radiant_eci_mini))
    eval_eci = pointOnTrajectory(traj, meas_len, meas_time)
    eval_jd = traj.jdt_ref + meas_time/86400
    eval_lat, eval_lon, _ = cartesian2Geo(eval_jd, *eval_eci)

    # The ground-fixed direction of motion at the evaluation point
    azim, elev, _ = derotatedRadiantAltAz(traj.v_init*traj.radiant_eci_mini, eval_eci, eval_jd, eval_lat, eval_lon)
    motion = -np.array([np.sin(azim)*np.cos(elev), np.cos(azim)*np.cos(elev), np.sin(elev)])
    across = np.array([np.cos(azim), -np.sin(azim), 0.0])

    displacement = np.array(ecef2ENU(eval_lat, eval_lon, *(_ecef(np.radians(lat), np.radians(lon), 1000*ele) \
        - _ecef(eval_lat, eval_lon, height))))

    assert np.dot(displacement, motion) == pytest.approx(sr.frag_main.length, abs=2.0)
    assert np.dot(displacement, across) == pytest.approx(0.0, abs=2.0)


### The velocity fit ###

def testRobustFitCovarianceIsTheLinearModelCovariance():
    """ On clean data the covariance equals sigma^2 (J^T J)^-1 of the linear model, with sigma from the
        MAD of the residuals; a fit of points that all share one time is refused with a clear error.
    """

    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1.0, 40)
    v = 20000.0 - 4000.0*t + rng.normal(0.0, 30.0, t.size)

    popt, pcov, perr = _robust_linear_fit(t, v, p0=(1.0, 1.0), loss='soft_l1', f_scale=30.0)

    resid = v - lineFunc(t, *popt)
    sigma = 1.4826*np.median(np.abs(resid - np.median(resid)))
    J = np.column_stack((t, np.ones_like(t)))

    assert popt[0] == pytest.approx(-4000.0, abs=200.0)
    assert pcov == pytest.approx(sigma**2*np.linalg.inv(J.T @ J), rel=1e-9)
    assert perr == pytest.approx(np.sqrt(np.diag(pcov)), rel=1e-12)

    with pytest.raises(RuntimeError, match="same time"):
        _robust_linear_fit(np.full(5, 0.3), v[:5], p0=(1.0, 1.0), loss='soft_l1')


def testFitVelocityRejectsOutliersAtTheRequestedSigma():
    """ A gross outlier is excluded from the final fit and the slope is recovered. """

    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 1.0, 30)
    v = 20000.0 - 4000.0*t + rng.normal(0.0, 30.0, t.size)
    v[12] += 3000.0

    with contextlib.redirect_stdout(io.StringIO()):
        popt, pcov, perr, mask = fitVelocity(t, v, p0=(1.0, 1.0), loss='soft_l1', sigma_clip=3.0)

    assert not mask[12]
    assert mask.sum() == t.size - 1
    assert popt[0] == pytest.approx(-4000.0, abs=150.0)


if __name__ == "__main__":

    import sys
    sys.exit(pytest.main([__file__, "-q"]))
