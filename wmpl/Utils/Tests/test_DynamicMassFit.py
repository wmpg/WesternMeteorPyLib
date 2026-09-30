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

from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import derotatedRadiantAltAz, cartesian2Geo
from wmpl.Utils.SampleTrajectoryPositions import sampleTrajectory
from wmpl.Utils.DynamicMassFit import pointOnTrajectory, _robust_linear_fit, fitVelocity, runFragSim, \
    SIM_HT_MIN, computeFragEndParams, _airSpeed, _motionENU
from wmpl.Utils.AtmosphereProfile import AtmosphereProfile
from wmpl.Utils.Physics import dynamicMass
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Math import lineFunc, vectMag


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


### The fragment simulation ###

def testFragmentSimulationFitsTheAtmosphereAtTheGivenLocation(traj):
    """ runFragSim() takes the location in degrees, as computeFragEndParams() passes it, while fitAtmPoly()
        takes it in radians. The atmosphere of the simulation must be the one at the given location.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        sr = runFragSim(0.1, 3500, np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), traj.jdt_ref, 30000, \
            5000, 45, 0.55)

    dens_co = fitAtmPoly(traj.rend_lat, traj.rend_lon, sr.const.h_kill, 180000, traj.jdt_ref)

    assert sr.const.dens_co == pytest.approx(dens_co, rel=1e-12)


def testFragmentSimulationUsesTheAtmosphereProfile(traj, tmp_path):
    """ With an atmosphere profile, the simulation's density polynomial is the profile's own fit over the
        heights the simulation descends through.
    """

    heights = np.arange(0.0, 60100.0, 100.0)
    path = str(tmp_path/"profile.csv")
    with open(path, 'w') as f:
        f.write("height,temperature,pressure,relative_humidity,wind_horizontal,wind_direction,wind_east,"
            "wind_north,wind_up,density\n")
        for ht in heights:
            f.write("{:.1f},250,1000,1,10,270,10,0,0,{:.10e}\n".format(ht, 1.3*np.exp(-ht/6900.0)))

    prof = AtmosphereProfile(path)

    with contextlib.redirect_stdout(io.StringIO()):
        sr = runFragSim(0.1, 3500, np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), traj.jdt_ref, 30000, \
            5000, 45, 0.55, atm_profile=prof)

    assert sr.const.dens_co == pytest.approx(prof.fitPoly(SIM_HT_MIN, 30000)[0], rel=1e-12)


### The dynamic mass ###

def testDynamicMassUsesTheGivenAirDensity():
    """ A given air density replaces the atmosphere model, and the mass goes as its cube. """

    args = (3500.0, 0.9, -0.03, 30000.0, 2459274.4, 6300.0, 3200.0)

    mass = dynamicMass(*args, gamma=1.0, shape_factor=0.55, atm_dens=0.02)

    assert mass == pytest.approx((0.55*6300.0**2*0.02/3200.0)**3/3500.0**2, rel=1e-12)
    assert dynamicMass(*args, gamma=1.0, shape_factor=0.55, atm_dens=0.04) == pytest.approx(8*mass, rel=1e-12)


### Winds ###

def _windProfile(path, speed, direction):
    """ An exponential atmosphere with a constant wind of the given speed (m/s), blowing from the given
        direction (deg). """

    with open(path, 'w') as f:
        f.write("height,temperature,pressure,relative_humidity,wind_horizontal,wind_direction,wind_east,"
            "wind_north,wind_up,density\n")
        for ht in np.arange(0.0, 120100.0, 100.0):
            f.write("{:.1f},250,1000,1,{:.3f},{:.3f},0,0,0,{:.10e}\n".format(ht, speed, direction, \
                1.3*np.exp(-ht/7000.0)))

    return AtmosphereProfile(path)


def _endState(traj, prof):

    with contextlib.redirect_stdout(io.StringIO()):
        return computeFragEndParams(traj, 1000.0, 3500, 100000.0, 20000.0, 0.55, atm_profile=prof)


def testDynamicMassUsesTheSpeedRelativeToTheAir(traj, tmp_path):
    """ A headwind adds its component along the motion to the speed the drag acts on. """

    elev = traj.orbit.elevation_apparent_norot
    headwind = _windProfile(str(tmp_path/"head.csv"), 50.0, np.degrees(traj.orbit.azimuth_apparent_norot) + 180)

    assert _airSpeed(headwind, traj, 100000.0, 20000.0) == pytest.approx( \
        np.sqrt(20000.0**2 + 2*20000.0*50.0*np.cos(elev) + 50.0**2), rel=1e-12)

    headwind.use_winds = False
    assert _airSpeed(headwind, traj, 100000.0, 20000.0) == 20000.0
    assert _airSpeed(None, traj, 100000.0, 20000.0) == 20000.0


def testWindsAreHandledInTheAirFrame(traj, tmp_path):
    """ In still air the wind path ends where the one without winds does. With a constant crosswind the final
        velocity relative to the air is the simulation's final speed, and the end moves across the track by
        the wind drift over the simulated time T minus the tilt of the path relative to the air, of length L:
        (w.c)*(T - L/|u0|), with c the horizontal direction across the track and u0 the initial velocity
        relative to the air.
    """

    still = _windProfile(str(tmp_path/"still.csv"), 0.0, 0.0)
    with_winds = _endState(traj, still)
    still.use_winds = False
    without_winds = _endState(traj, still)

    assert with_winds[1:5] + with_winds[6:] == pytest.approx(without_winds[1:5] + without_winds[6:], rel=1e-9)
    assert (with_winds[5] - without_winds[5] + 180)%360 - 180 == pytest.approx(0.0, abs=1e-7)

    windy = _windProfile(str(tmp_path/"windy.csv"), 50.0, 250.0)
    sr, _, lat, lon, ele, azim, elev, _, vel = _endState(traj, windy)
    wind = windy.wind(1000*ele)

    vel_air = vel*_motionENU(np.radians(azim), np.radians(elev)) - np.append(wind, 0.0)
    assert vectMag(vel_air) == pytest.approx(sr.frag_main.v, rel=1e-9)

    motion = _motionENU(traj.orbit.azimuth_apparent_norot, traj.orbit.elevation_apparent_norot)
    u0 = vectMag(20000.0*motion - np.append(windy.wind(100000.0), 0.0))

    # Across the track at the end, where the two runs differ only by the wind correction
    azim_motion = np.radians(without_winds[5]) + np.pi
    across = np.array([np.cos(azim_motion), -np.sin(azim_motion)])
    r = sr.const.r_earth + 1000*ele
    shift = np.array([np.radians(lon - without_winds[3])*r*np.cos(np.radians(lat)), \
        np.radians(lat - without_winds[2])*r])

    assert np.dot(shift, across) == pytest.approx( \
        np.dot(wind, across)*(sr.time_arr[-1] - sr.frag_main.length/u0), abs=1.0)


if __name__ == "__main__":

    import sys
    sys.exit(pytest.main([__file__, "-q"]))
