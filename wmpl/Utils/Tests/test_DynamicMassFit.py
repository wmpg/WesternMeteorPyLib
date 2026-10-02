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
import argparse
import contextlib

import numpy as np
import pytest

from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import derotatedRadiantAltAz, cartesian2Geo, jd2LST, latLonAlt2ECEF, \
    altAz2RADec, raDec2ECI, eci2RaDec, raDec2AltAz, ecef2ENU, enu2ECEF
from wmpl.Utils.GeoidHeightEGM96 import mslToWGS84Height
from wmpl.MetSim import MetSimErosion
from wmpl.Utils.SampleTrajectoryPositions import sampleTrajectory
from wmpl.Utils.DynamicMassFit import pointOnTrajectory, _robust_linear_fit, fitVelocity, runFragSim, \
    SIM_HT_MIN, computeFragEndParams, _airSpeed, _motionENU, _endDecel, groundSpeed, evalPointState, \
    runMonteCarloDynMass, setAtmosphere
from wmpl.Utils.AtmosphereProfile import AtmosphereProfile
from wmpl.Utils.Physics import dynamicMass
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, atmDensPoly, getAtmDensity
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

    dens_co = fitAtmPoly(traj.rend_lat, traj.rend_lon, sr.const.h_kill, 30000, traj.jdt_ref)

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


def testEndDecelerationOfASingleStepSimulationIsNaN(traj):
    """ A simulation that starts less than one step above the kill speed takes a single step, which has no
        deceleration; a normal one has the deceleration of its last step.
    """

    args = (0.1, 3500, np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), traj.jdt_ref, 30000, 5000, 45, 0.55)

    with contextlib.redirect_stdout(io.StringIO()):
        single = runFragSim(*args, v_kill=4999)
        normal = runFragSim(*args)

    assert len(single.time_arr) == 1
    assert np.isnan(_endDecel(single))
    assert _endDecel(normal) == pytest.approx((normal.main_vel_arr[-1] - normal.main_vel_arr[-2]) \
        /(normal.time_arr[-1] - normal.time_arr[-2]), rel=1e-12)


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

    azim, elev = traj.orbit.azimuth_apparent_norot, traj.orbit.elevation_apparent_norot
    headwind = _windProfile(str(tmp_path/"head.csv"), 50.0, np.degrees(azim) + 180)

    assert _airSpeed(headwind, 100000.0, 20000.0, azim, elev) == pytest.approx( \
        np.sqrt(20000.0**2 + 2*20000.0*50.0*np.cos(elev) + 50.0**2), rel=1e-12)

    headwind.use_winds = False
    assert _airSpeed(headwind, 100000.0, 20000.0, azim, elev) == 20000.0
    assert _airSpeed(None, 100000.0, 20000.0, azim, elev) == 20000.0


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
        sea level (m), independent of the ECI conversions of DynamicMassFit. """

    return np.array(latLonAlt2ECEF(lat, lon, mslToWGS84Height(lat, lon, ele_msl)))


def testEndOfAblationIsWhereMetSimTookTheBodyOverTheGround(traj, tmp_path):
    """ The final point is the evaluation point displaced by MetSim's 3D displacement in the ground-fixed frame of
        the evaluation point, and the final direction is MetSim's final velocity seen from the final point. Checked
        with plain geodetic conversions, without the Earth rotation or sidereal time used by DynamicMassFit; a
        missing rotation of the Earth would move the end by its rotation speed times the simulated time, hundreds
        of metres. """

    still = _windProfile(str(tmp_path/"still.csv"), 0.0, 0.0)
    sr, _, lat, lon, ele, azim, elev, time, vel = _endState(traj, still)
    frag = sr.frag_main

    meas_time, _, _, _, eval_lat, eval_lon, _, _ = evalPointState(traj, 100000.0)
    lat, lon = np.radians(lat), np.radians(lon)
    displacement = ecef2ENU(eval_lat, eval_lon, *(_ecef(lat, lon, 1000*ele) - _ecef(eval_lat, eval_lon, 100000.0)))
    vel_end = ecef2ENU(lat, lon, *enu2ECEF(eval_lat, eval_lon, frag.vx, frag.vy, frag.vz))

    assert displacement == pytest.approx([frag.px, frag.py, frag.pz], abs=1.0)
    assert vel_end == pytest.approx(vel*_motionENU(np.radians(azim), np.radians(elev)), abs=1e-3)
    assert time == pytest.approx(meas_time + sr.time_arr[-1], abs=1e-9)


def _inertialRun(traj, sr, height, vel, dt=2e-3):
    """ The single body of the simulation integrated with RK4 in the inertial ECI frame, where the drag acts on
        v - omega x r and the rotation of the Earth needs no Coriolis term, with the effective gravity of MetSim
        towards the centre of its sphere, plus the centripetal acceleration it leaves out. Only wmpl's horizontal
        coordinate conversions are shared with DynamicMassFit. Returns the final latitude, longitude (deg), height
        (km), radiant azimuth and elevation (deg), time (s) and mass (kg). """

    const = sr.const
    _, _, eci, jd, lat, lon, azim, elev = evalPointState(traj, height)
    omega = np.array([0.0, 0.0, MetSimErosion.EARTH_ROTATION_RATE])
    radiant = np.array(raDec2ECI(*altAz2RADec(azim, elev, jd, lat, lon)))
    up = np.array(raDec2ECI(*altAz2RADec(0.0, np.pi/2, jd, lat, lon)))
    centre0 = eci - (const.r_earth + height)*up
    K = const.gamma*const.shape_factor*const.rho**(-2/3.0)

    def centre(t):
        a = MetSimErosion.EARTH_ROTATION_RATE*t
        return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]) @ centre0

    def deriv(t, s):
        r, v, m = s[:3], s[3:6], s[6]
        down = centre(t) - r
        h = np.linalg.norm(down) - const.r_earth
        u = v - np.cross(omega, r)
        rho = atmDensPoly(h, const.dens_co)
        acc = -K*m**(-1/3.0)*rho*np.linalg.norm(u)*u \
            + MetSimErosion.G0*(const.r_earth/(const.r_earth + h))**2*down/np.linalg.norm(down) \
            + np.cross(omega, np.cross(omega, r))
        return np.concatenate([v, acc, [-K*const.sigma*m**(2/3.0)*rho*np.linalg.norm(u)**3]])

    s, t = np.concatenate([eci, -vel*radiant + np.cross(omega, eci), [const.m_init]]), 0.0
    while (np.linalg.norm(s[3:6] - np.cross(omega, s[:3])) > const.v_kill) \
            and (np.linalg.norm(s[:3] - centre(t)) - const.r_earth > const.h_kill):
        k1 = deriv(t, s); k2 = deriv(t + dt/2, s + dt/2*k1); k3 = deriv(t + dt/2, s + dt/2*k2)
        k4 = deriv(t + dt, s + dt*k3)
        s, t = s + dt/6*(k1 + 2*k2 + 2*k3 + k4), t + dt

    jd += t/86400
    lat, lon, ele = cartesian2Geo(jd, *s[:3])
    v_ground = s[3:6] - np.cross(omega, s[:3])
    azim, elev = raDec2AltAz(*eci2RaDec(-v_ground/np.linalg.norm(v_ground)), jd, lat, lon)

    return np.degrees(lat), np.degrees(lon), ele/1000, np.degrees(azim), np.degrees(elev), t, s[6]


def testEndOfAblationMatchesAnIntegrationInTheInertialFrame(traj):
    """ MetSim in the ground-fixed frame with gravity and the Coriolis acceleration, and the conversion of its end
        state, agree with an integration of the same body in the inertial frame. Here the path lasts 5 s and gravity
        turns it by 0.08-0.13 deg; without the Coriolis term the direction would be 0.04 deg off, with its sign flipped
        0.08 deg. """

    for mass in [1.0, 1000.0]:
        with contextlib.redirect_stdout(io.StringIO()):
            sr, final_mass, lat, lon, ele, azim, elev, _, _ = computeFragEndParams(traj, mass, 3500, 100000.0, \
                20000.0, 0.55)
        ref_lat, ref_lon, ref_ele, ref_azim, ref_elev, ref_time, ref_mass = _inertialRun(traj, sr, 100000.0, \
            20000.0)

        angle = np.degrees(np.arccos(np.clip(np.dot(_motionENU(np.radians(azim), np.radians(elev)), \
            _motionENU(np.radians(ref_azim), np.radians(ref_elev))), -1, 1)))
        horizontal = np.hypot(np.radians(lat - ref_lat), np.radians(lon - ref_lon)*np.cos(np.radians(lat))) \
            *sr.const.r_earth

        assert angle < 0.002
        assert horizontal < 30.0
        assert 1000*abs(ele - ref_ele) < 40.0
        assert sr.time_arr[-1] == pytest.approx(ref_time, abs=0.01)
        assert final_mass == pytest.approx(ref_mass, rel=0.01)


def testWindsInTheSimulationMoveTheEndAsExpected(traj, tmp_path):
    """ In still air the simulation with winds ends where the one without winds does. With a constant crosswind
        the final velocity relative to the air has the simulation's final speed, and the end moves across the
        track by the drift of the air over the simulated time T minus the tilt of the path relative to the air,
        of length L: (w.c)*(T - L/|u0|), with c the horizontal direction across the track and u0 the initial
        velocity relative to the air.
    """

    still = _windProfile(str(tmp_path/"still.csv"), 0.0, 0.0)
    with_winds = _endState(traj, still)
    still.use_winds = False
    without_winds = _endState(traj, still)

    assert with_winds[1:] == pytest.approx(without_winds[1:], rel=1e-9)

    windy = _windProfile(str(tmp_path/"windy.csv"), 50.0, 250.0)
    sr, _, lat, lon, ele, azim, elev, _, vel = _endState(traj, windy)
    wind = windy.wind(1000*ele)

    vel_air = vel*_motionENU(np.radians(azim), np.radians(elev)) - np.append(wind, 0.0)
    assert vectMag(vel_air) == pytest.approx(sr.frag_main.v, rel=1e-4)

    # The wind turns the final direction away from the one in still air
    angle = lambda a, b: np.degrees(np.arccos(np.clip(np.dot(a, b)/vectMag(a)/vectMag(b), -1, 1)))
    assert angle(_motionENU(np.radians(azim), np.radians(elev)), \
        _motionENU(np.radians(without_winds[5]), np.radians(without_winds[6]))) > 0.01

    _, _, _, _, _, _, eval_azim, eval_elev = evalPointState(traj, 100000.0)
    u0 = vectMag(20000.0*_motionENU(eval_azim, eval_elev) - np.append(windy.wind(100000.0), 0.0))

    # Across the track at the end, where the two runs differ only by the wind. The path length relative to the
    #   air is that of the simulation's speeds, which are relative to the air
    length_air = np.sum(sr.main_vel_arr)*sr.const.dt
    azim_motion = np.radians(without_winds[5]) + np.pi
    across = np.array([np.cos(azim_motion), -np.sin(azim_motion)])
    r = sr.const.r_earth + 1000*ele
    shift = np.array([np.radians(lon - without_winds[3])*r*np.cos(np.radians(lat)), \
        np.radians(lat - without_winds[2])*r])

    assert np.dot(shift, across) == pytest.approx( \
        np.dot(wind, across)*(sr.time_arr[-1] - length_air/u0), abs=1.0)


def testMonteCarloWorkersUseTheMSISModelOfTheMainProcess(traj):
    """ --atm selects the MSIS model in this process only. Where worker processes are spawned (macOS, Windows)
        they start from NRLMSISE-00, so runMonteCarloDynMass() hands the model to them: the realizations give
        the same masses on one core and on two, and not those of NRLMSISE-00. """

    args = ([traj, traj], traj.rbeg_ele/1000, traj.rend_ele/1000, 0.5, 3500, 0.55, 73000, 50, 3.0)
    kwargs = dict(run_final_sim=False, rng_seed=1)

    with contextlib.redirect_stdout(io.StringIO()):
        default = runMonteCarloDynMass(*args, cores=1, **kwargs)['dyn_mass_geom']
        setAtmosphere(argparse.Namespace(atm='2.1', atmtime=None))
        try:
            serial = runMonteCarloDynMass(*args, cores=1, **kwargs)['dyn_mass_geom']
            parallel = runMonteCarloDynMass(*args, cores=2, **kwargs)['dyn_mass_geom']
        finally:
            setAtmosphere(argparse.Namespace(atm='00', atmtime=None))

    assert parallel == pytest.approx(serial, rel=1e-12)
    assert abs(serial[0]/default[0] - 1) > 0.01


def testFragmentSimulationAtmosphereFollowsMSISOverTheSimulatedHeights(traj):
    """ The simulation only descends from its starting height, so its density polynomial has to follow MSIS
        there. A 7th order polynomial fitted up to 180 km misses it by 10-30% in the stratosphere.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        sr = runFragSim(0.1, 3500, np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), traj.jdt_ref, 30000, \
            5000, 45, 0.55)

    heights = np.linspace(sr.const.h_kill, 30000, 50)
    msis = np.array([getAtmDensity(traj.rend_lat, traj.rend_lon, ht, traj.jdt_ref) for ht in heights])

    assert atmDensPoly(heights, sr.const.dens_co) == pytest.approx(msis, rel=0.01)


if __name__ == "__main__":

    import sys
    sys.exit(pytest.main([__file__, "-q"]))
