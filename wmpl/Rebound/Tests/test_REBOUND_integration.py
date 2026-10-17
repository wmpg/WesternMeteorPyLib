""" Tests for REBOUND version compatibility and a real end-to-end integration.

The compatibility helpers are tested with stand-in objects, so those tests run without REBOUND. The
remaining tests integrate for real and are skipped when rebound/reboundx, the DE430 ephemeris or the
example trajectory are unavailable. They catch what the pure-numerics tests cannot: REBOUND API
changes between major versions, and a reboundx compiled against a different REBOUND version.
"""

import os
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dynesty",
                           "examples", "20191023_091225")
EXAMPLE_FILE = "20191023_091225_trajectory.pickle"


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_integration", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def realReb(reb):
    """ The REBOUND module, skipping the test if rebound/reboundx cannot be imported. """

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable: {}".format(reb._REBOUND_IMPORT_ERROR))

    return reb


### Version compatibility helpers (no REBOUND needed) ###


class _RecordingSim:
    """ Stand-in simulation recording the arguments passed to add(). """

    def __init__(self):
        self.added = []
        self.particles = []

    @property
    def N(self):
        return len(self.particles)

    def add(self, *args, **kwargs):
        self.added.append((args, kwargs))
        self.particles.append(SimpleNamespace())


@pytest.mark.parametrize("version, args, kwargs, expected_kwargs, named_after", [
    # REBOUND 4 names every particle with 'hash'
    ("4.3.0", ("Geocenter",), {"date": "JD2451545.0"}, {"date": "JD2451545.0", "hash": "Earth"}, False),
    ("4.3.0", (), {"m": 1.0}, {"m": 1.0, "hash": "Earth"}, False),
    # REBOUND 5 uses 'name', except for Horizons queries, which are named after being added
    ("5.1.1", (), {"m": 1.0}, {"m": 1.0, "name": "Earth"}, False),
    ("5.1.1", ("Geocenter",), {"date": "JD2451545.0"}, {"date": "JD2451545.0"}, True),
])
def testParticleIdentifiedWithTheVersionsKeyword(reb, monkeypatch, version, args, kwargs,
                                                 expected_kwargs, named_after):

    monkeypatch.setattr(reb, "rb", SimpleNamespace(__version__=version), raising=False)

    sim = _RecordingSim()
    reb._addNamedParticle(sim, "Earth", *args, **kwargs)

    assert sim.added == [(args, expected_kwargs)]
    assert getattr(sim.particles[0], "name", None) == ("Earth" if named_after else None)


def testHeartbeatSetterUsedWhenItWorks(reb):

    sim = SimpleNamespace()

    def func(sim_pointer):
        pass

    assert reb._setHeartbeat(sim, func) is None
    assert sim.heartbeat is func


def testHeartbeatFallsBackWhenTheSetterIsBroken(reb, monkeypatch):

    # Mimic the REBOUND 5.1.1 setter, which fails because '_hb' is missing from __slots__
    class BrokenSim:
        __slots__ = ["_heartbeat"]

        @property
        def heartbeat(self):
            return None

        @heartbeat.setter
        def heartbeat(self, func):
            self._hb = func

    wrapped = []
    monkeypatch.setattr(reb, "rb", SimpleNamespace(
        simulation=SimpleNamespace(AFF=lambda f: wrapped.append(f) or ("wrapped", f))), raising=False)

    def func(sim_pointer):
        pass

    sim = BrokenSim()
    ref = reb._setHeartbeat(sim, func)

    assert ref == ("wrapped", func)
    assert sim._heartbeat is ref
    assert wrapped == [func]


def testHeartbeatUnrelatedAttributeErrorPropagates(reb):

    class OtherSim:
        @property
        def heartbeat(self):
            return None

        @heartbeat.setter
        def heartbeat(self, func):
            raise AttributeError("something else")

    with pytest.raises(AttributeError, match="something else"):
        reb._setHeartbeat(OtherSim(), lambda p: None)


@pytest.mark.parametrize("version, expected", [
    ("4.3.0", 4), ("5.1.1", 5), ("5", 5),
    # A version string that does not start with a number is assumed to be a recent one, since the
    # naming keyword only changed going forward
    ("dev", 5), ("", 5),
])
def testMajorVersionIsReadOrAssumedRecent(reb, monkeypatch, version, expected):

    monkeypatch.setattr(reb, "rb", SimpleNamespace(__version__=version), raising=False)

    assert reb._reboundMajorVersion() == expected


def testUnattachedReboundxIsReportedWithTheFix(realReb):
    """ The whole point of the check: an Extras that did not attach must say how to repair it. """

    detached = SimpleNamespace(extras=None)

    # The message names the installed rebound and reboundx versions, so this needs them importable
    with pytest.raises(RuntimeError) as exc_info:
        realReb._checkReboundxAttached(detached)

    message = str(exc_info.value)
    assert "did not attach" in message
    assert "--no-build-isolation" in message, "the error must carry the command that repairs it"
    assert "pip uninstall reboundx" in message, "and the way out where reboundx cannot be built"


### Installed REBOUND/REBOUNDx ###


def testReboundxUsableCheckPassesOnAWorkingInstall(realReb):
    """ The early check used by reboundSimulate, on a throwaway simulation. """

    realReb.checkReboundxUsable()


def testReboundxAttachesToTheSimulation(realReb):
    """ Fails when reboundx was compiled against a different REBOUND version than the installed one. """

    sim = realReb.rb.Simulation()
    rebx = realReb.reboundx.Extras(sim)

    realReb._checkReboundxAttached(sim)
    assert rebx is not None


def testNamedParticlesAndHeartbeatWorkOnTheInstalledVersion(realReb):

    rb = realReb.rb

    sim = rb.Simulation()
    realReb._addNamedParticle(sim, "Sun", m=1.0)
    realReb._addNamedParticle(sim, "Earth", m=3e-6, x=1.0, vy=1.0)

    assert sim.particles["Earth"].m == 3e-6
    with pytest.raises(rb.ParticleNotFound):
        sim.particles["Mars"]

    calls = []

    def heartbeat(sim_pointer):
        calls.append(sim_pointer.contents.t)

    heartbeat_ref = realReb._setHeartbeat(sim, heartbeat)
    sim.integrate(0.1)

    assert len(calls) > 0
    del heartbeat_ref


def testEndToEndIntegrationMatchesReference(realReb):
    """ Integrate the example trajectory 60 days back and compare with a reference solution.

    The reference was computed with rebound 4.4.10 and reboundx 4.4.2, with the time unit consistent
    with G = 1 (DAYS_PER_SIM_TIME) and the Earth's J2/J4 symmetric about its pole of date. With the
    J2/J4 about the ecliptic pole instead, a was 10.99379 AU, and with the Julian time unit as well
    10.98466 AU. The tolerance leaves room for floating-point differences between platforms, while
    catching any real change in the dynamics. It needs REBOUNDx >= 4.4.0, which orients the J2/J4.
    """

    from wmpl.Config import config
    from wmpl.Utils.Pickling import loadPickle

    if not os.path.isfile(config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not found: {}".format(config.jpl_ephem_file))

    if not os.path.isfile(os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)):
        pytest.skip("Example trajectory not found")

    if not realReb.harmonicsSpinAxisSupported():
        pytest.skip("REBOUNDx < 4.4.0 cannot orient the Earth's J2/J4")

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)

    # The reference is the full post-Newtonian interaction between all bodies (the default)
    sim_outputs, sim_outputs_mc, diagnostics = realReb.reboundSimulate(
        None, None, traj=traj, direction="backward", sim_days=60, n_outputs=200, mc_runs=1, n_cpu=1,
        show_progress=False, return_diagnostics=True, random_seed=42, gr="gr_full")

    assert len(sim_outputs) == 200
    assert sim_outputs_mc == {}

    orbit = sim_outputs[-1][2]
    rtol = 1e-7
    assert orbit.a == pytest.approx(10.999618455077444, rel=rtol)
    assert orbit.e == pytest.approx(0.9470454373170485, rel=rtol)
    assert np.degrees(orbit.inc) == pytest.approx(164.56525762377314, rel=rtol)
    assert np.degrees(orbit.Omega) == pytest.approx(29.396553077832916, rel=rtol)
    assert np.degrees(orbit.omega) == pytest.approx(81.51210470078942, rel=rtol)

    # The Moon's closest approach comes from the heartbeat, so this also checks that it ran
    diag = next(iter(diagnostics.values()))
    assert diag["departed"]
    assert diag["impact"] is None
    assert diag["min_dist_au"]["Luna"] == pytest.approx(0.001819247055906065, rel=rtol)
    # The within-step refinement added in #99 moves this off the step-end value it had before
    assert diag["min_time_days"]["Luna"] == pytest.approx(-0.0426152665, rel=1e-6)

    # The nominal records every passage within PASSAGE_MAX_HILL Hill radii, its encounters included
    passages = diag["passages"]
    assert passages
    assert all(p["n_hill"] < realReb.PASSAGE_MAX_HILL for p in passages)
    for enc in diag["encounters"]:
        if enc["body"] != "Sun":
            assert any((p["body"] == enc["body"]) and (p["time_days"] == enc["time_days"]) for p in passages)


### Units ###


def testTimeUnitIsTheOneOfGEqualsOne(realReb):
    """ With G = 1, AU and solar masses, the time unit is REBOUND's yr2pi, sqrt(AU^3/GM_Sun). The
    planets added from Horizons and REBOUNDx's speed of light use it, so the local ephemeris and the
    meteoroid must too. """

    rb = realReb.rb

    assert realReb.DAYS_PER_SIM_TIME == pytest.approx(rb.units.times_SI["yr2pi"]/86400, rel=1e-14)

    # A simulation in the units REBOUND sets for Horizons queries has G = 1
    sim = rb.Simulation()
    sim.units = ("AU", "yr2pi", "Msun")
    assert sim.G == pytest.approx(1.0, rel=1e-12)

    # REBOUNDx's speed of light, in AU per time unit
    c_au_day = 299792.458*86400/149597870.7
    assert realReb.rbxConstants.C == pytest.approx(c_au_day*realReb.DAYS_PER_SIM_TIME, rel=1e-12)

    # The conversions are inverses of each other, and a Julian year is not 2 pi
    assert realReb.simTimeToDays(realReb.daysToSimTime(123.4)) == pytest.approx(123.4, rel=1e-15)
    assert realReb.daysToSimTime(365.25) == pytest.approx(2*np.pi*(1 - 1.89e-5), rel=1e-7)


def testPlanetsFollowTheEphemeris(realReb):
    """ The planets started from the local ephemeris and integrated for 10 years stay within tens of
    km of where the ephemeris puts them. With the Julian year over 2 pi as the time unit, the Earth
    was 720 000 km off, because G = 1 made the Sun 3.8e-5 too massive. """

    from wmpl.Config import config

    if not os.path.isfile(config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not found: {}".format(config.jpl_ephem_file))

    rb = realReb.rb
    eph = realReb.SPK.open(config.jpl_ephem_file)

    names = ["Sun", "Mercury", "Venus", "Earth", "Luna", "Mars", "Jupiter", "Saturn", "Uranus",
             "Neptune"]
    jd = 2458779.88
    days = -3652.5

    sim = rb.Simulation()
    for name in names:
        state, mass = realReb.ephemBodyStateRebound(name, jd, eph)
        realReb._addNamedParticle(sim, name, m=mass, x=state[0], y=state[1], z=state[2],
                                  vx=state[3], vy=state[4], vz=state[5])

    # General relativity, as in the integrations (Mercury drifts by ~1900 km in 10 years without it)
    rebx = realReb.reboundx.Extras(sim)
    gr = rebx.load_force("gr_full")
    rebx.add_force(gr)
    gr.params["c"] = realReb.rbxConstants.C

    sim.move_to_com()
    sim.integrate(realReb.daysToSimTime(days))

    def position(name):
        p = sim.particles[name]
        return np.array([p.x, p.y, p.z])

    # Measured: Mercury 1 km, Venus 0, Earth 1, Mars 4, Jupiter 26, Moon (from the Earth) 62 km
    au_km = 149597870.7
    for name, centre, limit_km in [("Mercury", "Sun", 20), ("Venus", "Sun", 20), ("Earth", "Sun", 20),
                                   ("Mars", "Sun", 40), ("Jupiter", "Sun", 100),
                                   ("Luna", "Earth", 300)]:
        ref = (np.array(realReb.ephemBodyStateRebound(name, jd + days, eph)[0][:3])
               - np.array(realReb.ephemBodyStateRebound(centre, jd + days, eph)[0][:3]))
        err_km = np.linalg.norm(position(name) - position(centre) - ref)*au_km
        assert err_km < limit_km, "{:s} is {:.0f} km off the ephemeris".format(name, err_km)


### Axis of the Earth's J2/J4 ###


def testEarthSpinAxisIsThePoleOfDate(reb):
    """ At J2000 the pole is at (0, sin eps, cos eps) in the ecliptic frame, 23.44 deg from the
    ecliptic pole; it then precesses by ~20 arcsec per year about the ecliptic pole. """

    eps = reb.J2000_OBLIQUITY
    pole_j2000 = np.array([0.0, np.sin(eps), np.cos(eps)])

    assert np.allclose(reb.earthSpinAxisEcliptic(reb.J2000_JD.days), pole_j2000, atol=1e-15)

    for years in [21.16, 100.0, -400.0]:
        pole = np.array(reb.earthSpinAxisEcliptic(reb.J2000_JD.days + years*365.25))
        assert np.linalg.norm(pole) == pytest.approx(1.0, abs=1e-15)

        # ~20.04 arcsec per year away from the J2000 pole, staying ~23.44 deg from the ecliptic pole
        moved = np.degrees(np.arccos(np.clip(pole @ pole_j2000, -1, 1)))*3600
        assert moved/abs(years) == pytest.approx(20.04, rel=0.005)
        assert np.degrees(np.arccos(pole[2])) == pytest.approx(np.degrees(eps), abs=0.001)

    # Same pole as precessing the coordinates with equatorialCoordPrecession (away from J2000,
    #   where it wraps a declination of exactly +90 deg to -90 deg)
    from wmpl.Utils.TrajConversions import equatorialCoordPrecession
    jd = 2459274.41
    ra, dec = equatorialCoordPrecession(jd, reb.J2000_JD.days, 0.0, np.pi/2)
    expected = reb.rotateVector(np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]),
                                np.array([1.0, 0.0, 0.0]), -eps)
    assert np.allclose(reb.earthSpinAxisEcliptic(jd), expected, atol=1e-12)


def _slowDepartureTask(reb, spin_axis, rotation=None):
    """ A meteoroid leaving the Earth from 100 km up at 14 km/s, with the Sun, the Earth and the Moon.
    rotation, if given, is a 3x3 matrix applied to every state, to integrate in a rotated frame. """

    au_km = 149597870.7
    day = 1.0/reb.DAYS_PER_SIM_TIME
    km_s = (86400.0/day)/au_km
    m_earth, m_moon = 3.003e-6, 3.69e-8

    earth = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    moon = earth + np.array([0.0, 384400/au_km, 0.0, -1.022*km_s, 0.0, 0.0])
    u = np.array([0.5, 0.2, 0.84])
    u /= np.linalg.norm(u)
    obj = np.concatenate([earth[:3] + u*6471.0/au_km, earth[3:] - u*np.sqrt(9.0**2 + 11.1**2)*km_s])

    states = [np.zeros(6), earth, moon, obj]
    if rotation is not None:
        states = [np.concatenate([rotation @ s[:3], rotation @ s[3:]]) for s in states]

    return {
        "planet_names": ["Sun", "Earth", "Luna"],
        "planet_states": [list(s) for s in states[:3]],
        "planet_masses": [1.0, m_earth, m_moon],
        "particle_states": [list(states[3])],
        "particle_names": ["obj"],
        "times": list(-np.linspace(0.0, 10*day, 5)),
        "direction": "backward",
        "reference_frame": "heliocentric",
        "earth_spin_axis": spin_axis,
    }


def testHarmonicsAreSymmetricAboutTheSpinAxis(realReb):
    """ Giving the Earth's spin axis gives the same orbit as rotating the whole simulation so that
    the axis is the z axis, which is what REBOUNDx uses by default. """

    if not realReb.harmonicsSpinAxisSupported():
        pytest.skip("REBOUNDx < 4.4.0 cannot orient the harmonics")

    reb = realReb
    eps = reb.J2000_OBLIQUITY
    axis = [0.0, np.sin(eps), np.cos(eps)]

    # Rotation about x taking the axis to z (ecliptic to equatorial)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(eps), -np.sin(eps)], [0.0, np.sin(eps), np.cos(eps)]])

    with_axis = reb._integrateParticles(_slowDepartureTask(reb, axis))["outputs"]["obj"][-1][1]
    rotated = reb._integrateParticles(_slowDepartureTask(reb, None, rot))["outputs"]["obj"][-1][1]
    ecliptic_z = reb._integrateParticles(_slowDepartureTask(reb, None))["outputs"]["obj"][-1][1]

    rotated_back = np.concatenate([rot.T @ np.array(rotated[:3]), rot.T @ np.array(rotated[3:])])

    ms = 149597870.7e3/reb.DAYS_PER_SIM_TIME/86400
    dv_same = np.linalg.norm(np.array(with_axis[3:]) - rotated_back[3:])*ms
    dv_z = np.linalg.norm(np.array(with_axis[3:]) - np.array(ecliptic_z[3:]))*ms

    assert dv_same < 1e-6
    # The ecliptic pole as the axis is m/s off (measured 5.2 m/s on Winchcombe)
    assert dv_z > 0.5


### The Sun's J2 ###


def testSunSpinAxisIsTheIauPole(reb):
    """ The Sun's rotation pole (IAU: RA 286.13 deg, Dec 63.87 deg) is 7.25 deg from the ecliptic
    pole, and the solar equator crosses the ecliptic at a longitude of 75.77 deg (J2000). """

    pole = np.array(reb.sunSpinAxisEcliptic())

    assert np.linalg.norm(pole) == pytest.approx(1.0, abs=1e-15)
    assert np.degrees(np.arccos(pole[2])) == pytest.approx(7.25, abs=0.005)
    assert np.degrees(np.arctan2(pole[0], -pole[1])) % 360 == pytest.approx(75.77, abs=0.02)


def testSunJ2KeepsMercuryOnTheEphemeris(realReb):
    """ DE430 includes the Sun's J2. With it, Mercury integrated from DE430 for 100 years stays within
    a km of the ephemeris (measured 0.6 km), against 13 km without it. """

    from wmpl.Config import config

    if not os.path.isfile(config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not found: {}".format(config.jpl_ephem_file))

    if not realReb.harmonicsSpinAxisSupported():
        pytest.skip("REBOUNDx < 4.4.0 cannot orient the harmonics")

    rb = realReb.rb
    eph = realReb.SPK.open(config.jpl_ephem_file)
    names = ["Sun", "Mercury", "Venus", "Earth", "Luna", "Mars", "Jupiter", "Saturn", "Uranus",
             "Neptune"]
    jd = 2458779.88
    days = -36525.0

    def mercuryError(sun_j2):

        sim = rb.Simulation()
        for name in names:
            state, mass = realReb.ephemBodyStateRebound(name, jd, eph)
            realReb._addNamedParticle(sim, name, m=mass, x=state[0], y=state[1], z=state[2],
                                      vx=state[3], vy=state[4], vz=state[5])

        rebx = realReb.reboundx.Extras(sim)
        gr = rebx.load_force("gr_full")
        rebx.add_force(gr)
        gr.params["c"] = realReb.rbxConstants.C

        if sun_j2:
            gh = rebx.load_force("gravitational_harmonics")
            rebx.add_force(gh)
            sun = sim.particles["Sun"]
            sun.params["J2"] = realReb.SUN_J2
            sun.params["R_eq"] = realReb.SUN_J2_RADIUS_AU
            sun.params["Omega"] = rb.Vec3d(*realReb.sunSpinAxisEcliptic())

        sim.move_to_com()
        sim.integrate(realReb.daysToSimTime(days))

        ps = sim.particles
        rel = np.array([ps["Mercury"].x - ps["Sun"].x, ps["Mercury"].y - ps["Sun"].y,
                        ps["Mercury"].z - ps["Sun"].z])
        ref = (np.array(realReb.ephemBodyStateRebound("Mercury", jd + days, eph)[0][:3])
               - np.array(realReb.ephemBodyStateRebound("Sun", jd + days, eph)[0][:3]))

        return np.linalg.norm(rel - ref)*149597870.7

    assert mercuryError(True) < 2.0
    assert mercuryError(False) > 8.0



### General relativity model ###


def testSunOnlyGrMatchesTheFullOne(realReb):
    """ The Sun's post-Newtonian correction on every body (--gr gr) gives the same orbit as the
    default, the interaction between all pairs of bodies, to ~1e-7 in a over 60 days (measured 9e-8
    on the example; a 1 m/s change of the initial velocity moves a by 1e-3). """

    from wmpl.Config import config
    from wmpl.Utils.Pickling import loadPickle

    if not os.path.isfile(config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not found: {}".format(config.jpl_ephem_file))

    if not os.path.isfile(os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)):
        pytest.skip("Example trajectory not found")

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)

    def finalOrbit(**kwargs):
        outputs, _ = realReb.reboundSimulate(
            None, None, traj=traj, direction="backward", sim_days=60, n_outputs=50, mc_runs=1,
            n_cpu=1, show_progress=False, **kwargs)
        return outputs[-1][2]

    sun_only, default = finalOrbit(gr="gr"), finalOrbit()

    assert sun_only.a == pytest.approx(default.a, rel=3e-7)
    assert sun_only.a != default.a
    assert sun_only.e == pytest.approx(default.e, abs=1e-7)

    with pytest.raises(ValueError):
        finalOrbit(gr="gr_potential")


### Chaos indicators on the default run ###


def _exampleWithCovariance(realReb):

    from wmpl.Config import config
    from wmpl.Utils.Pickling import loadPickle

    if not os.path.isfile(config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not found: {}".format(config.jpl_ephem_file))
    if not os.path.isfile(os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)):
        pytest.skip("Example trajectory not found")

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([100.0**2]*3 + [1.0**2]*3)

    return traj


def testDefaultRunGivesNoHorizonAndNoMegnoVerdict(realReb):
    """ With the default span of 60 days the clones keep the order of their start, so there is no
    predictability horizon, and the MEGNO is too short for a verdict; both say so instead of failing,
    and the MEGNO still returns ln|delta| and the departure. """

    traj = _exampleWithCovariance(realReb)

    outputs, outputs_mc, diag = realReb.reboundSimulate(
        None, None, traj=traj, direction="backward", sim_days=60, n_outputs=100, mc_runs=12, n_cpu=1,
        show_progress=False, return_diagnostics=True, random_seed=3, compute_megno=True,
        obj_name=traj.traj_id)

    megno = diag[traj.traj_id]["megno"]
    assert len(megno["ln_delta"]) == len(megno["megno"]) == len(megno["times_days"])
    # 3 Earth Hill radii (4.5 million km) take about a day at the example's speed
    assert megno["departure_days"] is not None and -3.0 < megno["departure_days"] < 0
    assert megno["verdict"]["status"] == "too_short"
    assert megno["verdict"]["lyapunov_time_years"] is None

    clone_diag = {name: d for name, d in diag.items() if name in outputs_mc}
    h = realReb.predictabilityHorizon(outputs, outputs_mc, clone_diag=clone_diag)
    assert h["n_clones"] == 12
    assert h["horizon_days"] is None
    assert h["elements"]["a"]["rank_corr_end"] > 0.9
    assert h["first_encounter_days"]["fraction"] == 0.0


def testMegnoRescalingLeavesTheDeviationUnchanged(realReb, monkeypatch):
    """ Rescaling the deviation to unit length at the outputs (so that it never overflows) changes
    neither ln|delta| nor <Y>: the variational equations are linear. """

    traj = _exampleWithCovariance(realReb)

    def megno():
        _, _, diag = realReb.reboundSimulate(
            None, None, traj=traj, direction="backward", sim_days=365.25, n_outputs=60, mc_runs=1,
            n_cpu=1, show_progress=False, return_diagnostics=True, compute_megno=True,
            obj_name=traj.traj_id)
        return diag[traj.traj_id]["megno"]

    reference = megno()
    monkeypatch.setattr(realReb, "MEGNO_RESCALE_ABOVE", 1.5)
    rescaled = megno()

    assert max(reference["ln_delta"]) > np.log(1.5)
    assert np.allclose(rescaled["ln_delta"], reference["ln_delta"], rtol=0, atol=1e-8)
    assert np.allclose(rescaled["megno"], reference["megno"], rtol=1e-9, atol=1e-12)
