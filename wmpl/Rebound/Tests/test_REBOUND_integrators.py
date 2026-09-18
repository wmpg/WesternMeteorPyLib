""" Tests for the selectable integrators (IAS15, WHFast, TRACE) and the MEGNO computation.

These integrate for real and are skipped when rebound/reboundx are not importable; the TRACE tests
are also skipped with a REBOUND older than 4.4, which does not provide TRACE. They need no
ephemeris file: the massive bodies are placed on circular orbits.
"""

import os
import importlib.util

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

AU_KM = 149597870.7
DAY = 2*np.pi/365.25                   # REBOUND time units (G = 1, AU, M_sun) per day
YEAR = 2*np.pi
KM_S = (86400.0/DAY)/AU_KM             # REBOUND velocity units per km/s


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name, skipping if REBOUND is unavailable. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_integrators", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    return module


def _traceAvailable(reb):
    try:
        reb.checkIntegratorAvailable("trace")
        return True
    except ValueError:
        return False


def _circular(r_au, m, phase, center=None, m_center=1.0):
    """ State [x, y, z, vx, vy, vz] of a circular orbit around a body of mass m_center. """

    v = np.sqrt((m_center + m)/r_au)
    state = np.array([r_au*np.cos(phase), r_au*np.sin(phase), 0.0,
                      -v*np.sin(phase), v*np.cos(phase), 0.0])
    if center is not None:
        state += np.array(center)

    return list(state)


def _departureTask(direction, days=365.25, n_outputs=40):
    """ A meteoroid leaving the Earth's surface at 20 km/s (backward in time for a backward run),
    with the Sun, the Earth, the Moon and Jupiter on circular orbits.
    """

    m_earth, m_moon, m_jupiter = 3.003e-6, 3.69e-8, 9.546e-4
    earth = _circular(1.0, m_earth, 0.7)
    moon = _circular(384400/AU_KM, m_moon, 2.0, center=earth, m_center=m_earth)
    jupiter = _circular(5.2, m_jupiter, 3.9)

    # Starting 100 km above the surface, moving away from the Earth in the integration's direction
    sign = 1.0 if direction == "forward" else -1.0
    u = np.array([0.3, 0.9, 0.3])
    u /= np.linalg.norm(u)
    r0 = np.array(earth[:3]) + u*(6471.0/AU_KM)
    v0 = np.array(earth[3:]) + sign*u*np.sqrt(20.0**2 + 11.1**2)*KM_S

    times = sign*np.linspace(0.0, days*DAY, n_outputs)

    return {
        "planet_names": ["Sun", "Earth", "Luna", "Jupiter"],
        "planet_states": [[0.0]*6, earth, moon, jupiter],
        "planet_masses": [1.0, m_earth, m_moon, m_jupiter],
        "particle_states": [list(r0) + list(v0)],
        "particle_names": ["obj"],
        "times": list(times),
        "direction": direction,
        "reference_frame": "heliocentric",
    }


def _finalOrbit(reb, task, integrator, dt_days=None, beta=None):
    result = reb._integrateParticles(dict(task, integrator=integrator, dt_days=dt_days, beta=beta))
    orbit = result["outputs"]["obj"][-1][2]

    return orbit, result["diagnostics"]["obj"]


### Integrator selection ###

def testUnknownIntegratorIsRejected(reb):
    with pytest.raises(ValueError, match="Unknown integrator"):
        reb.checkIntegratorAvailable("leapfrog")


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("integrator", ["whfast", "trace"])
def testFixedStepIntegratorsMatchIas15(reb, integrator, direction):
    """ Starting at the Earth's surface, WHFast and TRACE (with IAS15 through the departure) end a
    year later on the same orbit as IAS15. For TRACE backward this also checks the reversed-time
    integration that works around its negative-timestep bug.
    """

    if (integrator == "trace") and not _traceAvailable(reb):
        pytest.skip("TRACE needs REBOUND >= 4.4")

    task = _departureTask(direction)
    ref, _ = _finalOrbit(reb, task, "ias15")
    orbit, diag = _finalOrbit(reb, task, integrator, dt_days=0.5)

    assert orbit.a == pytest.approx(ref.a, rel=1e-5)
    assert orbit.e == pytest.approx(ref.e, rel=1e-5)
    assert orbit.inc == pytest.approx(ref.inc, abs=1e-6)

    # IAS15 handled the departure; the fixed-step integrator took over at an output time after it,
    # reported on the requested (signed) time axis
    assert diag["integrator"] == integrator
    t_from = diag["fixed_step_from_days"]
    assert t_from is not None
    assert np.sign(t_from) == (1 if direction == "forward" else -1)

    # The closest-approach times are reported on the same signed axis as the outputs
    t_moon = diag["min_time_days"]["Luna"]
    assert (t_moon*np.sign(t_from)) >= 0


def testTraceBackwardRefusesRadiationForces(reb):
    """ The reversed-time TRACE integration cannot handle dissipative Poynting-Robertson drag. """

    if not _traceAvailable(reb):
        pytest.skip("TRACE needs REBOUND >= 4.4")

    with pytest.raises(ValueError, match="radiation forces"):
        _finalOrbit(reb, _departureTask("backward", days=10.0), "trace", beta=0.01)


### MEGNO ###

def _keplerTask(direction, years, a=1.0, e=0.3, with_jupiter=False, jupiter_crosser=False):
    """ A test particle around the Sun (optionally with Jupiter), for MEGNO checks. """

    sim = __import__("rebound").Simulation()
    sim.add(m=1.0)
    names = ["Sun"]
    if with_jupiter:
        sim.add(m=9.546e-4, a=5.2, e=0.048, inc=0.02)
        names.append("Jupiter")
    if jupiter_crosser:
        sim.add(primary=sim.particles[0], a=4.2, e=0.5, inc=0.14, f=0.5)
    else:
        sim.add(primary=sim.particles[0], a=a, e=e, inc=0.1, f=1.0)

    ps = sim.particles
    sign = 1.0 if direction == "forward" else -1.0

    return {
        "planet_names": names,
        "planet_states": [[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in ps[:-1]],
        "planet_masses": [p.m for p in ps[:-1]],
        "particle_states": [[ps[-1].x, ps[-1].y, ps[-1].z, ps[-1].vx, ps[-1].vy, ps[-1].vz]],
        "particle_names": ["obj"],
        "times": list(sign*np.linspace(0.0, years*YEAR, 100)),
        "direction": direction,
    }


@pytest.mark.parametrize("direction", ["forward", "backward"])
def testMegnoConvergesToTwoForAKeplerOrbit(reb, direction):
    """ A Keplerian orbit is regular: <Y> converges to 2, in both time directions. """

    megno = reb.computeMegno(_keplerTask(direction, years=200.0))
    verdict = reb.classifyMegno(megno["times_days"], megno["megno"], megno["a_au"])

    assert megno["megno"][-1] == pytest.approx(2.0, abs=0.05)
    assert verdict["status"] == "regular"
    assert np.sign(megno["times_days"][-1]) == (1 if direction == "forward" else -1)


def testMegnoDetectsAChaoticJupiterCrosser(reb):
    """ An orbit crossing Jupiter's is chaotic: <Y> grows well past 2 within 300 years. """

    megno = reb.computeMegno(_keplerTask("backward", years=300.0, with_jupiter=True,
                                         jupiter_crosser=True))
    verdict = reb.classifyMegno(megno["times_days"], megno["megno"], megno["a_au"])

    assert megno["megno"][-1] > 4.0
    assert verdict["status"] == "chaotic"
    assert 0 < verdict["lyapunov_time_years"] < 300.0
