""" Regression tests for the closest approaches measured during a real REBOUND integration.

Away from the Earth IAS15 takes steps of 1-2 days, so the distances at the step ends alone can miss
a planetary flyby's minimum by a large fraction of the step length. These tests integrate synthetic
flybys with the integrator's own step choice and check the recorded minima, and the list of every
encounter, against dense re-integrations of the same flybys. They are skipped when rebound/reboundx
are not importable. They need no ephemeris file: the massive bodies are placed on circular orbits.
"""

import os
import importlib.util

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

AU_KM = 149597870.7
DAY = 2*np.pi/365.25                   # REBOUND time units (G = 1, AU, M_sun) per day
KM_S = (86400.0/DAY)/AU_KM             # REBOUND velocity units per km/s


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name, skipping if REBOUND is unavailable. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_encounters", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    return module


def _circular(r_au, m, phase):
    """ Barycentric-ish state [x, y, z, vx, vy, vz] of a circular orbit around a unit-mass Sun. """

    v = np.sqrt((1.0 + m)/r_au)

    return [r_au*np.cos(phase), r_au*np.sin(phase), 0.0, -v*np.sin(phase), v*np.cos(phase), 0.0]


def _flybyTask(v_rel_kms, b_hill, t_before_days, direction):
    """ A Mercury flyby at b_hill Mercury Hill radii, closest approach ~t_before_days after the
    start (forward) or before it (backward). The object starts far from the Earth, so every body is
    tracked from the start.
    """

    m_mercury, m_earth = 1.66e-7, 3.0e-6
    planet_names = ["Sun", "Mercury", "Earth"]
    planet_states = [[0.0]*6, _circular(0.387, m_mercury, 0.3), _circular(1.0, m_earth, 2.5)]
    planet_masses = [1.0, m_mercury, m_earth]

    mercury = np.array(planet_states[1])
    u = np.array([0.6, -0.3, 0.74])
    u /= np.linalg.norm(u)
    n = np.cross(u, [0.0, 0.0, 1.0])
    n /= np.linalg.norm(n)

    b = b_hill*0.001475
    sign = 1.0 if direction == "forward" else -1.0
    r0 = mercury[:3] + b*n - sign*v_rel_kms*KM_S*u*t_before_days*DAY
    v0 = mercury[3:] + v_rel_kms*KM_S*u

    return {
        "planet_names": planet_names,
        "planet_states": planet_states,
        "planet_masses": planet_masses,
        "particle_states": [list(r0) + list(v0)],
        "particle_names": ["obj"],
        "direction": direction,
        "reference_frame": "heliocentric",
    }


def _closestApproach(reb, task, times):
    """ Mercury closest approach (AU, days) recorded by _integrateParticles for the given outputs. """

    task = dict(task, times=list(times))
    diag = reb._integrateParticles(task)["diagnostics"]["obj"]

    return diag["min_dist_au"]["Mercury"], diag["min_time_days"]["Mercury"]


@pytest.mark.parametrize("direction", ["forward", "backward"])
def testFlybyMinimumIsNotLimitedByTheIntegratorStep(reb, direction):
    """ A fast Mercury flyby at ~2 Hill radii, crossed in steps of millions of km. The step-end
    distances alone overestimate the minimum here (by 1% forward and 12% backward), so this fails
    without the refinement. The refined minimum must match a dense re-integration.
    """

    sign = 1.0 if direction == "forward" else -1.0
    task = _flybyTask(v_rel_kms=45.0, b_hill=2.0, t_before_days=2.0, direction=direction)

    # Natural IAS15 steps: a single output at the end of the span
    d_coarse, t_coarse = _closestApproach(reb, task, [0.0, sign*4.0*DAY])

    # Reference: outputs every ~2 minutes force short steps through the whole span
    d_dense, t_dense = _closestApproach(reb, task, sign*np.linspace(0.0, 4.0*DAY, 3001))

    assert d_coarse == pytest.approx(d_dense, rel=1e-6)
    assert abs(t_coarse - t_dense)*1440 < 0.5

    # The flyby really is an encounter well inside the 3 Hill-radii threshold
    assert d_dense/reb.HILL_RADII_AU["Mercury"] < 2.5


def _resonantTask(reb, b_hill, inclination_deg, t_before_days):
    """ An object with Mercury's orbital period on an orbit inclined to Mercury's, meeting Mercury
    ~t_before_days after the start. With equal periods and nearly circular orbits the two meet
    again at the opposite node half an orbit later, and so on every ~44 days.
    """

    task = _flybyTask(v_rel_kms=10.0, b_hill=b_hill, t_before_days=0.0, direction="forward")
    mercury = np.array(task["planet_states"][1])

    # Just outside Mercury, moving along Mercury's direction of motion tilted out of its plane, with
    # the vis-viva speed for Mercury's semi-major axis, so both have the same orbital period
    inc = np.radians(inclination_deg)
    r_hat = mercury[:3]/np.linalg.norm(mercury[:3])
    v = mercury[3:]
    offset = b_hill*reb.HILL_RADII_AU["Mercury"]*r_hat
    v_dir = v*np.cos(inc) + np.cross(r_hat, v)*np.sin(inc)
    v_dir /= np.linalg.norm(v_dir)
    v_tilted = v_dir*np.sqrt(2.0/np.linalg.norm(mercury[:3] + offset) - 1.0/0.387)

    # Two-body propagation of the object back to the start
    sim = reb.rb.Simulation()
    sim.add(m=1.0)
    sim.add(x=mercury[0] + offset[0], y=mercury[1] + offset[1], z=mercury[2] + offset[2],
            vx=v_tilted[0], vy=v_tilted[1], vz=v_tilted[2])
    sim.integrate(-t_before_days*DAY)
    obj = sim.particles[1]

    # Mercury on its circular orbit at the start
    phase = np.arctan2(mercury[1], mercury[0]) - np.sqrt(1.0 + 1.66e-7)/0.387**1.5*t_before_days*DAY
    task["planet_states"][1] = _circular(0.387, 1.66e-7, phase)
    task["particle_states"] = [[obj.x, obj.y, obj.z, obj.vx, obj.vy, obj.vz]]

    return task


def testRepeatedEncountersWithTheSameBodyAreAllListed(reb):
    """ Three passages by Mercury, half an orbit apart, are all listed in order, while the per-body
    minimum keeps only the deepest. The list does not depend on the step sampling.
    """

    task = _resonantTask(reb, b_hill=1.5, inclination_deg=15.0, t_before_days=2.0)
    rh = reb.HILL_RADII_AU["Mercury"]

    def run(times):
        diag = reb._integrateParticles(dict(task, times=list(times)))["diagnostics"]["obj"]
        return diag, [e for e in diag["encounters"] if e["body"] == "Mercury"]

    diag, mercury = run([0.0, 100.0*DAY])
    _, mercury_dense = run(np.linspace(0.0, 100.0*DAY, 3001))

    # All passages, in order: ~2 days after the start, then every half Mercury orbit (~44 days)
    assert len(mercury) == 3
    assert mercury[0]["time_days"] == pytest.approx(2.0, abs=0.1)
    assert np.diff([e["time_days"] for e in mercury]) == pytest.approx([44.0, 44.0], abs=1.0)
    assert all(e["n_hill"] < 3.0 for e in mercury)

    # Nothing else comes close in this configuration
    assert [e["body"] for e in diag["encounters"]] == ["Mercury"]*3

    # The per-body minimum is the deepest passage
    deepest = min(mercury, key=lambda e: e["min_dist_au"])
    assert diag["min_dist_au"]["Mercury"] == deepest["min_dist_au"]
    assert diag["min_time_days"]["Mercury"] == deepest["time_days"]

    # Same passages with forced short steps (these slower, more bent passages leave the refined
    # distance within ~1e-5 of the dense one)
    assert len(mercury_dense) == 3
    for e, e_dense in zip(mercury, mercury_dense):
        assert e["min_dist_au"] == pytest.approx(e_dense["min_dist_au"], rel=1e-4)
        assert abs(e["time_days"] - e_dense["time_days"])*1440 < 0.5
        assert e["hill_radius_au"] == rh


### The Hill-radius gate on the encounter list ###

def testDistantPassageRefinesTheMinimumWithoutBeingAnEncounter(reb):
    """ A passage outside n_hill still refines the recorded minimum, but is not an encounter. """

    task = _flybyTask(30.0, 12.0, 2.0, "forward")
    task = dict(task, times=list(np.linspace(0.0, 4.0*DAY, 5)))
    diag = reb._integrateParticles(task)["diagnostics"]["obj"]

    # The flyby is well outside the 3 Hill radii threshold, so nothing is listed
    assert diag["encounters"] == []

    # The minimum is still refined below what the step ends alone would give
    d_dense, _ = _closestApproach(reb, task, np.linspace(0.0, 4.0*DAY, 3001))
    assert diag["min_dist_au"]["Mercury"] == pytest.approx(d_dense, rel=1e-6)
    assert diag["min_dist_au"]["Mercury"]/reb.HILL_RADII_AU["Mercury"] > 3.0


def _sunTask(q_au):
    """ An object released at 0.3 AU from the Sun on an orbit with perihelion q_au, reached ~13 days
    later. It starts far from the Earth, so the Sun is tracked from the start.
    """

    task = _flybyTask(30.0, 2.0, 2.0, "forward")
    v = np.sqrt(2/0.3 - 2/(0.3 + q_au))

    return dict(task, particle_states=[[-0.3, 0.0, 0.0, 0.0, -v, 0.0]],
                times=list(np.linspace(0.0, 20.0*DAY, 21)))


def testACloseSunPassageIsAnEncounter(reb):
    """ The Sun has no Hill radius, so its passage is listed by distance: inside SUN_ENCOUNTER_AU
    it is an encounter with no Hill radius, outside it only the closest approach is recorded.

    A body that has neither a Hill radius nor a distance threshold is still refused outright.
    """

    with pytest.raises(KeyError, match="No Hill radius"):
        reb._encounterRecord("Ceres", 0.5, 0.0)

    diag = reb._integrateParticles(_sunTask(0.05))["diagnostics"]["obj"]
    sun = [enc for enc in diag["encounters"] if enc["body"] == "Sun"]

    assert len(sun) == 1
    assert sun[0]["min_dist_au"] == pytest.approx(0.05, rel=1e-2)
    assert (sun[0]["hill_radius_au"] is None) and (sun[0]["n_hill"] is None)

    diag = reb._integrateParticles(_sunTask(0.15))["diagnostics"]["obj"]

    assert "Sun" not in [enc["body"] for enc in diag["encounters"]]
    assert diag["min_dist_au"]["Sun"] == pytest.approx(0.15, rel=1e-2)


def testSunMinimumIsNotLimitedByTheIntegratorStep(reb):
    """ The perihelion distance recorded with a single output over the whole span must match a dense
    re-integration, as for the planetary flybys: the Hermite refinement has to hold through the
    strongly curved motion at perihelion too. (Measured: 1e-6 down to q = 0.02 AU, 1e-4 at 1 R_Sun.)
    """

    task = _sunTask(0.02)

    diag_coarse = reb._integrateParticles(dict(task, times=[0.0, 20.0*DAY]))["diagnostics"]["obj"]
    diag_dense = reb._integrateParticles(dict(task, times=list(np.linspace(0.0, 20.0*DAY, 3001))))["diagnostics"]["obj"]

    assert diag_coarse["min_dist_au"]["Sun"] == pytest.approx(diag_dense["min_dist_au"]["Sun"], rel=1e-5)
    assert abs(diag_coarse["min_time_days"]["Sun"] - diag_dense["min_time_days"]["Sun"])*1440 < 0.5

    # Both runs list the same single passage
    for diag in (diag_coarse, diag_dense):
        sun = [enc for enc in diag["encounters"] if enc["body"] == "Sun"]
        assert len(sun) == 1
        assert sun[0]["min_dist_au"] == diag["min_dist_au"]["Sun"]


def testABackwardSunPassageIsListedAtNegativeTime(reb):
    """ In a backward run the Sun passage is listed at the (negative) time it happened, like the
    planetary flybys, and agrees with the tracked closest-approach time.
    """

    task = _sunTask(0.05)

    # Reverse the velocity, so the perihelion lies ~13 days in the past instead of the future
    state = task["particle_states"][0]
    task = dict(task, direction="backward", times=list(-np.linspace(0.0, 20.0*DAY, 21)),
                particle_states=[state[:3] + [-v for v in state[3:]]])

    diag = reb._integrateParticles(task)["diagnostics"]["obj"]
    sun = [enc for enc in diag["encounters"] if enc["body"] == "Sun"]

    assert len(sun) == 1
    assert sun[0]["time_days"] == pytest.approx(-13.4, abs=0.1)
    assert sun[0]["time_days"] == diag["min_time_days"]["Sun"]


@pytest.mark.parametrize("q_solar_radii", [0.5, 0.99])
def testAPerihelionInsideTheSunIsAnImpact(reb, q_solar_radii):
    """ The Sun has a physical radius, so an orbit that dives into it ends in an impact, whether it
    plunges deep or only grazes. The impact is recorded at the surface, and the aborted passage does
    not also appear as an encounter.
    """

    diag = reb._integrateParticles(_sunTask(q_solar_radii*reb.SUN_RADIUS_AU))["diagnostics"]["obj"]

    assert diag["impact"]["body"] == "Sun"
    assert diag["impact"]["dist_au"] == pytest.approx(reb.SUN_RADIUS_AU, rel=0.02)
    assert "Sun" not in [enc["body"] for enc in diag["encounters"]]


def testAPerihelionJustAboveTheSurfaceIsAnEncounterNotAnImpact(reb):
    """ One percent above the surface the object survives and the passage is listed. """

    diag = reb._integrateParticles(_sunTask(1.01*reb.SUN_RADIUS_AU))["diagnostics"]["obj"]
    sun = [enc for enc in diag["encounters"] if enc["body"] == "Sun"]

    assert diag["impact"] is None
    assert len(sun) == 1
    assert sun[0]["min_dist_au"] == pytest.approx(1.01*reb.SUN_RADIUS_AU, rel=1e-3)
    assert sun[0]["hill_radius_au"] is None


### Monte Carlo encounter summary ###

def _encounterEntry(body, dist_au):
    """ One encounter entry, in the format _integrateParticles produces. """

    hill = {"Mercury": 0.001475, "Venus": 0.006759}[body]

    return {"body": body, "min_dist_au": dist_au, "time_days": 0.0,
            "hill_radius_au": hill, "n_hill": dist_au/hill, "index": None}


def _cloneDiag(*encounters):
    """ A clone diagnostics dict holding the given (body, dist_au) encounters. """

    return {"encounters": [_encounterEntry(b, d) for b, d in encounters]}


def testCloneSummaryCountsClonesAndPassagesSeparately(reb):
    """ "count" is the clones that met a body, "n_encounters" is how many passages they made. """

    clone_diag = {
        "mc_0": _cloneDiag(("Mercury", 0.002), ("Mercury", 0.003)),
        "mc_1": _cloneDiag(("Mercury", 0.001)),
        "mc_2": _cloneDiag(("Venus", 0.01)),
        "mc_3": {"encounters": []},
    }

    summary = reb.cloneEncounterSummary(clone_diag)

    assert summary["Mercury"]["count"] == 2, "two clones met Mercury, one of them twice"
    assert summary["Mercury"]["n_encounters"] == 3
    assert summary["Mercury"]["closest_au"] == pytest.approx(0.001)

    assert summary["Venus"]["count"] == 1
    assert summary["Venus"]["n_encounters"] == 1


def testCloneSummaryIsEmptyWithoutEncounters(reb):
    """ Clones that met nothing, or diagnostics without the key, produce no entries. """

    assert reb.cloneEncounterSummary({}) == {}
    assert reb.cloneEncounterSummary({"mc_0": {}, "mc_1": {"encounters": []}}) == {}
