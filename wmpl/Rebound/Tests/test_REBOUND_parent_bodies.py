""" Tests for integrating the parent body candidates of a ParentBodySearch alongside the meteoroid.

The bookkeeping (output times, the D criteria along a run, the summaries) is tested with stand-in
outputs and runs without REBOUND. The rest integrates for real and is skipped when rebound/reboundx,
an attached REBOUNDx, the DE430 ephemeris or the example trajectory are unavailable.
"""

import json
import os
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from wmpl.Utils.Dcriteria import lookupCriterion, calcCriterionFromElements


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dynesty",
                           "examples", "20191023_091225")
EXAMPLE_FILE = "20191023_091225_trajectory.pickle"


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_parent_bodies", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def realReb(reb):
    """ The REBOUND module, skipping the test if it cannot integrate: rebound/reboundx missing, a
        REBOUNDx that does not attach to the installed REBOUND, or no DE430 ephemeris. """

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable: {}".format(reb._REBOUND_IMPORT_ERROR))

    try:
        reb.checkReboundxUsable()
    except RuntimeError as e:
        pytest.skip(str(e).split(".")[0])

    if not os.path.isfile(reb.config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not available")

    return reb


def _orb(q, e, incl=0.1, node=1.0, peri=2.0):
    """ A stand-in for the orbit of an output row, in the units REBOUND gives. """

    return SimpleNamespace(a=q/(1.0 - e), e=e, inc=incl, Omega=node, omega=peri, f=0.0)


def _row(t, orb, earth_dist):
    """ A stand-in output row of _integrateParticles: [time, state, orbit, body distances]. """

    return [t, [0.0]*6, orb, {"Earth": earth_dist}]


### Bookkeeping, no REBOUND needed ###


def testOutputTimesAreTheOnesReboundSimulateUsed(reb):
    """ The refactored output times are the ones reboundSimulate used to compute inline, in the time
    unit of the simulations (DAYS_PER_SIM_TIME), whatever it is. """

    for direction, sign in (("forward", 1.0), ("backward", -1.0)):

        times = reb.outputTimes(3652.5, 200, direction)
        expected = np.linspace(0, sign*reb.daysToSimTime(3652.5), 200)

        assert np.allclose(times, expected, rtol=0, atol=1e-12)


def testCriteriaAreLeftOutUntilTheMeteoroidLeavesTheEarth(reb):
    """ Before the departure the meteoroid's osculating orbit carries the Earth's pull, so nothing is
        compared there; after it the values are those of Dcriteria on the two orbits. """

    hill = reb.HILL_RADII_AU["Earth"]
    times = np.linspace(0, -1.0, 6)

    meteoroid = [_row(t, _orb(0.9 + 0.01*k, 0.6), (k + 0.5)*hill*1.5) for k, t in enumerate(times)]
    body = [_row(t, _orb(0.95, 0.62, peri=2.05), 1.0) for t in times]

    criteria = [lookupCriterion("DSH"), lookupCriterion("DN")]
    t_days, values = reb.parentBodyCriteria(meteoroid, body, criteria)

    departure = reb.findEarthDepartureIndex(meteoroid)
    assert departure == 2

    assert np.all(np.isnan(values[:, :departure]))

    for j in range(departure, len(times)):
        for c, criterion in enumerate(criteria):

            m, b = meteoroid[j][2], body[j][2]
            expected = calcCriterionFromElements(criterion,
                (m.a*(1 - m.e), m.e, m.inc, m.Omega, m.omega), (b.a*(1 - b.e), b.e, b.inc, b.Omega, b.omega))

            assert values[c, j] == pytest.approx(expected, abs=1e-12)

    assert np.allclose(t_days, times*reb.SIM_TIME_DAYS)



def testTheMoidIsEvaluatedOverAllOutputsAtOnce(reb):
    """ The MOID takes arrays (Dcriteria.VECTORISED_CRITERIA), so it is evaluated over every output in
        one call; the values are those of one call per output, alongside a criterion that is not. """

    hill = reb.HILL_RADII_AU["Earth"]
    times = np.linspace(0, -1.0, 8)

    meteoroid = [_row(t, _orb(0.9 + 0.01*k, 0.6, peri=2.0 + 0.01*k), (k + 0.5)*hill*1.5)
                 for k, t in enumerate(times)]
    body = [_row(t, _orb(0.95, 0.62, peri=2.05 - 0.005*k), 1.0) for k, t in enumerate(times)]

    criteria = [lookupCriterion("DSH"), lookupCriterion("MOID")]
    _, values = reb.parentBodyCriteria(meteoroid, body, criteria)

    departure = reb.findEarthDepartureIndex(meteoroid)
    assert np.all(np.isnan(values[:, :departure]))

    for j in range(departure, len(times)):
        m, b = meteoroid[j][2], body[j][2]
        orbit_m = (m.a*(1 - m.e), m.e, m.inc, m.Omega, m.omega)
        orbit_b = (b.a*(1 - b.e), b.e, b.inc, b.Omega, b.omega)
        for c, criterion in enumerate(criteria):
            assert values[c, j] == pytest.approx(calcCriterionFromElements(criterion, orbit_m, orbit_b),
                                                 abs=1e-12)

def testSummaryStartsAtTheDepartureAndBandsTheClones(reb):

    t = np.array([0.0, -1.0, -2.0, -3.0])
    nominal = np.array([np.nan, 0.30, 0.10, 0.20])

    clones = [(t, np.array([np.nan, 0.3, 0.2, v])) for v in np.linspace(0.1, 0.5, 41)]

    sm = reb.summariseParentBodyCriteria((t, nominal), clones)

    assert sm["departure"] == 0.30
    assert sm["departure_time_days"] == -1.0
    assert sm["end"] == 0.20
    assert sm["min"] == 0.10 and sm["min_time_days"] == -2.0

    median, low, high = sm["clones_end"]
    assert median == pytest.approx(0.3)
    assert low == pytest.approx(np.percentile(np.linspace(0.1, 0.5, 41), 2.5))
    assert high == pytest.approx(np.percentile(np.linspace(0.1, 0.5, 41), 97.5))

    assert reb.summariseParentBodyCriteria((t, nominal), [])["clones_end"] is None


def testPlotIsWritten(reb, tmp_path):

    t = np.linspace(0, -100, 20)
    nominal = np.vstack([np.linspace(0.1, 0.2, 20), np.linspace(0.3, 0.2, 20)])
    clones = np.array([nominal + 0.01*k for k in range(-3, 4)])

    path = str(tmp_path/"bodies.png")
    reb.plotParentBodyCriteria([("#1 A", t, nominal, clones), ("#2 B", t, nominal + 0.05, None)],
                               ["D_SH", "D_N"], path, title="test")

    assert os.path.getsize(path) > 1000


@pytest.mark.parametrize("q, e", [(0.5, 0.7), (0.3, 1.0), (0.674, 1.0011), (1.2, 2.5)])
def testStateFromPerihelionTimeFollowsTheTwoBodyMotion(reb, q, e):
    """ For every conic, the state from the time of perihelion is at q at perihelion, and moves as a
        two-body integration moves it. """

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound not importable")

    tp, t1, t2 = 2451545.0, 2451545.0 - 300.0, 2451545.0 + 500.0
    days = reb.SIM_TIME_DAYS

    at_peri = reb.heliocentricStateFromPerihelionTime(q, e, 30.0, 40.0, 50.0, tp, tp)
    assert np.linalg.norm(at_peri[:3]) == pytest.approx(q, abs=1e-14)

    s1 = reb.heliocentricStateFromPerihelionTime(q, e, 30.0, 40.0, 50.0, tp, t1)
    s2 = reb.heliocentricStateFromPerihelionTime(q, e, 30.0, 40.0, 50.0, tp, t2)

    sim = reb.rb.Simulation()
    sim.add(m=1.0)
    sim.add(x=s1[0], y=s1[1], z=s1[2], vx=s1[3]*days, vy=s1[4]*days, vz=s1[5]*days)
    sim.integrate((t2 - t1)/days)

    p = sim.particles[1]
    assert np.allclose([p.x, p.y, p.z], s2[:3], atol=1e-12)
    assert np.allclose([p.vx/days, p.vy/days, p.vz/days], s2[3:], atol=1e-14)


def testEllipticStateFromPerihelionTimeIsTheOneFromTheMeanAnomaly(reb):

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound not importable")

    q, e, tp, t = 0.8, 0.5, 2451545.0, 2451545.0 + 123.4
    a = q/(1 - e)
    mean_anomaly = np.degrees(((t - tp)/reb.SIM_TIME_DAYS/a**1.5) % (2*np.pi))

    assert np.allclose(reb.heliocentricStateFromPerihelionTime(q, e, 30, 40, 50, tp, t),
                       reb.heliocentricStateFromElements(q, e, 30, 40, 50, mean_anomaly), atol=1e-14)


### Real integrations ###


def testStateFromElementsIsTheKeplerOrbit(realReb):
    """ The state REBOUND gives for the elements is the one of the Kepler orbit, here at perihelion. """

    q, e = 0.8, 0.5
    state = realReb.heliocentricStateFromElements(q, e, 0.0, 0.0, 0.0, 0.0)

    v_peri = np.sqrt((1 + e)/q)/realReb.SIM_TIME_DAYS

    assert np.allclose(state, [q, 0, 0, 0, v_peri, 0], atol=1e-12)


def testBodiesAtTheirOwnEpochAreNotMovedAndUnusableOnesAreSkipped(realReb):

    body = {"q": 0.8, "e": 0.5, "incl": 10.0, "peri": 60.0, "node": 120.0, "mean_anomaly": 200.0,
            "epoch_jd": 2459000.5}

    out = realReb.propagateBodiesToEpoch([body, dict(body, e=1.0), dict(body, mean_anomaly=None),
                                          dict(body, epoch_jd=None)], 2459000.5)

    # A parabolic orbit has no mean anomaly, and without a time of perihelion it cannot be placed
    assert out[1] is None and out[2] is None and out[3] is None

    rec = out[0]
    for key in ("q", "e", "incl", "peri", "node", "mean_anomaly"):
        assert rec[key] == pytest.approx(body[key], abs=1e-9)


def testParabolicAndHyperbolicCometsArePropagatedFromTheirPerihelionTime(realReb):
    """ Comets with no mean anomaly are placed by their time of perihelion and propagated. At their
        own epoch they come back with their catalogue elements, q from the angular momentum included. """

    jd = 2459000.5
    comets = [{"q": 0.9, "e": 1.0, "incl": 70.0, "peri": 100.0, "node": 200.0, "mean_anomaly": None,
               "perihelion_jd": jd + 20.0, "epoch_jd": jd},
              {"q": 0.674, "e": 1.0011, "incl": 160.0, "peri": 47.1, "node": 338.9, "mean_anomaly": None,
               "perihelion_jd": jd - 40.0, "epoch_jd": jd}]

    here = realReb.propagateBodiesToEpoch(comets, jd)

    for comet, rec in zip(comets, here):
        assert rec["q"] == pytest.approx(comet["q"], abs=1e-10)
        assert rec["e"] == pytest.approx(comet["e"], abs=1e-10)
        assert rec["mean_anomaly"] is None

    # A year later they have moved, and are still on (nearly) the same conics
    later = realReb.propagateBodiesToEpoch(comets, jd + 365.25)

    for k, (comet, rec) in enumerate(zip(comets, later)):
        assert rec is not None
        assert rec["q"] == pytest.approx(comet["q"], abs=0.05)
        assert not np.allclose(rec["position"], here[k]["position"], atol=0.1)


def testMatchesFileIsReadAtTheEpochOrPropagated(realReb, tmp_path):
    """ A body saved at this epoch is taken as it is, and one saved at another epoch is propagated. """

    jd = 2459000.5
    state_here = realReb.heliocentricStateFromElements(0.8, 0.5, 10.0, 120.0, 60.0, 200.0)

    data = {
        "criteria": ["D_SH", "D_N"], "criteria_keys": ["DSH", "DN"],
        "matches": [
            {"rank": 1, "name": "Saved here", "q_AU": 9.0, "e": 0.1, "incl_deg": 0.0, "peri_deg": 0.0,
             "node_deg": 0.0, "catalogue_epoch_jd": None, "catalogue_mean_anomaly_deg": None,
             "criteria": {"D_SH": 0.1, "D_N": None},
             "at_epoch": {"jd_tdb": jd, "position": state_here[:3], "velocity": state_here[3:]}},
            {"rank": 2, "name": "From the catalogue", "q_AU": 0.8, "e": 0.5, "incl_deg": 10.0,
             "peri_deg": 60.0, "node_deg": 120.0, "catalogue_epoch_jd": jd + 365.25,
             "catalogue_mean_anomaly_deg": 200.0, "criteria": {"D_SH": 0.2, "D_N": 0.3},
             "at_epoch": None},
            {"rank": 3, "name": "Parabolic", "q_AU": 0.8, "e": 1.0, "incl_deg": 10.0, "peri_deg": 60.0,
             "node_deg": 120.0, "catalogue_epoch_jd": jd, "catalogue_mean_anomaly_deg": None,
             "criteria": {"D_SH": 0.4, "D_N": 0.5}, "at_epoch": None},
        ]}

    path = str(tmp_path/"matches.json")
    with open(path, "w") as f:
        json.dump(data, f)

    bodies, keys, skipped = realReb.loadParentBodies(path, jd)

    assert keys == ["DSH", "DN"]
    assert skipped == ["Parabolic"]
    assert [b["name"] for b in bodies] == ["Saved here", "From the catalogue"]

    # Taken as saved, not recomputed from its (deliberately wrong) catalogue elements
    assert np.allclose(bodies[0]["state"], state_here, atol=0)

    # Propagated a year back: on the same orbit to first order, but not at the same place
    assert not np.allclose(bodies[1]["state"], state_here, atol=1e-3)
    assert bodies[1]["catalogue_criteria"] == {"D_SH": 0.2, "D_N": 0.3}

    # Only the first one
    assert [b["name"] for b in realReb.loadParentBodies(path, jd, top_n=1)[0]] == ["Saved here"]


def testBodiesRunOverTheSameTimesAsTheMeteoroid(realReb):
    """ End to end on the example trajectory: the meteoroid and a body on its own orbit, integrated
        over the same output times, stay close in D_SH once the meteoroid has left the Earth. """

    pickle_path = os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)
    if not os.path.isfile(pickle_path):
        pytest.skip("example trajectory not available")

    from wmpl.Utils.Pickling import loadPickle
    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)

    days, n_out = 30.0, 31
    outputs, _ = realReb.reboundSimulate(None, None, traj=traj, direction="backward", sim_days=days,
                                         n_outputs=n_out, mc_runs=0, show_progress=False)

    # A body on the meteoroid's own catalogue orbit, at the reference epoch
    o = traj.orbit
    mean_anomaly = np.degrees(o.M) if getattr(o, "M", None) is not None else 0.0
    state = realReb.heliocentricStateFromElements(o.q, o.e, np.degrees(o.i), np.degrees(o.node),
                                                  np.degrees(o.peri), mean_anomaly)

    jd_tdb = realReb.utcToTdb(traj.jdt_ref)
    run = realReb.integrateHeliocentricBodies(["twin"], [state], jd_tdb,
                                              realReb.outputTimes(days, n_out, "backward"))

    rows = run["outputs"]["twin"]
    assert len(rows) == n_out
    assert [r[0] for r in rows] == pytest.approx([r[0] for r in outputs])

    t, values = realReb.parentBodyCriteria(outputs, rows, [lookupCriterion("DSH")])

    compared = values[0][~np.isnan(values[0])]
    assert len(compared) > 0

    # The orbit of the trajectory solution removes the Earth's attraction, and so does the departure,
    #   so the two orbits agree to the precision of that correction
    assert np.all(compared < 0.05)


def testParentBodiesAreIntegratedWithTheMeteoroidForces(reb, monkeypatch):
    """ The candidates get the same forces as the meteoroid: the Earth's J2/J4 about its pole of date,
        and the Sun's J2 and the GR model as chosen for the run. """

    tasks = []
    monkeypatch.setattr(reb, "_integrateParticles", lambda task: tasks.append(task) or {})

    jd = 2459274.41
    planets = (["Sun", "Earth"], [[0.0]*6, [1.0, 0, 0, 0, 1.0, 0]], [1.0, 3.0e-6])
    reb.integrateHeliocentricBodies(["body"], [[1.5, 0, 0, 0, 0.0136, 0]], jd, [0.0, -1.0],
                                    planets=planets, sun_j2=False, gr="gr")

    task = tasks[0]
    assert task["sun_j2"] is False
    assert task["gr"] == "gr"
    if reb.harmonicsSpinAxisSupported():
        assert np.allclose(task["earth_spin_axis"], reb.earthSpinAxisEcliptic(jd))
