""" Tests for the clones of a meteor's orbit (wmpl.Utils.OrbitClones): that they are the draws of
wmpl.Rebound.REBOUND, that the nominal state gives the orbit of the trajectory again, and how the parent
body search and the orbital classification are summarised over them.

The trajectory is made here, from a state vector, a radiant and a covariance, and the search runs on a
small catalogue, so that the tests need neither a solved pickle nor the catalogue files.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_OrbitClones.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_OrbitClones
"""

import csv
import json
import os
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

import wmpl.Utils.ParentBodySearch as pbs
from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Utils import OrbitClones as oc
from wmpl.Utils.TrajConversions import geo2Cartesian, raDec2ECI


# The reference time of the trajectory made here, and the speed it starts with
JD_REF = 2458907.896206816
V_INIT = 22000.0


def _trajectory(los_success=True, cov=None):
    """ A trajectory as the solver saves it, with only what the clones read: the state vector at the
        beginning (80 km above 45.8 N, 15.2 E), the radiant, the speeds, the orbit computed as the solver
        computes it, the covariance of the state vector and the solver's uncertainties. """

    position = np.array(geo2Cartesian(np.radians(45.8), np.radians(15.2), 80000.0, JD_REF))
    radiant = np.array(raDec2ECI(np.radians(200.0), np.radians(10.0)))
    v_avg = V_INIT - 500.0

    orbit = calcOrbit(radiant, V_INIT, v_avg, position, JD_REF, stations_fixed=(not los_success),
        reference_init=los_success)

    if cov is None:
        cov = np.diag([100.0**2]*3 + [30.0**2]*3)

    return SimpleNamespace(state_vect_mini=position, radiant_eci_mini=radiant, v_init=V_INIT, v_avg=v_avg,
        rbeg_jd=JD_REF, jdt_ref=JD_REF, orbit=orbit, orbit_los_success=los_success, v_init_stddev=None,
        state_vect_cov=cov, uncertainties=SimpleNamespace(q=1e-3, e=1e-3, i=1e-4, peri=1e-3, node=1e-6))


def _catalogue():
    """ A small catalogue in the format of the loaders of wmpl.Utils.ParentBodySearch: name, q, e, i,
        peri, node (degrees), H, epoch JD, mean anomaly and time of perihelion. """

    return [
        ["Phaethon", 0.1400, 0.8900, 22.25, 322.2, 265.2, 14.4, 2461200.5, 100.0, None],
        ["Near twin", 0.1500, 0.8800, 22.00, 320.0, 263.0, 20.0, 2461200.5, 10.0, None],
        ["Apollo", 0.8000, 0.5000, 10.00, 60.0, 120.0, 18.0, 2461200.5, 200.0, None],
        ["Never reaches Earth", 1.8000, 0.2000, 5.00, 30.0, 40.0, 17.0, 2461200.5, 5.0, None],
        ["Parabolic comet", 0.9000, 1.0000, 70.00, 100.0, 200.0, -1, 2451800.5, None, 2451820.5],
        ]


def _search(orbit, keys, geocentric=None, heliocentric=None, top_n=-1):
    """ The search of the small catalogue for an orbit (q, e, incl, node, peri, radians). """

    q, e, incl, node, peri = orbit

    saved = (pbs.comets_elements, pbs.asteroids_elements)
    pbs.comets_elements, pbs.asteroids_elements = [], _catalogue()

    try:
        return pbs.searchParentBodies(q, e, incl, peri, node, d_crit=keys, top_n=top_n,
            geocentric=geocentric, heliocentric=heliocentric)

    finally:
        pbs.comets_elements, pbs.asteroids_elements = saved


def _clonesOf(orbits, geocentric=None, heliocentric=None, seed=0):
    """ A set of clones as cloneOrbits returns it, with the given orbits (q, e, incl, node, peri). """

    orbits = np.array(orbits, dtype=float)
    n = len(orbits)

    clones = {'n': n, 'seed': seed, 'valid': np.all(np.isfinite(orbits), axis=1),
        'geocentric': geocentric or [None]*n, 'heliocentric': heliocentric or [None]*n,
        'spread': {}, 'solver_sigma': {}, 'nominal_offset_sigma': 0.0}

    for k, key in enumerate(('q', 'e', 'incl', 'node', 'peri')):
        clones[key] = orbits[:, k]

    return clones


def _bodyOrbit(body):
    """ The orbit of a body of the catalogue as (q, e, incl, node, peri), radians. """

    return (body[1], body[2], np.radians(body[3]), np.radians(body[5]), np.radians(body[4]))


def test_clones_are_the_draws_of_rebound():
    """ The clones are drawn as wmpl.Rebound.REBOUND draws its realizations: one multivariate normal draw
        per clone from a generator seeded with the seed, so clone k does not depend on how many are drawn.
    """

    traj = _trajectory()
    states = oc.cloneStateVectors(traj, 10, random_seed=1234)

    # The loop of REBOUND.py's reboundSimulate
    state_vect = list(traj.state_vect_mini) + list(traj.v_init*traj.radiant_eci_mini)
    rng = np.random.default_rng(1234)
    expected = np.array([rng.multivariate_normal(state_vect, traj.state_vect_cov) for _ in range(10)])

    assert np.array_equal(states, expected)
    assert np.array_equal(oc.cloneStateVectors(traj, 4, random_seed=1234), states[:4])

    # Where REBOUND.py has the draw as a function of its own (from #119), the two are the same
    from wmpl.Rebound import REBOUND

    if hasattr(REBOUND, 'sampleStateVectors'):
        assert np.array_equal(np.array(REBOUND.sampleStateVectors(traj, 10, random_seed=1234)), states)


def test_nominal_state_gives_the_orbit_of_the_trajectory():
    """ With the convention the solver used, moving stations or fixed ones, the nominal state vector
        gives the trajectory's orbit again; with the other convention it would not. """

    for los_success in (True, False):

        traj = _trajectory(los_success=los_success)
        orbit = oc.orbitOfStateVector(traj, np.concatenate([traj.state_vect_mini,
            traj.v_init*traj.radiant_eci_mini]))

        for name in ('q', 'e', 'i', 'node', 'peri', 'ra_g', 'dec_g', 'v_g'):
            assert abs(getattr(orbit, name) - getattr(traj.orbit, name)) < 1e-12*max(1.0,
                abs(getattr(traj.orbit, name))), (los_success, name)

    traj = _trajectory(los_success=False)
    traj.orbit_los_success = True
    orbit = oc.orbitOfStateVector(traj, np.concatenate([traj.state_vect_mini,
        traj.v_init*traj.radiant_eci_mini]))
    assert abs(orbit.q - traj.orbit.q) > 1e-6


def test_clone_orbits_spread_and_stay_centred():
    """ Without spread every clone has the nominal orbit; with it they spread, around the nominal one,
        and the recomputed nominal orbit is where the trajectory's is. """

    traj = _trajectory(cov=np.zeros((6, 6)))
    clones = oc.cloneOrbits(traj, 5, random_seed=3)

    assert clones['seed'] == 3 and clones['valid'].all()
    assert np.allclose(clones['q'], traj.orbit.q, rtol=0, atol=1e-12)
    assert np.allclose(clones['peri'], traj.orbit.peri, rtol=0, atol=1e-12)
    assert clones['nominal_offset_sigma'] < 1e-6
    assert all(v < 1e-12 for v in clones['spread'].values())

    traj = _trajectory()
    clones = oc.cloneOrbits(traj, 400, random_seed=3)

    assert clones['spread']['q'] > 1e-5
    assert abs(np.median(clones['q']) - traj.orbit.q) < 0.2*clones['spread']['q']
    assert abs(np.median(oc._wrappedDifference(clones['node'], traj.orbit.node))) \
        < 0.2*clones['spread']['node']

    # A seed is drawn and returned if none is given, and gives the same clones again
    first = oc.cloneOrbits(traj, 3)
    again = oc.cloneOrbits(traj, 3, random_seed=first['seed'])
    assert np.array_equal(first['q'], again['q'])


def test_a_trajectory_without_monte_carlo_has_no_clones():

    traj = _trajectory()
    traj.uncertainties = None

    try:
        oc.cloneOrbits(traj, 10)

    except ValueError as err:
        assert "Monte Carlo" in str(err)

    else:
        raise AssertionError("No error without uncertainties")


def test_the_nominal_as_a_clone_gives_the_values_of_the_search():
    """ A clone with the nominal orbit gets, from every kind of criterion, the values the search gives,
        and the ranks of the search; with a criterion on the radiant first too, over the whole
        catalogue. """

    phaethon = _bodyOrbit(_catalogue()[0])
    orbit = (0.145, 0.885, np.radians(22.1), np.radians(264.0), np.radians(321.0))

    for keys in (['DSH', 'DD', 'DN', 'DVVIDA', 'MOID'], ['DN', 'DSH'], ['MOID', 'DH']):

        search = _search(orbit, keys)
        clones = _clonesOf([orbit, orbit])
        stats = oc.cloneSearch(search, _catalogue(), clones)

        for c in range(len(keys)):
            for m, match in enumerate(search['matches']):

                expected = match['d_values'][c]
                got = stats['values'][c, 0, m]

                if np.isfinite(expected):
                    assert abs(got - expected) < 1e-12, (keys, c, match['name'])
                else:
                    assert not np.isfinite(got)

        n_ranked = sum(np.isfinite(m['d_values'][0]) for m in search['matches'])
        assert list(stats['ranks'][0, :n_ranked]) == list(range(1, n_ranked + 1))
        assert stats['best_fraction'][0] == 1.0
        assert stats['best_bodies'] == [(search['matches'][0]['name'], 1.0)]

    # The orbit of a body is at D = 0 from it
    stats = oc.cloneSearch(_search(phaethon, ['DSH']), _catalogue(), _clonesOf([phaethon]*2))
    assert stats['interval'][0, 0, 1] < 1e-12


def test_closest_bodies_thresholds_and_ranks_over_the_clones():
    """ Clones on the orbits of different bodies: each is the closest body for its clones, the share
        under a threshold and the ranks follow, and a clone without an orbit is left out. """

    catalogue = _catalogue()
    phaethon, twin, apollo = [_bodyOrbit(b) for b in catalogue[:3]]

    search = _search(phaethon, ['DSH', 'DD'], top_n=3)
    names = [m['name'] for m in search['matches']]
    assert names[:2] == ["Phaethon", "Near twin"]

    clones = _clonesOf([phaethon, phaethon, twin, apollo, [np.nan]*5])
    stats = oc.cloneSearch(search, catalogue, clones, dmax=[0.1])

    assert stats['n'] == 4 and stats['n_drawn'] == 5
    assert dict(stats['best_bodies']) == {"Phaethon": 0.5, "Near twin": 0.25, "Apollo": 0.25}

    m_phaethon, m_twin = names.index("Phaethon"), names.index("Near twin")
    assert stats['best_fraction'][m_phaethon] == 0.5
    assert stats['best_fraction'][m_twin] == 0.25

    # Phaethon is first for its two clones, second for the twin's, and further for the Apollo's
    assert list(stats['ranks'][:3, m_phaethon]) == [1, 1, 2]
    assert stats['ranks'][3, m_phaethon] > 2
    assert np.isnan(stats['ranks'][4, m_phaethon])

    # Under D_SH = 0.1: Phaethon's clones and the twin's (D_SH 0.069), not the Apollo's
    d = stats['values'][0, :4, m_phaethon]
    assert stats['below'][0, m_phaethon] == np.mean(d < 0.1) == 0.75
    assert np.isnan(stats['below'][1, m_phaethon])

    assert np.allclose(stats['interval'][0, m_phaethon], np.percentile(d, oc.CLONE_INTERVAL))

    try:
        oc.cloneSearch(search, catalogue, clones, dmax=[0.1, 0.1, 0.1])
    except ValueError:
        pass
    else:
        raise AssertionError("More thresholds than criteria were taken")


def test_report_and_saved_files_hold_the_clones():
    """ The report names the closest bodies and gives each match its interval, threshold share, closest
        share and rank, and the CSV and JSON files of the search hold them. """

    catalogue = _catalogue()
    phaethon, twin, apollo = [_bodyOrbit(b) for b in catalogue[:3]]

    search = _search(phaethon, ['DSH', 'DD'], top_n=3)
    clones = _clonesOf([phaethon, phaethon, twin, apollo], seed=42)
    stats = oc.cloneSearch(search, catalogue, clones, dmax=[0.1])

    text = "\n".join(oc.formatCloneSearch(stats, search))

    assert "Closest body by D_SH, share of the clones: Phaethon 50.0%" in text
    assert "D_SH < 0.1" in text and "D_D [95% of the clones]" in text
    assert "D_D < " not in text
    assert "no estimate" in text

    with tempfile.TemporaryDirectory() as dir_path:

        json_path = os.path.join(dir_path, "matches.json")
        pbs.saveMatches(json_path, search, clones=stats)

        with open(json_path) as f:
            data = json.load(f)

        assert data['clones']['n'] == 4 and data['clones']['seed'] == 42
        assert data['clones']['closest'][0] == {'name': "Phaethon", 'fraction': 0.5}
        assert data['clones']['dmax'] == {'D_SH': 0.1, 'D_D': None}

        first = data['matches'][0]
        assert first['clones']['closest_fraction'] == 0.5
        assert first['clones']['criteria']['D_SH']['below_dmax'] == 0.75
        assert 'below_dmax' not in first['clones']['criteria']['D_D']
        assert not any(key.startswith('clones_') for key in first)

        csv_path = os.path.join(dir_path, "matches.csv")
        pbs.saveMatches(csv_path, search, clones=stats)

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]['clones_closest_fraction']) == 0.5
        assert float(rows[0]['clones_D_SH_below_dmax']) == 0.75
        assert 'clones_D_D_2.5' in rows[0] and 'clones_rank_97.5' in rows[0]


def test_classes_of_the_clones():
    """ A bound clone gets every class the command line gives; an unbound one is a class of its own and
        has no dynamical parameters. """

    traj = _trajectory()
    o = traj.orbit
    assert o.e < 1

    nominal = oc.classifyOrbit(o.q, o.e, o.i, o.node, o.peri)
    keys = [key for _, key, _, _ in nominal]

    assert keys[:5] == ['T_J', 'K', 'P', 'Q', 'E']
    assert 'BOROVICKA_CLASS' in keys and 'TANCREDI_ASTEROID' in keys and 'MOID_JUPITER' in keys
    assert 'TANCREDI_ASTEROID' not in [key for _, key, _, _ in oc.classifyOrbit(o.q, o.e, o.i)]

    orbit = (o.q, o.e, o.i, o.node, o.peri)
    clones = _clonesOf([orbit, orbit, orbit, (o.q, 1.2, o.i, o.node, o.peri)], seed=5)
    summary = oc.cloneClasses(clones)

    assert summary['n'] == 4 and summary['n_unbound'] == 1

    items = dict((key, s) for _, key, _, s in summary['items'])
    t_j = dict((key, value) for _, key, _, value in nominal)['T_J']

    assert np.allclose(items['T_J']['interval'], [t_j, t_j])
    assert dict(items['BOROVICKA_CLASS']['fractions']) == {
        dict((key, value) for _, key, _, value in nominal)['BOROVICKA_CLASS']: 0.75, oc.UNBOUND: 0.25}

    lines = oc.formatCloneClasses(summary, nominal, quiet=True)
    assert "CLONES 4" in lines and "CLONES_SEED 5" in lines and "CLONES_UNBOUND 1" in lines
    assert any(line.startswith("T_J_CLONES_95 ") for line in lines)
    assert any(line.startswith("BOROVICKA_CLASS_CLONES ") and "unbound: 0.2500" in line for line in lines)

    text = "\n".join(oc.formatCloneClasses(summary, nominal))
    assert "1 clones are unbound" in text and "Tisserand parameter T_J" in text


def test_command_line_of_the_classification_with_clones():
    """ The command line of wmpl.Utils.OrbitClassification classifies the clones of a pickle, and
        classifyOrbit gives exactly what it prints for the nominal orbit. """

    traj = _trajectory()

    with tempfile.TemporaryDirectory() as dir_path:

        path = os.path.join(dir_path, "traj.pickle")
        with open(path, 'wb') as f:
            pickle.dump(traj, f)

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__)))))
        env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))

        out = subprocess.run([sys.executable, "-m", "wmpl.Utils.OrbitClassification", path, "--quiet",
            "--clones", "20", "--seed", "8"], capture_output=True, text=True, env=env, cwd=dir_path)

        assert out.returncode == 0, out.stderr

        printed = dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)

        o = traj.orbit
        for _, key, _, value in oc.classifyOrbit(o.q, o.e, o.i, o.node, o.peri):

            if isinstance(value, str):
                assert printed[key] == value, key
            else:
                assert abs(float(printed[key]) - value) <= 1e-4*max(1.0, abs(value)), key

        assert printed['CLONES'] == "20" and printed['CLONES_SEED'] == "8"
        assert 'T_J_CLONES_95' in printed and 'NEO_CLASS_CLONES' in printed

        # Without the Monte Carlo there are no clones, and it says so
        traj.uncertainties = None
        with open(path, 'wb') as f:
            pickle.dump(traj, f)

        out = subprocess.run([sys.executable, "-m", "wmpl.Utils.OrbitClassification", path, "--clones",
            "20"], capture_output=True, text=True, env=env, cwd=dir_path)

        assert out.returncode == 1 and "Monte Carlo" in out.stderr


if __name__ == "__main__":

    test_functions = [obj for name, obj in sorted(globals().items()) \
        if name.startswith("test_") and callable(obj)]

    failed = 0

    for test_func in test_functions:

        try:
            test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {!r}".format(test_func.__name__, e))

    print()

    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
