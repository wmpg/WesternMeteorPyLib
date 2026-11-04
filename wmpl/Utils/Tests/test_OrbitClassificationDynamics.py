""" Tests for the Jupiter-family dynamics metrics of an integrated ensemble of clones.

The metrics are checked on synthetic ensembles whose answer is known by construction: series that
change only at the outputs, clones that end early or leave the solar system, and encounter
sequences placed on either side of a crossing. The binomial interval is checked against its closed
form at k = 0 and against its symmetry.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_OrbitClassificationDynamics.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_OrbitClassificationDynamics
"""

import os
import math

import numpy as np

from types import SimpleNamespace

import wmpl.Rebound.REBOUND as reb
from wmpl.Utils.OrbitClassification import (calcKozaiEnergy, calcExcursionFractions,
    countPerihelionReach, splitPerihelionReachByEncounter, calcBinomialInterval,
    classifyPerihelionReach, classifyJupiterFamilyDynamics, cloneEnsemble, formatJupiterFamilyDynamics,
    classifyJupiterFamilyDynamicsShober2024, elementUncertainties, drawElementClones,
    elementCloneStates, addEarthAttraction, SHOBER2024_DEFAULT_SIGMA, SHOBER2024_START_HILL,
    JFC_DYNAMICS_Q_LIMIT, JFC_DYNAMICS_A_LIMIT)


# Tests that integrate with REBOUND itself, skipped where it cannot be imported (it has no wheels for
#   Linux on ARM64). The standalone runner skips the same ones.
_REBOUND_TESTS = set()

def _needsRebound(test_func):
    _REBOUND_TESTS.add(test_func.__name__)

    try:
        import pytest

    except ImportError:
        return test_func

    return pytest.mark.skipif(not reb.REBOUND_FOUND, reason="rebound/reboundx not importable")(test_func)


# Ten one-year intervals of a backward integration
T_BACKWARD = -np.arange(11, dtype=float)

Q_LOW = 1.0
Q_HIGH = JFC_DYNAMICS_Q_LIMIT + 0.5
A_LOW = 3.0
A_HIGH = JFC_DYNAMICS_A_LIMIT + 1.0


def _raises(exception, func, *args, **kwargs):
    """ Return True if calling func raises the given exception. """

    try:
        func(*args, **kwargs)

    except exception:
        return True

    return False


def test_excursion_fractions_are_exact_for_series_that_change_at_the_outputs():

    # Beyond both limits for the first 3 of 10 intervals, then within them
    q = np.full(11, Q_LOW)
    q[:3] = Q_HIGH
    a = np.full(11, A_LOW)
    a[:3] = A_HIGH

    f_q, f_a = calcExcursionFractions(T_BACKWARD, q, a)
    assert f_q == 0.3
    assert f_a == 0.3

    # A second clone that never leaves halves the ensemble fraction
    f_q, f_a = calcExcursionFractions(T_BACKWARD, [q, np.full(11, Q_LOW)], [a, np.full(11, A_LOW)])
    assert f_q == 0.15
    assert f_a == 0.15


def test_metrics_do_not_depend_on_the_direction_of_the_integration():

    rng = np.random.default_rng(1)
    q = rng.uniform(0.5, 4.0, size=(5, 11))
    a = rng.uniform(2.0, 9.0, size=(5, 11))
    encounters = [rng.uniform(0, 10, size=2) for _ in range(5)]

    t_forward = -T_BACKWARD
    encounters_forward = [-times for times in encounters]

    assert calcExcursionFractions(T_BACKWARD, q, a) == calcExcursionFractions(t_forward, q, a)
    assert splitPerihelionReachByEncounter(-t_forward, q, encounters) \
        == splitPerihelionReachByEncounter(t_forward, q, encounters_forward)


def test_excursion_fraction_is_unchanged_by_finer_sampling_of_the_same_series():

    # A perihelion that steps up at 4 yr, sampled every year and every half year
    def q_of(t):
        return np.where(np.abs(t) >= 4.0, Q_HIGH, Q_LOW)

    t_coarse = -np.arange(0, 10.5, 1.0)
    t_fine = -np.arange(0, 10.25, 0.5)

    f_coarse = calcExcursionFractions(t_coarse, q_of(t_coarse), np.full_like(t_coarse, A_LOW))
    f_fine = calcExcursionFractions(t_fine, q_of(t_fine), np.full_like(t_fine, A_LOW))

    assert f_coarse == f_fine == (0.6, 0.0)


def test_unbound_and_escaped_clones_count_beyond_the_limits():

    q = np.full(11, Q_LOW)
    a = np.full(11, A_LOW)

    # Unbound from 5 yr on: a negative semi-major axis, q still finite and low
    a_unbound = a.copy()
    a_unbound[5:] = -4.0
    assert calcExcursionFractions(T_BACKWARD, q, a_unbound) == (0.0, 0.5)

    # Out of the solar system from 5 yr on
    q_escaped = q.copy()
    q_escaped[5:] = np.inf
    a_escaped = a.copy()
    a_escaped[5:] = np.inf
    assert calcExcursionFractions(T_BACKWARD, q_escaped, a_escaped) == (0.5, 0.5)
    assert countPerihelionReach(q_escaped) == (1, 1)


def test_clone_that_ended_contributes_only_the_time_it_was_followed():

    # Beyond the perihelion limit at its 5 outputs, then gone. Each output is held until the next,
    #   so the clone counts as followed for 5 intervals
    q_ended = np.full(11, np.nan)
    q_ended[:5] = Q_HIGH
    a_ended = np.full(11, np.nan)
    a_ended[:5] = A_LOW

    f_q, _ = calcExcursionFractions(T_BACKWARD, q_ended, a_ended)
    assert f_q == 1.0

    # Pooled with a clone followed for all 10 intervals and never beyond the limit
    f_q, _ = calcExcursionFractions(T_BACKWARD, [q_ended, np.full(11, Q_LOW)],
        [a_ended, np.full(11, A_LOW)])
    assert f_q == 5.0/15.0


def test_clone_without_any_state_is_not_counted():

    q = [np.full(11, Q_HIGH), np.full(11, np.nan)]
    a = [np.full(11, A_LOW), np.full(11, np.nan)]

    assert countPerihelionReach(q) == (1, 1)
    assert calcExcursionFractions(T_BACKWARD, q, a)[0] == 1.0

    f_q, f_a = calcExcursionFractions(T_BACKWARD, np.full(11, np.nan), np.full(11, np.nan))
    assert math.isnan(f_q) and math.isnan(f_a)


def test_reach_counts_a_single_output_beyond_the_limit():

    q = np.full((3, 11), Q_LOW)
    q[0, 7] = Q_HIGH
    q[1, -1] = Q_HIGH

    assert countPerihelionReach(q) == (2, 3)
    assert countPerihelionReach(q, q_limit=Q_HIGH) == (0, 3)


def test_split_by_encounter_uses_the_first_output_beyond_the_limit():

    # Every clone first gets beyond the limit at the 5 yr output
    q = np.full((5, 11), Q_LOW)
    q[:4, 5:] = Q_HIGH

    encounters = [
        [-3.0],         # before the crossing: lifted by the encounter
        [-7.0],         # after it: lifted without one
        [-5.0],         # at the output that first shows the crossing: counted as before
        [],             # no encounter at all
        [-2.0],         # never gets beyond the limit, so it is in neither count
        ]

    n_with, n_without = splitPerihelionReachByEncounter(T_BACKWARD, q, encounters)

    assert (n_with, n_without) == (2, 2)
    assert n_with + n_without == countPerihelionReach(q)[0]


def test_inputs_that_cannot_be_interpreted_are_refused():

    q = np.full((2, 11), Q_LOW)

    # One list of encounters per clone
    assert _raises(ValueError, splitPerihelionReachByEncounter, T_BACKWARD, q, [[]])

    # Outputs that do not move away from the epoch have no integration order
    t_unordered = T_BACKWARD.copy()
    t_unordered[[3, 4]] = t_unordered[[4, 3]]
    assert _raises(ValueError, calcExcursionFractions, t_unordered, q, q)

    # A series sampled at other times
    assert _raises(ValueError, calcExcursionFractions, T_BACKWARD, q[:, :-1], q[:, :-1])

    # An interval needs at least one trial
    assert _raises(ValueError, calcBinomialInterval, 0, 0)


def test_binomial_interval_closed_form_at_zero_successes():

    # At k = 0 the Wilson interval is [0, z^2/(n + z^2)]
    z = 1.959963984540054
    for n in (10, 50, 1000):
        low, high = calcBinomialInterval(0, n)
        assert low == 0.0
        assert abs(high - z**2/(n + z**2)) < 1e-12


def test_binomial_interval_is_symmetric_and_narrows_with_more_trials():

    for k, n in ((0, 50), (7, 50), (25, 50), (85, 100)):
        low, high = calcBinomialInterval(k, n)
        low_mirror, high_mirror = calcBinomialInterval(n - k, n)
        assert abs(low - (1.0 - high_mirror)) < 1e-12
        assert abs(high - (1.0 - low_mirror)) < 1e-12
        assert low <= float(k)/n <= high

    width_50 = np.diff(calcBinomialInterval(25, 50))[0]
    width_1000 = np.diff(calcBinomialInterval(500, 1000))[0]
    assert width_1000 < width_50


def test_classification_gives_both_verdicts():

    stable = classifyPerihelionReach(0, 50)
    assert stable['dynamics_class'] == 'not jupiter family'
    assert not stable['jupiter_family_shober']

    chaotic = classifyPerihelionReach(50, 50)
    assert chaotic['dynamics_class'] == 'jupiter family'
    assert chaotic['jupiter_family_shober']

    # Just above half: the rule of Shober et al. calls it Jupiter-family, the interval cannot tell
    marginal = classifyPerihelionReach(26, 50)
    assert marginal['jupiter_family_shober']
    assert marginal['dynamics_class'] == 'undetermined'

    # Exactly half is not above the threshold
    assert not classifyPerihelionReach(25, 50)['jupiter_family_shober']

    # Enough clones settle a fraction that 50 clones leave open
    assert classifyPerihelionReach(600, 1000)['dynamics_class'] == 'jupiter family'


def test_kozai_energy_analytic_values():

    assert calcKozaiEnergy(0.0, 0.0) == 1.0
    assert abs(calcKozaiEnergy(0.6, np.radians(60.0)) - 0.4) < 1e-12

    # Retrograde orbits have a negative H of the same size as their prograde mirror
    assert abs(calcKozaiEnergy(0.6, np.radians(120.0)) + 0.4) < 1e-12

    # Works on arrays
    h = calcKozaiEnergy(np.array([0.0, 0.6]), np.array([0.0, np.radians(60.0)]))
    assert np.allclose(h, [1.0, 0.4])


class _FakeRebound(object):
    """ Stand-in for wmpl.Rebound.REBOUND.reboundSimulate with known clones, numbered across calls.

    clone(k) says what clone k does: 'reach' (q rises above the limit halfway, after a Jupiter
    encounter at 0.5 Hill radii), 'kozai' (it rises with no encounter), 'stay' (it never does),
    'escape' (after that encounter it turns hyperbolic halfway and leaves two outputs later),
    'ejected' (the same, with no encounter with Jupiter inside 1 Hill radius), 'unbound' (hyperbolic
    at every output but the epoch's, and leaves halfway), 'far' (bound with its aphelion beyond the
    simulation volume, and leaves halfway) or 'impact' (it stays and ends halfway on a body).
    """

    def __init__(self, clone):
        self.clone = clone
        self.calls = []
        self.n = 0

    def __call__(self, julian_date, state_vect, traj=None, direction=None, sim_days=None,
            n_outputs=None, mc_runs=None, random_seed=None, **kwargs):

        self.calls.append(dict(kwargs, direction=direction, sim_days=sim_days, n_outputs=n_outputs,
            mc_runs=mc_runs, random_seed=random_seed))

        half = n_outputs//2
        outputs_mc, diagnostics = {}, {}

        for i in range(mc_runs):

            kind = self.clone(self.n)
            self.n += 1

            a = np.full(n_outputs, 3.5)
            e = np.full(n_outputs, 0.7)
            if kind in ('reach', 'kozai'):
                a[half:] = 8.0
                e[half:] = 0.6
            if kind in ('escape', 'ejected'):
                a[half:], e[half:] = -5.0, 1.3
                a, e = a[:half + 2], e[:half + 2]
            if kind == 'unbound':
                a[:], e[:] = -5.0, 1.2
                a[0], e[0] = 40.0, 0.95
            if kind == 'far':
                a[:], e[:] = 600.0, 0.999
            if kind in ('unbound', 'far', 'impact'):
                a, e = a[:half], e[:half]

            encounters = [{'body': 'Saturn', 'time_days': -365.25, 'n_hill': 0.1},
                {'body': 'Jupiter', 'time_days': -365.25, 'n_hill': 2.0}] \
                if kind in ('reach', 'escape', 'ejected') else []
            if kind in ('reach', 'escape'):
                encounters.append({'body': 'Jupiter', 'time_days': -half*365.25, 'n_hill': 0.5})

            name = 'obj_MC_{:d}'.format(i)
            outputs_mc[name] = {'t': np.zeros(len(a)), 'a': a, 'e': e}
            diagnostics[name] = {
                'escaped': {'time_days': -1.0} if kind in ('escape', 'ejected', 'unbound', 'far') else None,
                'impact': {'body': 'Earth'} if kind == 'impact' else None, 'encounters': encounters}

        return None, outputs_mc, diagnostics


_TRAJ = SimpleNamespace(uncertainties=object(), state_vect_cov=np.eye(6), jdt_ref=2459000.5)


def _runDriver(clone, traj=_TRAJ, **kwargs):
    """ Run classifyJupiterFamilyDynamics on the stand-in, and return its result and the stand-in. """

    fake = _FakeRebound(clone)
    original = (reb.reboundSimulate, reb.REBOUND_FOUND)
    reb.reboundSimulate, reb.REBOUND_FOUND = fake, True

    try:
        result = classifyJupiterFamilyDynamics(traj, years=1000.0, output_years=10.0, **kwargs)

    finally:
        reb.reboundSimulate, reb.REBOUND_FOUND = original

    return result, fake


def test_dynamics_stops_as_soon_as_the_class_is_settled():

    for kind, verdict in (('reach', 'jupiter family'), ('stay', 'not jupiter family')):

        result, fake = _runDriver(lambda k: kind, seed=7)

        # 20 of 20 settles it even at the 99.5% of each of the 10 possible looks
        assert result['n_batches'] == 1 and result['n_clones'] == 20
        assert result['dynamics_class'] == verdict
        assert len(fake.calls) == 1

        call = fake.calls[0]
        assert call['direction'] == 'backward' and call['lean_outputs'] and call['return_diagnostics']
        assert call['mc_runs'] == 20 and call['n_outputs'] == 101 and call['random_seed'] == 7


def test_dynamics_runs_every_batch_when_the_class_stays_open():

    result, fake = _runDriver(lambda k: 'reach' if k%2 else 'stay', seed=100)

    assert result['dynamics_class'] == 'undetermined'
    assert result['n_batches'] == 10 and result['n_clones'] == 200
    assert result['p'] == 0.5

    # Every batch draws its own clones
    assert [c['random_seed'] for c in fake.calls] == list(range(100, 110))

    # The plain interval is narrower than the one each look used
    assert result['look_confidence'] == 1.0 - 0.05/10
    lo, hi = calcBinomialInterval(100, 200)
    assert (result['p_low'], result['p_high']) == (lo, hi)


def test_escaped_clones_are_told_apart_by_their_eccentricity_after_the_epoch():

    kinds = ['escape', 'unbound', 'far']
    _, outputs_mc, diagnostics = _FakeRebound(lambda k: kinds[k])(None, None, n_outputs=11, mc_runs=3)
    q, a, _, left_unbound = cloneEnsemble(outputs_mc, diagnostics, 11)

    # Thrown out: its states, then beyond every limit
    assert np.all(np.isfinite(q[0, :7])) and np.all(np.isinf(q[0, 7:])) and np.all(np.isinf(a[0, 7:]))

    # Hyperbolic after the epoch, though bound at it: not part of the ensemble
    assert np.all(np.isnan(q[1])) and np.all(np.isnan(a[1]))

    # Bound beyond the simulation volume: it keeps its orbit
    assert np.allclose(q[2], 600.0*(1.0 - 0.999)) and np.all(a[2] == 600.0)

    assert list(left_unbound) == [False, True, False]


def test_dynamics_counts_escapes_impacts_and_encounters():

    kinds = ['reach', 'kozai', 'escape', 'ejected', 'impact', 'stay', 'unbound', 'far']
    result, _ = _runDriver(lambda k: kinds[k%8], max_clones=16, batch=16, seed=1)

    # The two unbound clones are integrated but not classified
    assert result['n_integrated'] == 16 and result['n_clones'] == 14
    assert result['n_left_unbound'] == 2
    assert result['n_escaped'] == 8 and result['n_impacted'] == 2

    # Reach, Kozai and the thrown-out clones reach the limit; only the Jupiter encounter inside 1 R_H
    #   counts, so the Kozai and the ejected clones are the ones without one, and of those the
    #   ejected ones got there by being thrown out
    assert result['n_reached'] == 8
    assert (result['n_reached_after_encounter'], result['n_reached_without_encounter']) == (4, 4)
    assert result['n_reached_without_encounter_by_escape'] == 2


def test_dynamics_of_an_unbound_orbit_is_not_classified():

    result, fake = _runDriver(lambda k: 'unbound', seed=5)

    # Every clone left unbound, so one batch is enough to tell, and it gets no Jupiter-family verdict
    assert len(fake.calls) == 1
    assert result['dynamics_class'] == 'unbound'
    assert result['n_clones'] == 0 and result['n_left_unbound'] == 20
    assert math.isnan(result['p'])

    # The report gives the verdict without fractions of an empty ensemble
    report = "\n".join(formatJupiterFamilyDynamics(result))
    assert "Verdict: unbound" in report
    assert "nan" not in report


def test_dynamics_refuses_what_it_cannot_run():

    no_mc = SimpleNamespace(uncertainties=None, state_vect_cov=None, jdt_ref=2459000.5)
    assert _raises(ValueError, _runDriver, lambda k: 'stay', traj=no_mc)
    assert _raises(ValueError, _runDriver, lambda k: 'stay', batch=1)


def test_dynamics_report_states_its_assumptions():

    result, _ = _runDriver(lambda k: 'reach', seed=3, gr='none', beta=1.9e-4, integrator='trace')
    text = "\n".join(formatJupiterFamilyDynamics(result))

    for expected in ("1000 yr backward", "TRACE", "no GR", "beta = 1.900e-04", "seed 3",
            "every 10 yr", "q > 2.5 AU", "20/20", "jupiter family"):
        assert expected in text, expected



# A JFC-like orbit at its descending node, for the shober2024 mode
_ORBIT = SimpleNamespace(a=3.5, e=0.7, i=np.radians(10.0), node=np.radians(100.0),
    peri=np.radians(170.0), true_anomaly=np.radians(10.0), v_g=15000.0)
_UNCERTAINTIES = SimpleNamespace(a=0.02, e=0.003, i=np.radians(0.2), node=np.radians(0.05),
    peri=np.radians(0.04))


class _FakeHeliocentric(object):
    """ Stand-in for wmpl.Rebound.REBOUND.integrateHeliocentricBodies, with the clones of
    _FakeRebound, numbered in the order of the calls.
    """

    def __init__(self, clone):
        self.rebound = _FakeRebound(clone)
        self.calls = []

    def __call__(self, names, states_hel, jd_tdb, times, **kwargs):

        self.calls.append(dict(kwargs, names=names, states_hel=states_hel, jd_tdb=jd_tdb, times=times))
        _, outputs_mc, diagnostics = self.rebound(None, None, n_outputs=len(times), mc_runs=1)

        return {'outputs': {names[0]: outputs_mc['obj_MC_0']},
            'diagnostics': {names[0]: diagnostics['obj_MC_0']}}


# The Sun and an Earth far from the clones, standing in for planetStatesAtEpoch
_PLANETS = (["Sun", "Earth"], [[0.0]*6, [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]], [1.0, 3.0035e-6])


def _runShober2024(clone, traj, **kwargs):
    """ Run classifyJupiterFamilyDynamicsShober2024 on the stand-in, in this process, and return its
    result and the stand-in, which also holds the dates the planets were asked for.
    """

    fake = _FakeHeliocentric(clone)
    fake.planet_dates = []
    names = ('integrateHeliocentricBodies', 'planetStatesAtEpoch', 'REBOUND_FOUND', 'checkReboundxUsable')
    original = [getattr(reb, name) for name in names]
    reb.checkReboundxUsable = lambda: None
    reb.integrateHeliocentricBodies, reb.REBOUND_FOUND = fake, True
    reb.planetStatesAtEpoch = lambda jd_tdb: fake.planet_dates.append(jd_tdb) or _PLANETS

    try:
        result = classifyJupiterFamilyDynamicsShober2024(traj, years=1000.0, output_years=10.0,
            n_cpu=1, **kwargs)

    finally:
        for name, value in zip(names, original):
            setattr(reb, name, value)

    return result, fake


def test_shober2024_draws_the_clones_from_the_diagonal_of_the_uncertainties():

    traj = SimpleNamespace(orbit=_ORBIT, uncertainties=_UNCERTAINTIES)
    sigma, source = elementUncertainties(traj)
    assert source == "Monte Carlo of the trajectory"
    assert np.isclose(sigma['i'], 0.2) and np.isclose(sigma['peri'], 0.04)

    n = 20000
    clones = drawElementClones(_ORBIT, sigma, n, np.random.default_rng(1))

    nominal = {'a': _ORBIT.a, 'e': _ORBIT.e, 'i': 10.0, 'node': 100.0, 'peri': 170.0}
    for key in nominal:
        assert abs(np.mean(clones[key]) - nominal[key]) < 4*sigma[key]/np.sqrt(n), key
        assert abs(np.std(clones[key])/sigma[key] - 1.0) < 0.03, key

    # Independent draws, as from the diagonal of a covariance
    assert abs(np.corrcoef(clones['a'], clones['e'])[0, 1]) < 4/np.sqrt(n)

    # Every clone keeps the argument of latitude, so it is at the node as the meteoroid was
    assert np.allclose(clones['peri'] + clones['f'], 180.0)


def test_shober2024_uses_the_uncertainties_the_paper_assumed_without_monte_carlo():

    sigma, source = elementUncertainties(SimpleNamespace(orbit=_ORBIT, uncertainties=None))
    assert sigma == SHOBER2024_DEFAULT_SIGMA
    assert "MORP" in source


def test_shober2024_refuses_clones_that_make_no_conic():

    near_parabolic = SimpleNamespace(**dict(vars(_ORBIT), a=50.0, e=0.995))
    sigma = dict(SHOBER2024_DEFAULT_SIGMA, e=0.01)
    assert _raises(ValueError, drawElementClones, near_parabolic, sigma, 100, np.random.default_rng(2))


@_needsRebound
def test_shober2024_clones_start_three_hill_radii_before_the_earth():

    from wmpl.Utils.Pickling import loadPickle
    from wmpl.Trajectory.Orbit import eclipticToRectangularVelocityVect

    examples = os.path.join(os.path.dirname(__file__), "..", "..", "Dynesty", "examples",
        "20191023_091225")
    orbit = loadPickle(examples, "20191023_091225_trajectory.pickle").orbit

    zero = dict.fromkeys(SHOBER2024_DEFAULT_SIGMA, 0.0)
    clone = drawElementClones(orbit, zero, 1, np.random.default_rng(0))

    # At the epoch the elements put the clone where the trajectory placed the meteoroid
    au_km = reb.AU_M/1000.0
    start = np.array(elementCloneStates(clone, 0.0)[0])
    assert np.linalg.norm(start[:3]*au_km - np.array(orbit.meteor_pos)) < 1e-3

    # The Earth at the epoch, moving at the meteoroid's heliocentric velocity less the geocentric one
    v_g = np.array(eclipticToRectangularVelocityVect(orbit.L_g, orbit.B_g, orbit.v_g/1000.0))
    v_earth = (start[3:] - v_g*86400.0/au_km)*reb.SIM_TIME_DAYS
    sim = reb.rb.Simulation()
    sim.add(m=1.0)
    sim.add(x=start[0], y=start[1], z=start[2], vx=v_earth[0], vy=v_earth[1], vz=v_earth[2])

    dt_days = SHOBER2024_START_HILL*reb.HILL_RADII_AU["Earth"]*reb.AU_M/orbit.v_g/86400.0
    sim.integrate(-dt_days/reb.SIM_TIME_DAYS)
    before = np.array(elementCloneStates(clone, dt_days)[0])

    distance = np.linalg.norm(before[:3] - np.array(sim.particles[1].xyz))
    assert abs(distance/reb.HILL_RADII_AU["Earth"] - SHOBER2024_START_HILL) < 1e-3


@_needsRebound
def test_shober2024_start_does_not_depend_on_the_distance_from_the_earth():

    from wmpl.Utils.Pickling import loadPickle
    from wmpl.Trajectory.Orbit import eclipticToRectangularVelocityVect

    examples = os.path.join(os.path.dirname(__file__), "..", "..", "Dynesty", "examples",
        "20191023_091225")
    traj = loadPickle(examples, "20191023_091225_trajectory.pickle")
    orbit = traj.orbit

    zero = dict.fromkeys(SHOBER2024_DEFAULT_SIGMA, 0.0)
    clone = drawElementClones(orbit, zero, 1, np.random.default_rng(0))
    start = np.array(elementCloneStates(clone, 0.0)[0])

    # The Sun and the Earth alone, the Earth where the meteoroid met it
    au_km = reb.AU_M/1000.0
    v_g = np.array(eclipticToRectangularVelocityVect(orbit.L_g, orbit.B_g, orbit.v_g/1000.0))
    v_earth = (start[3:] - v_g*86400.0/au_km)*reb.SIM_TIME_DAYS
    m_earth = 3.0035e-6
    jd = reb.utcToTdb(traj.jdt_ref)

    offset = {}
    for hill in (3.0, 10.0):

        dt_days = hill*reb.HILL_RADII_AU["Earth"]*reb.AU_M/orbit.v_g/86400.0
        sim = reb.rb.Simulation()
        sim.add(m=1.0)
        sim.add(m=m_earth, x=start[0], y=start[1], z=start[2], vx=v_earth[0], vy=v_earth[1],
            vz=v_earth[2])
        sim.move_to_com()
        sim.integrate(-dt_days/reb.SIM_TIME_DAYS)
        planets = (["Sun", "Earth"], [list(p.xyz) + list(p.vxyz) for p in sim.particles], [1.0, m_earth])

        for corrected in (False, True):
            states = elementCloneStates(clone, dt_days)
            if corrected:
                states = addEarthAttraction(states, planets)
            run = reb.integrateHeliocentricBodies(["c"], states, jd - dt_days,
                [0.0, -365.25/reb.SIM_TIME_DAYS], planets=planets, gr='none', sun_j2=False,
                lean_outputs=True)
            offset[(hill, corrected)] = run['outputs']['c']['a'][-1] - orbit.a

    # Far from the Earth the orbit is the one without its attraction, wherever the clone started
    #   (2.7e-4 and 2.2e-4 AU when written)
    assert abs(offset[(3.0, True)]) < 1e-3 and abs(offset[(10.0, True)]) < 1e-3

    # Started at the speed at infinity, it loses GM/d on the way out: -0.014 AU from 3 R_H
    assert offset[(3.0, False)] < -0.01
    assert abs(offset[(3.0, False)]/offset[(10.0, False)] - 10.0/3.0) < 0.1*10.0/3.0


@_needsRebound
def test_shober2024_integrates_every_clone_with_the_setup_of_the_paper():

    traj = SimpleNamespace(orbit=_ORBIT, uncertainties=_UNCERTAINTIES, jdt_ref=2459000.5)
    result, fake = _runShober2024(lambda k: 'stay', traj, n_clones=30, seed=4)

    # No early stop: every clone, though the first 20 would have settled the class
    assert len(fake.calls) == 30 and result['n_integrated'] == 30 and result['n_batches'] == 1
    assert sorted(name for call in fake.calls for name in call['names']) == \
        sorted('obj_MC_{:d}'.format(k) for k in range(30))

    dt_days = SHOBER2024_START_HILL*reb.HILL_RADII_AU["Earth"]*reb.AU_M/_ORBIT.v_g/86400.0
    jd_start = reb.utcToTdb(traj.jdt_ref) - dt_days
    for call in fake.calls:
        assert (call['integrator'], call['gr'], call['sun_j2'], call['lean_outputs']) == \
            ('ias15', 'none', False, True)
        assert call['jd_tdb'] == jd_start and call['planets'] is _PLANETS
        assert len(call['times']) == 101
        assert np.isclose(call['times'][-1], -1000.0*365.25/reb.SIM_TIME_DAYS)

    assert fake.planet_dates == [jd_start]
    assert result['assumptions']['mode'] == 'shober2024'
    assert result['assumptions']['start_days_before'] == dt_days
    assert result['dynamics_class'] == 'not jupiter family' and result['p'] == 0.0


@_needsRebound
def test_shober2024_counts_like_the_adapted_mode():

    kinds = ['reach', 'kozai', 'escape', 'ejected', 'impact', 'stay', 'unbound', 'far']
    traj = SimpleNamespace(orbit=_ORBIT, uncertainties=_UNCERTAINTIES, jdt_ref=2459000.5)
    result, _ = _runShober2024(lambda k: kinds[k%8], traj, n_clones=16, seed=1)
    adapted, _ = _runDriver(lambda k: kinds[k%8], max_clones=16, batch=16, seed=1)

    for key in ('n_integrated', 'n_clones', 'n_left_unbound', 'n_escaped', 'n_impacted', 'n_reached',
            'n_reached_after_encounter', 'n_reached_without_encounter',
            'n_reached_without_encounter_by_escape', 'f_q', 'f_a', 'p', 'p_low', 'p_high'):
        assert result[key] == adapted[key], key


@_needsRebound
def test_shober2024_report_states_its_assumptions():

    traj = SimpleNamespace(orbit=_ORBIT, uncertainties=None, jdt_ref=2459000.5)
    result, _ = _runShober2024(lambda k: 'reach', traj, n_clones=20, seed=6)
    text = "\n".join(formatJupiterFamilyDynamics(result))

    for expected in ("setup of Shober et al. 2024", "1000 yr backward", "IAS15", "Newtonian",
            "no GR, Sun J2, radiation", "20, a, e, i, node, peri drawn independently, seed 6",
            "assumed by Shober et al. for MORP", "a 0.01 AU", "argument of latitude kept",
            "3 Hill radii before the Earth", "every 10 yr", "20/20",
            "Verdict: jupiter family (from the interval at 95%)"):
        assert expected in text, expected


if __name__ == "__main__":

    test_functions = [
        test_excursion_fractions_are_exact_for_series_that_change_at_the_outputs,
        test_metrics_do_not_depend_on_the_direction_of_the_integration,
        test_excursion_fraction_is_unchanged_by_finer_sampling_of_the_same_series,
        test_unbound_and_escaped_clones_count_beyond_the_limits,
        test_clone_that_ended_contributes_only_the_time_it_was_followed,
        test_clone_without_any_state_is_not_counted,
        test_reach_counts_a_single_output_beyond_the_limit,
        test_split_by_encounter_uses_the_first_output_beyond_the_limit,
        test_inputs_that_cannot_be_interpreted_are_refused,
        test_binomial_interval_closed_form_at_zero_successes,
        test_binomial_interval_is_symmetric_and_narrows_with_more_trials,
        test_classification_gives_both_verdicts,
        test_kozai_energy_analytic_values,
        test_dynamics_stops_as_soon_as_the_class_is_settled,
        test_dynamics_runs_every_batch_when_the_class_stays_open,
        test_escaped_clones_are_told_apart_by_their_eccentricity_after_the_epoch,
        test_dynamics_counts_escapes_impacts_and_encounters,
        test_dynamics_of_an_unbound_orbit_is_not_classified,
        test_dynamics_refuses_what_it_cannot_run,
        test_dynamics_report_states_its_assumptions,
        test_shober2024_draws_the_clones_from_the_diagonal_of_the_uncertainties,
        test_shober2024_uses_the_uncertainties_the_paper_assumed_without_monte_carlo,
        test_shober2024_refuses_clones_that_make_no_conic,
        test_shober2024_clones_start_three_hill_radii_before_the_earth,
        test_shober2024_start_does_not_depend_on_the_distance_from_the_earth,
        test_shober2024_integrates_every_clone_with_the_setup_of_the_paper,
        test_shober2024_counts_like_the_adapted_mode,
        test_shober2024_report_states_its_assumptions,
        ]

    failed = 0
    for test_func in test_functions:

        if (test_func.__name__ in _REBOUND_TESTS) and (not reb.REBOUND_FOUND):
            print("SKIP: {:s}: rebound/reboundx not importable".format(test_func.__name__))
            continue

        try:
            test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {:s}".format(test_func.__name__, str(e)))

    print()
    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
