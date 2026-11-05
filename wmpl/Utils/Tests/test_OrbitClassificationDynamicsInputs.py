""" Tests of what classifyJupiterFamilyDynamics refuses and of the clones it has to set apart: spans
with no output, no clones, clones that fail, unbound clones still inside the simulation volume, and
the warning of the integrator that cannot resolve an encounter. No integration is run. """

import inspect

import numpy as np
import pytest

from wmpl.Rebound import REBOUND as reb
from wmpl.Utils.OrbitClassification import (classifyJupiterFamilyDynamics, cloneEnsemble,
    dynamicsOutputTimes, formatJupiterFamilyDynamics)
from wmpl.Utils.Tests.test_OrbitClassificationDynamics import _FakeRebound, _runDriver, _TRAJ


def _withFake(fake, **kwargs):
    """ Run the driver on a stand-in of reboundSimulate, at the defaults of its span. """

    original = (reb.reboundSimulate, reb.REBOUND_FOUND)
    reb.reboundSimulate, reb.REBOUND_FOUND = fake, True

    try:
        return classifyJupiterFamilyDynamics(_TRAJ, years=1000.0, output_years=10.0, seed=1, **kwargs)

    finally:
        reb.reboundSimulate, reb.REBOUND_FOUND = original


@pytest.mark.parametrize("years", [20.0, 0.0, -100.0])
def test_a_span_with_no_output_after_the_start_is_refused(years):

    with pytest.raises(ValueError, match="no output after the start"):
        dynamicsOutputTimes(years, 50.0)


def test_the_default_spacing_refuses_a_short_span_before_integrating():

    fake = _FakeRebound(lambda k: 'stay')
    original = (reb.reboundSimulate, reb.REBOUND_FOUND)
    reb.reboundSimulate, reb.REBOUND_FOUND = fake, True

    try:
        with pytest.raises(ValueError, match="no output after the start"):
            classifyJupiterFamilyDynamics(_TRAJ, years=20.0, seed=1)

    finally:
        reb.reboundSimulate, reb.REBOUND_FOUND = original

    assert fake.calls == []


def test_the_output_times_span_the_run():

    t = dynamicsOutputTimes(100.0, 50.0)

    assert t.tolist() == [0.0, -50.0, -100.0]


@pytest.mark.parametrize("max_clones", [0, -5])
def test_no_clones_are_refused_rather_than_failing_later(max_clones):

    with pytest.raises(ValueError, match="At least one clone"):
        _runDriver(lambda k: 'stay', max_clones=max_clones, seed=1)


def test_the_nominal_solution_is_not_integrated():

    _, fake = _runDriver(lambda k: 'stay', seed=1)

    assert all(call['skip_nominal'] for call in fake.calls)


def test_skip_nominal_is_a_keyword_of_the_real_simulation():

    assert inspect.signature(reb.reboundSimulate).parameters['skip_nominal'].default is False


def test_a_batch_where_every_clone_failed_is_an_error_and_not_an_unbound_orbit():

    def nothing(*args, **kwargs):
        return None, {}, {}

    with pytest.raises(RuntimeError, match="failed to integrate"):
        _withFake(nothing)


def test_failed_clones_are_counted_and_reported():

    fake = _FakeRebound(lambda k: 'stay')

    def lose_one(*args, **kwargs):

        nominal, outputs_mc, diagnostics = fake(*args, **kwargs)
        del outputs_mc['obj_MC_0']

        return nominal, outputs_mc, diagnostics

    result = _withFake(lose_one)

    assert result['n_failed'] == 1 and result['n_integrated'] == 19 and result['n_clones'] == 19
    assert any("1 clone(s) failed" in line for line in formatJupiterFamilyDynamics(result))


def _clone(e, escaped=None):
    """ The lean output and the diagnostics of a clone with the given eccentricities. """

    e = np.array(e, dtype=float)
    a = np.where(e < 1.0, 3.0, -5.0)

    return {'a': a, 'e': e}, {'escaped': {'time_days': -1.0} if escaped else None, 'impact': None,
        'encounters': []}


def _ensemble(*clones):

    outputs = {'obj_MC_{:d}'.format(k): clone[0] for k, clone in enumerate(clones)}
    diagnostics = {'obj_MC_{:d}'.format(k): clone[1] for k, clone in enumerate(clones)}

    return cloneEnsemble(outputs, diagnostics, 5)


def test_a_clone_unbound_inside_the_volume_is_left_out_like_one_that_left_it():

    inside, outside = _clone([0.9, 1.2, 1.2, 1.2, 1.2]), _clone([0.9, 1.2, 1.2], escaped=True)

    q, a, _, left_unbound = _ensemble(inside, outside)

    assert left_unbound.tolist() == [True, True]
    assert np.all(np.isnan(q)) and np.all(np.isnan(a))


def test_a_clone_that_turns_unbound_inside_the_volume_keeps_its_bound_time():

    q, _, _, left_unbound = _ensemble(_clone([0.7, 0.7, 0.7, 1.2, 1.2]))

    assert not left_unbound[0]
    assert np.all(np.isfinite(q))


def test_a_clone_that_hit_a_body_at_once_is_not_called_unbound():

    q, _, _, left_unbound = _ensemble(_clone([1.2]))

    assert not left_unbound[0]
    assert np.isfinite(q[0, 0])


def _encounterWhileFixedStep():

    return {'fixed_step_from_days': -1.0, 'encounters': [{'body': 'Jupiter', 'time_days': -3000.0,
        'n_hill': 0.5}]}


def test_whfast_warns_of_encounters_in_every_batch():

    fake = _FakeRebound(lambda k: 'stay')

    def with_encounters(*args, **kwargs):

        nominal, outputs_mc, diagnostics = fake(*args, **kwargs)
        for name in diagnostics:
            diagnostics[name].update(_encounterWhileFixedStep())

        return nominal, outputs_mc, diagnostics

    result = _withFake(with_encounters, integrator='whfast', batch=2, max_clones=2)
    assert "2 close encounter(s)" in result['integrator_warning']
    assert any("WHFast" in line for line in formatJupiterFamilyDynamics(result))

    # The same encounters under IAS15 need no warning
    assert _withFake(with_encounters, batch=2, max_clones=2)['integrator_warning'] is None
