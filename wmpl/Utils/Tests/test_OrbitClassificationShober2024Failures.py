""" Tests that one failed clone does not lose a --dynamics shober2024 run, and of what it refuses
before starting the hours of integration. No integration is run. """

import concurrent.futures
from types import SimpleNamespace

import pytest

from wmpl.Rebound import REBOUND as reb
from wmpl.Utils.OrbitClassification import (classifyJupiterFamilyDynamicsShober2024,
    formatJupiterFamilyDynamics)
from wmpl.Utils.Tests.test_OrbitClassificationDynamics import (_FakeHeliocentric, _needsRebound, _ORBIT,
    _PLANETS, _UNCERTAINTIES)


_TRAJ = SimpleNamespace(orbit=_ORBIT, uncertainties=_UNCERTAINTIES, jdt_ref=2459000.5)


class _Failing(_FakeHeliocentric):
    """ The stand-in of the integration, failing for the clones given. """

    def __init__(self, failing):
        super(_Failing, self).__init__(lambda k: 'stay')
        self.failing = failing

    def __call__(self, names, *args, **kwargs):

        if names[0] in self.failing or self.failing == 'all':
            raise RuntimeError("no luck with {:s}".format(names[0]))

        return super(_Failing, self).__call__(names, *args, **kwargs)


class _InlineExecutor(object):
    """ A stand-in of the process pool that runs each call at once, and keeps its result or error in
    a real future, since the stand-ins cannot be sent to another process. """

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args, **kwargs):

        future = concurrent.futures.Future()

        try:
            future.set_result(function(*args, **kwargs))

        except Exception as err:
            future.set_exception(err)

        return future


@pytest.fixture
def run(monkeypatch):

    monkeypatch.setattr(reb, 'REBOUND_FOUND', True)
    monkeypatch.setattr(reb, 'checkReboundxUsable', lambda: None)
    monkeypatch.setattr(reb, 'planetStatesAtEpoch', lambda jd_tdb: _PLANETS)

    def _run(failing, **kwargs):

        monkeypatch.setattr(reb, 'integrateHeliocentricBodies', _Failing(failing))

        return classifyJupiterFamilyDynamicsShober2024(_TRAJ, years=1000.0, output_years=10.0, seed=2,
            n_clones=6, **kwargs)

    return _run


@_needsRebound
def test_one_failed_clone_leaves_the_others_counted(run):

    result = run(['obj_MC_3'], n_cpu=1)

    assert result['n_failed'] == 1 and result['n_integrated'] == 5 and result['n_clones'] == 5
    assert 'no luck with obj_MC_3' in result['failed_clones']['obj_MC_3']

    report = "\n".join(formatJupiterFamilyDynamics(result))
    assert "1 clone(s) failed" in report and "no luck with obj_MC_3" in report


@_needsRebound
def test_one_failed_clone_leaves_the_others_counted_in_the_process_pool(run, monkeypatch):

    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', _InlineExecutor)

    result = run(['obj_MC_0', 'obj_MC_5'], n_cpu=4)

    assert result['n_failed'] == 2 and result['n_integrated'] == 4
    assert sorted(result['failed_clones']) == ['obj_MC_0', 'obj_MC_5']


@_needsRebound
def test_a_run_without_failures_reports_none(run):

    result = run([], n_cpu=1)

    assert result['n_failed'] == 0 and result['failed_clones'] == {}
    assert not any("failed" in line for line in formatJupiterFamilyDynamics(result))


@_needsRebound
def test_every_clone_failing_is_an_error_and_not_an_unbound_orbit(run):

    with pytest.raises(RuntimeError, match="All the 6 clones failed.*no luck"):
        run('all', n_cpu=1)


def _refused(monkeypatch, **kwargs):

    monkeypatch.setattr(reb, 'REBOUND_FOUND', True)
    monkeypatch.setattr(reb, 'checkReboundxUsable', lambda: None)

    # Nothing here may be reached
    monkeypatch.setattr(reb, 'planetStatesAtEpoch', lambda jd_tdb: pytest.fail("the run went on"))

    return classifyJupiterFamilyDynamicsShober2024(_TRAJ, seed=2, **kwargs)


def test_a_span_with_no_output_after_the_start_is_refused(monkeypatch):

    with pytest.raises(ValueError, match="no output after the start"):
        _refused(monkeypatch, years=2.0)


@pytest.mark.parametrize("n_clones", [0, -1])
def test_no_clones_are_refused(monkeypatch, n_clones):

    with pytest.raises(ValueError, match="At least one clone"):
        _refused(monkeypatch, n_clones=n_clones)


def test_a_reboundx_that_does_not_attach_stops_it_before_it_starts(monkeypatch):

    def broken():
        raise RuntimeError("REBOUNDx did not attach")

    monkeypatch.setattr(reb, 'REBOUND_FOUND', True)
    monkeypatch.setattr(reb, 'checkReboundxUsable', broken)
    monkeypatch.setattr(reb, 'planetStatesAtEpoch', lambda jd_tdb: pytest.fail("the run went on"))

    with pytest.raises(RuntimeError, match="did not attach"):
        classifyJupiterFamilyDynamicsShober2024(_TRAJ, seed=2)
