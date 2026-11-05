""" Tests of the checkpoint of the Jupiter-family dynamics: the file itself, and that a run cut and started
again gives the result of one that was not cut, integrating only what was missing. No integration is run. """

import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from wmpl.Rebound import REBOUND as reb
from wmpl.Utils.DynamicsCheckpoint import (DynamicsCheckpoint, DynamicsCheckpointMismatch, checkpointSeed,
    fingerprint, readCheckpoint)
from wmpl.Utils.OrbitClassification import (classifyJupiterFamilyDynamics,
    classifyJupiterFamilyDynamicsShober2024, formatJupiterFamilyDynamics)
from wmpl.Utils.Tests.test_OrbitClassificationDynamics import (_FakeRebound, _needsRebound, _ORBIT, _PLANETS,
    _UNCERTAINTIES)


SIGNATURE = {'seed': 3, 'years': 100.0}


### The file ###

def test_records_are_read_back(tmp_path):

    path = str(tmp_path/"run.pkl")

    checkpoint = DynamicsCheckpoint(path, SIGNATURE)
    assert checkpoint.records == {}
    checkpoint.add('a', [1, 2])
    checkpoint.add('b', np.arange(3))

    again = DynamicsCheckpoint(path, SIGNATURE)
    assert sorted(again.records) == ['a', 'b'] and again.records['a'] == [1, 2]
    assert checkpointSeed(path) == 3


def test_a_record_cut_while_written_is_dropped_and_the_file_goes_on(tmp_path):

    path = str(tmp_path/"run.pkl")

    checkpoint = DynamicsCheckpoint(path, SIGNATURE)
    checkpoint.add('a', 1)
    whole = len(open(path, 'rb').read())
    checkpoint.add('b', list(range(1000)))

    data = open(path, 'rb').read()
    open(path, 'wb').write(data[:whole + (len(data) - whole)//2])

    again = DynamicsCheckpoint(path, SIGNATURE)
    assert sorted(again.records) == ['a']

    again.add('c', 3)
    assert sorted(readCheckpoint(path)[1]) == ['a', 'c']


def test_a_run_with_other_parameters_is_refused_and_names_them(tmp_path):

    path = str(tmp_path/"run.pkl")
    DynamicsCheckpoint(path, SIGNATURE).add('a', 1)

    with pytest.raises(DynamicsCheckpointMismatch, match="years") as error:
        DynamicsCheckpoint(path, {'seed': 3, 'years': 200.0})

    assert error.value.differences == [('years', 100.0, 200.0)]
    assert sorted(readCheckpoint(path)[1]) == ['a']


@pytest.mark.parametrize("content", [b"not a pickle at all", pickle.dumps({'a trajectory': 1}),
    pickle.dumps({'kind': 'header', 'version': 1, 'signature': {}})])
def test_a_file_that_is_not_a_checkpoint_is_not_overwritten(tmp_path, content):

    path = tmp_path/"traj.pickle"
    path.write_bytes(content)

    with pytest.raises(ValueError, match="not a checkpoint"):
        DynamicsCheckpoint(str(path), SIGNATURE)

    assert path.read_bytes() == content


def test_an_empty_file_is_a_new_checkpoint(tmp_path):

    path = tmp_path/"run.pkl"
    path.write_bytes(b"")

    DynamicsCheckpoint(str(path), SIGNATURE).add('a', 1)

    assert sorted(readCheckpoint(str(path))[1]) == ['a']


def test_no_file_has_no_seed(tmp_path):

    assert checkpointSeed(str(tmp_path/"none.pkl")) is None


def test_the_fingerprint_sees_the_last_bit_and_numpy_values():

    assert fingerprint([1.0, 2.0]) == fingerprint([np.float64(1.0), 2.0])
    assert fingerprint([1.0]) != fingerprint([np.nextafter(1.0, 2.0)])


### A run of the adapted mode ###

class _Interrupt(Exception):
    pass


TRAJ_ADAPTED = SimpleNamespace(uncertainties=object(), state_vect_cov=np.eye(6), jdt_ref=2459000.5,
    state_vect_mini=np.ones(3), v_init=1.0, radiant_eci_mini=np.ones(3))

KEYS = ('n_clones', 'n_reached', 'n_reached_after_encounter', 'n_reached_without_encounter', 'n_left_unbound',
    'n_integrated', 'n_batches', 'dynamics_class', 'p', 'f_q', 'f_a')


def _same(a, b):

    for key in KEYS:
        assert (a[key] == b[key]) or (isinstance(a[key], float) and math.isnan(a[key]) and math.isnan(b[key]))


class _FakeAdapted(object):
    """ The stand-in of reboundSimulate, whose clones depend on the batch, so that they are the same however
    many were integrated before. It can stop before a given batch. """

    def __init__(self, stop_at=None):
        self.stop_at = stop_at
        self.seeds = []

    def __call__(self, julian_date, state_vect, random_seed=None, **kwargs):

        self.seeds.append(random_seed)

        if len(self.seeds) - 1 == self.stop_at:
            raise _Interrupt()

        fake = _FakeRebound(lambda k: 'reach' if (k + random_seed)%2 else 'stay')

        return fake(julian_date, state_vect, random_seed=random_seed, **kwargs)


def _runAdapted(fake, **kwargs):

    original = (reb.reboundSimulate, reb.REBOUND_FOUND)
    reb.reboundSimulate, reb.REBOUND_FOUND = fake, True

    try:
        return classifyJupiterFamilyDynamics(TRAJ_ADAPTED, years=1000.0, output_years=10.0, batch=4, max_clones=16,
            **kwargs)

    finally:
        reb.reboundSimulate, reb.REBOUND_FOUND = original


def test_adapted_run_cut_and_started_again_gives_the_same_result(tmp_path):

    path = str(tmp_path/"run.pkl")
    whole = _runAdapted(_FakeAdapted(), seed=100)

    with pytest.raises(_Interrupt):
        _runAdapted(_FakeAdapted(stop_at=2), seed=100, checkpoint_path=path)

    assert sorted(readCheckpoint(path)[1]) == ['batch_0', 'batch_1']

    fake = _FakeAdapted()
    result = _runAdapted(fake, seed=100, checkpoint_path=path)

    _same(whole, result)
    assert fake.seeds == [102, 103]
    assert result['n_resumed'] == 8


def test_adapted_run_that_set_no_seed_goes_on_with_the_seed_of_the_file(tmp_path):

    path = str(tmp_path/"run.pkl")

    with pytest.raises(_Interrupt):
        _runAdapted(_FakeAdapted(stop_at=1), checkpoint_path=path)

    seed = checkpointSeed(path)
    fake = _FakeAdapted()
    result = _runAdapted(fake, checkpoint_path=path)

    assert result['assumptions']['seed'] == seed and fake.seeds[0] == seed + 1


def test_adapted_finished_run_integrates_nothing_more(tmp_path):

    path = str(tmp_path/"run.pkl")
    first = _runAdapted(_FakeAdapted(), seed=100, checkpoint_path=path)

    fake = _FakeAdapted()
    second = _runAdapted(fake, seed=100, checkpoint_path=path)

    _same(first, second)
    assert fake.seeds == []
    assert any("read back" in line for line in formatJupiterFamilyDynamics(second))


@pytest.mark.parametrize("change", [dict(seed=101), dict(gr='none'), dict(integrator='trace'), dict(beta=1e-3)])
def test_adapted_run_with_other_parameters_is_refused_before_integrating(tmp_path, change):

    path = str(tmp_path/"run.pkl")

    with pytest.raises(_Interrupt):
        _runAdapted(_FakeAdapted(stop_at=1), seed=100, checkpoint_path=path)

    fake = _FakeAdapted()
    with pytest.raises(DynamicsCheckpointMismatch):
        _runAdapted(fake, checkpoint_path=path, **dict(dict(seed=100), **change))

    assert fake.seeds == []


def test_another_trajectory_is_refused(tmp_path):

    path = str(tmp_path/"run.pkl")
    with pytest.raises(_Interrupt):
        _runAdapted(_FakeAdapted(stop_at=1), seed=100, checkpoint_path=path)

    original = TRAJ_ADAPTED.v_init
    TRAJ_ADAPTED.v_init = 1.0 + 1e-9

    try:
        with pytest.raises(DynamicsCheckpointMismatch, match="trajectory"):
            _runAdapted(_FakeAdapted(), seed=100, checkpoint_path=path)

    finally:
        TRAJ_ADAPTED.v_init = original


def test_the_verdict_settings_can_change_on_a_finished_run(tmp_path):

    path = str(tmp_path/"run.pkl")
    _runAdapted(_FakeAdapted(), seed=100, checkpoint_path=path)

    fake = _FakeAdapted()
    result = _runAdapted(fake, seed=100, checkpoint_path=path, encounter_hill=0.5, confidence=0.9)

    assert fake.seeds == [] and result['n_resumed'] > 0


### A run of shober2024 ###

TRAJ_SHOBER = SimpleNamespace(orbit=_ORBIT, uncertainties=_UNCERTAINTIES, jdt_ref=2459000.5)


class _FakeShober(object):
    """ The stand-in of integrateHeliocentricBodies, whose clones depend on their name. """

    def __init__(self, stop_at=None, failing=()):
        self.stop_at = stop_at
        self.failing = failing
        self.names = []

    def __call__(self, names, states_hel, jd_tdb, times, **kwargs):

        self.names.append(names[0])

        if len(self.names) - 1 == self.stop_at:
            raise KeyboardInterrupt()

        if names[0] in self.failing:
            raise RuntimeError("no luck")

        k = int(names[0].rsplit('_', 1)[1])
        _, outputs, diagnostics = _FakeRebound(lambda _: 'reach' if k%3 == 0 else 'stay')(None, None,
            n_outputs=len(times), mc_runs=1)

        return {'outputs': {names[0]: outputs['obj_MC_0']}, 'diagnostics': {names[0]: diagnostics['obj_MC_0']}}


@pytest.fixture
def shober(monkeypatch):

    monkeypatch.setattr(reb, 'REBOUND_FOUND', True)
    monkeypatch.setattr(reb, 'checkReboundxUsable', lambda: None)
    monkeypatch.setattr(reb, 'planetStatesAtEpoch', lambda jd_tdb: _PLANETS)

    def _run(fake, **kwargs):

        monkeypatch.setattr(reb, 'integrateHeliocentricBodies', fake)

        arguments = dict(years=1000.0, output_years=10.0, n_clones=9, n_cpu=1)
        arguments.update(kwargs)

        return classifyJupiterFamilyDynamicsShober2024(TRAJ_SHOBER, **arguments)

    return _run


@_needsRebound
def test_shober_run_cut_and_started_again_gives_the_same_result(shober, tmp_path):

    path = str(tmp_path/"run.pkl")
    whole = shober(_FakeShober(), seed=5)

    with pytest.raises(KeyboardInterrupt):
        shober(_FakeShober(stop_at=4), seed=5, checkpoint_path=path)

    assert len(readCheckpoint(path)[1]) == 4

    fake = _FakeShober()
    result = shober(fake, seed=5, checkpoint_path=path)

    _same(whole, result)
    assert fake.names == ['obj_MC_{:d}'.format(k) for k in range(4, 9)]
    assert result['n_resumed'] == 4


@_needsRebound
def test_shober_clones_that_failed_are_tried_again(shober, tmp_path):

    path = str(tmp_path/"run.pkl")

    first = shober(_FakeShober(failing=['obj_MC_2']), seed=5, checkpoint_path=path)
    assert first['n_failed'] == 1

    fake = _FakeShober()
    second = shober(fake, seed=5, checkpoint_path=path)

    assert fake.names == ['obj_MC_2']
    assert second['n_failed'] == 0 and second['n_integrated'] == 9


@_needsRebound
def test_shober_run_that_set_no_seed_keeps_the_seed_and_the_clones_of_the_file(shober, tmp_path):

    path = str(tmp_path/"run.pkl")

    with pytest.raises(KeyboardInterrupt):
        shober(_FakeShober(stop_at=2), checkpoint_path=path)

    seed = checkpointSeed(path)
    assert seed is not None

    result = shober(_FakeShober(), checkpoint_path=path)
    assert result['assumptions']['seed'] == seed and result['n_resumed'] == 2


@_needsRebound
@pytest.mark.parametrize("change", [dict(seed=6), dict(n_clones=10), dict(years=2000.0), dict(output_years=20.0)])
def test_shober_run_with_other_parameters_is_refused_before_integrating(shober, tmp_path, change):

    path = str(tmp_path/"run.pkl")
    with pytest.raises(KeyboardInterrupt):
        shober(_FakeShober(stop_at=2), seed=5, checkpoint_path=path)

    fake = _FakeShober()
    with pytest.raises(DynamicsCheckpointMismatch):
        shober(fake, checkpoint_path=path, **dict(dict(seed=5), **change))

    assert fake.names == []


@_needsRebound
def test_shober_checkpoint_that_cannot_be_written_stops_the_run_and_is_no_failed_clone(shober, tmp_path,
    monkeypatch):

    path = str(tmp_path/"run.pkl")

    def full(self, key, data):
        raise OSError("no space left on device")

    monkeypatch.setattr(DynamicsCheckpoint, 'add', full)

    with pytest.raises(OSError, match="no space"):
        shober(_FakeShober(), seed=5, checkpoint_path=path)
