""" Tests for the output size estimate and the checkpoint of the integrated orbits.

The size estimate and the checkpoint file format are tested without REBOUND. The resume tests
integrate for real and are skipped when rebound/reboundx, the DE430 ephemeris or the example
trajectory are unavailable.
"""

import os
import pickle
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

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_checkpoint", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def exampleTraj(reb):
    """ The example trajectory with a covariance for the clones, skipping if anything is missing. """

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable: {}".format(reb._REBOUND_IMPORT_ERROR))

    from wmpl.Config import config
    from wmpl.Utils.Pickling import loadPickle

    if not os.path.isfile(config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not found: {}".format(config.jpl_ephem_file))

    if not os.path.isfile(os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)):
        pytest.skip("Example trajectory not found")

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)

    # The example has no uncertainties, so give it a covariance (100 m, 1 m/s) for the clones
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([100.0**2]*3 + [1.0**2]*3)

    return traj


### Output size (no REBOUND needed) ###


def testOutputSizeGrowsWithSamplesOrbitsAndWorkers(reb):

    base = reb.estimateOutputSize(500, 101)
    assert base["memory"] == 500*101*reb.OUTPUT_SAMPLE_MEMORY_BYTES
    assert base["json"] == 500*101*reb.OUTPUT_SAMPLE_JSON_BYTES
    assert base["checkpoint"] == 500*101*reb.OUTPUT_SAMPLE_CHECKPOINT_BYTES

    # Each worker holds one orbit and its pickled copy on top of what the main process keeps
    with_workers = reb.estimateOutputSize(500, 101, n_workers=4)
    assert with_workers["memory"] == base["memory"] + 4*500*reb.OUTPUT_SAMPLE_WORKER_BYTES
    assert with_workers["json"] == base["json"]


def testMemoryWarningOnlyWhenTheOutputsFillMostOfTheMemory(reb):

    gb = 1024**3

    # 101 orbits x 500 samples are ~100 MB
    assert reb.outputMemoryWarning(500, 101, available_bytes=16*gb) is None

    # 1001 orbits x 100000 samples are ~190 GB
    text = reb.outputMemoryWarning(100000, 1001, available_bytes=16*gb)
    assert "1001 orbits x 100000 samples" in text
    assert reb.formatBytes(1001*100000*reb.OUTPUT_SAMPLE_MEMORY_BYTES) in text
    assert "--outputs" in text

    # Just below and just above the warning threshold
    need = 1001*100000*reb.OUTPUT_SAMPLE_MEMORY_BYTES
    limit = need/reb.OUTPUT_MEMORY_WARN_FRACTION
    assert reb.outputMemoryWarning(100000, 1001, available_bytes=int(limit*1.01)) is None
    assert reb.outputMemoryWarning(100000, 1001, available_bytes=int(limit*0.99)) is not None

    # Unknown memory: no warning
    assert reb.outputMemoryWarning(100000, 1001, available_bytes=0) is None


@pytest.mark.parametrize("n_bytes, text", [
    (512, "512 B"), (2048, "2.0 kB"), (5*1024**2, "5.0 MB"), (int(3.25*1024**3), "3.2 GB"),
    (2*1024**4, "2.0 TB"),
])
def testFormatBytes(reb, n_bytes, text):

    assert reb.formatBytes(n_bytes) == text


def testAvailableMemoryIsPositive(reb):

    mem = reb.availableMemoryBytes()
    assert (mem is None) or (mem > 0)


### Checkpoint file (no REBOUND needed) ###


def testCheckpointKeepsTheCompleteRecordsOfACutFile(reb, tmp_path):
    """ A record cut while it was being written is dropped, and the run goes on after the last
    complete one. """

    path = str(tmp_path/"checkpoint.pkl")
    signature = {"sim_days": 60.0, "random_seed": 7}

    writer = reb._CheckpointWriter(path, signature)
    writer.add("obj", {"value": 1})
    writer.add("obj_MC_0", {"value": 2})
    full_size = os.path.getsize(path)

    # Half of a third record, as if the run had been killed while writing it
    record = pickle.dumps({"kind": "record", "key": "obj_MC_1", "data": {"value": 3}},
                          protocol=pickle.HIGHEST_PROTOCOL)
    with open(path, "ab") as f:
        f.write(record[:len(record)//2])

    header, records, valid_size = reb.readReboundCheckpoint(path)
    assert header["signature"] == signature
    assert header["version"] == reb.CHECKPOINT_VERSION
    assert records == {"obj": {"value": 1}, "obj_MC_0": {"value": 2}}
    assert valid_size == full_size

    # Continuing cuts the partial record off before appending
    writer = reb._CheckpointWriter(path, signature, valid_size=valid_size)
    writer.add("obj_MC_1", {"value": 3})

    header, records, _ = reb.readReboundCheckpoint(path)
    assert records == {"obj": {"value": 1}, "obj_MC_0": {"value": 2}, "obj_MC_1": {"value": 3}}


def testCheckpointMissingOrWithoutHeaderIsEmpty(reb, tmp_path):

    assert reb.readReboundCheckpoint(str(tmp_path/"missing.pkl")) == (None, {}, 0)

    path = tmp_path/"garbage.pkl"
    path.write_bytes(b"not a pickle at all")
    assert reb.readReboundCheckpoint(str(path)) == (None, {}, 0)


def testCheckpointDifferences(reb):

    saved = {"sim_days": 60.0, "state_vect": [1.0, 2.0, 3.0], "random_seed": 7, "integrator": "ias15"}

    # Equal up to rounding
    current = dict(saved, sim_days=60.0*(1 + 1e-15), state_vect=[1.0, 2.0, 3.0*(1 + 1e-15)])
    assert reb._checkpointDifferences(saved, current) == []

    # Without a seed of its own, a run adopts the one of the checkpoint
    assert reb._checkpointDifferences(saved, dict(saved, random_seed=None)) == []

    differences = reb._checkpointDifferences(
        saved, dict(saved, sim_days=100.0, random_seed=8, integrator="trace"))
    assert differences == [("integrator", "ias15", "trace"), ("random_seed", 7, 8),
                           ("sim_days", 60.0, 100.0)]


### Resuming a real integration ###


def _run(reb, traj, checkpoint_path, **kwargs):

    args = dict(direction="backward", sim_days=60, n_outputs=40, mc_runs=3, n_cpu=1,
                obj_name=traj.traj_id,
                show_progress=False, return_diagnostics=True, random_seed=42,
                checkpoint_path=checkpoint_path)
    args.update(kwargs)

    return reb.reboundSimulate(None, None, traj=traj, **args)


def _countIntegrations(reb, monkeypatch):
    """ Count the calls of _integrateParticles, by the names of the particles integrated. """

    calls = []
    integrate = reb._integrateParticles

    def counted(task, progress=None):
        calls.extend(task["particle_names"])
        return integrate(task, progress=progress)

    monkeypatch.setattr(reb, "_integrateParticles", counted)

    return calls


def testCheckpointHoldsEveryOrbit(reb, exampleTraj, tmp_path):

    path = str(tmp_path/"checkpoint.pkl")
    outputs, outputs_mc, diagnostics = _run(reb, exampleTraj, path)

    header, records, _ = reb.readReboundCheckpoint(path)
    name = exampleTraj.traj_id
    assert set(records) == {name} | set(outputs_mc)
    assert header["signature"]["random_seed"] == 42
    assert header["signature"]["mc_runs"] == 3

    assert records[name]["outputs"][name] == outputs
    for mc_name, rows in outputs_mc.items():
        assert records[mc_name]["outputs"][mc_name] == rows
        assert records[mc_name]["diagnostics"][mc_name] == diagnostics[mc_name]

    # The size estimate of the checkpoint matches what was written to within 25%
    expected = reb.estimateOutputSize(40, 4)["checkpoint"]
    assert os.path.getsize(path) == pytest.approx(expected, rel=0.25)


def testCutRunResumesWithTheSameResults(reb, exampleTraj, tmp_path, monkeypatch):
    """ After a cut, only the orbits missing from the checkpoint are integrated, and the results
    are the same as those of an uninterrupted run. """

    full_path = str(tmp_path/"full.pkl")
    outputs, outputs_mc, diagnostics = _run(reb, exampleTraj, full_path)

    # A run cut after the nominal and the first clone, partway through writing the second one
    header, records, _ = reb.readReboundCheckpoint(full_path)
    name = exampleTraj.traj_id
    cut_path = str(tmp_path/"cut.pkl")
    writer = reb._CheckpointWriter(cut_path, header["signature"])
    writer.add(name, records[name])
    writer.add(name + "_MC_0", records[name + "_MC_0"])
    with open(cut_path, "ab") as f:
        f.write(pickle.dumps({"kind": "record", "key": name + "_MC_1",
                              "data": records[name + "_MC_1"]})[:100])

    calls = _countIntegrations(reb, monkeypatch)

    # No seed given: the one of the checkpoint is used, so the clones are drawn as before
    outputs_r, outputs_mc_r, diagnostics_r = _run(reb, exampleTraj, cut_path, random_seed=None)

    assert calls == [name + "_MC_1", name + "_MC_2"]
    assert outputs_r == outputs
    assert outputs_mc_r == outputs_mc
    assert diagnostics_r == diagnostics

    # The checkpoint is complete again
    assert set(reb.readReboundCheckpoint(cut_path)[1]) == set(records)


def testCheckpointOfOtherParametersIsRefused(reb, exampleTraj, tmp_path):

    path = str(tmp_path/"checkpoint.pkl")
    _run(reb, exampleTraj, path, mc_runs=1)

    with pytest.raises(reb.ReboundCheckpointMismatch) as err:
        _run(reb, exampleTraj, path, mc_runs=1, sim_days=30)

    assert err.value.differences == [("sim_days", 60.0, 30.0)]


def testMegnoIsCheckpointed(reb, exampleTraj, tmp_path, monkeypatch):

    path = str(tmp_path/"checkpoint.pkl")
    megno_calls = []
    compute = reb.computeMegno

    def counted(task, **kwargs):
        megno_calls.append(1)
        return compute(task, **kwargs)

    monkeypatch.setattr(reb, "computeMegno", counted)

    first = _run(reb, exampleTraj, path, mc_runs=1, compute_megno=True)[2]
    second = _run(reb, exampleTraj, path, mc_runs=1, compute_megno=True)[2]

    name = exampleTraj.traj_id
    assert len(megno_calls) == 1
    assert second[name]["megno"] == first[name]["megno"]
