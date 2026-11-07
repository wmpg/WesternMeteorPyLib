""" Tests for REBOUND.py --neomod, the source regions of the integrated orbit in NEOMOD2 (see
wmpl.Utils.NEOMODSourceRegions).

The command line is run on the example trajectory of the repository. Its orbit (a = 11 AU, i = 165 deg)
lies outside the model, so the results do not depend on which model file is read: a small one in the
format of the NEOMOD Simulator is put in the cache of a temporary home directory, which is used when
the published file has not been downloaded to wmpl/share, so that the tests never download it. They are
skipped when rebound/reboundx, the DE430 ephemeris or the example trajectory are unavailable.
"""

import os
import sys
import json
import subprocess
import importlib.util

import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dynesty",
    "examples", "20191023_091225")
EXAMPLE_FILE = "20191023_091225_trajectory.pickle"



@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_neomod", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module



@pytest.fixture
def runReboundCli(reb, tmp_path):
    """ Run the REBOUND.py command line on a copy of the example trajectory, with a home directory
        whose cache holds a small NEOMOD model file. Returns a function of the arguments that gives the
        completed process and the directory of the outputs. """

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable: {}".format(reb._REBOUND_IMPORT_ERROR))

    try:
        reb.checkReboundxUsable()
    except RuntimeError as e:
        pytest.skip(str(e).split(".")[0])

    if not os.path.isfile(reb.config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not available")

    pickle_path = os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)
    if not os.path.isfile(pickle_path):
        pytest.skip("example trajectory not available")

    from wmpl.Utils.Tests.SyntheticNEOMOD import writeModel, shares
    from wmpl.Utils.NEOMODSourceRegions import NEOMOD2

    home = tmp_path/"home"
    cache = home/".cache"/"wmpl"
    cache.mkdir(parents=True)
    writeModel(str(cache/NEOMOD2['file_name']), [(52, 27, 16, 1, 1.0, shares(_3_1=1.0))])

    run_dir = tmp_path/"run"
    run_dir.mkdir()
    with open(pickle_path, 'rb') as f_in, open(str(run_dir/EXAMPLE_FILE), 'wb') as f_out:
        f_out.write(f_in.read())

    repo = os.path.dirname(os.path.dirname(os.path.dirname(REBOUND_PATH)))
    env = dict(os.environ, MPLBACKEND="Agg", HOME=str(home),
        PYTHONPATH=repo + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def run(*args):
        done = subprocess.run([sys.executable, "-m", "wmpl.Rebound.REBOUND", str(run_dir/EXAMPLE_FILE),
            "--days", "20", "--outputs", "6", "--no_moid", *args], cwd=repo, env=env, capture_output=True,
            text=True, timeout=600)
        return done, run_dir

    return run



def testEveryOutputIsSavedWithItsReason(runReboundCli):
    """ With --neomod the source regions of every output go to the JSON, the CSV, the figure and the
        report, and an orbit outside the model says why, with the elements of the run. """

    done, run_dir = runReboundCli("--neomod")
    assert done.returncode == 0, done.stderr[-1000:]

    with open(str(run_dir/"rebound_simulation_results.json")) as f:
        results = json.load(f)

    neomod = results["neomod"]
    assert neomod["mode"] == "all"
    assert neomod["h_bin"] == [27.75, 28.0]
    assert len(neomod["time_days"]) == len(results["nominal"]["time_days"])
    assert all(v is None for v in neomod["nominal"]["3:1"])

    # The reason at the end is given with the final elements of the run
    a_final = results["final_elements"]["a"]
    assert "a = {:.3f} AU is beyond the model".format(a_final) in neomod["nominal_reason"][-1]

    with open(str(run_dir/"rebound_neomod_sources.csv")) as f:
        assert len(f.read().splitlines()) == 1 + 12*len(neomod["time_days"])

    assert (run_dir/"rebound_neomod_sources.png").exists()

    # Without clones there is neither a horizon nor a representative clone
    assert neomod["horizon"] is None
    assert neomod["representative"] is None

    # The report warns that the size is extrapolated and gives the papers
    assert "no size was given" in neomod["h_warning"]
    assert "doi:10.1016/j.icarus.2023.115922" in neomod["reference"]

    with open(str(run_dir/"rebound_simulation_results.txt")) as f:
        report = f.read()
    assert "Source regions in NEOMOD2" in report
    assert "WARNING: no size was given" in report
    assert "Please cite Nesvorny" in report

    assert "Saved NEOMOD" in done.stdout



def testTheEndOnlyWithTheSizeGiven(runReboundCli):
    """ --neomod final gives the last output only, at the H of the size given, without a figure. """

    done, run_dir = runReboundCli("--neomod", "final", "--neomod_size", "19m")
    assert done.returncode == 0, done.stderr[-1000:]

    with open(str(run_dir/"rebound_simulation_results.json")) as f:
        neomod = json.load(f)["neomod"]

    assert neomod["mode"] == "final"
    assert len(neomod["time_days"]) == 1
    assert neomod["h_bin"] == [26.25, 26.5]
    assert not neomod["h_extrapolated"]
    assert neomod["h_warning"] is None
    assert not (run_dir/"rebound_neomod_sources.png").exists()



def testTheSizeNeedsNeomodAndAValidValue(runReboundCli):
    """ --neomod_size alone, or with a value that is not a size, stops before integrating. """

    done, _ = runReboundCli("--neomod_size", "19m")
    assert (done.returncode != 0) and ("--neomod_size goes with --neomod" in done.stderr)

    done, _ = runReboundCli("--neomod", "--neomod_size", "19x")
    assert (done.returncode != 0) and ("--neomod_size" in done.stderr)
