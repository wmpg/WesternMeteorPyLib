""" Tests for REBOUND.py --meteomod, the source regions of the integrated orbit among the asteroid families
of METEOMOD (see wmpl.Utils.METEOMODSourceRegions), alone and with --neomod.

The command line is run on the example trajectory of the repository. Its orbit (a = 11 AU, i = 165 deg)
lies outside both models, so the results do not depend on which model files are read: small ones in the
formats of the authors are put in the cache of a temporary home directory, which is used when the
published files have not been downloaded to wmpl/share, so that the tests never download them. They are
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

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_meteomod", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module



@pytest.fixture
def runReboundCli(reb, tmp_path):
    """ Run the REBOUND.py command line on a copy of the example trajectory, with a home directory whose
        cache holds small METEOMOD and NEOMOD files. Returns a function of the arguments that gives the
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

    from wmpl.Utils.Tests.SyntheticMETEOMOD import writeArchive
    from wmpl.Utils.Tests.SyntheticNEOMOD import writeModel, shares
    from wmpl.Utils.METEOMODSourceRegions import METEOMOD
    from wmpl.Utils.NEOMODSourceRegions import NEOMOD2

    home = tmp_path/"home"
    cache = home/".cache"/"wmpl"
    cache.mkdir(parents=True)
    writeArchive(str(cache/METEOMOD['file_name']))
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



def testEveryOutputIsSavedWithItsReasonAndTheNeomodBeside(runReboundCli):
    """ With --meteomod the source regions of every output go to the JSON, the CSV, the figure and the
        report, an orbit outside the maps says why, and --neomod gives its own beside them. """

    done, run_dir = runReboundCli("--meteomod", "--neomod")
    assert done.returncode == 0, done.stderr[-1000:]

    with open(str(run_dir/"rebound_simulation_results.json")) as f:
        results = json.load(f)

    meteomod = results["meteomod"]
    assert meteomod["mode"] == "all"
    assert meteomod["weight"] == "flux"
    assert len(meteomod["time_days"]) == len(results["nominal"]["time_days"])
    assert all(v is None for values in meteomod["nominal"].values() for v in values)
    assert "doi:10.1051/0004-6361/202450532" in meteomod["reference"]

    # The reason at the end is given with the final elements of the run
    a_final = results["final_elements"]["a"]
    assert "a = {:.3f} AU is beyond the maps".format(a_final) in meteomod["nominal_reason"][-1]

    # NEOMOD's results beside, under their own key
    assert results["neomod"]["mode"] == "all"

    with open(str(run_dir/"rebound_meteomod_sources.csv")) as f:
        assert len(f.read().splitlines()) == 1 + len(meteomod["sources"])*len(meteomod["time_days"])

    assert (run_dir/"rebound_meteomod_sources.png").exists()
    assert (run_dir/"rebound_neomod_sources.png").exists()

    with open(str(run_dir/"rebound_simulation_results.txt")) as f:
        report = f.read()
    assert "Source regions in METEOMOD" in report
    assert "Source regions in NEOMOD2" in report
    assert "Please cite Broz" in report

    assert "Saved METEOMOD" in done.stdout



def testTheEndOnlyAmongTheFamiliesOfAType(runReboundCli):
    """ --meteomod final with --meteomod_types gives the last output only, among the families of that
        type, without a figure. """

    done, run_dir = runReboundCli("--meteomod", "final", "--meteomod_types", "CM")
    assert done.returncode == 0, done.stderr[-1000:]

    with open(str(run_dir/"rebound_simulation_results.json")) as f:
        meteomod = json.load(f)["meteomod"]

    assert meteomod["mode"] == "final"
    assert meteomod["types"] == ["CM"]
    assert len(meteomod["time_days"]) == 1
    assert meteomod["sources"]
    assert set(meteomod["family_types"].values()) == {"CM"}
    assert not (run_dir/"rebound_meteomod_sources.png").exists()



def testTheSelectionNeedsMeteomodAndKnownFamilies(runReboundCli):
    """ --meteomod_types alone, or a type without families, stops before integrating. """

    done, _ = runReboundCli("--meteomod_types", "CM")
    assert (done.returncode != 0) and ("go with --meteomod" in done.stderr)

    done, _ = runReboundCli("--meteomod", "--meteomod_types", "XYZ")
    assert (done.returncode != 0) and ("No family of METEOMOD is of type XYZ" in done.stderr)
