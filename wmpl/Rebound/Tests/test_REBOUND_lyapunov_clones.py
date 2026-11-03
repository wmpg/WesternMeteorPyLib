""" Tests for the rescaling of the MEGNO deviation at its real scale, and for the Lyapunov time over the
clones of a run.

The summaries, the report and the figure are pure numerics on synthetic values, so they run without
REBOUND or REBOUNDx installed. The integrations skip without them.
"""

import os
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

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_lyapunov_clones", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def realReb(reb):
    """ The REBOUND module, skipping the test if rebound/reboundx cannot be imported. """

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable: {}".format(reb._REBOUND_IMPORT_ERROR))

    return reb


def _chaoticTask(reb, span_years, n_outputs):
    """ A particle in the chaotic zone just outside Jupiter's orbit around a Sun with only Jupiter: it
    has a Lyapunov time of tens of years and takes well under a second per century. """

    import rebound as rb

    sim = rb.Simulation()
    sim.add(m=1.0)
    sim.add(primary=sim.particles[0], m=9.5479e-4, a=5.2, e=0.048)
    sim.add(primary=sim.particles[0], a=4.5, e=0.05, inc=0.02, omega=1.0, Omega=2.0, f=3.0)

    states = [[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles]

    return {"planet_names": ["Sun", "Jupiter"], "planet_states": states[:2], "planet_masses": [1.0, 9.5479e-4],
            "particle_states": [states[2]], "direction": "forward",
            "times": [reb.daysToSimTime(t) for t in np.linspace(0, span_years*365.25, n_outputs)]}


def testRescalingAtItsOwnLimitAddsNoJump(realReb, monkeypatch):
    """ At the limit in use, ln|delta| right after the first rescale is that of a run without rescaling.
    IAS15 keeps the round-off of its compensated summation at the old scale, so the test has to reach the
    limit: at a limit of 1e20 the same difference was about 8. """

    limit = realReb.MEGNO_RESCALE_ABOVE
    task = _chaoticTask(realReb, 1500.0, 3000)

    rescaled = np.array(realReb.computeMegno(task)["ln_delta"])

    monkeypatch.setattr(realReb, "MEGNO_RESCALE_ABOVE", 1e300)
    unscaled = np.array(realReb.computeMegno(task)["ln_delta"])

    # The run has to reach the limit for this to test anything
    first = int(np.argmax(unscaled > np.log(limit)))
    assert unscaled[first] > np.log(limit)

    # The orbit itself is the same to round-off until then, so the series can be compared there
    assert abs(rescaled[first + 1] - unscaled[first + 1]) < 1e-5

    # No output-to-output step stands out at the rescales
    assert np.max(np.abs(np.diff(rescaled)[first:first + 5])) < 3.0


def testSummaryOfTheClones(reb):
    """ Median and percentiles over the clones with a time, and what doubling the span changes. """

    results = [{"status": "chaotic", "lyapunov_time_years": tau, "half_run_lyapunov_time_years": 0.8*tau}
               for tau in (10.0, 20.0, 30.0, 40.0, 50.0)]
    results.append({"status": "regular", "lyapunov_time_years": None, "half_run_lyapunov_time_years": None})

    s = reb.summariseCloneLyapunov(results)

    assert s["n_clones"] == 6
    assert s["status_counts"] == {"chaotic": 5, "regular": 1}
    assert s["lyapunov_times_years"] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert s["lyapunov_time_median_years"] == pytest.approx(30.0)
    assert s["lyapunov_time_p16_years"] == pytest.approx(np.percentile([10, 20, 30, 40, 50], 16))
    assert s["lyapunov_time_p84_years"] == pytest.approx(np.percentile([10, 20, 30, 40, 50], 84))
    assert s["half_run_median_years"] == pytest.approx(24.0)
    assert s["change_on_doubling"] == pytest.approx(0.2)


def testSummaryWithoutTimes(reb):
    """ Regular clones have no time; too few times over the first half give no comparison. """

    s = reb.summariseCloneLyapunov([{"status": "regular", "lyapunov_time_years": None,
                                     "half_run_lyapunov_time_years": None}]*3)
    assert s["lyapunov_time_median_years"] is None
    assert s["change_on_doubling"] is None

    s = reb.summariseCloneLyapunov(
        [{"status": "chaotic", "lyapunov_time_years": 30.0, "half_run_lyapunov_time_years": None}]*3
        + [{"status": "chaotic", "lyapunov_time_years": 30.0, "half_run_lyapunov_time_years": 10.0}])
    assert s["lyapunov_time_median_years"] == pytest.approx(30.0)
    assert s["half_run_median_years"] is None
    assert s["change_on_doubling"] is None


def _megnoWith(reb, clones):
    """ A regular MEGNO result with these clones. """

    t = np.linspace(100.0, 365250.0, 200)
    verdict = reb.classifyMegno(t, np.full(200, 2.0), np.full(200, 2.0), ln_delta=np.log(t))

    return {"times_days": list(-t), "megno": [2.0]*200, "a_au": [2.0]*200, "ln_delta": list(np.log(t)),
            "verdict": verdict, "clones": clones}


def testReportGivesTheClones(reb):
    """ The verdicts, the median to quote, and whether it holds when the span doubles. """

    settled = reb.summariseCloneLyapunov(
        [{"status": "chaotic", "lyapunov_time_years": tau, "half_run_lyapunov_time_years": 0.95*tau}
         for tau in (20.0, 40.0, 60.0)] + [{"status": "not_converged", "lyapunov_time_years": None,
                                            "half_run_lyapunov_time_years": None}])
    text = "\n".join(reb._megnoReportLines(_megnoWith(reb, settled)))

    assert "The same over 4 clones of the run:" in text
    assert "Verdicts: 3 chaotic, 1 not converged." in text
    assert "Lyapunov time: median 40 years (16-84% of the clones: 26-54), from the 3 with a time." in text
    assert "so this is the value to quote" in text
    assert "From the first half of the run the median is 38 years: it changes by 5% when the span doubles," in text
    assert "so it has settled." in text

    drifting = reb.summariseCloneLyapunov(
        [{"status": "chaotic", "lyapunov_time_years": tau, "half_run_lyapunov_time_years": 0.6*tau}
         for tau in (20.0, 30.0, 40.0)])
    text = "\n".join(reb._megnoReportLines(_megnoWith(reb, drifting)))

    assert "it changes by 40% when the span doubles," in text
    assert "so this is a value for this span, not yet a limit." in text


def testReportWithoutClonesIsUnchanged(reb):
    """ A run without clones has no such lines, and regular clones only give their verdicts. """

    assert "clones of the run" not in "\n".join(reb._megnoReportLines(_megnoWith(reb, None)))

    regular = reb.summariseCloneLyapunov([{"status": "regular", "lyapunov_time_years": None,
                                           "half_run_lyapunov_time_years": None}]*2)
    text = "\n".join(reb._megnoReportLines(_megnoWith(reb, regular)))

    assert "Verdicts: 2 regular." in text
    assert "value to quote" not in text


def testFigureDrawsTheClones(reb):
    """ The lower panel gets the clones' median and their 16-84% band, within its limits. """

    import matplotlib.pyplot as plt

    t = np.linspace(0.5, 1000.0, 2000)
    ln_delta = np.where(t > 200.0, (t - 200.0)/40.0, 0.0)
    y = np.where(t > 200.0, 10.0, 2.0)
    verdict = reb.classifyMegno(list(-t*365.25), list(y), [2.0]*2000, ln_delta=list(ln_delta))
    clones = {"n_clones": 16, "lyapunov_time_median_years": 25.0, "lyapunov_time_p16_years": 15.0,
              "lyapunov_time_p84_years": 90.0}

    fig, axes = reb.plotMegno({"times_days": list(-t*365.25), "megno": list(y), "a_au": [2.0]*2000,
                               "ln_delta": list(ln_delta), "verdict": verdict, "clones": clones})

    labels = axes[-1].get_legend_handles_labels()[1]
    assert "median of 16 clones: 25 years" in labels
    assert "16-84% of the clones" in labels

    low, high = axes[-1].get_ylim()
    assert (low < 15.0) and (high > 90.0)

    plt.close(fig)


def _exampleWithCovariance():
    from wmpl.Config import config
    from wmpl.Utils.Pickling import loadPickle

    if not os.path.isfile(config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not found: {}".format(config.jpl_ephem_file))
    if not os.path.isfile(os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)):
        pytest.skip("Example trajectory not found")

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([100.0**2]*3 + [1.0**2]*3)

    return traj


def testRunGivesTheClonesOnlyWithAVerdict(realReb, monkeypatch):
    """ A run of 60 days is too short for a verdict, so the clones are not integrated for nothing; a long
    enough one gives them. """

    monkeypatch.setattr(realReb, "MEGNO_CLONES", 2)
    traj = _exampleWithCovariance()

    def run(days):
        _, _, diag = realReb.reboundSimulate(
            None, None, traj=traj, direction="backward", sim_days=days, n_outputs=100, mc_runs=3, n_cpu=1,
            show_progress=False, return_diagnostics=True, random_seed=3, compute_megno=True,
            obj_name=traj.traj_id, gr="gr")
        return diag[traj.traj_id]["megno"]

    assert run(60).get("clones") is None

    # This example has a period of decades, so the shortest span that gets a verdict is lowered for the test
    monkeypatch.setattr(realReb, "MEGNO_MIN_ORBITS", 1)
    clones = run(300*365.25)["clones"]
    assert clones["n_clones"] == 2
    assert sum(clones["status_counts"].values()) == 2
