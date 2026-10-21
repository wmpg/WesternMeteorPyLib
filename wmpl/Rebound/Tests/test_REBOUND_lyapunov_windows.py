""" Tests for the windows of the Lyapunov fit to ln|delta|: the growth phase is split into quarters,
each fitted on its own, and the report lists every quarter so that the range can be traced back.

Pure numerics on synthetic series, so they run without REBOUND or REBOUNDx installed.
"""

import os
import importlib.util

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_lyapunov_windows", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _steppedGrowth():
    """ Outputs every half year over 1000 years, <Y> above 2.5 from 200 years on, and ln|delta| that
    grows with a Lyapunov time of 20, 50 and 100 years over the first three quarters of that phase and
    shrinks over the last one. Returns times (years), ln|delta| and <Y>. """

    t = np.linspace(0.5, 1000.0, 2000)
    rate = np.select([t <= 200.0, t <= 400.0, t <= 600.0, t <= 800.0],
                     [0.0, 1/20.0, 1/50.0, 1/100.0], -1/50.0)
    ln_delta = np.concatenate([[0.0], np.cumsum(rate[1:]*np.diff(t))])
    megno = np.where(t > 200.0, 10.0, 2.0)

    return t, ln_delta, megno


def testWindowsAreTheQuartersOfTheGrowthPhase(reb):
    """ Four consecutive windows of equal length, from the onset of the growth to the end of the run,
    each with the time of its own segment. """

    t, ln_delta, megno = _steppedGrowth()

    fit = reb.lyapunovFromDeviation(t, ln_delta, megno, 1.0)
    windows = fit["windows"]

    assert len(windows) == 4
    assert windows[0]["start_years"] == pytest.approx(fit["onset_years"])
    assert windows[0]["start_years"] == pytest.approx(200.5)
    assert windows[-1]["end_years"] == pytest.approx(1000.0)

    lengths = [w["end_years"] - w["start_years"] for w in windows]
    assert max(lengths) - min(lengths) <= 0.5 + 1e-9
    for prev, nxt in zip(windows[:-1], windows[1:]):
        assert nxt["start_years"] == pytest.approx(prev["end_years"] + 0.5)

    for w, tau in zip(windows[:3], (20.0, 50.0, 100.0)):
        assert w["lyapunov_time_years"] == pytest.approx(tau, rel=1e-6)


def testWindowWithoutGrowthIsListedButLeftOutOfTheRange(reb):
    """ The last quarter shrinks: it has no time, and the range comes from the other three. """

    t, ln_delta, megno = _steppedGrowth()

    fit = reb.lyapunovFromDeviation(t, ln_delta, megno, 1.0)

    assert fit["windows"][-1]["lyapunov_time_years"] is None
    assert fit["lyapunov_time_range_years"] == pytest.approx([20.0, 100.0], rel=1e-6)


def testVerdictCarriesTheWindows(reb):
    """ classifyMegno passes the windows on; the legacy fit to <Y> has none. """

    t, ln_delta, megno = _steppedGrowth()
    a = [1.0]*len(t)

    v = reb.classifyMegno(list(-t*365.25), list(megno), a, ln_delta=list(ln_delta))
    assert v["status"] == "chaotic"
    assert len(v["lyapunov_windows"]) == 4

    legacy = reb.classifyMegno(list(-t*365.25), list(2.0 + t/40.0), a)
    assert legacy["lyapunov_method"] == "megno_second_half"
    assert legacy["lyapunov_windows"] is None


def _report(reb, t, ln_delta, megno, sign):

    times_days = list(sign*t*365.25)
    a = [1.0]*len(t)
    result = {"times_days": times_days, "megno": list(megno), "ln_delta": list(ln_delta), "a_au": a,
              "escaped": False,
              "verdict": reb.classifyMegno(times_days, list(megno), a, ln_delta=list(ln_delta))}

    return reb._megnoReportLines(result)


def testReportListsEveryQuarterOfABackwardRun(reb):

    lines = _report(reb, *_steppedGrowth(), sign=-1)
    text = "\n".join(lines)

    assert "from 200 to 1000 years back" in text
    assert "since <Y> last exceeded 2.5" in text
    assert any(line.strip() == "200-400 years back: 20 years" for line in lines)
    assert any(line.strip() == "400-600 years back: 50 years" for line in lines)
    assert any(line.strip() == "600-800 years back: 100 years" for line in lines)
    assert any(line.strip() == "800-1000 years back: no growth" for line in lines)
    assert "four windows" not in text

    # The 1 sigma from the scatter of the quarters' rates, and the fit from the onset to each quarter's end
    v = reb.classifyMegno(*[list(x) for x in (-_steppedGrowth()[0]*365.25, _steppedGrowth()[2], [1.0]*2000)],
                          ln_delta=list(_steppedGrowth()[1]))
    assert reb.lyapunovSigmaText(v["lyapunov_time_years"], v["lyapunov_time_1sigma_years"]) in text
    cum = ", ".join("{:.0f}".format(c["lyapunov_time_years"]) for c in v["lyapunov_cumulative"])
    assert "Fitted from the onset to the end of each quarter: {:s} years.".format(cum) in text


def testReportCountsTheQuartersFromTheStartOfAForwardRun(reb):

    text = "\n".join(_report(reb, *_steppedGrowth(), sign=1))

    assert "from 200 to 1000 years after the start" in text
    assert "200-400 years after the start: 20 years" in text
    assert "back" not in text


def testReportSaysWhenThereAreTooFewOutputsForQuarters(reb):
    """ Fewer than 40 outputs in the growth phase: a time, but no quarters and no range. """

    t = np.linspace(10.0, 1000.0, 100)
    ln_delta = np.maximum(t - 700.0, 0.0)/30.0
    megno = np.where(t > 700.0, 10.0, 2.0)

    fit = reb.lyapunovFromDeviation(t, ln_delta, megno, 1.0)
    assert fit["lyapunov_time_years"] == pytest.approx(30.0, rel=1e-6)
    assert fit["windows"] is None
    assert fit["lyapunov_time_range_years"] is None

    text = "\n".join(_report(reb, t, ln_delta, megno, sign=-1))
    assert "Lyapunov time ~ 30 years," in text
    assert "Too few outputs in the growth phase" in text
    assert "1 sigma" not in text and "Fitted from the onset to the end" not in text


@pytest.fixture
def agg():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
    yield
    plt.close("all")


def testMegnoFigureShowsEachQuarter(reb, agg, tmp_path):
    """ With ln|delta|, the figure has a second panel with the fit over the growth phase and one per
    quarter, each labelled with its time or "no growth"; it is saved. """

    t, ln_delta, megno = _steppedGrowth()
    times_days = list(-t*365.25)
    a = [1.0]*len(t)
    result = {"times_days": times_days, "megno": list(megno), "ln_delta": list(ln_delta), "a_au": a,
              "escaped": False,
              "verdict": reb.classifyMegno(times_days, list(megno), a, ln_delta=list(ln_delta))}

    fig, axes = reb.plotMegno(result, file_path=str(tmp_path/"megno.png"))
    assert (tmp_path/"megno.png").exists()
    assert len(axes) == 3
    assert axes[1].get_ylabel() == "$\\ln|\\delta|$  (each unit: the deviation $\\times$ 2.72)"
    assert axes[0].get_title() == "MEGNO and Lyapunov time of the nominal orbit (Newtonian gravity): chaotic"

    # The third panel: the time fitted from the onset, the quarters' ends on it, and the time with its 1 sigma
    v = result["verdict"]
    dots = [x for x in axes[2].get_lines() if x.get_label() == "cumulative quarters"][0]
    assert np.allclose(dots.get_ydata(), [c["lyapunov_time_years"] for c in v["lyapunov_cumulative"]])
    notes = [x.get_text() for x in axes[2].texts if x.get_gid() == "cumulative"]
    assert notes[-1] == "{:.0f} yr (whole fit)".format(v["lyapunov_cumulative"][-1]["lyapunov_time_years"])

    # The start of the fit is drawn at the same time in all three panels
    for ax in axes:
        starts = [x for x in ax.get_lines() if x.get_linestyle() == "-." and len(x.get_xdata()) == 2
                  and x.get_xdata()[0] == pytest.approx(-v["lyapunov_onset_years"])]
        assert starts, ax.get_ylabel()
    final = [x for x in axes[2].get_lines() if x.get_label().startswith("Lyapunov time")][0]
    assert final.get_ydata()[0] == pytest.approx(v["lyapunov_time_years"])
    band = [c for c in axes[2].patches + axes[2].collections if c.get_label() == "1$\\sigma$"][0]
    xy = band.get_paths()[0].vertices
    ys = xy[:, 1]

    # Over the fitted span only, not before the start of the fit
    assert min(xy[:, 0]) == pytest.approx(-1000.0, abs=1.0) and max(xy[:, 0]) == pytest.approx(-v["lyapunov_onset_years"])
    assert "growth phase (fitted)" in axes[2].get_legend_handles_labels()[1]
    assert axes[2].get_title() == ""
    assert min(ys) == pytest.approx(v["lyapunov_time_1sigma_years"][0])
    assert max(ys) == pytest.approx(v["lyapunov_time_1sigma_years"][1])

    notes = {text.get_text() for text in axes[1].texts if text.get_gid() == "quarter"}
    assert notes == {"20 yr", "50 yr", "100 yr", "no growth"}

    # The triangle under the fit gives the time as the span over the rise of ln|delta|
    tau = result["verdict"]["lyapunov_time_years"]
    slope = [text.get_text() for text in axes[1].texts if text.get_gid() == "slope"]
    assert len(slope) == 3
    rise = [x for x in axes[1].texts if x.get_gid() == "slope"][1]
    assert rise.get_rotation() == 90, "the rise runs along the vertical leg"
    d_t = float(slope[0].split("= ")[1].split(" yr")[0])
    d_ln = float(slope[1].split("= ")[1])
    assert d_t == pytest.approx(1000.0 - result["verdict"]["lyapunov_onset_years"], abs=1.0)
    assert d_t/d_ln == pytest.approx(tau, rel=0.01)
    assert slope[2].endswith("= {:.0f} yr".format(tau))

    # The limits of the verdicts are in the legend of <Y>
    top = axes[0].get_legend_handles_labels()[1]
    assert "regular if it ends within 0.1 of 2" in top
    assert "chaotic above 2.5; the fit starts at the last crossing" in top

    labels = axes[1].get_legend_handles_labels()[1]
    sigma = reb.lyapunovSigmaText(tau, result["verdict"]["lyapunov_time_1sigma_years"]).replace("1 sigma", "1$\\sigma$")
    assert "Lyapunov time {:.0f} years{:s}".format(tau, sigma) in labels
    assert "each quarter" in labels


def testMegnoFigureWithoutTheDeviationHasOnePanel(reb, agg):

    t = np.linspace(0.5, 1000.0, 2000)
    times_days = list(-t*365.25)
    result = {"times_days": times_days, "megno": [2.0]*len(t), "a_au": [1.0]*len(t), "escaped": False,
              "verdict": reb.classifyMegno(times_days, [2.0]*len(t), [1.0]*len(t))}

    _, axes = reb.plotMegno(result)
    assert len(axes) == 1
    assert axes[0].get_title() == "MEGNO of the nominal orbit (Newtonian gravity): regular"


def testCumulativeSlopesAreTheLeastSquaresFits(reb):

    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(100.0, 1000.0, 300))
    y = 0.03*t + rng.normal(size=300)
    slopes = reb.cumulativeSlopes(t, y)

    assert np.all(np.isnan(slopes[:2]))
    for k in (2, 10, 150, 299):
        assert slopes[k] == pytest.approx(np.polyfit(t[:k + 1], y[:k + 1], 1)[0], rel=1e-9)


def testOneSigmaOfTheTimeComesFromTheQuartersRates(reb):
    """ The rate's standard error is the quarters' scatter over sqrt(4); the time's interval is its
    inverse, asymmetric, and without an upper limit when the rate is within 1 sigma of 0. """

    t, ln_delta, megno = _steppedGrowth()
    fit = reb.lyapunovFromDeviation(t, ln_delta, megno, 1.0)

    on = t > 200.0
    slope = np.polyfit(t[on], ln_delta[on], 1)[0]
    se = np.std([1/20.0, 1/50.0, 1/100.0, -1/50.0], ddof=1)/2.0
    assert fit["lyapunov_time_1sigma_years"] == pytest.approx([1/(slope + se), 1/(slope - se)], rel=1e-3)
    assert [c["lyapunov_time_years"] for c in fit["cumulative"]] == pytest.approx(
        [1/np.polyfit(t[on][:k], ln_delta[on][:k], 1)[0] for k in (400, 800, 1200, 1600)], rel=1e-9)

    tau = fit["lyapunov_time_years"]
    lo, hi = fit["lyapunov_time_1sigma_years"]
    assert reb.lyapunovSigmaText(tau, [lo, hi]) == " (+{:.0f}/-{:.0f}, 1 sigma)".format(hi - tau, tau - lo)
    assert reb.lyapunovSigmaText(54.0, [25.0, None]).startswith(" (-29/+inf, 1 sigma")
    assert reb.lyapunovSigmaText(54.0, None) == ""


def testReportSaysWhetherTheTimeHasSettled(reb):
    """ A steady exponential settles; the stepped growth, whose fit from the onset still moves over the
    last quarter, does not. """

    t = np.linspace(0.5, 1000.0, 2000)
    steady = np.where(t > 200.0, (t - 200.0)/40.0, 0.0)
    megno = np.where(t > 200.0, 10.0, 2.0)
    text = "\n".join(_report(reb, t, steady, megno, sign=-1))
    assert "Fitted from the onset to the end of each quarter: 40, 40, 40, 40 years." in text
    assert "It changes by 0% over the last quarter: settled." in text

    text = "\n".join(_report(reb, *_steppedGrowth(), sign=-1))
    assert "so it has not settled: integrate for longer." in text
