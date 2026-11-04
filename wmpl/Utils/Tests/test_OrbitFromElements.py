""" Tests of --dynamics from published elements: the node the Earth was at, and what the command line refuses.
The checks against published fireballs need rebound and the DE430 ephemeris, and are skipped without them. """

import os
import subprocess
import sys

import numpy as np
import pytest

from wmpl.Rebound import REBOUND as reb
from wmpl.Utils.OrbitClassification import drawElementClones, elementUncertainties, \
    formatJupiterFamilyDynamics, classifyJupiterFamilyDynamicsShober2024
from wmpl.Utils.OrbitFromElements import nodeAtTheEarth, trajectoryFromElements
from wmpl.Utils.TrajConversions import datetime2JD

import datetime


_needsEphemeris = pytest.mark.skipif(not (reb.REBOUND_FOUND and os.path.isfile(reb.config.jpl_ephem_file)),
    reason="rebound or the DE430 ephemeris not available")


def _jd(date):
    return datetime2JD(datetime.datetime.fromisoformat(date))


# a, e, i, node, peri, time, then the published geocentric speed and the node, from the tables of each paper
PUBLISHED = {
    'Trigo-Rodriguez 2009': (4.5, 0.775, 43.8, 109.80489, 187.4, "2008-07-11 21:17:38.9", 27.3, 'descending', 0.001),
    'Madiedo 2014': (4.3, 0.79, 7.1, 203.9450, 40.95, "2013-04-13 21:43:55.4", 15.5, 'ascending', 0.02),
    'Hughes 2022': (3.58, 0.725, 9.06, 202.619, 20.18, "2021-04-13 02:16:44.345", 11.9, 'ascending', 0.02),
}


@_needsEphemeris
@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_published_fireballs_are_at_the_earth_at_their_node_and_speed(name):

    a, e, i, node, peri, date, v_g, which, miss = PUBLISHED[name]

    passage = trajectoryFromElements(a, e, i, node, peri, _jd(date)).node_passage

    assert passage['node'] == which
    assert passage['miss_au'] < miss
    assert abs(passage['v_g']/1000.0 - v_g) < 0.2


@_needsEphemeris
def test_the_true_anomaly_is_that_of_the_node():

    a, e, i, node, peri, date, _, _, _ = PUBLISHED['Trigo-Rodriguez 2009']
    orbit = trajectoryFromElements(a, e, i, node, peri, _jd(date)).orbit

    assert np.degrees(orbit.peri + orbit.true_anomaly)%360 == pytest.approx(180.0)
    assert orbit.q == pytest.approx(a*(1 - e))


@_needsEphemeris
def test_an_orbit_that_does_not_reach_the_earth_is_refused_unless_allowed():

    # The a, e and i of the fireball of Trigo-Rodriguez et al. in Shober et al., with its angles
    args = (2.89, 0.737, 29.9, 109.80489, 187.4, _jd("2008-07-11 21:17:38.9"))

    with pytest.raises(ValueError, match="0.25. AU from the Earth"):
        trajectoryFromElements(*args)

    assert trajectoryFromElements(*args, max_node_miss=0.3).node_passage['miss_au'] == pytest.approx(0.25, abs=0.01)


@_needsEphemeris
def test_the_wrong_date_is_refused():

    a, e, i, node, peri, _, _, _, _ = PUBLISHED['Madiedo 2014']

    with pytest.raises(ValueError, match="from the Earth"):
        trajectoryFromElements(a, e, i, node, peri, _jd("2013-10-13 21:43:55"))


@pytest.mark.parametrize("a, e", [(-3.0, 1.2), (3.0, 1.0), (0.0, 0.5)])
def test_only_an_ellipse_is_taken(a, e, monkeypatch):

    monkeypatch.setattr(reb, 'REBOUND_FOUND', True)

    with pytest.raises(ValueError, match="ellipse"):
        trajectoryFromElements(a, e, 10.0, 20.0, 30.0, 2459000.5)


@_needsEphemeris
def test_the_uncertainties_come_back_in_degrees_and_say_where_they_are_from():

    a, e, i, node, peri, date, _, _, _ = PUBLISHED['Madiedo 2014']
    sigma = {'a': 0.7, 'e': 0.04, 'i': 0.1, 'node': 1e-4, 'peri': 0.03}

    back, source = elementUncertainties(trajectoryFromElements(a, e, i, node, peri, _jd(date), sigma=sigma))

    assert source == "given with the elements"
    for key in sigma:
        assert back[key] == pytest.approx(sigma[key])

    assert elementUncertainties(trajectoryFromElements(a, e, i, node, peri, _jd(date)))[1].startswith("assumed")


### The command line ###

def _cli(*args):

    return subprocess.run([sys.executable, "-W", "ignore", "-m", "wmpl.Utils.OrbitClassification"] + list(args),
        capture_output=True, text=True)


ELEMENTS = ["-a", "4.3", "-e", "0.79", "-i", "7.1", "-p", "40.95", "-n", "203.945"]
DATE = ["--date", "2013-04-13 21:43:55.4"]
SIGMA = ["--sigma_a", "0.7", "--sigma_e", "0.04", "--sigma_i", "0.1", "--sigma_node", "1e-4", "--sigma_peri", "0.03"]


@pytest.mark.parametrize("args, message", [
    (ELEMENTS + DATE + ["--dynamics"], "adapted mode"),
    (ELEMENTS + ["--dynamics", "shober2024"], "needs --q or --a"),
    (["-a", "4.3", "-e", "0.79", "-i", "7.1", "--dynamics", "shober2024"] + DATE, "needs --q or --a"),
    (ELEMENTS + DATE + ["--dynamics", "shober2024", "--sigma_a", "0.7"], "or none"),
    (ELEMENTS + ["--date", "yesterday", "--dynamics", "shober2024"], "YYYY-MM-DD"),
    (["some.pickle", "--dynamics", "shober2024"] + DATE, None),
    ])
def test_what_the_command_line_refuses(args, message):

    result = _cli(*args)

    assert result.returncode != 0
    if message is not None:
        assert message in result.stderr + result.stdout


@_needsEphemeris
def test_the_command_line_integrates_from_the_elements_and_reports_the_node():

    result = _cli(*(ELEMENTS + DATE + SIGMA + ["--dynamics", "shober2024", "--max_clones", "2", "--years", "60",
        "--cores", "1", "--seed", "1"]))

    assert result.returncode == 0, result.stderr
    assert "at the ascending node" in result.stdout
    assert "given with the elements" in result.stdout
    assert "Verdict" in result.stdout


@_needsEphemeris
def test_the_command_line_refuses_elements_that_miss_the_earth():

    result = _cli(*(["-a", "2.89", "-e", "0.737", "-i", "29.9", "-p", "187.4", "-n", "109.80489", "--date",
        "2008-07-11 21:17:38.9", "--dynamics", "shober2024"]))

    assert result.returncode == 1
    assert "from the Earth" in result.stderr + result.stdout


### Drawing q instead of e ###

_ELEMENTS_Q = dict(a=4.5, e=0.775, q=1.0125, i=np.radians(43.8), node=np.radians(109.8), peri=np.radians(187.4),
    true_anomaly=np.radians(-7.4), v_g=27300.0)


def _orbit(**changes):

    from types import SimpleNamespace

    return SimpleNamespace(**dict(_ELEMENTS_Q, **changes))


SIGMA_E = {'a': 0.4, 'e': 0.021, 'i': 0.4, 'node': 1e-5, 'peri': 0.6}
SIGMA_Q = {'a': 0.4, 'q': 0.0006, 'i': 0.4, 'node': 1e-5, 'peri': 0.6}


def test_drawing_q_keeps_it_within_its_error_where_drawing_e_does_not():

    with_e = drawElementClones(_orbit(), SIGMA_E, 2000, np.random.default_rng(1))
    with_q = drawElementClones(_orbit(), SIGMA_Q, 2000, np.random.default_rng(1))

    assert np.std(with_e['a']*(1 - with_e['e'])) > 0.1
    assert np.std(with_q['a']*(1 - with_q['e'])) == pytest.approx(0.0006, rel=0.1)
    assert np.mean(with_q['a']*(1 - with_q['e'])) == pytest.approx(1.0125, abs=1e-4)


def test_drawing_q_leaves_the_other_elements_of_the_clones_as_they_were():

    with_e = drawElementClones(_orbit(), SIGMA_E, 50, np.random.default_rng(1))
    with_q = drawElementClones(_orbit(), SIGMA_Q, 50, np.random.default_rng(1))

    for key in ('a', 'i', 'node', 'peri', 'f'):
        assert np.array_equal(with_e[key], with_q[key])


@pytest.mark.parametrize("sigma_a", [3.0, 9.0])
def test_a_semi_major_axis_drawn_below_q_is_refused(sigma_a):

    with pytest.raises(ValueError, match="no ellipse"):
        drawElementClones(_orbit(), dict(SIGMA_Q, a=sigma_a), 1000, np.random.default_rng(1))


@_needsEphemeris
def test_the_elements_keep_the_error_of_q_and_the_report_says_it_is_not_the_paper():

    a, e, i, node, peri, date, _, _, _ = PUBLISHED['Trigo-Rodriguez 2009']
    traj = trajectoryFromElements(a, e, i, node, peri, _jd(date), sigma=SIGMA_Q)

    sigma, _ = elementUncertainties(traj)
    assert 'e' not in sigma and sigma['q'] == 0.0006

    result = classifyJupiterFamilyDynamicsShober2024(traj, n_clones=2, years=60.0, seed=1, n_cpu=1)
    report = "\n".join(formatJupiterFamilyDynamics(result))
    assert "a, q, i, node, peri drawn independently and e = 1 - q/a, unlike the paper" in report
    assert "q 0.0006 AU" in report


@pytest.mark.parametrize("args, message", [
    (ELEMENTS + DATE + ["--dynamics", "shober2024"] + SIGMA + ["--sigma_q", "0.001"], "either --sigma_e or --sigma_q"),
    (ELEMENTS + DATE + ["--dynamics", "shober2024", "--sigma_q", "0.001"], "or none"),
    ])
def test_what_the_command_line_refuses_of_sigma_q(args, message):

    result = _cli(*args)

    assert result.returncode == 1
    assert message in result.stderr + result.stdout


def test_sigma_q_goes_with_the_elements_and_not_with_a_pickle():

    example = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dynesty", "examples",
        "20191023_091225", "20191023_091225_trajectory.pickle")

    result = _cli(example, "--dynamics", "shober2024", "--sigma_q", "0.001")

    assert result.returncode == 1
    assert "--sigma_q goes with the elements" in result.stderr + result.stdout
