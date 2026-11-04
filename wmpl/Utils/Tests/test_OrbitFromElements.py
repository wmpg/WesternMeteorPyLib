""" Tests of --dynamics from published elements: the node the Earth was at, and what the command line refuses.
The checks against published fireballs need rebound and the DE430 ephemeris, and are skipped without them. """

import os
import subprocess
import sys

import numpy as np
import pytest

from wmpl.Rebound import REBOUND as reb
from wmpl.Utils.OrbitClassification import elementUncertainties, formatJupiterFamilyDynamics, \
    classifyJupiterFamilyDynamicsShober2024
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
