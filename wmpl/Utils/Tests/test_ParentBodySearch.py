""" Tests for the parent body search: several D criteria at once, the criteria on the geocentric
radiant against bodies that have only elements, the files and the plot it saves, and bringing
the bodies to the reference epoch of a meteor.

The search runs on a small catalogue made here, so that the tests do not depend on the catalogue
files that are downloaded separately.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_ParentBodySearch.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_ParentBodySearch
"""

import csv
import json
import os
import tempfile

import numpy as np

import wmpl.Utils.ParentBodySearch as pbs
from wmpl.Utils.Dcriteria import (calcGeocentricFromElements, calcHeliocentricFromElements,
    calcHeliocentricFromGeocentric, calcDN, calcDV, SPEED_EARTH)
from wmpl.Utils.TrajConversions import correctedEclipticCoord


# The Geminid shower radiant at its peak and the elements of (3200) Phaethon, its parent body
GEMINID_RADIANT = (np.radians(112.3), np.radians(32.3), np.radians(262.1), 33.8)
PHAETHON = (0.1400, 0.8900, 22.25, 322.2, 265.2)


def _elementsFromState(r_vec, v_vec):
    """ Heliocentric elements through a state, for GM = 1: 'q', 'e', 'incl', 'node', 'peri' (radians).
    """

    r = np.sqrt(r_vec @ r_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.sqrt(h_vec @ h_vec)
    e_vec = np.cross(v_vec, h_vec) - r_vec/r
    e = np.sqrt(e_vec @ e_vec)

    node_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    node_mag = np.sqrt(node_vec @ node_vec)

    peri = np.arccos(np.clip((node_vec @ e_vec)/(node_mag*e), -1.0, 1.0))
    if e_vec[2] < 0:
        peri = 2*np.pi - peri

    return {'q': h**2/(1.0 + e), 'e': e, 'incl': np.arccos(np.clip(h_vec[2]/h, -1.0, 1.0)),
        'node': np.arctan2(node_vec[1], node_vec[0]) % (2*np.pi), 'peri': peri}


def _catalogue():
    """ A small catalogue in the format of the loaders: name, q, e, i, peri, node, H, epoch JD, mean
        anomaly and time of perihelion. """

    return [
        ["Phaethon", 0.1400, 0.8900, 22.25, 322.2, 265.2, 14.4, 2461200.5, 100.0, None],
        ["Near twin", 0.1500, 0.8800, 22.00, 320.0, 263.0, 20.0, 2461200.5, 10.0, None],
        ["Apollo", 0.8000, 0.5000, 10.00, 60.0, 120.0, 18.0, 2461200.5, 200.0, None],
        ["Never reaches Earth", 1.8000, 0.2000, 5.00, 30.0, 40.0, 17.0, 2461200.5, 5.0, None],
        ["Parabolic comet", 0.9000, 1.0000, 70.00, 100.0, 200.0, -1, 2451800.5, None, 2451820.5],
        ]


class _Catalogue(object):
    """ Put the small catalogue in place of the real ones while a test runs. """

    def __enter__(self):
        self.saved = (pbs.comets_elements, pbs.asteroids_elements)
        pbs.comets_elements = _catalogue()[4:]
        pbs.asteroids_elements = _catalogue()[:4]
        return self

    def __exit__(self, *args):
        pbs.comets_elements, pbs.asteroids_elements = self.saved


def _search(d_crit, **kwargs):

    q, e, i, peri, node = PHAETHON

    with _Catalogue():
        return pbs.searchParentBodies(q, e, np.radians(i), np.radians(peri), np.radians(node),
            d_crit=d_crit, **kwargs)


def test_geocentric_from_elements_recovers_the_orbit():
    """ An orbit with its node at 1 AU meets the Earth there, so the radiant derived from its
        elements has to give the elements back when the heliocentric velocity is rebuilt from it. """

    # Argument of perihelion 60 deg and p = 1.25 put the ascending node at exactly 1 AU
    e = 0.5
    peri = np.radians(60.0)
    q = (1.0 + e*np.cos(peri))/(1.0 + e)
    incl = np.radians(15.0)
    node = np.radians(130.0)

    radiants = calcGeocentricFromElements(q, e, incl, node, peri)

    assert len(radiants) >= 1

    found = False

    for ra, dec, sol, vg in radiants:

        # Rebuild the velocity from the radiant, in the ecliptic, in units of the Earth's speed
        eps = 0.40909280
        radiant_eq = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
        radiant_ecl = np.array([radiant_eq[0], radiant_eq[1]*np.cos(eps) + radiant_eq[2]*np.sin(eps),
            -radiant_eq[1]*np.sin(eps) + radiant_eq[2]*np.cos(eps)])

        lon = sol - np.pi
        v_geo = -radiant_ecl*vg/SPEED_EARTH
        v_earth = np.array([-np.sin(lon), np.cos(lon), 0.0])

        el = _elementsFromState(np.array([np.cos(lon), np.sin(lon), 0.0]), v_geo + v_earth)

        # The closest point is found on a grid refined to about 0.003 deg, so the orbit comes back to
        #   that order
        if np.allclose([el['q'], el['e'], el['incl'], el['node'], el['peri']],
            [q, e, incl, node, peri], atol=1e-4):
            found = True

    assert found


def test_geocentric_from_elements_gives_the_geminid_radiant_for_phaethon():
    """ The elements of Phaethon come back as the Geminid radiant, a few degrees off and with
        D_N well within the 0.2 quoted for association. """

    radiants = calcGeocentricFromElements(PHAETHON[0], PHAETHON[1], np.radians(PHAETHON[2]),
        np.radians(PHAETHON[4]), np.radians(PHAETHON[3]))

    assert len(radiants) == 1

    ra, dec, sol, vg = radiants[0]

    assert abs(np.degrees(ra) - 112.3) < 3.0
    assert abs(np.degrees(dec) - 32.3) < 3.0
    assert abs(np.degrees(sol) - 262.1) < 1.0
    assert abs(vg - 33.8) < 1.0
    assert calcDN(*(GEMINID_RADIANT + radiants[0])) < 0.06


def test_orbit_that_does_not_reach_the_earth_has_no_radiant():

    # Perihelion beyond, and aphelion inside, the reach of 0.25 AU
    assert calcGeocentricFromElements(1.8, 0.2, 0.1, 0.5, 0.5) == []
    assert calcGeocentricFromElements(0.3, 0.1, 0.1, 0.5, 0.5) == []

    # A hyperbolic orbit that passes 0.38 AU from the Earth's
    assert calcGeocentricFromElements(0.9, 1.002, np.radians(70), np.radians(200), np.radians(100)) == []


def test_known_parents_give_the_radiants_of_their_showers():
    """ Parent bodies and their showers, among them two parents that never reach 1 AU (8P/Tuttle,
        21P/Giacobini-Zinner) and one with a shower at each node (1P/Halley). The D_N limits are
        loose bounds around what this derivation gives against the IAU MDC shower radiants. """

    showers = {
        "Geminids": (112.3, 32.3, 262.1, 33.8), "Orionids": (95.5, 15.9, 208.0, 66.5),
        "eta Aquariids": (338.0, -1.0, 45.5, 65.5), "Perseids": (48.2, 58.1, 140.0, 59.1),
        "Leonids": (153.6, 21.6, 235.3, 70.0), "Ursids": (219.4, 75.3, 270.7, 32.9),
        "Draconids": (263.1, 55.8, 195.4, 20.4)}

    # q, e, i, node, peri, the showers, and the largest D_N accepted
    parents = [
        ((0.140, 0.890, 22.25, 265.2, 322.2), ["Geminids"], 0.05),
        ((0.575, 0.968, 162.19, 59.1, 112.24), ["Orionids", "eta Aquariids"], 0.2),
        ((0.960, 0.963, 113.45, 139.38, 152.98), ["Perseids"], 0.05),
        ((0.976, 0.906, 162.49, 235.27, 172.5), ["Leonids"], 0.05),
        ((1.027, 0.820, 54.98, 270.34, 207.51), ["Ursids"], 0.08),
        ((1.010, 0.710, 31.99, 195.4, 172.8), ["Draconids"], 0.05)]

    for (q, e, i, node, peri), names, limit in parents:

        radiants = calcGeocentricFromElements(q, e, np.radians(i), np.radians(node), np.radians(peri))

        assert radiants

        for name in names:

            ra, dec, sol, vg = showers[name]
            shower = (np.radians(ra), np.radians(dec), np.radians(sol), vg)

            assert min(calcDN(*(shower + r)) for r in radiants) < limit, name


def test_single_criterion_keeps_the_original_return_shape():

    q, e, i, peri, node = PHAETHON

    with _Catalogue():
        rows = pbs.findParentBodies(q, e, np.radians(i), np.radians(peri), np.radians(node),
            d_crit='dsh', top_n=3)

    # name, q, e, i, peri, node, H, D
    assert len(rows) == 3
    assert all(len(row) == 8 for row in rows)
    assert rows[0][0] == "Phaethon"
    assert rows[0][7] < 0.01
    assert rows[0][7] <= rows[1][7] <= rows[2][7]


def test_several_criteria_are_ranked_by_the_first_and_all_are_carried():

    for first, second in (('DSH', 'DN'), ('DN', 'DSH')):

        search = _search([first, second], top_n=-1)

        assert search['keys'] == [first, second]
        assert search['labels'] == [{'DSH': 'D_SH', 'DN': 'D_N'}[first], {'DSH': 'D_SH', 'DN': 'D_N'}[second]]

        ranking = [m['d_values'][0] for m in search['matches']]

        assert ranking == sorted(ranking)
        assert all(len(m['d_values']) == 2 for m in search['matches'])

    # The legacy shape carries the values in the order given
    q, e, i, peri, node = PHAETHON

    with _Catalogue():
        rows = pbs.findParentBodies(q, e, np.radians(i), np.radians(peri), np.radians(node),
            d_crit=['DSH', 'DN'], top_n=2)

    assert all(len(row) == 9 for row in rows)


def test_dn_of_a_body_equals_the_function_on_the_derived_radiants():

    search = _search(['DN'], top_n=-1)
    q, e, i, peri, node = PHAETHON

    query = calcGeocentricFromElements(q, e, np.radians(i), np.radians(node), np.radians(peri))

    for m in search['matches']:

        body = calcGeocentricFromElements(m['q'], m['e'], np.radians(m['incl']), np.radians(m['node']),
            np.radians(m['peri']))

        expected = min(calcDN(*(g1 + g2)) for g1 in query for g2 in body)

        assert abs(m['d_values'][0] - expected) < 1e-12


def test_a_body_that_does_not_reach_the_earth_is_excluded_by_dn_and_kept_as_na_otherwise():

    names = [m['name'] for m in _search(['DN'], top_n=-1)['matches']]

    assert "Never reaches Earth" not in names
    assert "Phaethon" in names

    # Ranked by D_SH the body stays and has no value for D_N
    matches = _search(['DSH', 'DN'], top_n=-1)['matches']
    never = [m for m in matches if m['name'] == "Never reaches Earth"][0]

    assert np.isfinite(never['d_values'][0])
    assert np.isinf(never['d_values'][1])


def test_given_radiant_is_used_instead_of_the_derived_one():

    given = _search(['DN'], top_n=-1, geocentric=GEMINID_RADIANT)

    assert given['geocentric_source'] == 'given'
    assert given['query_geocentric'] == [GEMINID_RADIANT]

    derived = _search(['DN'], top_n=-1)

    assert derived['geocentric_source'] == 'elements'

    # Against the measured radiant Phaethon is no longer at exactly zero
    assert derived['matches'][0]['d_values'][0] < 1e-6
    assert 1e-6 < given['matches'][0]['d_values'][0] < 0.06


def test_unknown_key_falls_back_to_dsh():

    search = _search(['nonsense'], top_n=1)

    assert search['keys'] == ['DSH']


def test_save_csv_and_json_hold_every_criterion():

    search = _search(['DSH', 'DN', 'DD'], top_n=3)

    with tempfile.TemporaryDirectory() as tmp:

        csv_path = os.path.join(tmp, "matches.csv")
        pbs.saveMatches(csv_path, search)

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert rows[0]['name'] == 'Phaethon'
        assert all(label in rows[0] for label in ('D_SH', 'D_N', 'D_D'))
        assert [float(r['D_SH']) for r in rows] == sorted(float(r['D_SH']) for r in rows)

        json_path = os.path.join(tmp, "matches.json")
        pbs.saveMatches(json_path, search, meteor={'q': PHAETHON[0]})

        with open(json_path) as f:
            data = json.load(f)

        assert data['ranked_by'] == 'D_SH'
        assert data['criteria'] == ['D_SH', 'D_N', 'D_D']
        assert data['meteor']['q'] == PHAETHON[0]
        assert set(data['matches'][0]['criteria']) == {'D_SH', 'D_N', 'D_D'}


def test_bar_plot_is_written():

    search = _search(['DSH', 'DN'], top_n=4)

    with tempfile.TemporaryDirectory() as tmp:

        path = os.path.join(tmp, "bars.png")
        pbs.plotCriteriaBars(search, path, title="test")

        assert os.path.getsize(path) > 1000


def test_propagation_goes_through_rebound_py_and_leaves_unusable_bodies_alone():
    """ The bodies are brought to the epoch by wmpl.Rebound.REBOUND, and integrating back to the
        starting epoch returns the starting elements. """

    from wmpl.Rebound import REBOUND as reb

    if not reb.REBOUND_FOUND:
        print("  (skipped, REBOUND is not available)")
        return

    try:
        reb.checkReboundxUsable()

    except RuntimeError:
        print("  (skipped, REBOUNDx does not attach to REBOUND in this environment)")
        return

    if not os.path.isfile(pbs.config.jpl_ephem_file):
        print("  (skipped, the DE430 ephemeris is not available)")
        return

    body = {'name': 'Apollo', 'q': 0.8, 'e': 0.5, 'incl': 10.0, 'peri': 60.0, 'node': 120.0,
        'epoch_jd': 2461200.5, 'mean_anomaly': 200.0}
    parabolic = dict(body, e=1.0)
    no_epoch = dict(body, epoch_jd=None)

    jd_utc = 2459000.5
    out = pbs.propagateBodies([body, parabolic, no_epoch], jd_utc)

    assert out[1] is None and out[2] is None

    there = out[0]

    assert there['jd'] == jd_utc
    assert abs(there['jd_tdb'] - reb.utcToTdb(jd_utc)) < 1e-12

    # Over six years the elements move, but only a little for an orbit like this
    assert abs(there['q'] - 0.8) < 0.05

    # The saved state is the one of the saved elements
    state = reb.heliocentricStateFromElements(there['q'], there['e'], there['incl'], there['node'],
        there['peri'], there['mean_anomaly'])

    assert np.allclose(state, there['position'] + there['velocity'], atol=1e-9)

    # Integrating back to the starting epoch returns the starting elements. Not to machine precision:
    #   each leg starts the planets from DE430 at its own epoch rather than from where the other leg
    #   left them (the integrated Moon drifts from the ephemeris in six years), which leaves 1e-6 in
    #   q and e and 0.003 deg in the mean anomaly after the twelve years there and back
    back = reb.propagateBodiesToEpoch([{'q': there['q'], 'e': there['e'], 'incl': there['incl'],
        'peri': there['peri'], 'node': there['node'], 'epoch_jd': there['jd_tdb'],
        'mean_anomaly': there['mean_anomaly']}], 2461200.5)[0]

    assert abs(back['q'] - 0.8) < 1e-5
    assert abs(back['e'] - 0.5) < 1e-5
    assert abs(back['incl'] - 10.0) < 1e-4
    assert abs(back['peri'] - 60.0) < 1e-3
    assert abs(back['node'] - 120.0) < 1e-3
    assert abs(((back['mean_anomaly'] - 200.0) + 180) % 360 - 180) < 1e-2


def test_loaders_read_the_epoch_and_the_mean_anomaly():

    with tempfile.TemporaryDirectory() as tmp:

        comet = os.path.join(tmp, "comets.txt")
        asteroid = os.path.join(tmp, "asteroids.txt")

        with open(comet, 'w') as f:
            f.write("Num  Name\n---\n")
            f.write("  2P/Encke                                    59936  0.33891075 0.84731142  11.36818 "
                "187.20599 334.11280 20231022.39522 JPL K273/14\n")

        with open(asteroid, 'w') as f:
            f.write("Designation\n\n")
            f.write("                           2026 SB      0.863   4.266  0.00084   28.13  20260609  "
                "344.8    0.7 301.4   0.1 0.664   2.564  (  1d)   E2026-S16\n")

        c = pbs.loadCometsElements(comet)[0]
        a = pbs.loadAsteroidsElements(asteroid)[0]

        # MJD 59936 is the epoch, and the perihelion is 303.4 days after it
        assert abs(c[7] - (59936 + 2400000.5)) < 1e-9

        a_au = 0.33891075/(1.0 - 0.84731142)
        tp_jd = pbs._calendarToJD(2023, 10, 22.39522)
        expected = np.degrees(pbs.GAUSS_K/a_au**1.5*(c[7] - tp_jd))

        assert abs(c[8] - expected) < 1e-9
        assert -180.0 < c[8] <= 180.0

        # The time of perihelion is kept, for the comets that have no mean anomaly
        assert abs(c[9] - tp_jd) < 1e-9
        assert a[9] is None
        assert a[0] == "2026 SB"
        assert a[8] == 344.8
        assert abs(a[7] - pbs._calendarToJD(2026, 6, 9)) < 1e-9


def test_criteria_at_the_epoch_are_those_of_the_propagated_elements():
    """ The values printed with --propagate are the criteria of the search, with the body's
        propagated elements in place of its catalogue ones, against the same meteor orbit. """

    search = _search(['DSH', 'DN', 'DVVIDA'], top_n=3)
    q, e, i, peri, node = PHAETHON

    # Stand-in propagated records: the catalogue elements nudged, and one body not propagated
    propagated = []
    for k, m in enumerate(search['matches']):
        propagated.append(None if k == 1 else {'q': m['q'] + 0.01, 'e': m['e'], 'incl': m['incl'] + 0.5,
            'peri': m['peri'], 'node': m['node'] + 1.0})

    values = pbs.criteriaAtEpoch(search, propagated, q, e, np.radians(i), np.radians(peri),
        np.radians(node))

    assert values[1] is None

    from wmpl.Utils.Dcriteria import lookupCriterion, calcCriterionFromElements

    query = (q, e, np.radians(i), np.radians(node), np.radians(peri))

    for k in (0, 2):

        p = propagated[k]
        orbit = (p['q'], p['e'], np.radians(p['incl']), np.radians(p['node']), np.radians(p['peri']))

        for c, key in enumerate(search['keys']):
            assert values[k][c] == calcCriterionFromElements(lookupCriterion(key), query, orbit)

        # Not the catalogue values: the elements moved
        assert values[k][0] != search['matches'][k]['d_values'][0]

    # Saved beside the propagated state, in the JSON and in the CSV
    for k in (0, 2):
        propagated[k].update({'jd': 2459000.5, 'jd_tdb': 2459000.5008, 'a': None, 'mean_anomaly': None,
            'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.017, 0.0]})

    with tempfile.TemporaryDirectory() as tmp:

        json_path = os.path.join(tmp, "m.json")
        pbs.saveMatches(json_path, search, propagated=propagated, at_epoch_values=values)

        with open(json_path) as f:
            data = json.load(f)

        assert data['matches'][1]['at_epoch'] is None
        assert data['matches'][0]['at_epoch']['criteria']['D_SH'] == values[0][0]
        assert not any(key.startswith('at_epoch_') for key in data['matches'][0])

        csv_path = os.path.join(tmp, "m.csv")
        pbs.saveMatches(csv_path, search, propagated=propagated, at_epoch_values=values)

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]['at_epoch_D_SH']) == values[0][0]
        assert rows[1]['at_epoch_D_SH'] == ''


def test_heliocentric_from_elements_follows_the_wmpl_convention():
    """ The corrected heliocentric direction derived from the elements has to be what
        correctedEclipticCoord, which the trajectory solver uses, makes of the derived geocentric
        radiant, and the same as converting that radiant directly. """

    eps = 0.40909280

    for q, e, i, node, peri in ((0.14, 0.89, 22.25, 265.2, 322.2), (0.5749, 0.96794, 162.19, 59.1, 112.24),
                                (0.5, 0.6, 2.0, 10.0, 40.0)):

        el = (q, e, np.radians(i), np.radians(node), np.radians(peri))

        geo = calcGeocentricFromElements(*el)
        helio = calcHeliocentricFromElements(*el)

        assert len(geo) == len(helio) >= 1

        for (ra, dec, sol, vg), (l_h, b_h, sol_h, v_h) in zip(geo, helio):

            x, y, z = np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)
            l_g = np.arctan2(y*np.cos(eps) + z*np.sin(eps), x)
            b_g = np.arcsin(-y*np.sin(eps) + z*np.cos(eps))

            lon = sol - np.pi
            earth_vel = SPEED_EARTH*np.array([-np.sin(lon), np.cos(lon), 0.0])

            l_ref, b_ref, v_ref = correctedEclipticCoord(l_g, b_g, vg, earth_vel)

            assert abs(l_h - l_ref) < 1e-9
            assert abs(b_h - b_ref) < 1e-9
            assert abs(v_h - np.linalg.norm(v_ref)) < 1e-9
            assert sol_h == sol

            assert np.allclose(calcHeliocentricFromGeocentric(ra, dec, sol, vg), (l_h, b_h, sol_h, v_h),
                atol=1e-9)


def test_vida_criterion_is_offered_and_uses_the_heliocentric_direction():

    search = _search(['DVVIDA', 'DSH'], top_n=-1)

    assert search['labels'] == ['D_V (Vida)', 'D_SH']
    assert search['heliocentric_source'] == 'elements'
    assert search['geocentric_source'] is None
    assert search['matches'][0]['name'] == 'Phaethon'

    query = search['query_heliocentric']

    for m in search['matches']:

        body = calcHeliocentricFromElements(m['q'], m['e'], np.radians(m['incl']),
            np.radians(m['node']), np.radians(m['peri']))

        expected = min(calcDV(*(h1 + h2)) for h1 in query for h2 in body)

        assert abs(m['d_values'][0] - expected) < 1e-12

    # A given direction is used instead
    given = calcHeliocentricFromGeocentric(*GEMINID_RADIANT)
    search = _search(['DVVIDA'], top_n=1, heliocentric=given)

    assert search['heliocentric_source'] == 'given'
    assert search['query_heliocentric'] == [given]


def test_moid_criterion_ranks_every_body_including_open_comets():
    """ MOID is offered like the D criteria: as the first criterion it ranks the whole catalogue,
        the parabolic comet included, with the values of the MOID between the orbits, and the other
        criteria are carried along. """

    from wmpl.Utils.MOID import calcMOIDFromPerihelion

    search = _search(['MOID', 'DSH'], top_n=-1)
    assert search['keys'] == ['MOID', 'DSH']
    assert len(search['matches']) == len(_catalogue())

    q, e, i, peri, node = PHAETHON
    moids = []
    for match in search['matches']:
        expected = calcMOIDFromPerihelion(q, e, np.radians(i), np.radians(node), np.radians(peri),
            match['q'], match['e'], np.radians(match['incl']), np.radians(match['node']),
            np.radians(match['peri']))
        assert abs(match['d_values'][0] - expected) < 1e-12
        moids.append(match['d_values'][0])

    assert moids == sorted(moids)
    assert search['matches'][0]['name'] == 'Phaethon' and moids[0] < 1e-9
    assert 'Parabolic comet' in [m['name'] for m in search['matches']]

    # As a second criterion it is carried beside the first, with the same values
    by_dsh = _search(['DSH', 'MOID'], top_n=-1)
    moid_of = {m['name']: m['d_values'][0] for m in search['matches']}
    for match in by_dsh['matches']:
        assert abs(match['d_values'][1] - moid_of[match['name']]) < 1e-12


if __name__ == "__main__":

    test_functions = [obj for name, obj in sorted(globals().items()) \
        if name.startswith("test_") and callable(obj)]

    failed = 0

    for test_func in test_functions:

        try:
            test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {!r}".format(test_func.__name__, e))

    print()

    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
