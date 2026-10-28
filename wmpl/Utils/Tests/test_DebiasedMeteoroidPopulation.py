""" Tests for the background of meteoroids that hit the Earth, from the near-Earth object model of Deam
et al. (2026), and the chance-coincidence test in wmpl.Utils.DebiasedMeteoroidPopulation.

The tests write a small model in the format of the published files, a posterior and an archive of
residence-time maps whose occupied bins are chosen here, so they need neither the network nor the
3.5 MB of the real files. The one test that reads the real files is skipped when they have not been
downloaded.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DebiasedMeteoroidPopulation.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_DebiasedMeteoroidPopulation
"""

import io
import os
import math
import tarfile
import tempfile

import numpy as np

import wmpl.Utils.DebiasedMeteoroidPopulation as dmp
import wmpl.Utils.DebiasedNEOPopulation as dnp
from wmpl.Utils.Dcriteria import calcDSH


# Bins (ia, ie, ii, from 0) of the small model, all of whose orbits cross 1 AU: a = 1.5-1.6 AU with
#   e = 0.60-0.64, a = 0.8-0.9 AU with e = 0.40-0.44, and a = 2.5-2.6 AU with e = 0.64-0.68
BIN_3MY = (15, 15, 2)
BIN_1MY = (8, 10, 5)
BIN_LOW_I = (25, 16, 0)

# Two posterior samples: the strengths of the six sources at the four H nodes, and the crossover
STRENGTHS_A = np.array([[0.5, 0.4, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
STRENGTHS_B = np.array([[0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
CROSSOVERS = (16, 17)


def _mapRows(bins):
    """ The rows of a residence-time map with all its time in the given bins, shared equally. """

    return "ia,ie,ii,prob\n" + "".join("{:d},{:d},{:d},{:.12f}\n".format(ia + 1, ie + 1, ii + 1,
        1.0/len(bins)) for ia, ie, ii in bins)


def writeModel(tmp, maps_3my=(BIN_3MY,), maps_1my=(BIN_1MY,), strengths=(STRENGTHS_A, STRENGTHS_B),
    crossovers=CROSSOVERS):
    """ Write a model in the format of the published files: the nu6 source in the given bins of its 3 Myr
        and 1 Myr maps, every other source in the bin of the 3 Myr map of nu6 for both lifetimes. Return
        the paths of the posterior and of the archive. """

    tar_path = os.path.join(tmp, 'residence_time_distributions.tar.gz')

    with tarfile.open(tar_path, 'w:gz') as tar:

        for source in dmp.SOURCES:

            for maps, bins in ((dmp.MAPS_3MY, maps_3my), (dmp.MAPS_1MY, maps_1my)):

                data = _mapRows(bins if source == 'nu6' else maps_3my).encode()
                info = tarfile.TarInfo('residence_time_distributions/' + maps[source] + '.csv')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    posterior_path = os.path.join(tmp, 'posterior_samples.dat')

    with open(posterior_path, 'w') as f:
        for s, crossover in zip(strengths, crossovers):
            f.write("   ".join("{:.18E}".format(v) for v in list(s.ravel()) + [crossover, -8231.0])
                + "\n")

    return posterior_path, tar_path


def _model(tmp, **kwargs):

    posterior_path, tar_path = writeModel(tmp, **kwargs)

    return dmp.loadDeamModel(posterior_path=posterior_path, residence_path=tar_path)


def _filled(n_bins):
    """ The value of a bin holding all the time of a map, after the empty bins are filled. """

    return 1.0/(1.0 + dmp.EMPTY_BIN*(n_bins - 1))


def test_model_is_the_posterior_mean_of_the_sources_with_their_lifetimes():
    """ Each H slice sums to 1; below both crossovers the 3 Myr maps are used, above both the 1 Myr
        ones, and between them half of each, with the strengths interpolated in H and averaged over
        the two samples. """

    with tempfile.TemporaryDirectory() as tmp:
        deam = _model(tmp)

    population = deam['population']
    n_bins = population[..., 0].size
    full = _filled(n_bins)

    assert population.shape == (42, 25, 22, 28)
    assert np.allclose(population.sum(axis=(0, 1, 2)), 1.0, rtol=1e-12)

    def nu6(strengths, k):
        return np.interp(dmp.H_CENTRES[k], dmp.H_NODES, strengths[0])

    def onlyNu6InTheBin(strength):
        """ A sample whose nu6 map has the bin full and the other sources' maps have it empty. """
        return strength*full + (1.0 - strength)*dmp.EMPTY_BIN*full

    # H bin 15, below both crossovers: every source in the 3 Myr bin, none in the 1 Myr one
    k = 15
    assert math.isclose(population[BIN_3MY + (k,)], full, rel_tol=1e-9)
    assert math.isclose(population[BIN_1MY + (k,)], dmp.EMPTY_BIN*full, rel_tol=1e-9)

    # H bin 16: sample A has switched to the 1 Myr maps, sample B has not
    k = 16
    expected = 0.5*onlyNu6InTheBin(nu6(STRENGTHS_A, k)) + 0.5*dmp.EMPTY_BIN*full
    assert math.isclose(population[BIN_1MY + (k,)], expected, rel_tol=1e-9)

    # H bin 17, above both: nu6 in its 1 Myr bin in both samples
    k = 17
    expected = 0.5*onlyNu6InTheBin(nu6(STRENGTHS_A, k)) + 0.5*onlyNu6InTheBin(nu6(STRENGTHS_B, k))
    assert math.isclose(population[BIN_1MY + (k,)], expected, rel_tol=1e-9)

    assert list(deam['crossover']) == list(CROSSOVERS)
    assert deam['strengths'].shape == (2, 6, 4)


def test_population_at_h_takes_the_bins_within_a_quarter():
    """ H = 38.0 takes the bins centred on 37.875 and 38.125, H = 38.1 the same two. """

    with tempfile.TemporaryDirectory() as tmp:
        deam = _model(tmp)

    k = np.searchsorted(dmp.H_CENTRES, [37.875, 38.125])
    expected = deam['population'][..., k].sum(axis=-1)

    assert np.allclose(dmp.populationAtH(deam, 38.0), expected)
    assert np.allclose(dmp.populationAtH(deam, 38.1), expected)


def test_impact_rate_is_that_of_opik_with_focusing():
    """ impactRate against the same rate written through the perihelion and aphelion distances,
        (1 - q)(Q - 1) = a U_x^2, and zero for orbits that do not cross 1 AU. """

    a = np.array([1.5, 0.85, 2.55, 1.2])
    e = np.array([0.62, 0.42, 0.66, 0.1])
    incl = np.radians([10.0, 22.0, 2.0, 5.0])

    q, big_q = a*(1 - e), a*(1 + e)

    # The last orbit has no encounter speed: U^2 < 0
    with np.errstate(invalid='ignore'):
        u = np.sqrt(3.0 - 1.0/a - 2.0*np.sqrt(a*(1 - e**2))*np.cos(incl))
        v_inf = u*dmp.SPEED_EARTH
        expected = u*(1.0 + dmp.ESCAPE_SPEED**2/v_inf**2)/(a**1.5*np.sin(incl)
            *np.sqrt((1.0 - q)*(big_q - 1.0)/a))

    rate = dmp.impactRate(a, e, incl)

    assert np.allclose(rate[:3], expected[:3], rtol=1e-12)

    # q = 1.08 AU: never at 1 AU
    assert rate[3] == 0.0

    assert math.isclose(dmp.SPEED_EARTH, 29.7847, abs_tol=1e-4)
    assert math.isclose(dmp.ESCAPE_SPEED, 11.099, abs_tol=1e-3)


def test_every_impactor_has_a_node_at_one_au():
    """ The orbits drawn cross 1 AU and have a node there, the weights are positive with mean 1, and
        the four arguments of perihelion are all used. """

    with tempfile.TemporaryDirectory() as tmp:
        deam = _model(tmp, maps_3my=(BIN_3MY, BIN_LOW_I))

    imp = dmp.sampleImpactors(deam, 38.0, n=20000, seed=3)

    p = imp['a']*(1 - imp['e']**2)
    r_ascending = p/(1 + imp['e']*np.cos(-imp['peri']))
    r_descending = p/(1 + imp['e']*np.cos(np.pi - imp['peri']))

    assert np.max(np.minimum(np.abs(r_ascending - 1), np.abs(r_descending - 1))) < 1e-9
    assert np.all(imp['q'] < 1.0) and np.all(imp['a']*(1 + imp['e']) > 1.0)
    assert np.all(imp['weight'] > 0) and math.isclose(np.mean(imp['weight']), 1.0, rel_tol=1e-12)
    assert 0.8*imp['n'] < imp['n_eff'] <= imp['n']

    # The four solutions: the perihelion before or after each node
    near_ascending = np.abs(r_ascending - 1) < 1e-9
    assert 0.4 < np.mean(near_ascending) < 0.6
    assert 0.4 < np.mean(np.sin(imp['peri']) > 0) < 0.6

    # The same seed gives the same orbits
    again = dmp.sampleImpactors(deam, 38.0, n=20000, seed=3)
    assert again is imp


def test_weights_reproduce_the_population_times_the_impact_rate():
    """ The weighted impactors against an independent draw: the population uniform in a and e and in
        cos i within each bin, weighted directly by impactRate. The means of a, e, i, v_inf and the
        share of the impacts from each bin agree within the sampling error, the lowest-inclination
        bin included, where the sin i of the density cancels the 1/sin i of the rate. """

    with tempfile.TemporaryDirectory() as tmp:
        deam = _model(tmp, maps_3my=(BIN_3MY, BIN_LOW_I), maps_1my=(BIN_1MY,))

    # Past both crossovers, where nu6 is in its 1 Myr bin and the other sources in the two others
    h_mag = 39.0
    n = 400000

    imp = dmp.sampleImpactors(deam, h_mag, n=n, seed=5)

    # The independent draw, with the bins in proportion to the population, i uniform within the bin and
    #   the density's sin i folded into the weight, which keeps the variance finite at i = 0
    rng = np.random.default_rng(6)
    population = dmp.populationAtH(deam, h_mag)
    occupied = np.argwhere(population > 1e-6)
    shares = population[tuple(occupied.T)]/np.sum(population[tuple(occupied.T)])

    which = rng.choice(len(occupied), size=n, p=shares)
    ia, ie, ii = occupied[which].T

    a = rng.uniform(dmp.A_EDGES[ia], dmp.A_EDGES[ia + 1])
    e = rng.uniform(dmp.E_EDGES[ie], dmp.E_EDGES[ie + 1])
    i1, i2 = np.radians(dmp.INCL_EDGES[ii]), np.radians(dmp.INCL_EDGES[ii + 1])
    incl = rng.uniform(i1, i2)

    w = dmp.impactRate(a, e, incl)*np.sin(incl)*(i2 - i1)/(np.cos(i1) - np.cos(i2))
    u = np.sqrt(3.0 - 1.0/a - 2.0*np.sqrt(a*(1 - e**2))*np.cos(incl))

    def weightedMean(x, weights):
        """ The weighted mean and its standard error. """
        weights = weights/np.sum(weights)
        mean = np.sum(weights*x)
        return mean, np.sqrt(np.sum(weights**2*(x - mean)**2))

    v_imp = imp['v_inf']
    for name, x_imp, x_ref in (('a', imp['a'], a), ('e', imp['e'], e), ('i', imp['incl'], incl),
            ('v_inf', v_imp, u*dmp.SPEED_EARTH)):
        m1, s1 = weightedMean(x_imp, imp['weight'])
        m2, s2 = weightedMean(x_ref, w)
        assert abs(m1 - m2) < 4*np.hypot(s1, s2), (name, m1, m2, s1, s2)

    for bin_ in (BIN_3MY, BIN_LOW_I, BIN_1MY):
        lo_a, lo_e, lo_i = dmp.A_EDGES[bin_[0]], dmp.E_EDGES[bin_[1]], math.radians(dmp.INCL_EDGES[bin_[2]])
        in_imp = (imp['a'] >= lo_a) & (imp['a'] < lo_a + 0.1) & (imp['e'] >= lo_e) \
            & (imp['e'] < lo_e + 0.04) & (imp['incl'] >= lo_i) & (imp['incl'] < lo_i + math.radians(4))
        in_ref = (ia == bin_[0]) & (ie == bin_[1]) & (ii == bin_[2])
        m1, s1 = weightedMean(in_imp.astype(float), imp['weight'])
        m2, s2 = weightedMean(in_ref.astype(float), w)
        assert m1 > 0.01
        assert abs(m1 - m2) < 4*np.hypot(s1, s2), (bin_, m1, m2, s1, s2)


def _impactors(n_far=990):
    """ Background orbits around a candidate's: ten that differ from it only in q, by 0.01 to 0.10 AU,
        so at those values of D_SH, and n_far far away; the closest three weigh 2 and the others are
        scaled so that the weights have mean 1. """

    q0, e0, i0, node0, peri0 = CANDIDATE

    close_q = q0 + 0.01*np.arange(1, 11)
    q = np.concatenate([close_q, np.full(n_far, 0.2)])
    e = np.concatenate([np.full(10, e0), np.full(n_far, 0.1)])
    incl = np.concatenate([np.full(10, i0), np.full(n_far, 0.5)])
    node = np.concatenate([np.full(10, node0), np.full(n_far, 3.0)])
    peri = np.concatenate([np.full(10, peri0), np.full(n_far, 1.0)])

    weight = np.ones(len(q))
    weight[:3] = 2.0
    weight = weight/np.mean(weight)

    return {'q': q, 'a': q/(1 - e), 'e': e, 'incl': incl, 'node': node, 'peri': peri, 'weight': weight,
        'v_inf': np.full(len(q), 10.0), 'h_mag': 38.0, 'n': len(q), 'n_eff': float(len(q)),
        'seed': None, 'model': dmp.DEAM2026}


# The candidate and the meteoroid of the tests, in the model's region: q (AU), e, i, node, peri (rad)
CANDIDATE = (0.80, 0.50, 0.17, 2.1, 1.0)
METEOR = (0.85, 0.55, 0.20, 2.0, 1.1)


def test_p_counts_the_weight_of_the_background_at_least_as_close():
    """ p = (k + 1)/(N + 1), k the weight of the background orbits at least as close to the candidate
        as the meteoroid, here D_SH = 0.045: the four orbits at dq = 0.01 to 0.04, three of weight 2. """

    imp = _impactors()
    weight = imp['weight']

    result = dmp.chanceAssociation(METEOR, 38.0, [(0.045, CANDIDATE)], 'DSH', imp)[0]

    k = weight[0] + weight[1] + weight[2] + weight[3]

    assert result['reason'] is None
    assert result['n_closer'] == 4
    assert math.isclose(result['p_value'], (k + 1.0)/(imp['n'] + 1.0), rel_tol=1e-12)
    assert math.isclose(result['fraction'], k/imp['n'], rel_tol=1e-12)
    assert result['p_value_global'] is None

    # A value equal to a background orbit's counts it as at least as close
    d_fourth = calcDSH(*([imp[name][3] for name in ('q', 'e', 'incl', 'node', 'peri')]
        + list(CANDIDATE)))
    assert dmp.chanceAssociation(METEOR, 38.0, [(d_fourth, CANDIDATE)], 'DSH', imp)[0]['n_closer'] == 4

    none = dmp.chanceAssociation(METEOR, 38.0, [(0.001, CANDIDATE)], 'DSH', imp)[0]
    assert none['n_closer'] == 0 and none['is_limit']
    assert math.isclose(none['p_value'], 1.0/(imp['n'] + 1.0))


def test_p_global_counts_the_background_with_some_body_as_close():
    """ With a catalogue of the candidate and a far body, a background orbit has a body within
        D_SH = 0.045 when it is one of the four near the candidate; p_global is that weight + 1 over
        n_global + 1, and never below p. """

    imp = _impactors()
    weight = imp['weight']

    far = (0.3, 0.2, 1.2, 0.5, 0.5)
    catalogue = tuple(np.array([c, f]) for c, f in zip(CANDIDATE, far))

    result = dmp.chanceAssociation(METEOR, 38.0, [(0.045, CANDIDATE)], 'DSH', imp,
        catalogue_orbits=catalogue, n_global=imp['n'])[0]

    k = weight[0] + weight[1] + weight[2] + weight[3]

    assert result['n_closer_global'] == 4
    assert math.isclose(result['p_value_global'], (k + 1.0)/(imp['n'] + 1.0), rel_tol=1e-12)
    assert math.isclose(result['sigma'], dnp.pValueToSigma(result['p_value_global']), rel_tol=1e-12)
    assert result['p_value_global'] >= result['p_value']

    # With only 500 far orbits compared, p_global would be 1/501, below p: it is taken as p
    few = dmp.chanceAssociation(METEOR, 38.0, [(0.045, CANDIDATE)], 'DSH', dict(imp,
        **dict((name, imp[name][::-1]) for name in ('q', 'a', 'e', 'incl', 'node', 'peri', 'weight'))),
        catalogue_orbits=catalogue, n_global=500)[0]
    assert few['n_closer_global'] == 0 and few['is_limit_global']
    assert few['p_value_global'] == few['p_value']


def test_meteoroid_outside_the_model_gets_no_value():
    """ An H outside the model, no H, an unbound or a distant orbit, a retrograde one and one that does
        not reach the Earth's all get a reason instead of a p-value. """

    cases = [
        ((0.85, 0.55, 0.2, 2.0, 1.1), 34.5, "H = 34.50"),
        ((0.85, 0.55, 0.2, 2.0, 1.1), None, "no absolute magnitude"),
        ((0.85, 1.05, 0.2, 2.0, 1.1), 38.0, "not bound"),
        ((0.9, 0.8, 0.2, 2.0, 1.1), 38.0, "a = 4.500"),
        ((0.85, 0.55, math.radians(150), 2.0, 1.1), 38.0, "i = 150.00"),
        ((1.05, 0.3, 0.2, 2.0, 1.1), 38.0, "does not reach"),
        ]

    for orbit, h_mag, text in cases:

        result = dmp.chanceAssociation(orbit, h_mag, [(0.05, CANDIDATE)], 'DSH', None)[0]

        assert result['p_value'] is None and result['p_value_global'] is None, text
        assert text in result['reason'], (text, result['reason'])

    # q = 1.01 AU reaches the Earth near its aphelion
    assert dmp.isInModelRegion(1.01, 0.3, 0.2, 38.0)[0]


def test_criteria_on_the_radiant_and_the_moid_are_refused():
    """ The background has only elements here, and the MOID is too slow against it. """

    imp = _impactors()

    for key in ('DN', 'DVVIDA', 'MOID'):
        try:
            dmp.chanceAssociation(METEOR, 38.0, [(0.05, CANDIDATE)], key, imp)
            raise AssertionError("{:s} was accepted".format(key))

        except ValueError:
            pass


def test_background_drawn_for_another_h_is_refused():

    imp = _impactors()

    try:
        dmp.chanceAssociation(METEOR, 39.0, [(0.05, CANDIDATE)], 'DSH', imp)
        raise AssertionError("a background for H = 38 was used for H = 39")

    except ValueError:
        pass


def test_a_missing_file_is_reported_when_downloading_is_off():
    """ With download=False a missing file raises, naming its URL. """

    saved = dnp.CACHE_DIR

    with tempfile.TemporaryDirectory() as tmp:

        dnp.CACHE_DIR = os.path.join(tmp, 'cache')

        try:
            files = tuple(dict(f, share_path=os.path.join(tmp, 'share', f['file_name']))
                for f in dmp.DEAM2026['files'])
            model = dict(dmp.DEAM2026, files=files)

            try:
                dmp.loadDeamModel(model, download=False)
                raise AssertionError("a missing file did not raise")

            except IOError as err:
                assert files[0]['url'] in str(err)

        finally:
            dnp.CACHE_DIR = saved


def test_the_published_files_when_they_have_been_downloaded():
    """ The real model: 4125 posterior samples, the crossover at H = 38.5 in 4122 of them, and the
        background at H = 38, measured when the module was written: an effective number 0.928 times
        the number, and 34.5 % of the population, by the centres of the bins, not crossing the Earth's
        orbit. """

    paths = dmp.findDeamModel()

    if None in paths:
        print("  skipped: the model of Deam et al. (2026) has not been downloaded")
        return

    deam = dmp.loadDeamModel(posterior_path=paths[0], residence_path=paths[1])

    assert deam['strengths'].shape == (4125, 6, 4)
    assert np.count_nonzero(deam['crossover'] == 16) == 4122
    assert dmp.H_EDGES[16] == 38.5
    assert np.allclose(deam['strengths'].sum(axis=1), 1.0)

    imp = dmp.sampleImpactors(deam, 38.0, n=200000, seed=2)
    assert math.isclose(imp['n_eff']/imp['n'], 0.928, abs_tol=0.005)

    population = dmp.populationAtH(deam, 38.0)
    a_c = 0.5*(dmp.A_EDGES[1:] + dmp.A_EDGES[:-1])[:, None]
    e_c = 0.5*(dmp.E_EDGES[1:] + dmp.E_EDGES[:-1])[None, :]
    not_crossing = (a_c*(1 - e_c) >= 1) | (a_c*(1 + e_c) <= 1)
    share = np.sum(population[not_crossing])/np.sum(population)
    assert math.isclose(share, 0.345, abs_tol=0.001), share


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
