""" Tests for the debiased NEO population and the chance-coincidence test in
wmpl.Utils.DebiasedNEOPopulation.

The tests write a small synthetic population in the format of Granvik et al. (2018), whose distances
to a given orbit are known exactly, so they need neither the network nor the 28 MB file. The one
test that reads the real file is skipped when it has not been downloaded.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DebiasedNEOPopulation.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_DebiasedNEOPopulation
"""

import os
import gzip
import glob
import math
import tempfile

import numpy as np
import scipy.stats

import wmpl.Utils.DebiasedNEOPopulation as dnp


# The meteoroid of the tests: q (AU), e, i, node, peri (rad)
METEOR = (0.9, 0.5, 0.1, 1.0, 2.0)

# Synthetic orbits that differ from the meteoroid's only in q, by these amounts. With the same e, i,
#   node and argument of perihelion the mutual inclination and the difference of the longitudes of
#   perihelion are zero, so D_SH is exactly |dq|
CLOSE_DQ = 0.01*np.arange(1, 11)

# The rest of the population, far from the meteoroid (D_SH > 0.7)
N_FAR = 990


def _populationRows():
    """ Rows of a synthetic population in the column order of the published file: a (AU), e, i,
        node, argument of perihelion, mean anomaly (deg), H. """

    q0, e0, i0, node0, peri0 = METEOR

    rows = []

    for dq in CLOSE_DQ:
        q = q0 + dq
        rows.append([q/(1.0 - e0), e0, math.degrees(i0), math.degrees(node0), math.degrees(peri0),
            0.0, 0.0])

    for _ in range(N_FAR):
        rows.append([0.2/(1.0 - 0.1), 0.1, 30.0, 200.0, 100.0, 0.0, 0.0])

    rows = np.array(rows)

    # H spread evenly over the range of the model, so the counts below are known
    rows[:, 6] = np.linspace(17.004, 24.996, len(rows))

    return rows


def _writePopulation(file_path, rows=None):
    """ Write a population file, gzipped, as the authors publish it. """

    rows = _populationRows() if rows is None else rows

    with gzip.open(file_path, 'wt') as f:
        np.savetxt(f, rows, fmt='%.9f')

    return file_path


def _model(**kwargs):
    """ A copy of the published model description, with entries replaced. """

    model = dict(dnp.GRANVIK2018)
    model.update(kwargs)

    return model


def _population(tmp):
    """ The synthetic population, loaded through the module. """

    path = _writePopulation(os.path.join(tmp, 'population.dat.gz'))

    return dnp.loadDebiasedNEOPopulation(file_path=path)


def test_loader_reads_the_published_columns():
    """ The loader turns a and e into q and the angles into radians, and keeps H. """

    with tempfile.TemporaryDirectory() as tmp:

        rows = _populationRows()
        population = _population(tmp)

        assert len(population['q']) == len(rows)
        assert np.allclose(population['q'], rows[:, 0]*(1.0 - rows[:, 1]))
        assert np.allclose(population['incl'], np.radians(rows[:, 2]))
        assert np.allclose(population['node'], np.radians(rows[:, 3]))
        assert np.allclose(population['peri'], np.radians(rows[:, 4]))
        assert np.allclose(population['h_mag'], rows[:, 6])

        # A second load of the same file is the same object, not a new parse
        assert dnp.loadDebiasedNEOPopulation(file_path=population['file_path']) is population


def test_count_at_least_as_large_joins_the_known_and_the_synthetic_bodies():
    """ The known NEOs count up to the faint end where the model starts, the synthetic ones above. """

    with tempfile.TemporaryDirectory() as tmp:

        population = _population(tmp)
        known = np.array([10.0, 15.0, 16.9, 17.0, 17.5, 20.0])

        # Brighter than the model: the known ones only
        assert dnp.countAtLeastAsLarge(15.5, population, known) == (2, 0)

        # Fainter: every known one up to H = 17, and none of the known ones above it
        n_known, n_model = dnp.countAtLeastAsLarge(20.0, population, known)

        assert n_known == 4
        assert n_model == int(np.count_nonzero(population['h_mag'] <= 20.0))


def test_chance_association_counts_the_closer_orbits_and_applies_poisson():
    """ k is the number of synthetic orbits at least as close, and
        lambda = N(H <= H_cand) (k + 1)/(N_pop + 1). """

    with tempfile.TemporaryDirectory() as tmp:

        population = _population(tmp)
        known = np.array([12.0, 16.0])
        n_pop = len(population['q'])

        # Up to 0.035, three of the close orbits (0.01, 0.02, 0.03); at exactly 0.05, five, the
        #   orbit at the candidate's distance being at least as close
        result_3, result_5 = dnp.chanceAssociation(METEOR, [(0.035, 20.0), (0.05, 22.0)], 'DSH',
            population, known)

        assert result_3['n_closer'] == 3
        assert result_5['n_closer'] == 5

        for result, k, h_mag in ((result_3, 3, 20.0), (result_5, 5, 22.0)):

            n_large = 2 + int(np.count_nonzero(population['h_mag'] <= h_mag))

            assert result['reason'] is None
            assert result['n_at_least_as_large'] == n_large
            assert result['n_population'] == n_pop
            assert abs(result['expected'] - n_large*(k + 1)/(n_pop + 1)) < 1e-12
            assert abs(result['p_value'] - (1.0 - math.exp(-result['expected']))) < 1e-12

            upper = 0.5*scipy.stats.chi2.ppf(0.95, 2*(k + 1))
            assert abs(result['expected_upper'] - n_large*upper/n_pop) < 1e-12
            assert result['p_value_upper'] > result['p_value']


def test_no_closer_orbit_gives_the_smallest_resolvable_p():
    """ A candidate closer than any synthetic orbit has k = 0, p1 = 1/(N_pop + 1) rather than 0, and an
        upper limit set by the 3.0 events a Poisson count of zero allows at 95 %. """

    assert abs(dnp.poissonUpperLimit(0) - 2.9957) < 1e-4

    with tempfile.TemporaryDirectory() as tmp:

        population = _population(tmp)

        result = dnp.chanceAssociation(METEOR, [(0.005, 18.0)], 'DSH', population, [])[0]

        assert result['n_closer'] == 0
        assert abs(result['p1'] - 1.0/(len(population['q']) + 1)) < 1e-15
        assert result['p_value'] > 0.0
        assert abs(result['expected_upper'] - result['n_at_least_as_large']*dnp.poissonUpperLimit(0)
            /len(population['q'])) < 1e-12


def test_global_expected_is_the_sum_over_the_background_ranked_by_size():
    """ lambda_global = sum over n = 1..N of min(1, lambda/n), summed here term by term, and about
        lambda*(ln N + 0.58) when lambda is small. """

    for n_background in (1, 7, 1000, 802448):

        n = np.arange(1, n_background + 1)

        for lam in (1e-6, 0.003, 0.5, 1.0, 2.7, 40.0, 5e6):

            brute = np.minimum(1.0, lam/n).sum()
            closed = dnp.globalExpected(lam, n_background)

            assert abs(closed - brute) < 1e-9*max(1.0, brute), (n_background, lam, closed, brute)

    assert abs(dnp.globalExpected(0.01, 802448)/0.01 - (math.log(802448) + np.euler_gamma)) < 1e-3

    # Arrays in, arrays out
    assert np.shape(dnp.globalExpected(np.array([0.1, 1.0]), 100)) == (2,)


def test_sigma_is_the_two_sided_gaussian_equivalent():
    """ 2 sigma is p = 0.0455 and 3 sigma p = 0.0027, the two-sided convention. """

    assert abs(dnp.pValueToSigma(0.0455) - 2.0) < 2e-3
    assert abs(dnp.pValueToSigma(0.0027) - 3.0) < 2e-3
    assert dnp.pValueToSigma(1.0) == 0.0


def test_global_p_value_corrects_for_the_search_and_is_reported():
    """ The global p-value is the local one corrected for having searched bodies of every size, so it
        is never smaller, and sigma follows from it. """

    with tempfile.TemporaryDirectory() as tmp:

        population = _population(tmp)
        known = np.array([12.0, 16.0])

        n_background = 2 + len(population['q'])

        for result in dnp.chanceAssociation(METEOR, [(0.005, 18.0), (0.035, 20.0), (0.5, 24.0)],
            'DSH', population, known):

            assert result['n_background'] == n_background
            assert abs(result['expected_global'] - dnp.globalExpected(result['expected'],
                n_background)) < 1e-12
            assert result['p_value_global'] >= result['p_value']
            assert abs(result['p_value_global'] - (1.0 - math.exp(-result['expected_global']))) < 1e-12
            assert abs(result['sigma'] - dnp.pValueToSigma(result['p_value_global'])) < 1e-12


def test_candidates_without_a_p_value_say_why():
    """ A body fainter than the model, one without H, and one without a value get no p-value. """

    with tempfile.TemporaryDirectory() as tmp:

        population = _population(tmp)

        results = dnp.chanceAssociation(METEOR, [(0.03, 25.5), (0.03, -1), (0.03, None),
            (np.inf, 20.0)], 'DSH', population, [])

        for result in results:
            assert result['p_value'] is None
            assert result['reason']

        assert 'fainter than the model' in results[0]['reason']
        assert 'absolute magnitude' in results[1]['reason']
        assert 'absolute magnitude' in results[2]['reason']
        assert 'no value' in results[3]['reason']


def test_meteoroid_outside_the_model_gets_no_p_value():
    """ The model holds only q < 1.3 AU and a < 4.2 AU, and is empty rather than unknown outside, so
        a p-value there would be 0 for the wrong reason. """

    with tempfile.TemporaryDirectory() as tmp:

        population = _population(tmp)

        for q, e, word in ((1.35, 0.1, 'q ='), (0.9, 0.8, 'a ='), (0.9, 1.0, 'not bound')):

            result = dnp.chanceAssociation((q, e, 0.1, 1.0, 2.0), [(0.03, 20.0)], 'DSH',
                population, [])[0]

            assert result['p_value'] is None
            assert word in result['reason']


def test_criteria_on_the_radiant_are_refused():
    """ The synthetic orbits have no radiant, so a criterion on it cannot be evaluated against them. """

    with tempfile.TemporaryDirectory() as tmp:

        population = _population(tmp)

        for key in ('DN', 'DR', 'DX', 'DVVIDA'):

            try:
                dnp.chanceAssociation(METEOR, [(0.1, 20.0)], key, population, [])
                raise AssertionError("{:s} was not refused".format(key))

            except ValueError as err:
                assert 'radiant' in str(err)


def test_every_criterion_on_the_elements_can_be_used():
    """ The population is compared with the meteoroid in one call per criterion, which every criterion
        on the elements supports. """

    from wmpl.Utils.Dcriteria import ELEMENT_CRITERIA_KEYS

    with tempfile.TemporaryDirectory() as tmp:

        population = _population(tmp)

        for key in ELEMENT_CRITERIA_KEYS:

            result = dnp.chanceAssociation(METEOR, [(0.5, 20.0)], key, population, [])[0]

            assert result['reason'] is None, key
            assert 0 <= result['n_closer'] <= len(population['q']), key


def test_download_keeps_only_a_file_that_matches_its_hash():
    """ A good download is moved into place; a corrupted one is removed and leaves no partial file. """

    with tempfile.TemporaryDirectory() as tmp:

        source = _writePopulation(os.path.join(tmp, 'source.dat.gz'))
        url = 'file://' + source
        target = os.path.join(tmp, 'share', 'population.dat.gz')

        good = _model(sha256=dnp._sha256(source))

        path = dnp.downloadDebiasedNEOPopulation(good, file_path=target, url=url, verbose=False)

        assert path == target
        assert dnp._sha256(target) == good['sha256']
        assert oct(os.stat(target).st_mode & 0o044) == oct(0o044 & ~_umask()), \
            "the downloaded file is not readable by others"

        os.remove(target)

        bad = _model(sha256='0'*64)

        try:
            dnp.downloadDebiasedNEOPopulation(bad, file_path=target, url=url, verbose=False)
            raise AssertionError("a file with the wrong hash was accepted")

        except IOError as err:
            assert 'SHA-256' in str(err)

        assert not os.path.exists(target)
        assert not glob.glob(os.path.join(tmp, 'share', '*.part'))


def test_a_file_already_there_is_not_downloaded_again():
    """ With the file in place, loading needs no network: the URL here cannot be reached. """

    with tempfile.TemporaryDirectory() as tmp:

        share_path = _writePopulation(os.path.join(tmp, 'population.dat.gz'))
        model = _model(share_path=share_path, url='file:///nonexistent/population.dat.gz')

        assert dnp.findDebiasedNEOPopulation(model) == share_path

        population = dnp.loadDebiasedNEOPopulation(model)

        assert population['file_path'] == os.path.abspath(share_path)


def test_a_missing_file_is_reported_when_downloading_is_off():
    """ With download=False a missing file raises, naming where it was looked for and its URL. """

    saved = dnp.CACHE_DIR

    with tempfile.TemporaryDirectory() as tmp:

        dnp.CACHE_DIR = os.path.join(tmp, 'cache')

        try:
            model = _model(share_path=os.path.join(tmp, 'share', 'missing.dat.gz'))

            try:
                dnp.loadDebiasedNEOPopulation(model, download=False)
                raise AssertionError("a missing file did not raise")

            except IOError as err:
                assert model['url'] in str(err)

        finally:
            dnp.CACHE_DIR = saved


def test_the_published_file_when_it_has_been_downloaded():
    """ The real population: its size, its region, and one count measured when the module was written,
        the 84 synthetic orbits within D_SH = 0.1 of the orbit of the Winchcombe meteorite. """

    path = dnp.findDebiasedNEOPopulation()

    if path is None:
        print("  skipped: the population of Granvik et al. (2018) has not been downloaded")
        return

    population = dnp.loadDebiasedNEOPopulation(file_path=path)

    assert len(population['q']) == dnp.GRANVIK2018['n_objects']
    assert population['q'].max() < dnp.GRANVIK2018['q_max']
    assert population['a'].max() < dnp.GRANVIK2018['a_max']
    assert population['h_mag'].min() > dnp.GRANVIK2018['h_min']
    assert population['h_mag'].max() <= dnp.GRANVIK2018['h_max']

    winchcombe = (2.585*(1 - 0.618), 0.618, math.radians(0.460), math.radians(160.195),
        math.radians(351.79))

    result = dnp.chanceAssociation(winchcombe, [(0.1, 20.0)], 'DSH', population, [])[0]

    assert result['n_closer'] == 84


def _umask():
    """ The process umask, which os.umask can only read by setting it. """

    umask = os.umask(0)
    os.umask(umask)

    return umask


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
