""" A debiased model of the near-Earth object population, and the probability that a parent body
candidate lies as close to a meteoroid's orbit by chance.

The model is that of Granvik et al. (2018): 802 000 synthetic NEOs with 17 < H < 25, q < 1.3 AU and
a < 4.2 AU, given by their orbits and absolute magnitudes. It is not the known population: it is a
realization of the model fitted to the Catalina Sky Survey detections once the survey biases are
removed, at the true numbers, so the count of synthetic objects brighter than some H is the number
of real NEOs brighter than that H the model predicts. The angular elements are drawn uniformly, the
model being one in (a, e, i, H).

The file is not distributed with wmpl. It is downloaded on first use, 28 MB, from the authors' page,
checked against its SHA-256 and kept in wmpl/share (or ~/.cache/wmpl if that cannot be written).
Granvik et al. publish it under CC BY 4.0: cite the paper when using it.

    Granvik, Morbidelli, Jedicke, Bolin, Bottke, Beshore, Vokrouhlicky, Nesvorny & Michel (2018),
    Debiased orbit and absolute-magnitude distributions for near-Earth objects, Icarus 312, 181-207,
    doi:10.1016/j.icarus.2018.04.018

The chance-coincidence test (chanceAssociation) asks: given the meteoroid's orbit, how many NEOs at
least as large as the candidate would lie at least as close to it by chance? The null hypothesis is
that the candidate is an unrelated member of the background population. Under it the candidate's
orbit is the random one and the meteoroid's is the observed datum, so the meteoroid's orbit is kept
fixed and compared with every synthetic orbit:

    p1 = (k + 1)/(N_pop + 1)
                           the probability that a background orbit is at least as close, k of the
                           N_pop synthetic orbits being so. Under the null hypothesis the candidate's
                           orbit is one more draw of the same distribution, so its rank among the
                           N_pop + 1 is uniform and this estimate is exact rather than approximate
                           (the usual Monte Carlo p-value, e.g. North, Curtis & Sham 2002, AJHG 71,
                           439). Unlike k/N_pop it is never 0.
    N(H <= H_cand)         the number of NEOs at least as large as the candidate
    lambda = N(H <= H_cand) * p1
                           the expected number of such NEOs as close by chance
    p = 1 - exp(-lambda)   the probability of at least one, the p-value of this body

N(H <= H_cand) counts every body at least as large that the candidate could have been. p is
therefore the right value for a body named before the search, such as one proposed in the
literature.

The global p-value. A body found by the search, as the best of thousands of every size, needs a
further correction: a small body needs a smaller D to reach the same p, but small bodies are
numerous, so somewhere in the background some body reaches a low p by chance. With the background
NEOs ranked by size, n = 1 the largest up to N_tot, all those to H = 25, and the orbit taken as
independent of size, F(D) is uniform for each of them. The n-th then has lambda_n = n*F(D) <= lambda
with probability min(1, lambda/n), and the expected number of background bodies with a lambda at
least as small is

    lambda_global = sum over n = 1..N_tot of min(1, lambda/n), about lambda*(ln N_tot + 0.58)
    p_global = 1 - exp(-lambda_global)

N_tot is 802 448, so the factor is about 14. Measured on the synthetic population itself, averaged
over 200 Earth-crossing orbits taken as the meteoroid, the true count is 0.6 to 1.0 times
lambda_global (0.14 against 0.22 at lambda = 0.03, 7.4 against 7.5 at lambda = 1), so p_global is
conservative. It is the value to quote for a body the search found.

Significance is given as the two-sided Gaussian equivalent of p_global, the convention in which
p = 0.0455 is 2 sigma and p = 0.0027 is 3 sigma. With several criteria, choose one before looking:
the smallest p over several criteria is not itself a p-value.

Why not random pairs. A procedure used in the literature, after Borovicka et al. (2013), draws pairs
of synthetic NEOs until one is as close as the observed pair, which gives the fraction of all pairs
of NEOs that are that close. That replaces the meteoroid by a typical NEO. It is not one: a meteoroid
has to hit the Earth, which favours the low-inclination Earth-crossing orbits where the population
is densest.
Measured with D_SH on this model, the fraction of the population within D_SH = 0.1 of the orbit of
the Winchcombe meteorite is 2.6 times the random-pair fraction, while for a Geminid-like orbit
(q = 0.14 AU, i = 22 deg) it is 0.06 times. Random pairs would overstate the significance of a match
by a factor of 2 to 3 for a typical fireball, and understate it by a factor of 16 in a sparse region.
That method is therefore not offered.

What the test assumes:
    - The orbital distribution is taken as independent of H: p1 is measured over the whole
      population, for statistics, rather than over the objects as large as the candidate.
    - The model covers q < 1.3 AU and a < 4.2 AU only. A meteoroid outside that region gets no
      p-value, and one near its edge gets a p-value biased low, the population beyond the edge being
      missing rather than empty.
    - The model has no object with H < 17, where the known population is nearly complete, so
      N(H <= H_cand) adds the known NEOs with H <= 17 to the synthetic ones with 17 < H <= H_cand.
      The two join without a step: at H = 17.5 the sum is 726 against 723 known NEOs.
    - The candidate needs an absolute magnitude, so comets and meteorites get no p-value.
    - Only the criteria on the orbital elements are offered. Those on the radiant need the radiant
      of each synthetic orbit, which exists only where it comes close to the Earth's.
    - The meteoroid's orbit is taken as exact. Its uncertainty is not propagated.

Resolution. The population holds 802 000 orbits, so p1 is never below 1/802 001: a candidate closer
than every synthetic orbit has k = 0 and that value of p1, which is then an upper bound on how unusual
it is. The count k is also a Poisson draw of the model, and an upper limit on lambda from it, at the
given confidence, is kept with every value.
"""

from __future__ import print_function, division, absolute_import

import os
import gzip
import shutil
import hashlib
import tempfile

import numpy as np
import scipy.special
import scipy.stats

from wmpl.Config import config
from wmpl.Utils.Dcriteria import lookupCriterion


# The Granvik et al. (2018) model as published on the authors' page
GRANVIK2018 = {
    'name': 'Granvik et al. (2018)',
    'file_name': 'Granvik+_2018_Icarus.dat.gz',
    'share_path': config.granvik2018_neo_file,
    'url': 'https://www.mv.helsinki.fi/home/mgranvik/data/Granvik+_2018_Icarus/'
        'Granvik+_2018_Icarus.dat.gz',
    'sha256': '5e6304118a9c408a9715d9a6bcaf9fc816d8606bebc3f61f5e6acab687c947a3',
    'size_mb': 28,
    'n_objects': 802000,
    'reference': 'Granvik, M., Morbidelli, A., Jedicke, R., Bolin, B., Bottke, W. F., Beshore, E., '
        'Vokrouhlicky, D., Nesvorny, D. & Michel, P. (2018), Debiased orbit and absolute-magnitude '
        'distributions for near-Earth objects, Icarus 312, 181-207, doi:10.1016/j.icarus.2018.04.018',
    'license': 'CC BY 4.0',
    # The region and the H range the model covers
    'h_min': 17.0,
    'h_max': 25.0,
    'q_max': 1.3,
    'a_max': 4.2,
    }

# The confidence of the Poisson upper limits
CHANCE_CONFIDENCE = 0.95

# Where the file goes if wmpl/share cannot be written, as in an installed package
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'wmpl')

# The population, once loaded
_population_cache = {}


def _candidatePaths(model=GRANVIK2018):
    """ The places the file of the model is looked for, in order: wmpl/share, then the cache. """

    return [model['share_path'], os.path.join(CACHE_DIR, model['file_name'])]


def _sha256(file_path):
    """ SHA-256 of a file, as a hex string. """

    digest = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)

    return digest.hexdigest()


def findDebiasedNEOPopulation(model=GRANVIK2018):
    """ The path of the file of the model if it has been downloaded, None otherwise. """

    for path in _candidatePaths(model):
        if os.path.isfile(path):
            return path

    return None


def downloadDebiasedNEOPopulation(model=GRANVIK2018, file_path=None, url=None, verbose=True):
    """ Download the file of the model and check it against its SHA-256.

        The file is written beside a temporary name and moved into place only once it has been
        checked, so an interrupted or corrupted download never leaves a file that looks complete.

    Keyword arguments:
        model: [dict] The model, GRANVIK2018.
        file_path: [str] Where to write it. By default wmpl/share, or ~/.cache/wmpl if that cannot be
            written.
        url: [str] Where to download it from, the authors' page by default.
        verbose: [bool] Print what is being downloaded.

    Return:
        [str] The path of the file.
    """

    from urllib.request import urlopen

    url = model['url'] if url is None else url

    targets = [file_path] if file_path is not None else _candidatePaths(model)

    last_error = None

    for target in targets:

        target_dir = os.path.dirname(os.path.abspath(target))

        try:
            os.makedirs(target_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix='.' + model['file_name'] + '.', suffix='.part',
                dir=target_dir)

        except OSError as err:
            last_error = err
            continue

        try:

            if verbose:
                print("Downloading the debiased NEO population of {:s} ({:d} MB), once,".format(
                    model['name'], model['size_mb']))
                print("  from {:s}".format(url))
                print("  to {:s}".format(target))

            with os.fdopen(fd, 'wb') as f_out, urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, f_out)

            digest = _sha256(tmp_path)

            if digest != model['sha256']:
                raise IOError("The downloaded file does not match the published one (SHA-256 {:s}, "
                    "expected {:s}). It was not kept.".format(digest, model['sha256']))

            os.replace(tmp_path, target)

            # mkstemp makes the file readable by its owner only, which is wrong for shared data
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(target, 0o644 & ~umask)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if verbose:
            print("  done. Please cite {:s}, published under {:s}.".format(model['reference'],
                model['license']))

        return target

    raise IOError("Could not write the debiased NEO population to {:s}: {:s}".format(
        " or ".join(targets), str(last_error)))


def loadDebiasedNEOPopulation(model=GRANVIK2018, file_path=None, download=True, verbose=True):
    """ Load the synthetic NEOs of the model, downloading the file first if it is not there.

    Keyword arguments:
        model: [dict] The model, GRANVIK2018.
        file_path: [str] The file to read, instead of looking for it in wmpl/share and the cache.
        download: [bool] Download the file if it is not found. If False, raise instead.
        verbose: [bool] Print a note when downloading.

    Return:
        [dict] Arrays of 'q' and 'a' (AU), 'e', 'incl', 'node', 'peri' (rad) and 'h_mag', a value per
            synthetic object, and 'model' and 'file_path'.
    """

    if file_path is None:

        file_path = findDebiasedNEOPopulation(model)

        if file_path is None:

            if not download:
                raise IOError("The debiased NEO population of {:s} is not in {:s}. Download it from "
                    "{:s}.".format(model['name'], " or ".join(_candidatePaths(model)), model['url']))

            file_path = downloadDebiasedNEOPopulation(model, verbose=verbose)

    file_path = os.path.abspath(file_path)

    if file_path in _population_cache:
        return _population_cache[file_path]

    opener = gzip.open if file_path.endswith('.gz') else open

    with opener(file_path, 'rt') as f:
        data = np.loadtxt(f, ndmin=2)

    # Columns: a (AU), e, i, node, argument of perihelion, mean anomaly (deg), H
    a, e = data[:, 0], data[:, 1]

    population = {'q': a*(1.0 - e), 'a': a, 'e': e, 'incl': np.radians(data[:, 2]),
        'node': np.radians(data[:, 3]), 'peri': np.radians(data[:, 4]), 'h_mag': data[:, 6],
        'model': model, 'file_path': file_path}

    _population_cache[file_path] = population

    return population


def isInModelRegion(q, e, model=GRANVIK2018):
    """ Whether an orbit lies in the region the model covers, q < q_max and a < a_max.

    Return:
        (inside, reason): [tuple] True and None, or False and why not.
    """

    if e >= 1.0:
        return False, "the orbit is not bound, and the model holds only orbits with a < {:.1f} AU".format(
            model['a_max'])

    a = q/(1.0 - e)

    if q >= model['q_max']:
        return False, "q = {:.3f} AU is outside the model, which holds only q < {:.1f} AU".format(q,
            model['q_max'])

    if a >= model['a_max']:
        return False, "a = {:.3f} AU is outside the model, which holds only a < {:.1f} AU".format(a,
            model['a_max'])

    return True, None


def countAtLeastAsLarge(h_mag, population, known_h_mags):
    """ The number of NEOs at least as large as a body of absolute magnitude h_mag: the known ones with
        H <= min(h_mag, h_min of the model), where the known population is nearly complete, plus the
        synthetic ones with h_min < H <= h_mag.

    Arguments:
        h_mag: [float or ndarray] Absolute magnitude of the body, or of several.
        population: [dict] Return of loadDebiasedNEOPopulation.
        known_h_mags: [ndarray] Absolute magnitudes of the known NEOs.

    Return:
        (n_known, n_model): [tuple] the two parts of the count, ints or arrays of them.
    """

    h_min = population['model']['h_min']

    known_sorted = np.sort(np.asarray(known_h_mags, dtype=float))
    model_sorted = np.sort(population['h_mag'][population['h_mag'] > h_min])

    h_mag = np.asarray(h_mag, dtype=float)

    n_known = np.searchsorted(known_sorted, np.minimum(h_mag, h_min), side='right')
    n_model = np.searchsorted(model_sorted, h_mag, side='right')

    if h_mag.ndim == 0:
        return int(n_known), int(n_model)

    return n_known, n_model


def poissonUpperLimit(k, confidence=CHANCE_CONFIDENCE):
    """ One-sided upper limit on the mean of a Poisson count of which k were observed (Garwood 1936).
        For k = 0 at 95 % it is 3.0.
    """

    return 0.5*scipy.stats.chi2.ppf(confidence, 2*(np.asarray(k) + 1))


def globalExpected(expected, n_background):
    """ The expected number of background NEOs, of any size, with an expected count at least as small
        as the candidate's: the sum over n = 1..n_background of min(1, expected/n). See the module
        docstring.

    Arguments:
        expected: [float or ndarray] lambda of the candidate, N(H <= H_cand)*p1.
        n_background: [int] The number of background NEOs to the faint end of the model.

    Return:
        [float or ndarray] lambda_global.
    """

    def harmonic(n):
        return scipy.special.digamma(n + 1.0) + np.euler_gamma

    expected = np.asarray(expected, dtype=float)

    # The terms with n <= lambda are 1, the others lambda/n
    m = np.minimum(np.floor(expected), n_background)

    out = m + expected*(harmonic(n_background) - harmonic(m))

    return out[()]


def pValueToSigma(p_value):
    """ The two-sided Gaussian equivalent of a p-value, in which 0.0455 is 2 sigma and 0.0027 is
        3 sigma. """

    return scipy.stats.norm.isf(np.asarray(p_value, dtype=float)/2.0)[()]


def chanceAssociation(meteor_orbit, candidates, criterion_key, population, known_h_mags,
    confidence=CHANCE_CONFIDENCE):
    """ For each candidate, the probability that a NEO at least as large lies at least as close to the
        meteoroid's orbit by chance, for the candidate alone and corrected for having searched bodies
        of every size. See the module docstring for the test and what it assumes.

    Arguments:
        meteor_orbit: [tuple] (q (AU), e, incl, node, peri) of the meteoroid, the angles in radians.
        candidates: [list] (d_value, h_mag) of each candidate: its value of the criterion against the
            meteoroid's orbit and its absolute magnitude, -1 or None if it has none.
        criterion_key: [str] Key of the criterion on the orbital elements (wmpl.Utils.Dcriteria).
        population: [dict] Return of loadDebiasedNEOPopulation.
        known_h_mags: [ndarray] Absolute magnitudes of the known NEOs, for the bodies brighter than
            the model covers.

    Keyword arguments:
        confidence: [float] Confidence of the Poisson upper limits, 0.95 by default.

    Return:
        [list] For each candidate a dictionary: 'p_value' of the candidate alone, 'p_value_global'
            corrected for the search and its 'sigma', the 'expected' and 'expected_global' counts
            (lambda), the same from the Poisson upper limit on k as 'p_value_upper',
            'p_value_global_upper', 'expected_upper' and 'expected_global_upper', 'n_closer' (k,
            synthetic orbits at least as close), 'n_population', 'p1', 'n_at_least_as_large' with its
            'n_known' and 'n_model' parts, 'n_background', 'is_limit' (True when no synthetic orbit is
            as close, so that p is at most and sigma at least the value given, which the size of the
            population cannot resolve), and 'reason', None if the values were computed or why not, in
            which case they are None.
    """

    key, label, func, kind = lookupCriterion(criterion_key)

    if kind != 'elements':
        raise ValueError("{:s} is a criterion on the radiant. The synthetic NEOs have only orbital "
            "elements, and a radiant derived from them exists only where an orbit comes close to "
            "the Earth's, so it cannot be compared with the whole population.".format(label))

    model = population['model']
    n_population = len(population['q'])

    # Every background NEO to the faint end of the model, the count the global correction runs over
    n_background = sum(countAtLeastAsLarge(model['h_max'], population, known_h_mags))

    fields = ['p_value', 'p_value_global', 'sigma', 'expected', 'expected_global', 'p_value_upper',
        'p_value_global_upper', 'expected_upper', 'expected_global_upper', 'n_closer', 'p1',
        'n_at_least_as_large', 'n_known', 'n_model', 'is_limit']

    empty = dict((field, None) for field in fields)
    empty.update({'n_population': n_population, 'n_background': n_background, 'reason': None})

    inside, reason = isInModelRegion(meteor_orbit[0], meteor_orbit[1], model)

    if not inside:
        return [dict(empty, reason="for the meteoroid, " + reason) for _ in candidates]

    # Why a candidate gets no value, or None if it gets one
    reasons = []

    for d_value, h_mag in candidates:

        if (d_value is None) or (not np.isfinite(d_value)):
            reasons.append("no value of {:s}".format(label))

        elif (h_mag is None) or (h_mag < 0):
            reasons.append("no absolute magnitude")

        elif h_mag > model['h_max']:
            reasons.append("H = {:.2f} is fainter than the model, which reaches H = {:.0f}".format(
                h_mag, model['h_max']))

        else:
            reasons.append(None)

    valid = np.array([r is None for r in reasons], dtype=bool)

    values = {}

    if np.any(valid):

        d_values = np.array([c[0] for c in candidates], dtype=object)[valid].astype(float)
        h_mags = np.array([c[1] for c in candidates], dtype=object)[valid].astype(float)

        # The criterion between the meteoroid and every synthetic orbit, once for all the candidates
        population_orbits = (population['q'], population['e'], population['incl'],
            population['node'], population['peri'])

        with np.errstate(divide='ignore', invalid='ignore'):
            d_population = np.asarray(func(tuple(meteor_orbit), population_orbits), dtype=float)

        # A pair the criterion is undefined for (e.g. the MOID of two open orbits) is not a closer
        #   pair
        d_population = np.sort(d_population[np.isfinite(d_population)])

        # At least as close: the synthetic orbits with a value up to the candidate's
        k = np.searchsorted(d_population, d_values, side='right')

        n_known, n_model = countAtLeastAsLarge(h_mags, population, known_h_mags)
        n_large = n_known + n_model

        p1 = (k + 1.0)/(n_population + 1.0)
        expected = n_large*p1
        expected_upper = n_large*poissonUpperLimit(k, confidence)/n_population

        expected_global = globalExpected(expected, n_background)
        expected_global_upper = globalExpected(expected_upper, n_background)

        p_value_global = -np.expm1(-expected_global)

        values = {'p_value': -np.expm1(-expected), 'p_value_global': p_value_global,
            'sigma': pValueToSigma(p_value_global), 'expected': expected,
            'expected_global': expected_global, 'p_value_upper': -np.expm1(-expected_upper),
            'p_value_global_upper': -np.expm1(-expected_global_upper),
            'expected_upper': expected_upper, 'expected_global_upper': expected_global_upper,
            'n_closer': k, 'p1': p1, 'n_at_least_as_large': n_large, 'n_known': n_known,
            'n_model': n_model, 'is_limit': k == 0}

    results = []
    j = 0

    for reason in reasons:

        if reason is not None:
            results.append(dict(empty, reason=reason))
            continue

        result = dict(empty)

        for field, array in values.items():
            value = np.atleast_1d(array)[j]
            if field == 'is_limit':
                result[field] = bool(value)

            elif field in ('n_closer', 'n_at_least_as_large', 'n_known', 'n_model'):
                result[field] = int(value)

            else:
                result[field] = float(value)

        results.append(result)
        j += 1

    return results
