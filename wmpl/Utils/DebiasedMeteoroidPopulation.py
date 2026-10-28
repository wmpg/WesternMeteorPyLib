""" The orbits of the meteoroids that hit the Earth, from the near-Earth object model of Deam et al.
(2026), and the probability that a meteoroid lies as close to a parent body candidate by chance.

The model of Deam et al. (2026) describes the near-Earth objects at meteoroid sizes, about 1 cm to
1 m, fitted to sporadic meteoroids observed by the Global Fireball Observatory, on the bins and the
residence-time maps of the model of Nesvorny et al. (2023). It is binned in (a, e, i, H): 42 x 25 x 22
bins over a < 4.2 AU, e < 1, i < 88 deg, and 28 bins of 0.25 over 34.5 < H < 41.5. At each H it is a
sum over six source regions (the nu6 and the 3:1, 5:2 and 2:1 resonances, the inner belt and the
Jupiter-family comets) of the residence-time map of the source, the time its objects spend in each
bin, weighted by the strength of the source. The strengths
are linear in H between H = 34.5, 36.75, 39.25 and 41.5. The maps are those for a collisional
lifetime of 3 Myr up to an H that is fitted, and of 1 Myr past it: H = 38.5 in 4122 of the 4125
posterior samples. The model gives the relative orbital distribution at each H, not the numbers of
objects.

Two files are needed, 3.5 MB together, and are not distributed with wmpl. They are downloaded on
first use from the authors' Zenodo record, checked against their SHA-256 and kept in wmpl/share (or
~/.cache/wmpl if that cannot be written), as wmpl.Utils.DebiasedNEOPopulation does: the posterior
samples of the source strengths, and the residence-time maps. Deam & Nesvorny publish them under CC BY
4.0: cite the paper and the data.

    Deam, S. E., et al. (2026), A Near-Earth Object Model Calibrated to Earth Impactors, AJ 171, 126,
    doi:10.3847/1538-3881/ae2fc5
    Deam, S. E. & Nesvorny, D. (2025), Sampling and querying code for the Near-Earth Object Model
    Calibrated to Earth Impactors, Zenodo, doi:10.5281/zenodo.17809512

The background. The model gives where the objects are, not which of them hit the Earth, and a
meteoroid that was observed is one that did: 34 to 41 % of the model's population at H = 35 to 41 does
not even cross the Earth's orbit. Each orbit is therefore weighted by its rate of impact on the Earth,
that of Opik (1951) for a target on a circular orbit of 1 AU, with gravitational focusing:

    P ~ U (1 + v_esc^2/v_inf^2) / (a^1.5 sin i |U_x|)
        U^2 = 3 - 1/a - 2 sqrt(a (1 - e^2)) cos i       the encounter speed, in units of the Earth's
        U_x^2 = 2 - 1/a - a (1 - e^2)                  its radial part, zero where the orbit only
                                                       touches 1 AU
        v_inf = U v_E                                  v_E = 29.78 km/s, v_esc = 11.10 km/s at 100 km

and zero for an orbit that does not cross 1 AU. This changes the background a great deal: measured
with D_SH at H = 38, the fraction of the impactors within 0.1 of a low-inclination Apollo orbit
(q = 0.99 AU, i = 0.5 deg) is 7.0 times that of the population, and of an Apollo orbit with q = 0.8 AU
and i = 10 deg 4.0 times.

Within a bin a and e are taken uniform, as the authors' sampler takes them, and the orbital pole
uniform on the sphere, so that the density in i goes as sin i. The authors' sampler takes i uniform
instead, which for the population is a fine difference but not for the impacts: with i uniform down to
0, Opik's 1/sin i makes the impacts from the lowest bin depend on where 1/sin i is cut off. Cut off at
the Earth's radius, 57 to 67 % of the impacts at H = 35 to 41 come from i < 4 deg, 6 to 8 % of the
population. With the pole uniform, the sin i of the density cancels the 1/sin i of the rate and there
is nothing to cut off.

An impactor needs its angles too. Its node is taken uniform, the Earth meeting it at any time of the
year, and its argument of perihelion is one of the four that put a node at 1 AU, chosen at random.
The orbits are drawn by importance sampling, with e drawn so that the proposal follows 1/|U_x|, and
carry weights of mean 1. At H = 35 to 41 their effective number is 0.93 to 0.94 times their number,
and the largest weight is about 10.

The chance-coincidence test (chanceAssociation) asks: given the candidate's orbit, how often would
an unrelated meteoroid lie at least as close to it? The null hypothesis is that the meteoroid is a
member of the background, delivered to that orbit by the dynamical evolution of its source region and
not by the candidate. Under it the meteoroid's orbit is the random one and the candidate's is fixed,
the converse of the test of wmpl.Utils.DebiasedNEOPopulation, which keeps the meteoroid fixed and
draws the candidate:

    p = (k + 1)/(N + 1)
                           the probability that a background meteoroid is at least as close to the
                           candidate, k being the weight of the N background orbits that are. Under
                           the null hypothesis the meteoroid is one more draw, so for weights of 1
                           its rank among the N + 1 is uniform and this is the usual Monte Carlo
                           p-value. It is never 0.

The candidate's size plays no part, so comets and bodies without an absolute magnitude get a p too. p
is the value for a body named before the search, such as one proposed in the literature.

The global p-value. A body found by the search, as the closest of a catalogue of tens of thousands,
needs a correction: some body is close to almost any orbit. With K background meteoroids, each
compared with every body of the catalogue,

    p_global = (k_global + 1)/(K + 1)
                           k_global being the weight of those whose closest body is at least as close
                           as the candidate is to the meteoroid,

the probability that a background meteoroid has some body of the catalogue at least as close. The
candidate is one of the bodies, so p_global is at least p. When the smaller K gives less, by sampling,
it is taken as p. Significance is given as the two-sided Gaussian equivalent of p_global, as in
wmpl.Utils.DebiasedNEOPopulation.

What the test assumes:
    - The meteoroid's H selects the model: the bins within H +- 0.25, as the authors' sampler takes
      them, so H can be given between 34.75 and 41.25. The source strengths change with H, and with
      them the fraction of the background within D_SH = 0.1 of the two Apollo orbits above, by 2.0 and
      0.66 times from H = 35 to 41. H is that of the model, D = 1329 km 10^(-H/5)/sqrt(p_V).
    - The posterior is averaged: the population is the mean of the 4125 posterior samples, and the
      empty bins are filled with 1e-8 before each map is normalised, as in the authors' sampler.
    - The Earth's orbit is taken circular. An orbit with q up to 1.017 AU, or Q down to 0.983 AU, can
      hit the Earth but has no counterpart in the background, so a meteoroid there gets a p-value
      biased low.
    - The model covers a < 4.2 AU and i < 88 deg, prograde. A meteoroid outside it gets no p-value.
    - Only the criteria on the orbital elements are offered, and not the MOID, which costs 0.09 to
      0.15 ms a pair, 1.5 to 2.5 minutes a body against 10^6 background orbits.
    - The global correction is on the criterion, not on p: a body in a sparse part of orbit space at a
      larger D could be the more significant, and is not found as such.
    - The meteoroid's orbit is taken as exact. Its uncertainty is not propagated.

Resolution. N background orbits give p no lower than 1/(N + 1), and K give p_global no lower than
1/(K + 1). When no background orbit is as close, the value is an upper bound.
"""

from __future__ import print_function, division, absolute_import

import os
import tarfile

import numpy as np

from wmpl.Config import config
from wmpl.Utils.Dcriteria import lookupCriterion
from wmpl.Utils.OrbitConstants import GAUSS_K
from wmpl.Utils.DebiasedNEOPopulation import (CACHE_DIR, findDebiasedNEOPopulation,
    downloadDebiasedNEOPopulation, pValueToSigma)


_ZENODO = 'https://zenodo.org/records/17809512/files/'

_REFERENCE = ('Deam, S. E., et al. (2026), A Near-Earth Object Model Calibrated to Earth Impactors, '
    'AJ 171, 126, doi:10.3847/1538-3881/ae2fc5')

_DATA_REFERENCE = ('Deam, S. E. & Nesvorny, D. (2025), Sampling and querying code for the Near-Earth '
    'Object Model Calibrated to Earth Impactors, Zenodo, doi:10.5281/zenodo.17809512')

# The files of the model, each described as wmpl.Utils.DebiasedNEOPopulation describes a model, so that
#   its functions find, download and check them
DEAM2026_POSTERIOR = {
    'name': 'Deam et al. (2026), posterior samples',
    'file_name': 'Deam+_2026_posterior_samples.dat',
    'share_path': config.deam2026_posterior_file,
    'url': _ZENODO + 'posterior_samples.dat?download=1',
    'sha256': 'ccc509645f850fa2b85613700d740b3d6ce6b90a7c564a0dc9ece0156c79dc32',
    'size_mb': 3.0,
    'reference': _REFERENCE + ', and ' + _DATA_REFERENCE,
    'license': 'CC BY 4.0',
    }

DEAM2026_RESIDENCE = {
    'name': 'Deam et al. (2026), residence-time maps',
    'file_name': 'Deam+_2026_residence_time_distributions.tar.gz',
    'share_path': config.deam2026_residence_file,
    'url': _ZENODO + 'residence_time_distributions.tar.gz?download=1',
    'sha256': '9b9e3a9ffb446dcffdbd61ec306ae9b2c73ceee4ad1ab24d39f89f47f749a7fe',
    'size_mb': 0.5,
    'reference': _REFERENCE + ', and ' + _DATA_REFERENCE,
    'license': 'CC BY 4.0',
    }

DEAM2026 = {
    'name': 'Deam et al. (2026)',
    'files': (DEAM2026_POSTERIOR, DEAM2026_RESIDENCE),
    'reference': _REFERENCE,
    'data_reference': _DATA_REFERENCE,
    'license': 'CC BY 4.0',
    # The H the meteoroid can have, so that H +- 0.25 stays inside the model, and the region it covers
    'h_min': 34.75,
    'h_max': 41.25,
    'a_max': 4.2,
    'incl_max': 88.0,
    }

# The bins of the model (Nesvorny et al. 2023), a (AU), e, i (deg), H
A_EDGES = np.linspace(0.0, 4.2, 43)
E_EDGES = np.linspace(0.0, 1.0, 26)
INCL_EDGES = np.linspace(0.0, 88.0, 23)
H_EDGES = np.linspace(34.5, 41.5, 29)
H_CENTRES = 0.5*(H_EDGES[1:] + H_EDGES[:-1])

# The H at which each source's strength is given, in the order of the posterior columns
H_NODES = np.array([34.5, 36.75, 39.25, 41.5])
SOURCES = ('nu6', '31', 'inner', '52', '21', 'jfc')

# The residence-time map of each source, for the 3 Myr and the 1 Myr collisional lifetime
MAPS_3MY = {'nu6': 'nu6_tc3My', '31': '31', 'inner': 'inn_tc3My', '52': '52', '21': '21', 'jfc': 'JFC'}
MAPS_1MY = {'nu6': 'nu6_tc1My', '31': '31_tc1My', 'inner': 'inn_tc1My', '52': '52_tc1My',
    '21': '21_tc1My', 'jfc': 'jfc_tc1My'}

# What the authors' sampler fills the empty bins of a map with before normalising it
EMPTY_BIN = 1e-8

# The Earth's orbital speed on a circle of 1 AU, and the escape speed at 100 km, where a meteoroid
#   becomes a meteor [km/s]
SPEED_EARTH = GAUSS_K*149597870.7/86400.0
ESCAPE_SPEED = np.sqrt(2*6.67408e-11*5.9722e24/6471.0e3)/1000.0

# Background orbits for the p of each body, and the part of them compared with the whole catalogue
#   for p_global
N_SAMPLES = 1000000
N_GLOBAL = 2000
SEED = 1

# The model, once loaded
_model_cache = {}


def findDeamModel(model=DEAM2026):
    """ The paths of the files of the model if they have been downloaded, None for each one that has
        not. """

    return [findDebiasedNEOPopulation(f) for f in model['files']]


def _readMaps(tar_path):
    """ The residence-time maps of the sources, 42 x 25 x 22 each, read from the archive without
        unpacking it. """

    maps = {}

    with tarfile.open(tar_path, 'r:gz') as tar:

        for member in tar.getmembers():

            name = os.path.basename(member.name)

            if not (member.isfile() and name.endswith('.csv')):
                continue

            # Columns: ia, ie, ii (from 1), prob
            rows = np.loadtxt(tar.extractfile(member), delimiter=',', skiprows=1, ndmin=2)

            counts = np.zeros((len(A_EDGES) - 1, len(E_EDGES) - 1, len(INCL_EDGES) - 1))
            counts[rows[:, 0].astype(int) - 1, rows[:, 1].astype(int) - 1,
                rows[:, 2].astype(int) - 1] = rows[:, 3]

            counts[counts == 0.0] = EMPTY_BIN
            maps[name[:-len('.csv')]] = counts/np.sum(counts)

    missing = [m for m in list(MAPS_3MY.values()) + list(MAPS_1MY.values()) if m not in maps]

    if missing:
        raise IOError("The residence-time maps {:s} are not in {:s}.".format(", ".join(missing),
            tar_path))

    return maps


def loadDeamModel(model=DEAM2026, posterior_path=None, residence_path=None, download=True,
    verbose=True):
    """ Load the model, downloading its files first if they are not there, and average it over the
        posterior.

    Keyword arguments:
        model: [dict] The model, DEAM2026.
        posterior_path: [str] The file of posterior samples to read, instead of looking for it in
            wmpl/share and the cache.
        residence_path: [str] The archive of residence-time maps to read, likewise.
        download: [bool] Download a file that is not found. If False, raise instead.
        verbose: [bool] Print a note when downloading.

    Return:
        [dict] 'population', the posterior mean of the model, an array over (a, e, i, H) whose slice
            at each H sums to 1; 'strengths', the posterior samples of the strengths of the sources
            (samples x sources x H nodes); 'crossover', the index of the first H bin of each sample
            that uses the 1 Myr maps; 'model' and 'file_paths'; and 'impactors', where sampleImpactors
            keeps the orbits it draws from it.
    """

    paths = [posterior_path, residence_path]

    for k, (path, file_model) in enumerate(zip(paths, model['files'])):

        if path is not None:
            continue

        path = findDebiasedNEOPopulation(file_model)

        if path is None:

            if not download:
                raise IOError("The file {:s} of the model of {:s} is not in wmpl/share or {:s}. "
                    "Download it from {:s}.".format(file_model['file_name'], model['name'], CACHE_DIR,
                        file_model['url']))

            path = downloadDebiasedNEOPopulation(file_model, verbose=verbose)

        paths[k] = path

    paths = [os.path.abspath(p) for p in paths]
    key = tuple(paths)

    if key in _model_cache:
        return _model_cache[key]

    posterior = np.loadtxt(paths[0], ndmin=2)

    # Columns: four strengths per source, the crossover index and the log-likelihood
    if posterior.shape[1] != 4*len(SOURCES) + 2:
        raise IOError("{:s} has {:d} columns, not the {:d} of the posterior samples of the model."
            .format(paths[0], posterior.shape[1], 4*len(SOURCES) + 2))

    strengths = posterior[:, :4*len(SOURCES)].reshape(-1, len(SOURCES), len(H_NODES))
    crossover = posterior[:, 4*len(SOURCES)].astype(int)

    maps = _readMaps(paths[1])

    # The strength of each source at each H bin, as the authors' sampler interpolates it, and split
    #   between the two lifetimes by each sample's crossover. The model is linear in the strengths, so
    #   averaging these averages the model
    strength_h = np.empty(strengths.shape[:2] + (len(H_CENTRES),))
    for k in range(strengths.shape[0]):
        for s in range(len(SOURCES)):
            strength_h[k, s] = np.interp(H_CENTRES, H_NODES, strengths[k, s])

    long_lived = np.arange(len(H_CENTRES))[None, :] < crossover[:, None]
    mean_3my = np.mean(strength_h*long_lived[:, None, :], axis=0)
    mean_1my = np.mean(strength_h*(~long_lived)[:, None, :], axis=0)

    population = np.zeros(maps[MAPS_3MY[SOURCES[0]]].shape + (len(H_CENTRES),))

    for s, source in enumerate(SOURCES):
        population += maps[MAPS_3MY[source]][..., None]*mean_3my[s] \
            + maps[MAPS_1MY[source]][..., None]*mean_1my[s]

    deam = {'population': population, 'strengths': strengths, 'crossover': crossover,
        'model': model, 'file_paths': paths, 'impactors': {}}

    _model_cache[key] = deam

    return deam


def isInModelRegion(q, e, incl, h_mag, model=DEAM2026):
    """ Whether a meteoroid can be a member of the background: H within the model, an orbit that
        reaches the Earth's, which is not quite 1 AU from the Sun, a < a_max and i < incl_max.

    Arguments:
        q: [float] Perihelion distance (AU).
        e: [float] Eccentricity.
        incl: [float] Inclination (radians).
        h_mag: [float] Absolute magnitude of the meteoroid, in the model's scale.

    Return:
        (inside, reason): [tuple] True and None, or False and why not.
    """

    if (h_mag is None) or (not np.isfinite(h_mag)):
        return False, "the meteoroid has no absolute magnitude, and the model changes with it"

    if not (model['h_min'] <= h_mag <= model['h_max']):
        return False, "H = {:.2f} is outside the model, which takes {:.2f} to {:.2f}".format(h_mag,
            model['h_min'], model['h_max'])

    if e >= 1.0:
        return False, "the orbit is not bound, and the model holds only orbits with a < {:.1f} AU" \
            .format(model['a_max'])

    a = q/(1.0 - e)

    if a >= model['a_max']:
        return False, "a = {:.3f} AU is outside the model, which holds only a < {:.1f} AU".format(a,
            model['a_max'])

    if np.degrees(incl) >= model['incl_max']:
        return False, "i = {:.2f} deg is outside the model, which holds only i < {:.0f} deg".format(
            np.degrees(incl), model['incl_max'])

    # The Earth's distance from the Sun, 0.983 to 1.017 AU
    if (q > 1.0167) or (a*(1.0 + e) < 0.9833):
        return False, "the orbit does not reach the Earth's (q = {:.3f} AU, Q = {:.3f} AU)".format(q,
            a*(1.0 + e))

    return True, None


def impactRate(a, e, incl):
    """ The rate at which an object on an orbit hits the Earth, to a constant factor: that of Opik
        (1951) for a target on a circular orbit of 1 AU, with gravitational focusing. See the module
        docstring.

    Arguments:
        a: [float or ndarray] Semi-major axis (AU).
        e: [float or ndarray] Eccentricity.
        incl: [float or ndarray] Inclination (radians).

    Return:
        [float or ndarray] U (1 + v_esc^2/v_inf^2)/(a^1.5 sin i |U_x|), 0 where the orbit does not
            cross 1 AU.
    """

    a, e, incl = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (a, e, incl)))

    p = a*(1.0 - e**2)
    u_x2 = 2.0 - 1.0/a - p
    u2 = 3.0 - 1.0/a - 2.0*np.sqrt(np.maximum(p, 0.0))*np.cos(incl)

    rate = np.zeros(a.shape)
    crossing = (u_x2 > 0) & (u2 > 0) & (np.sin(incl) > 0)

    u = np.sqrt(u2[crossing])
    rate[crossing] = u*(1.0 + (ESCAPE_SPEED/(u*SPEED_EARTH))**2) \
        /(a[crossing]**1.5*np.sin(incl[crossing])*np.sqrt(u_x2[crossing]))

    return rate[()]


def populationAtH(deam, h_mag):
    """ The population of the model for a meteoroid of absolute magnitude h_mag: the sum of the H bins
        whose centre is within 0.25 of it, over (a, e, i). """

    keep = np.abs(H_CENTRES - h_mag) <= 0.25 + 1e-9

    return np.sum(deam['population'][..., keep], axis=-1)


def sampleImpactors(deam, h_mag, n=N_SAMPLES, seed=SEED):
    """ Draw the orbits of meteoroids of absolute magnitude h_mag that hit the Earth, by importance
        sampling of the model weighted by impactRate. See the module docstring.

        Within a bin a is drawn uniform, i uniform and e as e_c cosh(u), u uniform, e_c = |1 - 1/a|
        being the smallest e that crosses 1 AU. With the density of the population uniform in a and e
        and going as sin i, and the rate going as 1/(sin i sqrt(e^2 - e_c^2)), the weight that is left
        varies only with U and a. The bins are chosen in proportion to the rest of the rate at the
        centre of their a, so that the weights stay close to 1.

    Arguments:
        deam: [dict] Return of loadDeamModel.
        h_mag: [float] Absolute magnitude of the meteoroid, in the model's scale.

    Keyword arguments:
        n: [int] How many orbits to draw. The 2 to 3 % that fall where their bin does not cross 1 AU
            are dropped.
        seed: [int] Seed of the random generator, so that the same call gives the same orbits.

    Return:
        [dict] Arrays of 'q', 'a' (AU), 'e', 'incl', 'node', 'peri' (rad), 'weight' (mean 1), 'v_inf'
            (km/s); 'h_mag', 'n' (the orbits kept), 'n_eff' (their effective number,
            (sum w)^2/sum w^2), 'seed' and 'model'.
    """

    cache_key = (float(h_mag), int(n), seed)

    if cache_key in deam['impactors']:
        return deam['impactors'][cache_key]

    rng = np.random.default_rng(seed)

    population = populationAtH(deam, h_mag)

    incl_edges = np.radians(INCL_EDGES)
    a_centres = 0.5*(A_EDGES[1:] + A_EDGES[:-1])

    # The density in i going as sin i, an impactor is uniform in i within a bin, which then weighs
    #   di/(cos i1 - cos i2)
    incl_factor = np.diff(incl_edges)/(np.cos(incl_edges[:-1]) - np.cos(incl_edges[1:]))

    # The integral over e of 1/sqrt(e^2 - e_c^2) across each bin, at the centre of each a bin
    e_c = np.abs(1.0 - 1.0/a_centres)[:, None]
    e_lo = np.maximum(E_EDGES[:-1][None, :], e_c)
    e_hi = np.broadcast_to(E_EDGES[1:][None, :], e_lo.shape)

    with np.errstate(divide='ignore', invalid='ignore'):
        e_factor = np.where(e_hi > e_lo, np.arccosh(e_hi/e_c) - np.arccosh(e_lo/e_c), 0.0)

    e_factor = e_factor/np.diff(E_EDGES)[None, :]/a_centres[:, None]**2

    guide = population*e_factor[:, :, None]*incl_factor[None, None, :]
    pmf = guide.ravel()/np.sum(guide)

    index = rng.choice(pmf.size, size=n, p=pmf)
    ia, ie, ii = np.unravel_index(index, population.shape)

    a = rng.uniform(A_EDGES[ia], A_EDGES[ia + 1])
    incl = rng.uniform(incl_edges[ii], incl_edges[ii + 1])

    e_c = np.abs(1.0 - 1.0/a)
    e_lo = np.maximum(E_EDGES[ie], e_c)
    e_hi = E_EDGES[ie + 1]

    # A draw whose a is past the crossing of its e bin, possible when the bin's a centre is not, does
    #   not cross 1 AU and gets no weight
    crossing = e_hi > e_lo

    with np.errstate(divide='ignore', invalid='ignore'):
        u_lo = np.where(crossing, np.arccosh(e_lo/e_c), 0.0)
        u_hi = np.where(crossing, np.arccosh(e_hi/e_c), 0.0)

    e = e_c*np.cosh(rng.uniform(u_lo, u_hi))
    e = np.where(crossing, e, e_hi)

    p = a*(1.0 - e**2)
    u = np.sqrt(np.maximum(3.0 - 1.0/a - 2.0*np.sqrt(p)*np.cos(incl), 0.0))

    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.where(crossing & (u > 0), population[ia, ie, ii]*incl_factor[ii]*(u_hi - u_lo)
            /np.diff(E_EDGES)[ie]/a**2*u*(1.0 + (ESCAPE_SPEED/(u*SPEED_EARTH))**2)/pmf[index], 0.0)

    # The draws that do not cross 1 AU are dropped, so that every orbit kept can hit the Earth
    kept = weight > 0
    a, e, incl, p, u = a[kept], e[kept], incl[kept], p[kept], u[kept]
    weight = weight[kept]/np.mean(weight[kept])
    n_kept = len(a)

    # The angles: the node uniform, and one of the four arguments of perihelion that put a node at
    #   1 AU, r = p/(1 + e cos(nu)) with nu = -peri at the ascending node and pi - peri at the
    #   descending one
    node = rng.uniform(0.0, 2*np.pi, n_kept)
    peri_0 = np.arccos(np.clip((p - 1.0)/e, -1.0, 1.0))
    solution = rng.integers(0, 4, n_kept)
    peri = np.choose(solution, [peri_0, -peri_0, np.pi - peri_0, np.pi + peri_0]) % (2*np.pi)

    impactors = {'q': a*(1.0 - e), 'a': a, 'e': e, 'incl': incl, 'node': node, 'peri': peri,
        'weight': weight, 'v_inf': u*SPEED_EARTH, 'h_mag': float(h_mag), 'n': n_kept,
        'n_eff': float(np.sum(weight)**2/np.sum(weight**2)), 'seed': seed, 'model': deam['model']}

    deam['impactors'][cache_key] = impactors

    return impactors


def closestInCatalogue(impactors, catalogue_orbits, criterion_key, n_global=N_GLOBAL):
    """ For each of the first n_global background orbits, the value of the criterion to the closest
        body of a catalogue.

    Arguments:
        impactors: [dict] Return of sampleImpactors.
        catalogue_orbits: [tuple] Arrays of (q (AU), e, incl, node, peri) of the bodies, the angles in
            radians.
        criterion_key: [str] Key of the criterion on the orbital elements (wmpl.Utils.Dcriteria).

    Keyword arguments:
        n_global: [int] How many background orbits to compare with the catalogue.

    Return:
        (closest, weight): [tuple] The value to the closest body for each orbit, inf if there is none,
            and their weights, of mean 1.
    """

    _, _, func, _ = lookupCriterion(criterion_key)

    n_global = min(int(n_global), impactors['n'])
    closest = np.full(n_global, np.inf)

    for k in range(n_global):

        orbit = tuple(float(impactors[name][k]) for name in ('q', 'e', 'incl', 'node', 'peri'))

        with np.errstate(divide='ignore', invalid='ignore'):
            values = np.asarray(func(orbit, catalogue_orbits), dtype=float)

        values = values[np.isfinite(values)]

        if values.size:
            closest[k] = np.min(values)

    weight = impactors['weight'][:n_global]

    return closest, weight/np.mean(weight)


def chanceAssociation(meteor_orbit, h_mag, candidates, criterion_key, impactors,
    catalogue_orbits=None, n_global=N_GLOBAL, model=DEAM2026):
    """ For each candidate, the probability that a background meteoroid lies at least as close to it
        as the meteoroid does, for the candidate alone and, given the catalogue that was searched,
        corrected for the search. See the module docstring for the test and what it assumes.

    Arguments:
        meteor_orbit: [tuple] (q (AU), e, incl, node, peri) of the meteoroid, the angles in radians.
        h_mag: [float] Absolute magnitude of the meteoroid, in the model's scale.
        candidates: [list] (d_value, orbit) of each candidate: its value of the criterion against the
            meteoroid's orbit and its orbit as (q (AU), e, incl, node, peri), the angles in radians.
        criterion_key: [str] Key of the criterion on the orbital elements (wmpl.Utils.Dcriteria).
        impactors: [dict] Return of sampleImpactors, for this h_mag. It may be None for a meteoroid
            outside the model, which gets no value.

    Keyword arguments:
        catalogue_orbits: [tuple] Arrays of (q, e, incl, node, peri) of every body searched, for
            p_global. Without them p_global is not computed.
        n_global: [int] How many background orbits to compare with the catalogue for p_global.
        model: [dict] The model, DEAM2026, for the region it covers.

    Return:
        [list] For each candidate a dictionary: 'p_value' of the candidate alone, 'fraction' of the
            background at least as close (by weight), 'n_closer' (the background orbits at least as
            close), 'n_background', 'is_limit' (True when none is, so that p is at most the value
            given); 'p_value_global' corrected for the search and its 'sigma', 'n_closer_global',
            'n_global' and 'is_limit_global', likewise, None without the catalogue; and 'reason',
            None if the values were computed or why not, in which case they are None.
    """

    key, label, func, kind = lookupCriterion(criterion_key)

    if kind != 'elements':
        raise ValueError("{:s} is a criterion on the radiant. Only the criteria on the orbital "
            "elements are offered against the background meteoroids.".format(label))

    if key == 'MOID':
        raise ValueError("The MOID costs 0.09 to 0.15 ms a pair, 1.5 to 2.5 minutes a body against 10^6 "
            "background orbits, so it is not offered.")

    fields = ['p_value', 'fraction', 'n_closer', 'is_limit', 'p_value_global', 'sigma',
        'n_closer_global', 'is_limit_global']

    empty = dict((field, None) for field in fields)
    empty.update({'n_background': None, 'n_global': None, 'reason': None})

    inside, reason = isInModelRegion(meteor_orbit[0], meteor_orbit[1], meteor_orbit[2], h_mag, model)

    if not inside:
        return [dict(empty, reason="for the meteoroid, " + reason) for _ in candidates]

    n_background = impactors['n']
    n_used = None if catalogue_orbits is None else min(int(n_global), n_background)
    empty.update({'n_background': n_background, 'n_global': n_used})

    if abs(impactors['h_mag'] - h_mag) > 1e-9:
        raise ValueError("The background orbits were drawn for H = {:.2f}, not {:.2f}.".format(
            impactors['h_mag'], h_mag))

    background = tuple(impactors[name] for name in ('q', 'e', 'incl', 'node', 'peri'))
    weight = impactors['weight']

    closest = None
    if catalogue_orbits is not None:
        closest, weight_global = closestInCatalogue(impactors, catalogue_orbits, key, n_used)
        order = np.argsort(closest)
        closest = closest[order]
        cumulative_global = np.concatenate([[0.0], np.cumsum(weight_global[order])])

    results = []

    for d_value, orbit in candidates:

        if (d_value is None) or (not np.isfinite(d_value)):
            results.append(dict(empty, reason="no value of {:s}".format(label)))
            continue

        with np.errstate(divide='ignore', invalid='ignore'):
            d_background = np.asarray(func(background, tuple(orbit)), dtype=float)

        # A pair the criterion is undefined for is not a closer pair
        closer = np.isfinite(d_background) & (d_background <= d_value)

        n_closer = int(np.count_nonzero(closer))
        k = float(np.sum(weight[closer]))

        result = dict(empty)
        result.update({'p_value': (k + 1.0)/(n_background + 1.0), 'fraction': k/n_background,
            'n_closer': n_closer, 'is_limit': n_closer == 0})

        if closest is not None:

            j = int(np.searchsorted(closest, d_value, side='right'))
            p_global = max((cumulative_global[j] + 1.0)/(n_used + 1.0), result['p_value'])

            result.update({'p_value_global': p_global, 'sigma': float(pValueToSigma(p_global)),
                'n_closer_global': j, 'is_limit_global': j == 0})

        results.append(result)

    return results
