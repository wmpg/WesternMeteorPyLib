""" The orbits of the meteoroids that hit the Earth, from the near-Earth object model of Deam et al.
(2026), and the probability that a meteoroid lies as close to a parent body candidate by chance.

The model of Deam et al. (2026) describes the near-Earth objects at meteoroid sizes, H = 34.5 to 41.5,
fitted to 1202 sporadic meteoroids observed by the Global Fireball Observatory, on the residence-time
maps of the model of Nesvorny et al. (2023). The authors convert the masses of the fireballs to H with
the density (3500 kg/m^3) and albedo (0.147) of an ordinary chondrite (massToH): 1 kg is H = 38.14, and
the model spans 150 kg to 10 g, 0.44 m to 1.7 cm in diameter. It is binned in (a, e, i, H): 42 x 25 x 22
bins over a < 4.2 AU, e < 1, i < 88 deg, and 28 bins of 0.25 over 34.5 < H < 41.5. The authors fitted it
on bins twice as wide in a and e, over e > 0.04 and i < 60 deg (their Table 1), and their sampler draws
it, as here, on the finer bins. At each H it is a sum over six source regions (the nu6 and the 3:1, 5:2
and 2:1 resonances, the inner belt and the Jupiter-family comets) of the residence-time map of the
source, the time its objects spend in each bin, weighted by the strength of the source. The strengths
are linear in H between H = 34.5, 36.75, 39.25 and 41.5. The maps are those for a collisional
lifetime of 3 Myr up to an H that is fitted, and of 1 Myr past it: H = 38.5 in 4122 of the 4125
posterior samples. The model gives the relative orbital distribution at each H, not the numbers of
objects.

The strengths were fitted with each map normalised over the fitted region, so that the strength of a
source is its share of that region. Here too each map is normalised over the region, and the
population of each posterior sample at each H then over all the bins. The authors' sampler normalises
each map over all its bins instead, which gives a source with more of its time outside the region less
weight than it was fitted with: 7 to 18 % of the time of the 3:1, 5:2 and 2:1 resonances and of the
Jupiter-family comets in their 3 Myr maps is at i > 60 deg, against 0.3 to 0.4 % for nu6 and the inner
belt. Measured on the two orbits below, that would raise the fraction of the background near them by
1 to 3 %.

Two files are needed, 3.5 MB together, and are not distributed with wmpl. They are downloaded on
first use from the authors' Zenodo record, checked against their SHA-256 and kept in wmpl/share (or
~/.cache/wmpl if that cannot be written), as wmpl.Utils.DebiasedNEOPopulation does: the posterior
samples of the source strengths, and the residence-time maps. Deam & Nesvorny publish them under CC BY
4.0: cite the paper and the data.

    Deam, S. E., Devillepoix, H. A. R., Nesvorny, D., et al. (2026), A Near-Earth Object Model
    Calibrated to Earth Impactors, AJ 171, 126, doi:10.3847/1538-3881/ae2fc5
    Deam, S. E. & Nesvorny, D. (2025), Sampling and querying code for the Near-Earth Object Model
    Calibrated to Earth Impactors, Zenodo, doi:10.5281/zenodo.17809512

The background. The model gives where the objects are, not which of them hit the Earth, and a
meteoroid that was observed is one that did: 34.5 to 42 % of the model's population at H = 35 to 41
does not even cross the Earth's orbit. To fit the model, Deam et al. went from the fireballs to the
population by dividing each fireball by its probability of impact, from 100 000 years of backward
integration of its clones with the method of Fuentes-Munoz et al. (2023, AJ 166, 10), and by the
gravitational focusing (their Appendix B). Going back, each orbit of the model is multiplied here by its
rate of impact on the Earth. The model has no angles to integrate, so the rate is that of Opik (1951),
which averages over them, for a target on a circular orbit of 1 AU, with the same focusing:

    P ~ U (1 + v_esc^2/v_inf^2) / (a^1.5 sin i |U_x|)
        U^2 = 3 - 1/a - 2 sqrt(a (1 - e^2)) cos i       the encounter speed, in units of the Earth's
        U_x^2 = 2 - 1/a - a (1 - e^2)                  its radial part, zero where the orbit only
                                                       touches 1 AU
        v_inf = U v_E                                  v_E = 29.78 km/s, v_esc = 11.10 km/s at 100 km

and zero for an orbit that does not cross 1 AU. This changes the background a great deal. Two Apollo
orbits are used for the figures here, both with the ascending node at 0 and a node at 1 AU: a
low-inclination one, q = 0.99 AU, e = 0.4, i = 0.5 deg (argument of perihelion 15.2 deg), and one with
q = 0.8 AU, e = 0.5, i = 10 deg (66.4 deg). Measured with D_SH at H = 38, the fraction of the impactors
within 0.1 of them is 20 and 4.2 times that of the population, whose angles are uniform.

Within a bin a and e are taken uniform, as the authors' sampler takes them, and the orbital pole
uniform on the sphere, so that the density in i goes as sin i. The authors' sampler takes i uniform
instead, which for the population is a fine difference but not for the impacts: with i uniform down to
0, Opik's 1/sin i makes the impacts from the lowest bin depend on where 1/sin i is cut off. Cut off at
the Earth's radius, 60 to 68 % of the impacts at H = 35 to 41 come from i < 4 deg, 6 to 8 % of the
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

Resolving a small p. N = 10^6 background orbits resolve p only down to 1/(N + 1), but the bodies that
matter after the correction below have p of 1e-5 and less. When fewer than 100 of the N are at least as
close, p is computed again from orbits drawn only in the bins that can hold an orbit that close to the
candidate, in batches of N until 100 are as close or ten batches are drawn. The bins are those that
overlap the box the criterion puts around the candidate: |dq| <= D, |de| <= D and a mutual inclination
of at most 2 arcsin(D/2) for D_SH; |dq| <= D (q1 + q2) instead for D_H; and |dq| <= D (q1 + q2),
|de| <= D (e1 + e2) and a mutual inclination of at most pi D for D_D. With R the share of the impacts
in those bins,

    p = R (k_R + 1)/(N_R + 1)

the meteoroid being one more draw in the box, where it lies, and k_R the weight of the N_R orbits drawn
there that are at least as close. For a Geminid of 50 g (q = 0.140 AU, e = 0.896, i = 23.6 deg), whose
closest body is Phaethon at D_SH = 0.046, none of the N orbits is as close; R = 0.0025, 82 of 9.7 x 10^6
orbits drawn there are, and p = 2.2e-8. Where the N orbits have hundreds as close, eight seeds of each
give means 0.987 to 1.001 times apart, within the 4 to 9 % error of the N orbits. The other criteria
keep the resolution of the N orbits.

The global p-value. A body found by the search, as the closest of a catalogue of M bodies, tens of
thousands, needs a correction: some body is close to almost any orbit. Two corrections are computed,
each valid and each powerful where the other is not:

    p_bonferroni = min(1, M p)
                           no body of the catalogue is as unusual for a background meteoroid, at its
                           own value of the criterion, more often than this (Bonferroni, which holds
                           however the bodies are related and however the candidate was chosen). It
                           finds a body in a part of orbit space the background rarely reaches.
    p_closest = (k_global + 1)/(K + 1)
                           K background meteoroids being compared with every body of the catalogue,
                           and k_global the weight of those whose closest body is at least as close as
                           the candidate is to the meteoroid: how often a background meteoroid has some
                           body this close. It finds a body very close to a meteoroid among the many
                           near-Earth asteroids at low inclination, where M p is large. The candidate
                           is one of the bodies, so it is at least p; when the smaller K gives less, by
                           sampling, it is taken as p.
    p_global = min(1, 2 min(p_bonferroni, p_closest))
                           the smaller, doubled because two were tried: the value to quote.

Measured with the comets of the JPL and the near-Earth asteroids of the MPC, 46 501 bodies: for the
Geminid above, Phaethon has p_closest = 0.32, as a third of the background meteoroids have some body
that close, but p_bonferroni = 1.0e-3, so that p_global = 2.0e-3, 3.1 sigma (with D_D, 3.5e-5, 4.1
sigma). For Winchcombe (13 kg), whose closest body, 2016 EF195, is at D_SH = 0.025, p_bonferroni = 0.39
but p_closest = 0.035, and p_global = 0.069. Significance is given as the two-sided Gaussian equivalent
of p_global, as in wmpl.Utils.DebiasedNEOPopulation.

What the test assumes:
    - The meteoroid's H selects the model: the bins within H +- 0.25, as the authors' sampler takes
      them, so H can be given between 34.75 and 41.25, 108 kg to 14 g. The source strengths change with
      H, and with them the fraction of the background within D_SH = 0.1 of the two Apollo orbits above:
      over H = 35 to 41 it is 0.86 to 1.10 and 0.57 to 1.06 times its value at H = 38.
    - H is that of the model. Convert a mass with massToH and its defaults whatever the meteoroid's
      type: every fireball the model was fitted to was converted so.
    - The posterior is averaged: the population is the mean of the 4125 posterior samples (3566 of them
      distinct), and the empty bins are filled with 1e-8 before each map is normalised, as in the
      authors' sampler. Their sampler draws from the samples of highest likelihood instead, which with
      its recommended five are the best fit five times over. The best fit alone puts 0.75 to 1.03 times
      the background of the posterior mean near the two orbits over H = 35 to 41, the least at H = 35.
    - The rate of impact is Opik's, which takes the angles of the orbit as uniform over its secular
      cycle. For the fireballs, whose positions they know, Deam et al. preferred integrations, so the
      weighting here is not the exact inverse of their debiasing.
    - The Earth's orbit is taken circular. An orbit with q up to 1.017 AU, or Q down to 0.983 AU, can
      hit the Earth but has no counterpart in the background, so a meteoroid there gets a p-value
      biased low.
    - The model covers a < 4.2 AU and i < 88 deg, prograde. A meteoroid outside it gets no p-value.
      Past i = 60 deg and below e = 0.04 it is the maps' own extension of the fitted region, not fitted
      to any fireball.
    - Only the criteria on the orbital elements are offered, and not the MOID, which costs 0.09 to
      0.15 ms a pair, 1.5 to 2.5 minutes a body against 10^6 background orbits.
    - Only the bodies the search ranked first by the criterion are tested. A body further down, in a
      sparse part of orbit space, could have a smaller p and is not found as such.
    - The meteoroid's orbit is taken as exact. Its uncertainty is not propagated.

Resolution. N background orbits give p no lower than 1/(N + 1), or R/(10 N + 1) when it is computed
again near the candidate, and K give p_closest no lower than 1/(K + 1). When no background orbit is as
close, the value is an upper bound.
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

_REFERENCE = ('Deam, S. E., Devillepoix, H. A. R., Nesvorny, D., et al. (2026), A Near-Earth Object '
    'Model Calibrated to Earth Impactors, AJ 171, 126, doi:10.3847/1538-3881/ae2fc5')

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

# The bins of the region the model was fitted on, e > 0.04 and i < 60 deg (Deam et al. 2026, Table 1)
FIT_REGION = (slice(None), slice(1, None), slice(0, 15))

# The density (kg/m^3) and geometric albedo of an ordinary chondrite, with which Deam et al. (2026)
#   convert the masses of the fireballs to H
DENSITY = 3500.0
ALBEDO = 0.147

# The Earth's orbital speed on a circle of 1 AU, and the escape speed at 100 km, where a meteoroid
#   becomes a meteor [km/s]
SPEED_EARTH = GAUSS_K*149597870.7/86400.0
ESCAPE_SPEED = np.sqrt(2*6.67408e-11*5.9722e24/6471.0e3)/1000.0

# Background orbits for the p of each body, and the part of them compared with the whole catalogue
#   for p_closest
N_SAMPLES = 1000000
N_GLOBAL = 2000
SEED = 1

# Below this many background orbits at least as close, p is computed again from orbits drawn near the
#   candidate, in batches as large as the background until as many are as close, at most this many
N_REFINE = 100
N_REFINE_BATCHES = 10

# The model, once loaded
_model_cache = {}


def findDeamModel(model=DEAM2026):
    """ The paths of the files of the model if they have been downloaded, None for each one that has
        not. """

    return [findDebiasedNEOPopulation(f) for f in model['files']]


def massToH(mass, density=DENSITY, albedo=ALBEDO):
    """ The absolute magnitude of a meteoroid in the scale of the model, from its mass, as Deam et al.
        (2026) convert the masses of the fireballs: a sphere of the density and albedo of an ordinary
        chondrite, with D = 1329 km 10^(-H/5)/sqrt(p_V). 1 kg is H = 38.14.

    Arguments:
        mass: [float or ndarray] Mass (kg).

    Keyword arguments:
        density: [float] Bulk density (kg/m^3).
        albedo: [float] Geometric albedo.

    Return:
        [float or ndarray] H.
    """

    diameter_km = (6.0*np.asarray(mass, dtype=float)/(np.pi*density))**(1.0/3)/1000.0

    return 5.0*np.log10(1329.0/(diameter_km*np.sqrt(albedo)))


def parseMeteoroidSize(value):
    """ The meteoroid's H from a command-line value: H itself, or a mass ending in kg or g, converted
        with massToH.

    Arguments:
        value: [str] For example '38.0', '0.6kg' or '600g'.

    Return:
        (h_mag, mass): [tuple] H, and the mass in kg if one was given, else None.
    """

    text = str(value).strip().lower()

    for unit, factor in (('kg', 1.0), ('g', 1e-3)):

        if text.endswith(unit):

            mass = float(text[:-len(unit)])*factor

            if not (mass > 0):
                raise ValueError("The mass has to be positive, not {:s}.".format(str(value)))

            return float(massToH(mass)), mass

    return float(text), None


def _readMaps(tar_path):
    """ The residence-time maps of the sources, 42 x 25 x 22 each, read from the archive without
        unpacking it, and each normalised over the region the model was fitted on, FIT_REGION. """

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

            # The strengths were fitted with each map normalised over the region, so that a source's
            #   strength is its share of the region; normalised over all its bins, as the authors'
            #   sampler does, a source with more time outside it would weigh less than it was fitted to
            counts[counts == 0.0] = EMPTY_BIN
            maps[name[:-len('.csv')]] = counts/np.sum(counts[FIT_REGION])

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
    #   between the two lifetimes by each sample's crossover
    strength_h = np.empty(strengths.shape[:2] + (len(H_CENTRES),))
    for k in range(strengths.shape[0]):
        for s in range(len(SOURCES)):
            strength_h[k, s] = np.interp(H_CENTRES, H_NODES, strengths[k, s])

    long_lived = np.arange(len(H_CENTRES))[None, :] < crossover[:, None]

    # Each sample's population at each H sums to 1 over all the bins, the strengths being scaled by
    #   the sum of the maps, which are normalised over the fitted region only. The model is then linear
    #   in the scaled strengths, so averaging them averages the model
    total_3my = np.array([np.sum(maps[MAPS_3MY[source]]) for source in SOURCES])
    total_1my = np.array([np.sum(maps[MAPS_1MY[source]]) for source in SOURCES])
    total = np.sum(strength_h*np.where(long_lived[:, None, :], total_3my[None, :, None],
        total_1my[None, :, None]), axis=1)
    strength_h = strength_h/total[:, None, :]

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


def sampleImpactors(deam, h_mag, n=N_SAMPLES, seed=SEED, region=None):
    """ Draw the orbits of meteoroids of absolute magnitude h_mag that hit the Earth, by importance
        sampling of the model weighted by impactRate, from all the bins or from a region of them. See
        the module docstring.

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
        seed: [int or tuple] Seed of the random generator, so that the same call gives the same
            orbits.
        region: [ndarray] Boolean mask over the (a, e, i) bins, to draw only in those bins. The orbits
            drawn this way are not kept for later calls.

    Return:
        [dict] Arrays of 'q', 'a' (AU), 'e', 'incl', 'node', 'peri' (rad), 'weight' (mean 1), 'v_inf'
            (km/s); 'h_mag', 'n' (the orbits kept), 'n_eff' (their effective number,
            (sum w)^2/sum w^2), 'mass' (the rate of impact of the bins drawn from, to a constant
            factor, so that the ratio of two calls is the share of the impacts in a region), 'seed'
            and 'model'.
    """

    cache_key = (float(h_mag), int(n), seed)

    if (region is None) and (cache_key in deam['impactors']):
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

    # Within a region the bins are drawn from the same guide, restricted to it; the weights are those
    #   of the whole, and the share of the guide in the region scales their mean into its mass
    draw_pmf = pmf
    region_share = 1.0

    if region is not None:
        draw_pmf = pmf*np.asarray(region, dtype=bool).ravel()
        region_share = np.sum(draw_pmf)

    if region_share <= 0:
        index = np.zeros(0, dtype=int)

    else:
        index = rng.choice(pmf.size, size=n, p=draw_pmf/region_share)
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

    mass = region_share*np.sum(weight)/n

    # The draws that do not cross 1 AU are dropped, so that every orbit kept can hit the Earth
    kept = weight > 0
    a, e, incl, p, u = a[kept], e[kept], incl[kept], p[kept], u[kept]
    n_kept = len(a)
    weight = weight[kept]/np.mean(weight[kept]) if n_kept else weight[kept]

    # The angles: the node uniform, and one of the four arguments of perihelion that put a node at
    #   1 AU, r = p/(1 + e cos(nu)) with nu = -peri at the ascending node and pi - peri at the
    #   descending one
    node = rng.uniform(0.0, 2*np.pi, n_kept)
    peri_0 = np.arccos(np.clip((p - 1.0)/e, -1.0, 1.0))
    solution = rng.integers(0, 4, n_kept)
    peri = np.choose(solution, [peri_0, -peri_0, np.pi - peri_0, np.pi + peri_0]) % (2*np.pi)

    impactors = {'q': a*(1.0 - e), 'a': a, 'e': e, 'incl': incl, 'node': node, 'peri': peri,
        'weight': weight, 'v_inf': u*SPEED_EARTH, 'h_mag': float(h_mag), 'n': n_kept,
        'n_eff': float(np.sum(weight)**2/np.sum(weight**2)) if n_kept else 0.0, 'mass': float(mass),
        'seed': seed, 'model': deam['model']}

    if region is None:
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


def criterionRegion(orbit, criterion_key, d_value):
    """ The bins of the model that can hold an orbit within d_value of the given one by the criterion:
        those that overlap the box in q, e and i the criterion puts around the orbit (see the module
        docstring). Only D_SH, D_H and D_D have one.

    Arguments:
        orbit: [tuple] (q (AU), e, incl, ...) of the orbit, the inclination in radians.
        criterion_key: [str] Key of the criterion (wmpl.Utils.Dcriteria).
        d_value: [float] The value of the criterion.

    Return:
        [ndarray] Boolean mask over the (a, e, i) bins, or None for another criterion.
    """

    q, e, incl = float(orbit[0]), float(orbit[1]), float(orbit[2])
    d_value = float(d_value)
    key = criterion_key.upper()

    def relative(x):
        """ The range of x' with |x' - x| <= d_value (x' + x). """
        return x*(1.0 - d_value)/(1.0 + d_value), (x*(1.0 + d_value)/(1.0 - d_value) if d_value < 1
            else np.inf)

    if key in ('DSH', 'DH'):

        e_lo, e_hi = e - d_value, e + d_value
        q_lo, q_hi = (q - d_value, q + d_value) if key == 'DSH' else relative(q)

        # The mutual inclination is at least the difference of the inclinations
        d_incl = 2*np.arcsin(min(d_value/2.0, 1.0))

    elif key == 'DD':

        (e_lo, e_hi), (q_lo, q_hi) = relative(e), relative(q)
        d_incl = np.pi*d_value

    else:
        return None

    incl_edges = np.radians(INCL_EDGES)

    # A bin of (a, e) holds q from a_lo (1 - e_hi) to a_hi (1 - e_lo)
    q_min = A_EDGES[:-1][:, None]*(1.0 - E_EDGES[1:][None, :])
    q_max = A_EDGES[1:][:, None]*(1.0 - E_EDGES[:-1][None, :])

    in_ae = (q_max >= q_lo) & (q_min <= q_hi) & (E_EDGES[1:] >= e_lo)[None, :] \
        & (E_EDGES[:-1] <= e_hi)[None, :]
    in_incl = (incl_edges[1:] >= incl - d_incl) & (incl_edges[:-1] <= incl + d_incl)

    return in_ae[:, :, None] & in_incl[None, None, :]


def chanceAssociation(meteor_orbit, h_mag, candidates, criterion_key, impactors,
    catalogue_orbits=None, n_global=N_GLOBAL, model=DEAM2026, deam=None, n_refine=N_REFINE):
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
        n_global: [int] How many background orbits to compare with the catalogue for p_closest.
        model: [dict] The model, DEAM2026, for the region it covers.
        deam: [dict] Return of loadDeamModel, to compute p again near a candidate that fewer than
            n_refine background orbits are as close to. Without it p keeps the resolution of the
            background orbits given.
        n_refine: [int] See deam.

    Return:
        [list] For each candidate a dictionary: 'p_value' of the candidate alone, 'fraction' of the
            background at least as close (by weight), 'n_closer' (the background orbits at least as
            close), 'n_background', 'is_limit' (True when none is, so that p is at most the value
            given); when p was computed again near the candidate, 'region_share' (R), 'n_local'
            (N_R) and 'n_closer_local', else None; 'p_value_bonferroni', 'p_value_closest',
            'n_closer_global' and 'n_global', 'n_bodies' (M), and 'p_value_global' with its 'sigma'
            and 'is_limit_global', None without the catalogue; and 'reason', None if the values were
            computed or why not, in which case they are None.
    """

    key, label, func, kind = lookupCriterion(criterion_key)

    if kind != 'elements':
        raise ValueError("{:s} is a criterion on the radiant. Only the criteria on the orbital "
            "elements are offered against the background meteoroids.".format(label))

    if key == 'MOID':
        raise ValueError("The MOID costs 0.09 to 0.15 ms a pair, 1.5 to 2.5 minutes a body against 10^6 "
            "background orbits, so it is not offered.")

    fields = ['p_value', 'fraction', 'n_closer', 'is_limit', 'region_share', 'n_local',
        'n_closer_local', 'p_value_bonferroni', 'p_value_closest', 'n_closer_global', 'p_value_global',
        'sigma', 'is_limit_global']

    empty = dict((field, None) for field in fields)
    empty.update({'n_background': None, 'n_global': None, 'n_bodies': None, 'reason': None})

    inside, reason = isInModelRegion(meteor_orbit[0], meteor_orbit[1], meteor_orbit[2], h_mag, model)

    if not inside:
        return [dict(empty, reason="for the meteoroid, " + reason) for _ in candidates]

    n_background = impactors['n']
    n_used = None if catalogue_orbits is None else min(int(n_global), n_background)
    n_bodies = None if catalogue_orbits is None else int(np.size(catalogue_orbits[0]))
    empty.update({'n_background': n_background, 'n_global': n_used, 'n_bodies': n_bodies})

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

    for c, (d_value, orbit) in enumerate(candidates):

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

        # Few background orbits as close: p again from orbits drawn in the bins that can be as close,
        #   R (k_R + 1)/(N_R + 1), the meteoroid being one more draw there. Batches of as many orbits as
        #   the background are drawn until n_refine of them are as close, or N_REFINE_BATCHES were
        region = None
        if (deam is not None) and (n_closer < n_refine) and impactors.get('mass'):
            region = criterionRegion(orbit, key, d_value)

        if region is not None:

            masses = []
            k_local, n_local, n_closer_local = 0.0, 0, 0
            seed = SEED if impactors['seed'] is None else impactors['seed']

            for batch in range(N_REFINE_BATCHES):

                local = sampleImpactors(deam, h_mag, n=n_background, seed=(seed, c + 1, batch),
                    region=region)

                if local['n'] == 0:
                    break

                with np.errstate(divide='ignore', invalid='ignore'):
                    d_local = np.asarray(func(tuple(local[name] for name in ('q', 'e', 'incl', 'node',
                        'peri')), tuple(orbit)), dtype=float)

                closer_local = np.isfinite(d_local) & (d_local <= d_value)

                masses.append(local['mass'])
                k_local += float(np.sum(local['weight'][closer_local]))
                n_local += local['n']
                n_closer_local += int(np.count_nonzero(closer_local))

                if n_closer_local >= n_refine:
                    break

            if n_local > 0:

                share = float(np.mean(masses))/impactors['mass']

                result.update({'p_value': share*(k_local + 1.0)/(n_local + 1.0),
                    'fraction': share*k_local/n_local, 'is_limit': n_closer_local == 0,
                    'region_share': share, 'n_local': n_local, 'n_closer_local': n_closer_local})

        if closest is not None:

            p_value = result['p_value']
            p_bonferroni = min(1.0, n_bodies*p_value)

            j = int(np.searchsorted(closest, d_value, side='right'))
            p_closest = max((cumulative_global[j] + 1.0)/(n_used + 1.0), p_value)

            # The smaller of the two, doubled for having tried two, and a limit if that one is
            if p_bonferroni <= p_closest:
                p_global, is_limit_global = min(1.0, 2*p_bonferroni), result['is_limit']
            else:
                p_global, is_limit_global = min(1.0, 2*p_closest), j == 0

            result.update({'p_value_bonferroni': p_bonferroni, 'p_value_closest': p_closest,
                'n_closer_global': j, 'p_value_global': p_global,
                'sigma': float(pValueToSigma(p_global)), 'is_limit_global': is_limit_global})

        results.append(result)

    return results
