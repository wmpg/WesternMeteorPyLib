""" Methods for choosing the threshold of a D criterion.

A D criterion is only half of a stream search: the threshold below which two orbits count as
associated has to be chosen as well, and a threshold taken from the literature is generally tied
to the criterion, the database size and the sporadic background of the study it came from.

Most of the methods here are independent of which criterion is used, and are passed the criterion
as a callable where they need one. thresholdRandomPairing is the exception: its thresholds were
measured per criterion and per population, and it is selected by name.
"""

from __future__ import print_function, division, absolute_import

import numpy as np


def thresholdDr(n_orbits):
    """ Calculate the Southworth & Hawkins threshold scaled to the size of the database.

        The threshold shrinks as the database grows, because the number of pairs available to
        associate by chance grows with it. The reference value of 0.20 applies to a sample of 360
        orbits; the formula was calibrated on 359 Super-Schmidt photographic meteors. An equivalent
        form is 0.8712*N**(-1/4).

        This threshold is far too permissive for pair searches. Jopek & Bronikowska (2017) measured
        the probability of a coincidental pair at these thresholds as 1.0 for bolide samples of 200
        to 800 orbits, i.e. two "similar" orbits are certain to be found whether or not any are
        related. Use thresholdRandomPairing for a threshold tied to a stated probability.

        Reference: Southworth & Hawkins (1963), Smithson. Contrib. Astrophys. 7, 261; Lindblad
        (1971), Smithson. Contrib. Astrophys. 12, 1, eq. 1 of Jopek & Bronikowska (2017).

    Arguments:
        n_orbits: [int] number of orbits in the database

    Return:
        [float] threshold value
    """

    n_orbits = np.asarray(n_orbits, dtype=np.float64)

    return 0.20*(360.0/n_orbits)**0.25


# Coefficient and exponent of the fitted threshold D_c = A*N**(-b) corresponding to a probability of
#   0.01 of a coincidental pair, from Jopek & Bronikowska (2017). The bolide entries are their eqs
#   14 to 16 and the rest are their table 9, eqs 17 to 25
RANDOM_PAIRING_COEFFS = {
    ('D_SH', 'bolides'): (0.3186, 0.431),
    ('D_H',  'bolides'): (0.3143, 0.438),
    ('D_D',  'bolides'): (0.1240, 0.423),
    ('D_SH', 'NEAs'):    (0.2558, 0.408),
    ('D_H',  'NEAs'):    (0.2193, 0.405),
    ('D_D',  'NEAs'):    (0.1049, 0.408),
    ('D_SH', 'radar'):   (0.4007, 0.450),
    ('D_H',  'radar'):   (0.3768, 0.447),
    ('D_D',  'radar'):   (0.1543, 0.436),
    ('D_SH', 'video'):   (0.5837, 0.487),
    ('D_H',  'video'):   (0.4808, 0.464),
    ('D_D',  'video'):   (0.1724, 0.446),
    }

# Probability of a coincidental pair that the coefficients above correspond to
RANDOM_PAIRING_PROBABILITY = 0.01


def thresholdRandomPairing(n_orbits, d_criterion='D_SH', population='bolides'):
    """ Calculate the threshold at which the probability of a coincidental pair is 0.01.

        The threshold depends on the criterion and on the population as well as on the sample size.
        Both dependencies are substantial: at a fixed sample size the D_D thresholds are about 2.4
        times smaller than the D_SH ones, and the video thresholds about 1.15 times larger than the
        NEA ones, so a threshold is not transferable between criteria or between datasets.

        The thresholds were obtained by searching synthetic samples that reproduce the orbital
        distributions of the observed ones, including the Earth-crossing condition, which matters:
        drawing the elements uniformly instead inflates the threshold by about a factor of two.

        Reference: Jopek & Bronikowska (2017), P&SS 143, 43, doi:10.1016/j.pss.2016.12.004, eqs 14
        to 16 and table 9.

        The fits hold over 200 to 1000 orbits for the bolide coefficients and 1000 to 16000 for the
        others, which the authors expect to extend to about 50000. Nothing is clamped outside those
        ranges.

    Arguments:
        n_orbits: [int] number of orbits in the database

    Keyword arguments:
        d_criterion: [str] criterion the threshold is for, one of 'D_SH', 'D_H' or 'D_D'.
            Default 'D_SH'.
        population: [str] population the database is drawn from, one of 'bolides', 'NEAs', 'radar'
            or 'video'. Default 'bolides'.

    Return:
        [float] threshold value
    """

    key = (d_criterion, population)

    if key not in RANDOM_PAIRING_COEFFS:
        raise ValueError("No published threshold for criterion {!r} and population {!r}. "
            "Available combinations: {!s}.".format(d_criterion, population,
                sorted(RANDOM_PAIRING_COEFFS)))

    coeff, exponent = RANDOM_PAIRING_COEFFS[key]

    n_orbits = np.asarray(n_orbits, dtype=np.float64)

    return coeff*n_orbits**(-exponent)


def thresholdBreakPoint(d_values, n_bins=100, d_max=None):
    """ Locate the break point of a cumulative distribution of D values.

        The number of orbits within a distance D of a stream's mean orbit rises steeply while the
        stream itself is being counted, then flattens into the much shallower rise of the sporadic
        background. The D at which the slope drops separates the two populations and is used in
        place of a fixed threshold.

        The break is taken to be the bin of strongest negative curvature of the cumulative
        distribution. Note that the break is only meaningful when the stream is a large enough
        fraction of the sample to produce a visible change of slope; a sample that is almost all
        sporadic has no break to find, and this function will still return its best candidate.

        Reference: Neslusan, Svoren & Porubcan (1995), EM&P 68, 427; Neslusan, Hajdukova & Jakubik
        (2013), A&A 560, A47.

    Arguments:
        d_values: [ndarray] D values of the orbits with respect to the mean orbit

    Keyword arguments:
        n_bins: [int] number of bins of the cumulative distribution. Default 100.
        d_max: [float] largest D value to consider. Defaults to the largest value given.

    Return:
        [float] D value of the break point
    """

    d_values = np.asarray(d_values, dtype=np.float64)
    d_values = d_values[np.isfinite(d_values)]

    if d_values.size < 3:
        raise ValueError("At least 3 finite D values are needed to locate a break point.")

    if d_max is None:
        d_max = np.max(d_values)

    edges = np.linspace(0.0, d_max, n_bins + 1)

    # Cumulative count of orbits within each D
    cumulative = np.searchsorted(np.sort(d_values), edges, side='right').astype(np.float64)

    # The break point is where the cumulative count bends over most sharply
    curvature = np.diff(cumulative, n=2)

    return edges[np.argmin(curvature) + 1]


def thresholdReliability(orbit_params, d_func, reliability=0.99, n_trials=100, random_state=None):
    """ Calculate the threshold at which a chance association is unlikely at a stated reliability.

        The threshold is not read off the real sample but derived from synthetic samples that share
        the marginal distribution of each variable with the real one while carrying none of its
        correlations, which is what shuffling each variable independently produces. The threshold
        is the largest value at which, in the requested fraction of the synthetic samples, no pair
        of orbits is associated. A group found below it in the real sample is therefore not
        reproducible by chance at that reliability.

        Only pairs are considered, so this is the threshold for a group of two. Larger minimum
        group sizes need a clustering algorithm, which is a separate concern from the threshold.

        Reference: Jopek, Valsecchi & Froeschle (1999), MNRAS 304, 751.

    Arguments:
        orbit_params: [ndarray] n_orbits by n_params array of the parameters the criterion takes,
            one row per orbit, in the order in which d_func takes them for a single orbit
        d_func: [function] criterion, called as d_func(*row1, *row2) and returning the distance

    Keyword arguments:
        reliability: [float] fraction of the synthetic samples in which no pair may be associated.
            Default 0.99.
        n_trials: [int] number of synthetic samples. Default 100.
        random_state: [int] seed for the shuffling, for reproducibility. Default None.

    Return:
        [float] threshold value
    """

    orbit_params = np.atleast_2d(np.asarray(orbit_params, dtype=np.float64))

    n_orbits, n_params = orbit_params.shape

    if n_orbits < 2:
        raise ValueError("At least 2 orbits are needed to derive a threshold.")

    if not 0.0 < reliability < 1.0:
        raise ValueError("The reliability has to lie strictly between 0 and 1.")

    rng = np.random.RandomState(random_state)

    min_distances = np.zeros(n_trials)

    for trial in range(n_trials):

        # Shuffling each parameter independently keeps its marginal distribution and destroys the
        #   correlations between parameters that a real stream would show
        synthetic = np.empty_like(orbit_params)
        for param in range(n_params):
            synthetic[:, param] = rng.permutation(orbit_params[:, param])

        smallest = np.inf
        for a in range(n_orbits - 1):
            for b in range(a + 1, n_orbits):
                d_value = d_func(*np.concatenate((synthetic[a], synthetic[b])))
                smallest = min(smallest, float(d_value))

        min_distances[trial] = smallest

    # No pair is associated in a trial when the threshold is at or below that trial's smallest
    #   distance, so the threshold that achieves the reliability is the corresponding lower
    #   quantile of the smallest distances
    return float(np.quantile(min_distances, 1.0 - reliability))
