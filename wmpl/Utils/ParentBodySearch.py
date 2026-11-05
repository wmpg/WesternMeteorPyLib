""" Functions for comparing the given orbit with orbits of comets and asteroids.

The D criterion keys and labels this module takes and prints are not its own: they are
shared with wmpl.Utils.Dcriteria, which defines them in ELEMENT_CRITERIA and
GEOCENTRIC_CRITERIA and offers them through -c. A key means the same criterion in both, and
editing one module changes the other.

Several criteria can be given at once, with -d on the command line or a list for d_crit. The
bodies are ranked by the first one and the others are printed beside it. With more than one
criterion a bar plot of the matches is saved too.

MOID is one of the keys: the minimum orbit intersection distance between the given orbit and each
body's, in AU (wmpl.Utils.MOID), parabolic and hyperbolic comets included. It says how close the two
orbits come in space, not how similar they are, so two unrelated orbits can cross; it is best given
with a D criterion, -d DSH MOID to rank by similarity and check the distance, or -d MOID DSH the other
way round. It is evaluated over the whole catalogue in one call (~2 s for the ~46 000 bodies).

--chance_neo (chanceNEO) gives, for each asteroid, the probability that a NEO at least as large lies at
least as close to the orbit by chance, the p-value of the match, for the body alone and corrected for
having searched bodies of every size, against the debiased NEO population of Granvik et al. (2018),
which is downloaded on first use, and ranks the whole catalogue by it (rankByChance). See
wmpl.Utils.DebiasedNEOPopulation.
"""

# The MIT License

# Copyright (c) 2017 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import sys
import csv
import math
import json
import datetime
import numpy as np

from wmpl.Config import config
from wmpl.Utils.Dcriteria import (GEOCENTRIC_CRITERIA_KEYS, HELIOCENTRIC_CRITERIA_KEYS,
    ALL_CRITERIA_KEYS, DERIVED_FROM_ELEMENTS, lookupCriterion, radiantCriterionMin,
    calcHeliocentricFromGeocentric, RADIANT_MAX_DISTANCE, VECTORISED_CRITERIA, CRITERIA_DECIMALS)
from wmpl.Utils.OrbitConstants import GAUSS_K
from wmpl.Utils.Pickling import loadPickle


# Reference Julian date of J2000.0
JD_J2000 = 2451545.0


def _calendarToJD(year, month, day):
    """ Julian date of a calendar date with a fractional day, or None if it cannot be represented. """

    try:
        whole = int(day)
        date = datetime.datetime(int(year), int(month), whole) + datetime.timedelta(days=day - whole)

    except (ValueError, OverflowError):
        return None

    return JD_J2000 + (date - datetime.datetime(2000, 1, 1, 12)).total_seconds()/86400.0


def _jdToString(jd):
    """ Calendar date of a Julian date for printing, or '-' if there is none. """

    if jd is None:
        return '-'

    try:
        date = datetime.datetime(2000, 1, 1, 12) + datetime.timedelta(days=jd - JD_J2000)

    except (ValueError, OverflowError):
        return '-'

    return date.strftime('%Y-%m-%d')


def _meanAnomalyFromPerihelionTime(q, e, epoch_jd, tp_jd):
    """ Mean anomaly (deg) at the epoch of an elliptic orbit given its time of perihelion. """

    if (e >= 1.0) or (e < 0) or (epoch_jd is None) or (tp_jd is None):
        return None

    a = q/(1.0 - e)
    n = GAUSS_K/a**1.5
    m = n*(epoch_jd - tp_jd)

    # Wrap to (-180, 180] deg
    return float(np.degrees((m + np.pi) % (2*np.pi) - np.pi))


# Every body of the catalogues is [name, q, e, incl, peri, node, h_mag, epoch_jd, mean_anomaly,
#   perihelion_jd]: the angles in degrees, the epoch the osculating elements refer to as a Julian
#   date, the mean anomaly at that epoch, and the time of perihelion as a Julian date. The last three
#   are None when the file does not give them, and they are only used to bring the body to another
#   epoch. The mean anomaly exists only for an elliptic orbit, so a parabolic or hyperbolic comet is
#   placed on its orbit by its time of perihelion instead.

def loadCometsElements(comets_file):
    """ Loads the orbital elements of comets from the JPL comets element list. """

    comets_list = []

    with open(comets_file, encoding="utf-8", errors="replace") as f:

        for k, line in enumerate(f):

            # Skip the first two lines
            if k < 2:
                continue


            comet_name = ' '.join(line[:44].strip().split())
            q = float(line[52:63])
            e = float(line[64:74])
            incl = float(line[75:84])
            peri = float(line[85:94])
            node = float(line[95:104])
            h_mag = -1 # Not present in the JPL file, but needed for the MPC format

            # The epoch is a modified Julian date and the time of perihelion is YYYYMMDD.dddd
            epoch_jd = None
            mean_anomaly = None
            tp_jd = None

            try:
                epoch_jd = float(line[44:52]) + 2400000.5

                tp = float(line[104:].split()[0])
                tp_jd = _calendarToJD(int(tp//10000), int((tp//100) % 100), tp % 100)
                mean_anomaly = _meanAnomalyFromPerihelionTime(q, e, epoch_jd, tp_jd)

            except (ValueError, IndexError):
                pass

            comets_list.append([comet_name, q, e, incl, peri, node, h_mag, epoch_jd, mean_anomaly,
                tp_jd])

    return comets_list


def loadAsteroidsElements(asteroids_file):
    """ Loads the elements of asteroids in the MPC format. """

    asteroids_list = []

    with open(asteroids_file, encoding="utf-8", errors="replace") as f:

        for i, line in enumerate(f):
            
            # Skip first 2 lines
            if i < 2:
                continue

            asteroid_name = ' '.join(line[:38].strip().split())
            try:
                q = float(line[39:45])
            except ValueError:
                continue

            e = float(line[107:112])
            incl = float(line[101:106])
            peri = float(line[89:94])
            node = float(line[95:100])

            try:
                h_mag = float(line[63:70])  # Asteroid H magnitude
            except ValueError:
                h_mag = -1

            # The epoch is YYYYMMDD, and the mean anomaly at it is given in the file
            epoch_jd = None
            mean_anomaly = None

            try:
                epoch = int(line[72:80])
                epoch_jd = _calendarToJD(epoch//10000, (epoch//100) % 100, epoch % 100)
                mean_anomaly = float(line[80:88])

            except ValueError:
                epoch_jd = None
                mean_anomaly = None

            asteroids_list.append([asteroid_name, q, e, incl, peri, node, h_mag, epoch_jd, mean_anomaly,
                None])


    return asteroids_list


def loadMeteoritesElements(meteorites_file):
    """ Loads orbital elements of meteorites with orbits."""

    meteorites_list = []

    with open(meteorites_file, encoding="utf-8", errors="replace") as f:

        for line in f:

            # Skip commented lines
            if line.startswith("#"):
                continue


            line = line.split(';')

            meteorite_name = line[1]

            a = float(line[8])
            e = float(line[10])

            # Compute perihelion
            q = a*(1 - e)

            try:
                incl = float(line[12])
            except:
                # print("No inclination for meteorite", meteorite_name)
                continue

            try:
                peri = float(line[14])
            except:
                # print("No argument of perihelion for meteorite", meteorite_name)
                continue

            try:
                node = float(line[16])
            except:
                # print("No ascending node for meteorite", meteorite_name)
                continue
            
            # Meteorites do not have H magnitude in the orbit file, nor an epoch for the elements
            h_mag = -1

            meteorites_list.append([meteorite_name, q, e, incl, peri, node, h_mag, None, None, None])


    return meteorites_list


# Load the parent body database - avoid crash on first initialisation
comets_elements = []
asteroids_elements = []
meteorites_elements = []

if os.path.isfile(config.comets_elements_file):
    comets_elements = loadCometsElements(config.comets_elements_file)
if os.path.isfile(config.asteroids_amors_file):
    asteroids_elements = loadAsteroidsElements(config.asteroids_amors_file)
if os.path.isfile(config.asteroids_apollos_file):
    asteroids_elements += loadAsteroidsElements(config.asteroids_apollos_file)
if os.path.isfile(config.asteroids_atens_file):
    asteroids_elements += loadAsteroidsElements(config.asteroids_atens_file)
if os.path.isfile(config.meteorites_orbit_file):
    meteorites_elements = loadMeteoritesElements(config.meteorites_orbit_file)


def _resolveCriteria(d_crit):
    """ Turn a criterion key, or a list of them, into a list of lookups. A key that is unknown is
        reported and dropped, and D_SH is used if none is left. A repeated key is used once.
    """

    keys = [d_crit] if isinstance(d_crit, str) else list(d_crit)

    criteria = []

    for key in keys:

        try:
            criterion = lookupCriterion(key)

        except ValueError as err:
            print(str(err))
            continue

        if criterion[0] not in [c[0] for c in criteria]:
            criteria.append(criterion)

    if not criteria:
        print('Using D_SH by default...')
        criteria = [lookupCriterion('DSH')]

    return criteria


def searchParentBodies(q, e, incl, peri, node, d_crit='dsh', top_n=10, meteorites=False,
    geocentric=None, heliocentric=None):
    """ Compares the given orbit to the orbit of asteroids and comets from the database using the given
        D criteria and returns the top N best matches, ranked by the first criterion.

    Arguments;
        q: [float] Perihelion distance in AU.
        e: [float] Eccentricity.
        incl: [float] Inclination (radians).
        peri: [float] Argument of perihelion (radians).
        node: [float] Ascending node (radians).

    Keyword arguments:
        d_crit: [str or list] Key of the D criterion function, or a list of keys, not case
            sensitive. The bodies are ranked by the first key and the values of the others are
            carried along. Warning: the keys are not defined here. They are those of
            wmpl.Utils.Dcriteria, ELEMENT_CRITERIA, GEOCENTRIC_CRITERIA and HELIOCENTRIC_CRITERIA,
            which the command line
            of that module offers through -c, so a key means the same criterion and prints the same
            label in both, and editing them there changes them here. The criteria that read only
            part of the orbit, D_T from the Tisserand parameter and D_SAC without a node or a
            perihelion term, rank large numbers of bodies equally and say little here.
            The criteria on the geocentric radiant (DN, DR, DX) need the radiant of each orbit. A
            body of the catalogues has only its elements, so its radiant is derived from them at the
            places where its orbit comes closest to the Earth's (see
            wmpl.Utils.Dcriteria.calcGeocentricFromElements), and a body whose orbit never comes
            within 0.25 AU of the Earth's has an infinite value, which excludes it if the criterion is
            the first one. The Vida
            criterion (DVVIDA), on the corrected heliocentric direction, gets it from the same
            place with calcHeliocentricFromElements.
        top_n: [int] How many objects with the highest orbit similarity will be returned. 10 is default.
            If -1 is given, the whole list of bodies will be returned.
        meteorites: [bool] Use known orbits of meteorites for comparison instead of comets and asteroids.
        geocentric: [tuple] (ra, dec, solar longitude (radians), geocentric speed (km/s)) of the given
            orbit, for the criteria on the radiant. If None, they are derived from its elements.
        heliocentric: [tuple] (Lh, Bh, solar longitude (radians), heliocentric speed (km/s)) of the
            given orbit, the corrected heliocentric direction of the Vida criterion. If None, they are
            derived from its elements.

    Return:
        [dict] 'keys' and 'labels' of the criteria used, in order; 'query_geocentric', a list of
            the radiants of the given orbit that the criteria on the radiant used; 'geocentric_source',
            'given', 'elements' or None; 'query_heliocentric' and 'heliocentric_source', the same for
            the Vida criterion; 'matches', a list of dictionaries sorted by the first
            criterion with 'name', 'q', 'e', 'incl', 'peri', 'node' (degrees), 'h_mag',
            'd_values' (a value per criterion, in order), and 'epoch_jd', 'mean_anomaly' (deg) and
            'perihelion_jd' of the catalogue elements, which are None if the catalogue does not give
            them.
    """

    criteria = _resolveCriteria(d_crit)

    query = (q, e, incl, node, peri)

    # Radiant of the given orbit for each kind of criterion that needs one, the given one if there
    #   is one, otherwise derived from the elements
    given = {'geocentric': geocentric, 'heliocentric': heliocentric}
    query_radiants = {'geocentric': [], 'heliocentric': []}
    sources = {'geocentric': None, 'heliocentric': None}

    for kind in set(c[3] for c in criteria if c[3] != 'elements'):

        if given[kind] is not None:
            query_radiants[kind] = [tuple(given[kind])]
            sources[kind] = 'given'

        else:
            query_radiants[kind] = DERIVED_FROM_ELEMENTS[kind](q, e, incl, node, peri)
            sources[kind] = 'elements'


    # Use meteorite orbits instead of comets and asteroids
    if meteorites:
        parent_body_elements = meteorites_elements

    else:

        # Combine comets and asteroids into one parent body list
        parent_body_elements = comets_elements + asteroids_elements


    dcrit_list = []
    values_list = []

    # The criteria that take arrays (see wmpl.Utils.Dcriteria.VECTORISED_CRITERIA) are evaluated over
    #   the whole catalogue in one call: the MOID takes ~2 s for ~46 000 bodies this way, against ~40 s
    #   one body at a time
    precomputed = {}
    vectorised = [c for c, criterion in enumerate(criteria)
                  if (criterion[3] == 'elements') and (criterion[0] in VECTORISED_CRITERIA)]
    if vectorised and parent_body_elements:
        elements = np.array([body[1:6] for body in parent_body_elements], dtype=float)
        catalogue_orbits = (elements[:, 0], elements[:, 1], np.radians(elements[:, 2]),
                            np.radians(elements[:, 4]), np.radians(elements[:, 3]))
        for c in vectorised:
            precomputed[c] = np.asarray(criteria[c][2](query, catalogue_orbits), dtype=float)

    # Calculate D criteria for every comet and asteroid
    for k, body in enumerate(parent_body_elements):

        # Extract body orbital elements
        name, q2, e2, incl2, peri2, node2 = body[:6]  # Exclude H magnitude

        # Convert the body elements to radians
        incl2 = np.radians(incl2)
        peri2 = np.radians(peri2)
        node2 = np.radians(node2)

        body_orbit = (q2, e2, incl2, node2, peri2)
        body_radiants = {}

        values = []

        # Calculate the D criteria between the body and the meteoroid
        for c, (_, _, dFunc, kind) in enumerate(criteria):

            if c in precomputed:
                values.append(float(precomputed[c][k]))
                continue

            if kind == 'elements':
                values.append(float(dFunc(query, body_orbit)))
                continue

            if kind not in body_radiants:
                body_radiants[kind] = DERIVED_FROM_ELEMENTS[kind](*body_orbit)

            values.append(radiantCriterionMin(dFunc, query_radiants[kind], body_radiants[kind]))

        # A criterion written in terms of the semi-major axis is infinite against the parabolic
        #   orbits in the comet catalogue, and such a body is not a match under it. The same holds
        #   for the first criterion, by which the bodies are ranked, of a body whose orbit does not
        #   reach the Earth, and for the MOID between two open orbits (NaN). The others are kept as
        #   they are
        if not np.isfinite(values[0]):
            continue

        # Add the body index and the ranking D criterion to list
        dcrit_list.append([k, values[0]])
        values_list.append(values)


    # Sort the bodies by ascending value of the first D criterion
    if not dcrit_list:
        order = []

    else:
        order = np.argsort(np.array(dcrit_list)[:, 1], kind='stable')

    # Return the whole list of bodies if top_n is -1, otherwise the top N most similar bodies
    if top_n != -1:
        order = order[:top_n]


    matches = []

    for j in order:

        body = parent_body_elements[dcrit_list[j][0]]

        matches.append({'name': body[0], 'q': body[1], 'e': body[2], 'incl': body[3],
            'peri': body[4], 'node': body[5], 'h_mag': body[6], 'd_values': values_list[j],
            'epoch_jd': body[7], 'mean_anomaly': body[8], 'perihelion_jd': body[9]})


    return {'keys': [c[0] for c in criteria], 'labels': [c[1] for c in criteria],
        'query_geocentric': query_radiants['geocentric'], 'geocentric_source': sources['geocentric'],
        'query_heliocentric': query_radiants['heliocentric'],
        'heliocentric_source': sources['heliocentric'], 'matches': matches}


def findParentBodies(q, e, incl, peri, node, d_crit='dsh', top_n=10, meteorites=False,
    geocentric=None, heliocentric=None):
    """ Compares the given orbit to the orbit of asteroids and comets from the database using the given
        D criterion function and returns top N best matches.

        This is searchParentBodies with the matches as lists. See it for the arguments, which are the
        same.
    
    Return:
        [list] A list of best matching objects: [Object name, q, e, i, peri, node, H magnitude,
            D criterion value]. If a list of criteria was given, there is a value for each, in the
            order they were given, and the objects are sorted by the first.
    """

    search = searchParentBodies(q, e, incl, peri, node, d_crit=d_crit, top_n=top_n,
        meteorites=meteorites, geocentric=geocentric, heliocentric=heliocentric)

    return [[m['name'], m['q'], m['e'], m['incl'], m['peri'], m['node'], m['h_mag']] + m['d_values'] \
        for m in search['matches']]


def chanceNEO(search, q, e, incl, peri, node, population=None, download=True, confidence=None):
    """ For each match of a search, the probability that a NEO at least as large lies at least as close
        to the given orbit by chance, for every criterion of the search on the orbital elements: for
        the body alone, and corrected for having searched bodies of every size.

        The synthetic NEOs of the debiased population of Granvik et al. (2018) are compared with the
        given orbit, and the known NEOs of the catalogues count the bodies brighter than the model
        reaches. See wmpl.Utils.DebiasedNEOPopulation for the test and what it assumes. The criteria
        on the radiant are skipped: they would need the radiant of every synthetic NEO, which exists
        only where its orbit comes close to the Earth's.

        To rank the whole catalogue by significance, give it a search with top_n=-1 (rankByChance).

    Arguments:
        search: [dict] Return of searchParentBodies.
        q, e, incl, peri, node: [float] The orbit given to searchParentBodies, the angles in radians.

    Keyword arguments:
        population: [dict] Return of wmpl.Utils.DebiasedNEOPopulation.loadDebiasedNEOPopulation. It is
            loaded, and downloaded on first use, if None.
        download: [bool] Download the population if it is not there. If False, raise instead.
        confidence: [float] Confidence of the Poisson upper limits, 0.95 by default.

    Return:
        [dict] 'model', 'reference', 'license', 'file_path' and 'n_population' of the population, and
            'n_background', the NEOs to its faint end, and 'h_max', the absolute magnitude it reaches;
            'confidence'; 'meteor_in_model' and
            'meteor_outside_reason'; 'labels', the criteria evaluated, in the order of the search,
            and 'skipped', those on the radiant; and 'results', for each label a list with a
            dictionary per match, as returned by wmpl.Utils.DebiasedNEOPopulation.chanceAssociation.
    """

    from wmpl.Utils import DebiasedNEOPopulation as dnp

    if confidence is None:
        confidence = dnp.CHANCE_CONFIDENCE

    if population is None:
        population = dnp.loadDebiasedNEOPopulation(download=download)

    model = population['model']

    # The known NEOs, for the bodies brighter than the model reaches
    known_h_mags = np.array([body[6] for body in asteroids_elements], dtype=float)
    known_h_mags = known_h_mags[known_h_mags >= 0]

    meteor_orbit = (q, e, incl, node, peri)
    inside, reason = dnp.isInModelRegion(q, e, model)

    results = {}
    labels = []
    skipped = []

    for c, (key, label) in enumerate(zip(search['keys'], search['labels'])):

        if lookupCriterion(key)[3] != 'elements':
            skipped.append(label)
            continue

        candidates = [(m['d_values'][c], m['h_mag']) for m in search['matches']]

        results[label] = dnp.chanceAssociation(meteor_orbit, candidates, key, population, known_h_mags,
            confidence=confidence)
        labels.append(label)

    return {'model': model['name'], 'reference': model['reference'], 'license': model['license'],
        'file_path': population['file_path'], 'n_population': len(population['q']),
        'n_background': sum(dnp.countAtLeastAsLarge(model['h_max'], population, known_h_mags)),
        'h_max': model['h_max'],
        'confidence': confidence, 'meteor_in_model': inside, 'meteor_outside_reason': reason,
        'labels': labels, 'skipped': skipped, 'results': results}


def firstMatches(search, chance, top_n):
    """ The first top_n matches of a search and of its chanceNEO, or all of them if top_n is -1, so
        that one search of the whole catalogue serves both the table of the closest bodies and the
        ranking by significance. """

    if top_n == -1:
        return search, chance

    search = dict(search, matches=search['matches'][:top_n])

    if chance is not None:
        chance = dict(chance, results=dict((label, results[:top_n])
            for label, results in chance['results'].items()))

    return search, chance


def rankByChance(search, chance, label, top_n=10):
    """ The matches with a p-value under one criterion, the most significant first.

    Arguments:
        search: [dict] Return of searchParentBodies, best with top_n=-1 to rank the whole catalogue.
        chance: [dict] Return of chanceNEO for it.
        label: [str] Label of the criterion, one of chance['labels'].

    Keyword arguments:
        top_n: [int] How many to return, -1 for all.

    Return:
        [list] (match, its criterion value, result of chanceAssociation) tuples, by ascending p-value
            and, for equal ones, by the criterion.
    """

    c = search['labels'].index(label)

    ranked = [(m, m['d_values'][c], r) for m, r in zip(search['matches'], chance['results'][label])
        if r['p_value'] is not None]

    ranked.sort(key=lambda entry: (entry[2]['p_value'], entry[1]))

    return ranked if top_n == -1 else ranked[:top_n]


# The significance levels the output flags, in Gaussian sigma: 2 sigma is a hint, 3 sigma is the usual
#   bar for evidence in physics
SIGMA_HINT = 2.0
SIGMA_EVIDENCE = 3.0


def formatProbability(p_value, limit=False):
    """ A probability for the tables: plain decimals with 3 significant digits above 0.001, scientific
        notation below it, and '>0.999' for what rounds to certainty, which would otherwise be printed
        as 1.00. A limit (no synthetic orbit is as close) is prefixed with '<=': the population cannot
        resolve it.
    """

    if p_value >= 0.9995:
        text = ">0.999"

    elif p_value >= 0.001:
        text = "{:.3g}".format(p_value)

    else:
        text = "{:.2e}".format(p_value)

    return ("<=" + text) if limit else text


def formatSigma(sigma, limit=False):
    """ A significance for the tables: one decimal from 1 sigma up, '<1' below it, where the number
        measures nothing, and prefixed with '>=' for a limit. One decimal is all the value supports:
        p_global is conservative by up to 1.6 times, which moves it by 0.1 sigma at 3.5 sigma.
    """

    if sigma < 1.0:
        return "<1" if not limit else ">={:.1f}".format(sigma)

    return "{:s}{:.1f}".format(">=" if limit else "", sigma)


def formatOdds(p_value):
    """ A probability as '1 in N' with 2 significant digits, for the ones small enough to be read that
        way. """

    if p_value >= 0.1:
        return ""

    odds = 1.0/p_value
    digits = 2 - 1 - int(np.floor(np.log10(odds)))

    return "1 in {:,.0f}".format(round(odds, digits)).replace(",", " ")


def _significanceText(sigma):
    """ The flag of a significance level. """

    if sigma >= SIGMA_EVIDENCE:
        return "3 sigma"

    if sigma >= SIGMA_HINT:
        return "2 sigma"

    return ""


def summariseChanceNEO(chance, search, search_all, chance_all):
    """ The summary that closes the output of --chance_neo: what was tested, the most significant
        body, whether anything is distinguishable from chance, and what cannot be concluded.

        It is written for the criterion the bodies were ranked by, which is the one to choose before
        looking, with one line for each of the others as a check. """

    h_max = chance_all['h_max']

    # The criterion the bodies were ranked by, if it was evaluated
    primary = search['labels'][0] if search['labels'][0] in chance_all['labels'] \
        else chance_all['labels'][0]

    asteroids = [m for m in search_all['matches'] if m['h_mag'] >= 0]

    ranked = rankByChance(search_all, chance_all, primary, top_n=-1)
    n_tested = len(ranked)
    n_faint = sum(1 for m in asteroids if m['h_mag'] > h_max)
    n_other = len(search_all['matches']) - len(asteroids)

    def _describe(entry, label):

        match, d_value, r = entry

        p_text = formatProbability(r['p_value_global'], r['is_limit'])

        # A limit reads as "p_global <= x", a value as "p_global = x"
        text = "{:s}, {:s} = {:.3f}, H = {:.1f}, p_global {:s}{:s} ({:s} sigma)".format(match['name'],
            label, d_value, match['h_mag'], "" if p_text.startswith("<=") else "= ", p_text,
            formatSigma(r['sigma'], r['is_limit']))

        odds = formatOdds(r['p_value_global'])

        if odds:
            text += ", " + odds + (" or rarer" if r['is_limit'] else "")

        return text

    lines = ["Summary, for {:s}, the criterion the bodies were ranked by:".format(primary)]
    lines.append("  Tested: {:d} of the {:d} bodies searched. {:d} asteroids are fainter than H = {:.0f} and "
        "{:d} comets or meteorites have no H, so the".format(n_tested, len(search_all['matches']),
            n_faint, h_max, n_other))
    lines.append("  model cannot say whether they are a chance match.")

    if not ranked:
        lines.append("  No body could be tested.")
        return lines

    significant = [entry for entry in ranked if entry[2]['sigma'] >= SIGMA_EVIDENCE]
    hints = [entry for entry in ranked if SIGMA_HINT <= entry[2]['sigma'] < SIGMA_EVIDENCE]

    lines.append("  Most significant: " + _describe(ranked[0], primary) + ".")

    if significant:

        lines.append("  Verdict: {:d} {:s} {:s} {:.0f} sigma, the usual bar for evidence:".format(
            len(significant), "body reaches" if len(significant) == 1 else "bodies reach",
            "at least", SIGMA_EVIDENCE))

        for entry in significant[:5]:
            lines.append("    " + _describe(entry, primary))

        if len(significant) > 5:
            lines.append("    and {:d} more.".format(len(significant) - 5))

    elif hints:

        lines.append("  Verdict: a hint, not evidence. {:d} {:s} {:.0f} sigma but none {:.0f}, the usual bar:"
            .format(len(hints), "body reaches" if len(hints) == 1 else "bodies reach", SIGMA_HINT,
                SIGMA_EVIDENCE))

        for entry in hints[:5]:
            lines.append("    " + _describe(entry, primary))

    else:
        lines.append("  Verdict: no body is distinguishable from chance. None reaches {:.0f} sigma "
            "(p_global > {:.3f}).".format(SIGMA_HINT, math.erfc(SIGMA_HINT/math.sqrt(2.0))))

    if any(entry[2]['is_limit'] for entry in (significant + hints)[:5] + ranked[:1]):
        lines.append("  No synthetic orbit is as close as some of these, so their p is at most and their sigma "
            "at least the value shown: the {:d} synthetic orbits cannot resolve more.".format(
                chance_all['n_population']))

    # How many of the closest bodies could not be tested
    n_closest = len(search['matches'])
    n_untested = sum(1 for r in chance['results'][primary] if r['reason'] is not None)

    if n_untested:
        lines.append("  {:d} of the {:d} closest by {:s} could not be tested. A small parent body cannot be "
            "ruled in or out by this method.".format(n_untested, n_closest, search['labels'][0]))

    others = [label for label in chance_all['labels'] if label != primary]

    if others:

        lines.append("  Other criteria, as a check and not to pick the best of (the smallest p over several "
            "criteria is not a p-value):")

        for label in others:

            other = rankByChance(search_all, chance_all, label, top_n=1)

            lines.append("    {:s}: ".format(label) + (_describe(other[0], label) if other
                else "no body could be tested"))

    lines.append("  A low p says the match is unlikely by chance, not that the body is the parent. "
        "The meteoroid's orbit is taken as exact.")

    return lines


def formatChanceNEO(chance, search, top_n=10, search_all=None, chance_all=None):
    """ The lines that print the result of chanceNEO: for each criterion the matches of the search,
        and, given the search and chanceNEO of the whole catalogue, the top_n most significant
        asteroids, and the summary. """

    model_range = "17 < H < 25, q < 1.3 AU, a < 4.2 AU"

    lines = ["Chance coincidence with the debiased NEO population of {:s}: {:d} synthetic NEOs "
        "({:s}), with the known ones with".format(chance['model'], chance['n_population'],
            model_range),
        "H <= 17, {:d} background NEOs to H = {:.0f}.".format(chance['n_background'], chance['h_max']),
        "  N(H<=Hmag)  NEOs at least as large as the body.",
        "  closer      synthetic orbits at least as close to the meteoroid's as the body's.",
        "  expected    NEOs at least as large expected as close by chance, "
            "N(H<=Hmag)*(closer + 1)/{:d}.".format(chance['n_population'] + 1),
        "  p           probability of at least one. Valid for a body named before the search.",
        "  p_global    the same counting a search over bodies of every size: the one to quote for a "
            "body the search found.",
        "  sigma       p_global as Gaussian standard deviations: 2 is a hint (p_global 0.046), 3 is "
            "evidence (0.0027).",
        "              Below 1 it measures nothing and is printed <1. A <= or >= marks a limit.",
        "  A summary follows the tables."]

    if chance['skipped']:
        lines.append("  Skipped {:s}: a criterion on the radiant needs the radiant of every synthetic "
            "NEO, which exists only where its orbit comes close to the Earth's.".format(
                ", ".join(chance['skipped'])))

    if not chance['meteor_in_model']:
        lines.append("  No p-value for this meteoroid: " + chance['meteor_outside_reason'] + ".")
        return lines

    def _table(label, entries):

        rows = ["Name                               ,  Hmag, {:>9s}, N(H<=Hmag),  closer,   expected,"
            "          p,   p_global,  sigma,".format(label[:9])]

        for match, d_value, r in entries:

            start = "{:35s}, {:5.2f}, {:9.4f}, ".format(match['name'], match['h_mag'], d_value) \
                if np.isfinite(d_value) else "{:35s}, {:5.2f}, {:>9s}, ".format(match['name'],
                    match['h_mag'], "n/a")

            if r['reason'] is not None:
                rows.append(start + "n/a: " + r['reason'])
                continue

            limit = r['is_limit']

            rows.append(start + "{:10d}, {:7d}, {:>10s}, {:>10s}, {:>10s}, {:>6s}, {:s}".format(
                r['n_at_least_as_large'], r['n_closer'],
                ("<=" if limit else "") + "{:.3g}".format(r['expected']),
                formatProbability(r['p_value'], limit), formatProbability(r['p_value_global'], limit),
                formatSigma(r['sigma'], limit), _significanceText(r['sigma'])).rstrip())

        return rows

    for label in chance['labels']:

        c = search['labels'].index(label)

        lines.append("")
        lines.append("{:s}, the matches above, ranked by {:s}:".format(label, search['labels'][0]))
        lines += _table(label, [(m, m['d_values'][c], r) for m, r in zip(search['matches'],
            chance['results'][label])])

        if (search_all is not None) and (chance_all is not None):

            ranked = rankByChance(search_all, chance_all, label, top_n=top_n)

            lines.append("")
            lines.append("{:s}, the {:d} most significant of the {:d} bodies searched, ranked by "
                "p:".format(label, len(ranked), len(search_all['matches'])))
            lines += _table(label, ranked)

    lines.append("")
    lines.append("Please cite " + chance['reference'] + ".")

    if (search_all is not None) and (chance_all is not None):
        lines.append("")
        lines += summariseChanceNEO(chance, search, search_all, chance_all)

    return lines




def propagateBodies(bodies, jd_target):
    """ Bring bodies of the catalogues to the reference epoch of a meteor, so that they can be handed
        to a REBOUND integration with the meteoroid (wmpl.Rebound.REBOUND, --parent_bodies).

        The integration is that of wmpl.Rebound.REBOUND.propagateBodiesToEpoch: the same simulation
        the meteoroid is integrated in, with the Sun, the planets and the Moon from DE430, the Earth's
        J2 and J4 and general relativity. Non-gravitational forces are not modelled, which matters for
        comets. The catalogue epochs are taken as TDB (they are TT, which differs from it by less than
        2 ms), and the reference epoch of the meteor, which is UTC, is converted to TDB, as
        wmpl.Rebound.REBOUND does with the trajectory.

        This is for preparing the initial conditions. The ranking of the bodies is not redone with
        the propagated elements.

    Arguments:
        bodies: [list] Dictionaries of the matches of searchParentBodies, which need 'q', 'e', 'incl',
            'peri', 'node', 'epoch_jd', and 'mean_anomaly' or, for a parabolic or hyperbolic comet,
            'perihelion_jd'.
        jd_target: [float] Julian date (UTC) to bring the bodies to, the reference time of the
            trajectory.

    Return:
        [list] For each body, None if it could not be propagated (no epoch, or neither a mean anomaly
            nor a time of perihelion in the catalogue, an epoch outside the ephemeris, an impact or an
            escape on the way), or a dictionary: 'jd' (UTC) and 'jd_tdb', 'q', 'a' (AU), 'e',
            'incl', 'peri', 'node', 'mean_anomaly' (degrees), 'position' (AU) and 'velocity'
            (AU/day), heliocentric in the ecliptic J2000 frame.
    """

    from wmpl.Rebound import REBOUND as reb

    if not reb.REBOUND_FOUND:
        raise ImportError(reb._REBOUND_IMPORT_ERROR)

    # A REBOUNDx that cannot attach would fail inside the integration with a less helpful message
    reb.checkReboundxUsable()

    jd_tdb = reb.utcToTdb(jd_target)

    elements = [{'q': b['q'], 'e': b['e'], 'incl': b['incl'], 'peri': b['peri'], 'node': b['node'],
        'mean_anomaly': b['mean_anomaly'], 'perihelion_jd': b.get('perihelion_jd'),
        'epoch_jd': b['epoch_jd']} for b in bodies]

    results = reb.propagateBodiesToEpoch(elements, jd_tdb)

    for rec in results:
        if rec is not None:
            rec['jd'] = jd_target

    return results


def criteriaAtEpoch(search, propagated, q, e, incl, peri, node):
    """ Value of the criteria of a search with the elements of the bodies brought to the reference epoch
        of the meteor (propagateBodies), against the same orbit of the meteor the search used.

        The ranking stays the one of the catalogue elements. This shows how much the values move
        when both orbits refer to the same epoch. A criterion on the radiant reads the radiant the
        search used for the meteor (measured, or derived from its elements) and the one derived from
        the propagated elements for the body.

    Arguments:
        search: [dict] Return of searchParentBodies.
        propagated: [list] Return of propagateBodies for its matches.
        q, e, incl, peri, node: [float] The orbit of the meteor, as given to searchParentBodies.

    Return:
        [list] For each match, None if it was not propagated, or a list of the values of the criteria
            in the order of the search, infinite for a criterion on the radiant when the propagated
            orbit does not come close enough to the Earth's.
    """

    criteria = [lookupCriterion(key) for key in search['keys']]
    query = (q, e, incl, node, peri)

    out = []

    for p in propagated:

        if p is None:
            out.append(None)
            continue

        body_orbit = (p['q'], p['e'], np.radians(p['incl']), np.radians(p['node']),
            np.radians(p['peri']))

        values = []

        for _, _, func, kind in criteria:

            if kind == 'elements':
                values.append(float(func(query, body_orbit)))

            else:
                values.append(radiantCriterionMin(func, search['query_' + kind],
                    DERIVED_FROM_ELEMENTS[kind](*body_orbit)))

        out.append(values)

    return out


# The values of chanceNEO saved per match and criterion, as chance_neo_<label>_<name>
CHANCE_NEO_FIELDS = ['n_at_least_as_large', 'n_closer', 'expected', 'p_value', 'expected_global',
    'p_value_global', 'sigma', 'is_limit', 'expected_upper', 'p_value_upper', 'p_value_global_upper',
    'reason']


def _bodyRows(search, propagated=None, at_epoch_values=None, chance=None):
    """ The matches of a search as flat rows of (name, value) pairs, for the saved files. """

    rows = []

    for rank, match in enumerate(search['matches']):

        row = [('rank', rank + 1), ('name', match['name']), ('q_AU', match['q']),
            ('e', match['e']), ('incl_deg', match['incl']), ('peri_deg', match['peri']),
            ('node_deg', match['node']), ('h_mag', match['h_mag']),
            ('catalogue_epoch_jd', match['epoch_jd']),
            ('catalogue_mean_anomaly_deg', match['mean_anomaly']),
            ('catalogue_perihelion_jd', match.get('perihelion_jd'))]

        row += [(label, value) for label, value in zip(search['labels'], match['d_values'])]

        if propagated is not None:

            p = propagated[rank]
            keys = ['jd', 'jd_tdb', 'q', 'a', 'e', 'incl', 'peri', 'node', 'mean_anomaly']
            names = ['jd', 'jd_tdb', 'q_AU', 'a_AU', 'e', 'incl_deg', 'peri_deg', 'node_deg',
                'mean_anomaly_deg']

            for key, name in zip(keys, names):
                row.append(('at_epoch_' + name, None if p is None else p[key]))

            for axis, value in zip('xyz', [None]*3 if p is None else p['position']):
                row.append(('at_epoch_' + axis + '_AU', value))

            for axis, value in zip('xyz', [None]*3 if p is None else p['velocity']):
                row.append(('at_epoch_v' + axis + '_AU_day', value))

            if at_epoch_values is not None:

                values = at_epoch_values[rank]

                for c, label in enumerate(search['labels']):
                    row.append(('at_epoch_' + label, None if values is None else values[c]))

        if chance is not None:

            for label in chance['labels']:

                r = chance['results'][label][rank]

                for field in CHANCE_NEO_FIELDS:
                    row.append(('chance_neo_' + label + '_' + field, r[field]))

        rows.append(row)

    return rows


def _rankedRows(ranked):
    """ The bodies ranked by significance (rankByChance), for each criterion, as flat rows of
        (name, value) pairs. """

    rows = []

    for label, entries in ranked.items():

        for rank, (match, d_value, r) in enumerate(entries):

            rows.append([('criterion', label), ('rank', rank + 1), ('name', match['name']),
                ('q_AU', match['q']), ('e', match['e']), ('incl_deg', match['incl']),
                ('peri_deg', match['peri']), ('node_deg', match['node']), ('h_mag', match['h_mag']),
                ('d_value', d_value)] + [(field, r[field]) for field in CHANCE_NEO_FIELDS])

    return rows


def saveMatches(file_path, search, meteor=None, propagated=None, at_epoch_values=None, chance=None,
    ranked=None):
    """ Save the matches of a search, with the value of every D criterion that was used.

        The format follows the extension: JSON for .json, a CSV table otherwise. The CSV has a row
        per body with its elements, a column per criterion in the order they were given, which is
        also the ranking order, and, if the bodies were propagated, their elements and heliocentric
        state at the epoch and the criteria with them, and, with chance, the chance-coincidence
        values per criterion. The JSON also holds the orbit that was searched for.

        The bodies ranked by significance (ranked) go into the JSON as 'most_significant', or, with a
        CSV, into a second table beside it, <name>_chance_neo.csv, a row per criterion and body.

    Arguments:
        file_path: [str] Path of the file to write.
        search: [dict] Return of searchParentBodies.

    Keyword arguments:
        meteor: [dict] The orbit that was searched for, saved to the JSON file.
        propagated: [list] Return of propagateBodies for the matches, in the same order.
        at_epoch_values: [list] Return of criteriaAtEpoch for them.
        chance: [dict] Return of chanceNEO for the search.
        ranked: [dict] For each criterion label, the return of rankByChance.

    Return:
        [list] The paths of the files written.
    """

    def _finite(v):
        """ Infinite values are not JSON. """
        return v if (v is not None) and np.isfinite(v) else None

    rows = _bodyRows(search, propagated, at_epoch_values, chance)

    if file_path.lower().endswith('.json'):

        matches = []

        for row, match in zip(rows, search['matches']):

            k = len(matches)

            # The flat columns of the state at the epoch are kept in 'at_epoch' instead, and those of
            #   the chance coincidence in 'chance_neo'
            entry = dict((name, v) for name, v in row
                if (name not in search['labels']) and (not name.startswith('at_epoch_'))
                and (not name.startswith('chance_neo_')))
            entry['criteria'] = dict((label, _finite(v)) for label, v in zip(search['labels'],
                match['d_values']))

            entry['at_epoch'] = None
            if (propagated is not None) and (propagated[k] is not None):

                entry['at_epoch'] = dict(propagated[k])

                if at_epoch_values is not None:
                    entry['at_epoch']['criteria'] = dict((label, _finite(v))
                        for label, v in zip(search['labels'], at_epoch_values[k]))

            if chance is not None:
                entry['chance_neo'] = dict((label, dict((field, chance['results'][label][k][field])
                    for field in CHANCE_NEO_FIELDS)) for label in chance['labels'])

            matches.append(entry)

        out = {'meteor': meteor, 'criteria': search['labels'], 'criteria_keys': search['keys'],
            'ranked_by': search['labels'][0],
            'geocentric_source': search['geocentric_source'],
            'heliocentric_source': search['heliocentric_source'],
            'frame': 'heliocentric ecliptic J2000, AU and AU/day' if propagated is not None else None,
            'matches': matches}

        if chance is not None:
            out['chance_neo'] = dict((name, chance[name]) for name in ('model', 'reference', 'license',
                'n_population', 'n_background', 'h_max', 'confidence', 'meteor_in_model',
                'meteor_outside_reason', 'labels', 'skipped'))

            if ranked is not None:
                out['chance_neo']['most_significant'] = dict((label, [dict(row[1:])
                    for row in _rankedRows({label: entries})]) for label, entries in ranked.items())

        with open(file_path, 'w') as f:
            json.dump(out, f, indent=2)

        return [file_path]

        return

    with open(file_path, 'w', newline='') as f:

        writer = csv.writer(f)

        if rows:
            writer.writerow([k for k, _ in rows[0]])

        for row in rows:
            writer.writerow(['' if v is None else v for _, v in row])

    written = [file_path]

    if ranked:

        ranked_path = os.path.splitext(file_path)[0] + '_chance_neo.csv'
        ranked_rows = _rankedRows(ranked)

        with open(ranked_path, 'w', newline='') as f:

            writer = csv.writer(f)

            if ranked_rows:
                writer.writerow([k for k, _ in ranked_rows[0]])

            for row in ranked_rows:
                writer.writerow(['' if v is None else v for _, v in row])

        written.append(ranked_path)

    return written


def plotCriteriaBars(search, file_path, title=None):
    """ Save a bar plot of the matches, a panel for each D criterion that was used.

        The bodies are listed top to bottom in the order of the first criterion, which is the same in
        every panel so that the others can be read against it. Each panel has its own scale because
        the criteria are not in the same units. A body that does not reach the Earth has no value
        for a criterion on the radiant and is marked n/a.

    Arguments:
        file_path: [str] Path of the image to write.
        search: [dict] Return of searchParentBodies.

    Keyword arguments:
        title: [str] Title of the figure.
    """

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    matches = search['matches']
    labels = search['labels']

    surface, ink, ink_secondary, bar_color, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#2a78d6', '#e6e5e1'

    n_panels = len(labels)
    n_bodies = max(len(matches), 1)

    fig = Figure(figsize=(2.6*n_panels + 2.8, 0.42*n_bodies + 1.4), dpi=100, facecolor=surface)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, n_panels, sharey=True, squeeze=False)[0]

    names = [m['name'] for m in matches]
    positions = np.arange(len(matches))

    for k, ax in enumerate(axes):

        ax.set_facecolor(surface)

        values = np.array([m['d_values'][k] for m in matches], dtype=float)
        finite = np.isfinite(values)

        ax.barh(positions[finite], values[finite], height=0.6, color=bar_color, zorder=2)

        top = values[finite].max() if finite.any() else 1.0

        for pos, value, ok in zip(positions, values, finite):
            ax.text((value if ok else 0) + 0.02*top, pos, '{:.3f}'.format(value) if ok else 'n/a',
                va='center', ha='left', fontsize=8, color=ink_secondary)

        ax.set_xlim(0, top*1.25)
        ax.set_title(labels[k] + (' (ranking)' if k == 0 else ''), fontsize=10, color=ink, loc='left')
        ax.xaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink_secondary, labelsize=8, length=0)

        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)

        ax.spines['bottom'].set_color(grid)

    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names, fontsize=8, color=ink)

    # The best match on top
    axes[0].set_ylim(len(matches) - 0.5, -0.5)

    if title:
        fig.suptitle(title, fontsize=11, color=ink, x=0.01, ha='left')

    fig.tight_layout()
    fig.savefig(file_path, facecolor=surface)




if __name__ == "__main__":


    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run parent body search on given trajectory pickle file or orbital elements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Using several D criteria:
  Give them to -d one after another. The bodies are ranked by the FIRST one, which is the one the
  top N is taken by, and the others are printed beside it for the same bodies:

      python -m wmpl.Utils.ParentBodySearch traj.pickle -d DSH DD DN

  With more than one criterion a bar plot, a panel per criterion with the bodies in the ranking
  order, is saved as <pickle name>_parent_bodies_<keys>.png beside the pickle (or in the working
  directory), unless --no_plot is given. -d takes everything up to the next option, so put
  the pickle before it or end the line with another option.

  The criteria on the radiant (DN, DR, DX) need a geocentric radiant, solar longitude and speed. A
  pickle has them, orbital elements given by hand may be completed with --ra, --dec, --sol and
  --vg, and where they are missing they are derived from the elements. The bodies of the
  catalogues have only elements, so theirs are always derived where their orbit comes closest to the
  Earth's (one radiant near each node, such as the eta Aquariids and the Orionids of 1P/Halley), and a
  body whose orbit never comes within 0.25 AU of the Earth's has no value (n/a).
  The Vida criterion (DVVIDA) takes the corrected heliocentric direction and speed instead. A
  pickle has them, a radiant given by hand is converted to them, and otherwise they are derived
  from the elements at the same place.

  The ranking uses the catalogue elements as they are. --propagate does not change it: it brings the
  top N to the reference epoch of the meteor, to be used as initial conditions of a REBOUND integration.
  --save writes the matches, with every criterion, to a CSV or JSON file.

Is a match more than chance?
  --chance_neo gives, for each asteroid and each criterion on the elements, the number of NEOs at least
  as large expected as close to the orbit by chance and the probability of at least one, from the
  debiased NEO population of Granvik et al. (2018): p for a body named before the search, p_global
  corrected for having searched bodies of every size, the one to quote for a body the search found,
  and its sigma (2 sigma = 0.046 is a hint, 3 sigma = 0.0027 is evidence). It lists the top N by D and the N
  most significant asteroids of the whole catalogue, and ends with a summary and a verdict. The
  population, 28 MB, is downloaded to wmpl/share the first time:

      python -m wmpl.Utils.ParentBodySearch traj.pickle -d DSH RHO2 --chance_neo""")

    arg_parser.add_argument('traj_path', nargs="?", metavar='TRAJ_PATH', type=str, \
        help="Path to the trajectory pickle file.")

    arg_parser.add_argument('-q', '--q', metavar='PERIHELION_DIST', help="Perihelion distance in AU.", \
        type=float)

    arg_parser.add_argument('-e', '--e', metavar='ECCENTRICITY', help="Eccentricity.", \
        type=float)

    arg_parser.add_argument('-i', '--i', metavar='INCLINATION', help="Inclination (deg).", \
        type=float)

    arg_parser.add_argument('-p', '--peri', metavar='ARG_OF_PERI', help="Argument of perihelion (deg).", \
        type=float)

    arg_parser.add_argument('-n', '--node', metavar='ASCENDING_NODE', help="Ascending node (deg).", \
        type=float)

    arg_parser.add_argument('--ra', metavar='RA_G', type=float, \
        help="Right ascension of the geocentric radiant (deg), for DN, DR and DX.")

    arg_parser.add_argument('--dec', metavar='DEC_G', type=float, \
        help="Declination of the geocentric radiant (deg), for DN, DR and DX.")

    arg_parser.add_argument('--sol', metavar='SOLAR_LON', type=float, \
        help="Solar longitude (deg), for DN, DR and DX.")

    arg_parser.add_argument('--vg', metavar='V_G', type=float, \
        help="Geocentric speed (km/s), for DN, DR and DX.")

    arg_parser.add_argument('-d', '--dcrit', metavar='D_CRIT', nargs='+', help="D criterion type, 'DSH' by "
        "default. One or more keys, case insensitive: "
        + ", ".join(ALL_CRITERIA_KEYS) + ". With several the bodies are "
        "ranked by the first. DT and DSAC read only part of the orbit and rank "
        "large numbers of bodies equally, so they say little in a search. MOID is the minimum orbit "
        "intersection distance in AU: how close the orbits come in space, not how similar they are, "
        "so give it with a D criterion (e.g. -d DSH MOID). The keys are those the "
        "Dcriteria command line takes.", \
        type=str, default=['DSH'])

    arg_parser.add_argument('-t', '--top_n', metavar='TOP_N', help="How many matches to list, 10 by default.", \
        type=int, default=10)

    arg_parser.add_argument('-m', '--meteorites', action="store_true", \
        help="""Use meteorites with orbits for comparison instead of asteroids and comets. """
        )

    arg_parser.add_argument('--save', metavar='FILE', type=str, \
        help="Save the matches with the value of every D criterion to this file: JSON if it ends in "
        ".json, a CSV table otherwise.")

    arg_parser.add_argument('--propagate', action="store_true", \
        help="Bring the top N bodies to the reference epoch of the meteor with the integrator of "
        "wmpl.Rebound.REBOUND, and list their elements there with every D criterion recomputed from "
        "them, and save both with --save. Comets, parabolic and hyperbolic ones included, are placed "
        "by their time of perihelion. The epoch is that of the pickle, or --jd (UTC). This does not "
        "change the ranking. The JSON saved with it can be given to wmpl.Rebound.REBOUND "
        "--parent_bodies, which then starts the bodies from these states instead of propagating "
        "them again.")

    arg_parser.add_argument('--jd', metavar='JD', type=float, \
        help="Reference epoch of the meteor as a Julian date, if there is no pickle.")

    arg_parser.add_argument('--chance_neo', action="store_true", \
        help="For each asteroid, the probability that a NEO at least as large lies at least as close "
        "to the orbit by chance (the p-value of the match), alone and corrected for the search "
        "(p_global, with its sigma), for every criterion on the orbital elements given with -d, "
        "against the debiased NEO population of Granvik et al. (2018), the N most significant "
        "asteroids of the catalogue, and a summary with a verdict. The population (28 MB) is downloaded to wmpl/share on first use. "
        "Comets, meteorites and asteroids fainter than H = 25 get no p-value, and the criteria on the "
        "radiant are skipped. See wmpl.Utils.DebiasedNEOPopulation.")

    arg_parser.add_argument('--no_plot', action="store_true", \
        help="Do not save the bar plot that is saved by default when several D criteria are given.")

    arg_parser.add_argument('--plot_path', metavar='FILE', type=str, \
        help="Where to save the bar plot, instead of beside the pickle.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # If the trajectory pickle was given, load the orbital elements from it
    geocentric = None
    heliocentric = None
    jd_ref = cml_args.jd

    if cml_args.traj_path is not None:

        # Load the trajectory pickle
        traj = loadPickle(*os.path.split(cml_args.traj_path))

        # Load orbital elements
        q = traj.orbit.q
        e = traj.orbit.e
        incl = np.degrees(traj.orbit.i)
        peri = np.degrees(traj.orbit.peri)
        node = np.degrees(traj.orbit.node)

        # The radiant, the solar longitude and the speed, for the criteria on the radiant
        orbit = traj.orbit

        if all(v is not None for v in (orbit.ra_g, orbit.dec_g, orbit.la_sun, orbit.v_g)):
            geocentric = (orbit.ra_g, orbit.dec_g, orbit.la_sun, orbit.v_g/1000.0)

        # The corrected heliocentric direction and speed, for the Vida criterion
        if all(getattr(orbit, name, None) is not None for name in ('L_h', 'B_h', 'la_sun', 'v_h')):
            heliocentric = (orbit.L_h, orbit.B_h, orbit.la_sun, orbit.v_h/1000.0)

        if (jd_ref is None) and (getattr(traj, 'jdt_ref', None) is not None):
            jd_ref = traj.jdt_ref


    # Otherwise, load orbital elements from the manual entry
    else:

        # Check that all elements are given
        if (cml_args.q is not None) and (cml_args.e is not None) and (cml_args.i is not None) \
            and (cml_args.peri is not None) and (cml_args.node is not None):

            q = cml_args.q
            e = cml_args.e
            incl = cml_args.i
            peri = cml_args.peri
            node = cml_args.node

            radiant = (cml_args.ra, cml_args.dec, cml_args.sol, cml_args.vg)

            if all(v is not None for v in radiant):
                geocentric = (np.radians(radiant[0]), np.radians(radiant[1]), np.radians(radiant[2]),
                    radiant[3])

                # The Vida criterion uses the same measured radiant, not the elements
                heliocentric = calcHeliocentricFromGeocentric(*geocentric)

            elif any(v is not None for v in radiant):
                print("The radiant needs --ra, --dec, --sol and --vg all together!")
                sys.exit(1)

        else:
            print("All orbital elements need to be specified: q, e, i, peri, node!")
            sys.exit()


    # Check the D criterion keys, and take their labels, which are the ones the Dcriteria command line
    #   prints. searchParentBodies warns and drops an unknown key, but a typed key is worth
    #   stopping for, so that a run is not silently labelled with a criterion it did not use
    try:
        d_crit_keys = [lookupCriterion(key)[0] for key in cml_args.dcrit]

    except ValueError as err:
        print(str(err), file=sys.stderr)
        sys.exit(1)

    if cml_args.propagate and (jd_ref is None):
        print("--propagate needs the reference epoch of the meteor: a trajectory pickle or --jd.",
            file=sys.stderr)
        sys.exit(1)



    # Print orbital elements
    print("Orbital elements:")
    print("  q = {:.5f} AU".format(q))
    print("  e = {:.5f}".format(e))
    print("  i = {:.5f} deg".format(incl))
    print("  w = {:.5f} deg".format(peri))
    print("  O = {:.5f} deg".format(node))
    if jd_ref is not None:
        print("  reference epoch JD {:.5f} ({:s})".format(jd_ref, _jdToString(jd_ref)))
    print()


    ##########################################################################################################

    # Find parent bodies for the given orbit. To rank them by significance, --chance_neo needs the
    #   whole catalogue: the criteria are computed for every body anyway, and the top N is taken from it
    search_all = searchParentBodies(q, e, np.radians(incl), np.radians(peri), np.radians(node), \
        d_crit=d_crit_keys, top_n=-1 if cml_args.chance_neo else cml_args.top_n,
        meteorites=cml_args.meteorites, geocentric=geocentric, heliocentric=heliocentric)

    search, _ = firstMatches(search_all, None, cml_args.top_n)

    labels = search['labels']
    matches = search['matches']


    # The radiant of the given orbit that each kind of criterion on the radiant used
    for kind, keys, line in (
        ('geocentric', GEOCENTRIC_CRITERIA_KEYS,
            "  RA = {:.3f} deg, Dec = {:.3f} deg, solar longitude = {:.3f} deg, Vg = {:.3f} km/s"),
        ('heliocentric', HELIOCENTRIC_CRITERIA_KEYS,
            "  Lh = {:.3f} deg, Bh = {:.3f} deg, solar longitude = {:.3f} deg, Vh = {:.3f} km/s")):

        source = search[kind + '_source']

        if source is None:
            continue

        print("{:s} radiant for {:s}".format(kind.capitalize(), ", ".join(l for l, k \
            in zip(labels, search['keys']) if k in keys)) + (" (given):" if source == 'given' \
            else " (derived from the elements, where the orbit comes closest to the Earth's):"))

        if not search['query_' + kind]:
            print("  The orbit does not come within {:.2f} AU of the Earth's, so it has no radiant and these "
                "criteria have no value.".format(RADIANT_MAX_DISTANCE))

        for lon, lat, sol, speed in search['query_' + kind]:
            print(line.format(np.degrees(lon), np.degrees(lat), np.degrees(sol), speed))

        print()


    print("Top {:d} matches, ranked by {:s}:".format(len(matches), labels[0]))
    # Each column as wide as its label, so that the values line up under it, with the decimals of its
    #   criterion (the MOID needs more than the D criteria)
    decimals = [CRITERIA_DECIMALS.get(k, 3) for k in search['keys']]
    widths = [max(8, len(l), d + 3) for l, d in zip(labels, decimals)]

    print('Name                               ,     q,     e,   incl,    peri,    node,  Hmag,',
        ', '.join('{:>{:d}s}'.format(l, w) for l, w in zip(labels, widths)))

    for m in matches:
        print("{:35s}, {:.3f}, {:.3f}, {:6.2f}, {:7.3f}, {:7.3f}, {:5.2f}, ".format(m['name'], m['q'],
            m['e'], m['incl'], m['peri'], m['node'], m['h_mag']) \
            + ", ".join("{:{:d}.{:d}f}".format(v, w, d) if np.isfinite(v) else "{:>{:d}s}".format("n/a", w) \
            for v, w, d in zip(m['d_values'], widths, decimals)))


    # The probability that each match is a chance coincidence
    chance = None
    ranked = None

    if cml_args.chance_neo and matches:

        print()

        if all(lookupCriterion(key)[3] != 'elements' for key in search['keys']):
            print("--chance_neo needs a criterion on the orbital elements: {:s} {:s} on the radiant, "
                "which would need the radiant of every synthetic NEO, and that exists only where its "
                "orbit comes close to the Earth's.".format(", ".join(labels),
                    "is" if len(labels) == 1 else "are"))

        else:

            if 'MOID' in search['keys']:
                print("The MOID against the 802000 synthetic NEOs takes about half a minute...")

            chance_all = None

            try:
                chance_all = chanceNEO(search_all, q, e, np.radians(incl), np.radians(peri),
                    np.radians(node))

            except IOError as err:
                print("--chance_neo: {:s}".format(str(err)), file=sys.stderr)

            if chance_all is not None:

                _, chance = firstMatches(search_all, chance_all, cml_args.top_n)

                ranked = dict((label, rankByChance(search_all, chance_all, label,
                    top_n=cml_args.top_n)) for label in chance_all['labels'])

                print("\n".join(formatChanceNEO(chance, search, top_n=cml_args.top_n,
                    search_all=search_all, chance_all=chance_all)))


    # Bring the bodies to the reference epoch of the meteor
    propagated = None
    at_epoch_values = None

    if cml_args.propagate:

        print()

        try:
            propagated = propagateBodies(matches, jd_ref)

        except ImportError as err:
            print("REBOUND and REBOUNDx are needed to propagate the bodies: {:s}".format(str(err)),
                file=sys.stderr)
            sys.exit(1)

        # A REBOUNDx compiled against another REBOUND, which wmpl.Rebound.REBOUND explains
        except RuntimeError as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

        at_epoch_values = criteriaAtEpoch(search, propagated, q, e, np.radians(incl), np.radians(peri),
            np.radians(node))

        print("At the reference epoch of the meteor, JD {:.5f}: heliocentric elements (ecliptic J2000) "
            "and the".format(jd_ref))
        print("criteria with them. The ranking above, from the catalogue elements, is kept.")
        print('Name                               ,      q,      e,   incl,    peri,    node, cat. epoch,',
            ', '.join('{:>{:d}s}'.format(l, w) for l, w in zip(labels, widths)))

        for m, p, values in zip(matches, propagated, at_epoch_values):

            if p is None:
                print("{:35s}, not propagated: no epoch, mean anomaly or time of perihelion in the "
                    "catalogue, or outside the ephemeris".format(m['name']))
                continue

            print("{:35s}, {:.4f}, {:.4f}, {:6.2f}, {:7.3f}, {:7.3f}, {:>10s}, ".format(m['name'],
                p['q'], p['e'], p['incl'], p['peri'], p['node'], _jdToString(m['epoch_jd'])) \
                + ", ".join("{:{:d}.{:d}f}".format(v, w, d) if np.isfinite(v) else "{:>{:d}s}".format("n/a", w) \
                for v, w, d in zip(values, widths, decimals)))


    # Save the matches
    if cml_args.save is not None:

        meteor = {'q': q, 'e': e, 'incl_deg': incl, 'peri_deg': peri, 'node_deg': node,
            'reference_epoch_jd': jd_ref, 'trajectory_pickle': cml_args.traj_path}

        if search['query_geocentric']:
            ra_g, dec_g, sol_g, vg_g = search['query_geocentric'][0]
            meteor.update({'ra_g_deg': np.degrees(ra_g), 'dec_g_deg': np.degrees(dec_g),
                'solar_longitude_deg': np.degrees(sol_g), 'v_g_kms': vg_g})

        if search['query_heliocentric']:
            l_h, b_h, sol_h, v_h = search['query_heliocentric'][0]
            meteor.update({'L_h_deg': np.degrees(l_h), 'B_h_deg': np.degrees(b_h),
                'solar_longitude_deg': np.degrees(sol_h), 'v_h_kms': v_h})

        written = saveMatches(cml_args.save, search, meteor=meteor, propagated=propagated,
            at_epoch_values=at_epoch_values, chance=chance, ranked=ranked)

        print()
        print("Saved the matches to", cml_args.save)

        if len(written) > 1:
            print("Saved the most significant bodies to", written[1])


    # With more than one criterion, plot them side by side by default
    if (len(labels) > 1) and (not cml_args.no_plot) and matches:

        plot_path = cml_args.plot_path

        if plot_path is None:

            stem = "parent_bodies_{:s}".format("_".join(search['keys']))

            if cml_args.traj_path is not None:
                dir_path, file_name = os.path.split(cml_args.traj_path)
                plot_path = os.path.join(dir_path, os.path.splitext(file_name)[0] + "_" + stem + ".png")

            else:
                plot_path = stem + ".png"

        plotCriteriaBars(search, plot_path, title="Parent body candidates, ranked by {:s}".format(labels[0]))

        print("Saved the bar plot to", plot_path)


    # ##########################################################################################################

    # ### Plot orbits of all JFCs
    # from wmpl.Utils.PlotOrbits import plotOrbits
    # import datetime

    # # Get orbital elements of comets
    # orbit_list = []

    # count = 0

    # for entry in comets_elements:

    #     comet_name, q, e, incl, peri, node = entry

    #     if e >= 1.0:
    #         continue

    #     a = q/(1.0 - e)


    #     # Take only short period comets
    #     if a > 34:
    #         continue

    #     orbit_list.append([a, e, incl, peri, node])

    #     count += 1

    #     # Limit the number of plotted objects
    #     if count > 4000:
    #         break

    # orbit_list = np.array(orbit_list)

    # orb_time = datetime.datetime.now()

    # plt = plotOrbits(orbit_list, orb_time, linewidth=0.2, color_scheme='dark')

    # plt.show()


    # ##########################################################################################################

    # ### Plot orbits of all KBOs
    # from wmpl.Utils.PlotOrbits import plotOrbits
    # import datetime

    # # Load a list of KBOs
    # #file_path = "C:\Users\delorayn1\Desktop\distant_extended.dat"
    # file_path = "/home/dvida/Desktop/distant_extended.dat"
    # with open(file_path) as f:
    #     kbos_elements = []

    #     for line in f:
    #         line = line.split()

    #         peri = float(line[5])
    #         node = float(line[6])
    #         incl = float(line[7])
    #         e = float(line[8])

    #         try:
    #             a = float(line[10])

    #         except:
    #             continue

    #         kbos_elements.append([a, e, incl, peri, node])
    


    # # Get orbital elements of KBOs
    # orbit_list = []

    # count = 0

    # for entry in kbos_elements:

    #     a, e, incl, peri, node = entry

    #     if e >= 1.0:
    #         continue


    #     # Take only TNOs
    #     if (a < 30) or (a > 55):
    #         continue

    #     orbit_list.append([a, e, incl, peri, node])

    #     count += 1

    #     # Limit the number of plotted objects
    #     if count > 5000:
    #         break

    # orbit_list = np.array(orbit_list)

    # orb_time = datetime.datetime.now()

    # # Plot KBOs
    # plt = plotOrbits(orbit_list, orb_time, linewidth=0.1, color_scheme='dark', figsize=(20, 20))




    # # Get orbital elements of Centaurs
    # orbit_list = []

    # count = 0

    # for entry in kbos_elements:

    #     a, e, incl, peri, node = entry

    #     if e >= 1.0:
    #         continue


    #     # Take only Centaurs
    #     if (a > 30):
    #         continue

    #     orbit_list.append([a, e, incl, peri, node])

    #     count += 1

    #     # Limit the number of plotted objects
    #     if count > 5000:
    #         break


    # orbit_list = np.array(orbit_list)

    # orb_time = datetime.datetime.now()

    # # Plot Centaurs
    # plt = plotOrbits(orbit_list, orb_time, linewidth=0.4, color_scheme='dark', orbit_colors=['#2072e3']*len(orbit_list), plt_handle=plt)



    # # Get orbital elements of comets
    # orbit_list = []

    # count = 0

    # for entry in comets_elements:

    #     comet_name, q, e, incl, peri, node = entry

    #     if e >= 1.0:
    #         continue

    #     a = q/(1.0 - e)


    #     # Take only short period comets
    #     if a > 34:
    #         continue

    #     orbit_list.append([a, e, incl, peri, node])

    #     count += 1

    #     # Limit the number of plotted objects
    #     if count > 4000:
    #         break

    # orbit_list = np.array(orbit_list)

    # orb_time = datetime.datetime.now()

    # plt = plotOrbits(orbit_list, orb_time, linewidth=0.05, color_scheme='dark', orbit_colors=['r']*len(orbit_list), plt_handle=plt)




    # plt.show()
