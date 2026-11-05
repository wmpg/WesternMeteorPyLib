""" Clones of a meteor's orbit within the uncertainty of its trajectory, and how the parent body search
(wmpl.Utils.ParentBodySearch) and the orbital classification (wmpl.Utils.OrbitClassification) come out
over them. Both command lines take --clones N [--seed S] for this.

The clones are drawn from the covariance of the trajectory's state vector, traj.state_vect_cov, which
the trajectory solver computes from its Monte Carlo runs, exactly as wmpl.Rebound.REBOUND draws its Monte
Carlo realizations: with the same seed, clone k here is clone k of REBOUND.py --mc N --seed S, and these
are the orbits of those clones at the epoch of the meteor. Each state vector is turned into an orbit by
wmpl.Trajectory.Orbit.calcOrbit with the convention the solver used for the nominal orbit (moving or
fixed stations, see Trajectory.orbit_los_success), so that the nominal state gives the orbit of the
pickle again. A trajectory saved without its Monte Carlo uncertainties has no covariance and no clones.

What the clones say, and what they do not:
- They carry the measurement uncertainty of the trajectory, nothing else. For a well-measured fireball
  they change nothing: Novo Mesto (sigma of the initial speed 14 m/s), 500 clones, keeps the same best
  match, 2005 OX, in every clone, its D_SH within 0.0781-0.0790, and the same classes. They matter for
  poorly constrained orbits: with its covariance scaled by 10 (initial speed to 140 m/s, the radiant
  alike) the share of the clones under D_SH = 0.10 of the second candidate falls from 77% to 52%, and
  scaled by 30 (410 m/s) the best match changes in 29% of the clones.
- The values to quote are those of the nominal orbit. A D criterion is a distance between two orbits,
  and noise moves it away on average: the clones' median lies above the nominal value (0.0785 against
  0.0854 for that best match at ~410 m/s) and is not an estimate of it. The clones give an interval
  around the nominal value, the share of them under a threshold, and how often a body stays the best
  match.
- The p-values of the parent body search (--chance_neo, --chance_meteoroid) are those of the nominal
  orbit; the clones do not change them.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Utils.Dcriteria import DERIVED_FROM_ELEMENTS, lookupCriterion, radiantCriterionMin
from wmpl.Utils.OrbitClassification import (calcTisserand, calcWhippleK, calcKresakP,
    calcAphelionDistance, calcOrbitalEnergy, classifyNEO, isArjuna, classifyBorovicka, isProgrograde,
    isRetrograde, isEarthCrossing, isSunApproaching, SUN_APPROACHING_Q_LIMIT, isCometaryQi,
    isCometaryEi, isCometaryTi, isCometaryPi, isCometaryKi, classifyTancrediComet,
    calcGiantPlanetMOIDs, classifyTancrediAsteroid, GIANT_PLANETS)


# The interval the clones are summarised by, in percent: 95% of them
CLONE_INTERVAL = (2.5, 97.5)

# A recomputed nominal orbit further than this many sigma from the pickle's is reported: the pickle was
#   solved with another version of the orbit code, and the clones, computed now, are off-centre
NOMINAL_OFFSET_WARNING_SIGMA = 0.5

# The elements the spread of the clones is given for, as (key, label, unit, is an angle)
SPREAD_ELEMENTS = [
    ('q', 'q', ' AU', False),
    ('e', 'e', '', False),
    ('incl', 'i', ' deg', True),
    ('peri', 'peri', ' deg', True),
    ('node', 'node', ' deg', True),
    ]

# The names of the same elements in the solver's uncertainties (traj.uncertainties)
_SOLVER_NAMES = {'q': 'q', 'e': 'e', 'incl': 'i', 'peri': 'peri', 'node': 'node'}

NO_COVARIANCE_MESSAGE = ("The trajectory has no Monte Carlo uncertainties (traj.state_vect_cov), so it has "
    "no clones. Solve it again with the Monte Carlo uncertainties.")


def cloneStateVectors(traj, n_clones, random_seed=None):
    """ Draw clones of the trajectory's state vector, [x, y, z, vx, vy, vz] (traj.state_vect_mini and
        traj.v_init*traj.radiant_eci_mini, ECI, m and m/s), from its covariance traj.state_vect_cov.

        The draws are those of wmpl.Rebound.REBOUND: a generator seeded with random_seed, and one
        multivariate normal draw per clone, in order. Clone k therefore depends only on the seed and on k,
        not on how many clones are drawn, and is clone k of REBOUND.py --mc N --seed random_seed.

    Arguments:
        traj: [Trajectory] A trajectory solved with the Monte Carlo uncertainties.
        n_clones: [int] How many clones to draw.

    Keyword arguments:
        random_seed: [int] Seed of the generator.

    Return:
        [ndarray] The state vectors, one row per clone.
    """

    if (getattr(traj, 'uncertainties', None) is None) or (getattr(traj, 'state_vect_cov', None) is None):
        raise ValueError(NO_COVARIANCE_MESSAGE)

    state_vect = np.concatenate([traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini])
    rng = np.random.default_rng(random_seed)

    return np.array([rng.multivariate_normal(state_vect, traj.state_vect_cov) for _ in range(n_clones)])


def orbitOfStateVector(traj, state_vect):
    """ The orbit of a state vector of the trajectory, computed as the solver computed the nominal orbit.

        The solver gives calcOrbit the radiant, the initial speed, the average speed, the state vector
        position and the time of the beginning of the trajectory, with moving stations if the line of
        sight fit converged and fixed ones otherwise (Trajectory.orbit_los_success; a pickle from before
        that attribute was stored is taken to have converged, as MetSim's saveUpdatedOrbit does). A
        clone's average speed is shifted by the change of its initial speed, as there.

    Arguments:
        traj: [Trajectory] The trajectory.
        state_vect: [ndarray] [x, y, z, vx, vy, vz], ECI, m and m/s, as cloneStateVectors draws them.

    Return:
        [Orbit] The orbit.
    """

    los_success = getattr(traj, 'orbit_los_success', True)

    position = np.asarray(state_vect[:3], dtype=float)
    velocity = np.asarray(state_vect[3:], dtype=float)
    v_init = np.sqrt(velocity @ velocity)

    return calcOrbit(velocity/v_init, v_init, traj.v_avg + (v_init - traj.v_init), position, traj.rbeg_jd,
        stations_fixed=(not los_success), reference_init=los_success,
        v_init_stddev_direct=getattr(traj, 'v_init_stddev', None))


def _orbitRow(orbit):
    """ What the parent body search and the classification read from an orbit: the elements (q, e,
        incl, node, peri, angles in radians), the geocentric radiant (ra, dec, solar longitude, speed in
        km/s) and the corrected heliocentric direction (Lh, Bh, solar longitude, speed in km/s), the last
        two None where the orbit lacks them, as ParentBodySearch takes them from a pickle. """

    elements = tuple(np.nan if v is None else float(v) for v in (orbit.q, orbit.e, orbit.i, orbit.node,
        orbit.peri))

    geocentric = None
    if all(v is not None for v in (orbit.ra_g, orbit.dec_g, orbit.la_sun, orbit.v_g)):
        geocentric = (orbit.ra_g, orbit.dec_g, orbit.la_sun, orbit.v_g/1000.0)

    heliocentric = None
    if all(getattr(orbit, name, None) is not None for name in ('L_h', 'B_h', 'la_sun', 'v_h')):
        heliocentric = (orbit.L_h, orbit.B_h, orbit.la_sun, orbit.v_h/1000.0)

    return elements, geocentric, heliocentric


def _wrappedDifference(angles, reference):
    """ Differences of angles from a reference, in radians, within -pi to pi. """

    return (np.asarray(angles) - reference + np.pi) % (2*np.pi) - np.pi


def cloneOrbits(traj, n_clones, random_seed=None):
    """ The orbits of clones of a trajectory, at the epoch of the meteor (see cloneStateVectors and
        orbitOfStateVector).

    Arguments:
        traj: [Trajectory] A trajectory solved with the Monte Carlo uncertainties.
        n_clones: [int] How many clones, at least 2.

    Keyword arguments:
        random_seed: [int] Seed of the draws. If None, one is drawn, as REBOUND.py does without --seed,
            and returned, so that the clones can be drawn again.

    Return:
        [dict] 'n' and 'seed'; 'states', the state vectors; 'q', 'e', 'incl', 'node', 'peri', arrays of
            the elements (radians), NaN for a clone without an orbit; 'geocentric' and 'heliocentric',
            lists with the radiant and the heliocentric direction of each clone or None; 'valid', the
            clones with an orbit; 'spread', the standard deviation of each element over them (the angles
            in radians, about the nominal), and 'solver_sigma', the solver's uncertainty of each, for
            comparison; 'nominal_offset_sigma', how far the orbit computed now from the nominal state lies
            from the pickle's, in units of the solver's sigma, the largest over the elements.
    """

    if n_clones < 2:
        raise ValueError("At least 2 clones are needed, {:d} were asked for.".format(n_clones))

    if random_seed is None:
        random_seed = int(np.random.SeedSequence().entropy % (2**32))

    states = cloneStateVectors(traj, n_clones, random_seed=random_seed)

    elements = []
    geocentric = []
    heliocentric = []

    for state_vect in states:

        row, geo, helio = _orbitRow(orbitOfStateVector(traj, state_vect))

        elements.append(row)
        geocentric.append(geo)
        heliocentric.append(helio)

    elements = np.array(elements, dtype=float)
    valid = np.all(np.isfinite(elements), axis=1)

    out = {'n': n_clones, 'seed': random_seed, 'states': states, 'valid': valid,
        'geocentric': geocentric, 'heliocentric': heliocentric}

    for k, key in enumerate(('q', 'e', 'incl', 'node', 'peri')):
        out[key] = elements[:, k]


    # The spread of the clones, and the solver's for comparison
    nominal = {'q': traj.orbit.q, 'e': traj.orbit.e, 'incl': traj.orbit.i, 'node': traj.orbit.node,
        'peri': traj.orbit.peri}

    out['spread'] = {}
    out['solver_sigma'] = {}

    for key, _, _, angle in SPREAD_ELEMENTS:

        values = out[key][valid]
        if angle:
            values = _wrappedDifference(values, nominal[key])

        out['spread'][key] = float(np.std(values)) if len(values) else None

        sigma = getattr(traj.uncertainties, _SOLVER_NAMES[key], None)
        out['solver_sigma'][key] = None if sigma is None else float(sigma)


    # The orbit of the nominal state, computed now, against the pickle's
    recomputed, _, _ = _orbitRow(orbitOfStateVector(traj, np.concatenate([traj.state_vect_mini,
        traj.v_init*traj.radiant_eci_mini])))

    offsets = []
    for k, key in enumerate(('q', 'e', 'incl', 'node', 'peri')):

        sigma = out['solver_sigma'][key]
        if (nominal[key] is None) or (not sigma):
            continue

        diff = recomputed[k] - nominal[key]
        if key in ('incl', 'node', 'peri'):
            diff = _wrappedDifference(recomputed[k], nominal[key])

        offsets.append(abs(diff)/sigma)

    out['nominal_offset_sigma'] = float(max(offsets)) if offsets else None

    return out


def cloneHeaderLines(clones):
    """ Lines that say which clones these are, how many have an orbit, and how they spread compared with
        the solver's uncertainties. """

    lines = [
        "{:d} clones of the trajectory's state vector, drawn from its covariance with seed {:d}".format(
            clones['n'], clones['seed']),
        "(wmpl.Rebound.REBOUND --mc {:d} --seed {:d} integrates the same clones).".format(clones['n'],
            clones['seed']),
        ]

    n_invalid = int(np.sum(~clones['valid']))
    if n_invalid:
        lines.append("{:d} clones have no orbit and are left out.".format(n_invalid))

    parts = []
    for key, label, unit, angle in SPREAD_ELEMENTS:

        spread = clones['spread'][key]
        if spread is None:
            continue

        sigma = clones['solver_sigma'][key]
        scale = np.degrees(1.0) if angle else 1.0

        parts.append("{:s} {:.2g}{:s}".format(label, spread*scale, unit) + ("" if sigma is None \
            else " ({:.2g})".format(sigma*scale)))

    lines += ["1 sigma of the clones' orbits (the solver's in brackets):", "  " + ", ".join(parts)]

    offset = clones['nominal_offset_sigma']
    if (offset is not None) and (offset > NOMINAL_OFFSET_WARNING_SIGMA):
        lines.append("WARNING: the orbit computed now from the nominal state vector lies {:.1f} sigma from "
            "the pickle's.".format(offset))
        lines.append("The pickle was solved with another version of the orbit code; the clones, computed "
            "now, are not centred on its orbit.")

    return lines


##############################################################################################################
### PARENT BODY SEARCH ###


def _bodyOrbits(bodies):
    """ The elements of the bodies of a catalogue (lists as the loaders of wmpl.Utils.ParentBodySearch
        give them: name, q, e, incl, peri, node in degrees, ...) as arrays (q, e, incl, node, peri), the
        angles in radians, the order of the criteria's orbits. """

    elements = np.array([body[1:6] for body in bodies], dtype=float).reshape(-1, 5)

    return (elements[:, 0], elements[:, 1], np.radians(elements[:, 2]), np.radians(elements[:, 4]),
        np.radians(elements[:, 3]))


def _matchIndices(matches, bodies):
    """ The place in the catalogue of each match of a search, found by name and elements. """

    places = {}
    for k, body in enumerate(bodies):
        places.setdefault(body[0], []).append(k)

    indices = []
    for match in matches:

        candidates = places.get(match['name'], [])
        same = [k for k in candidates if (bodies[k][1] == match['q']) and (bodies[k][2] == match['e'])]

        if not (same or candidates):
            raise ValueError("{:s} is not in the catalogue that was searched.".format(match['name']))

        indices.append((same or candidates)[0])

    return np.array(indices, dtype=int)


def _criterionValues(criterion, orbit, radiants, body_orbits, body_radiants, index):
    """ Values of one criterion between an orbit and the bodies of a catalogue at the places index.

        A criterion on the elements is evaluated over all of them in one call, which every one of them
        allows (see wmpl.Utils.Dcriteria.ELEMENT_CRITERIA). One on the radiant takes the smallest value
        over the radiants of the two orbits, as the search does (radiantCriterionMin), infinite for a body
        without a radiant.
    """

    _, _, func, kind = criterion

    if kind == 'elements':
        values = func(orbit, tuple(x[index] for x in body_orbits))
        return np.broadcast_to(np.asarray(values, dtype=float), (len(index),)).copy()

    return np.array([radiantCriterionMin(func, radiants[kind], body_radiants[kind][b]) for b in index],
        dtype=float)


def cloneSearch(search, bodies, clones, dmax=None, progress=None):
    """ How the matches of a parent body search come out over the clones of the meteor's orbit.

        For every clone, each criterion of the search is computed with each match, and the first one,
        which ranks the bodies, with every body of the catalogue, which gives the place of each match
        among them and the body that is the closest. A clone's orbit is compared as the search compares
        the nominal one: with its own radiant for the criteria on the radiant. The catalogue's radiants
        are derived once.

    Arguments:
        search: [dict] Return of wmpl.Utils.ParentBodySearch.searchParentBodies for the nominal orbit,
            with the matches to follow (the top N).
        bodies: [list] The catalogue that was searched, as its loaders give it.
        clones: [dict] Return of cloneOrbits.

    Keyword arguments:
        dmax: [list] A threshold for each criterion, in order, or None; also shorter than the criteria,
            None for those left. The share of the clones under it is given.
        progress: [function] Called as progress(k, n) after each clone, to report the progress.

    Return:
        [dict] 'labels' and 'keys' of the criteria; 'n' (clones with an orbit) and 'seed'; 'values',
            an array (criterion, clone, match), NaN for a clone without an orbit; 'interval', an array
            (criterion, match, 2) with the 95% interval of the finite values; 'n_finite', (criterion,
            match), how many clones have a finite value; 'dmax' and 'below', (criterion, match) the share
            of the clones under the threshold, NaN without one; 'ranks', (clone, match), the place among
            all the bodies by the first criterion, 1 for the closest, NaN where the value is not finite;
            'rank_interval', (match, 3) with the median and the 95% interval; 'best_fraction', (match,)
            the share of the clones for which the match is the closest body; 'best_bodies', [(name,
            share), ...] every body that is the closest for some clone, the most frequent first; and
            'no_best', the share of clones with no finite value for any body; and 'spread', 'solver_sigma'
            and 'nominal_offset_sigma' of the clones (see cloneOrbits).
    """

    criteria = [lookupCriterion(key) for key in search['keys']]
    matches = search['matches']

    n_criteria = len(criteria)
    n_matches = len(matches)

    dmax = list(dmax or [])
    if len(dmax) > n_criteria:
        raise ValueError("{:d} thresholds were given for {:d} criteria.".format(len(dmax), n_criteria))
    dmax = dmax + [None]*(n_criteria - len(dmax))

    match_index = _matchIndices(matches, bodies)
    all_index = np.arange(len(bodies))
    body_orbits = _bodyOrbits(bodies)


    # The radiants of the bodies, derived from their elements once: of all of them if the first criterion
    #   is on the radiant, otherwise only of the matches
    body_radiants = {}

    for c, (_, _, _, kind) in enumerate(criteria):

        if kind == 'elements':
            continue

        needed = all_index if (c == 0) else match_index
        known = body_radiants.setdefault(kind, {})

        for b in needed:
            if b not in known:
                known[b] = DERIVED_FROM_ELEMENTS[kind](*(x[b] for x in body_orbits))


    valid = np.flatnonzero(clones['valid'])
    n_clones = len(clones['valid'])

    values = np.full((n_criteria, n_clones, n_matches), np.nan)
    ranks = np.full((n_clones, n_matches), np.nan)
    best = np.full(n_clones, -1, dtype=int)

    for done, k in enumerate(valid):

        orbit = tuple(clones[key][k] for key in ('q', 'e', 'incl', 'node', 'peri'))

        # The clone's own radiant, as the search uses the nominal one of a pickle, or derived from its
        #   elements if it has none
        radiants = {}
        for kind in body_radiants:
            given = clones[kind][k]
            radiants[kind] = [tuple(given)] if given is not None else DERIVED_FROM_ELEMENTS[kind](*orbit)

        # The ranking criterion against the whole catalogue
        first = _criterionValues(criteria[0], orbit, radiants, body_orbits, body_radiants, all_index)
        values[0, k] = first[match_index]

        finite = np.isfinite(first)
        if finite.any():

            best[k] = int(np.argmin(np.where(finite, first, np.inf)))

            finite_values = np.sort(first[finite])
            for m, b in enumerate(match_index):
                if finite[b]:
                    ranks[k, m] = 1 + np.searchsorted(finite_values, first[b], side='left')

        for c in range(1, n_criteria):
            values[c, k] = _criterionValues(criteria[c], orbit, radiants, body_orbits, body_radiants,
                match_index)

        if progress is not None:
            progress(done + 1, len(valid))


    # Summaries over the clones with an orbit
    n_valid = len(valid)
    interval = np.full((n_criteria, n_matches, 2), np.nan)
    n_finite = np.zeros((n_criteria, n_matches), dtype=int)
    below = np.full((n_criteria, n_matches), np.nan)

    for c in range(n_criteria):
        for m in range(n_matches):

            v = values[c, valid, m]
            f = np.isfinite(v)
            n_finite[c, m] = int(f.sum())

            if f.any():
                interval[c, m] = np.percentile(v[f], CLONE_INTERVAL)

            if (dmax[c] is not None) and n_valid:
                below[c, m] = float(np.mean(np.where(f, v, np.inf) < dmax[c]))

    rank_interval = np.full((n_matches, 3), np.nan)
    for m in range(n_matches):
        r = ranks[valid, m]
        r = r[np.isfinite(r)]
        if len(r):
            rank_interval[m] = np.percentile(r, (50,) + CLONE_INTERVAL)

    best_valid = best[valid]
    best_fraction = np.array([np.mean(best_valid == b) if n_valid else np.nan for b in match_index])

    counts = {}
    for b in best_valid[best_valid >= 0]:
        counts[b] = counts.get(b, 0) + 1

    best_bodies = [(bodies[b][0], n/n_valid) for b, n in sorted(counts.items(), key=lambda kv: -kv[1])]

    return {'labels': search['labels'], 'keys': search['keys'], 'n': n_valid, 'n_drawn': n_clones,
        'seed': clones['seed'], 'values': values, 'interval': interval, 'n_finite': n_finite,
        'dmax': dmax, 'below': below, 'ranks': ranks, 'rank_interval': rank_interval,
        'best_fraction': best_fraction, 'best_bodies': best_bodies,
        'no_best': float(np.mean(best_valid < 0)) if n_valid else np.nan,
        'spread': clones['spread'], 'solver_sigma': clones['solver_sigma'],
        'nominal_offset_sigma': clones['nominal_offset_sigma']}


def _formatValue(value, decimals):
    """ A value of a criterion, or n/a. """

    return "{:.{:d}f}".format(value, decimals) if np.isfinite(value) else "n/a"


def formatCloneSearch(stats, search, decimals=None, max_best=5):
    """ Lines that report cloneSearch: the bodies that are the closest for the clones, and for each
        match its nominal value of every criterion with the 95% interval of the clones, the share of the
        clones under each threshold given, how often it is the closest body and its place among them.

    Arguments:
        stats: [dict] Return of cloneSearch.
        search: [dict] The search of the nominal orbit it was computed for.

    Keyword arguments:
        decimals: [list] Decimals of each criterion. 4 by default.
        max_best: [int] How many of the closest bodies to name.

    Return:
        [list] The lines.
    """

    labels = stats['labels']
    decimals = decimals or [4]*len(labels)

    lines = []

    if stats['best_bodies']:

        named = stats['best_bodies'][:max_best]
        others = sum(share for _, share in stats['best_bodies'][max_best:])

        lines.append("Closest body by {:s}, share of the clones: ".format(labels[0]) + ", ".join(
            "{:s} {:.1f}%".format(name, 100*share) for name, share in named) + ("" if others <= 0 \
            else ", {:d} others {:.1f}%".format(len(stats['best_bodies']) - len(named), 100*others)))

    if stats['no_best'] > 0:
        lines.append("{:.1f}% of the clones have no value of {:s} with any body.".format(
            100*stats['no_best'], labels[0]))

    # The columns: every criterion, nominal [95% of the clones], then its threshold if one was given
    columns = []
    for c, label in enumerate(labels):

        columns.append("{:s} [95% of the clones]".format(label))

        if stats['dmax'][c] is not None:
            columns.append("{:s} < {:g}".format(label, stats['dmax'][c]))

    columns += ["closest", "rank [95%]"]

    cells = []
    for m, match in enumerate(search['matches']):

        row = []
        for c in range(len(labels)):

            lo, hi = stats['interval'][c, m]
            row.append("{:s} [{:s}, {:s}]".format(_formatValue(match['d_values'][c], decimals[c]),
                _formatValue(lo, decimals[c]), _formatValue(hi, decimals[c])))

            if stats['dmax'][c] is not None:
                row.append("{:.1f}%".format(100*stats['below'][c, m]))

        row.append("{:.1f}%".format(100*stats['best_fraction'][m]))

        median, lo, hi = stats['rank_interval'][m]
        row.append("n/a" if not np.isfinite(median) else "{:.0f} [{:.0f}, {:.0f}]".format(median, lo, hi))

        cells.append(row)

    widths = [max([len(col)] + [len(row[j]) for row in cells]) for j, col in enumerate(columns)]

    lines.append('{:35s}, '.format('Name') + ', '.join('{:>{:d}s}'.format(col, w) for col, w in zip(columns,
        widths)))

    for match, row in zip(search['matches'], cells):
        lines.append('{:35s}, '.format(match['name']) + ', '.join('{:>{:d}s}'.format(cell, w)
            for cell, w in zip(row, widths)))

    lines += [
        "The values are the nominal orbit's, the ones to quote, with 95% of the clones in brackets. "
        "'closest' is the",
        "share of the clones for which the body is the closest of the catalogue by {:s}, and 'rank' its "
        "place among".format(labels[0]),
        "all the bodies over the clones. A criterion is a distance between orbits, so the clones' noise "
        "moves it away",
        "on average: their median lies above the nominal value and is no estimate of it. The p-values, if "
        "asked for,",
        "are the nominal orbit's.",
        ]

    return lines


def cloneSearchColumns(stats, m):
    """ The values of cloneSearch for match m as (name, value) pairs, for the files the search saves. """

    columns = []

    for c, label in enumerate(stats['labels']):

        lo, hi = stats['interval'][c, m]
        columns += [('clones_' + label + '_2.5', _finiteOrNone(lo)), ('clones_' + label + '_97.5',
            _finiteOrNone(hi))]

        if stats['dmax'][c] is not None:
            columns.append(('clones_' + label + '_below_dmax', _finiteOrNone(stats['below'][c, m])))

    median, lo, hi = stats['rank_interval'][m]

    columns += [('clones_closest_fraction', _finiteOrNone(stats['best_fraction'][m])),
        ('clones_rank_median', _finiteOrNone(median)), ('clones_rank_2.5', _finiteOrNone(lo)),
        ('clones_rank_97.5', _finiteOrNone(hi))]

    return columns


def cloneSearchMatch(stats, m):
    """ The values of cloneSearch for match m, for the JSON file the search saves. """

    criteria = {}
    for c, label in enumerate(stats['labels']):

        lo, hi = stats['interval'][c, m]
        criteria[label] = {'interval_95': [_finiteOrNone(lo), _finiteOrNone(hi)],
            'n_finite': int(stats['n_finite'][c, m])}

        if stats['dmax'][c] is not None:
            criteria[label]['below_dmax'] = _finiteOrNone(stats['below'][c, m])

    median, lo, hi = stats['rank_interval'][m]

    return {'criteria': criteria, 'closest_fraction': _finiteOrNone(stats['best_fraction'][m]),
        'rank': {'median': _finiteOrNone(median), 'interval_95': [_finiteOrNone(lo), _finiteOrNone(hi)]}}


def cloneSearchSummary(stats):
    """ What the JSON file of the search holds about the clones as a whole: how many, the seed, the
        thresholds, the closest bodies and the spread of the orbits. """

    return {'n': stats['n_drawn'], 'n_with_orbit': stats['n'], 'seed': stats['seed'],
        'interval_percent': list(CLONE_INTERVAL), 'ranked_by': stats['labels'][0],
        'dmax': dict((label, d) for label, d in zip(stats['labels'], stats['dmax'])),
        'closest': [{'name': name, 'fraction': share} for name, share in stats['best_bodies']],
        'no_closest_fraction': _finiteOrNone(stats['no_best']),
        'spread': _elementsWithUnits(stats['spread']), 'solver_sigma': _elementsWithUnits(stats['solver_sigma']),
        'nominal_offset_sigma': stats['nominal_offset_sigma']}


def _elementsWithUnits(values):
    """ A spread of the elements with the names and units of the saved files: q_AU, e, and the angles in
        degrees, incl_deg, peri_deg, node_deg. """

    out = {}
    for key, _, unit, angle in SPREAD_ELEMENTS:

        value = values.get(key)
        name = key + ('_deg' if angle else ('_AU' if unit else ''))

        out[name] = None if value is None else float(np.degrees(value) if angle else value)

    return out


def _finiteOrNone(value):
    """ A float, or None where it is not finite, which JSON does not take. """

    return float(value) if (value is not None) and np.isfinite(value) else None


##############################################################################################################
### ORBITAL CLASSIFICATION ###


# The sections of the classification, in the order and with the titles of the command line of
#   wmpl.Utils.OrbitClassification
CLASSIFICATION_SECTIONS = [
    "Dynamical parameters",
    "Orbital class",
    "Jopek & Williams (2013) two-parameter classification",
    "Tancredi (2014) classification",
    ]


def classifyOrbit(q, e, incl, node=None, peri=None):
    """ Everything the command line of wmpl.Utils.OrbitClassification prints for a bound orbit, with the
        keys of its --quiet output, so that the classes of the clones can be put beside those of the
        nominal orbit. The functions are those of wmpl.Utils.OrbitClassification.

    Arguments:
        q: [float] Perihelion distance (AU).
        e: [float] Eccentricity, under 1.
        incl: [float] Inclination (radians).

    Keyword arguments:
        node, peri: [float] Ascending node and argument of perihelion (radians). Without them the MOIDs
            with the giant planets and the Tancredi class of an asteroid are left out, as there.

    Return:
        [list] (section, key, label, value) for each quantity, in the order printed there; a value is a
            float or, for a class, a string.
    """

    if e >= 1.0:
        raise ValueError("A dynamical class is defined only for a bound orbit, e = {:.5f}.".format(e))

    a = q/(1.0 - e)

    direction = "prograde" if isProgrograde(incl) else ("retrograde" if isRetrograde(incl) else "polar")

    dynamical, orbital, cometary, tancredi = CLASSIFICATION_SECTIONS

    out = [
        (dynamical, 'T_J', 'Tisserand parameter T_J', float(calcTisserand(a, e, incl))),
        (dynamical, 'K', 'Whipple K', float(calcWhippleK(a, e))),
        (dynamical, 'P', 'Kresak P (yr)', float(calcKresakP(a, e))),
        (dynamical, 'Q', 'Aphelion Q (AU)', float(calcAphelionDistance(a, e))),
        (dynamical, 'E', 'Orbital energy E (AU^2/day^2)', float(calcOrbitalEnergy(a))),
        (orbital, 'NEO_CLASS', "near-Earth class", classifyNEO(q, e)),
        (orbital, 'ARJUNA', "Arjuna orbital domain", "yes" if bool(isArjuna(q, e, incl)) else "no"),
        (orbital, 'BOROVICKA_CLASS', "Borovicka et al. (2005) class", classifyBorovicka(q, e, incl)),
        (orbital, 'DIRECTION', "direction", direction),
        (orbital, 'EARTH_CROSSING', "Earth crossing", "yes" if bool(isEarthCrossing(q, e)) else "no"),
        (orbital, 'SUN_APPROACHING', "Sun approaching (q < {:.1f} AU)".format(SUN_APPROACHING_Q_LIMIT),
            "yes" if bool(isSunApproaching(q)) else "no"),
        ]

    for label, test in (("Q-i", isCometaryQi), ("E-i", isCometaryEi), ("T-i", isCometaryTi),
                        ("P-i", isCometaryPi), ("K-i", isCometaryKi)):
        out.append((cometary, label, label, "cometary" if bool(test(a, e, incl)) else "asteroidal"))

    out.append((tancredi, 'TANCREDI_COMET', "comet class", classifyTancrediComet(a, e, incl)))

    if (node is not None) and (peri is not None):

        moids = calcGiantPlanetMOIDs(a, e, incl, node, peri)

        for name, _, _, _, _, _, _ in GIANT_PLANETS:
            out.append((tancredi, 'MOID_' + name.upper(), "MOID " + name.capitalize() + " (Hill radii)",
                float(moids[name])))

        out.append((tancredi, 'TANCREDI_ASTEROID', "asteroid class", classifyTancrediAsteroid(a, e, incl,
            moids['jupiter'], min(moids.values()))))

    return out


# The class given to an unbound clone, which has none
UNBOUND = "unbound"


def cloneClasses(clones, with_orientation=True):
    """ The classification of every clone with an orbit (see classifyOrbit), summarised: for a number, the
        95% interval of the bound clones, and for a class, the share of the clones in each, the unbound
        ones in a class of their own.

    Arguments:
        clones: [dict] Return of cloneOrbits.

    Keyword arguments:
        with_orientation: [bool] Compute the MOIDs with the giant planets and the Tancredi class of an
            asteroid, which need the node and the argument of perihelion.

    Return:
        [dict] 'n' (clones with an orbit), 'n_unbound', 'seed', and 'items', a list of (section, key,
            label, summary), where summary is {'interval': [lo, hi]} for a number (None without bound
            clones) and {'fractions': [(class, share), ...]} for a class, the most frequent first.
    """

    valid = np.flatnonzero(clones['valid'])
    bound = [k for k in valid if clones['e'][k] < 1.0]

    results = [classifyOrbit(clones['q'][k], clones['e'][k], clones['incl'][k],
        clones['node'][k] if with_orientation else None, clones['peri'][k] if with_orientation else None)
        for k in bound]

    # The items, in order, from a bound orbit's; a clone set without one has no numbers
    template = results[0] if results else []

    items = []
    for j, (section, key, label, value) in enumerate(template):

        values = [r[j][3] for r in results]

        if isinstance(value, str):

            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1

            if len(valid) > len(bound):
                counts[UNBOUND] = len(valid) - len(bound)

            summary = {'fractions': [(name, n/len(valid)) for name, n in sorted(counts.items(),
                key=lambda kv: -kv[1])]}

        else:
            v = np.array(values, dtype=float)
            v = v[np.isfinite(v)]
            summary = {'interval': [float(x) for x in np.percentile(v, CLONE_INTERVAL)] if len(v) else None}

        items.append((section, key, label, summary))

    return {'n': len(valid), 'n_unbound': len(valid) - len(bound), 'seed': clones['seed'], 'items': items}


def formatCloneClasses(summary, nominal, quiet=False):
    """ Lines that report cloneClasses beside the nominal orbit's classification.

    Arguments:
        summary: [dict] Return of cloneClasses.
        nominal: [list] Return of classifyOrbit for the nominal orbit.

    Keyword arguments:
        quiet: [bool] One 'KEY VALUE' line per quantity, as the --quiet output of
            wmpl.Utils.OrbitClassification: KEY_CLONES_95 with the interval of a number, KEY_CLONES with
            'class: share' pairs for a class.

    Return:
        [list] The lines.
    """

    nominal_values = dict((key, value) for _, key, _, value in nominal)

    if quiet:

        lines = ["CLONES {:d}".format(summary['n']), "CLONES_SEED {:d}".format(summary['seed']),
            "CLONES_UNBOUND {:d}".format(summary['n_unbound'])]

        for _, key, _, s in summary['items']:

            if 'fractions' in s:
                lines.append("{:s}_CLONES {:s}".format(key, ", ".join("{:s}: {:.4f}".format(name, share)
                    for name, share in s['fractions'])))

            elif s['interval'] is not None:
                lines.append("{:s}_CLONES_95 {:.6g} {:.6g}".format(key, *s['interval']))

        return lines

    lines = ["The nominal value, and the 95% interval of the {:d} clones or the share of them in "
        "each class:".format(summary['n'])]

    if summary['n_unbound']:
        lines.append("{:d} clones are unbound and have no dynamical parameters.".format(
            summary['n_unbound']))

    section = None
    for this_section, key, label, s in summary['items']:

        if this_section != section:
            lines += ["", this_section + ":"]
            section = this_section

        value = nominal_values.get(key)

        if 'fractions' in s:
            lines.append("  {:<30s} {:s}".format(label, ", ".join("{:s} {:.1f}%".format(name, 100*share)
                for name, share in s['fractions'])))

        else:
            fmt = '{:.6e}' if key == 'E' else '{:.6f}'
            interval = "n/a" if s['interval'] is None else "[{:s}, {:s}]".format(fmt.format(s['interval'][0]),
                fmt.format(s['interval'][1]))
            lines.append("  {:<30s} {:>14s}  {:s}".format(label, "n/a" if value is None else fmt.format(value),
                interval))

    return lines
