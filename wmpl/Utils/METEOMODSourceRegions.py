""" The source regions of a meteoroid's orbit in METEOMOD, the orbital distribution model of metre-sized
meteoroids of Broz et al. (2024): the probability that it came from each of 52 asteroid families.

METEOMOD (Broz et al. 2024a, 2024b) follows 1-m bodies released from 52 asteroid families, with the
Yarkovsky and YORP effects and collisional reorientations, from the main belt to the NEO region. For each
family j it gives a map M_j(a, i) of where its bodies are once in the NEO region (e > 1 - 1.3 AU/a), in
bins of 0.01 AU in a, 0 to 3.5 AU, and 1 deg in i, 0 to 90 deg, normalised to 1 over the map. Only a and
i are kept, as the eccentricity of the source is lost by then. The family's contribution is weighted
by its meteoroid flux on the Earth, Phi_j = p_j N_neo,j (Eq. 9 of Broz et al. 2024a), where N_neo is
its number of 1-m bodies in the NEO region from its size distribution and p_j their mean collision
probability with the Earth (Table 5 of Broz et al. 2024a). The probability that a meteoroid at (a, i)
came from family j is then

    P_j = M_j(a, i) Phi_j / sum over the selected families of M_k(a, i) Phi_k

as the METEOMOD page states: "Since we study meteoroids colliding with the Earth, the maps should be
weighted by the flux Phi = pcoll Nneo(>1 m) [...] The probability of coming from a source is then
estimated as pj = Mj(a, i) Phi". The families carry the taxonomy of their bodies, so the families to
compare are those of the meteorite's type (--types L for an L chondrite, as in the examples of the
page).

This module reads the distribution itself, and does not run the authors' meteomod.py, because:
    - meteomod.py weights the maps by N_neo, not by Phi. It is a copy of neomod.py, the authors' tool for
      NEOs, for which N_neo is the right weight, and it reads Phi from Table 5 without using it. For a
      body that hit the Earth this leaves out the collision probability, which differs between the
      families by a factor of 11 (Hygiea 0.70, Nysa 7.8 in 1e-9 km^-2 y^-1 per 1e8 bodies). It matters
      most for the carbonaceous families: with the CM families, Winchcombe is from Veritas (0.30)
      weighted by N_neo, and from Astrid (0.43, Veritas 0.11) weighted by Phi. Of the 47 meteorites of
      wmpl/share/Meteorites_with_Orbits.csv whose type has families, 3 change their most probable family
      (Dishchii'bikoh, Sutter's Mill, Maribo), and the probability of the first family of others moves
      by up to 0.24 (Benesov, Golden, Grimsby). The L and H chondrites of the examples still come from
      Massalia(2) and Koronis(2). weight='neo' gives the script's values, to compare.
    - meteomod.py reads the maps from maps.npz when it is there, a pickled dictionary: loading it would
      run code from a downloaded file. The maps are read here from the text files they were made from
      (map/*/map.out), which give the same maps to the last digit (checked for the 55 of them).
    - meteomod.py does not check the orbit: a < 0 (hyperbolic) indexes the map from its far end without
      a message, a >= 3.5 AU or i >= 90 deg stop it with an IndexError, and an orbit outside the NEO
      region gets the probabilities of its (a, i) bin. Here such orbits get none, with the reason.

Otherwise the values are those of meteomod.py: with weight='neo' they agree within 6e-16 with a
transcription of its compute_p, for 36 626 orbits and selections of families, 1000 of the orbits on the
edges of the bins.

The names and types of the families are those of Table 5 (tab5.tex), as in the paper and on the page:
maps.in, which gives the map of each family, calls Misa CM and Brasilia CH, which Table 5 calls CI and M.
The values of Table 5 are those of tab5.tex in the download, which the authors' page uses. It is dated
after the publication, and differs from the published Table 5 (arXiv:2406.19727v2) for two families:
Hungaria, N_neo 29-46 and Phi 37-59 against 5.9-9.4 and 7.6-12 (tau_mb 61 My against "300?"), and
Baptistina, Phi 8.0-13 against 11-19. Witt and Brucato, in the published table, are left out of the
download. The middle of each range is used, as by the authors.

What the probabilities assume:
    - The maps are of 1-m bodies, and Phi of 1-m meteoroids. A body much larger, of a size whose
      population differs between the families, would need other weights.
    - The bin is taken as it is, without interpolation, as meteomod.py does.
    - The orbit is in the model if it is bound, with q < 1.3 AU, 0 <= a < 3.5 AU and i < 90 deg. A bin
      where none of the selected families has bodies gives no probabilities.
    - Variants of a family share its map: Koronis(2) and Karin the map of Karin; Massalia and
      Massalia(2), the young dust-band population, that of Massalia; Nysa (LL) and Nysa (E) that of Nysa,
      with the same Phi, so that selecting both counts Nysa twice. Select the families by type.
    - Averaged over clones, each clone weighs the same.

The file is not distributed with wmpl. It is downloaded on first use, 19 MB, from the authors' page,
checked against its SHA-256 and kept in wmpl/share (or ~/.cache/wmpl if that cannot be written), as
wmpl.Utils.DebiasedNEOPopulation does for Granvik et al. (2018). Nothing is extracted: the files are
read from the archive. No licence is stated on the page. Cite the papers when using it.

    Broz, M., Vernazza, P., Marsset, M., Binzel, R. P., DeMeo, F., Birlan, M., Colas, F., Anghel, S.,
    Bouley, S., Blanpain, C., Gattacceca, J., Jeanne, S., Jorda, L., Lecubin, J., Malgoyre, A.,
    Steinhausser, A., Vaubaillon, J. & Zanda, B. (2024a), Source regions of carbonaceous meteorites and
    near-Earth objects, A&A 689, A183, doi:10.1051/0004-6361/202450532

    Broz, M., Vernazza, P., Marsset, M., DeMeo, F. E., Binzel, R. P., Vokrouhlicky, D. & Nesvorny, D.
    (2024b), Young asteroid families as the primary source of meteorites, Nature 634, 566-571,
    doi:10.1038/s41586-024-08006-7

Usage:

    python -m wmpl.Utils.METEOMODSourceRegions path/to/trajectory.pickle --types L
    python -m wmpl.Utils.METEOMODSourceRegions path/to/trajectory.pickle --types CM --clones 1000 --seed 1
    python -m wmpl.Utils.METEOMODSourceRegions --a 2.5855 --e 0.6183 --i 0.460 --types CM --weight neo
    python -m wmpl.Utils.METEOMODSourceRegions --list

wmpl.Rebound.REBOUND takes --meteomod to give them at every output of an integration, or at its end only.
"""

from __future__ import print_function, division, absolute_import

import os
import re
import json
import tarfile

import numpy as np

from wmpl.Config import config
from wmpl.Utils.DebiasedNEOPopulation import CACHE_DIR, findDebiasedNEOPopulation, \
    downloadDebiasedNEOPopulation
from wmpl.Utils.SourceRegions import IN_MODEL, OUTSIDE, EMPTY_BIN, CLONE_INTERVAL, PLOT_COLORS, wrapped, \
    orderedSources, leftOutLine


# METEOMOD as the authors' page distributes it, in the form wmpl.Utils.DebiasedNEOPopulation finds,
#   downloads and checks a model file
METEOMOD = {
    'name': 'METEOMOD (Broz et al. 2024)',
    'file_name': 'Broz+_2024_METEOMOD.tar.gz',
    'share_path': config.meteomod_file,
    'url': 'https://sirrah.troja.mff.cuni.cz/~mira/meteomod/meteomod.tar.gz',
    'sha256': '3bac0abadb3007c0612012c58da522331d0e678a4244ff9dc1b8dae25317e4d4',
    'size_mb': 19,
    'reference': "Broz, M., Vernazza, P., Marsset, M., et al. (2024), Source regions of carbonaceous "
        "meteorites and near-Earth objects, A&A 689, A183, doi:10.1051/0004-6361/202450532, and Broz, M., "
        "Vernazza, P., Marsset, M., et al. (2024), Young asteroid families as the primary source of "
        "meteorites, Nature 634, 566-571, doi:10.1038/s41586-024-08006-7",
    'short_reference': "Broz et al. (2024), A&A 689, A183; Broz et al. (2024), Nature 634, 566",
    }

# The files of the model in the archive
ARCHIVE_DIR = 'meteomod'

# The grid of the maps: a in AU and i in deg as (number of bins, min, max)
GRID = {'a': (350, 0.0, 3.5), 'incl_deg': (90, 0.0, 90.0)}

# The maps are of the NEO region
NEO_Q_MAX = 1.3

# The weights of the families: the meteoroid flux, as the model defines it, or the number of bodies in
#   the NEO region, as the authors' meteomod.py computes it
WEIGHTS = {'flux': "meteoroid flux Phi of Table 5", 'neo': "N_neo of Table 5, as the authors' meteomod.py"}

# Sources below this probability everywhere are left out of the printed tables
PRINT_MIN = 0.005

# The model, once loaded
_model_cache = {}



def findMETEOMOD(model=METEOMOD):
    """ The path of the model file if it has been downloaded, None otherwise. """

    return findDebiasedNEOPopulation(model)



def downloadMETEOMOD(model=METEOMOD, file_path=None, url=None, verbose=True):
    """ Download the model file from the authors' page and check it against its SHA-256 (see
        wmpl.Utils.DebiasedNEOPopulation.downloadDebiasedNEOPopulation).

    Keyword arguments:
        model: [dict] The model, METEOMOD by default.
        file_path: [str] Where to write it. wmpl/share by default, or ~/.cache/wmpl if that cannot be
            written.
        url: [str] Where to download it from. The authors' page by default.
        verbose: [bool] Print what is being downloaded. True by default.

    Return:
        [str] The path of the file.
    """

    if not model.get('sha256'):
        raise ValueError("The model has no SHA-256 to check its file against.")

    url = model['url'] if url is None else url

    if verbose:
        print("Downloading the {:s} model file ({:d} MB), once, from".format(model['name'],
            model['size_mb']))
        print("  {:s}".format(url))

    path = downloadDebiasedNEOPopulation(model, file_path=file_path, url=url, verbose=False)

    if verbose:
        print("  to {:s}".format(path))
        for line in wrapped("Please cite {:s}.".format(model['reference'])):
            print(line)

    return path



def _cleanLatex(text):
    """ A family name of the files without its LaTeX: bold, subscripts and umlauts. """

    text = re.sub(r"\{\\bf\s*(.*?)\}", r"\1", text)
    text = re.sub(r"\$_\{(.*?)\}\$", r"\1", text)
    text = re.sub(r'\\"\{?([A-Za-z])\}?', r"\1", text)

    return text.replace("$", "").strip()



def _parseRange(text):
    """ The (low, high) of a range of Table 5, such as 2-7, 320-370! or $\\sim$100-200, or a single
        value as both. """

    text = re.sub(r"\$[^$]*\$", "", text.replace("!", "").replace("?", "")).strip()

    if '-' in text:
        low, high = text.split('-', 1)
        return float(low), float(high)

    return float(text), float(text)



def readTable5(text):
    """ The N_neo and Phi of each family from tab5.tex, the rows with a family number and a Phi, as the
        authors' read_tab5 takes them.

    Arguments:
        text: [str] The content of tab5.tex.

    Return:
        [dict] {number: {'label': the family as the table names it, 'n_neo': (low, high) in 1e8 bodies,
            'flux': (low, high) in 1e-9 km^-2 y^-1}}.
    """

    table = {}

    for line in text.splitlines():

        if line.startswith('%'):
            continue

        fields = line.split('&')
        if len(fields) < 10:
            continue

        number = fields[0].replace(" ", "")
        flux = fields[9].strip()

        if (not number) or (flux in ('--', '')):
            continue

        try:
            table[number] = {'label': _cleanLatex(fields[1]), 'n_neo': _parseRange(fields[8]),
                'flux': _parseRange(flux)}
        except ValueError:
            continue

    return table



def _nameAndType(label):
    """ The name and the type of a family from its label, the type being its last parenthesised part. """

    match = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", label)

    return (match.group(1), match.group(2)) if match else (label, "?")



def readMapsIndex(text):
    """ The families of maps.in: their numbers, names, types and map directories, in order.

    Arguments:
        text: [str] The content of maps.in.

    Return:
        [list] (number, name, type, directory) of each family.
    """

    families = []

    for line in text.splitlines():

        if line.startswith('%'):
            continue

        fields = line.split('&')
        if len(fields) < 3:
            continue

        number = fields[0].replace(" ", "")
        name, kind = _nameAndType(_cleanLatex(fields[1]))

        families.append((number, name, kind, fields[2].strip()))

    return families



def readMap(file_obj):
    """ A map of the NEO region in a and i from a map.out file, normalised to 1.

    Arguments:
        file_obj: [file] The map.out file: bin in a, bin in i (from 1), a, i and the count, one bin per
            line, with blank lines between the bins in a.

    Return:
        [ndarray] The map, (bins in a, bins in i).
    """

    data = np.loadtxt(file_obj, ndmin=2)

    counts = np.zeros((GRID['a'][0], GRID['incl_deg'][0]))
    counts[data[:, 0].astype(int) - 1, data[:, 1].astype(int) - 1] = data[:, 4]

    total = np.sum(counts)
    if total <= 0:
        raise ValueError("A map of METEOMOD is empty.")

    return counts/total



def loadMETEOMOD(model=METEOMOD, file_path=None, download=True, verbose=True):
    """ Load METEOMOD from its archive, downloading it first if it is not there. The files are read
        from the archive without extracting it.

    Keyword arguments:
        model: [dict] The model, METEOMOD by default.
        file_path: [str] The archive to read instead of the one in wmpl/share or the cache. None by
            default.
        download: [bool] Download the file if it is not found. If False, raise instead. True by default.
        verbose: [bool] Print a note when downloading. True by default.

    Return:
        [dict] 'model' and 'file_path'; 'sources', the family numbers in the order of the authors' list;
            'names' and 'types', {number: name or type}; 'directories', {number: map directory};
            'maps', (source, bins in a, bins in i), each normalised to 1; 'n_neo' and 'flux',
            {number: (low, high)}.
    """

    if file_path is None:

        file_path = findMETEOMOD(model)

        if file_path is None:

            if not download:
                raise IOError("The {:s} model file is not in {:s}. Download it from {:s}.".format(
                    model['name'], " or ".join([model['share_path'], os.path.join(CACHE_DIR,
                    model['file_name'])]), model['url']))

            file_path = downloadMETEOMOD(model, verbose=verbose)

    file_path = os.path.abspath(file_path)

    if file_path in _model_cache:
        return _model_cache[file_path]

    with tarfile.open(file_path, 'r:gz') as archive:

        def _member(name):
            member = archive.extractfile("{:s}/{:s}".format(ARCHIVE_DIR, name))
            if member is None:
                raise IOError("{:s} has no {:s}.".format(file_path, name))
            return member

        families = readMapsIndex(_member('maps.in').read().decode('utf-8', 'replace'))
        table = readTable5(_member('tab5.tex').read().decode('utf-8', 'replace'))

        missing = [number for number, _, _, _ in families if number not in table]
        if missing:
            raise ValueError("Table 5 of {:s} has no flux for {:s}.".format(file_path, ", ".join(missing)))

        # Each map once, some families sharing one
        maps_of = {}
        for _, _, _, directory in families:
            if directory not in maps_of:
                maps_of[directory] = readMap(_member("map/{:s}/map.out".format(directory)))

    sources = [number for number, _, _, _ in families]

    # The names and types of Table 5, which the paper and the page give, as maps.in differs for two
    #   families (Misa CM and Brasilia CH there, CI and M in the table)
    names_types = {number: _nameAndType(table[number]['label']) for number in sources}

    meteomod = {
        'model': model,
        'file_path': file_path,
        'sources': sources,
        'names': {number: names_types[number][0] for number in sources},
        'types': {number: names_types[number][1] for number in sources},
        'directories': {number: directory for number, _, _, directory in families},
        'maps': np.array([maps_of[directory] for _, _, _, directory in families]),
        'n_neo': {number: table[number]['n_neo'] for number in sources},
        'flux': {number: table[number]['flux'] for number in sources},
        }

    _model_cache[file_path] = meteomod

    return meteomod



def selectSources(meteomod, types=None, sources=None):
    """ The families to compare: those given by number, or those of the given types, or all.

    Arguments:
        meteomod: [dict] Return of loadMETEOMOD.

    Keyword arguments:
        types: [list] Types of the meteorite, such as ['L'] or ['CM', 'CI']. A family is kept if one of the
            parts of its type, split at '/', is one of them (Juno, L/LL, is kept for L and for LL), or if
            its whole type is. None by default.
        sources: [list] Family numbers, as in Table 5 (20 and 20' are Massalia and Massalia(2)). None by
            default.

    Return:
        [list] The family numbers, in the order of the authors' list.
    """

    if (types is not None) and (sources is not None):
        raise ValueError("Select the families either by type or by number, not both.")

    if sources is not None:

        unknown = [s for s in sources if s not in meteomod['sources']]
        if unknown:
            raise ValueError("No family {:s} in METEOMOD. Its families are {:s}.".format(", ".join(unknown),
                ", ".join(meteomod['sources'])))

        return [s for s in meteomod['sources'] if s in sources]

    if types is not None:

        wanted = set(t.upper() for t in types)
        kept = [s for s in meteomod['sources'] if (meteomod['types'][s].upper() in wanted)
            or (wanted & set(part.upper() for part in meteomod['types'][s].split('/')))]

        if not kept:
            raise ValueError("No family of METEOMOD is of type {:s}. Its types are {:s}.".format(
                ", ".join(types), ", ".join(sorted(set(meteomod['types'].values())))))

        return kept

    return list(meteomod['sources'])



def familyWeights(meteomod, sources, weight='flux'):
    """ The weight of each family: the middle of its range of Phi, or of N_neo.

    Arguments:
        meteomod: [dict] Return of loadMETEOMOD.
        sources: [list] Family numbers.

    Keyword arguments:
        weight: [str] 'flux' for Phi, as the model defines it, 'neo' for N_neo, as the authors' meteomod.py
            computes it. 'flux' by default.

    Return:
        [ndarray] The weights.
    """

    if weight not in WEIGHTS:
        raise ValueError("The weight is 'flux' or 'neo', not {:s}.".format(str(weight)))

    key = 'flux' if weight == 'flux' else 'n_neo'

    return np.array([0.5*(meteomod[key][s][0] + meteomod[key][s][1]) for s in sources])



def _binIndices(a, e, incl):
    """ The bins in a and i (0-based) of orbits, as meteomod.py finds them, and whether each orbit is in
        the region of the maps.

    Arguments:
        a: [ndarray] Semi-major axes (AU).
        e: [ndarray] Eccentricities.
        incl: [ndarray] Inclinations (rad).

    Return:
        (ia, ii, inside): [tuple] The bins, 0 where the orbit is outside, and the mask of the orbits
            inside.
    """

    # Rounded to 1e-9 deg, so that an inclination given in degrees on the edge of a bin falls in the bin
    #   meteomod.py puts it in after the round trip through radians
    incl_deg = np.round(np.degrees(incl), 9)

    with np.errstate(invalid='ignore'):

        q = a*(1.0 - e)

        inside = np.isfinite(a) & np.isfinite(e) & np.isfinite(incl_deg) & (e < 1.0) & (q < NEO_Q_MAX)

        indices = []
        for values, (n, v_min, v_max) in zip((a, incl_deg), (GRID['a'], GRID['incl_deg'])):

            inside &= (values >= v_min) & (values < v_max)

            # The bin as meteomod.py computes it
            dv = (v_max - v_min)/n
            k = np.zeros(len(values), dtype=int)
            k[inside] = np.minimum(np.floor((values[inside] - v_min)/dv).astype(int), n - 1)

            indices.append(k)

    ia, ii = [np.where(inside, k, 0) for k in indices]

    return ia, ii, inside



def sourceProbabilities(meteomod, a, e, incl, sources=None, weight='flux'):
    """ The probability that a meteoroid on each of the given orbits came from each of the selected
        families, M_j(a, i) Phi_j normalised over them.

    Arguments:
        meteomod: [dict] Return of loadMETEOMOD.
        a: [float or ndarray] Semi-major axis (AU).
        e: [float or ndarray] Eccentricity.
        incl: [float or ndarray] Inclination (rad).

    Keyword arguments:
        sources: [list] The family numbers to compare (see selectSources). All, by default.
        weight: [str] 'flux' or 'neo' (see familyWeights). 'flux' by default.

    Return:
        (probabilities, map_values, status, bins):
            - probabilities: [ndarray] The probability of each family, with a last axis over sources, NaN
                where the orbit has none.
            - map_values: [ndarray] M_j(a, i) of each family, the same shape, NaN outside the maps.
            - status: [ndarray] IN_MODEL, OUTSIDE or EMPTY_BIN for each orbit.
            - bins: [ndarray] The bin of each orbit as a flat index in (a, i), -1 outside.

    Reference: the probability as the METEOMOD page and Eq. (9) of Broz et al. (2024), A&A 689, A183,
        doi:10.1051/0004-6361/202450532, define it, with the maps and Table 5 of the download.
    """

    if sources is None:
        sources = meteomod['sources']

    columns = [meteomod['sources'].index(s) for s in sources]
    weights = familyWeights(meteomod, sources, weight=weight)

    a, e, incl = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(e, dtype=float),
        np.asarray(incl, dtype=float))

    shape = a.shape
    a, e, incl = a.ravel(), e.ravel(), incl.ravel()

    ia, ii, inside = _binIndices(a, e, incl)

    n = len(a)
    map_values = np.full((n, len(sources)), np.nan)
    map_values[inside] = meteomod['maps'][columns][:, ia[inside], ii[inside]].T

    weighted = map_values*weights
    total = np.sum(weighted, axis=1)

    status = np.full(n, OUTSIDE, dtype=int)
    status[inside] = EMPTY_BIN

    found = inside & (total > 0)
    status[found] = IN_MODEL

    probabilities = np.full((n, len(sources)), np.nan)
    probabilities[found] = weighted[found]/total[found, None]

    bins = np.full(n, -1, dtype=np.int64)
    bins[inside] = ia[inside]*GRID['incl_deg'][0] + ii[inside]

    return probabilities.reshape(shape + (len(sources),)), map_values.reshape(shape + (len(sources),)), \
        status.reshape(shape), bins.reshape(shape)



def outsideReason(a, e, incl, status):
    """ Why an orbit has no source probabilities, or None if it has them.

    Arguments:
        a: [float] Semi-major axis (AU).
        e: [float] Eccentricity.
        incl: [float] Inclination (rad).
        status: [int] Its status from sourceProbabilities.

    Return:
        [str] The reason, or None.
    """

    if status == IN_MODEL:
        return None

    if status == EMPTY_BIN:
        return "none of the selected families has bodies in the bin of the orbit"

    if not all(np.isfinite(v) for v in (a, e, incl)):
        return "the orbit has no elements"

    if e >= 1.0:
        return "the orbit is not bound (e = {:.4f}), and the maps hold only bound orbits".format(e)

    q = a*(1.0 - e)
    if q >= NEO_Q_MAX:
        return "q = {:.3f} AU: outside the NEO region of the maps, q < {:.1f} AU".format(q, NEO_Q_MAX)

    if a >= GRID['a'][2]:
        return "a = {:.3f} AU is beyond the maps, which end at a = {:.1f} AU".format(a, GRID['a'][2])

    return "i = {:.2f} deg is beyond the maps, which end at i = {:.0f} deg".format(np.degrees(incl),
        GRID['incl_deg'][2])



def sourceRegions(meteomod, a, e, incl, sources=None, weight='flux'):
    """ The source probabilities of one orbit, with what they were computed from.

    Arguments:
        meteomod: [dict] Return of loadMETEOMOD.
        a: [float] Semi-major axis (AU).
        e: [float] Eccentricity.
        incl: [float] Inclination (rad).

    Keyword arguments:
        sources: [list] The family numbers to compare. All, by default.
        weight: [str] 'flux' or 'neo' (see familyWeights). 'flux' by default.

    Return:
        [dict] 'a', 'e', 'q' (AU), 'incl' (rad); 'weight'; 'sources'; 'in_model'; 'reason', why the
            orbit has no probabilities, or None; 'bin', the limits of its bin in 'a' and 'incl_deg', or
            None outside the maps; 'probabilities' and 'map_values', {number: value}, or None.
    """

    if sources is None:
        sources = meteomod['sources']

    probabilities, map_values, status, _ = sourceProbabilities(meteomod, a, e, incl, sources=sources,
        weight=weight)

    status = int(status)
    in_model = status == IN_MODEL

    ia, ii, inside = _binIndices(np.array([a], dtype=float), np.array([e], dtype=float),
        np.array([incl], dtype=float))

    bin_ranges = None
    if inside[0]:
        bin_ranges = {}
        for key, k in (('a', ia[0]), ('incl_deg', ii[0])):
            n, v_min, v_max = GRID[key]
            dv = (v_max - v_min)/n
            bin_ranges[key] = (v_min + k*dv, v_min + (k + 1)*dv)

    return {
        'a': float(a), 'e': float(e), 'q': float(a*(1.0 - e)), 'incl': float(incl),
        'weight': weight,
        'sources': list(sources),
        'in_model': in_model,
        'reason': outsideReason(a, e, incl, status),
        'bin': bin_ranges,
        'probabilities': {s: float(p) for s, p in zip(sources, probabilities)} if in_model else None,
        'map_values': {s: float(m) for s, m in zip(sources, map_values)} if inside[0] else None,
        }



def cloneSourceRegions(meteomod, a, e, incl, sources=None, weight='flux', nominal=None,
        interval=CLONE_INTERVAL):
    """ The source probabilities over clones of an orbit: their mean over the clones in the model, which
        is the probability of each family given the uncertainty of the orbit, and the interval they span.

    Arguments:
        meteomod: [dict] Return of loadMETEOMOD.
        a: [ndarray] Semi-major axes of the clones (AU).
        e: [ndarray] Eccentricities of the clones.
        incl: [ndarray] Inclinations of the clones (rad).

    Keyword arguments:
        sources: [list] The family numbers to compare. All, by default.
        weight: [str] 'flux' or 'neo' (see familyWeights). 'flux' by default.
        nominal: [tuple] (a, e, incl) of the nominal orbit, to count the clones in its bin. None by
            default.
        interval: [tuple] Percentiles of the interval. CLONE_INTERVAL by default.

    Return:
        [dict] 'n', the clones; 'n_in_model'; 'n_bins', the bins they fall in; 'share_in_nominal_bin',
            or None without the nominal; 'interval'; and {number: value} dicts 'mean', 'low' and 'high',
            None if no clone is in the model.
    """

    if sources is None:
        sources = meteomod['sources']

    probabilities, _, status, bins = sourceProbabilities(meteomod, a, e, incl, sources=sources,
        weight=weight)

    in_model = status == IN_MODEL

    out = {'n': int(len(status)), 'n_in_model': int(np.sum(in_model)),
        'n_bins': int(len(np.unique(bins[in_model]))), 'share_in_nominal_bin': None,
        'interval': tuple(interval), 'mean': None, 'low': None, 'high': None}

    if nominal is not None:
        _, _, _, nominal_bin = sourceProbabilities(meteomod, *nominal, sources=sources, weight=weight)
        if nominal_bin >= 0:
            out['share_in_nominal_bin'] = float(np.mean(bins == nominal_bin))

    if np.any(in_model):

        values = probabilities[in_model]
        low, high = np.percentile(values, interval, axis=0)

        out['mean'] = {s: float(v) for s, v in zip(sources, np.mean(values, axis=0))}
        out['low'] = {s: float(v) for s, v in zip(sources, low)}
        out['high'] = {s: float(v) for s, v in zip(sources, high)}

    return out



def sourceNames(meteomod, sources=None):
    """ The name of each family with its type, {number: 'Name (type)'}. """

    if sources is None:
        sources = meteomod['sources']

    return {s: "{:s} ({:s})".format(meteomod['names'][s], meteomod['types'][s]) for s in sources}



def selectionWarning(meteomod, sources, weight='flux'):
    """ A warning about the weights or the families selected, or None. """

    warnings = []

    if weight == 'neo':
        warnings.append("WARNING: weighted by N_neo, as the authors' meteomod.py computes it, not by the "
            "meteoroid flux Phi that METEOMOD defines for bodies that hit the Earth.")

    for first, second in (('44', "44'"),):
        if (first in sources) and (second in sources):
            warnings.append("WARNING: {:s} and {:s} are the same family, Nysa, read as LL and as E, with "
                "the same map and flux, so it is counted twice. Select the families by type.".format(first,
                second))

    return " ".join(warnings) if warnings else None



def headingText(meteomod, sources, weight='flux', types=None):
    """ What the probabilities were computed for, in words. """

    if len(sources) == len(meteomod['sources']):
        selection = "all {:d} families".format(len(sources))
    elif types is not None:
        selection = "the {:d} families of type {:s}".format(len(sources), ", ".join(types))
    else:
        selection = "{:d} families".format(len(sources))

    return "{:s}, weighted by the {:s} (1-m meteoroids)".format(selection, WEIGHTS[weight])



def sourceModel(meteomod, sources=None, weight='flux', types=None):
    """ METEOMOD with a selection of families, described for wmpl.Utils.SourceRegions, whose series,
        report, JSON, CSV and figure it is given to.

    Arguments:
        meteomod: [dict] Return of loadMETEOMOD.

    Keyword arguments:
        sources: [list] The family numbers to compare. All, by default.
        weight: [str] 'flux' or 'neo' (see familyWeights). 'flux' by default.
        types: [list] The types the families were selected by, to say so. None by default.

    Return:
        [dict] The model (see wmpl.Utils.SourceRegions).
    """

    if sources is None:
        sources = list(meteomod['sources'])

    names = sourceNames(meteomod, sources)

    def _evaluate(a, e, incl):
        probabilities, _, status, _ = sourceProbabilities(meteomod, a, e, incl, sources=sources,
            weight=weight)
        return probabilities, status

    def _reason(a, e, incl, status):
        return outsideReason(a, e, incl, status)

    # A colour per family when they are few, else for the families with the largest probabilities of
    #   the series drawn
    plot_groups = None
    if len(sources) <= len(PLOT_COLORS):
        plot_groups = [(names[s], color, [s]) for s, color in zip(sources, PLOT_COLORS)]

    warning = selectionWarning(meteomod, sources, weight=weight)

    return {
        'name': METEOMOD['name'],
        'label': "METEOMOD",
        'reference': METEOMOD['reference'],
        'short_reference': METEOMOD['short_reference'],
        'sources': list(sources),
        'source_names': names,
        'evaluate': _evaluate,
        'reason': _reason,
        'heading': headingText(meteomod, sources, weight=weight, types=types),
        'warning': warning,
        'title_note': ", weighted by {:s}".format("the meteoroid flux" if weight == 'flux' else "N_neo"),
        'print_min': PRINT_MIN,
        'name_width': 22,
        'plot_groups': plot_groups,
        'json': {
            'model_file': os.path.basename(meteomod['file_path']),
            'weight': weight,
            'types': types,
            'warning': warning,
            'family_types': {s: meteomod['types'][s] for s in sources},
            'n_neo_1e8': {s: list(meteomod['n_neo'][s]) for s in sources},
            'flux_1e-9_km-2_yr-1': {s: list(meteomod['flux'][s]) for s in sources},
            },
        }



def formatSourceRegions(meteomod, result, clones=None, types=None):
    """ Lines of text giving the source probabilities of one orbit.

    Arguments:
        meteomod: [dict] Return of loadMETEOMOD.
        result: [dict] Return of sourceRegions.

    Keyword arguments:
        clones: [dict] Return of cloneSourceRegions, or None. None by default.
        types: [list] The types the families were selected by, to say so. None by default.

    Return:
        [list] The lines.
    """

    sources = result['sources']
    names = sourceNames(meteomod, sources)
    weights = dict(zip(sources, familyWeights(meteomod, sources, weight=result['weight'])))

    lines = ["Source regions in {:s}, {:s}:".format(METEOMOD['name'], headingText(meteomod, sources,
        weight=result['weight'], types=types))]

    warning = selectionWarning(meteomod, sources, weight=result['weight'])
    if warning is not None:
        lines += wrapped(warning)

    lines.append("  Orbit: a = {:.4f} AU, e = {:.4f}, i = {:.3f} deg, q = {:.4f} AU".format(result['a'],
        result['e'], np.degrees(result['incl']), result['q']))

    if result['bin'] is not None:
        lines.append("  Bin:   a {:.2f}-{:.2f} AU, i {:.0f}-{:.0f} deg".format(*(result['bin']['a']
            + result['bin']['incl_deg'])))

    if clones is not None:
        lines.append("  Clones: {:d}, {:d} in the model, over {:d} bin(s){:s}".format(clones['n'],
            clones['n_in_model'], clones['n_bins'], "" if clones['share_in_nominal_bin'] is None
            else ", {:.1f}% in the bin of the nominal orbit".format(100*clones['share_in_nominal_bin'])))

    if not result['in_model']:
        lines.append("  No source probabilities: {:s}.".format(result['reason']))
        return lines


    columns = [result['probabilities']] + ([clones['mean']] if (clones is not None) else [])
    order = orderedSources(sources, columns, PRINT_MIN)

    unit = "Phi [1e-9/km2/yr]" if result['weight'] == 'flux' else "N_neo [1e8]"

    lines.append("")
    header = "  {:<5s} {:<24s} {:>8s}  {:>10s}  {:>17s}".format("No.", "Family", "p", "M_j(a,i)", unit)
    if clones is not None:
        header += "  {:>26s}".format("clones mean [{:g}-{:g}%]".format(*clones['interval']))
    lines.append(header)

    for s in order:

        line = "  {:<5s} {:<24s} {:8.3f}  {:10.3e}  {:17.2f}".format(s, names[s], result['probabilities'][s],
            result['map_values'][s], weights[s])

        if clones is not None:
            line += "  {:>26s}".format("n/a" if clones['mean'] is None else "{:.3f} [{:.3f}, {:.3f}]".format(
                clones['mean'][s], clones['low'][s], clones['high'][s]))

        lines.append(line)

    left_out = leftOutLine({'source_names': names, 'print_min': PRINT_MIN}, sources, order)
    if left_out is not None:
        lines.append(left_out)

    return lines



def formatSourceList(meteomod):
    """ Lines of text listing the families of the model, their types, maps, N_neo and Phi. """

    lines = ["The families of {:s}, with N_neo [1e8] and Phi [1e-9 km^-2 yr^-1] of Table 5:".format(
        METEOMOD['name']), ""]
    lines.append("  {:<5s} {:<16s} {:<10s} {:<16s} {:>11s} {:>11s}".format("No.", "Family", "Type", "Map",
        "N_neo", "Phi"))

    for s in meteomod['sources']:
        lines.append("  {:<5s} {:<16s} {:<10s} {:<16s} {:>11s} {:>11s}".format(s, meteomod['names'][s],
            meteomod['types'][s], meteomod['directories'][s], "{:g}-{:g}".format(*meteomod['n_neo'][s]),
            "{:g}-{:g}".format(*meteomod['flux'][s])))

    return lines



if __name__ == "__main__":

    import sys
    import argparse

    from wmpl.Utils.Pickling import loadPickle


    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="The probability that a meteoroid on an orbit came "
        "from each asteroid family of METEOMOD (Broz et al. 2024), the map of each family in a and i "
        "weighted by its meteoroid flux on the Earth. The model file, 19 MB, is downloaded from the "
        "authors' page on first use and kept in wmpl/share.")

    arg_parser.add_argument('traj_path', metavar='TRAJ_PATH', type=str, nargs='?', default=None, \
        help="Path to the trajectory pickle of the orbit. Omit it to give the orbit with --q or --a, --e "
            "and --i.")

    elem = arg_parser.add_argument_group('orbit, given by hand')
    elem.add_argument('-q', '--q', metavar='PERIHELION_DIST', type=float, dest='q', \
        help="Perihelion distance of the orbit in AU.")
    elem.add_argument('-a', '--a', metavar='SEMI_MAJOR_AXIS', type=float, dest='a', \
        help="Semi-major axis of the orbit in AU, in place of the perihelion distance.")
    elem.add_argument('-e', '--e', metavar='ECCENTRICITY', type=float, dest='e', \
        help="Eccentricity of the orbit.")
    elem.add_argument('-i', '--i', metavar='INCLINATION', type=float, dest='i', \
        help="Inclination of the orbit (deg).")

    selection = arg_parser.add_argument_group('families compared')
    selection.add_argument('--types', metavar='TYPE', type=str, nargs='+', default=None, \
        help="Types of the meteorite, to compare the families of those types only, e.g. L, or CM CI. A "
            "family of type L/LL is kept for L and for LL. All families by default.")
    selection.add_argument('--sources', metavar='NUMBER', type=str, nargs='+', default=None, \
        help="Family numbers to compare, as in Table 5 (20 and \"20'\" are Massalia and Massalia(2)).")
    selection.add_argument('--weight', type=str, choices=sorted(WEIGHTS), default='flux', \
        help="Weight of the families: flux, the meteoroid flux on the Earth, as the model defines it "
            "(default), or neo, the number of bodies in the NEO region, as the authors' meteomod.py "
            "computes it.")
    selection.add_argument('--list', action='store_true', \
        help="List the families, their types, N_neo and flux, and stop.")

    arg_parser.add_argument('--clones', metavar='N', type=int, \
        help="Give the mean and the 95%% interval of the probabilities over N clones of the orbit, drawn "
            "from the covariance of a trajectory pickle solved with the Monte Carlo uncertainties, with "
            "the draws of wmpl.Rebound.REBOUND (see wmpl.Utils.OrbitClones).")
    arg_parser.add_argument('--seed', metavar='S', type=int, \
        help="Seed of the clones, to draw the same ones again. Drawn and printed if not given.")

    arg_parser.add_argument('--model_file', metavar='FILE', type=str, default=None, \
        help="The METEOMOD archive to read instead of downloading it.")

    arg_parser.add_argument('--save', metavar='FILE', type=str, default=None, \
        help="Save the results to this JSON file.")

    cml_args = arg_parser.parse_args()


    #########################


    def reportError(message):
        """ Print a message to stderr and stop.

        Arguments:
            message: [str] What went wrong.
        """

        print(message, file=sys.stderr)
        sys.exit(1)


    meteomod = loadMETEOMOD(file_path=cml_args.model_file)

    if cml_args.list:
        for line in formatSourceList(meteomod):
            print(line)
        sys.exit(0)

    try:
        sources = selectSources(meteomod, types=cml_args.types, sources=cml_args.sources)
    except ValueError as err:
        reportError(str(err))


    # Assemble the orbit, either from a trajectory pickle or from the elements given by hand
    traj = None

    if cml_args.traj_path is not None:

        traj = loadPickle(*os.path.split(os.path.abspath(cml_args.traj_path)))

        if traj.orbit is None or traj.orbit.a is None:
            reportError("The trajectory has no heliocentric orbit.")

        a, e, incl = traj.orbit.a, traj.orbit.e, traj.orbit.i

    else:

        if (cml_args.q is not None) and (cml_args.a is not None):
            reportError("Give the size of the orbit either as --q or as --a, not as both.")

        if ((cml_args.q is None) and (cml_args.a is None)) or (cml_args.e is None) or (cml_args.i is None):
            reportError("The orbit needs --q or --a, --e and --i, or a trajectory pickle to take it from.")

        e = cml_args.e
        incl = np.radians(cml_args.i)

        if cml_args.a is not None:
            a = cml_args.a
        elif e < 1.0:
            a = cml_args.q/(1.0 - e)
        else:
            a = np.nan

    if (cml_args.clones is None) and (cml_args.seed is not None):
        reportError("--seed goes with --clones.")

    if cml_args.clones is not None:

        if traj is None:
            reportError("--clones needs a trajectory pickle solved with the Monte Carlo uncertainties: the "
                "clones are drawn from its covariance.")

        if (getattr(traj, 'uncertainties', None) is None) or (getattr(traj, 'state_vect_cov', None) is None):
            reportError("--clones: the trajectory has no Monte Carlo uncertainties (traj.state_vect_cov), "
                "so it has no clones. Solve it again with the Monte Carlo uncertainties.")

        if cml_args.clones < 2:
            reportError("--clones needs at least 2 clones.")

    result = sourceRegions(meteomod, a, e, incl, sources=sources, weight=cml_args.weight)


    # The clones, from the trajectory's covariance
    clone_result = None
    clones = None

    if cml_args.clones is not None:

        from wmpl.Utils.OrbitClones import cloneOrbits, cloneHeaderLines

        clones = cloneOrbits(traj, cml_args.clones, random_seed=cml_args.seed)

        with np.errstate(divide='ignore', invalid='ignore'):
            a_clones = clones['q']/(1.0 - clones['e'])
        a_clones[clones['e'] >= 1.0] = np.nan

        clone_result = cloneSourceRegions(meteomod, a_clones, clones['e'], clones['incl'], sources=sources,
            weight=cml_args.weight, nominal=(a, e, incl))

        for line in cloneHeaderLines(clones):
            print(line)
        print()

    if cml_args.traj_path is not None:
        print("Trajectory: {:s}".format(cml_args.traj_path))

    for line in formatSourceRegions(meteomod, result, clones=clone_result, types=cml_args.types):
        print(line)

    print()
    for line in wrapped("Please cite {:s}.".format(METEOMOD['reference']), indent=""):
        print(line)


    # Save the results
    if cml_args.save is not None:

        saved = {
            'model': METEOMOD['name'],
            'model_file': os.path.basename(meteomod['file_path']),
            'reference': METEOMOD['reference'],
            'trajectory': cml_args.traj_path,
            'types': cml_args.types,
            'weight': cml_args.weight,
            'family_names': sourceNames(meteomod, sources),
            'result': dict(result, incl_deg=np.degrees(result['incl'])),
            'clones': None if clone_result is None else {'n': clones['n'], 'seed': clones['seed'],
                'result': clone_result},
            }

        with open(cml_args.save, 'w') as f:
            json.dump(saved, f, indent=1, default=float)

        print("Saved: {:s}".format(cml_args.save))
