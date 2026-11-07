""" Tests for the source regions of a meteoroid's orbit in METEOMOD, wmpl.Utils.METEOMODSourceRegions.

The tests write a small archive in the format of the authors' meteomod.tar.gz (SyntheticMETEOMOD.py),
so they need neither the network nor the 19 MB archive. The probabilities with the weight of the authors'
script are compared with a transcription of its compute_p. The one test that reads the real archive is
skipped when it has not been downloaded.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_METEOMODSourceRegions.py -v
"""

import os
import json
import hashlib
import tempfile

import numpy as np
import pytest

import wmpl.Utils.METEOMODSourceRegions as mm
import wmpl.Utils.SourceRegions as SourceRegions
from wmpl.Utils.SourceRegions import sourceRegionSeries, sourceRegionReportLines, \
    sourceRegionSeriesToJSON, plotSourceRegionSeries
from wmpl.Utils.Tests.SyntheticMETEOMOD import FAMILIES, writeArchive, familyFlux, familyNeo, familyMap


def _orbit(ia, ii):
    """ a (AU), e and incl (rad) of an orbit at q = 0.9 AU in the middle of a bin of the maps, the bin in
        a and i given from 1. """

    a = 0.01*(ia - 0.5)
    return a, 1.0 - 0.9/a, np.radians(ii - 0.5)



def _load(table='download', **kwargs):
    """ Write the test archive in a temporary directory and load it, with the values of its Table 5 as
        they are unless told otherwise. """

    path = os.path.join(tempfile.mkdtemp(), 'meteomod.tar.gz')
    writeArchive(path, **kwargs)

    return mm.loadMETEOMOD(file_path=path, download=False, table=table)



def _transcribedComputeP(meteomod, a, inc_deg, keys):
    """ compute_p of the authors' meteomod.py, transcribed: the bin as int((a - 0)/0.01) and
        int((inc - 0)/1), the map value times the middle of N_neo, normalised over the families. """

    i = int((a - 0.0)/0.01)
    j = int((inc_deg - 0.0)/1.0)

    p = {}
    s = 0.0
    for key in keys:
        val = meteomod['maps'][meteomod['sources'].index(key)][i, j]
        val *= 0.5*(meteomod['n_neo'][key][0] + meteomod['n_neo'][key][1])
        p[key] = val
        s += val

    return {k: v/s for k, v in p.items()} if s > 0 else None



def testLoaderReadsTheArchive():
    """ The families, their names and types from Table 5, their ranges with the LaTeX of the table, and
        the maps normalised, a map shared by two families given to both. """

    meteomod = _load()

    assert meteomod['sources'] == [f[0] for f in FAMILIES]
    assert meteomod['names']["20'"] == "Massalia(2)"
    assert meteomod['types']['3'] == "L/LL"

    # Misa is CM in maps.in and CI in Table 5, which is used
    assert meteomod['types']['569'] == "CI"

    assert meteomod['n_neo']["20'"] == (28.0, 55.0)
    assert meteomod['flux']['569'] == (7.3, 42.0)
    assert meteomod['n_neo']['490'] == (215.0, 249.0)

    assert meteomod['maps'].shape == (len(FAMILIES), 350, 90)
    assert np.allclose(np.sum(meteomod['maps'], axis=(1, 2)), 1.0)
    assert np.array_equal(meteomod['maps'][0], meteomod['maps'][1])
    assert meteomod['maps'][0][164, 1] == pytest.approx(0.3)
    assert meteomod['maps'][0][229, 4] == pytest.approx(0.6)



def testProbabilitiesAreTheMapsWeightedByTheFlux():
    """ P_j = M_j(a, i) Phi_j normalised over the selected families, and with N_neo for weight='neo'. """

    meteomod = _load()
    keys = ['20', "20'", '569', '3']
    a, e, incl = _orbit(165, 2)

    for weight, middle in (('flux', familyFlux), ('neo', familyNeo)):

        probabilities, map_values, status, _ = mm.sourceProbabilities(meteomod, a, e, incl, sources=keys,
            weight=weight)

        maps = np.array([familyMap(k).get((165, 2), 0.0)/sum(familyMap(k).values()) for k in keys])
        expected = maps*np.array([middle(k) for k in keys])
        expected /= np.sum(expected)

        assert status == mm.IN_MODEL
        assert np.allclose(map_values, maps)
        assert np.allclose(probabilities, expected)
        assert np.sum(probabilities) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        mm.sourceProbabilities(meteomod, a, e, incl, weight='mass')



def testTheNeoWeightIsTheAuthorsScript():
    """ With weight='neo' the probabilities are those of the authors' compute_p, transcribed, for orbits
        across the maps and on the edges of their bins. """

    meteomod = _load()
    rng = np.random.default_rng(3)

    # The orbits in the bins of the maps, and around them, and on the edges of bins
    centres = [(165, 2), (166, 2), (230, 5), (260, 1), (61, 61)]
    a = []
    inc = []
    for ia, ii in centres:
        a += list(0.01*(ia - 1) + 0.01*rng.uniform(0.0, 1.0, 40)) + [0.01*(ia - 1)]
        inc += list((ii - 1) + rng.uniform(0.0, 1.0, 40)) + [float(ii - 1)]
    a = np.array(a)
    inc = np.array(inc)
    e = 1.0 - 0.5/a

    n_compared = 0
    for keys in (meteomod['sources'], ['20', '3'], ['569', '490']):

        probabilities, _, status, _ = mm.sourceProbabilities(meteomod, a, e, np.radians(inc), sources=keys,
            weight='neo')

        for k in range(len(a)):

            expected = _transcribedComputeP(meteomod, a[k], inc[k], keys)

            if expected is None:
                assert status[k] == mm.EMPTY_BIN
                continue

            assert status[k] == mm.IN_MODEL
            assert np.allclose([expected[s] for s in keys], probabilities[k], rtol=0, atol=1e-15)
            n_compared += 1

    assert n_compared > 100



def testOrbitsOutsideTheMapsHaveNoProbabilitiesAndSayWhy():
    """ Unbound orbits, q >= 1.3 AU, a >= 3.5 AU, i >= 90 deg, a bin without bodies of the selected
        families and an orbit without elements give no probabilities, each with its reason. """

    meteomod = _load()

    cases = [
        ((-1.0, 1.2, 0.1), "not bound"),
        ((2.0, 0.3, 0.1), "outside the NEO region"),
        ((3.6, 0.8, 0.1), "beyond the maps, which end at a = 3.5 AU"),
        ((2.0, 0.6, np.radians(95.0)), "end at i = 90 deg"),
        ((1.5, 0.5, 0.1), "none of the selected families"),
        ((np.nan, 0.5, 0.1), "no elements"),
        ]

    for (a, e, incl), reason in cases:

        res = mm.sourceRegions(meteomod, a, e, incl)

        assert not res['in_model']
        assert res['probabilities'] is None
        assert reason in res['reason']

    # Every orbit outside is refused, as one array
    _, _, status, _ = mm.sourceProbabilities(meteomod, [-1.0, 3.6, 2.0], [1.2, 0.8, 0.6],
        [0.1, 0.1, np.radians(95.0)])
    assert np.all(status == mm.OUTSIDE)



def testAnInclinationOnTheEdgeOfABinGoesWhereTheScriptPutsIt():
    """ 1 deg given in radians comes back as 0.9999999999999999 deg, and must still fall in the 1-2 deg
        bin, as int(1.0) does in meteomod.py. """

    meteomod = _load()

    res = mm.sourceRegions(meteomod, 1.645, 1.0 - 0.9/1.645, np.radians(1.0), sources=['20'])

    assert res['bin']['incl_deg'] == pytest.approx((1.0, 2.0))
    assert res['in_model']



def testFamiliesAreSelectedByTypeOrNumber():
    """ A type keeps the families of which it is one of the parts, L/LL for L and for LL; numbers keep
        those; both, or an unknown one, are refused. """

    meteomod = _load()

    # All the families, Nysa once
    assert mm.selectSources(meteomod) == [s for s in meteomod['sources'] if s != "44'"]
    assert mm.selectSources(meteomod, types=['L']) == ['20', "20'", '3']
    assert mm.selectSources(meteomod, types=['ll']) == ['3', '44']
    assert mm.selectSources(meteomod, types=['E']) == ["44'"]
    assert mm.selectSources(meteomod, types=['CM', 'CI']) == ['569', '490']
    assert mm.selectSources(meteomod, sources=['490', '20']) == ['20', '490']

    for kwargs in ({'types': ['X']}, {'sources': ['999']}, {'types': ['L'], 'sources': ['20']}):
        with pytest.raises(ValueError):
            mm.selectSources(meteomod, **kwargs)



def testClonesGiveTheMeanOverThoseInTheModel():
    """ The mean is over the clones in the model, and the clones in the bin of the nominal are counted. """

    meteomod = _load()
    keys = ['20', '3']

    orbits = [_orbit(165, 2), _orbit(165, 2), _orbit(230, 5), (4.0, 0.9, 0.1)]
    a, e, incl = [np.array(v) for v in zip(*orbits)]

    clones = mm.cloneSourceRegions(meteomod, a, e, incl, sources=keys, nominal=orbits[0])

    first = mm.sourceRegions(meteomod, *orbits[0], sources=keys)['probabilities']
    third = mm.sourceRegions(meteomod, *orbits[2], sources=keys)['probabilities']

    assert clones['n'] == 4
    assert clones['n_in_model'] == 3
    assert clones['n_bins'] == 2
    assert clones['share_in_nominal_bin'] == pytest.approx(0.5)
    assert clones['mean']['20'] == pytest.approx((2*first['20'] + third['20'])/3)



def testTheModelGivesTheSeriesTheReportAndTheFigure(tmp_path):
    """ The model described for wmpl.Utils.SourceRegions: the series with the horizon and the
        representative clone, the report with its warnings and the papers, the JSON and the figure. """

    meteomod = _load()

    nominal = [_orbit(165, 2), _orbit(230, 5), _orbit(230, 5)]
    clones = [[_orbit(165, 2), _orbit(165, 2), _orbit(230, 5)], [_orbit(230, 5), (4.0, 0.9, 0.1),
        _orbit(230, 5)]]
    horizon = (-10.0, _orbit(230, 5), [_orbit(165, 2), _orbit(230, 5)])

    model = mm.sourceModel(meteomod, sources=['20', "20'", '3'], types=['L'])

    series = sourceRegionSeries(model, [0.0, -10.0, -100.0], nominal, clones=clones, horizon=horizon,
        representative=('mc_1', clones[0]))

    expected = mm.sourceRegions(meteomod, *_orbit(230, 5), sources=['20', "20'", '3'])['probabilities']
    assert series['nominal'][1, 2] == pytest.approx(expected['3'])
    assert series['clones_in_model'] == pytest.approx([1.0, 0.5, 1.0])
    assert series['horizon']['nominal'][0, 0] == pytest.approx(expected['20'])

    lines = sourceRegionReportLines(series, first=1)
    text = "\n".join(lines)
    assert "the 3 families of type L, weighted by the meteoroid flux" in text
    assert "Table 5 of the download" in text
    assert "Massalia(2) (L)" in text
    assert "representative (mc_1)" in text
    assert "Please cite Broz" in text
    assert "WARNING" not in text

    saved = json.loads(json.dumps(sourceRegionSeriesToJSON(series, mode='all')))
    assert saved['weight'] == 'flux'
    assert saved['table'] == 'download'
    assert saved['types'] == ['L']
    assert saved['flux_1e-9_km-2_yr-1']["20'"] == [200.0, 400.0]
    assert saved['horizon']['nominal']['3'] == pytest.approx(expected['3'])
    assert "doi:10.1051/0004-6361/202450532" in saved['reference']

    # Few families, a colour each; more than colours, those with the largest probabilities of the series
    #   (the families are listed twice here to be more than the colours)
    assert [label for label, _, _ in model['plot_groups']] == ["Massalia (L)", "Massalia(2) (L)",
        "Juno (L/LL)"]
    many = mm.sourceModel(meteomod, sources=meteomod['sources']*2)
    assert many['plot_groups'] is None

    plotSourceRegionSeries(series, str(tmp_path/"sources.png"))
    assert os.path.getsize(str(tmp_path/"sources.png")) > 0

    largest = SourceRegions._largestGroups(sourceRegionSeries(many, [0.0, -10.0], nominal[:2]))
    assert len(largest) == len(SourceRegions.PLOT_COLORS)
    assert largest[0][2] == ["20'"]

    # The weight of the authors' script and a family counted twice are warned about
    neo = mm.sourceModel(meteomod, sources=['20', '3'], weight='neo')
    assert "weighted by N_neo" in neo['warning']
    twice = mm.selectionWarning(meteomod, ['44', "44'"])
    assert "counted twice" in twice



def testThePublishedTableReplacesTheDownloadWhereTheyDiffer():
    """ By default the values of Table 5 as published replace those of the download where they differ,
        here the type of Juno, and say so; the download's are kept with table='download'. """

    path = os.path.join(tempfile.mkdtemp(), 'meteomod.tar.gz')
    writeArchive(path)

    published = mm.loadMETEOMOD(file_path=path, download=False)
    download = mm.loadMETEOMOD(file_path=path, download=False, table='download')

    assert published['table'] == 'published'
    assert published['table_changes'] == ['3']
    assert published['types']['3'] == "L"
    assert download['table_changes'] == []
    assert download['types']['3'] == "L/LL"

    # Juno is no longer an LL family
    assert mm.selectSources(published, types=['LL']) == ['44']

    model = mm.sourceModel(published, sources=['20', '3'])
    assert "Table 5 as published" in model['heading']
    assert model['json']['table_changes'] == ['3']

    with pytest.raises(ValueError):
        mm.loadMETEOMOD(file_path=path, download=False, table='preprint')



def testAllTheFamiliesAreComparedOnceAndSummedByType():
    """ Without a selection every family is compared once, Nysa as LL/E, and the probabilities are summed
        by type, for one orbit and over clones. """

    meteomod = _load(table='published')
    a, e, incl = _orbit(165, 2)

    res = mm.sourceRegions(meteomod, a, e, incl)

    assert res['merged']
    assert "44'" not in res['sources']
    assert sum(res['probabilities'].values()) == pytest.approx(1.0)

    p = res['probabilities']
    assert res['by_type']['L'] == pytest.approx(p['20'] + p["20'"] + p['3'])
    assert res['by_type']['LL/E'] == pytest.approx(p['44'])
    assert res['by_type']['CI'] == pytest.approx(p['569'])
    assert sum(res['by_type'].values()) == pytest.approx(1.0)
    assert mm.sourceNames(meteomod, res['sources'], merged=True)['44'] == "Nysa (LL/E)"

    # Given by number, Nysa keeps its type
    explicit = mm.sourceRegions(meteomod, a, e, incl, sources=['20', '44'])
    assert not explicit['merged']
    assert set(explicit['by_type']) == {'L', 'LL'}

    # Over clones, the mean of the sums is the sum of the means, and the interval is that of the sums
    orbits = [_orbit(165, 2), _orbit(230, 5), _orbit(230, 5), _orbit(165, 2)]
    clones = mm.cloneSourceRegions(meteomod, *[np.array(v) for v in zip(*orbits)])

    sums_l = [mm.sourceRegions(meteomod, *o)['by_type']['L'] for o in orbits]
    assert clones['by_type']['mean']['L'] == pytest.approx(np.mean(sums_l))
    assert clones['by_type']['low']['L'] == pytest.approx(np.percentile(sums_l, 2.5))
    assert clones['by_type']['mean']['L'] == pytest.approx(clones['mean']['20'] + clones['mean']["20'"]
        + clones['mean']['3'])

    text = "\n".join(mm.formatSourceRegions(meteomod, res))
    assert "all 6 families, each once" in text
    assert "NOTE: no type given" in text
    assert "By type" in text
    assert "Nysa (LL/E)" in text



def testTheReportSumsByTypeOnlyWithoutAType():
    """ The model of all the families gives the sums by type in the report, at the horizon too, and in the
        JSON, with the note; the families of one type give neither, and Nysa given twice is warned of. """

    meteomod = _load(table='published')

    model = mm.sourceModel(meteomod)
    assert "NOTE: no type given" in model['warning']
    assert [kind for kind, _ in model['type_groups']] == ['L', 'CI', 'CM', 'LL/E']
    assert model['json']['all_families_once']
    assert model['json']['family_types']['44'] == "LL/E"

    nominal = [_orbit(165, 2), _orbit(230, 5)]
    clones = [[_orbit(165, 2), _orbit(230, 5)], [_orbit(230, 5), _orbit(230, 5)]]
    horizon = (-10.0, _orbit(230, 5), [_orbit(165, 2), _orbit(230, 5)])

    series = sourceRegionSeries(model, [0.0, -10.0], nominal, clones=clones, horizon=horizon)

    lines = sourceRegionReportLines(series)
    text = "\n".join(lines)
    assert text.count("Type ") == 2
    assert "LL/E" in text

    saved = json.loads(json.dumps(sourceRegionSeriesToJSON(series, mode='all')))
    expected = mm.sourceRegions(meteomod, *_orbit(230, 5))['by_type']
    assert saved['by_type']['nominal']['L'][1] == pytest.approx(expected['L'])
    assert saved['by_type']['horizon']['nominal']['CI'] == pytest.approx(expected['CI'])
    assert saved['by_type']['members']['LL/E'] == ['44']

    # The families of one type: no sums, no note
    one_type = mm.sourceModel(meteomod, types=['L'])
    assert one_type['type_groups'] is None
    assert one_type['warning'] is None
    one_series = sourceRegionSeries(one_type, [0.0], nominal[:1])
    assert "By type" not in "\n".join(sourceRegionReportLines(one_series))
    assert sourceRegionSeriesToJSON(one_series)['by_type'] is None

    # Nysa given twice by number is counted twice, and warned about
    twice = mm.sourceModel(meteomod, sources=['44', "44'", '20'])
    assert "counted twice" in twice['warning']
    assert "NOTE" not in twice['warning']



def testDownloadKeepsOnlyAFileThatMatchesItsHash(tmp_path):
    """ The download goes through the downloader of wmpl.Utils.DebiasedNEOPopulation: a matching file is
        kept, any other is refused and leaves nothing behind. """

    source = str(tmp_path/"published.tar.gz")
    writeArchive(source)
    url = "file://" + source

    with open(source, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    good = dict(mm.METEOMOD, sha256=digest)
    target = str(tmp_path/"kept.tar.gz")
    assert mm.downloadMETEOMOD(good, file_path=target, url=url, verbose=False) == target
    assert mm.loadMETEOMOD(file_path=target, download=False)['sources'] == [f[0] for f in FAMILIES]

    bad = dict(mm.METEOMOD, sha256="0"*64)
    target = str(tmp_path/"refused.tar.gz")
    with pytest.raises(IOError):
        mm.downloadMETEOMOD(bad, file_path=target, url=url, verbose=False)
    assert not os.path.exists(target)



def testAFamilyWithoutFluxIsRefused(tmp_path):
    """ A family of maps.in without a row in Table 5 stops the loading, naming it. """

    path = str(tmp_path/"broken.tar.gz")
    writeArchive(path, maps_in="  20 & Massalia (L) & massalia-4\n 999 & Nowhere (L) & massalia-4\n")

    with pytest.raises(ValueError) as err:
        mm.loadMETEOMOD(file_path=path, download=False)

    assert "999" in str(err.value)



def testThePublishedArchiveWhenItHasBeenDownloaded():
    """ The published archive: its 55 families, and the probabilities of Winchcombe among the CM families
        and of Novo Mesto among the L families, the weight of the authors' script changing the first. """

    path = mm.findMETEOMOD()
    if path is None:
        pytest.skip("the METEOMOD archive has not been downloaded")

    meteomod = mm.loadMETEOMOD(file_path=path, download=False)

    assert len(meteomod['sources']) == 55
    assert meteomod['types']['569'] == "CI"

    cm = mm.selectSources(meteomod, types=['CM'])
    assert len(cm) == 9

    flux = mm.sourceRegions(meteomod, 2.5855, 0.6183, np.radians(0.460), sources=cm)['probabilities']
    neo = mm.sourceRegions(meteomod, 2.5855, 0.6183, np.radians(0.460), sources=cm,
        weight='neo')['probabilities']

    assert max(flux, key=flux.get) == '1128'
    assert flux['1128'] == pytest.approx(0.435, abs=0.001)
    assert max(neo, key=neo.get) == '490'
    assert neo['490'] == pytest.approx(0.304, abs=0.001)

    # Hungaria, Baptistina and Juno as published, and as in the download
    assert meteomod['flux']['434'] == (7.6, 12.0)
    assert meteomod['n_neo']['434'] == (5.9, 9.4)
    assert meteomod['flux']['298'] == (11.0, 19.0)
    assert meteomod['types']['3'] == "L"

    download = mm.loadMETEOMOD(file_path=path, download=False, table='download')
    assert download['flux']['434'] == (37.0, 59.0)
    assert download['flux']['298'] == (8.0, 13.0)
    assert download['types']['3'] == "L/LL"

    l_types = mm.selectSources(meteomod, types=['L'])
    novo = mm.sourceRegions(meteomod, 1.451, 0.60866, np.radians(8.755), sources=l_types)['probabilities']
    assert max(novo, key=novo.get) == "20'"
