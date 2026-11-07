""" Tests for the source regions of an orbit in the NEOMOD model, wmpl.Utils.NEOMODSourceRegions.

The tests write small model files in the format of the NEOMOD Simulator, with the binning of NEOMOD2
(SyntheticNEOMOD.py), so they need neither the network nor the 62 MB file. In one of them the shares
of the sources give the bin they are in, so that the bin found for an orbit can be read back and
compared with the bin the authors' give_me_prob.f finds, transcribed here. The one test that reads
the real file is skipped when it has not been downloaded.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_NEOMODSourceRegions.py -v
"""

import os
import ssl
import json
import hashlib
import tempfile

import numpy as np
import pytest

import wmpl.Utils.NEOMODSourceRegions as nm
from wmpl.Utils.DebiasedMeteoroidPopulation import massToH
from wmpl.Utils.Tests.SyntheticNEOMOD import BINNING, SOURCES, writeModel, shares



def _model(rows, **kwargs):
    """ Write a model file in a temporary directory and load it. """

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'model.dat')
    writeModel(path, rows, **kwargs)

    return nm.loadNEOMOD(file_path=path, download=False)



def _indexModel():
    """ A model with every bin in a, e and i at two H bins, 1 and 52, where the shares of the first
        four sources are the bin indices divided by 1000, so that the bin of an orbit can be read back
        from its probabilities. """

    rows = []
    for ih in (1, 52):
        for ia in range(1, 43):
            for ie in range(1, 26):
                for ii in range(1, 23):
                    rows.append((ih, ia, ie, ii, 1.0, [ih/1000, ia/1000, ie/1000, ii/1000] + [0.0]*8))

    return _model(rows)



def _giveMeProbBin(a, e, incl_deg, h_mag):
    """ The bin (iH, ia, ie, ii, from 1) give_me_prob.f finds for an orbit, or None outside its
        ranges, transcribed from the Fortran. """

    n_h, h_min, h_max = BINNING['h']
    n_a, a_min, a_max = BINNING['a']
    n_e, e_min, e_max = BINNING['e']
    n_i, i_min, i_max = BINNING['incl_deg']

    d_a = (a_max - a_min)/float(n_a)
    d_e = (e_max - e_min)/float(n_e)
    d_i = (i_max - i_min)/float(n_i)
    d_h = (h_max - h_min)/float(n_h)

    if (a < a_min) or (a > a_max) or (e < e_min) or (e > e_max) or (incl_deg < i_min) \
        or (incl_deg > i_max) or (h_mag < h_min) or (h_mag > h_max):
        return None

    ia = min(int((a - a_min)/d_a) + 1, n_a)
    ie = min(int((e - e_min)/d_e) + 1, n_e)
    ii = min(int((incl_deg - i_min)/d_i) + 1, n_i)
    ih = min(int((h_mag - h_min)/d_h) + 1, n_h)

    return ih, ia, ie, ii



def _sha256(file_path):
    """ SHA-256 of a file. """

    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()



def testLoaderReadsTheHeaderAndTheTable():
    """ The binning, the name and the sources come from the header, and each row lands in its bin. """

    neomod = _model([(52, 27, 16, 1, 867.4, shares(nu6=0.145, _3_1=0.701, _5_2=0.132))])

    assert neomod['model_name'] == "TestModel"
    assert neomod['sources'] == SOURCES

    for key, value in BINNING.items():
        assert neomod[key] == value

    assert neomod['row_of'].shape == (52, 42, 25, 22)
    assert neomod['row_of'][51, 26, 15, 0] == 0
    assert np.sum(neomod['row_of'] >= 0) == 1



def testProbabilitiesAreTheSharesOfTheBin():
    """ An orbit inside a bin gets the shares of its sources and the number of NEOs of the bin. """

    neomod = _model([(52, 27, 16, 1, 867.4, shares(nu6=0.145, _3_1=0.701, _5_2=0.132, _8_3=0.013))])

    res = nm.sourceRegions(neomod, 2.649, 0.627, np.radians(0.45))

    assert res['in_model']
    assert res['reason'] is None
    assert res['probabilities']['3:1'] == 0.701
    assert res['probabilities']['nu6'] == 0.145
    assert res['probabilities']['JFC'] == 0.0
    assert res['n_neo_bin'] == pytest.approx(867.4)
    assert res['bin']['a'] == pytest.approx((2.6, 2.7))
    assert res['bin']['e'] == pytest.approx((0.6, 0.64))
    assert res['bin']['incl_deg'] == pytest.approx((0.0, 4.0))



def testBinsAreTheOnesGiveMeProbFinds():
    """ For random orbits and orbits on the edges of the bins, the bin found is the one give_me_prob.f
        finds, wherever the orbit is in the region of the model. """

    neomod = _indexModel()
    rng = np.random.default_rng(1)

    n = 3000
    a = np.concatenate([rng.uniform(0.0, 4.3, n), 0.1*rng.integers(0, 43, n)])
    e = np.concatenate([rng.uniform(0.0, 1.0, n), 0.04*rng.integers(0, 26, n)])
    incl_deg = np.concatenate([rng.uniform(0.0, 90.0, n), 4.0*rng.integers(0, 23, n)])

    n_compared = 0

    for h_mag in (15.0, 15.2, 27.75, 27.9, 28.0):

        probabilities, _, status, _, _ = nm.sourceProbabilities(neomod, a, e, np.radians(incl_deg),
            h_mag=h_mag)

        for k in range(len(a)):

            expected = _giveMeProbBin(a[k], e[k], incl_deg[k], h_mag)

            # The module also requires a bound orbit with q < 1.3 AU
            in_region = (expected is not None) and (e[k] < 1.0) and (a[k]*(1.0 - e[k]) < 1.3)

            if not in_region:
                assert status[k] != nm.IN_MODEL
                continue

            # The model has rows only at the H bins 1 and 52
            if expected[0] not in (1, 52):
                assert status[k] == nm.EMPTY_BIN
                continue

            assert status[k] == nm.IN_MODEL
            found = tuple(int(round(1000*p)) for p in probabilities[k, :4])
            assert found == expected, (a[k], e[k], incl_deg[k], h_mag)

            n_compared += 1

    assert n_compared > 1000



def testAnInclinationOnTheEdgeOfABinGoesWhereGiveMeProbPutsIt():
    """ 60 deg given in radians comes back as 59.99999999999999 deg, and must still fall in the 60-64
        deg bin, as it does in give_me_prob.f. """

    neomod = _indexModel()

    probabilities, _, status, _, _ = nm.sourceProbabilities(neomod, 1.6, 0.8, np.radians(60.0),
        h_mag=27.9)

    assert status == nm.IN_MODEL
    assert int(round(1000*probabilities[3])) == 16



def testUpperLimitsFallInTheLastBin():
    """ a = 4.2 AU and i = 88 deg are inside the model, in its last bins, as in give_me_prob.f. """

    neomod = _indexModel()

    probabilities, _, status, _, _ = nm.sourceProbabilities(neomod, 4.2, 0.9, np.radians(88.0),
        h_mag=27.9)

    assert status == nm.IN_MODEL
    assert [int(round(1000*p)) for p in probabilities[:4]] == [52, 42, 23, 22]



def testOrbitsOutsideTheModelHaveNoProbabilitiesAndSayWhy():
    """ Unbound orbits, q >= 1.3 AU, a > 4.2 AU, i > 88 deg, a bin without NEOs and an orbit without
        elements give no probabilities, each with its reason. """

    neomod = _model([(52, 27, 16, 1, 1.0, shares(_3_1=1.0))])

    cases = [
        ((2.0, 1.2, 0.1), "not bound"),
        ((2.0, 0.3, 0.1), "not a near-Earth object"),
        ((5.0, 0.8, 0.1), "beyond the model, which ends at a = 4.2 AU"),
        ((2.0, 0.6, np.radians(100.0)), "ends at i = 88 deg"),
        ((1.5, 0.5, 0.1), "no model NEO"),
        ((np.nan, 0.5, 0.1), "no elements"),
        ]

    for (a, e, incl), reason in cases:

        res = nm.sourceRegions(neomod, a, e, incl)

        assert not res['in_model']
        assert res['probabilities'] is None
        assert res['n_neo_bin'] is None
        assert reason in res['reason']



def testTheHBinOfASmallBodyIsTheFaintestOfTheModel():
    """ Without H, and for H beyond the model, the bin at the end of the model is used and flagged. """

    neomod = _indexModel()

    meteoroid = nm.hBin(neomod)
    assert meteoroid['index'] == 51
    assert meteoroid['range'] == pytest.approx((27.75, 28.0))
    assert meteoroid['extrapolated']

    faint = nm.hBin(neomod, 36.3)
    assert (faint['index'] == 51) and faint['extrapolated']

    bright = nm.hBin(neomod, 10.0)
    assert (bright['index'] == 0) and bright['extrapolated']

    inside = nm.hBin(neomod, 20.1)
    assert (inside['index'] == 20) and (not inside['extrapolated'])
    assert inside['range'] == pytest.approx((20.0, 20.25))

    edge = nm.hBin(neomod, 28.0)
    assert (edge['index'] == 51) and (not edge['extrapolated'])



def testSizesAreConvertedToH():
    """ H is taken as it is, a mass with massToH and a diameter with the same albedo, 8.7 m being
        H = 28. """

    assert nm.parseBodySize('26.3') == (26.3, None)

    h_mag, description = nm.parseBodySize('13kg')
    assert h_mag == pytest.approx(massToH(13.0))
    assert description == "mass 13 kg"

    h_mag, description = nm.parseBodySize('19m')
    assert h_mag == pytest.approx(nm.diameterToH(19.0))
    assert description == "diameter 19 m"

    assert nm.diameterToH(8.7) == pytest.approx(28.0, abs=0.01)

    # The diameter of a sphere of the mass and density massToH takes gives the same H
    diameter = (6.0*13.0/(np.pi*3500.0))**(1.0/3)
    assert nm.diameterToH(diameter) == pytest.approx(massToH(13.0))

    for bad in ('-1m', '0m', 'big'):
        with pytest.raises(ValueError):
            nm.parseBodySize(bad)



def testClonesGiveTheMeanOverThoseInTheModel():
    """ The mean is over the clones in the model, and the clones in the bin of the nominal orbit are
        counted. """

    neomod = _model([
        (52, 27, 16, 1, 1.0, shares(nu6=0.2, _3_1=0.8)),
        (52, 28, 16, 1, 1.0, shares(nu6=0.6, _5_2=0.4)),
        ])

    a = np.array([2.65, 2.65, 2.65, 2.75, 5.0])
    e = np.full(5, 0.62)
    incl = np.full(5, np.radians(1.0))

    clones = nm.cloneSourceRegions(neomod, a, e, incl, nominal=(2.65, 0.62, np.radians(1.0)))

    assert clones['n'] == 5
    assert clones['n_in_model'] == 4
    assert clones['n_bins'] == 2
    assert clones['share_in_nominal_bin'] == pytest.approx(0.6)
    assert clones['mean']['nu6'] == pytest.approx((3*0.2 + 0.6)/4)
    assert clones['mean']['3:1'] == pytest.approx(3*0.8/4)
    assert clones['mean']['5:2'] == pytest.approx(0.4/4)
    assert clones['low']['nu6'] == pytest.approx(np.percentile([0.2, 0.2, 0.2, 0.6], 2.5))
    assert clones['high']['5:2'] == pytest.approx(np.percentile([0.0, 0.0, 0.0, 0.4], 97.5))



def testSeriesOverAnIntegrationIsSavedAndReported(tmp_path):
    """ The series of an integration: the nominal and the clones at each output, those outside the
        model left out of the mean, and the JSON, CSV, report and figure made from it. """

    neomod = _model([
        (52, 27, 16, 1, 1.0, shares(nu6=0.2, _3_1=0.8)),
        (52, 28, 16, 1, 1.0, shares(nu6=0.6, _5_2=0.4)),
        ])

    time_days = np.array([0.0, -10.0, -1000.0])
    i0 = np.radians(1.0)

    # At the first output the object is still at the Earth, on a hyperbolic orbit
    nominal = [[-1.0, 1.1, i0], [2.65, 0.62, i0], [2.75, 0.62, i0]]
    clones = [
        [[-1.0, 1.1, i0], [2.65, 0.62, i0], [2.65, 0.62, i0]],
        [[-1.0, 1.1, i0], [2.75, 0.62, i0], [5.0, 0.62, i0]],
        ]

    series = nm.sourceRegionSeries(neomod, time_days, nominal, clones=clones)

    assert np.all(np.isnan(series['nominal'][0]))
    assert "not bound" in series['nominal_reason'][0]
    assert series['nominal'][1, 1] == 0.8
    assert series['nominal'][2, 2] == 0.4
    assert series['clones_in_model'] == pytest.approx([0.0, 1.0, 0.5])
    assert series['clones_mean'][1, 0] == pytest.approx(0.4)
    assert series['clones_mean'][2, 1] == pytest.approx(0.8)
    assert np.all(np.isnan(series['clones_mean'][0]))

    # JSON, with null where there is no value
    saved = json.loads(json.dumps(nm.sourceRegionSeriesToJSON(series, mode='all', neomod=neomod)))
    assert saved['nominal']['3:1'] == [None, 0.8, 0.0]
    assert saved['clones_in_model'] == [0.0, 1.0, 0.5]
    assert saved['h_bin'] == [27.75, 28.0]
    assert saved['model_name'] == "TestModel"

    # CSV, one row per output and source
    csv_path = str(tmp_path/"sources.csv")
    nm.writeSourceRegionCSV(series, csv_path)
    with open(csv_path) as f:
        rows = f.read().splitlines()
    assert len(rows) == 1 + 3*12
    assert rows[1 + 12 + 1].startswith("-10.000000,3:1,0.8000,")

    # The report starts where it is told to, after the departure from the Earth
    lines = nm.sourceRegionReportLines(series, first=1, first_note="leaving the Earth")
    assert any("t = -10.00 d (leaving the Earth)" in line for line in lines)
    assert any(line.lstrip().startswith("3:1") for line in lines)
    assert any("50.0% of the 2 clones are in the model" in line for line in lines)

    png_path = str(tmp_path/"sources.png")
    nm.plotSourceRegionSeries(series, png_path)
    assert os.path.getsize(png_path) > 0



def testDownloadKeepsOnlyAFileThatMatchesItsHash(tmp_path):
    """ The download goes through the downloader of wmpl.Utils.DebiasedNEOPopulation with the SSL
        context of the authors' server: a matching file is kept, any other is refused. """

    source = str(tmp_path/"published.dat")
    writeModel(source, [(52, 27, 16, 1, 1.0, shares(_3_1=1.0))])
    url = "file://" + source

    good = dict(nm.NEOMOD2, sha256=_sha256(source))
    target = str(tmp_path/"kept.dat")
    path = nm.downloadNEOMOD(good, file_path=target, url=url, verbose=False)
    assert path == target
    assert _sha256(path) == good['sha256']

    bad = dict(nm.NEOMOD2, sha256="0"*64)
    target = str(tmp_path/"refused.dat")
    with pytest.raises(IOError):
        nm.downloadNEOMOD(bad, file_path=target, url=url, verbose=False)
    assert not os.path.exists(target)
    assert not [f for f in os.listdir(str(tmp_path)) if f.endswith('.part')]



def testTheIntermediateCertificateOfTheServerIsAValidCertificate():
    """ The certificate given for the authors' server loads into an SSL context that still verifies
        the chain and the host name. """

    context = ssl.create_default_context()
    context.load_verify_locations(cadata=nm.SERVER_INTERMEDIATE_PEM)

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname



def testThePublishedFileWhenItHasBeenDownloaded():
    """ The published NEOMOD2 file: its sources, its name, and the shares of its faintest bin over the
        whole population, which are those of Table 2 of NEOMOD2 at H = 28 within the difference between
        the best fit and the median (nu6 0.595, 3:1 0.313). """

    path = nm.findNEOMOD()
    if path is None:
        pytest.skip("the NEOMOD2 model file has not been downloaded")

    neomod = nm.loadNEOMOD(file_path=path, download=False)

    assert neomod['model_name'] == "Model136"
    assert neomod['sources'] == SOURCES
    assert len(neomod['n_neo']) == 566828

    rows = neomod['row_of'][51][neomod['row_of'][51] >= 0]
    weights = neomod['n_neo'][rows]
    faint_shares = np.sum(weights[:, None]*neomod['shares'][rows], axis=0)/np.sum(weights)

    assert faint_shares[0] == pytest.approx(0.636, abs=0.001)
    assert faint_shares[1] == pytest.approx(0.299, abs=0.001)

    # Winchcombe in the faintest bin: the 3:1 first
    res = nm.sourceRegions(neomod, 2.586, 0.618, np.radians(0.46))
    assert max(res['probabilities'], key=res['probabilities'].get) == '3:1'
