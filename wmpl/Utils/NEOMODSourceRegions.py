""" The source regions of an orbit in the NEOMOD model of near-Earth objects: the probability that a body
on that orbit came from each of the 12 sources of the model, as the authors' give_me_prob.f gives it.

NEOMOD is the orbital and absolute magnitude model of NEOs of Nesvorny et al. (2023, NEOMOD), fitted
again to ten years of Catalina Sky Survey detections in NEOMOD2 (Nesvorny et al. 2024a) and extended
with albedos and sizes in NEOMOD3 (Nesvorny et al. 2024b). Its sources are the nu6 secular resonance,
the 3:1, 5:2, 7:3, 8:3, 9:4, 11:5 and 2:1 resonances with Jupiter, the weak resonances of the inner
belt, the Hungarias, the Phocaeas and the Jupiter-family comets. The model is binned in a (0 to 4.2 AU
in 0.1 AU), e (0 to 1 in 0.04), i (0 to 88 deg in 4 deg) and H (15 to 28 in 0.25 mag), for q < 1.3 AU.
Each bin holds the number of model NEOs in it and the share of them from each source, so the
probability that a body came from a source is that share in the bin of its orbit and H. That is how
the authors mean it to be used: "Researchers interested in the probability that a specific NEO evolved
from a particular source can obtain this information from the ASCII table that is available along
with the Simulator" (Nesvorny et al. 2023, 2024a).

Which version. The NEOMOD Simulator page distributes three model files, input_model.dat,
input_neomod2.dat and input_neomod3.dat. Below their headers, which give the distribution in H
(NEOMOD2, in two H ranges) or in size (NEOMOD3), the three hold the same table, line for line: the
best fit of NEOMOD2, "Model136". The orbits and the source probabilities of NEOMOD, NEOMOD2 and
NEOMOD3 as distributed are therefore the same, and the first NEOMOD fit (2005-2012 survey data) is no
longer distributed. NEOMOD3 adds the albedos and sizes of the bodies, which do not enter the source
probabilities. This module downloads input_neomod2.dat; any of the three gives the same results here.

The size. The model is calibrated on the Catalina Sky Survey for 15 < H < 28, and H = 28 is a body
of about 9 m (8.7 m for an albedo of 0.147, 1.3e6 kg for 3500 kg/m^3). Nearly every fireball and
meteorite fall is smaller, and takes the faintest bin, 27.75 < H < 28, as the NEOMOD Simulator does for
H > 28. That is the default, as a trajectory does not give the size of the body, and the result says
so. A larger body, such as Chelyabinsk (about 19 m), needs its H, given directly or from its mass or
diameter. The choice matters. The share of each source changes with H, linearly between H = 15 and 28
(NEOMOD2, Sect. 3): nu6 brings 6% of the NEOs at H = 15 and 60% at H = 28, the inner weak resonances
18% and 1% (Table 2 of NEOMOD2). For the 51 meteorites in wmpl/share/Meteorites_with_Orbits.csv, the
largest change of the probability of a source between the bin at H = 17 and the faintest is 0.52 at the
median. In the faintest bin, nu6 is the most probable source of 36 of them, the 3:1 of 11 and the 5:2
of 4. For bodies smaller than the model reaches these are an extrapolation, and
wmpl.Utils.DebiasedMeteoroidPopulation (Deam et al. 2026) fits the strengths of 6 of these sources to
fireballs at meteoroid sizes.

What the probabilities assume:
    - The orbit is the one the body had in the NEO region, in steady state. NEOMOD gives where the NEOs
      of each source are, averaged over their lifetimes, not where a single body was at a given time.
    - The bin is taken as it is, without interpolation, as give_me_prob.f does, and the module gives
      the same values as a line-by-line transcription of give_me_prob.f for every orbit inside the
      model (checked on 220 000 orbits, 20 000 of them on the edges of the bins). An orbit near the
      edge of a bin can change source probabilities a lot with a small change of its elements: of the
      34 meteorites of that file with published errors of a, e and i, 22 have fewer than 68% of clones
      drawn from those errors in the bin of the nominal orbit. Clones of the orbit within its
      uncertainty (--clones here, REBOUND.py --mc --neomod) show how much it matters.
    - The shares of the sources are given to three decimals in the file, and sum to 1 within 0.004.
      They are returned as they are.
    - The orbit is in the model only if it is bound, q < 1.3 AU, a <= 4.2 AU and i <= 88 deg. A bin
      inside these limits that holds no model NEO gives no probabilities either.
    - Averaged over clones, each clone weighs the same: the mean is over the uncertainty of the orbit,
      not weighted by how many NEOs the model puts in each bin.

The file is not distributed with wmpl. It is downloaded on first use, 62 MB, from the authors' page,
checked against its SHA-256 and kept in wmpl/share (or ~/.cache/wmpl if that cannot be written), as
wmpl.Utils.DebiasedNEOPopulation does for Granvik et al. (2018). The authors' server does not send the
intermediate certificate of its chain, so the module supplies it (SERVER_INTERMEDIATE_PEM). No licence
is stated on the page. Cite the NEOMOD2 paper when using it.

    Nesvorny, D., Deienno, R., Bottke, W. F., Jedicke, R., Naidu, S., Chesley, S. R., Chodas, P. W.,
    Granvik, M., Vokrouhlicky, D., Broz, M., Morbidelli, A., Christensen, E., Shelly, F. E. & Bolin,
    B. T. (2023), NEOMOD: A New Orbital Distribution Model for Near-Earth Objects, AJ 166, 55,
    doi:10.3847/1538-3881/ace040

    Nesvorny, D., Vokrouhlicky, D., Shelly, F., Deienno, R., Bottke, W. F., Christensen, E., Jedicke,
    R., Naidu, S., Chesley, S. R., Chodas, P. W., Farnocchia, D. & Granvik, M. (2024a), NEOMOD 2: An
    updated model of Near-Earth Objects from a decade of Catalina Sky Survey observations, Icarus 411,
    115922, doi:10.1016/j.icarus.2023.115922

    Nesvorny, D., Vokrouhlicky, D., Shelly, F., Deienno, R., Bottke, W. F., Fuls, C., Jedicke, R.,
    Naidu, S., Chesley, S. R., Chodas, P. W., Farnocchia, D. & Delbo, M. (2024b), NEOMOD 3: The
    debiased size distribution of Near Earth Objects, Icarus 417, 116110,
    doi:10.1016/j.icarus.2024.116110

Usage:

    python -m wmpl.Utils.NEOMODSourceRegions path/to/trajectory.pickle
    python -m wmpl.Utils.NEOMODSourceRegions path/to/trajectory.pickle --clones 1000 --seed 1
    python -m wmpl.Utils.NEOMODSourceRegions --a 2.59 --e 0.62 --i 0.5 --size 20 28
    python -m wmpl.Utils.NEOMODSourceRegions path/to/trajectory.pickle --size 19m --save sources.json

wmpl.Rebound.REBOUND takes --neomod to give them at every output of an integration, or at its end only.
"""

from __future__ import print_function, division, absolute_import

import os
import ssl
import json
import textwrap
from urllib.error import URLError

import numpy as np

from wmpl.Config import config
from wmpl.Utils.DebiasedNEOPopulation import CACHE_DIR, findDebiasedNEOPopulation, \
    downloadDebiasedNEOPopulation
from wmpl.Utils.DebiasedMeteoroidPopulation import ALBEDO, parseMeteoroidSize
from wmpl.Utils.SourceRegions import IN_MODEL, OUTSIDE, EMPTY_BIN, CLONE_INTERVAL, PLOT_COLORS, wrapped, \
    orderedSources, reboundSourceRegionSeries, writeSourceRegionCSV, sourceRegionReportLines, \
    plotSourceRegionSeries
from wmpl.Utils.SourceRegions import sourceRegionSeries as seriesOfModel, \
    sourceRegionSeriesToJSON as seriesToJSON


# NEOMOD2 as the NEOMOD Simulator page distributes it, in the form wmpl.Utils.DebiasedNEOPopulation
#   finds, downloads and checks a model file
NEOMOD2 = {
    'name': 'NEOMOD2 (Nesvorny et al. 2024)',
    'file_name': 'Nesvorny+_2024_NEOMOD2.dat',
    'share_path': config.neomod2_file,
    'url': 'https://www2.boulder.swri.edu/~davidn/NEOMOD_Simulator/input_neomod2.dat',
    'sha256': '1e34265a932cdd4f028657e88f0e86019c8fa91da3b42d6aafe8e8739c961442',
    'size_mb': 62,
    'reference': "Nesvorny, D., Vokrouhlicky, D., Shelly, F., et al. (2024), NEOMOD 2: An updated model "
        "of Near-Earth Objects from a decade of Catalina Sky Survey observations, Icarus 411, 115922, "
        "doi:10.1016/j.icarus.2023.115922, and Nesvorny, D., Deienno, R., Bottke, W. F., et al. (2023), "
        "NEOMOD: A New Orbital Distribution Model for Near-Earth Objects, AJ 166, 55, "
        "doi:10.3847/1538-3881/ace040",
    'short_reference': "Nesvorny et al. (2024), Icarus 411, 115922; Nesvorny et al. (2023), AJ 166, 55",
    }

# The intermediate certificate of the authors' server, which the server does not send with its own, so
#   that OpenSSL cannot verify it without it. It is DigiCert Global G2 TLS RSA SHA256 2020 CA1, valid to
#   2031-03-29, from http://cacerts.digicert.com/DigiCertGlobalG2TLSRSASHA2562020CA1-1.crt (SHA-256
#   fingerprint C8:02:5F:9F:C6:5F:DF:C9:5B:3C:A8:CC:78:67:B9:A5:87:B5:27:79:73:95:79:17:46:3F:C8:13:D0:
#   B6:25:A9). It only completes the chain: the chain is still verified up to a root of the system, and
#   the host name is checked
SERVER_INTERMEDIATE_PEM = """-----BEGIN CERTIFICATE-----
MIIEyDCCA7CgAwIBAgIQDPW9BitWAvR6uFAsI8zwZjANBgkqhkiG9w0BAQsFADBh
MQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3
d3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBH
MjAeFw0yMTAzMzAwMDAwMDBaFw0zMTAzMjkyMzU5NTlaMFkxCzAJBgNVBAYTAlVT
MRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxMzAxBgNVBAMTKkRpZ2lDZXJ0IEdsb2Jh
bCBHMiBUTFMgUlNBIFNIQTI1NiAyMDIwIENBMTCCASIwDQYJKoZIhvcNAQEBBQAD
ggEPADCCAQoCggEBAMz3EGJPprtjb+2QUlbFbSd7ehJWivH0+dbn4Y+9lavyYEEV
cNsSAPonCrVXOFt9slGTcZUOakGUWzUb+nv6u8W+JDD+Vu/E832X4xT1FE3LpxDy
FuqrIvAxIhFhaZAmunjZlx/jfWardUSVc8is/+9dCopZQ+GssjoP80j812s3wWPc
3kbW20X+fSP9kOhRBx5Ro1/tSUZUfyyIxfQTnJcVPAPooTncaQwywa8WV0yUR0J8
osicfebUTVSvQpmowQTCd5zWSOTOEeAqgJnwQ3DPP3Zr0UxJqyRewg2C/Uaoq2yT
zGJSQnWS+Jr6Xl6ysGHlHx+5fwmY6D36g39HaaECAwEAAaOCAYIwggF+MBIGA1Ud
EwEB/wQIMAYBAf8CAQAwHQYDVR0OBBYEFHSFgMBmx9833s+9KTeqAx2+7c0XMB8G
A1UdIwQYMBaAFE4iVCAYlebjbuYP+vq5Eu0GF485MA4GA1UdDwEB/wQEAwIBhjAd
BgNVHSUEFjAUBggrBgEFBQcDAQYIKwYBBQUHAwIwdgYIKwYBBQUHAQEEajBoMCQG
CCsGAQUFBzABhhhodHRwOi8vb2NzcC5kaWdpY2VydC5jb20wQAYIKwYBBQUHMAKG
NGh0dHA6Ly9jYWNlcnRzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2VydEdsb2JhbFJvb3RH
Mi5jcnQwQgYDVR0fBDswOTA3oDWgM4YxaHR0cDovL2NybDMuZGlnaWNlcnQuY29t
L0RpZ2lDZXJ0R2xvYmFsUm9vdEcyLmNybDA9BgNVHSAENjA0MAsGCWCGSAGG/WwC
ATAHBgVngQwBATAIBgZngQwBAgEwCAYGZ4EMAQICMAgGBmeBDAECAzANBgkqhkiG
9w0BAQsFAAOCAQEAkPFwyyiXaZd8dP3A+iZ7U6utzWX9upwGnIrXWkOH7U1MVl+t
wcW1BSAuWdH/SvWgKtiwla3JLko716f2b4gp/DA/JIS7w7d7kwcsr4drdjPtAFVS
slme5LnQ89/nD/7d+MS5EHKBCQRfz5eeLjJ1js+aWNJXMX43AYGyZm0pGrFmCW3R
bpD0ufovARTFXFZkAdl9h6g4U5+LXUZtXMYnhIHUfoyMo5tS58aI7Dd8KvvwVVo4
chDYABPPTHPbqjc1qCmBaZx2vN4Ye5DUys/vZwP9BFohFrH/6j/f3IL16/RZkiMN
JCqVJUzKoZHm1Lesh3Sz8W2jmdv51b2EQJ8HmA==
-----END CERTIFICATE-----
"""

# The number of header lines of the model files, the column names being the 12th
HEADER_LINES = 13

# The names of the sources to print, by the names the header of the file gives them
SOURCE_NAMES = {
    'nu6': 'nu6',
    '3:1': '3:1',
    '5:2': '5:2',
    '7:3': '7:3',
    '8:3': '8:3',
    '9:4': '9:4',
    '11:5': '11:5',
    '2:1': '2:1',
    'inner': 'inner weak res.',
    'Hun': 'Hungarias',
    'Pho': 'Phocaeas',
    'JFC': 'JFCs',
    }

# Near-Earth objects have q < 1.3 AU, and the model holds no others
NEO_Q_MAX = 1.3

# Sources below this probability everywhere are left out of the printed tables
PRINT_MIN = 0.001

# The sources drawn in the figures, in the order of the categorical palette, the rest drawn together
#   as one, so that a source keeps its colour whatever the orbit
PLOT_SOURCES = ['nu6', '3:1', '5:2', 'inner', 'Hun', 'Pho', 'JFC']

# The model, once loaded
_model_cache = {}



def findNEOMOD(model=NEOMOD2):
    """ The path of the model file if it has been downloaded, None otherwise. """

    return findDebiasedNEOPopulation(model)



def downloadNEOMOD(model=NEOMOD2, file_path=None, url=None, verbose=True):
    """ Download the model file from the authors' page and check it against its SHA-256 (see
        wmpl.Utils.DebiasedNEOPopulation.downloadDebiasedNEOPopulation).

    Keyword arguments:
        model: [dict] The model, NEOMOD2 by default.
        file_path: [str] Where to write it. wmpl/share by default, or ~/.cache/wmpl if that cannot be
            written.
        url: [str] Where to download it from. The authors' page by default.
        verbose: [bool] Print what is being downloaded. True by default.

    Return:
        [str] The path of the file.
    """

    url = model['url'] if url is None else url

    if verbose:
        print("Downloading the {:s} model file ({:d} MB), once, from".format(model['name'],
            model['size_mb']))
        print("  {:s}".format(url))

    # The default verification, with the intermediate certificate the server leaves out
    context = ssl.create_default_context()
    context.load_verify_locations(cadata=SERVER_INTERMEDIATE_PEM)

    try:
        path = downloadDebiasedNEOPopulation(model, file_path=file_path, url=url, verbose=False,
            context=context)

    except URLError as err:
        raise IOError("Could not download {:s}: {:s}. Download it with a web browser and give it with "
            "file_path (--model_file on the command line), or put it at {:s}.".format(url,
            str(err.reason), model['share_path']))

    if verbose:
        print("  to {:s}".format(path))
        for line in wrapped("Please cite {:s}.".format(model['reference'])):
            print(line)

    return path



def _readHeader(file_path):
    """ The binning, the name and the sources of a NEOMOD model file, from its header. """

    with open(file_path) as f:
        lines = [f.readline() for _ in range(HEADER_LINES)]

    def _numbers(line, n):
        return [float(v) for v in line.split('!')[0].split()[:n]]

    n_h, h_min, h_max = _numbers(lines[1], 3)
    n_a, a_min, a_max = _numbers(lines[6], 3)
    n_e, e_min, e_max = _numbers(lines[7], 3)
    n_i, i_min, i_max = _numbers(lines[8], 3)
    n_sources = int(_numbers(lines[9], 1)[0])

    # The column names end with the sources
    names = lines[11].split()

    if len(names) < n_sources + 5:
        raise ValueError("The header of {:s} names fewer columns than its {:d} sources need.".format(
            file_path, n_sources))

    sources = names[-n_sources:]

    return {
        'model_name': lines[0].strip(),
        'h': (int(n_h), h_min, h_max),
        'a': (int(n_a), a_min, a_max),
        'e': (int(n_e), e_min, e_max),
        'incl_deg': (int(n_i), i_min, i_max),
        'sources': sources,
        }



def loadNEOMOD(model=NEOMOD2, file_path=None, download=True, verbose=True):
    """ Load the binned NEOMOD model, downloading its file first if it is not there.

    Keyword arguments:
        model: [dict] The model, NEOMOD2 by default.
        file_path: [str] A model file to read instead of the one in wmpl/share or the cache, for
            example input_model.dat or input_neomod3.dat from the NEOMOD Simulator page. None by
            default.
        download: [bool] Download the file if it is not found. If False, raise instead. True by
            default.
        verbose: [bool] Print a note when downloading. True by default.

    Return:
        [dict] 'model' and 'file_path'; 'model_name', the name in the first line of the file;
            'sources', the short names of the sources; 'h', 'a', 'e' and 'incl_deg', the binning of
            each as (number of bins, min, max); 'row_of', the row of each bin (H, a, e, i, 0-based) in
            the table, -1 for a bin without NEOs; 'n_neo', the number of model NEOs in each row's bin;
            'shares', the share of each source in each row's bin.
    """

    if file_path is None:

        file_path = findNEOMOD(model)

        if file_path is None:

            if not download:
                raise IOError("The {:s} model file is not in {:s}. Download it from {:s}.".format(
                    model['name'], " or ".join([model['share_path'], os.path.join(CACHE_DIR,
                    model['file_name'])]), model['url']))

            file_path = downloadNEOMOD(model, verbose=verbose)

    file_path = os.path.abspath(file_path)

    if file_path in _model_cache:
        return _model_cache[file_path]

    neomod = _readHeader(file_path)

    # Columns: the bin in H, a, e and i (from 1), the number of NEOs in it, and the share of each source
    data = np.loadtxt(file_path, skiprows=HEADER_LINES, ndmin=2)

    shape = (neomod['h'][0], neomod['a'][0], neomod['e'][0], neomod['incl_deg'][0])

    index = data[:, :4].astype(int) - 1

    for k in range(4):
        if (index[:, k].min() < 0) or (index[:, k].max() >= shape[k]):
            raise ValueError("A bin of {:s} lies outside the binning its header gives.".format(file_path))

    row_of = np.full(shape, -1, dtype=np.int32)
    row_of[index[:, 0], index[:, 1], index[:, 2], index[:, 3]] = np.arange(len(data), dtype=np.int32)

    neomod.update({
        'model': model,
        'file_path': file_path,
        'row_of': row_of,
        'n_neo': data[:, 4],
        'shares': data[:, 5:],
        })

    _model_cache[file_path] = neomod

    return neomod



def hBin(neomod, h_mag=None):
    """ The H bin of the model used for a body of absolute magnitude h_mag.

        The model covers H from 15 to 28. A body outside that range takes the bin at the end of it, as
        the NEOMOD Simulator does, and a meteoroid, given without H, the faintest bin.

    Arguments:
        neomod: [dict] Return of loadNEOMOD.

    Keyword arguments:
        h_mag: [float] The absolute magnitude. None for a meteoroid, by default.

    Return:
        [dict] 'index', the bin (0-based); 'range', its H limits; 'h_mag', the H given; 'extrapolated',
            True if H was outside the model or not given.

    Reference: the calibrated range 15 < H < 28 and the last bin used beyond it, Nesvorny et al. (2024),
        Icarus 411, 115922, Sect. 3, and neomod2_simulator.f.
    """

    n_h, h_min, h_max = neomod['h']
    dh = (h_max - h_min)/n_h

    if h_mag is None:
        index = n_h - 1
        extrapolated = True

    else:
        index = int(np.floor((h_mag - h_min)/dh))
        extrapolated = (h_mag < h_min) or (h_mag > h_max)
        index = min(max(index, 0), n_h - 1)

    return {'index': index, 'range': (h_min + index*dh, h_min + (index + 1)*dh), 'h_mag': h_mag,
        'extrapolated': extrapolated}



def diameterToH(diameter, albedo=ALBEDO):
    """ The absolute magnitude of a body from its diameter, D = 1329 km 10^(-H/5)/sqrt(p_V), with the
        albedo wmpl.Utils.DebiasedMeteoroidPopulation.massToH takes for a mass. 8.7 m is H = 28.

    Arguments:
        diameter: [float or ndarray] Diameter (m).

    Keyword arguments:
        albedo: [float] Geometric albedo. ALBEDO by default.

    Return:
        [float or ndarray] H.

    Reference: the relation neomod3_simulator.f uses for NEOMOD3, Nesvorny et al. (2024), Icarus 417,
        116110, doi:10.1016/j.icarus.2024.116110.
    """

    return 5.0*np.log10(1329.0/(np.asarray(diameter, dtype=float)/1000.0*np.sqrt(albedo)))



def parseBodySize(value):
    """ The absolute magnitude of a body from a command-line value: H itself, a mass ending in kg or g
        (see wmpl.Utils.DebiasedMeteoroidPopulation.parseMeteoroidSize), or a diameter ending in m.

    Arguments:
        value: [str] For example '26.5', '13kg', '600g' or '19m'.

    Return:
        (h_mag, description): [tuple] H, and what it was computed from, or None if H was given.
    """

    text = str(value).strip().lower()

    if text.endswith('m') and not text.endswith('km'):

        diameter = float(text[:-1])

        if not (diameter > 0):
            raise ValueError("The diameter has to be positive, not {:s}.".format(str(value)))

        return float(diameterToH(diameter)), "diameter {:g} m".format(diameter)

    h_mag, mass = parseMeteoroidSize(value)

    return h_mag, None if mass is None else "mass {:g} kg".format(mass)



def _binIndices(neomod, a, e, incl):
    """ The bins in a, e and i (0-based) of orbits, as give_me_prob.f finds them, and whether each orbit
        is in the region of the model.

    Arguments:
        neomod: [dict] Return of loadNEOMOD.
        a: [ndarray] Semi-major axes (AU).
        e: [ndarray] Eccentricities.
        incl: [ndarray] Inclinations (rad).

    Return:
        (ia, ie, ii, inside): [tuple] The bins, 0 where the orbit is outside, and the mask of the orbits
            inside.
    """

    # Rounded to 1e-9 deg, so that an inclination given in degrees on the edge of a bin (60 deg) falls
    #   in the bin give_me_prob.f puts it in after the round trip through radians (59.99999999999999)
    incl_deg = np.round(np.degrees(incl), 9)

    with np.errstate(invalid='ignore'):

        q = a*(1.0 - e)

        inside = np.isfinite(a) & np.isfinite(e) & np.isfinite(incl_deg) & (e < 1.0) & (q < NEO_Q_MAX)

        indices = []
        for values, (n, v_min, v_max) in zip((a, e, incl_deg), (neomod['a'], neomod['e'],
                neomod['incl_deg'])):

            inside &= (values >= v_min) & (values <= v_max)

            # The bin as give_me_prob.f computes it, the upper limit falling in the last bin
            dv = (v_max - v_min)/n
            k = np.zeros(len(values), dtype=int)
            k[inside] = np.minimum(np.floor((values[inside] - v_min)/dv).astype(int), n - 1)

            indices.append(k)

    # The orbits found outside by a later element have to be reset in the earlier ones
    ia, ie, ii = [np.where(inside, k, 0) for k in indices]

    return ia, ie, ii, inside



def sourceProbabilities(neomod, a, e, incl, h_mag=None):
    """ The probability that a body on each of the given orbits came from each source of the model:
        the share of the source in the bin of the orbit and of H.

    Arguments:
        neomod: [dict] Return of loadNEOMOD.
        a: [float or ndarray] Semi-major axis (AU).
        e: [float or ndarray] Eccentricity.
        incl: [float or ndarray] Inclination (rad).

    Keyword arguments:
        h_mag: [float] The absolute magnitude of the body, one for all the orbits. None for a meteoroid,
            which takes the faintest bin of the model (see hBin), by default.

    Return:
        (probabilities, n_neo, status, bins, h_bin):
            - probabilities: [ndarray] The probability of each source, with a last axis over the sources
                of neomod['sources'], NaN where the orbit has none.
            - n_neo: [ndarray] The number of model NEOs in the bin of the orbit, NaN where it has none.
            - status: [ndarray] IN_MODEL, OUTSIDE or EMPTY_BIN for each orbit.
            - bins: [ndarray] The bin of each orbit as a flat index in (H, a, e, i), -1 outside.
            - h_bin: [dict] The H bin used (see hBin).

    Reference: the binned model of NEOMOD2 distributed with the NEOMOD Simulator, read as its
        give_me_prob.f reads it. Nesvorny et al. (2024), Icarus 411, 115922,
        doi:10.1016/j.icarus.2023.115922; the sources are those of Nesvorny et al. (2023), AJ 166, 55,
        doi:10.3847/1538-3881/ace040.
    """

    a, e, incl = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(e, dtype=float),
        np.asarray(incl, dtype=float))

    shape = a.shape
    a, e, incl = a.ravel(), e.ravel(), incl.ravel()

    h_bin = hBin(neomod, h_mag)

    ia, ie, ii, inside = _binIndices(neomod, a, e, incl)

    rows = np.where(inside, neomod['row_of'][h_bin['index'], ia, ie, ii], -1)

    found = rows >= 0

    n_sources = len(neomod['sources'])
    probabilities = np.full((len(a), n_sources), np.nan)
    probabilities[found] = neomod['shares'][rows[found]]

    n_neo = np.full(len(a), np.nan)
    n_neo[found] = neomod['n_neo'][rows[found]]

    status = np.full(len(a), OUTSIDE, dtype=int)
    status[inside] = EMPTY_BIN
    status[found] = IN_MODEL

    bins = np.full(len(a), -1, dtype=np.int64)
    bins[inside] = np.ravel_multi_index((np.full(np.sum(inside), h_bin['index']), ia[inside],
        ie[inside], ii[inside]), neomod['row_of'].shape)

    return probabilities.reshape(shape + (n_sources,)), n_neo.reshape(shape), status.reshape(shape), \
        bins.reshape(shape), h_bin



def outsideReason(neomod, a, e, incl, status):
    """ Why an orbit has no source probabilities, or None if it has them.

    Arguments:
        neomod: [dict] Return of loadNEOMOD.
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
        return "no model NEO lies in the bin of the orbit"

    if not all(np.isfinite(v) for v in (a, e, incl)):
        return "the orbit has no elements"

    if e >= 1.0:
        return "the orbit is not bound (e = {:.4f}), and the model holds only bound orbits".format(e)

    q = a*(1.0 - e)
    if q >= NEO_Q_MAX:
        return "q = {:.3f} AU: not a near-Earth object, which needs q < {:.1f} AU".format(q, NEO_Q_MAX)

    if a > neomod['a'][2]:
        return "a = {:.3f} AU is beyond the model, which ends at a = {:.1f} AU".format(a, neomod['a'][2])

    return "i = {:.2f} deg is beyond the model, which ends at i = {:.0f} deg".format(np.degrees(incl),
        neomod['incl_deg'][2])



def _binRanges(neomod, a, e, incl):
    """ The limits of the bin in a, e and i of an orbit inside the model. """

    ia, ie, ii, inside = _binIndices(neomod, np.array([a]), np.array([e]), np.array([incl]))

    if not inside[0]:
        return None

    ranges = {}
    for key, k in (('a', ia[0]), ('e', ie[0]), ('incl_deg', ii[0])):
        n, v_min, v_max = neomod[key]
        dv = (v_max - v_min)/n
        ranges[key] = (v_min + k*dv, v_min + (k + 1)*dv)

    return ranges



def sourceRegions(neomod, a, e, incl, h_mag=None):
    """ The source probabilities of one orbit, with what they were computed from.

    Arguments:
        neomod: [dict] Return of loadNEOMOD.
        a: [float] Semi-major axis (AU).
        e: [float] Eccentricity.
        incl: [float] Inclination (rad).

    Keyword arguments:
        h_mag: [float] The absolute magnitude of the body. None for a meteoroid, by default.

    Return:
        [dict] 'a', 'e', 'q' (AU), 'incl' (rad); 'h_bin' (see hBin); 'in_model'; 'reason', why the
            orbit has no probabilities, or None; 'bin', the limits in 'a', 'e' and 'incl_deg' of its bin,
            or None outside the model; 'n_neo_bin', the number of model NEOs in the bin; 'sources'; and
            'probabilities', {source: probability} in the order of the file, or None.
    """

    probabilities, n_neo, status, _, h_bin = sourceProbabilities(neomod, a, e, incl, h_mag=h_mag)

    status = int(status)
    in_model = status == IN_MODEL

    return {
        'a': float(a), 'e': float(e), 'q': float(a*(1.0 - e)), 'incl': float(incl),
        'h_bin': h_bin,
        'in_model': in_model,
        'reason': outsideReason(neomod, a, e, incl, status),
        'bin': _binRanges(neomod, a, e, incl),
        'n_neo_bin': float(n_neo) if in_model else None,
        'sources': list(neomod['sources']),
        'probabilities': {s: float(p) for s, p in zip(neomod['sources'], probabilities)} if in_model \
            else None,
        }



def cloneSourceRegions(neomod, a, e, incl, h_mag=None, nominal=None, interval=CLONE_INTERVAL):
    """ The source probabilities over clones of an orbit: their mean over the clones in the model, which
        is the probability of each source given the uncertainty of the orbit, and the interval they span.

    Arguments:
        neomod: [dict] Return of loadNEOMOD.
        a: [ndarray] Semi-major axes of the clones (AU).
        e: [ndarray] Eccentricities of the clones.
        incl: [ndarray] Inclinations of the clones (rad).

    Keyword arguments:
        h_mag: [float] The absolute magnitude of the body. None for a meteoroid, by default.
        nominal: [tuple] (a, e, incl) of the nominal orbit, to count the clones in its bin. None by
            default.
        interval: [tuple] Percentiles of the interval. CLONE_INTERVAL by default.

    Return:
        [dict] 'n', the clones; 'n_in_model'; 'n_bins', the bins they fall in; 'share_in_nominal_bin',
            or None without the nominal; 'interval'; and {source: value} dicts 'mean', 'low' and 'high',
            None if no clone is in the model.
    """

    probabilities, _, status, bins, _ = sourceProbabilities(neomod, a, e, incl, h_mag=h_mag)

    in_model = status == IN_MODEL
    sources = neomod['sources']

    out = {'n': int(len(status)), 'n_in_model': int(np.sum(in_model)),
        'n_bins': int(len(np.unique(bins[in_model]))), 'share_in_nominal_bin': None,
        'interval': tuple(interval), 'mean': None, 'low': None, 'high': None}

    if nominal is not None:
        _, _, _, nominal_bin, _ = sourceProbabilities(neomod, *nominal, h_mag=h_mag)
        if nominal_bin >= 0:
            out['share_in_nominal_bin'] = float(np.mean(bins == nominal_bin))

    if np.any(in_model):

        values = probabilities[in_model]
        low, high = np.percentile(values, interval, axis=0)

        out['mean'] = {s: float(v) for s, v in zip(sources, np.mean(values, axis=0))}
        out['low'] = {s: float(v) for s, v in zip(sources, low)}
        out['high'] = {s: float(v) for s, v in zip(sources, high)}

    return out



def sourceModel(neomod, h_mag=None):
    """ NEOMOD2 at the H of a body, described for wmpl.Utils.SourceRegions, whose series, report, JSON,
        CSV and figure it is given to.

    Arguments:
        neomod: [dict] Return of loadNEOMOD.

    Keyword arguments:
        h_mag: [float] The absolute magnitude of the body. None for a meteoroid, by default.

    Return:
        [dict] The model (see wmpl.Utils.SourceRegions).
    """

    h_bin = hBin(neomod, h_mag)

    def _evaluate(a, e, incl):
        probabilities, _, status, _, _ = sourceProbabilities(neomod, a, e, incl, h_mag=h_mag)
        return probabilities, status

    def _reason(a, e, incl, status):
        return outsideReason(neomod, a, e, incl, status)

    return {
        'name': NEOMOD2['name'],
        'label': "NEOMOD2",
        'reference': NEOMOD2['reference'],
        'short_reference': NEOMOD2['short_reference'],
        'sources': list(neomod['sources']),
        'source_names': SOURCE_NAMES,
        'evaluate': _evaluate,
        'reason': _reason,
        'heading': _hBinText(h_bin),
        'warning': extrapolationWarning(h_bin),
        'title_note': ", H {:.2f}-{:.2f}{:s}".format(*h_bin['range'],
            " (extrapolated)" if h_bin['extrapolated'] else ""),
        'print_min': PRINT_MIN,
        'name_width': 16,
        'plot_groups': [(SOURCE_NAMES[s], color, [s]) for s, color in zip(PLOT_SOURCES, PLOT_COLORS)],
        'json': {
            'model_name': neomod['model_name'],
            'model_file': os.path.basename(neomod['file_path']),
            'h_mag': h_bin['h_mag'],
            'h_bin': list(h_bin['range']),
            'h_extrapolated': h_bin['extrapolated'],
            'h_warning': extrapolationWarning(h_bin),
            },
        'h_bin': h_bin,
        }



def sourceRegionSeries(neomod, time_days, nominal, clones=None, h_mag=None, interval=CLONE_INTERVAL,
        horizon=None, representative=None):
    """ The source probabilities of an integrated orbit in NEOMOD2 at the H of the body (see
        wmpl.Utils.SourceRegions.sourceRegionSeries, and sourceModel).

    Arguments:
        neomod: [dict] Return of loadNEOMOD.
        time_days: [ndarray] The output times (days).
        nominal: [ndarray] a (AU), e and incl (rad) of the nominal solution at each output, one row each.

    Keyword arguments:
        clones: [ndarray] The same for the clones, (clone, output, element). None by default.
        h_mag: [float] The absolute magnitude of the body. None for a meteoroid, by default.
        interval: [tuple] Percentiles of the interval of the clones. CLONE_INTERVAL by default.
        horizon: [tuple] (time_days, nominal, clones) at the predictability horizon. None by default.
        representative: [tuple] (name, elements) of the representative clone. None by default.

    Return:
        [dict] Return of wmpl.Utils.SourceRegions.sourceRegionSeries, with 'h_bin' (see hBin).
    """

    model = sourceModel(neomod, h_mag=h_mag)

    series = seriesOfModel(model, time_days, nominal, clones=clones, interval=interval, horizon=horizon,
        representative=representative)
    series['h_bin'] = model['h_bin']

    return series



def sourceRegionSeriesToJSON(series, mode=None, neomod=None):
    """ The return of sourceRegionSeries as plain lists, for a JSON file (see
        wmpl.Utils.SourceRegions.sourceRegionSeriesToJSON).

    Arguments:
        series: [dict] Return of sourceRegionSeries.

    Keyword arguments:
        mode: [str] 'all' or 'final', which outputs were computed. None by default.
        neomod: [dict] Not used: the model file is named by the series. None by default.

    Return:
        [dict] The series.
    """

    return seriesToJSON(series, mode=mode)



def extrapolationWarning(h_bin):
    """ A warning that the H bin used extrapolates the model beyond the sizes it was calibrated for, or
        None.

    Arguments:
        h_bin: [dict] Return of hBin.

    Return:
        [str] The warning, or None.
    """

    if not h_bin['extrapolated']:
        return None

    if h_bin['h_mag'] is None:
        return "WARNING: no size was given, so the body is taken as smaller than about 9 m (H > 28). " \
            "NEOMOD was calibrated on the Catalina Sky Survey for 15 < H < 28 only: its faintest bin is " \
            "used, an extrapolation to smaller bodies. Give the size of a body larger than about 9 m."

    if h_bin['h_mag'] > h_bin['range'][1]:
        return "WARNING: H = {:.2f} is smaller than NEOMOD was calibrated for on the Catalina Sky Survey " \
            "(15 < H < 28, D > ~9 m): its faintest bin is used, an extrapolation.".format(h_bin['h_mag'])

    return "WARNING: H = {:.2f} is larger than NEOMOD was calibrated for on the Catalina Sky Survey " \
        "(15 < H < 28): its brightest bin is used, an extrapolation.".format(h_bin['h_mag'])



def _hBinText(h_bin):
    """ The H bin used, in words. """

    text = "H {:.2f}-{:.2f}".format(*h_bin['range'])

    if h_bin['h_mag'] is None:
        return text + " (no size given: the faintest bin of the model, for a body smaller than about 9 m)"

    if h_bin['extrapolated'] and (h_bin['h_mag'] > h_bin['range'][1]):
        return text + " (H = {:.2f} is fainter than the model reaches: its faintest bin)".format(
            h_bin['h_mag'])

    if h_bin['extrapolated']:
        return text + " (H = {:.2f} is brighter than the model reaches: its brightest bin)".format(
            h_bin['h_mag'])

    return text + " (H = {:.2f})".format(h_bin['h_mag'])



def _hLabel(h_bin):
    """ A short label of the H of a column. """

    return "no size" if h_bin['h_mag'] is None else "H = {:.2f}".format(h_bin['h_mag'])



def formatSourceRegions(results, clones=None, model_name=None):
    """ Lines of text giving the source probabilities of one orbit, at one or several H.

    Arguments:
        results: [list] Returns of sourceRegions, one per H, for the same orbit.

    Keyword arguments:
        clones: [list] Returns of cloneSourceRegions, one per H, or None. None by default.
        model_name: [str] The name of the fit in the model file. None by default.

    Return:
        [list] The lines.
    """

    first = results[0]

    lines = ["Source regions in {:s}{:s}:".format(NEOMOD2['name'], "" if model_name is None
        else ", {:s}".format(model_name))]

    lines.append("  Orbit: a = {:.4f} AU, e = {:.4f}, i = {:.3f} deg, q = {:.4f} AU".format(first['a'],
        first['e'], np.degrees(first['incl']), first['q']))

    if first['bin'] is not None:
        lines.append("  Bin:   a {:.1f}-{:.1f} AU, e {:.2f}-{:.2f}, i {:.0f}-{:.0f} deg".format(
            *(first['bin']['a'] + first['bin']['e'] + first['bin']['incl_deg'])))

    for k, res in enumerate(results):

        h_label = "  H:     " if k == 0 else "         "
        line = h_label + _hBinText(res['h_bin'])

        if res['in_model']:
            line += ", {:.4g} model NEOs in the bin".format(res['n_neo_bin'])

        lines.append(line)

    # The warnings of the sizes outside the model, each once
    warnings = []
    for res in results:
        warning = extrapolationWarning(res['h_bin'])
        if (warning is not None) and (warning not in warnings):
            warnings.append(warning)

    for warning in warnings:
        lines += wrapped(warning)

    if clones is not None:

        c = clones[0]
        lines.append("  Clones: {:d}, {:d} in the model, over {:d} bin(s){:s}".format(c['n'],
            c['n_in_model'], c['n_bins'], "" if c['share_in_nominal_bin'] is None
            else ", {:.1f}% in the bin of the nominal orbit".format(100*c['share_in_nominal_bin'])))

    if not any(res['in_model'] for res in results):

        lines.append("  No source probabilities: {:s}.".format(first['reason']))

        return lines


    # One column per H, with the clones' mean and interval after the nominal if there are clones
    columns = [res['probabilities'] for res in results]
    if clones is not None:
        columns += [c['mean'] for c in clones]

    order = orderedSources(first['sources'], columns, PRINT_MIN)

    lines.append("")

    header = "  {:<16s}".format("Source")
    for res in results:
        header += "  {:>11s}".format(_hLabel(res['h_bin']))
        if clones is not None:
            header += "  {:>26s}".format("clones mean [{:g}-{:g}%]".format(*clones[0]['interval']))

    lines.append(header)

    for s in order:

        line = "  {:<16s}".format(SOURCE_NAMES.get(s, s))

        for k, res in enumerate(results):

            p = None if res['probabilities'] is None else res['probabilities'][s]
            line += "  {:>11s}".format("n/a" if p is None else "{:.3f}".format(p))

            if clones is not None:
                c = clones[k]
                if c['mean'] is None:
                    line += "  {:>26s}".format("n/a")
                else:
                    line += "  {:>26s}".format("{:.3f} [{:.3f}, {:.3f}]".format(c['mean'][s], c['low'][s],
                        c['high'][s]))

        lines.append(line)

    left_out = [SOURCE_NAMES.get(s, s) for s in first['sources'] if s not in order]
    if left_out:
        lines.append("  ({:s}: below {:g})".format(", ".join(left_out), PRINT_MIN))

    for res in results:
        if not res['in_model']:
            lines.append("  At {:s}: {:s}.".format(_hLabel(res['h_bin']), res['reason']))

    return lines



if __name__ == "__main__":

    import sys
    import argparse

    from wmpl.Utils.Pickling import loadPickle


    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="The probability that a body on an orbit came from "
        "each source region of the NEOMOD model of near-Earth objects (Nesvorny et al. 2023, 2024), as the "
        "authors' give_me_prob.f gives it. The model file, 62 MB, is downloaded from the authors' page on "
        "first use and kept in wmpl/share.")

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

    arg_parser.add_argument('--size', metavar='H|MASS', type=str, nargs='+', default=None, \
        help="Absolute magnitude of the body, its mass ending in kg or g, or its diameter ending in m "
            "(converted to H with a density of 3500 kg/m^3 and an albedo of 0.147, as "
            "wmpl.Utils.DebiasedMeteoroidPopulation.massToH does; 8.7 m or 1.3e6 kg is H = 28). Several "
            "values give a column each. The model covers 15 < H < 28, so a body smaller than about 9 m "
            "takes the faintest bin, as it does without --size. A trajectory does not give the size: "
            "give it here for a body larger than that.")

    arg_parser.add_argument('--clones', metavar='N', type=int, \
        help="Give the mean and the 95%% interval of the probabilities over N clones of the orbit, drawn "
            "from the covariance of a trajectory pickle solved with the Monte Carlo uncertainties, with "
            "the draws of wmpl.Rebound.REBOUND (see wmpl.Utils.OrbitClones).")
    arg_parser.add_argument('--seed', metavar='S', type=int, \
        help="Seed of the clones, to draw the same ones again. Drawn and printed if not given.")

    arg_parser.add_argument('--model_file', metavar='FILE', type=str, default=None, \
        help="A NEOMOD model file to read instead of downloading one, for example input_neomod3.dat from "
            "the NEOMOD Simulator page.")

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

    # The absolute magnitudes, None for a meteoroid
    sizes = [(None, None)]
    if cml_args.size is not None:
        try:
            sizes = [parseBodySize(value) for value in cml_args.size]
        except ValueError as err:
            reportError("--size: {:s}".format(str(err)))

    neomod = loadNEOMOD(file_path=cml_args.model_file)

    results = [sourceRegions(neomod, a, e, incl, h_mag=h_mag) for h_mag, _ in sizes]


    # The clones, from the trajectory's covariance
    clone_results = None
    clones = None

    if cml_args.clones is not None:

        from wmpl.Utils.OrbitClones import cloneOrbits, cloneHeaderLines

        clones = cloneOrbits(traj, cml_args.clones, random_seed=cml_args.seed)

        with np.errstate(divide='ignore', invalid='ignore'):
            a_clones = clones['q']/(1.0 - clones['e'])
        a_clones[clones['e'] >= 1.0] = np.nan

        clone_results = [cloneSourceRegions(neomod, a_clones, clones['e'], clones['incl'], h_mag=h_mag,
            nominal=(a, e, incl)) for h_mag, _ in sizes]

        for line in cloneHeaderLines(clones):
            print(line)
        print()

    if cml_args.traj_path is not None:
        print("Trajectory: {:s}".format(cml_args.traj_path))

    for (h_mag, description) in sizes:
        if description is not None:
            print("{:s}: H = {:.2f}".format(description[0].upper() + description[1:], h_mag))

    for line in formatSourceRegions(results, clones=clone_results, model_name=neomod['model_name']):
        print(line)

    print()
    for line in wrapped("Please cite {:s}.".format(NEOMOD2['reference']), indent=""):
        print(line)


    # Save the results
    if cml_args.save is not None:

        def _jsonable(res):
            out = dict(res)
            out['incl_deg'] = np.degrees(out.pop('incl'))
            out['h_bin'] = dict(res['h_bin'])
            out['h_bin']['range'] = list(res['h_bin']['range'])
            return out

        saved = {
            'model': NEOMOD2['name'],
            'model_name': neomod['model_name'],
            'model_file': os.path.basename(neomod['file_path']),
            'reference': NEOMOD2['reference'],
            'trajectory': cml_args.traj_path,
            'sizes': [description for _, description in sizes],
            'results': [_jsonable(res) for res in results],
            'clones': None if clone_results is None else {'n': clones['n'], 'seed': clones['seed'],
                'by_h': clone_results},
            }

        with open(cml_args.save, 'w') as f:
            json.dump(saved, f, indent=1, default=float)

        print("Saved: {:s}".format(cml_args.save))
