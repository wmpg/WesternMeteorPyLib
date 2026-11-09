""" Tests for the MSIS atmosphere mass density in wmpl.Utils.AtmosphereDensity.

The density is computed with pymsis, NRLMSIS 2.1 with the measured solar and geomagnetic activity by
default. The tests pin NRLMSISE-00 with the constant activity against the bundled Python NRLMSISE-00 port,
called with the local solar time the model expects, and check the scalar and array interfaces, the
polynomial fit, the space weather and the command line options which select the model and the date.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_AtmosphereDensity.py -v
"""

from __future__ import print_function, division, absolute_import

import argparse
import datetime
import multiprocessing
import os

import numpy as np
import pytest

import pymsis.utils

from wmpl.Utils.AtmosphereDensity import MSIS_JD_ENV, MSIS_VERSION_ENV, addAtmosphereArguments, atmDensPoly, \
    atmDensityTableInfo, fitAtmPoly, getAtmDensity, getAtmDensity_vect, getAtmTemperature, getMSISVersion, \
    msisModelName, setAtmosphere, spaceWeather, spaceWeatherArrays, SPACE_WEATHER_STATE
from wmpl.Utils.TrajConversions import datetime2JD


# Reference location and time
LAT = np.radians(44.327)
LON = np.radians(-81.372)
DT_REF = datetime.datetime(2020, 4, 20, 16, 15, 0)
JD_REF = datetime2JD(DT_REF)



def setAtmosphereOptions(*args):
    """ Set the atmosphere with the given command line options.

    Arguments:
        *args: [list] Command line options.

    Return:
        None
    """

    arg_parser = argparse.ArgumentParser()
    addAtmosphereArguments(arg_parser)
    setAtmosphere(arg_parser.parse_args(list(args)))



def keepSpaceWeatherState(monkeypatch):
    """ Restore the space weather state of the module after the test, and clear its warnings.

    Arguments:
        monkeypatch: [MonkeyPatch] Fixture of the test.

    Return:
        None
    """

    for key, value in list(SPACE_WEATHER_STATE.items()):
        monkeypatch.setitem(SPACE_WEATHER_STATE, key, value)

    SPACE_WEATHER_STATE['warned'] = set()



def referenceMSIS(height, dt=DT_REF):
    """ Compute the total mass density and temperature at the reference location with the bundled
        NRLMSISE-00 port.

    Arguments:
        height: [float] Height (m).

    Keyword arguments:
        dt: [datetime] Date and time (UTC). DT_REF by default.

    Return:
        (density, temperature): [tuple] Mass density (kg/m^3) and temperature (K).
    """

    nrlmsise_00 = pytest.importorskip('wmpl.PythonNRLMSISE00.nrlmsise_00')
    header = pytest.importorskip('wmpl.PythonNRLMSISE00.nrlmsise_00_header')
    gtd7 = nrlmsise_00.gtd7
    nrlmsise_input, nrlmsise_flags, nrlmsise_output = header.nrlmsise_input, header.nrlmsise_flags, \
        header.nrlmsise_output

    inp = nrlmsise_input()
    flags = nrlmsise_flags()
    out = nrlmsise_output()

    inp.year = 0
    inp.doy = dt.timetuple().tm_yday
    inp.sec = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).seconds
    inp.alt = height/1000
    inp.g_lat = np.degrees(LAT)
    inp.g_long = np.degrees(LON)

    # Local solar time consistent with the UT second and the longitude, as the model requires
    inp.lst = inp.sec/3600 + inp.g_long/15

    inp.f107A = 150
    inp.f107 = 150
    inp.ap = 4

    # Output in kilograms and meters, all switches on
    for i in range(24):
        flags.switches[i] = 1

    gtd7(inp, flags, out)

    return out.d[5], out.t[1]



def testMSIS00MatchesTheBundledNRLMSISEPort():
    """ Check that NRLMSISE-00 with the constant activity is the model of the bundled port, for both the
        density and the temperature.
    """

    try:
        setAtmosphereOptions('--atm', '00', '--atmsw', 'constant')

        for height in np.arange(20000, 180001, 5000):

            dens_ref, temp_ref = referenceMSIS(height)

            assert abs(getAtmDensity(LAT, LON, float(height), JD_REF)/dens_ref - 1) < 1e-4
            assert abs(getAtmTemperature(LAT, LON, float(height), JD_REF)/temp_ref - 1) < 1e-4

    finally:
        setAtmosphereOptions()



def testScalarAndArrayInputsAgree():
    """ Check that scalars return floats and arrays are evaluated point by point and keep their shape. """

    height_arr = np.linspace(20000, 180000, 12)

    dens_scalar = [getAtmDensity(LAT, LON, float(ht), JD_REF) for ht in height_arr]
    dens_arr = getAtmDensity(LAT, LON, height_arr, JD_REF)

    assert isinstance(dens_scalar[0], float)
    assert dens_arr.shape == height_arr.shape
    np.testing.assert_allclose(dens_arr, dens_scalar, rtol=1e-12)

    # Latitudes and longitudes can be given per point, and the shape of the input is preserved
    lat_arr = LAT + np.zeros((3, 4))
    lon_arr = LON + np.zeros((3, 4))
    assert getAtmDensity(lat_arr, lon_arr, height_arr[:12].reshape(3, 4), JD_REF).shape == (3, 4)

    # The vectorized alias is the same function
    assert getAtmDensity_vect is getAtmDensity



def testFitAtmPolyReproducesTheProfile():
    """ Check that the fitted polynomial follows the MSIS profile over the fitted height range to 10%.
    """

    dens_co = fitAtmPoly(LAT, LON, 60000, 180000, JD_REF)

    height_arr = np.linspace(60000, 180000, 100)
    dens_msis = getAtmDensity(LAT, LON, height_arr, JD_REF)

    np.testing.assert_allclose(atmDensPoly(height_arr, dens_co), dens_msis, rtol=0.1)



def testAtmosphereArgumentsSelectTheModelAndTheDate():
    """ Check that NRLMSIS 2.1 is the default, that --atm selects the MSIS version and that --atmtime
        overrides the date of the evaluation.
    """

    dt_other = datetime.datetime(2020, 4, 20, 4, 15, 0)

    assert getMSISVersion() == "2.1"

    try:

        # Check that NRLMSIS 2.1 gives significantly lower densities at 120 km than NRLMSISE-00
        setAtmosphereOptions('--atmsw', 'constant')
        dens_msis21 = getAtmDensity(LAT, LON, 120000.0, JD_REF)
        setAtmosphereOptions('--atm', '00', '--atmsw', 'constant')
        assert dens_msis21/getAtmDensity(LAT, LON, 120000.0, JD_REF) < 0.9

        # Check that the date given on the command line is used instead of the one passed to the function
        dens_other_time = getAtmDensity(LAT, LON, 120000.0, datetime2JD(dt_other))
        setAtmosphereOptions('--atm', '00', '--atmsw', 'constant', '--atmtime', \
            dt_other.strftime("%Y%m%d-%H%M%S"))
        assert getAtmDensity(LAT, LON, 120000.0, JD_REF) == dens_other_time

    finally:

        # Restore the defaults for the other tests
        setAtmosphereOptions()

    assert getMSISVersion() == "2.1"



def testTheMeasuredSpaceWeatherIsUsedAndRecorded():
    """ Check that the measured F10.7 and Ap of the date are used and recorded, if the space weather file
        is available.
    """

    f107, f107a, ap, source = spaceWeather(JD_REF)

    if source != 'real':
        pytest.skip("The space weather file is not available")

    # The Sun was quiet in April 2020
    assert 60 < f107 < 80
    assert 60 < f107a < 80

    info = atmDensityTableInfo(LAT, LON, JD_REF)
    assert info['model'] == msisModelName()
    assert info['space_weather'] == 'real'
    assert info['f107'] == f107

    # Check that the quiet Sun gives a lower density at 150 km than the constant activity
    dens_real = getAtmDensity(LAT, LON, 150000.0, JD_REF)
    try:
        setAtmosphereOptions('--atmsw', 'constant')
        assert dens_real/getAtmDensity(LAT, LON, 150000.0, JD_REF) < 0.97
    finally:
        setAtmosphereOptions()



def testWithoutTheSpaceWeatherFileTheConstantValuesAreUsedAndItIsDownloadedOnce(monkeypatch, tmp_path):
    """ Check that without the space weather file, which cannot be downloaded, the constant values are used
        and the download is tried only once in the process, also for many calls.
    """

    downloads = []

    def failingDownload():
        """ Count the download attempts and fail. """

        downloads.append(1)
        raise OSError("offline")

    monkeypatch.setitem(spaceWeather.__globals__, 'spaceWeatherFile', lambda: tmp_path/'missing.csv')
    monkeypatch.setattr(pymsis.utils, 'download_f107_ap', failingDownload)
    keepSpaceWeatherState(monkeypatch)
    SPACE_WEATHER_STATE['checked'] = False
    SPACE_WEATHER_STATE['refreshed'] = False

    assert spaceWeather(JD_REF) == (150.0, 150.0, 4.0, 'constant')

    for ht in np.linspace(80000, 120000, 20):
        assert np.isfinite(getAtmDensity(LAT, LON, ht, JD_REF))

    assert atmDensityTableInfo(LAT, LON, JD_REF)['space_weather'] == 'constant'
    assert len(downloads) == 1



def testPredictedAndMissingDatesAreLabelled(monkeypatch):
    """ Check that dates the space weather file only predicts are labelled 'predicted', after one refresh,
        and that dates outside its daily values, before 1957 or in its monthly predictions, get the
        constant values without more downloads.
    """

    f107, f107a, ap, source = spaceWeather(JD_REF)
    if source != 'real':
        pytest.skip("The space weather file is not available")

    downloads = []

    def noDownload():
        """ Count the refreshes without downloading. """

        downloads.append(1)
        raise OSError("not refreshed in the test")

    monkeypatch.setattr(pymsis.utils, 'download_f107_ap', noDownload)
    keepSpaceWeatherState(monkeypatch)
    SPACE_WEATHER_STATE['refreshed'] = False

    predicted_start = SPACE_WEATHER_STATE['predicted_start']
    end = SPACE_WEATHER_STATE['end']
    dates = np.array([np.datetime64('2020-04-20T16:15'), predicted_start + np.timedelta64(12, 'h'), \
        end + np.timedelta64(30, 'D'), np.datetime64('1950-01-01')])

    _, _, _, sources = spaceWeatherArrays(dates)

    assert sources[0] == 'real'
    assert sources[2] == 'constant'
    assert sources[3] == 'constant'
    if predicted_start < end:
        assert sources[1] == 'predicted'

    assert len(downloads) == 1

    # Check that the densities of the dates outside the daily values are computed
    for date in (end + np.timedelta64(30, 'D'), np.datetime64('1950-01-01')):
        jd = datetime2JD(date.astype('datetime64[s]').astype(datetime.datetime))
        assert np.isfinite(getAtmDensity(LAT, LON, 100000.0, jd))

    assert len(downloads) == 1



def testNonFiniteInputsGiveNaN():
    """ Check that a non-finite height gives NaN for that point only.
    """

    height_arr = np.array([80000.0, np.nan, 100000.0])
    dens = getAtmDensity(LAT, LON, height_arr, JD_REF)

    assert np.isnan(dens[1])
    assert np.all(np.isfinite(dens[[0, 2]]))
    assert np.isnan(getAtmDensity(LAT, LON, np.nan, JD_REF))



def testABadAtmosphereTimeIsAParserError():
    """ Check that a malformed --atmtime is rejected by the command line parser.
    """

    arg_parser = argparse.ArgumentParser()
    addAtmosphereArguments(arg_parser)

    with pytest.raises(SystemExit):
        arg_parser.parse_args(['--atmtime', '2020-04-20'])



def childAtmosphere(_):
    """ Return the model and a density as seen by a process started by multiprocessing, the density also
        depending on the date.

    Arguments:
        _: [int] Unused argument of the pool map.

    Return:
        (version, density): [tuple] MSIS version and the density at 85 km (kg/m^3).
    """

    return getMSISVersion(), getAtmDensity(LAT, LON, 85000.0, JD_REF)



def testAProcessStartedBySpawnUsesTheChosenAtmosphere():
    """ Check that a process started with "spawn", the default on macOS and Windows, which imports the module
        afresh, uses the model and date chosen in its parent through the environment.
    """

    default = childAtmosphere(0)

    try:
        setAtmosphereOptions('--atm', '00', '--atmtime', '20240101-000000')
        parent = childAtmosphere(0)

        with multiprocessing.get_context("spawn").Pool(1) as pool:
            child = pool.map(childAtmosphere, [0])[0]

        assert parent == child
        assert child[0] == "00"
        assert abs(child[1]/default[1] - 1) > 1e-3

    finally:
        setAtmosphereOptions()

    # Check that the defaults leave no date behind for later processes
    assert os.environ[MSIS_VERSION_ENV] == "2.1"
    assert MSIS_JD_ENV not in os.environ
