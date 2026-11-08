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

from wmpl.PythonNRLMSISE00.nrlmsise_00 import gtd7
from wmpl.PythonNRLMSISE00.nrlmsise_00_header import nrlmsise_input, nrlmsise_flags, nrlmsise_output
from wmpl.Formats.GenericFunctions import addSolverOptions
import wmpl.Utils.AtmosphereDensity as AtmosphereDensity
from wmpl.Utils.AtmosphereDensity import MSIS_JD_ENV, MSIS_VERSION_ENV, addAtmosphereArguments, atmDensPoly, \
    atmDensityTableInfo, fitAtmPoly, getAtmDensity, getAtmDensity_vect, getAtmTemperature, getMSISVersion, \
    setAtmosphere, spaceWeather
from wmpl.Utils.TrajConversions import datetime2JD


# Reference location and time
LAT = np.radians(44.327)
LON = np.radians(-81.372)
DT_REF = datetime.datetime(2020, 4, 20, 16, 15, 0)
JD_REF = datetime2JD(DT_REF)



def _setAtmosphere(*args):
    """ Set the atmosphere with the given command line options.

    Arguments:
        *args: [list] Command line options.

    Return:
        None
    """

    arg_parser = argparse.ArgumentParser()
    addAtmosphereArguments(arg_parser)
    setAtmosphere(arg_parser.parse_args(list(args)))



def referenceMSIS(height, dt=DT_REF):
    """ Compute the total mass density (kg/m^3) and temperature (K) at the reference location with the
        bundled NRLMSISE-00 port.
    """

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
        _setAtmosphere('--atm', '00', '--atmsw', 'constant')

        for height in np.arange(20000, 180001, 5000):

            dens_ref, temp_ref = referenceMSIS(height)

            assert abs(getAtmDensity(LAT, LON, float(height), JD_REF)/dens_ref - 1) < 1e-4
            assert abs(getAtmTemperature(LAT, LON, float(height), JD_REF)/temp_ref - 1) < 1e-4

    finally:
        _setAtmosphere()



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
        _setAtmosphere('--atmsw', 'constant')
        dens_msis21 = getAtmDensity(LAT, LON, 120000.0, JD_REF)
        _setAtmosphere('--atm', '00', '--atmsw', 'constant')
        assert dens_msis21/getAtmDensity(LAT, LON, 120000.0, JD_REF) < 0.9

        # Check that the date given on the command line is used instead of the one passed to the function
        dens_other_time = getAtmDensity(LAT, LON, 120000.0, datetime2JD(dt_other))
        _setAtmosphere('--atm', '00', '--atmsw', 'constant', '--atmtime', \
            dt_other.strftime("%Y%m%d-%H%M%S"))
        assert getAtmDensity(LAT, LON, 120000.0, JD_REF) == dens_other_time

    finally:

        # Restore the defaults for the other tests
        _setAtmosphere()

    assert getMSISVersion() == "2.1"



def testTheMeasuredSpaceWeatherIsUsedOrTheConstantOne(monkeypatch):
    """ Check that the measured F10.7 and Ap of the date are used and recorded, and that the constant values
        are used when they are not available.
    """

    f107, f107a, ap, source = spaceWeather(JD_REF)

    # Check the measured values, which need the CelesTrak file
    if source == 'real':

        # The Sun was quiet in April 2020
        assert 60 < f107 < 80
        assert 60 < f107a < 80

        info = atmDensityTableInfo(LAT, LON, JD_REF)
        assert info['model'] == "NRLMSIS 2.1"
        assert info['space_weather'] == 'real'
        assert info['f107'] == f107

        # Check that the quiet Sun gives a lower density at 150 km than the constant activity
        dens_real = getAtmDensity(LAT, LON, 150000.0, JD_REF)
        try:
            _setAtmosphere('--atmsw', 'constant')
            assert dens_real/getAtmDensity(LAT, LON, 150000.0, JD_REF) < 0.97
        finally:
            _setAtmosphere()

    # Check the constant values when the measured ones are not available
    def failingDownload(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(AtmosphereDensity.pymsis.utils, 'get_f107_ap', failingDownload)

    assert spaceWeather(JD_REF) == (150.0, 150.0, 4.0, 'constant')
    assert atmDensityTableInfo(LAT, LON, JD_REF)['space_weather'] == 'constant'



def _childAtmosphere(_):
    """ Return the model and a density as seen by a process started by multiprocessing, the density also
        depending on the date.
    """

    return getMSISVersion(), getAtmDensity(LAT, LON, 85000.0, JD_REF)



def testAProcessStartedBySpawnUsesTheChosenAtmosphere():
    """ Check that a process started with "spawn", the default on macOS and Windows, which imports the module
        afresh, uses the model and date chosen in its parent through the environment.
    """

    default = _childAtmosphere(0)

    try:
        _setAtmosphere('--atm', '00', '--atmtime', '20240101-000000')
        parent = _childAtmosphere(0)

        with multiprocessing.get_context("spawn").Pool(1) as pool:
            child = pool.map(_childAtmosphere, [0])[0]

        assert parent == child
        assert child[0] == "00"
        assert abs(child[1]/default[1] - 1) > 1e-3

    finally:
        _setAtmosphere()

    # Check that the defaults leave no date behind for later processes
    assert os.environ[MSIS_VERSION_ENV] == "2.1"
    assert MSIS_JD_ENV not in os.environ



def testTheTrajectorySolverOptionsSelectTheAtmosphere():
    """ Check that the input formats of the trajectory solver take --atm and --atmtime through
        addSolverOptions(), and apply them with setAtmosphere() after parsing.
    """

    arg_parser = argparse.ArgumentParser()
    addSolverOptions(arg_parser)

    try:
        setAtmosphere(arg_parser.parse_args(['--atm', '2.0']))
        assert getMSISVersion() == "2.0"

    finally:
        setAtmosphere(arg_parser.parse_args([]))

    assert getMSISVersion() == "2.1"
