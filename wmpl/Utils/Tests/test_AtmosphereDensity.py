""" Tests for the MSIS atmosphere mass density in wmpl.Utils.AtmosphereDensity.

The density is computed with pymsis. The tests pin the default MSIS-00 densities against the bundled
Python NRLMSISE-00 port that WMPL used before pymsis, check the scalar and array interfaces and the
polynomial fit, and check the command line options which select the model version and the date.

The port is called here with the local solar time, as the model expects. WMPL used to pass the local
sidereal time instead, which is the one thing pymsis changes about the default model: it shifts the
densities by up to 35% at 110 km, depending on the time of day and the season.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_AtmosphereDensity.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_AtmosphereDensity
"""

import argparse
import datetime

import numpy as np

from wmpl.PythonNRLMSISE00.nrlmsise_00 import gtd7
from wmpl.PythonNRLMSISE00.nrlmsise_00_header import nrlmsise_input, nrlmsise_flags, nrlmsise_output
from wmpl.Utils.AtmosphereDensity import addAtmosphereArguments, atmDensPoly, fitAtmPoly, \
    getAtmDensity, getAtmDensity_vect, getAtmTemperature, setAtmosphere
from wmpl.Utils.TrajConversions import datetime2JD


# Reference location and time
LAT = np.radians(44.327)
LON = np.radians(-81.372)
DT_REF = datetime.datetime(2020, 4, 20, 16, 15, 0)
JD_REF = datetime2JD(DT_REF)


def referenceMSIS(height, dt=DT_REF):
    """ Total mass density (kg/m^3) and temperature (K) at the reference location, from the bundled
        NRLMSISE-00 port.
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
    """ The default model is the same NRLMSISE-00 that WMPL has always used, for both the density and
        the temperature.
    """

    for height in np.arange(20000, 180001, 5000):

        dens_ref, temp_ref = referenceMSIS(height)

        assert abs(getAtmDensity(LAT, LON, float(height), JD_REF)/dens_ref - 1) < 1e-4
        assert abs(getAtmTemperature(LAT, LON, float(height), JD_REF)/temp_ref - 1) < 1e-4


def testScalarAndArrayInputsAgree():
    """ Scalars return floats, arrays are evaluated point by point and keep their shape. """

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
    """ The fitted polynomial follows the MSIS profile over the fitted height range. The polynomial
        smooths the profile, so it is only expected to be good to a few per cent.
    """

    dens_co = fitAtmPoly(LAT, LON, 60000, 180000, JD_REF)

    height_arr = np.linspace(60000, 180000, 100)
    dens_msis = getAtmDensity(LAT, LON, height_arr, JD_REF)

    np.testing.assert_allclose(atmDensPoly(height_arr, dens_co), dens_msis, rtol=0.1)


def testAtmosphereArgumentsSelectTheModelAndTheDate():
    """ --atm selects the MSIS version and --atmtime overrides the date of the evaluation. """

    arg_parser = argparse.ArgumentParser()
    addAtmosphereArguments(arg_parser)

    dt_other = datetime.datetime(2020, 4, 20, 4, 15, 0)
    dens_msis00 = getAtmDensity(LAT, LON, 120000.0, JD_REF)
    dens_other_time = getAtmDensity(LAT, LON, 120000.0, datetime2JD(dt_other))

    try:

        # MSIS 2.1 gives significantly lower densities at 120 km than MSIS-00
        setAtmosphere(arg_parser.parse_args(['--atm', '2.1']))
        assert getAtmDensity(LAT, LON, 120000.0, JD_REF)/dens_msis00 < 0.9

        # The date given on the command line is used instead of the one passed to the function
        setAtmosphere(arg_parser.parse_args(['--atmtime', dt_other.strftime("%Y%m%d-%H%M%S")]))
        assert getAtmDensity(LAT, LON, 120000.0, JD_REF) == dens_other_time

    finally:

        # Restore the defaults for the other tests
        setAtmosphere(arg_parser.parse_args([]))

    # Without the options, the date given to the function is used again
    assert getAtmDensity(LAT, LON, 120000.0, JD_REF) == dens_msis00


if __name__ == "__main__":

    test_functions = [
        testMSIS00MatchesTheBundledNRLMSISEPort,
        testScalarAndArrayInputsAgree,
        testFitAtmPolyReproducesTheProfile,
        testAtmosphereArgumentsSelectTheModelAndTheDate,
    ]

    failed = 0
    for test_func in test_functions:

        try:
            test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {:s}".format(test_func.__name__, str(e)))

    print()
    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
