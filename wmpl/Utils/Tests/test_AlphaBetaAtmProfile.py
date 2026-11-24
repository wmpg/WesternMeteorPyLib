""" Tests of the air density of an atmosphere profile in the alpha-beta fits (--atm_profile): the density of
    the profile up to its top, and above it that of the MSIS model scaled to join it.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_AlphaBetaAtmProfile.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from wmpl.Utils.AtmosphereDensity import loadAtmosphereProfile, atmDensProfileMSIS, atmProfileJoinFactor, \
    getAtmDensity
from wmpl.Utils.AlphaBeta import airDensityProfile, rescaleHeightToExponentialAtmosphere, HT_NORM_CONST
from wmpl.Utils.Physics import dynamicPressure
from wmpl.Utils.Tests.AtmosphereProfileSynthetic import LAT, LON, JD, DENSER, PROFILE_HT, writeProfile, \
    msisProfile


# End of Winchcombe in radians, as the density functions take it
LAT_RAD = np.radians(LAT)
LON_RAD = np.radians(LON)



def testProfileJoinsTheMsisModelAtItsTop(tmp_path):
    """ Check that the density is that of the profile up to its top and above it that of MSIS scaled by the
        ratio of the two at the top, so that it does not jump at the top.

    Arguments:
        tmp_path: [Path] Temporary directory of pytest.
    """

    # A profile whose ratio to MSIS grows with height up to DENSER
    scale_arr = 1.0 + (DENSER - 1.0)*PROFILE_HT/PROFILE_HT[-1]
    file_path = str(tmp_path/'ModelWindProfile.csv')
    writeProfile(file_path, PROFILE_HT, scale_arr*getAtmDensity(LAT_RAD, LON_RAD, PROFILE_HT, JD))
    atm_table = loadAtmosphereProfile(file_path)
    ht_top = atm_table[0][-1]

    # Up to the top, the density of the profile
    ht_arr = np.array([1000.0, 20000.0, 40000.0])
    assert atmDensProfileMSIS(ht_arr, LAT_RAD, LON_RAD, JD, *atm_table) == pytest.approx( \
        10**np.interp(ht_arr, atm_table[0], atm_table[1]), rel=1e-6)

    # Above the top, MSIS times the ratio at the top
    assert atmProfileJoinFactor(LAT_RAD, LON_RAD, JD, *atm_table) == pytest.approx(DENSER, rel=1e-6)
    ht_arr = np.array([50000.0, 80000.0, 120000.0])
    assert atmDensProfileMSIS(ht_arr, LAT_RAD, LON_RAD, JD, *atm_table) == pytest.approx( \
        DENSER*getAtmDensity(LAT_RAD, LON_RAD, ht_arr, JD), rel=1e-6)

    # No jump at the top, where MSIS alone is DENSER times lighter
    dens_below, dens_above = atmDensProfileMSIS(np.array([ht_top - 1.0, ht_top + 1.0]), LAT_RAD, LON_RAD, \
        JD, *atm_table)
    assert dens_above/dens_below == pytest.approx(1.0, abs=1e-3)

    # A single height gives a single density
    assert isinstance(atmDensProfileMSIS(30000.0, LAT_RAD, LON_RAD, JD, *atm_table), float)



def testProfileOfMsisTimesAFactorScalesTheWholeColumn(tmp_path):
    """ Check that a profile of the MSIS density times a constant factor gives the density profile of the
        alpha-beta fits times that factor at all heights, also above the top of the profile, so the heights
        mapped by the column all move by the same amount, which changes only alpha.

    Arguments:
        tmp_path: [Path] Temporary directory of pytest.
    """

    # A profile of MSIS times DENSER, and the points of a meteor
    atm_table = msisProfile(tmp_path, DENSER)
    ht_data = np.linspace(90000.0, 25000.0, 50)
    lat_data = np.full_like(ht_data, LAT_RAD)
    lon_data = np.full_like(ht_data, LON_RAD)

    # The density profile of the fits, without and with the profile
    profile_hts, profile_dens = airDensityProfile(lat_data, lon_data, ht_data, JD)
    profile_hts_atm, profile_dens_atm = airDensityProfile(lat_data, lon_data, ht_data, JD, \
        atm_table=atm_table)

    assert np.array_equal(profile_hts, profile_hts_atm)
    assert profile_dens_atm/profile_dens == pytest.approx(DENSER, rel=1e-4)

    # The mapped heights all move down by h0*ln(DENSER)
    ht_rescaled = rescaleHeightToExponentialAtmosphere(lat_data, lon_data, ht_data, JD)
    ht_rescaled_atm = rescaleHeightToExponentialAtmosphere(lat_data, lon_data, ht_data, JD, \
        atm_table=atm_table)

    assert ht_rescaled - ht_rescaled_atm == pytest.approx(HT_NORM_CONST*np.log(DENSER), abs=1.0)



def testDynamicPressureTakesTheGivenDensity():
    """ Check that the dynamic pressure uses the given air density, and the MSIS density without it. """

    ht_arr = np.array([30000.0, 40000.0])
    vel_arr = np.array([8000.0, 10000.0])
    dens_arr = np.array([0.02, 0.004])

    # With the given density and without it
    assert dynamicPressure(LAT_RAD, LON_RAD, ht_arr, JD, vel_arr, atm_dens=dens_arr) == pytest.approx( \
        dens_arr*vel_arr**2, rel=1e-12)
    assert dynamicPressure(LAT_RAD, LON_RAD, ht_arr, JD, vel_arr) == pytest.approx( \
        getAtmDensity(LAT_RAD, LON_RAD, ht_arr, JD)*vel_arr**2, rel=1e-12)



if __name__ == "__main__":

    # Run the test which needs no pytest fixtures
    testDynamicPressureTakesTheGivenDensity()

    print("All tests passed.")
