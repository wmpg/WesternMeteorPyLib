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


# End of the Winchcombe fireball (2021-02-28), as in test_DynamicMassFitAtmProfile.py
LAT = np.radians(51.94011)
LON = np.radians(-2.09633)
JD = 2459274.412778

# Air 9.4% denser than NRLMSIS 2.1, as in the ERA5 reanalysis at the end of Winchcombe
DENSER = 1.094

# Heights of an ERA5 profile as OpenDarkflight writes it (m)
PROFILE_HT = np.arange(300.0, 46300.0, 100.0)

# Header of the CSV files OpenDarkflight writes for model profiles
PROFILE_HEADER = "height,temperature,pressure,relative_humidity,wind_horizontal,wind_direction,wind_east," \
    "wind_north,wind_up,density"



def msisProfile(tmp_path, scale_func):
    """ Write and load a profile of the MSIS density at Winchcombe multiplied by a factor of the height.

    Arguments:
        tmp_path: [Path] Directory for the file.
        scale_func: [function] Factor applied to the MSIS density, of the height (m).

    Return:
        (table_ht, table_log10_rho): [tuple] Density table of the profile.
    """

    file_path = str(tmp_path/'ModelWindProfile.csv')
    dens_arr = scale_func(PROFILE_HT)*getAtmDensity(LAT, LON, PROFILE_HT, JD)

    with open(file_path, 'w') as f:

        f.write(PROFILE_HEADER + "\n")

        for ht, dens in zip(PROFILE_HT, dens_arr):
            f.write("{:.1f},220.0,1000.0,0.0,10.0,270.0,10.0,0.0,0.0,{:.10e}\n".format(ht, dens))

    return loadAtmosphereProfile(file_path)



def testProfileJoinsTheMsisModelAtItsTop(tmp_path):
    """ Check that the density is that of the profile up to its top and above it that of MSIS scaled by the
        ratio of the two at the top, so that it does not jump at the top.

    Arguments:
        tmp_path: [Path] Temporary directory of pytest.
    """

    # A profile whose ratio to MSIS grows with height, from 1 at the ground to DENSER at its top
    atm_table = msisProfile(tmp_path, lambda ht: 1.0 + (DENSER - 1.0)*ht/PROFILE_HT[-1])
    ht_top = atm_table[0][-1]

    # Up to the top, the density of the profile
    ht_arr = np.array([1000.0, 20000.0, 40000.0])
    assert atmDensProfileMSIS(ht_arr, LAT, LON, JD, *atm_table) == pytest.approx( \
        10**np.interp(ht_arr, atm_table[0], atm_table[1]), rel=1e-6)

    # Above the top, MSIS times the ratio at the top
    assert atmProfileJoinFactor(LAT, LON, JD, *atm_table) == pytest.approx(DENSER, rel=1e-6)
    ht_arr = np.array([50000.0, 80000.0, 120000.0])
    assert atmDensProfileMSIS(ht_arr, LAT, LON, JD, *atm_table) == pytest.approx( \
        DENSER*getAtmDensity(LAT, LON, ht_arr, JD), rel=1e-6)

    # No jump at the top, where MSIS alone is DENSER times lighter
    dens_below, dens_above = atmDensProfileMSIS(np.array([ht_top - 1.0, ht_top + 1.0]), LAT, LON, JD, \
        *atm_table)
    assert dens_above/dens_below == pytest.approx(1.0, abs=1e-3)

    # A single height gives a single density
    assert isinstance(atmDensProfileMSIS(30000.0, LAT, LON, JD, *atm_table), float)



def testProfileOfMsisTimesAFactorScalesTheWholeColumn(tmp_path):
    """ Check that a profile of the MSIS density times a constant factor gives the density profile of the
        alpha-beta fits times that factor at all heights, also above the top of the profile, so the heights
        mapped by the column all move by the same amount, which changes only alpha.

    Arguments:
        tmp_path: [Path] Temporary directory of pytest.
    """

    atm_table = msisProfile(tmp_path, lambda ht: np.full_like(ht, DENSER))
    ht_data = np.linspace(90000.0, 25000.0, 50)
    lat_data = np.full_like(ht_data, LAT)
    lon_data = np.full_like(ht_data, LON)

    profile_hts, profile_dens = airDensityProfile(lat_data, lon_data, ht_data, JD)
    profile_hts_atm, profile_dens_atm = airDensityProfile(lat_data, lon_data, ht_data, JD, \
        atm_table=atm_table)

    assert np.array_equal(profile_hts, profile_hts_atm)
    assert profile_dens_atm/profile_dens == pytest.approx(DENSER, rel=1e-4)

    # The heights mapped to the exponential atmosphere by the column move down by h0*ln(DENSER)
    ht_rescaled = rescaleHeightToExponentialAtmosphere(lat_data, lon_data, ht_data, JD)
    ht_rescaled_atm = rescaleHeightToExponentialAtmosphere(lat_data, lon_data, ht_data, JD, \
        atm_table=atm_table)

    assert ht_rescaled - ht_rescaled_atm == pytest.approx(HT_NORM_CONST*np.log(DENSER), abs=1.0)



def testDynamicPressureTakesTheGivenDensity():
    """ Check that the dynamic pressure uses the given air density, and the MSIS density without it. """

    ht_arr = np.array([30000.0, 40000.0])
    vel_arr = np.array([8000.0, 10000.0])
    dens_arr = np.array([0.02, 0.004])

    assert dynamicPressure(LAT, LON, ht_arr, JD, vel_arr, atm_dens=dens_arr) == pytest.approx( \
        dens_arr*vel_arr**2, rel=1e-12)
    assert dynamicPressure(LAT, LON, ht_arr, JD, vel_arr) == pytest.approx( \
        getAtmDensity(LAT, LON, ht_arr, JD)*vel_arr**2, rel=1e-12)



if __name__ == "__main__":

    # Run the test which needs no pytest fixtures
    testDynamicPressureTakesTheGivenDensity()

    print("All tests passed.")
