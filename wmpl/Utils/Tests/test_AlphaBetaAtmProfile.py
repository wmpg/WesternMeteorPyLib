""" Tests of the air density of an atmosphere profile in the alpha-beta fits (--atm_profile): the density of
    the profile up to its top, and above it that of the MSIS model scaled to join it.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_AlphaBetaAtmProfile.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.AtmosphereDensity import loadAtmosphereProfile, atmDensProfileMSIS, atmProfileJoinFactor, \
    getAtmDensity
from wmpl.Utils.AlphaBeta import airDensityProfile, rescaleHeightToExponentialAtmosphere, HT_NORM_CONST
from wmpl.Utils.Physics import dynamicPressure
from wmpl.Utils.Pickling import savePickle
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



def testCommandLineAtmProfile(tmp_path):
    """ Check that the AlphaBeta CLI scales dynamic mass and peak dynamic pressure with --atm_profile and
        refuses a profile that does not reach down to the lowest point.

    Arguments:
        tmp_path: [Path] Temporary directory of pytest.
    """

    meteor = DragMeteor(15000.0, 90.0, 45.0, 0.005, 0.005, 2.7, 10.0, ground_drag=True)
    traj = observeMeteor(meteor, noise=0.0, fps=10.0)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    savePickle(traj, str(tmp_path), 'profile_trajectory.pickle')
    pkl_path = os.path.join(str(tmp_path), 'profile_trajectory.pickle')

    # Write a synthetic profile scaled by DENSER at the fireball mean location
    lat_m = 0.5*(traj.rbeg_lat + traj.rend_lat)
    lon_m = 0.5*(traj.rbeg_lon + traj.rend_lon)
    prof_path = str(tmp_path/'era5_scaled.csv')
    writeProfile(prof_path, PROFILE_HT, DENSER*getAtmDensity(lat_m, lon_m, PROFILE_HT, traj.jdt_ref))

    # Write a profile that starts above the lowest point of the fireball
    high_ht = np.arange(50000.0, 60000.0, 100.0)
    high_path = str(tmp_path/'era5_too_high.csv')
    writeProfile(high_path, high_ht, getAtmDensity(lat_m, lon_m, high_ht, traj.jdt_ref))

    res_msis = subprocess.run([sys.executable, '-m', 'wmpl.Utils.AlphaBeta', pkl_path, '--hideplots'], \
        capture_output=True, text=True, timeout=300)
    assert res_msis.returncode == 0, res_msis.stderr

    res_prof = subprocess.run([sys.executable, '-m', 'wmpl.Utils.AlphaBeta', pkl_path, '--hideplots', \
        '--atm_profile', prof_path], capture_output=True, text=True, timeout=300)
    assert res_prof.returncode == 0, res_prof.stderr

    res_high = subprocess.run([sys.executable, '-m', 'wmpl.Utils.AlphaBeta', pkl_path, '--hideplots', \
        '--atm_profile', high_path], capture_output=True, text=True, timeout=300)
    assert res_high.returncode != 0
    assert "ERROR:" in res_high.stdout

    # Dynamic mass scales as DENSER^3 and peak dynamic pressure as DENSER
    m_dyn_msis = float(re.search(r"Dynamic mass\s*=\s*([\d.eE+-]+)\s*kg", res_msis.stdout).group(1))
    m_dyn_prof = float(re.search(r"Dynamic mass\s*=\s*([\d.eE+-]+)\s*kg", res_prof.stdout).group(1))
    p_msis = float(re.search(r"AlphaBeta model\s*=\s*([\d.eE+-]+)\s*MPa", res_msis.stdout).group(1))
    p_prof = float(re.search(r"AlphaBeta model\s*=\s*([\d.eE+-]+)\s*MPa", res_prof.stdout).group(1))

    assert m_dyn_prof/m_dyn_msis == pytest.approx(DENSER**3, rel=0.01)
    assert p_prof/p_msis == pytest.approx(DENSER, rel=0.01)



if __name__ == "__main__":

    # Run the test which needs no pytest fixtures
    testDynamicPressureTakesTheGivenDensity()

    print("All tests passed.")
