""" Tests for the air density of an atmosphere profile in DynamicMassFit (--atm_profile): the profile reader,
the dynamic mass and the fragment simulation.

Run under pytest, or directly:

    python -m wmpl.Utils.Tests.test_DynamicMassFitAtmProfile
"""

import io
import os
import contextlib

import numpy as np
import pytest

from wmpl.Utils.AtmosphereDensity import loadAtmosphereProfile, atmDensTable, getAtmDensity, \
    getAtmDensityTable
from wmpl.Utils.Physics import dynamicMass
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.DynamicMassFit import runFragSim, computeFragEndParams


# End of the Winchcombe fireball (2021-02-28), where DynamicMassFit evaluates its dynamic mass
LAT = 51.94011
LON = -2.09633
JD = 2459274.412778
HT_EVAL = 29980.0
V_EVAL = 6090.0
DECEL = 3230.0

# Air 9.4% denser than NRLMSIS 2.1, as in the ERA5 reanalysis at that point
DENSER = 1.094

# Heights of an ERA5 profile as OpenDarkflight writes it (m)
PROFILE_HT = np.arange(300.0, 46300.0, 100.0)

# Header of the CSV files OpenDarkflight writes for model profiles
PROFILE_HEADER = "height,temperature,pressure,relative_humidity,wind_horizontal,wind_direction,wind_east," \
    "wind_north,wind_up,density"

# A solved trajectory shipped with the repository
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"



def writeProfile(file_path, ht_arr, dens_arr):
    """ Write an atmosphere profile in the CSV format OpenDarkflight writes for model profiles.

    Arguments:
        file_path: [str] Path to the CSV file.
        ht_arr: [ndarray] Heights (m).
        dens_arr: [ndarray] Air densities (kg/m^3).

    Return:
        None
    """

    with open(file_path, 'w') as f:

        f.write(PROFILE_HEADER + "\n")

        for ht, dens in zip(ht_arr, dens_arr):
            f.write("{:.1f},220.0,1000.0,0.0,10.0,270.0,10.0,0.0,0.0,{:.10e}\n".format(ht, dens))



def msisProfile(tmp_path, scale):
    """ Write and load a profile of the MSIS density at the end of Winchcombe, multiplied by a factor.

    Arguments:
        tmp_path: [Path] Directory for the file.
        scale: [float] Factor applied to the MSIS density.

    Return:
        (table_ht, table_log10_rho): [tuple] Density table of the profile.
    """

    file_path = str(tmp_path/"ModelWindProfile_{:.3f}.csv".format(scale))
    writeProfile(file_path, PROFILE_HT, \
        scale*getAtmDensity(np.radians(LAT), np.radians(LON), PROFILE_HT, JD))

    return loadAtmosphereProfile(file_path)



def testProfileIsReadAsADensityTable(tmp_path):
    """ The levels are sorted by height, and a level without a density is skipped. """

    file_path = str(tmp_path/"profile.csv")
    writeProfile(file_path, [3000.0, 1000.0, 2000.0, 4000.0], [0.9, 1.1, np.nan, 0.8])

    table_ht, table_log10_rho = loadAtmosphereProfile(file_path)

    assert table_ht == [1000.0, 3000.0, 4000.0]
    assert table_log10_rho == pytest.approx(np.log10([1.1, 0.9, 0.8]), abs=1e-12)



def testMsisProfileGivesTheMsisMass(tmp_path):
    """ With a profile of the MSIS density, the dynamic mass is the one of the MSIS model. """

    atm_table = msisProfile(tmp_path, 1.0)

    mass_msis = dynamicMass(3500, np.radians(LAT), np.radians(LON), HT_EVAL, JD, V_EVAL, DECEL, \
        shape_factor=0.55)
    mass_profile = dynamicMass(3500, np.radians(LAT), np.radians(LON), HT_EVAL, JD, V_EVAL, DECEL, \
        shape_factor=0.55, atm_dens=atmDensTable(HT_EVAL, *atm_table))

    assert mass_profile == pytest.approx(mass_msis, rel=1e-4)



def testMassGoesAsTheCubeOfTheAirDensity(tmp_path):
    """ In air 9.4% denser, the dynamic mass is 1.094^3 = 1.31 times larger. """

    mass_list = []
    for scale in [1.0, DENSER]:

        atm_dens = atmDensTable(HT_EVAL, *msisProfile(tmp_path, scale))
        mass_list.append(dynamicMass(3500, np.radians(LAT), np.radians(LON), HT_EVAL, JD, V_EVAL, DECEL, \
            shape_factor=0.55, atm_dens=atm_dens))

    assert mass_list[1]/mass_list[0] == pytest.approx(DENSER**3, rel=1e-9)



def testFragSimRunsInTheProfileAtmosphere(tmp_path):
    """ The fragment simulation takes the air density of the profile. In the MSIS profile it ends where it
        does with the MSIS model. In denser air, a body heavier by the cube of the density decelerates and
        ablates in the same way, so it ends at the same point with its mass scaled by the same factor.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        sr_msis = runFragSim(0.1, 3500, LAT, LON, JD, HT_EVAL, V_EVAL, 41.0, 0.55)
        sr_table = runFragSim(0.1, 3500, LAT, LON, JD, HT_EVAL, V_EVAL, 41.0, 0.55, \
            atm_table=msisProfile(tmp_path, 1.0))
        sr_dense = runFragSim(0.1*DENSER**3, 3500, LAT, LON, JD, HT_EVAL, V_EVAL, 41.0, 0.55, \
            atm_table=msisProfile(tmp_path, DENSER))

    assert sr_table.frag_main.h == pytest.approx(sr_msis.frag_main.h, abs=50.0)
    assert sr_table.frag_main.m == pytest.approx(sr_msis.frag_main.m, rel=0.01)

    assert sr_dense.frag_main.h == pytest.approx(sr_table.frag_main.h, abs=1.0)
    assert sr_dense.frag_main.m/sr_table.frag_main.m == pytest.approx(DENSER**3, rel=1e-6)



def testFragSimRefusesAProfileBelowItsStart(tmp_path):
    """ A profile which does not reach the starting height is refused, as it would be extrapolated. """

    table_ht, table_log10_rho = msisProfile(tmp_path, 1.0)
    n_low = np.searchsorted(table_ht, 25000.0)

    with pytest.raises(ValueError):
        runFragSim(0.1, 3500, LAT, LON, JD, HT_EVAL, V_EVAL, 41.0, 0.55, \
            atm_table=(table_ht[:n_low], table_log10_rho[:n_low]))



def testFragEndParamsTakesTheProfile():
    """ computeFragEndParams() passes the profile to the fragment simulation, so the end point is the same
        in denser air with the mass scaled by the cube of the density.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    table_ht, table_log10_rho = getAtmDensityTable(traj.rend_lat, traj.rend_lon, 10000, 120000, traj.jdt_ref)
    table_dense = (table_ht, [lr + np.log10(DENSER) for lr in table_log10_rho])

    with contextlib.redirect_stdout(io.StringIO()):
        end_msis = computeFragEndParams(traj, 1.0, 3500, 100000.0, 20000.0, 0.55, \
            atm_table=(table_ht, table_log10_rho))
        end_dense = computeFragEndParams(traj, DENSER**3, 3500, 100000.0, 20000.0, 0.55, \
            atm_table=table_dense)

    assert end_dense[1]/end_msis[1] == pytest.approx(DENSER**3, rel=1e-6)
    assert end_dense[2:5] == pytest.approx(end_msis[2:5], abs=1e-5)




if __name__ == "__main__":

    import sys
    sys.exit(pytest.main([__file__, "-q"]))
