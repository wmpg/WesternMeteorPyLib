""" Tests for reading atmosphere profiles as OpenDarkflight does (wmpl.Utils.AtmosphereProfile).

Run with: python -m pytest wmpl/Utils/Tests/test_AtmosphereProfile.py
"""

import hashlib

import numpy as np
import pytest

from wmpl.Utils.AtmosphereProfile import AtmosphereProfile, densityFromPressure, R_GAS, M_AIR
from wmpl.Utils.AtmosphereDensity import atmDensPoly


WRF_HEADER = "height,temperature,pressure,relative_humidity,wind_horizontal,wind_direction,wind_east," \
    "wind_north,wind_up,density\n"


def _writeWrf(path, heights, densities):
    with open(path, 'w') as f:
        f.write(WRF_HEADER)
        for ht, rho in zip(heights, densities):
            f.write("{:.1f},250.0,1000.0,1.0,10.0,270.0,10.0,0.0,0.0,{:.10e}\n".format(ht, rho))


def testWindIsTheVelocityOfTheAirFromTheSpeedAndTheDirectionItBlowsFrom(tmp_path):
    """ wrf and supracenter give the speed in m/s, wyoming in knots, and all give the direction the wind
        blows from; the wind returned is the velocity the air moves with.
    """

    path = str(tmp_path/"profile.csv")
    _writeWrf(path, [10000.0, 20000.0], [0.4, 0.09])
    assert AtmosphereProfile(path).wind(15000.0) == pytest.approx([10.0, 0.0], abs=1e-12)

    path = str(tmp_path/"supra.txt")
    with open(path, 'w') as f:
        f.write("   100   -3.8   8.0  180.0  1015.125\n   200   -0.9   8.0  180.0  1002.426\n")
    assert AtmosphereProfile(path, profile_type='supracenter').wind(150.0) == pytest.approx([0.0, 8.0], \
        abs=1e-12)


def _exponentialProfile(path, ht_top=60000.0, step=100.0, scale_height=7000.0):
    heights = np.arange(0.0, ht_top + step, step)
    _writeWrf(path, heights, 1.225*np.exp(-heights/scale_height))
    return heights


def testWrfUsesTheDensityColumnAndInterpolatesItsLogarithm(tmp_path):

    path = str(tmp_path/"profile.csv")
    _writeWrf(path, [30000.0, 10000.0, 20000.0], [0.02, 0.4, 0.09])

    prof = AtmosphereProfile(path)

    # Levels are sorted by height and their densities are the file's
    assert list(prof.heights) == [10000.0, 20000.0, 30000.0]
    assert prof.density([10000.0, 20000.0, 30000.0]) == pytest.approx([0.4, 0.09, 0.02], rel=1e-9)

    # Halfway between two levels the density is their geometric mean
    assert prof.density(15000.0) == pytest.approx(np.sqrt(0.4*0.09), rel=1e-12)

    with open(path, 'rb') as f:
        assert prof.sha256 == hashlib.sha256(f.read()).hexdigest()


def testHeightsOutsideTheProfileAreRefused(tmp_path):

    path = str(tmp_path/"profile.csv")
    _writeWrf(path, [10000.0, 20000.0], [0.4, 0.09])

    prof = AtmosphereProfile(path)

    with pytest.raises(ValueError, match="covers 10.00 to 20.00 km"):
        prof.density(25000.0)

    with pytest.raises(ValueError, match="covers"):
        prof.checkCoverage(5000.0, 15000.0)


def testWrfRowsMustHaveTenColumnsAndTheTypeMustBeKnown(tmp_path):

    path = str(tmp_path/"profile.csv")
    with open(path, 'w') as f:
        f.write(WRF_HEADER + "1000.0,280.0,90000.0,1.0,1.0,1.0,1.0,1.0,1.0\n")

    with pytest.raises(ValueError, match="10 comma-separated columns"):
        AtmosphereProfile(path)

    with pytest.raises(ValueError, match="Unknown atmosphere profile type"):
        AtmosphereProfile(path, profile_type='era5')


def testWyomingSkipsTheFirstDataRowAndIncompleteRows(tmp_path):
    """ As in OpenDarkflight, the first data row only marks the start of the data, rows without all 11 values
        are skipped, an empty line ends the data, and the density comes from pressure and temperature.
    """

    path = str(tmp_path/"sounding.txt")
    with open(path, 'w') as f:
        f.write("10393 Lindenberg Observations at 00Z 21 Jan 2024\n\n")
        f.write("-----------------------------------------------------------------------------\n")
        f.write("   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV\n")
        f.write("    hPa     m      C      C      %    g/kg    deg   knot     K      K      K \n")
        f.write("-----------------------------------------------------------------------------\n")
        f.write(" 1016.0    112   -3.3   -6.1     81   2.40    240      6  268.6  275.2  269.0\n")
        f.write(" 1007.0    183   -3.0   -6.8     75   2.29    235     16  269.6  275.9  269.9\n")
        f.write(" 1002.0    223   -2.9                         242     18  270.1  276.3  270.5\n")
        f.write("  500.0   5500  -30.0  -40.0     30   0.20    270     40  300.0  301.0  300.1\n")
        f.write("\nStation information and sounding indices\n")
        f.write("  100.0  16000  -60.0  -80.0      5   0.01    270     60  400.0  400.0  400.0\n")

    prof = AtmosphereProfile(path, profile_type='wyoming')

    assert list(prof.heights) == [183.0, 5500.0]

    # 40 knots from the west
    assert prof.wind(5500.0) == pytest.approx([40*0.514, 0.0], abs=1e-9)
    assert prof.density([183.0, 5500.0]) == pytest.approx( \
        [100*1007.0*M_AIR/(R_GAS*270.15), 100*500.0*M_AIR/(R_GAS*243.15)], rel=1e-12)


def testSupracenterComputesTheDensityFromPressureAndTemperature(tmp_path):

    path = str(tmp_path/"sounding.txt")
    with open(path, 'w') as f:
        f.write("   100   -3.8   5.135  198.633  1015.125\n")
        f.write("   200   -0.9   7.564  224.972  1002.426\n")

    prof = AtmosphereProfile(path, profile_type='supracenter')

    assert prof.density([100.0, 200.0]) == pytest.approx( \
        densityFromPressure([1015.125, 1002.426], [269.35, 272.25]), rel=1e-12)


def testRepeatedHeightsKeepTheFirstLevel(tmp_path):

    path = str(tmp_path/"profile.csv")
    _writeWrf(path, [10000.0, 20000.0, 20000.0], [0.4, 0.09, 0.5])

    assert AtmosphereProfile(path).density(20000.0) == pytest.approx(0.09, rel=1e-9)


def testSimulationPolynomialFollowsTheProfile(tmp_path):

    path = str(tmp_path/"profile.csv")
    _exponentialProfile(path)
    prof = AtmosphereProfile(path)

    dens_co, max_rel_err = prof.fitPoly(15000.0, 35000.0)

    heights = np.linspace(15000.0, 35000.0, 50)
    assert max_rel_err < 1e-3
    assert atmDensPoly(heights, dens_co) == pytest.approx(prof.density(heights), rel=1e-3)



if __name__ == "__main__":

    import sys

    sys.exit(pytest.main([__file__, "-q"]))
