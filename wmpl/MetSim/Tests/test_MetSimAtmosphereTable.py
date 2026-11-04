""" Tests for the tabulated atmosphere density profile used by MetSimErosion.

The ablation simulation used to take the air density only from a 6th degree polynomial fitted to log10 of
the MSIS density (fitAtmPoly). Over the wide height ranges used for fireballs (e.g. 14 - 180 km) the
polynomial misses the model by tens of percent at 80 - 120 km. The density can now be given as a table of
log10(density), interpolated linearly in height (getAtmDensityTable), which the simulation uses instead of
the polynomial when it is present. These tests check:

    - the table grid and its values at the nodes
    - the table follows the MSIS model between the nodes, far closer than the polynomial
    - the scalar engine interpolation and the vectorized utility agree, also outside the table
    - the exact air column of the table matches the numerical integral (used by the Pecina-Ceplecha fit)
    - the simulation actually uses the table, and falls back to the polynomial without one (incl. Constants
      objects created before the table existed)
    - tables given as numpy arrays (e.g. after a JSON load) are converted, invalid tables are rejected
    - the energy received before erosion integrates the table
    - the table survives the GUI JSON save/load round trip

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_MetSimAtmosphereTable
"""

import os
import sys
import tempfile

import numpy as np
import scipy.integrate

import wmpl.MetSim.MetSimErosion as MetSimErosion
from wmpl.Utils.AtmosphereDensity import AtmDensityTableInterp, atmColumnTable, atmDensPoly, atmDensTable, \
    fitAtmPoly, getAtmDensity, getAtmDensityTable
from wmpl.Utils.TrajConversions import date2JD


# Reference location and time
LAT = np.radians(45.3)
LON = np.radians(18.1)
JD = date2JD(2020, 4, 20, 16, 15, 0)

# Fireball-like density range (bottom at the 14 km floor used by the GUI and Dynesty)
HT_MIN = 14000.0
HT_MAX = 180000.0

# Shared table and polynomial, computed once
TABLE_HT, TABLE_LOG10_RHO = getAtmDensityTable(LAT, LON, HT_MIN, HT_MAX, JD)
DENS_CO = fitAtmPoly(LAT, LON, HT_MIN, HT_MAX, JD)


def _makeConstants(table=None):
    """ Single-body constants for a short, fast simulation, optionally with the given density table. """

    const = MetSimErosion.Constants()
    const.dens_co = DENS_CO
    const.h_init = 130000.0
    const.m_init = 1e-3
    const.v_init = 20000.0
    const.rho = 3000
    const.sigma = 0.02/1e6
    const.zenith_angle = np.radians(45.0)
    const.erosion_on = False
    const.disruption_on = False
    const.fragmentation_on = False
    const.h_kill = 20000.0
    const.v_kill = 3000.0

    if table is not None:
        const.atm_table_ht, const.atm_table_log10_rho = table

    return const


def _runMain(const):
    """ Run the simulation and return the time, height, velocity and luminosity of the main fragment. """

    _, results_list, _ = MetSimErosion.runSimulation(const, compute_wake=False)
    results = np.array(results_list, dtype=np.float64)

    # Columns of ablateAll()'s results list: time, total luminosity, ..., main height, main length, main vel
    return results[:, 0], results[:, 17], results[:, 19], results[:, 1]


def test_table_grid():
    """ The table spans the requested range in equal steps no larger than requested, holds plain floats,
    and stores log10 of the MSIS density at the nodes. """

    assert TABLE_HT[0] == HT_MIN
    assert TABLE_HT[-1] == HT_MAX
    assert len(TABLE_HT) == len(TABLE_LOG10_RHO)
    assert all(type(x) is float for x in TABLE_HT + TABLE_LOG10_RHO)

    steps = np.diff(TABLE_HT)
    assert np.all(steps <= 500.0 + 1e-9)
    assert np.allclose(steps, steps[0])

    # A step that doesn't divide the range evenly is shrunk, not exceeded
    ht, _ = getAtmDensityTable(LAT, LON, 70000.0, 71234.0, JD, step=500)
    assert len(ht) == 4 and np.all(np.diff(ht) <= 500.0)

    for i in [0, 100, len(TABLE_HT) - 1]:
        assert TABLE_LOG10_RHO[i] == np.log10(getAtmDensity(LAT, LON, TABLE_HT[i], JD))

    try:
        getAtmDensityTable(LAT, LON, 80000.0, 80000.0, JD)
        assert False, "An empty height range must raise"
    except ValueError:
        pass


def test_table_follows_msis():
    """ Between the nodes the table stays within 0.5% of the MSIS model over the whole fireball range, while the
    polynomial fitted to the same range misses it by more than 10% somewhere at 80 - 120 km. """

    # Node midpoints are the worst case of the interpolation
    ht = 0.5*(np.array(TABLE_HT[:-1]) + np.array(TABLE_HT[1:]))
    rho_msis = np.array([getAtmDensity(LAT, LON, h, JD) for h in ht])

    err_table = np.abs(atmDensTable(ht, TABLE_HT, TABLE_LOG10_RHO)/rho_msis - 1)
    assert err_table.max() < 0.005, err_table.max()

    meteor_band = (ht > 80000) & (ht < 120000)
    err_poly = np.abs(atmDensPoly(ht, DENS_CO)/rho_msis - 1)
    assert err_poly[meteor_band].max() > 0.10, err_poly[meteor_band].max()


def test_scalar_and_vector_interpolation_agree():
    """ The scalar engine interpolation equals the vectorized utility at the nodes, between them and outside
    the table (where the end segments are extrapolated). """

    ht = np.concatenate([[5000.0, HT_MIN, 14321.0, 95123.4, 100000.0, HT_MAX, 200000.0],
        np.linspace(10000, 190000, 97)])

    rho_vect = atmDensTable(ht, TABLE_HT, TABLE_LOG10_RHO)
    rho_scalar = np.array([MetSimErosion.atmDensityTable(h, TABLE_HT, TABLE_LOG10_RHO) for h in ht])
    assert np.allclose(rho_scalar, rho_vect, rtol=1e-12, atol=0)

    # Exact at the nodes
    assert np.isclose(MetSimErosion.atmDensityTable(TABLE_HT[50], TABLE_HT, TABLE_LOG10_RHO),
        10**TABLE_LOG10_RHO[50], rtol=1e-12)

    # Outside the table, the end segment continues as an exponential
    slope_top = (TABLE_LOG10_RHO[-1] - TABLE_LOG10_RHO[-2])/(TABLE_HT[-1] - TABLE_HT[-2])
    assert np.isclose(np.log10(atmDensTable(HT_MAX + 10000.0, TABLE_HT, TABLE_LOG10_RHO)),
        TABLE_LOG10_RHO[-1] + slope_top*10000.0, rtol=1e-12)

    # The dispatcher picks the table if there is one, the polynomial otherwise
    assert MetSimErosion.atmDensity(95123.4, _makeConstants((TABLE_HT, TABLE_LOG10_RHO))) == rho_scalar[3]
    assert MetSimErosion.atmDensity(95123.4, _makeConstants()) == \
        MetSimErosion.atmDensityPoly(95123.4, np.array(DENS_CO))


def test_table_column_exact():
    """ The exact air column of the table equals the numerical integral of the interpolated density, also
    across the table ends, and the callable table gives the same density as atmDensTable. """

    dens_interp = AtmDensityTableInterp(TABLE_HT, TABLE_LOG10_RHO)

    ht = np.array([10000.0, 50000.0, 95123.4])
    assert np.array_equal(dens_interp(ht), atmDensTable(ht, TABLE_HT, TABLE_LOG10_RHO))

    for ht_low, ht_high in [(70000.0, 180000.0), (14000.0, 30000.0), (95123.4, 95400.0), (10000.0, 20000.0),
            (170000.0, 190000.0)]:

        # Integrate segment by segment so quad never straddles a kink of the piecewise profile
        nodes = np.unique(np.concatenate([[ht_low, ht_high],
            [h for h in TABLE_HT if ht_low < h < ht_high]]))
        column_quad = sum(scipy.integrate.quad(dens_interp, a, b, epsrel=1e-12)[0]
            for a, b in zip(nodes[:-1], nodes[1:]))

        assert np.isclose(atmColumnTable(ht_low, ht_high, TABLE_HT, TABLE_LOG10_RHO), column_quad, rtol=1e-9)
        assert np.isclose(dens_interp.column(ht_low, ht_high), column_quad, rtol=1e-9)

    # Swapped limits give the negative column, and arrays work
    assert np.isclose(dens_interp.column(100000.0, 80000.0), -dens_interp.column(80000.0, 100000.0))
    columns = dens_interp.column(np.array([60000.0, 80000.0]), HT_MAX)
    assert columns.shape == (2,) and columns[0] > columns[1] > 0

    # Pecina-Ceplecha integrates with the exact column when it is available, quadrature otherwise
    from wmpl.Utils.PecinaCeplechaFunction import airColumn
    assert airColumn(dens_interp, 80000.0, HT_MAX) == float(dens_interp.column(80000.0, HT_MAX))
    assert np.isclose(airColumn(lambda h: 10**(-5 - h/20000.0), 80000.0, 100000.0),
        20000.0/np.log(10)*(10**(-9.0) - 10**(-10.0)), rtol=1e-8)


def test_simulation_uses_table():
    """ A table sampled from the polynomial itself reproduces the polynomial run, and the MSIS table
    changes the run - so the simulation reads the table, not dens_co. """

    t_poly, ht_poly, vel_poly, lum_poly = _runMain(_makeConstants())

    # Table of the polynomial on a fine grid: same atmosphere up to the interpolation error
    ht_fine = np.arange(HT_MIN, HT_MAX + 1, 100.0)
    table_poly = (ht_fine.tolist(), np.log10(atmDensPoly(ht_fine, DENS_CO)).tolist())
    t_tp, ht_tp, vel_tp, lum_tp = _runMain(_makeConstants(table_poly))

    assert len(t_tp) == len(t_poly)
    assert np.allclose(ht_tp, ht_poly, rtol=0, atol=1.0)
    assert np.allclose(vel_tp, vel_poly, rtol=1e-5)
    assert np.allclose(lum_tp, lum_poly, rtol=1e-3, atol=1e-6*np.max(lum_poly))

    # The MSIS table is a different atmosphere, so the light curve must change
    _, _, _, lum_msis = _runMain(_makeConstants((TABLE_HT, TABLE_LOG10_RHO)))
    n = min(len(lum_msis), len(lum_poly))
    assert np.max(np.abs(lum_msis[:n] - lum_poly[:n])) > 0.01*np.max(lum_poly)


def test_constants_without_table_attribute():
    """ Constants created before the table existed (e.g. unpickled, without the attributes) still run, on the
    polynomial, exactly as Constants with the table set to None. """

    const_old = _makeConstants()
    del const_old.atm_table_ht
    del const_old.atm_table_log10_rho

    t_old, ht_old, _, lum_old = _runMain(const_old)
    t_new, ht_new, _, lum_new = _runMain(_makeConstants())

    assert np.array_equal(t_old, t_new)
    assert np.array_equal(ht_old, ht_new)
    assert np.array_equal(lum_old, lum_new)


def test_numpy_table_is_converted():
    """ A table given as numpy arrays is converted to lists of floats and gives the same run as lists. """

    const_np = _makeConstants((np.array(TABLE_HT), np.array(TABLE_LOG10_RHO)))
    _, ht_np, _, lum_np = _runMain(const_np)

    assert isinstance(const_np.atm_table_ht, list) and isinstance(const_np.atm_table_log10_rho, list)
    assert all(type(x) is float for x in const_np.atm_table_ht)

    _, ht_list, _, lum_list = _runMain(_makeConstants((TABLE_HT, TABLE_LOG10_RHO)))
    assert np.array_equal(ht_np, ht_list)
    assert np.array_equal(lum_np, lum_list)


def test_invalid_table_rejected():
    """ Tables that are too short, have mismatched columns or are not ascending raise a ValueError. """

    bad_tables = [
        ([100000.0], [-6.0]),
        ([90000.0, 100000.0, 110000.0], [-5.0, -6.0]),
        ([110000.0, 100000.0, 90000.0], [-7.0, -6.0, -5.0]),
        ([90000.0, 90000.0, 110000.0], [-5.0, -5.0, -7.0]),
    ]

    for table in bad_tables:
        try:
            MetSimErosion.runSimulation(_makeConstants(table))
            assert False, "Table {} must be rejected".format(table)
        except ValueError:
            pass


def test_energy_received_uses_table():
    """ The energy received before erosion integrates the tabulated density when there is a table. """

    const = _makeConstants((TABLE_HT, TABLE_LOG10_RHO))
    const.erosion_height_start = 100000.0

    es, _ = MetSimErosion.energyReceivedBeforeErosion(const)

    dens_integ = scipy.integrate.quad(lambda h: atmDensTable(h, TABLE_HT, TABLE_LOG10_RHO),
        const.erosion_height_start, const.h_init, limit=200)[0]
    es_ref = 0.5*const.v_init**2*dens_integ/np.cos(const.zenith_angle)

    assert np.isclose(es, es_ref, rtol=1e-6)

    # The polynomial gives a different energy, so the table was used
    es_poly, _ = MetSimErosion.energyReceivedBeforeErosion(_makeConstants())
    assert not np.isclose(es, es_poly, rtol=1e-3)


def test_gui_json_round_trip():
    """ The GUI saves the table to the constants JSON and loads it back unchanged, also from numpy arrays. """

    from wmpl.MetSim.GUI import loadConstants, saveConstants

    const = _makeConstants((np.array(TABLE_HT), np.array(TABLE_LOG10_RHO)))

    with tempfile.TemporaryDirectory() as dir_path:
        saveConstants(const, dir_path, "const.json")
        const_loaded, const_json = loadConstants(os.path.join(dir_path, "const.json"))

    assert const_json['atm_table_ht'] == TABLE_HT
    assert const_json['atm_table_log10_rho'] == TABLE_LOG10_RHO
    assert const_loaded.atm_table_ht == TABLE_HT

    # A JSON without the table (saved before the table existed) loads with the table off
    with tempfile.TemporaryDirectory() as dir_path:
        const_old = _makeConstants()
        del const_old.atm_table_ht
        del const_old.atm_table_log10_rho
        saveConstants(const_old, dir_path, "const.json")
        const_loaded, _ = loadConstants(os.path.join(dir_path, "const.json"))

    assert const_loaded.atm_table_ht is None



if __name__ == "__main__":

    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]

    n_failed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except Exception as e:
            n_failed += 1
            print("FAIL", test.__name__, repr(e))

    print("{:d}/{:d} passed".format(len(tests) - n_failed, len(tests)))
    sys.exit(1 if n_failed else 0)
