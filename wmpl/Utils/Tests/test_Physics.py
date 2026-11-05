""" Tests of the functions in wmpl.Utils.Physics.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Physics.py -v
"""

from __future__ import print_function, division, absolute_import

import warnings

import numpy as np

from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect
from wmpl.Utils.Physics import ballisticCoeffMass, dynamicMass, calcMass, LUM_EFF_MODELS


# A synthetic light curve: a 2 s meteor peaking at -2 mag, decelerating from 20 to 15 km/s
TIME = np.linspace(0, 2, 41)
MAG_ABS = -2 + 3*(TIME - 1.0)**2
VELOCITY = np.linspace(20000, 15000, 41)



def testBallisticCoeffMass():
    """ Check the mass from the ballistic coefficient against its definition and against the dynamic mass
        from the deceleration.
    """

    # A sphere of 1 kg with Gamma = 0.5 and the bulk density 3500 kg/m^3
    gamma_a = 0.5*1.21
    k_sphere = gamma_a/(1.0**(1.0/3)*3500.0**(2.0/3))
    assert np.isclose(ballisticCoeffMass(k_sphere, 3500.0, gamma_a), 1.0)

    # The deceleration of a body with the ballistic coefficient K is K*rho_atm*v^2
    lat, lon, height, jd, vel = np.radians(45.0), np.radians(15.0), 30000.0, 2460600.25, 8000.0
    k_arr = np.array([1e-3, 5e-3, 2e-2])
    atm_dens = getAtmDensity_vect(lat, lon, height, jd)
    decel = k_arr*atm_dens*vel**2

    dyn_mass = dynamicMass(3500.0, lat, lon, height, jd, vel, decel, gamma=1.0, shape_factor=0.78)
    assert np.allclose(ballisticCoeffMass(k_arr, 3500.0, 0.78), dyn_mass, rtol=1e-6)



def _massAndWarnings(**kwargs):
    """ Compute the mass of the synthetic light curve with calcMass(), and keep the warnings it raised.

    Keyword arguments:
        **kwargs: Keyword arguments of calcMass().

    Return:
        (mass, caught):
            - mass: [float] Mass (kg).
            - caught: [list] Messages of the warnings.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mass = calcMass(TIME, MAG_ABS, VELOCITY, **kwargs)

    return mass, [str(w.message) for w in caught]



def testCalcMassTakesNumpyIntegersAsModelCodes():
    """ Check that a numpy integer is a model code, as a Python int, and not a constant efficiency of
        100*tau %.
    """

    # Every code, as Python and numpy integers of several types
    for code in range(9):

        mass_int, _ = _massAndWarnings(tau=code, P_0m=1500.0)

        for np_type in (np.int8, np.int32, np.int64, np.uint16):
            mass_np, _ = _massAndWarnings(tau=np_type(code), P_0m=1500.0)
            assert mass_np == mass_int, (code, np_type)

    # RC2001 Type I by code, numpy code and name, against a constant tau of 1 (100 %)
    mass_name, _ = _massAndWarnings(tau='rc2001', P_0m=1500.0)
    mass_np, _ = _massAndWarnings(tau=np.int64(1), P_0m=1500.0)
    mass_const, _ = _massAndWarnings(tau=1.0, P_0m=1500.0)
    assert mass_np == mass_name
    assert mass_np > 10*mass_const

    # Codes taken from an array, as when looping over a table of models
    names = ('rc2001', 'kosice', 'camo', 'cm1976')
    codes = np.array([LUM_EFF_MODELS[name] for name in names])
    for name, code in zip(names, codes):
        assert _massAndWarnings(tau=code, P_0m=1500.0)[0] == _massAndWarnings(tau=name, P_0m=1500.0)[0]



def testCalcMassWarnsAboutP0mForThe1500WModels():
    """ Check that the models calibrated for P_0m = 1500 W warn about another P_0m, among them the Kosice
        model of Borovicka et al. (2013) by name, code and numpy code, and that the mass is not corrected.
    """

    # The Kosice model, by name, by code and by numpy code
    for tau in ('kosice', 'borovicka2013', 4, np.int64(4)):

        _, caught = _massAndWarnings(tau=tau, P_0m=840.0)
        assert len(caught) == 1, tau
        assert "lum_eff_type=4" in caught[0]
        assert "P_0m = 840 W" in caught[0]

        _, caught = _massAndWarnings(tau=tau, P_0m=1500.0)
        assert caught == [], tau

    # Every 1500 W model warns, the constant efficiency and CAMO do not
    for code in range(9):
        _, caught = _massAndWarnings(tau=code, P_0m=840.0)
        assert len(caught) == (1 if code in (1, 2, 3, 4, 6, 7, 8) else 0), code

    _, caught = _massAndWarnings(tau=0.007, P_0m=840.0)
    assert caught == []

    # The warning gives the factor the mass is off by, and the mass is not corrected
    mass_840, caught = _massAndWarnings(tau='kosice', P_0m=840.0)
    mass_1500, _ = _massAndWarnings(tau='kosice', P_0m=1500.0)

    assert "factor of 0.56" in caught[0]
    assert np.isclose(mass_840/mass_1500, 840.0/1500.0, rtol=1e-2)



def testCalcMassAtZeroLumEffMassIsTheSmallMassLimit():
    """ Check that lum_eff_mass = 0 does not switch the mass term off but takes its small-mass limit, while
        at 1 kg the mass term of RC2001 vanishes (tanh(0.38*ln(1)) = 0). At 20 km/s, tau is 1.39 % at 0 kg
        and 4.38 % at 1 kg, so the mass is 3.16 times larger at 0 kg.
    """

    mass_0 = calcMass(TIME, MAG_ABS, 20000.0, tau='rc2001', P_0m=1500.0, lum_eff_mass=0)
    mass_1 = calcMass(TIME, MAG_ABS, 20000.0, tau='rc2001', P_0m=1500.0, lum_eff_mass=1.0)

    assert np.isfinite(mass_0)
    assert np.isclose(mass_0/mass_1, 4.38150/1.38734, rtol=1e-3)



def testCalcMassConstantCodeIsAlways0p7Percent():
    """ Check that code 0 ('constant') is the constant 0.7 %, the same as the default float tau, and that
        another constant efficiency is given as a float.
    """

    mass_default, _ = _massAndWarnings()
    for tau in (0, 'constant', np.int64(0)):
        assert np.isclose(_massAndWarnings(tau=tau)[0], mass_default, rtol=1e-12)

    assert np.isclose(_massAndWarnings(tau=0.014)[0], mass_default/2, rtol=1e-12)




if __name__ == "__main__":

    # Run all tests
    test_list = [testBallisticCoeffMass, testCalcMassTakesNumpyIntegersAsModelCodes, \
        testCalcMassWarnsAboutP0mForThe1500WModels, testCalcMassAtZeroLumEffMassIsTheSmallMassLimit, \
        testCalcMassConstantCodeIsAlways0p7Percent]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
