""" Tests for how wmpl.Utils.Physics.calcMass() selects the luminous efficiency model and warns about P_0m.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_PhysicsCalcMass.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_PhysicsCalcMass
"""

import warnings

import numpy as np

from wmpl.Utils.Physics import calcMass, LUM_EFF_MODELS


# A synthetic light curve: a 2 s meteor peaking at -2 mag, decelerating from 20 to 15 km/s
TIME = np.linspace(0, 2, 41)
MAG_ABS = -2 + 3*(TIME - 1.0)**2
VELOCITY = np.linspace(20000, 15000, 41)


def _massAndWarnings(**kwargs):
    """ calcMass() on the synthetic light curve, and the warnings it raised. """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mass = calcMass(TIME, MAG_ABS, VELOCITY, **kwargs)

    return mass, [str(w.message) for w in caught]


def test_numpy_integer_selects_the_model():
    """ A numpy integer is a model code, as a Python int, and not a constant efficiency of 100*tau %. """

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


def test_numpy_integer_codes_from_an_array():
    """ Codes taken from an array (as when looping over a table of models) give the named models. """

    codes = np.array([LUM_EFF_MODELS[name] for name in ('rc2001', 'kosice', 'camo', 'cm1976')])
    for name, code in zip(('rc2001', 'kosice', 'camo', 'cm1976'), codes):
        assert _massAndWarnings(tau=code, P_0m=1500.0)[0] == _massAndWarnings(tau=name, P_0m=1500.0)[0]


def test_kosice_warns_with_p0m_not_1500():
    """ Borovicka et al. (2013) used 1500 W for a zero magnitude meteor, so the Kosice model warns as the other
        1500 W models do, by name, by code and by numpy code. """

    for tau in ('kosice', 'borovicka2013', 4, np.int64(4)):

        _, caught = _massAndWarnings(tau=tau, P_0m=840.0)
        assert len(caught) == 1, tau
        assert "lum_eff_type=4" in caught[0]
        assert "P_0m = 840 W" in caught[0]

        _, caught = _massAndWarnings(tau=tau, P_0m=1500.0)
        assert caught == [], tau


def test_p0m_warning_models():
    """ Every 1500 W model warns with a P_0m of 840 W; the constant efficiency and CAMO do not. """

    for code in range(9):
        _, caught = _massAndWarnings(tau=code, P_0m=840.0)
        assert len(caught) == (1 if code in (1, 2, 3, 4, 6, 7, 8) else 0), code

    _, caught = _massAndWarnings(tau=0.007, P_0m=840.0)
    assert caught == []


def test_warning_does_not_change_the_mass():
    """ The warning reports the factor the mass is off by, and the mass is not corrected. """

    mass_840, caught = _massAndWarnings(tau='kosice', P_0m=840.0)
    mass_1500, _ = _massAndWarnings(tau='kosice', P_0m=1500.0)

    assert "factor of 0.56" in caught[0]
    assert np.isclose(mass_840/mass_1500, 840.0/1500.0, rtol=1e-2)


def test_lum_eff_mass_zero_is_the_small_mass_limit():
    """ lum_eff_mass = 0 does not switch the mass term off: it evaluates the model in its small-mass limit, while
        at 1 kg the mass term of RC2001 vanishes (tanh(0.38*ln(1)) = 0). At 20 km/s, tau is 1.39 % at 0 kg and
        4.38 % at 1 kg, so the mass is 3.16 times larger at 0 kg. """

    mass_0 = calcMass(TIME, MAG_ABS, 20000.0, tau='rc2001', P_0m=1500.0, lum_eff_mass=0)
    mass_1 = calcMass(TIME, MAG_ABS, 20000.0, tau='rc2001', P_0m=1500.0, lum_eff_mass=1.0)

    assert np.isfinite(mass_0)
    assert np.isclose(mass_0/mass_1, 4.38150/1.38734, rtol=1e-3)


if __name__ == "__main__":

    test_functions = [
        test_numpy_integer_selects_the_model,
        test_numpy_integer_codes_from_an_array,
        test_kosice_warns_with_p0m_not_1500,
        test_p0m_warning_models,
        test_warning_does_not_change_the_mass,
        test_lum_eff_mass_zero_is_the_small_mass_limit,
    ]

    failed = 0
    for test_func in test_functions:

        try:
            test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {!r}".format(test_func.__name__, e))

    print()
    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
