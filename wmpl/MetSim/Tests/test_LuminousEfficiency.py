""" Regression tests for the ReVelle & Ceplecha (2001) deceleration correction in luminousEfficiency().

The correction f(dv) = 0.26*ln(dv) + 0.0042*ln(dv)^3, dv = (v_init - vel) in km/s, diverges to -inf as
dv -> 0 and is undefined for vel >= v_init. It is bounded by flooring dv at 0.1 km/s, which keeps the
correction monotonic in dv (f is strictly increasing everywhere, so any bound that reshapes it below the
floor would put a spurious minimum at the splice). These tests pin that behaviour against the compiled
extension, so a future bound cannot silently reintroduce a dip or a NaN.

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_LuminousEfficiency
"""

import numpy as np
import pytest

# Importing the engine compiles the Cython extension through pyximport, if needed
import wmpl.MetSim.MetSimErosion  # noqa: F401
from wmpl.MetSim.MetSimErosionCyTools import luminousEfficiency


# Test point: a 20 km/s, 1 kg meteoroid, evaluated with the RC2001 Type I, II and III models
VEL = 20000.0
MASS = 1.0
RC2001_TYPES = [1, 2, 3]

# The floor of the velocity difference, as in luminousEfficiency()
DV_MIN_KMS = 0.1


def tauAtDv(lum_eff_type, dv_kms):
    """ Luminous efficiency with the correction evaluated at the given velocity difference (km/s). """

    return luminousEfficiency(lum_eff_type, 0.7, VEL, MASS, VEL + 1000.0*dv_kms)


def tauUncorrected(lum_eff_type):
    """ Luminous efficiency with the deceleration correction switched off (v_init <= 0). """

    return luminousEfficiency(lum_eff_type, 0.7, VEL, MASS, -1.0)


@pytest.mark.parametrize("lum_eff_type", RC2001_TYPES)
def test_rc2001_correction_is_finite_and_monotonic_in_dv(lum_eff_type):
    """ Over the whole physical range, including vel > v_init, tau is finite and never decreases with dv. """

    dv = np.linspace(-5.0, 40.0, 20001)
    tau = np.array([tauAtDv(lum_eff_type, d) for d in dv])

    assert np.all(np.isfinite(tau))
    assert np.all(tau > 0)

    # Non-decreasing: a linear taper to zero below the floor would fail this with a dip at dv = 0.1 km/s
    assert np.all(np.diff(tau) >= 0)


@pytest.mark.parametrize("lum_eff_type", RC2001_TYPES)
def test_rc2001_correction_is_floored_at_dv_min(lum_eff_type):
    """ Below the floor, and for vel >= v_init, the correction holds its dv = 0.1 km/s value exactly. """

    tau_floor = tauAtDv(lum_eff_type, DV_MIN_KMS)
    floor_factor = np.exp(0.26*np.log(DV_MIN_KMS) + 0.0042*np.log(DV_MIN_KMS)**3)

    assert tau_floor/tauUncorrected(lum_eff_type) == pytest.approx(floor_factor, rel=1e-12)

    for dv in [0.099, 0.05, 0.01, 0.0, -0.01, -5.0]:
        assert tauAtDv(lum_eff_type, dv) == tau_floor

    # Just above the floor the correction moves again
    assert tauAtDv(lum_eff_type, 0.101) > tau_floor


@pytest.mark.parametrize("lum_eff_type", RC2001_TYPES)
def test_rc2001_correction_is_neutral_at_one_kms(lum_eff_type):
    """ f(1 km/s) = 0, so at dv = 1 km/s the corrected and uncorrected efficiencies coincide. """

    assert tauAtDv(lum_eff_type, 1.0) == pytest.approx(tauUncorrected(lum_eff_type), rel=1e-12)


def test_other_models_ignore_v_init():
    """ Models that are not RC2001 must not react to v_init at all. """

    for lum_eff_type in [0, 4, 5, 6, 7, 8]:
        assert luminousEfficiency(lum_eff_type, 0.7, VEL, MASS, -1.0) \
            == luminousEfficiency(lum_eff_type, 0.7, VEL, MASS, VEL + 50.0)


if __name__ == "__main__":

    import sys
    sys.exit(pytest.main([__file__, "-q"]))
