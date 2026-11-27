""" Tests of the fragment loop of MetSim in Cython, besides the comparison with the Python reference in
test_MetSimLoopReference.py.

The tests check that:
    - the powers of the loop are the same doubles as the powers of Python floats, also where a C compiler
      would replace pow(x, 2.0) by x*x, which is not always the same double,
    - a fragment without speed raises ZeroDivisionError, as in the Python loop, by which Dynesty rejects a
      draw,
    - an object which is not a fragment raises TypeError,
    - the substep which resolved the motion of a small grain is returned, and a step started from it gives
      the same change of the mass as a step started from the full time step, to the accuracy of the
      integration.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimCythonLoop.py -v
"""

from __future__ import print_function, division, absolute_import

import math

import numpy as np
import pytest

import wmpl.MetSim.MetSimErosion as MetSimErosion
import wmpl.MetSim.Tests.MetSimErosionReference as MetSimErosionReference
from wmpl.MetSim.MetSimErosionCyTools import pythonPow, ablationStepRK4, ablationStepRK4Substep
from wmpl.MetSim.Tests.DefaultFragment import initFragment



def testPowerIsThePowerOfPython():
    """ Check that the powers of the loop are the same doubles as in Python, for squares of the values the
        loop squares (speeds, lengths, distances from the centre of the Earth, time steps) and other
        exponents.
    """

    # Take random speeds, distances from the centre of the Earth, lengths, time steps and ratios
    rng = np.random.default_rng(1)

    values = np.concatenate([rng.uniform(1e3, 8e4, 50000), rng.uniform(6.3e6, 6.6e6, 50000), \
        rng.uniform(0.0, 3e5, 50000), rng.uniform(1e-4, 0.1, 50000), rng.uniform(0.98, 1.02, 50000)])

    # Compare the powers of the loop with the powers of Python floats, bit for bit
    for exponent in [2.0, 3.0, 0.5, -2/3.0, 2/3.0]:
        for x in values:
            x = float(x)
            assert pythonPow(x, exponent) == x**exponent, (x, exponent)



def testZeroSpeedRaisesZeroDivisionError():
    """ Check that a fragment without speed raises ZeroDivisionError, as in the Python loop. """

    # Run the loop in Cython and the Python reference on a fragment without speed
    for module in [MetSimErosion, MetSimErosionReference]:

        frag, const = initFragment(module)
        const.atm_density_model = 'polynomial'
        frag.v = 0.0

        with pytest.raises(ZeroDivisionError):
            module.ablateAll([frag], const)



def testOtherObjectRaisesTypeError():
    """ Check that an object which is not a fragment in the list of fragments raises TypeError. """

    # A fragment of the Python reference is not a fragment of the Cython class
    frag, const = initFragment(MetSimErosionReference)
    const.atm_density_model = 'polynomial'

    with pytest.raises(TypeError):
        MetSimErosion.ablateAll([frag], const)



def testSubstepIsRemembered():
    """ Check that a small fast grain deep in the atmosphere is integrated in substeps, that the last of them
        is returned, and that a step started from it gives the same change of the mass as one started from the
        full time step, to the accuracy of the integration.
    """

    # Take the default constants and a time step of 5 ms
    const = MetSimErosion.Constants()
    dens_co = np.ascontiguousarray(const.dens_co, dtype=np.float64)
    dt = 0.005
    m = 1e-11

    # A 1e-11 kg grain at 60 km/s at 90 km, which loses most of its mass in a step
    args = (dt, 1.21*3000**(-2/3.0), 0.02e-6, m, 60000.0, 0.0, 90000.0, math.radians(45), const.r_earth, \
        0.0, None, dens_co, 1.0, const.lum_eff_type, const.lum_eff, const.mu, const.v_kill)

    # Integrate the step from the full time step, which the grain must split into substeps
    results = ablationStepRK4Substep(*args, 0.0)
    h_substep = results[5]

    assert 0 < h_substep < dt

    # Starting from the full time step is the same as ablationStepRK4
    assert results[:5] == ablationStepRK4(*args)

    # Starting from the remembered substep changes the mass by the same amount
    results_substep = ablationStepRK4Substep(*args, h_substep)
    assert abs(results_substep[0] - results[0]) < 1e-5*m
