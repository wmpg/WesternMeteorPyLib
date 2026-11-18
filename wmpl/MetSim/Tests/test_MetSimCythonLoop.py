""" Tests of the fragment loop of MetSim in Cython, besides the comparison with the Python reference in
test_MetSimLoopReference.py.

The tests check that:
    - the powers of the loop are the same doubles as the powers of Python floats, also where a C compiler
      would replace pow(x, 2.0) by x*x, which is not always the same double,
    - a fragment without speed raises ZeroDivisionError, as in the Python loop, by which Dynesty rejects a
      draw,
    - an object which is not a fragment raises TypeError.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimCythonLoop.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

import wmpl.MetSim.MetSimErosion as MetSimErosion
import wmpl.MetSim.Tests.MetSimErosionReference as MetSimErosionReference
from wmpl.MetSim.MetSimErosionCyTools import pythonPow
from wmpl.MetSim.Tests.DefaultFragment import initFragment



def testPowerIsThePowerOfPython():
    """ Check that the powers of the loop are the same doubles as in Python, for squares of the values the
        loop squares (speeds, lengths, distances from the centre of the Earth, time steps) and other
        exponents. """

    rng = np.random.default_rng(1)

    values = np.concatenate([rng.uniform(1e3, 8e4, 50000), rng.uniform(6.3e6, 6.6e6, 50000), \
        rng.uniform(0.0, 3e5, 50000), rng.uniform(1e-4, 0.1, 50000), rng.uniform(0.98, 1.02, 50000)])

    for exponent in [2.0, 3.0, 0.5, -2/3.0, 2/3.0]:
        for x in values:
            x = float(x)
            assert pythonPow(x, exponent) == x**exponent, (x, exponent)



def testZeroSpeedRaisesZeroDivisionError():
    """ Check that a fragment without speed raises ZeroDivisionError, as in the Python loop. """

    for module in [MetSimErosion, MetSimErosionReference]:

        frag, const = initFragment(module)
        const.atm_density_model = 'polynomial'
        frag.v = 0.0

        with pytest.raises(ZeroDivisionError):
            module.ablateAll([frag], const)



def testOtherObjectRaisesTypeError():
    """ Check that an object which is not a fragment in the list of fragments raises TypeError. """

    frag, const = initFragment(MetSimErosionReference)
    const.atm_density_model = 'polynomial'

    with pytest.raises(TypeError):
        MetSimErosion.ablateAll([frag], const)
