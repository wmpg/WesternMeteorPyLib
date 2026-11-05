""" Tests of the functions in wmpl.Utils.Math.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Math.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Utils.Math import medianStddev



def testMedianStddev():
    """ Check the standard deviation of the median against the known values for equal deviations. """

    # The median of one value has the same deviation, and of three about 0.670 of it
    assert np.isclose(medianStddev([10.0]), 10.0, rtol=0.02)
    assert np.isclose(medianStddev([10.0, 10.0, 10.0]), 6.70, rtol=0.02)

    # Values without a valid deviation are taken as exact
    assert medianStddev([None, np.inf]) == 0.0
    assert medianStddev([]) == 0.0




if __name__ == "__main__":

    testMedianStddev()
    print("PASS testMedianStddev")
