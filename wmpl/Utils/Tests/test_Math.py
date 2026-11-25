""" Tests of the functions in wmpl.Utils.Math.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Math.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Utils.Math import medianStddev, circPercentile, findClosestPoints, findClosestPoints_vect, \
    isGraphConnected, weightedMedian, integratedAutocorrTime



def testMedianStddev():
    """ Check the standard deviation of the median against the known values for equal deviations. """

    # The median of one value has the same deviation, and of three about 0.670 of it
    assert np.isclose(medianStddev([10.0]), 10.0, rtol=0.02)
    assert np.isclose(medianStddev([10.0, 10.0, 10.0]), 6.70, rtol=0.02)

    # Values without a valid deviation are taken as exact
    assert medianStddev([None, np.inf]) == 0.0
    assert medianStddev([]) == 0.0



def testWeightedMedian():
    """ Check the weighted median against the median and a brute force minimum of the absolute deviations. """

    # With equal weights it is the median, also of an even number of values
    assert weightedMedian([3.0, 1.0, 2.0], [1.0, 1.0, 1.0]) == 2.0
    assert weightedMedian([4.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]) == 2.5

    # A heavy value pulls the median to it
    assert weightedMedian([1.0, 2.0, 10.0], [1.0, 1.0, 5.0]) == 10.0

    # The weighted sum of absolute deviations is the smallest at the weighted median
    rng = np.random.RandomState(1)
    x = rng.normal(0, 1, 51)
    weights = rng.uniform(0.1, 2.0, 51)
    x_grid = np.linspace(-3, 3, 6001)
    cost = [np.sum(weights*np.abs(x - x_val)) for x_val in x_grid]
    assert abs(weightedMedian(x, weights) - x_grid[np.argmin(cost)]) < 1e-3



def testCircPercentileOfAnglesFrom0To360():
    """ Check the median of angles from 0 to 360 deg, also across the 0/360 break and with a given 0th
        percentile.
    """

    # Angles away from the break
    assert np.isclose(circPercentile(np.radians([10.0, 20.0, 30.0, 40.0, 50.0]), 50), np.radians(30.0))

    # Angles beyond 360 deg, with the median the middle value
    assert np.isclose(circPercentile([70.0, 70.1, 70.2], 50), 70.1%(2*np.pi))

    # Angles across the 0/360 break, with the median at 0
    median = circPercentile(np.radians([350.0, 355.0, 0.0, 5.0, 10.0]), 50)
    assert np.isclose(np.cos(median), 1.0)

    # A given 0th percentile outside 0-360 is taken as the same angle, so the median is the middle value
    for dtype in (np.float64, np.float32):
        data = np.array([82.0, 82.1, 82.2], dtype=dtype)
        assert np.isclose(circPercentile(data, 50, q0=82.0), 82.1%(2*np.pi), atol=1e-5)

    # A tiny negative angle with a given 0th percentile of 0 stays at the end of the ordering
    assert np.isclose(circPercentile([-1e-16, 0.1, 0.2], 50, q0=0), 0.2)

    # Tiny negative angles which wrap to the same value keep their order along the circle
    for data in ([-1e-16, -2e-16, 0.1], [-2e-16, -1e-16, 0.1]):
        assert np.isclose(np.cos(circPercentile(data, 50)), 1.0)

    # A sample without a value (e.g. the anomaly of a hyperbolic orbit) gives NaN
    assert np.isnan(circPercentile([0.1, np.nan, 0.2], 50))

    # The same angle given in different revolutions is the same sample
    a = 11*2*np.pi + 0.1
    assert np.isclose(circPercentile([a%(2*np.pi), a, a + 0.1], 50), 0.1)

    # The angles can be given in an array of any shape
    assert np.isclose(circPercentile([[0.4, 0.3], [0.2, 0.1]], 50), 0.25)



def testCircPercentileOfAnglesFromMinus180To180():
    """ Check that angles from -180 to 180 deg give the same percentiles as from 0 to 360, also when they
        are more than 180 deg apart and none is above 180 deg, which raised a ValueError (e.g. longitudes
        on both sides of the date line).
    """

    # Angles around 0
    data = np.radians([-20.0, -10.0, 0.0, 10.0, 20.0])
    assert np.isclose(circPercentile(data, 50), circPercentile(data%(2*np.pi), 50))

    # Longitudes on both sides of the date line
    data = np.radians([179.8, 179.9, -179.9, -179.8])
    for percentile in [2.5, 50, 97.5]:
        assert np.isclose(circPercentile(data, percentile), circPercentile(data%(2*np.pi), percentile))

    assert 179.8 <= np.degrees(circPercentile(data, 2.5)) <= 180.2




def testClosestPointsOfManyLinesAreThoseOfEachLine():
    """ Check that the closest points of many pairs of lines at once are those of each pair, with one or many
        second lines, and infinite for parallel lines.
    """

    rng = np.random.default_rng(1)
    P = rng.normal(0, 1e5, (20, 3))
    u = rng.normal(0, 1, (20, 3))
    Q = rng.normal(0, 1e5, (20, 3))
    v = rng.normal(0, 1, 3)

    # One second line, and one second line per pair
    for Q_test in (Q[0], Q):

        S, T, d = findClosestPoints_vect(P, u, Q_test, v)
        Q_rows = np.reshape(Q_test, (-1, 3))

        for i in range(len(P)):
            S_i, T_i, d_i = findClosestPoints(P[i], u[i], Q_rows[i%len(Q_rows)], v)

            assert np.allclose(S[i], S_i, rtol=1e-12, atol=1e-6)
            assert np.allclose(T[i], T_i, rtol=1e-12, atol=1e-6)
            assert np.isclose(d[i], d_i, rtol=1e-9, atol=1e-6)

    # Parallel lines have no closest points
    u[3] = 2*v
    S, T, d = findClosestPoints_vect(P, u, Q, v)

    assert np.all(np.isinf(S[3])) and np.all(np.isinf(T[3])) and np.isinf(d[3])
    assert np.all(np.isfinite(d[np.arange(len(P)) != 3]))



def testGraphConnectivity():
    """ A chain of nodes is connected, two groups of nodes are not, and edges to other nodes are not used. """

    assert isGraphConnected([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    assert isGraphConnected(['A', 'B', 'C'], [('C', 'A'), ('B', 'C')])

    # Two groups joined only through a node which is not in the graph
    assert not isGraphConnected([0, 2, 3, 4], [(0, 2), (3, 4), (0, 1), (1, 3)])

    # A node without edges
    assert not isGraphConnected([0, 1, 2], [(0, 1)])

    # One node or none
    assert isGraphConnected([5], [])
    assert isGraphConnected([], [])



def testIntegratedAutocorrTime():
    """ The autocorrelation time is about 1 for independent values and large for correlated ones. """

    rng = np.random.RandomState(0)

    # Independent values
    white_noise = rng.normal(0, 1, 2000)
    assert 1.0 <= integratedAutocorrTime(white_noise) < 1.2

    # An autoregressive series with the coefficient 0.8, whose autocorrelation time is (1 + 0.8)/(1 - 0.8) = 9
    correlated = np.zeros(20000)
    for i in range(1, len(correlated)):
        correlated[i] = 0.8*correlated[i - 1] + rng.normal()
    assert 6.0 < integratedAutocorrTime(correlated) < 10.0

    # A constant series has no autocorrelation
    assert np.isnan(integratedAutocorrTime(np.ones(10)))




if __name__ == "__main__":

    # Run all tests
    test_list = [testMedianStddev, testWeightedMedian, testCircPercentileOfAnglesFrom0To360, \
        testCircPercentileOfAnglesFromMinus180To180, testClosestPointsOfManyLinesAreThoseOfEachLine, \
        testGraphConnectivity, testIntegratedAutocorrTime]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
