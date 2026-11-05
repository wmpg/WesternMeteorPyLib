""" Tests for the percentiles of angles, which take any range of angles.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_CircPercentile.py -v
"""

import numpy as np

from wmpl.Utils.Math import circPercentile


def test_angles_within_0_and_360_are_as_before():

    data = np.radians([10.0, 20.0, 30.0, 40.0, 50.0])

    assert np.isclose(circPercentile(data, 50), np.radians(30.0))


def test_angles_on_the_0_360_break():

    data = np.radians([350.0, 355.0, 0.0, 5.0, 10.0])

    assert np.isclose(circPercentile(data, 50), 0.0) or np.isclose(circPercentile(data, 50), 2*np.pi)


def test_angles_from_minus_180_to_180_more_than_180_apart():
    """ This raised a ValueError: their range is above 180 degrees, and none of them is above 180. """

    data = np.radians([-120.0, -110.0, 80.0, 90.0])

    assert 0 <= circPercentile(data, 50) < 2*np.pi


def test_negative_angles_are_the_same_as_the_positive_ones():

    data = np.radians([-20.0, -10.0, 0.0, 10.0, 20.0])

    assert np.isclose(circPercentile(data, 50)%(2*np.pi), circPercentile(data%(2*np.pi), 50)%(2*np.pi))
