""" Settings of the tests of wmpl, run with pytest. """

from __future__ import print_function, division, absolute_import

import os


# Solve the trajectories of the tests with UTC as UT1, unless the environment gives UT1 - UTC
os.environ.setdefault('WMPL_UT1_UTC', '0')
