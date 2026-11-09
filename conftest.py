""" Settings of the tests of wmpl, run with pytest. """

import os


# The synthetic observations of the tests are computed with the conversions of wmpl, which take UTC as UT1,
#   and the reference values of the tests were computed in the same way. So the trajectories are solved
#   without UT1 - UTC (see wmpl.Utils.EarthRotation), unless the environment gives it. The tests of the
#   correction give it to the solver themselves.
os.environ.setdefault("WMPL_UT1_UTC", "0")
