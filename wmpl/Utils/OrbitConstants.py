""" Constants shared by the orbit dissimilarity criteria and the orbit classification.

These live in a module of their own so that wmpl.Utils.Dcriteria, which needs only three of them,
does not have to import wmpl.Utils.OrbitClassification and with it scipy.integrate and
scipy.optimize. Nothing here imports anything beyond numpy.
"""

from __future__ import print_function, division, absolute_import


# Gaussian gravitational constant [AU^1.5/day] and its square, the solar gravitational parameter in
#   units of AU, day and solar masses [AU^3/day^2]
GAUSS_K = 0.01720209895
GAUSS_K_SQUARED = GAUSS_K**2

# Semi-major axis of Jupiter [AU], the value Tancredi (2014) works in. It is needed to reproduce the
#   resonance semi-major axes of that paper's table 1 to the three decimals it prints, so it is not
#   interchangeable with the 5.204267 used by wmpl.Rebound.REBOUND and wmpl.Trajectory.Orbit:
#   substituting that value moves the 1:1 resonance to 5.2043 and misses the tabulated 5.203. The
#   difference is 1.7e-4 relative and shifts a Tisserand parameter by about 1e-4, far below any
#   class boundary, so the two coexist deliberately rather than by oversight.
A_JUPITER = 5.20336

# Semi-major axes of the other giant planets [AU]
A_SATURN = 9.5826
A_URANUS = 19.2018
A_NEPTUNE = 30.0470
