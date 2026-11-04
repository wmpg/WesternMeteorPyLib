""" Tests of the functions in wmpl.Utils.Physics.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Physics.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect
from wmpl.Utils.Physics import ballisticCoeffMass, dynamicMass



def testBallisticCoeffMass():
    """ Check the mass from the ballistic coefficient against its definition and against the dynamic mass
        from the deceleration.
    """

    # A sphere of 1 kg with Gamma = 0.5 and the bulk density 3500 kg/m^3
    gamma_a = 0.5*1.21
    k_sphere = gamma_a/(1.0**(1.0/3)*3500.0**(2.0/3))
    assert np.isclose(ballisticCoeffMass(k_sphere, 3500.0, gamma_a), 1.0)

    # The deceleration of a body with the ballistic coefficient K is K*rho_atm*v^2
    lat, lon, height, jd, vel = np.radians(45.0), np.radians(15.0), 30000.0, 2460600.25, 8000.0
    k_arr = np.array([1e-3, 5e-3, 2e-2])
    atm_dens = getAtmDensity_vect(lat, lon, height, jd)
    decel = k_arr*atm_dens*vel**2

    dyn_mass = dynamicMass(3500.0, lat, lon, height, jd, vel, decel, gamma=1.0, shape_factor=0.78)
    assert np.allclose(ballisticCoeffMass(k_arr, 3500.0, 0.78), dyn_mass, rtol=1e-6)




if __name__ == "__main__":

    # Run all tests
    test_list = [testBallisticCoeffMass]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
