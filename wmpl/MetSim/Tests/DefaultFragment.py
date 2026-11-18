""" A fragment of the default MetSim constants, shared by the tests of the fragment class and of the loop over
the fragments.
"""

from __future__ import print_function, division, absolute_import

import wmpl.MetSim.MetSimErosion as MetSimErosion



def initFragment(module):
    """ Make an initialized fragment of the given module with the default constants.

    Arguments:
        module: [module] MetSimErosion or MetSimErosionReference.

    Return:
        (frag, const): [tuple] Fragment and the constants.
    """

    # Take the default constants
    const = MetSimErosion.Constants()

    # Make the fragment and set its initial conditions from the constants
    frag = module.Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, const.zenith_angle, \
        const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)

    return frag, const
