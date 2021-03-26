
""" Implementation of the Borovicka (2007) meteor erosion model with added disruption.

References:
    Borovička, J., Spurný, P., & Koten, P. (2007). Atmospheric deceleration and light curves of Draconid 
    meteors and implications for the structure of cometary dust. Astronomy & Astrophysics, 473(2), 661-672.

    Campbell-Brown, M. D., Borovička, J., Brown, P. G., & Stokan, E. (2013). High-resolution modelling of 
    meteoroid ablation. Astronomy & Astrophysics, 557, A41.

"""

from __future__ import print_function, division, absolute_import


import math
import copy

import numpy as np
import scipy.stats


# Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from wmpl.MetSim.MetSimErosionCyTools import massLossRK4, decelerationRK4, luminousEfficiency, atmDensityPoly


### DEFINE CONSTANTS

# Earth acceleration in m/s^2 on the surface
G0 = 9.81

# Earth radius (m) at 43.930723 deg latitude
R_EARTH = 6367888.0

# The mass bin coefficient makes sure that there are 10 mass bins per order of magnitude
MASS_BIN_COEFF = 10**(-0.1)

###



class Constants(object):
    def __init__(self):
        """ Constant parameters for the ablation modelling. """

        ### Simulation parameters ###

        # Time step
        self.dt = 0.005

        # Time elapsed since the beginning
        self.total_time = 0

        # Number of active fragments
        self.n_active = 0

        # Minimum possible mass for ablation (kg)
        self.m_kill = 1e-14

        # Minimum ablation velocity (m/s)
        self.v_kill = 3000

        # Minimum height (m)
        self.h_kill = 60000

        # Initial meteoroid height (m)
        self.h_init = 180000

        # Power of a 0 magnitude meteor
        self.P_0m = 840

        # Atmosphere density coefficients
        self.dens_co = np.array([6.96795507e+01, -4.14779163e+03, 9.64506379e+04, -1.16695944e+06, \
            7.62346229e+06, -2.55529460e+07, 3.45163318e+07])


        self.total_fragments = 0

        ### ###


        ### Wake parameters ###

        # PSF stddev (m)
        self.wake_psf = 3.0

        # Wake extension from the leading fragment (m)
        self.wake_extension = 200

        ### ###



        ### Main meteoroid properties ###

        # Meteoroid bulk density (kg/m^3)
        self.rho = 1000

        # Initial meteoroid mass (kg)
        self.m_init = 2e-5

        # Initial meteoroid veocity (m/s)
        self.v_init = 23570

        # Shape factor (1.21 is sphere)
        self.shape_factor = 1.21

        # Main fragment ablation coefficient (s^2/km^2)
        self.sigma = 0.023/1e6

        # Zenith angle (radians)
        self.zenith_angle = math.radians(45)

        # Drag coefficient
        self.gamma = 1.0

        # Grain bulk density (kg/m^3)
        self.rho_grain = 3000


        # Luminous efficiency type
        #   0 - Constant
        #   1 - TDB
        #   2 - TDB ...
        self.lum_eff_type = 0

        # Constant luminous efficiency (percent)
        self.lum_eff = 0.7

        ### ###


        ### Erosion properties ###

        # Toggle erosion on/off
        self.erosion_on = True

        
        # Height at which the erosion starts (meters)
        self.erosion_height_start = 102000

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = 0.33/1e6

        
        # Height at which the erosion coefficient changes (meters)
        self.erosion_height_change = 90000

        # Erosion coefficient after the change (s^2/m^2)
        self.erosion_coeff_change = 0.33/1e6

        # Density after erosion change (density of small chondrules by default)
        self.erosion_rho_change = 3700

        # Abaltion coeff after erosion change
        self.erosion_sigma_change = self.sigma


        # Grain mass distribution index
        self.erosion_mass_index = 2.5

        # Mass range for grains (kg)
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10

        ###


        ### Disruption properties ###

        # Toggle disruption on/off
        self.disruption_on = True

        # Meteoroid compressive strength (Pa)
        self.compressive_strength = 2000

        # Height of disruption (will be assigned when the disruption occures)
        self.disruption_height = None

        # Erosion coefficient to use after disruption
        self.disruption_erosion_coeff = self.erosion_coeff

        # Disruption mass distribution index
        self.disruption_mass_index = 2.0


        # Mass ratio for disrupted fragments as the ratio of the disrupted mass
        self.disruption_mass_min_ratio = 1.0/100
        self.disruption_mass_max_ratio = 10.0/100

        # Ratio of mass that will disrupt into grains
        self.disruption_mass_grain_ratio = 0.25

        ### ###


        ### Complex fragmentation behaviour ###

        # Indicate if the complex fragmentation is used
        self.fragmentation_on = False

        # A list of fragmentation entries
        self.fragmentation_entries = []

        # Name of the fragmentation file
        self.fragmentation_file_name = "metsim_fragmentation.txt"

        ### ###




class Fragment(object):
    def __init__(self):

        self.id = 0

        self.const = None

        # Shape-density coeff
        self.K = 0

        # Initial fragment mass
        self.m_init = 0

        # Instantaneous fragment mass Mass (kg)
        self.m = 0

        # Density (kg/m^3)
        self.rho = 0

        # Ablation coefficient (s^2/m^2)
        self.sigma = 0

        # Velocity (m/s)
        self.v = 0

        # Velocity components (vertical and horizontal)
        self.vv = 0
        self.vh = 0

        # Length along the trajectory
        self.length = 0

        # Luminous intensity (Watts)
        self.lum = 0

        # Erosion coefficient value
        self.erosion_coeff = 0

        # Grain mass distribution index
        self.erosion_mass_index = 2.5

        # Mass range for grains (kg)
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10


        self.erosion_enabled = False

        self.disruption_enabled = False

        self.active = False
        self.n_grains = 1

        # Indicate that this is the main fragment
        self.main = False

        # Indicate that this is born out of complex fragmentation
        self.complex = False


    def init(self, const, m, rho, v_init, sigma, gamma, zenith_angle, erosion_mass_index, erosion_mass_min, \
        erosion_mass_max):

        self.const = const

        self.m = m
        self.m_init = m
        self.h = const.h_init
        self.rho = rho
        self.v = v_init
        self.sigma = sigma
        self.gamma = gamma
        self.zenith_angle = zenith_angle

        # Compute shape-density coeff
        self.updateShapeDensityCoeff()

        self.erosion_mass_index = erosion_mass_index
        self.erosion_mass_min = erosion_mass_min
        self.erosion_mass_max = erosion_mass_max

        # Compute velocity components
        self.vv = -v_init*math.cos(zenith_angle)
        self.vh = v_init*math.sin(zenith_angle)

        self.active = True
        self.n_grains = 1

    def updateShapeDensityCoeff(self):
        """ Update the value of the shape-density coefficient. """

        self.K = self.gamma*self.const.shape_factor*self.rho**(-2/3.0)




class Wake(object):
    def __init__(self, const, frag_list, leading_frag_length, length_array):
        """ Container for the evaluated wake. 
        
        Arguments:
            const: [Constants]
            frag_list: [list of Fragment object] A list of active fragments visible in the wake.
            leading_frag_length: [float] Length from the beginning of the simulation of the leading fragment.
            length_array: [ndarray] An array of lengths (zero centered to the leading fragment) over which 
                the lag will be evaluated.
        """

        # Constants
        self.const = const

        # List of active fragments within the window
        self.frag_list = frag_list

        # Length of the leading fragment
        self.leading_frag_length = leading_frag_length

        # Array of lengths for plotting (independent variable)
        self.length_array = length_array

        # Length of visible fragments
        self.length_points = np.array([frag.length - self.leading_frag_length for frag in self.frag_list])

        # Luminosity of visible fragments
        self.luminosity_points = np.array([frag.lum for frag in self.frag_list])


        # Evalute the Gaussian at every fragment an add to the estimated wake
        self.wake_luminosity_profile = np.zeros_like(length_array)
        
        for frag_lum, frag_len in zip(self.luminosity_points, self.length_points):

            self.wake_luminosity_profile += frag_lum*scipy.stats.norm.pdf(self.length_array, loc=frag_len, \
                scale=const.wake_psf)




def generateFragments(const, frag_parent, eroded_mass, mass_index, mass_min, mass_max, keep_eroding=False,
    disruption=False):
    """ Given the parent fragment, fragment it into daughter fragments using a power law mass distribution.

    Masses are binned and one daughter fragment may represent several fragments/grains, which is specified 
    with the n_grains atribute.

    Arguments:
        const: [object] Constants instance.
        frag_parent: [object] Fragment instance, the parent fragment.
        eroded_mass: [float] Mass to be distributed into daughter fragments. 
        mass_index: [float] Mass index to use to distribute the mass.
        mass_min: [float] Minimum mass bin (kg).
        mass_max: [float] Maximum mass bin (kg).

    Keyword arguments:
        keep_eroding: [bool] Whether the daughter fragments should keep eroding.
        disruption: [bool] Indicates that the disruption occured, uses a separate erosion parameter for
            disrupted daughter fragments.

    Return:
        frag_children: [list] A list of Fragment instances - these are the generated daughter fragments.

    """

    # Compute the number of mass bins
    k = math.ceil(abs(math.log10(mass_max/mass_min)/math.log10(MASS_BIN_COEFF)))

    # Compute the number of the largest grains
    if mass_index == 2:
        n0 = eroded_mass/(mass_max*(k + 1))
    else:
        n0 = abs((eroded_mass/mass_max)*(1 - MASS_BIN_COEFF**(2 - mass_index))/(1 - MASS_BIN_COEFF**((2 - mass_index)*(k + 1))))


    # Go though every mass bin
    frag_children = []
    leftover_mass = 0
    for i in range(0, k + 1):

        # Compute the mass of all grains in the bin (per grain)
        m_grain = mass_max*MASS_BIN_COEFF**i

        # Compute the number of grains in the bin
        n_grains_bin = n0*(mass_max/m_grain)**(mass_index - 1) + leftover_mass/m_grain
        n_grains_bin_round = int(math.floor(n_grains_bin))

        # Compute the leftover mass
        leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain

        # If there are any grains to erode, erode them
        if n_grains_bin_round > 0:

            # Init the new fragment with params of the parent
            frag_child = copy.deepcopy(frag_parent)

            # Assign the number of grains this fragment stands for (make sure to preserve the previous value
            #   if erosion is done for more fragments)
            frag_child.n_grains *= n_grains_bin_round

            # Assign the grain mass
            frag_child.m = m_grain
            frag_child.m_init = m_grain

            frag_child.active = True
            frag_child.main = False
            frag_child.disruption_enabled = False

            # Set the erosion coefficient value (disable in grans, only larger fragments)
            if keep_eroding:
                frag_child.erosion_enabled = True

                # If the disruption occured, use a different erosion coefficient for daguhter fragments
                if disruption:
                    frag_child.erosion_coeff = const.disruption_erosion_coeff
                else:
                    frag_child.erosion_coeff = getErosionCoeff(const, frag_parent.h)

            else:
                # Compute the grain density and shape-density coeff
                frag_child.rho = const.rho_grain
                frag_child.updateShapeDensityCoeff()

                frag_child.erosion_enabled = False
                frag_child.erosion_coeff = 0


            # Give every fragment a unique ID
            frag_child.id = const.total_fragments
            const.total_fragments += 1

            frag_children.append(frag_child)


    return frag_children, const



def getErosionCoeff(const, h):
    """ Return the erosion coeff for the given height. """

    # Return the changed erosion coefficient
    if const.erosion_height_change >= h:
        return const.erosion_coeff_change

    # Return the starting erosion coeff
    elif const.erosion_height_start >= h:
        return const.erosion_coeff

    # If the height is above the erosion start height, return 0
    else:
        return 0



def ablateAll(fragments, const, compute_wake=False):
    """ Perform single body ablation of all fragments using the 4th order Runge-Kutta method. 

    Arguments:
        fragments: [list] A list of Fragment instances.
        const: [object] Constants instance.

    Keyword arguments:
        compute_wake: [bool] If True, the wake profile will be computed. False by default.

    Return:
        ...
    """


    # Keep track of the total luminosity
    luminosity_total = 0.0

    # Keep track of height of the brightest fragment
    brightest_height = 0.0
    brightest_length = 0.0
    brightest_lum    = 0.0
    brightest_vel    = 0.0


    # Track total mass
    mass_total = sum([frag.m for frag in fragments])

    frag_children_all = []

    # Go through all active fragments
    for frag in fragments:

        # Skip the fragment if it's not active
        if not frag.active:
            continue


        # Get atmosphere density for the given height
        rho_atm = atmDensityPoly(frag.h, const.dens_co)


        # Compute the mass loss of the fragment due to ablation
        mass_loss_ablation = massLossRK4(const.dt, frag.K, frag.sigma, frag.m, rho_atm, frag.v)


        # Compute the mass loss due to erosion
        if frag.erosion_enabled and (frag.erosion_coeff > 0):
            mass_loss_erosion = massLossRK4(const.dt, frag.K, frag.erosion_coeff, frag.m, rho_atm, frag.v)
        else:
            mass_loss_erosion = 0


        # Compute the total mass loss
        mass_loss_total = mass_loss_ablation + mass_loss_erosion


        # If the total mass in below zero, ablate what's left
        if (frag.m + mass_loss_total ) < 0:
            mass_loss_total = mass_loss_total + frag.m


        # Compute new mass
        m_new = frag.m + mass_loss_total


        # Compute change in velocity
        deceleration_total = decelerationRK4(const.dt, frag.K, frag.m, rho_atm, frag.v)


        # ### Add velocity change due to Earth's gravity ###

        # # Compute g at given height
        # gv = G0/((1 + frag.h/R_EARTH)**2)

        # # Vertical component of a
        # av = -gv - deceleration_total*frag.vv/frag.v + frag.vh*frag.v/(R_EARTH + frag.h)

        # # Horizontal component of a
        # ah = -deceleration_total*frag.vh/frag.v - frag.vv*frag.v/(R_EARTH + frag.h)

        # ### ###


        ### Compute deceleration wihout effect of gravity (to reconstruct the initial velocity without the 
        #   gravity component)

        # Vertical component of a
        av = -deceleration_total*frag.vv/frag.v + frag.vh*frag.v/(R_EARTH + frag.h)

        # Horizontal component of a
        ah = -deceleration_total*frag.vh/frag.v - frag.vv*frag.v/(R_EARTH + frag.h)

        ###


        # Update the velocity
        frag.vv -= av*const.dt
        frag.vh -= ah*const.dt
        frag.v = math.sqrt(frag.vh**2 + frag.vv**2)


        # Update fragment parameters
        frag.m = m_new
        frag.h = frag.h + frag.vv*const.dt

        # # Compute ablated luminosity (including the deceleration term) for one fragment/grain
        # lum = -luminousEfficiency(const.lum_eff_type, const.lum_eff, frag.v, frag.m) \
        #     *((mass_loss_ablation/const.dt*frag.v**2)/2 - frag.m*frag.v*deceleration_total)

        # Compute luminosity without the deceleration term
        lum = -luminousEfficiency(const.lum_eff_type, const.lum_eff, frag.v, frag.m) \
            *((mass_loss_ablation/const.dt*frag.v**2)/2)

        # Compute the total luminosity
        frag.lum = lum*frag.n_grains


        # Keep track of the total luminosity across all fragments
        luminosity_total += frag.lum

        # Update length along the track
        frag.length += frag.v*const.dt


        # Track total mass loss
        mass_total += mass_loss_total


        # Keep track of the brightest fragment
        if frag.lum > brightest_lum:
            brightest_lum = lum
            brightest_height = frag.h
            brightest_length = frag.length
            brightest_vel = frag.v


        # Compute aerodynamic loading on the grain
        dyn_press = frag.gamma*rho_atm*frag.v**2

        # if frag.id == 0:
        #     print('----- id:', frag.id)
        #     print('t:', const.total_time)
        #     print('V:', frag.v/1000)
        #     print('H:', frag.h/1000)
        #     print('m:', frag.m)
        #     print('DynPress:', dyn_press/1000, 'kPa')



        # If the fragment is done, stop ablating
        if (frag.m <= const.m_kill) or (frag.v < const.v_kill) or (frag.h < const.h_kill) or (frag.lum < 0):
            frag.active = False
            const.n_active -= 1
            #print('Killing', frag.id)
            continue

        
        # # Change the erosion coefficient of the fragment below the given height
        # if (frag.erosion_coeff > 0):
        #     frag.erosion_coeff = getErosionCoeff(const, frag.h)



        # For non-complex fragmentation only: Check if the erosion should start, given the height, 
        #   and create grains
        if (not frag.complex) and (frag.h < const.erosion_height_start) and frag.erosion_enabled \
            and const.erosion_on:

            # Turn on the erosion of the fragment
            frag.erosion_coeff = getErosionCoeff(const, frag.h)


            # Update the main fragment physical parameters if it is changed after erosion coefficient change
            if frag.main and (const.erosion_height_change >= frag.h):

                # Update the density
                frag.rho = const.erosion_rho_change
                frag.updateShapeDensityCoeff()

                # Update the ablation coeff
                frag.sigma = const.erosion_sigma_change


        # Create grains for erosion-enabled fragments
        if frag.erosion_enabled:

            # Generate new grains if there is some mass to distribute
            if abs(mass_loss_erosion) > 0:

                grain_children, const = generateFragments(const, frag, abs(mass_loss_erosion), \
                    frag.erosion_mass_index, frag.erosion_mass_min, frag.erosion_mass_max, \
                    keep_eroding=False)

                const.n_active += len(grain_children)
                frag_children_all += grain_children

                # print('Eroding id', frag.id)
                # print('Eroded mass: {:e}'.format(abs(mass_loss_erosion)))
                # print('Mass distribution:')
                # grain_mass_sum = 0
                # for f in frag_children:
                #     print('    {:d}: {:e} kg'.format(f.n_grains, f.m))
                #     grain_mass_sum += f.n_grains*f.m
                # print('Grain total mass: {:e}'.format(grain_mass_sum))



        # Disrupt the fragment if the dynamic pressure exceeds its strength
        if frag.disruption_enabled and const.disruption_on:
            if dyn_press > const.compressive_strength:

                # Compute the mass that should be disrupted into fragments
                mass_frag_disruption = frag.m*(1 - const.disruption_mass_grain_ratio)


                fragments_total_mass = 0
                if mass_frag_disruption > 0:

                    # Disrupt the meteoroid into fragments
                    disruption_mass_min = const.disruption_mass_min_ratio*mass_frag_disruption
                    disruption_mass_max = const.disruption_mass_max_ratio*mass_frag_disruption
    
                    # Generate larger fragments, possibly assign them a separate erosion coefficient
                    frag_children, const = generateFragments(const, frag, mass_frag_disruption, \
                        const.disruption_mass_index, disruption_mass_min, disruption_mass_max, \
                        keep_eroding=const.erosion_on, disruption=True)


                    frag_children_all += frag_children
                    const.n_active += len(frag_children)


                    # Compute the mass that went into fragments
                    fragments_total_mass = sum([f.n_grains*f.m for f in frag_children])

                    # Assign the height of disruption
                    const.disruption_height = frag.h


                    print('Disrupting id', frag.id)
                    print('Height: {:.3f} km'.format(const.disruption_height/1000))
                    print('Disrupted mass: {:e}'.format(mass_frag_disruption))
                    print('Mass distribution:')
                    for f in frag_children:
                        print('{:4d}: {:e} kg'.format(f.n_grains, f.m))
                    print('Disrupted total mass: {:e}'.format(fragments_total_mass))


                # Disrupt a portion of the leftover mass into grains
                mass_grain_disruption = frag.m - fragments_total_mass
                if mass_grain_disruption > 0:
                    grain_children, const = generateFragments(const, frag, mass_grain_disruption, 
                        frag.erosion_mass_index, frag.erosion_mass_min, frag.erosion_mass_max, \
                        keep_eroding=False)

                    frag_children_all += grain_children
                    const.n_active += len(grain_children)


                # Deactive the disrupted fragment
                frag.active = False
                frag.m = 0
                const.n_active -= 1


        # If the fragment is done, stop ablating
        if (frag.m <= const.m_kill):
            frag.active = False
            const.n_active -= 1
            #print('Killing', frag.id)
            continue



        # Handle complex fragmentation and status changes of the main fragment
        if frag.main:

            # Get a list of complex fragmentations that are still to do
            frags_to_do = [frag_entry for frag_entry in const.fragmentation_entries if not frag_entry.done]

            if len(frags_to_do):

                # Go through all fragmentations that needs to be performed
                for frag_entry in frags_to_do:

                    # Check if the height of the main fragment is right to perform the operation
                    if frag.h < frag_entry.height:


                        # Change parameters of all fragments
                        if frag_entry.frag_type == "A":

                            for frag_tmp in (fragments + frag_children_all + [frag]):

                                # Update the ablation coefficient
                                if frag_entry.sigma is not None:
                                    frag_tmp.sigma = frag_entry.sigma

                                # Update the drag coefficient
                                if frag_entry.gamma is not None:
                                    frag_tmp.gamma = frag_entry.gamma
                                    frag_tmp.updateShapeDensityCoeff()



                        # Change the parameters of the main fragment
                        if frag_entry.frag_type == "M":

                            if frag_entry.sigma is not None:
                                frag.sigma = frag_entry.sigma

                            if frag_entry.erosion_coeff is not None:
                                frag.erosion_coeff = frag_entry.erosion_coeff

                            if frag_entry.mass_index is not None:
                                frag.erosion_mass_index = frag_entry.mass_index

                            if frag_entry.grain_mass_min is not None:
                                frag.erosion_mass_min = frag_entry.grain_mass_min

                            if frag_entry.grain_mass_max is not None:
                                frag.erosion_mass_max = frag_entry.grain_mass_max


                        # Create a new single-body or eroding fragment
                        if (frag_entry.frag_type == "F") or (frag_entry.frag_type == "EF"):

                            parent_initial_mass = frag.m

                            # Go through all new fragments
                            for frag_num in range(frag_entry.number):

                                # Mass of the new fragment
                                new_frag_mass = parent_initial_mass*(frag_entry.mass_percent/100.0)/frag_entry.number
                                frag_entry.mass = new_frag_mass*frag_entry.number

                                # Decrease the parent mass
                                frag.m -= new_frag_mass

                                # Create the new fragment
                                frag_new = copy.deepcopy(frag)

                                # Indicate that the fragments are born out of complex fragmentation
                                frag_new.complex = True

                                # Assing the mass to the new fragment
                                frag_new.m = new_frag_mass

                                # Assign possible new ablation coeff to this fragment
                                if frag_entry.sigma is not None:
                                    frag_new.sigma = frag_entry.sigma


                                # If the fragment is eroding, set erosion parameters
                                if frag_entry.frag_type == "EF":
                                    frag_new.erosion_enabled = True

                                    frag_new.erosion_coeff = frag_entry.erosion_coeff

                                    frag_new.erosion_mass_index = frag_entry.mass_index
                                    frag_new.erosion_mass_min = frag_entry.grain_mass_min
                                    frag_new.erosion_mass_max = frag_entry.grain_mass_max

                                    
                                else:
                                    # Disable erosion for single-body fragments
                                    frag_new.erosion_enabled = False


                                # Add the new fragment to the list of childern
                                frag_children_all.append(frag_new)
                                const.n_active += 1


                        # Release dust
                        if frag_entry.frag_type == "D":

                            # Compute the mass of the dust
                            dust_mass = frag.m*(frag_entry.mass_percent/100.0)
                            frag_entry.mass = dust_mass

                            # Subtract from the parent mass
                            frag.m -= dust_mass

                            # Create the new fragment
                            frag_new = copy.deepcopy(frag)

                            # Indicate that the fragments are born out of complex fragmentation
                            frag_new.complex = True

                            # Generate dust grains
                            grain_children, const = generateFragments(const, frag_new, dust_mass, \
                                frag_entry.mass_index, frag_entry.grain_mass_min, frag_entry.grain_mass_max, \
                                keep_eroding=False)

                            # Add fragments to the list
                            frag_children_all += grain_children
                            const.n_active += len(grain_children)


                        # Set the fragmentation as finished
                        frag_entry.done = True

                        # Set physical conditions at the moment of fragmentation
                        frag_entry.time = const.total_time
                        frag_entry.dyn_pressure = dyn_press
                        frag_entry.velocity = frag.v
                        frag_entry.parent_mass = frag.m






    # Track the leading fragment length
    active_fragments = [frag for frag in fragments if frag.active]
    if len(active_fragments):
        leading_frag = max(active_fragments, key=lambda x: x.length)
        leading_frag_length = leading_frag.length
        leading_frag_height = leading_frag.h
    else:
        leading_frag_length = None
        leading_frag_height = None


    ### Compute the wake profile ###

    if compute_wake and (leading_frag_length is not None):

        # Evaluate the Gaussian from +3 sigma in front of the leading fragment to behind
        front_len = leading_frag_length + 3*const.wake_psf
        back_len = leading_frag_length - const.wake_extension

        ### Compute the wake as convoluted luminosities with the PSF ###

        length_array = np.linspace(back_len, front_len, 500) - leading_frag_length

        frag_list = []
        
        for frag in fragments:

            # Take only those lengths inside the wake window
            if frag.active:
                if (frag.length > back_len) and (frag.length < front_len):
                    frag_list.append(copy.deepcopy(frag))


        # Store evaluated wake
        wake = Wake(const, frag_list, leading_frag_length, length_array)

        ### ###

    else:
        wake = None


    ### ###


    # Add generated fragment children to the list of fragments
    fragments += frag_children_all

    # Increment the running time
    const.total_time += const.dt


    return fragments, const, luminosity_total, brightest_height, brightest_length, brightest_vel, \
        leading_frag_height, leading_frag_length, mass_total, wake




def runSimulation(const, compute_wake=False):
    """ Run the ablation simulation. """

    ###

    fragments = []

    # Init the main fragment
    frag = Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, const.zenith_angle, \
        const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)
    frag.main = True
    
    # Erode the main fragment
    frag.erosion_enabled = True

    # Disrupt the main fragment
    frag.disruption_enabled = True

    fragments.append(frag)



    # Reset simulation parameters
    const.total_time = 0
    const.n_active = 1
    const.total_fragments = 1


    ###



    # Check that the grain density is larger than the bulk density, and if not, set the grain density
    #   to be the same as the bulk density
    if const.rho > const.rho_grain:
        const.rho_grain = const.rho

    # Run the simulation until all fragments stop ablating
    results_list = []
    wake_results = []
    while const.n_active > 0:

        # Ablate the fragments
        fragments, const, luminosity_total, brightest_height, brightest_length, brightest_vel, \
            leading_frag_height, leading_frag_length, mass_total, wake = ablateAll(fragments, const, \
                compute_wake=compute_wake)

        # Store wake estimation results
        wake_results.append(wake)

        # Stack results list
        results_list.append([const.total_time, luminosity_total, brightest_height, brightest_length, \
            brightest_vel, leading_frag_height, leading_frag_length, mass_total])



    return results_list, wake_results



if __name__ == "__main__":

    import matplotlib.pyplot as plt



    # Show wake
    show_wake = False


    # Init the constants
    const = Constants()


    # Run the ablation simulation
    results_list, wake_results = runSimulation(const, compute_wake=show_wake)



    ### ANALYZE RESULTS ###


    # System limiting magnitude
    lim_mag = 6.0

    # Unpack the results
    results_list = np.array(results_list).astype(np.float64)
    time_arr, luminosity_arr, brightest_height_arr, brightest_length_arr, brightest_vel_arr, \
        leading_frag_height_arr, leading_frag_length_arr, mass_total_arr = results_list.T


    # Calculate absolute magnitude (apparent @100km) from given luminous intensity
    abs_magnitude = -2.5*np.log10(luminosity_arr/const.P_0m)

    # plt.plot(abs_magnitude, brightest_height_arr/1000)
    # plt.gca().invert_xaxis()
    # plt.show()

    plt.plot(time_arr, abs_magnitude)
    plt.gca().invert_yaxis()

    plt.xlabel("Time (s)")
    plt.ylabel("Absolulte magnitude")

    plt.show()



    # Plot mass loss
    plt.plot(time_arr, 1000*mass_total_arr)
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (g)")
    plt.show()



    # Plot length vs time
    plt.plot(brightest_length_arr[:-1]/1000, brightest_height_arr[:-1]/1000, label='Brightest bin')
    plt.plot(leading_frag_length_arr[:-1]/1000, leading_frag_height_arr[:-1]/1000, label='Leading fragment')

    plt.ylabel("Height (km)")
    plt.xlabel("Length (km)")

    plt.legend()


    plt.show()


    # Plot the wake animation
    if show_wake and wake_results:
        
        plt.ion()
        fig, ax = plt.subplots(1,1)

        # Determine the plot upper limit
        max_lum_wake = max([max(wake.wake_luminosity_profile) for wake in wake_results if wake is not None])

        

        for wake, abs_mag in zip(wake_results, abs_magnitude):

            if wake is None:
                continue

            # Skip points below the limiting magnitude
            if (abs_mag > lim_mag) or np.isnan(abs_mag):
                continue

            plt.cla()
                
            # Plot the wake profile
            ax.plot(wake.length_array, wake.wake_luminosity_profile)

            # Plot the location of grains
            ax.scatter(wake.length_points, wake.luminosity_points/10, c='k', s=10*wake.luminosity_points/np.max(wake.luminosity_points))

            plt.ylim([0, max_lum_wake])

            plt.pause(2*const.dt)

            fig.canvas.draw()

        plt.ioff()
        plt.clf()
        plt.close()