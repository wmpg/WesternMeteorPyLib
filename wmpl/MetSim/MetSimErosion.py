
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
        self.dens_co = [-9.02726494,
                        0.108986696,
                        -0.0005189,
                        -2.0646e-5,
                        1.93881e-7,
                        -4.7231e-10]


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

        # Main fragment ablation coefficient
        self.sigma = 0.023/1e6

        # Zenith angle (radians)
        self.zenith_angle = math.radians(45)

        # Drag coefficient
        self.gamma = 1.0

        # Grain bulk density (kg/m^3)
        self.rho_grain = 3000

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




class Fragment(object):
    def __init__(self):

        self.id = 0

        # Shape-density coeff
        self.K = 0

        # Mass (kg)
        self.m = 0

        # Density (kg/m^3)
        self.rho = 0

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

        self.erosion_enabled = False

        self.disruption_enabled = False

        self.active = False
        self.n_grains = 1

        # Indicate that this is the main fragment
        self.main = False


    def init(self, const, m, rho, v_init, zenith_angle):


        self.m = m
        self.h = const.h_init
        self.rho = rho
        self.v = v_init
        self.zenith_angle = zenith_angle

        # Compute shape-density coeff
        self.K = const.gamma*const.shape_factor*self.rho**(-2/3.0)

        # Compute velocity components
        self.vv = -v_init*math.cos(zenith_angle)
        self.vh = v_init*math.sin(zenith_angle)

        self.active = True
        self.n_grains = 1



class Wake(object):
    def __init__(self, length_array, wake_luminosity_profile, length_points, luminosity_points):
        """ Container for the evaluated wake. """

        self.length_array = length_array
        self.wake_luminosity_profile = wake_luminosity_profile
        self.length_points = np.array(length_points)
        self.luminosity_points = np.array(luminosity_points)




def massLoss(K, sigma, m, rho_atm, v):
    """ Mass loss differential equation, the result is giving dm/dt.

    Arguments:
        K: [float] Shape-density coefficient (m^2/kg^(2/3)).
        sigma: [float] Ablation coefficient (s^2/m^2).
        m: [float] Mass (kg).
        rho_atm: [float] Atmosphere density (kg/m^3).
        v: [float] Velocity (m/S).

    Return:
        dm/dt: [float] Mass loss in kg/s.
    """

    return -K*sigma*m**(2/3.0)*rho_atm*v**3



def massLossRK4(frag, const, rho_atm, sigma):
    """ Computes the mass loss using the 4th order Runge-Kutta method. 
    
    Arguments:
        frag: [object] Fragment instance.
        cont: [object] Constants instance.
        rho_atm: [float] Atmosphere density (kg/m^3).
        sigma: [float] Ablation coefficient (s^2/m^2).

    Return:
        dm/dt: [float] Mass loss in kg/s.
    """

    # Compute the mass loss (RK4)
    # Check instances when there is no more mass to ablate

    mk1 = const.dt*massLoss(frag.K, sigma, frag.m,            rho_atm, frag.v)

    if -mk1/2 > frag.m:
        mk1 = -frag.m*2

    mk2 = const.dt*massLoss(frag.K, sigma, frag.m + mk1/2.0,  rho_atm, frag.v)

    if -mk2/2 > frag.m:
        mk2 = -frag.m*2

    mk3 = const.dt*massLoss(frag.K, sigma, frag.m + mk2/2.0,  rho_atm, frag.v)

    if -mk3 > frag.m:
        mk3 = -frag.m

    mk4 = const.dt*massLoss(frag.K, sigma, frag.m + mk3,      rho_atm, frag.v)


    mass_loss_total = mk1/6.0 + mk2/3.0 + mk3/3.0 + mk4/6.0

    return mass_loss_total




def deceleration(K, m, rho_atm, v):
    """ Computes the deceleration derivative.     

    Arguments:
        K: [float] Shape-density coefficient (m^2/kg^(2/3)).
        m: [float] Mass (kg).
        rho_atm: [float] Atmosphere density (kg/m^3).
        v: [float] Velocity (m/S).

    Return:
        dv/dt: [float] Deceleration.
    """

    return -K*m**(-1/3.0)*rho_atm*v**2




def luminousEfficiency(vel):
    """ Compute the luminous efficienty in percent for the given velocity. 
    
    Arguments:
        vel: [float] Velocity (m/s).

    Return:
        tau: [float] Luminous efficiency (ratio).

    """

    return 0.7/100



def atmDensity(h, const):
    """ Calculates the atmospheric density in kg/m^3. 
    
    Arguments:
        h: [float] Height in meters.

    Return:
        [float] Atmosphere density at height h (kg/m^3)

    """

    # # If the atmosphere dentiy interpolation is present, use it as the source of atm. density
    # if const.atm_density_interp is not None:
    #     return const.atm_density_interp(h)

    # # Otherwise, use the polynomial fit (WARNING: the fit is not as good as the interpolation!!!)
    # else:

    dens_co = const.dens_co

    rho_a = (10**(dens_co[0] + dens_co[1]*h/1000.0 + dens_co[2]*(h/1000)**2 + dens_co[3]*(h/1000)**3 \
        + dens_co[4]*(h/1000)**4 + dens_co[5]*(h/1000)**5))*1000

    return rho_a




def generateFragments(const, frag_parent, eroded_mass, mass_index, mass_min, mass_max, keep_eroding=False,
    disruption=False):
    """ Given the parent fragment, fragment it into daughter fragments using a power law mass distribution.

    Masses are binned and one daughter fragment may represent several fragments/grains, which is specified 
    with the n_grains atribute.

    Arguments:
        const: [object] Constants instance.
        frag_parent: [object] Fragment instance, the tparent fragment.
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
                frag_child.K = const.gamma*const.shape_factor*frag_child.rho**(-2/3.0)

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
    brightest_lum = 0.0
    brightest_vel = 0.0

    # Track total mass
    mass_total = sum([frag.m for frag in fragments])

    frag_children_all = []

    # Go through all active fragments
    for frag in fragments:

        # Skip the fragment if it's not active
        if not frag.active:
            continue


        # Get atmosphere density for the given height
        rho_atm = atmDensity(frag.h, const)


        # Compute the mass loss of the main fragment due to ablation
        mass_loss_ablation = massLossRK4(frag, const, rho_atm, const.sigma)


        # Compute the mass loss due to erosion
        if frag.erosion_enabled and (frag.erosion_coeff > 0):
            mass_loss_erosion = massLossRK4(frag, const, rho_atm, frag.erosion_coeff)
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
        vk1 = const.dt*deceleration(frag.K, frag.m, rho_atm, frag.v)
        vk2 = const.dt*deceleration(frag.K, frag.m, rho_atm, frag.v + vk1/2.0)
        vk3 = const.dt*deceleration(frag.K, frag.m, rho_atm, frag.v + vk2/2.0)
        vk4 = const.dt*deceleration(frag.K, frag.m, rho_atm, frag.v + vk3)
        deceleration_total = (vk1/6.0 + vk2/3.0 + vk3/3.0 + vk4/6.0)/const.dt


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

        # Compute ablated luminosity (including the deceleration term) for one fragment/grain
        lum = -luminousEfficiency(frag.v)*((mass_loss_ablation/const.dt*frag.v**2)/2 - frag.m*frag.v*deceleration_total)
        #lum = -luminousEfficiency(frag.v)*((mass_loss_ablation/const.dt*frag.v**2)/2)

        # Compute the total luminosity
        frag.lum = lum*frag.n_grains


        # Keep track of the total luminosity across all fragments
        luminosity_total += frag.lum

        # Update length along the track
        frag.length += frag.v*const.dt


        # Track total mass loss
        mass_total += mass_loss_total


        # Keep track of the brightest fragment
        if lum > brightest_lum:
            brightest_lum = lum
            brightest_height = frag.h
            brightest_length = frag.length
            brightest_vel = frag.v


        # Compute aerodynamic loading on the grain
        dyn_press = const.gamma*rho_atm*frag.v**2

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



        # Check if the erosion should start, given the height, and create grains
        if (frag.h < const.erosion_height_start) and frag.erosion_enabled and const.erosion_on:

            # Turn on the erosion of the fragment
            frag.erosion_coeff = getErosionCoeff(const, frag.h)

            # Generate new fragments if there is some mass to distribute
            if abs(mass_loss_erosion) > 0:

                grain_children, const = generateFragments(const, frag, abs(mass_loss_erosion), \
                    const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max, \
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


                    print('Disrupting id', frag.id)
                    print('Disrupted mass: {:e}'.format(mass_frag_disruption))
                    print('Mass distribution:')
                    for f in frag_children:
                        print('{:4d}: {:e} kg'.format(f.n_grains, f.m))
                    print('Disrupted total mass: {:e}'.format(fragments_total_mass))


                # Disrupt a portion of the leftover mass into grains
                mass_grain_disruption = frag.m - fragments_total_mass
                if mass_grain_disruption > 0:
                    grain_children, const = generateFragments(const, frag, mass_grain_disruption, 
                        const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max, \
                        keep_eroding=False)

                    frag_children_all += grain_children
                    const.n_active += len(grain_children)


                # Deactive the disrupted fragment
                frag.active = False
                frag.m = 0
                const.n_active -= 1



    # Track the leading fragment length
    active_fragments_length = [frag.length for frag in fragments if frag.active]
    if len(active_fragments_length):
        leading_frag_length = max(active_fragments_length)
    else:
        leading_frag_length = None


    ### Compute the wake profile ###

    if compute_wake and (leading_frag_length is not None):

        # Evaluate the Gaussian from +3 sigma in front of the leading fragment to behind
        front_len = leading_frag_length + 3*const.wake_psf
        back_len = leading_frag_length - const.wake_extension

        ### Compute the wake as convoluted luminosities with the PSF ###

        length_array = np.linspace(back_len, front_len, 500) - leading_frag_length
        wake_luminosity_profile = np.zeros_like(length_array)

        luminosity_points = []
        length_points = []

        # Evalue the Gaussian of every fragment
        for frag in fragments:

            # Take only those lengths inside the wake window
            if (frag.length > back_len) and (frag.length < front_len):

                luminosity_points.append(frag.lum)
                length_points.append(frag.length - leading_frag_length)

                # Evalute the Gaussian
                wake_luminosity_profile += frag.lum*scipy.stats.norm.pdf(length_array, \
                    loc=frag.length - leading_frag_length, scale=const.wake_psf)


        # Store evaluated wake
        wake = Wake(length_array, wake_luminosity_profile, length_points, luminosity_points)

        ### ###

    else:
        wake = None


    ### ###


    # Add generated fragment children to the list of fragments
    fragments += frag_children_all

    # Increment the running time
    const.total_time += const.dt


    return fragments, const, luminosity_total, brightest_height, brightest_length, brightest_vel, \
        leading_frag_length, mass_total, wake




def runSimulation(const, compute_wake=False):
    """ Run the ablation simulation. """

    ###

    fragments = []

    # Init the main fragment
    frag = Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.zenith_angle)
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



    # Run the simulation until all fragments stop ablating
    results_list = []
    wake_results = []
    while const.n_active > 0:

        # Ablate the fragments
        fragments, const, luminosity_total, brightest_height, brightest_length, brightest_vel, \
            leading_frag_length, mass_total, wake = ablateAll(fragments, const, compute_wake=compute_wake)

        # Store wake estimation results
        wake_results.append(wake)

        # Stack results list
        results_list.append([const.total_time, luminosity_total, brightest_height, brightest_length, \
            brightest_vel, leading_frag_length, mass_total])



    return results_list, wake_results



if __name__ == "__main__":

    import matplotlib.pyplot as plt



    # Show wake
    show_wake = True


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
        leading_frag_length_arr, mass_total_arr = results_list.T


    # Calculate absolute magnitude (apparent @100km) from given luminous intensity
    abs_magnitude = -2.5*np.log10(luminosity_arr/const.P_0m)

    # plt.plot(abs_magnitude, brightest_height_arr/1000)
    # plt.gca().invert_xaxis()
    # plt.show()

    plt.plot(time_arr, abs_magnitude)
    plt.gca().invert_yaxis()
    plt.show()


    # Plot mass loss

    plt.plot(time_arr, mass_total_arr)
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