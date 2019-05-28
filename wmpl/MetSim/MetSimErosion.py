
from __future__ import print_function, division, absolute_import

import sys
import math
import copy



### DEFINE CONSTANTS

# earth acceleration in m/s^2
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
        self.dt = 0.02

        # Time elapsed since the beginning
        self.total_time = 0

        # Number of active fragments
        self.n_active = 0

        # Minimum possible mass for ablation (kg)
        self.m_kill = 1e-14

        # Minimum ablation velocity (m/s)
        self.v_kill = 3000

        # Minimum height (m)
        self.h_kill = 40000


        # Atmosphere density coefficients
        self.dens_co = [-9.02726494,
                        0.108986696,
                        -0.0005189,
                        -2.0646e-5,
                        1.93881e-7,
                        -4.7231e-10]


        self.total_fragments = 0

        ### ###


        ### Main meteoroid properties ###

        # Meteoroid density (kg/m^3)
        self.rho = 120

        # Initial meteoroid mass (kg)
        self.m_init = 5.9e-5

        # Initial meteoroid veocity (m/s)
        self.v_init = 23570

        # Initial meteoroid height (m)
        self.h_init = 180000

        # Shape factor (1.21 is sphere)
        self.shape_factor = 1.21

        # Main fragment ablation coefficient
        self.sigma = 0.023/1e6

        # Zenith angle (radians)
        self.zenith_angle = math.radians(45)

        # Drag coefficient
        self.gamma = 1.0

        ### ###


        ### Erosion properties ###

        # Height at which the erosion starts (meters)
        self.erosion_height = 100000

        # Grain ablation coefficient (s^2/m^2)
        self.erosion_coeff = 0.33/1e6

        # Grain mass distribution index
        self.mass_index = 2.5

        # Grain density (kg/m^3)
        self.rho_grain = 3000

        # Define mass range for grains (kg)
        self.mass_min = 1.2e-10
        self.mass_max = 6e-10

        ###




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

        # Erosion coefficient value
        self.erosion_coeff = 0

        self.erosion_enabled = False

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




def massLoss(K, sigma, m, rho, v):
    return -K*sigma*m**(2/3.0)*rho*v**3


def massLossRK4(frag, const, rho_atm, sigma):

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


def deceleration(K, m, rho, v):
    return -K*m**(-1/3.0)*rho*v**2



def luminousEfficiency(vel):
    """ Compute the luminous efficienty in percent for the given velocity (m/s). """

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




def generateFragments(const, frag_parent, eroded_mass, mass_index, mass_min, mass_max):

    # Compute the number of mass bins
    k = math.ceil(abs(math.log10(mass_max/mass_min)/math.log10(MASS_BIN_COEFF)))

    # Compute the number of the largest grains
    if mass_index == 2:
        n0 = eroded_mass/(mass_max*(k + 1))
    else:
        n0 = abs((eroded_mass/mass_max)*(1 - MASS_BIN_COEFF**(2 - mass_index))/(1 - MASS_BIN_COEFF**((2 - mass_index)*(k + 1))))

    print('Largest grains:', n0, mass_max)

    # Go though every mass bin
    frag_children = []
    for i in range(0, k + 1):

        # Compute the mass of all grains in the bin (per grain)
        m_grain = mass_max*MASS_BIN_COEFF**i

        # Compute the number of grains in the bin
        n_grains_bin = int(n0*(mass_max/m_grain)**(mass_index - 1))


        # If there are any grains to erode, erode them
        if n_grains_bin > 0:


            # Init the new fragment with params of the parent
            frag_child = copy.deepcopy(frag_parent)

            # Assign the number of grains this fragment stands for
            frag_child.n_grains = n_grains_bin

            # Assign the grain mass
            frag_child.m = m_grain

            # Compute the grain density and shape-density coeff
            frag_child.rho = const.rho_grain
            frag_child.K = const.gamma*const.shape_factor*frag_child.rho**(-2/3.0)

            frag_child.erosion_enabled = False
            frag_child.active = True
            frag_child.main = False

            # Set the erosion coefficient value (disable in child fragments, only the parent erodes!)
            frag_child.erosion_coeff = 0

            # Give every fragment a unique ID
            frag_child.id = const.total_fragments
            const.total_fragments += 1

            frag_children.append(frag_child)


    return frag_children, const




def ablate(fragments, const):
    """ Perform single body ablation of the given grain using the 4th order Runge-Kutta method. """



    # Keep track of the total luminosity
    luminosity_total = 0.0

    # Keep track of height of the brightest fragment
    brightest_height = 0.0
    brightest_lum = 0.0

    # Track total mass
    mass_total = sum([frag.m for frag in fragments])

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


        ### Add velocity change due to Earth's gravity ###

        # Compute g at given height
        gv = G0/((1 + frag.h/R_EARTH)**2)

        # Vertical component of a
        av = -gv - deceleration_total*frag.vv/frag.v + frag.vh*frag.v/(R_EARTH + frag.h)

        # Horizontal component of a
        ah = -deceleration_total*frag.vh/frag.v - frag.vv*frag.v/(R_EARTH + frag.h)

        # # Deceleration magnitude
        # decel_mag = math.sqrt(av**2 + ah**2)

        # Update the velocity
        frag.vv -= av*const.dt
        frag.vh -= ah*const.dt
        frag.v = math.sqrt(frag.vh**2 + frag.vv**2)

        ### ###


        # Update fragment parameters
        frag.m = m_new
        frag.h = frag.h + frag.vv*const.dt

        # Compute ablated luminosity (including the deceleration term)
        lum = -luminousEfficiency(frag.v)*((mass_loss_ablation/const.dt*frag.v**2)/2 - frag.m*frag.v*deceleration_total)
        #lum = -luminousEfficiency(frag.v)*((mass_loss_ablation/const.dt*frag.v**2)/2)

        # Keep track of the brightest fragment
        if lum > brightest_lum:
            brightest_lum = lum
            brightest_height = frag.h

        # Keep track of the total luminosity across all fragments
        luminosity_total += lum*frag.n_grains

        # Update length along the track
        frag.length += frag.v*const.dt


        # Track total mass loss
        mass_total += mass_loss_total


        # Compute aerodynamic loading on the grain
        dyn_press = const.gamma*rho_atm*frag.v**2

        if frag.id == 0:
            print('----- id:', frag.id)
            print('V:', frag.v/1000)
            print('H:', frag.h/1000)
            print('m:', frag.m)
            print('DynPress:', dyn_press/1000, 'kPa')


        # ## TEST !!!!!!!!!!!!!!
        # sys.exit()


        # Check if the erosion should start, given the height and create grains
        if (frag.h < const.erosion_height) and frag.erosion_enabled:

            # Turn on the erosion of the fragment
            frag.erosion_coeff = const.erosion_coeff

            # Generate new fragments if there is some mass to distribute
            if abs(mass_loss_erosion) > 0:

                frag_children, const = generateFragments(const, frag, abs(mass_loss_erosion), 
                    const.mass_index, const.mass_min, const.mass_max)

                const.n_active += len(frag_children)

                fragments += frag_children

                # print('Eroding id', frag.id)
                # print('Eroded mass: {:e}'.format(abs(mass_loss_erosion)))
                # print('Mass distribution:')
                # grain_mass_sum = 0
                # for f in frag_children:
                #     print('    {:d}: {:e} kg'.format(f.n_grains, f.m))
                #     grain_mass_sum += f.n_grains*f.m
                # print('Grain total mass: {:e}'.format(grain_mass_sum))



        # If the fragment is done, stop ablating
        if (frag.m <= const.m_kill) or (frag.v < const.v_kill) or (frag.h < const.h_kill):
            frag.active = False
            const.n_active -= 1
            #print('Killing', frag.id)


            


    
    # Increment the running time
    const.total_time += const.dt


    return fragments, const, luminosity_total, brightest_height, mass_total




def runSimulation(const):

    ###

    fragments = []

    # Init the main fragment
    frag = Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.zenith_angle)
    frag.main = True
    
    # Erode the main fragment
    frag.erosion_enabled = True

    fragments.append(frag)



    # Reset simulation parameters
    const.total_time = 0
    const.n_active = 1
    const.total_fragments = 1


    ###



    # Run the simulation until all fragments stop ablating
    results_list = []
    while const.n_active > 0:
    #for i in range(5000):
        fragments, const, luminosity_total, brightest_height, mass_total = ablate(fragments, const)

        # Stack results list
        results_list.append([const.total_time, luminosity_total, brightest_height, mass_total])


    return results_list



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    # Init the constants
    const = Constants()


    results_list = runSimulation(const)


    results_list = np.array(results_list)

    time_arr, luminosity_arr, brightest_height_arr, mass_total_arr = results_list.T


    # Calculate absolute magnitude (apparent @100km) from given luminous intensity
    P_0m = 1500
    abs_magnitude = -2.5*np.log10(luminosity_arr/P_0m)

    # plt.plot(abs_magnitude, brightest_height_arr/1000)
    # plt.gca().invert_xaxis()
    # plt.show()

    plt.plot(time_arr, abs_magnitude)
    plt.gca().invert_yaxis()
    plt.show()


    # Plot mass loss

    plt.plot(time_arr, mass_total_arr)
    plt.show()