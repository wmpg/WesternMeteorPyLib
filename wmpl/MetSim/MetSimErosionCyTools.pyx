""" Fast cython functions for the MetSimErosion module. """


import cython
cimport cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, M_PI, M_PI_2, atan2, tanh, log, exp, log10


# Define cython types for numpy arrays
FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t


### ATTEMPTED TO CONVERT THESE CLASSES TO CYTHON, BUT THERE ARE STILL ERRORS AND THE EXECUTION IS NOT ANY 
### FASTER

# @cython.auto_pickle(True)
# cdef class ConstantsCy:
#     cdef double dt, total_time, m_kill, v_kill, h_kill, h_init, P_0m
#     cdef double wake_psf, wake_extension
#     cdef double rho, m_init, v_init, shape_factor, sigma, zenith_angle, gamma, rho_grain
#     cdef double erosion_height_start, erosion_coeff, erosion_height_change, erosion_coeff_change, \
#         erosion_mass_index, erosion_mass_min, erosion_mass_max
#     cdef double compressive_strength, disruption_height, disruption_erosion_coeff, disruption_mass_index, \
#         disruption_mass_min_ratio, disruption_mass_max_ratio, disruption_mass_grain_ratio
#     cdef int n_active, total_fragments
#     cdef FLOAT_TYPE_t[:] dens_co
#     cdef bint erosion_on, disruption_on
    

#     def __init__(self):
#         """ Constant parameters for the ablation modelling. """

#         ### Simulation parameters ###

#         # Time step
#         self.dt = 0.005

#         # Time elapsed since the beginning
#         self.total_time = 0

#         # Number of active fragments
#         self.n_active = 0

#         # Minimum possible mass for ablation (kg)
#         self.m_kill = 1e-14

#         # Minimum ablation velocity (m/s)
#         self.v_kill = 3000

#         # Minimum height (m)
#         self.h_kill = 60000

#         # Initial meteoroid height (m)
#         self.h_init = 180000

#         # Power of a 0 magnitude meteor
#         self.P_0m = 840

#         # Atmosphere density coefficients
#         self.dens_co = np.array([-9.02726494,
#                         0.108986696,
#                         -0.0005189,
#                         -2.0646e-5,
#                         1.93881e-7,
#                         -4.7231e-10])


#         self.total_fragments = 0

#         ### ###


#         ### Wake parameters ###

#         # PSF stddev (m)
#         self.wake_psf = 3.0

#         # Wake extension from the leading fragment (m)
#         self.wake_extension = 200

#         ### ###



#         ### Main meteoroid properties ###

#         # Meteoroid bulk density (kg/m^3)
#         self.rho = 1000

#         # Initial meteoroid mass (kg)
#         self.m_init = 2e-5

#         # Initial meteoroid veocity (m/s)
#         self.v_init = 23570

#         # Shape factor (1.21 is sphere)
#         self.shape_factor = 1.21

#         # Main fragment ablation coefficient
#         self.sigma = 0.023/1e6

#         # Zenith angle (radians)
#         self.zenith_angle = np.radians(45)

#         # Drag coefficient
#         self.gamma = 1.0

#         # Grain bulk density (kg/m^3)
#         self.rho_grain = 3000

#         ### ###


#         ### Erosion properties ###

#         # Toggle erosion on/off
#         self.erosion_on = True

        
#         # Height at which the erosion starts (meters)
#         self.erosion_height_start = 102000

#         # Erosion coefficient (s^2/m^2)
#         self.erosion_coeff = 0.33/1e6

        
#         # Height at which the erosion coefficient changes (meters)
#         self.erosion_height_change = 90000

#         # Erosion coefficient after the change (s^2/m^2)
#         self.erosion_coeff_change = 0.33/1e6


#         # Grain mass distribution index
#         self.erosion_mass_index = 2.5

#         # Mass range for grains (kg)
#         self.erosion_mass_min = 1.0e-11
#         self.erosion_mass_max = 5.0e-10

#         ###


#         ### Disruption properties ###

#         # Toggle disruption on/off
#         self.disruption_on = True

#         # Meteoroid compressive strength (Pa)
#         self.compressive_strength = 2000

#         # Height of disruption (will be assigned when the disruption occures)
#         self.disruption_height = -1

#         # Erosion coefficient to use after disruption
#         self.disruption_erosion_coeff = self.erosion_coeff

#         # Disruption mass distribution index
#         self.disruption_mass_index = 2.0


#         # Mass ratio for disrupted fragments as the ratio of the disrupted mass
#         self.disruption_mass_min_ratio = 1.0/100
#         self.disruption_mass_max_ratio = 10.0/100

#         # Ratio of mass that will disrupt into grains
#         self.disruption_mass_grain_ratio = 0.25

#         ### ###


#     ### These two functions enable the object to be pickled
#     ### More info: https://stackoverflow.com/questions/43760919/save-cython-extension-by-pickle
#     ### Be careful when returning numpy arrays, .base needs to be added (e.g. see self.dens_co)

#     def __getstate__(self):
#         return self.dt, self.total_time, self.n_active, self.m_kill, self.v_kill, self.h_kill, self.h_init, \
#         self.P_0m, self.dens_co.base, self.total_fragments, self.wake_psf, self.wake_extension, self.rho, \
#         self.m_init, self.v_init, self.shape_factor, self.sigma, self.zenith_angle, self.gamma, \
#         self.rho_grain, self.erosion_on, self.erosion_height_start, self.erosion_coeff, \
#         self.erosion_height_change, self.erosion_coeff_change, self.erosion_mass_index, \
#         self.erosion_mass_min, self.erosion_mass_max, self.disruption_on, self.compressive_strength, \
#         self.disruption_height, self.disruption_erosion_coeff, self.disruption_mass_index, \
#         self.disruption_mass_min_ratio, self.disruption_mass_max_ratio, self.disruption_mass_grain_ratio


#     def __setstate__(self, x):
#         self.dt, self.total_time, self.n_active, self.m_kill, self.v_kill, self.h_kill, self.h_init, \
#         self.P_0m, self.dens_co, self.total_fragments, self.wake_psf, self.wake_extension, self.rho, \
#         self.m_init, self.v_init, self.shape_factor, self.sigma, self.zenith_angle, self.gamma, \
#         self.rho_grain, self.erosion_on, self.erosion_height_start, self.erosion_coeff, \
#         self.erosion_height_change, self.erosion_coeff_change, self.erosion_mass_index, \
#         self.erosion_mass_min, self.erosion_mass_max, self.disruption_on, self.compressive_strength, \
#         self.disruption_height, self.disruption_erosion_coeff, self.disruption_mass_index, \
#         self.disruption_mass_min_ratio, self.disruption_mass_max_ratio, self.disruption_mass_grain_ratio = x

#     ### ###


# # Create a class that Python can access and has a __dict__
# class Constants(ConstantsCy):
#     pass



# cdef class FragmentCy:
#     cdef public:
#         int id, n_grains
#         double K, m, rho, v, vv, vh, length, lum, erosion_coeff
#         bint erosion_enabled, disruption_enabled, active, main

#     def __init__(self):

#         self.id = 0

#         # Shape-density coeff
#         self.K = 0

#         # Mass (kg)
#         self.m = 0

#         # Density (kg/m^3)
#         self.rho = 0

#         # Velocity (m/s)
#         self.v = 0

#         # Velocity components (vertical and horizontal)
#         self.vv = 0
#         self.vh = 0

#         # Length along the trajectory
#         self.length = 0

#         # Luminous intensity (Watts)
#         self.lum = 0

#         # Erosion coefficient value
#         self.erosion_coeff = 0

#         self.erosion_enabled = False

#         self.disruption_enabled = False

#         self.active = False
#         self.n_grains = 1

#         # Indicate that this is the main fragment
#         self.main = False


#     cpdef void init(self, ConstantsCy const, double m, double rho, double v_init, double zenith_angle):


#         self.m = m
#         self.h = const.h_init
#         self.rho = rho
#         self.v = v_init
#         self.zenith_angle = zenith_angle

#         # Compute shape-density coeff
#         self.K = const.gamma*const.shape_factor*self.rho**(-2/3.0)

#         # Compute velocity components
#         self.vv = -v_init*np.cos(zenith_angle)
#         self.vh = v_init*np.sin(zenith_angle)

#         self.active = True
#         self.n_grains = 1



# # Create a class that Python can access and has a __dict__
# class Fragment(FragmentCy):
#     pass



@cython.cdivision(True) 
cdef double massLoss(double K, double sigma, double m, double rho_atm, double v):
    """ Mass loss differential equation, the result is giving dm/dt.

    Arguments:
        K: [double] Shape-density coefficient (m^2/kg^(2/3)).
        sigma: [double] Ablation coefficient (s^2/m^2).
        m: [double] Mass (kg).
        rho_atm: [double] Atmosphere density (kg/m^3).
        v: [double] Velocity (m/s).

    Return:
        dm/dt: [double] Mass loss in kg/s.
    """

    return -K*sigma*m**(2/3.0)*rho_atm*v**3



@cython.cdivision(True) 
cpdef double massLossRK4(double dt, double K, double sigma, double m, double rho_atm, double v):
    """ Computes the mass loss using the 4th order Runge-Kutta method. 
    
    Arguments:
        frag: [object] Fragment instance.
        cont: [object] Constants instance.
        rho_atm: [double] Atmosphere density (kg/m^3).
        sigma: [double] Ablation coefficient (s^2/m^2).

    Return:
        dm/dt: [double] Mass loss in kg/s.
    """

    cdef double mk1, mk2, mk3, mk4

    # Compute the mass loss (RK4)
    # Check instances when there is no more mass to ablate

    mk1 = dt*massLoss(K, sigma, m,            rho_atm, v)

    if -mk1/2 > m:
        mk1 = -m*2

    mk2 = dt*massLoss(K, sigma, m + mk1/2.0,  rho_atm, v)

    if -mk2/2 > m:
        mk2 = -m*2

    mk3 = dt*massLoss(K, sigma, m + mk2/2.0,  rho_atm, v)

    if -mk3 > m:
        mk3 = -m

    mk4 = dt*massLoss(K, sigma, m + mk3,      rho_atm, v)


    return mk1/6.0 + mk2/3.0 + mk3/3.0 + mk4/6.0



@cython.cdivision(True) 
cdef double deceleration(double K, double m, double rho_atm, double v):
    """ Computes the deceleration derivative.     

    Arguments:
        K: [double] Shape-density coefficient (m^2/kg^(2/3)).
        m: [double] Mass (kg).
        rho_atm: [double] Atmosphere density (kg/m^3).
        v: [double] Velocity (m/S).

    Return:
        dv/dt: [double] Deceleration.
    """

    return -K*m**(-1/3.0)*rho_atm*v**2




@cython.cdivision(True) 
cpdef double decelerationRK4(double dt, double K, double m, double rho_atm, double v):
    """ Computes the deceleration using the 4th order Runge-Kutta method. """

    cdef double vk1, vk2, vk3, vk4

    # Compute change in velocity
    vk1 = dt*deceleration(K, m, rho_atm, v)
    vk2 = dt*deceleration(K, m, rho_atm, v + vk1/2.0)
    vk3 = dt*deceleration(K, m, rho_atm, v + vk2/2.0)
    vk4 = dt*deceleration(K, m, rho_atm, v + vk3)
    
    return (vk1/6.0 + vk2/3.0 + vk3/3.0 + vk4/6.0)/dt



@cython.cdivision(True) 
cpdef double luminousEfficiency(int lum_eff_type, double lum_eff, double vel, double mass):
    """ Compute the luminous efficienty in percent for the given velocity. 
    
    Arguments:
        lum_eff_type: [int] Lum. eff. model: 
            0 - constant, 
            1 - Revelle & Ceplecha (2001): a) Type I, 
            2 - b) Type II, 
            3 - c) Type III, 
            4 - Borovicka et al. (2013) Kosice
            5 - CAMO faint meteors
            6 - Celpecha & McCrosky (1976)
            7 - Borovicka et al. (2020) - Two strengths
        lum_eff: [double] Value of the constant luminous efficiency (percent).
        vel: [double] Velocity (m/s).
        mass: [double] Mass (kg).

    Return:
        tau: [double] Luminous efficiency (ratio).

    """

    cdef double c1, c2

    # Constant luminous efficiency
    if lum_eff_type == 0:
        return lum_eff/100.0

    # Revelle & Ceplecha (2001)
    elif (lum_eff_type == 1) or (lum_eff_type == 2) or (lum_eff_type == 3):

        # Type I meteoroids
        if lum_eff_type == 1:
            c1 = 0.466
            c2 = -1.538

        # Type II meteoroids
        elif lum_eff_type == 2:
            c1 = -0.955
            c2 = -2.959

        # Type III meteoroids
        else:
            c1 = -2.670
            c2 = -4.674


        # Slow meteoroids
        if vel < 25372:
            return (exp(c1 - 10.307*log(vel/1000.0) + 9.781*log(vel/1000.0)**2 - 3.0414*log(vel/1000.0)**3 \
                + 0.3213*log(vel/1000.0)**4 + 1.15*tanh(0.38*log(mass))))/100.0

        # Fast meteoroids
        else:
            return (exp(c2 + log(vel/1000.0) + 1.15*tanh(0.38*log(mass))))/100.0

    # Borovicka et al. (2013) - Kosice
    elif lum_eff_type == 4:
        return (exp(-1.45 + log(vel/1000.0) + 0.35*tanh(0.38*log(mass))))/100.0

    # CAMO fast meteors - Subasinghe 2018 & Brown 2020
    elif lum_eff_type == 5:
        return exp(-12.59 + 5.58*log(vel/1000.0) - 0.17*log(vel/1000.0)**3 - 1.21*tanh(0.2*log(1e6*mass)))/100.0

    # Ceplecha & McCrosky (1976)
    elif lum_eff_type == 6:

        # Correction factor to convert from panchromatic magnitude to fraction definition
        correction_factor = 1500*10000000

        if vel <= 9300:
            return correction_factor*10**(-12.75)

        elif vel <= 12500:
            return correction_factor*10**(-15.60 + 2.92*log10(vel/1000.0))

        elif vel <= 17000:
            return correction_factor*10**(-13.24 + 0.77*log10(vel/1000.0))

        elif vel <= 27000:
            return correction_factor*10**(-12.50 + 0.17*log10(vel/1000.0))

        else:
            return correction_factor*10**(-13.69 + 1.00*log10(vel/1000.0))

    # Borovicka et al. (2020) - Two strengths
    elif lum_eff_type == 7:

        if vel < 25372:
            return exp(0.567 - 10.307*log(vel/1000.0) 
                             +  9.781*log(vel/1000.0)**2 
                             - 3.0414*log(vel/1000.0)**3 
                             + 0.3213*log(vel/1000.0)**4 
                             + 0.3470*tanh(0.38*log(mass))
                        )/100.0
        else:
            return exp(-1.4286 + log(vel/1000.0) + 0.347*tanh(0.38*log(mass)))/100.0




@cython.cdivision(True) 
cpdef atmDensityPoly(double ht, np.ndarray[FLOAT_TYPE_t, ndim=1] dens_co):
    """ Calculates the atmospheric density in kg/m^3. 
    
    Arguments:
        ht: [double] Height in meters.
        dens_co: [ndarray] Array of 7th order poly coeffs.

    Return:
        [double] Atmosphere density at height h (kg/m^3)

    """

    return 10**(dens_co[0] 
               + dens_co[1]*(ht/1e6) 
               + dens_co[2]*(ht/1e6)**2 
               + dens_co[3]*(ht/1e6)**3 
               + dens_co[4]*(ht/1e6)**4 
               + dens_co[5]*(ht/1e6)**5
               + dens_co[6]*(ht/1e6)**6
               )