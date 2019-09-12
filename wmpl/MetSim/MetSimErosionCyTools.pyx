""" Fast cython functions for the MetSimErosion module. """


import cython
cimport cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, M_PI, M_PI_2, atan2


# Define cython types for numpy arrays
FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t



@cython.cdivision(True) 
cdef float massLoss(float K, float sigma, float m, float rho_atm, float v):
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



@cython.cdivision(True) 
cpdef float massLossRK4(float dt, float K, float sigma, float m, float rho_atm, float v):
    """ Computes the mass loss using the 4th order Runge-Kutta method. 
    
    Arguments:
        frag: [object] Fragment instance.
        cont: [object] Constants instance.
        rho_atm: [float] Atmosphere density (kg/m^3).
        sigma: [float] Ablation coefficient (s^2/m^2).

    Return:
        dm/dt: [float] Mass loss in kg/s.
    """

    cdef float mk1, mk2, mk3, mk4

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
cdef float deceleration(float K, float m, float rho_atm, float v):
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




@cython.cdivision(True) 
cpdef float decelerationRK4(float dt, float K, float m, float rho_atm, float v):
    """ Computes the deceleration using the 4th order Runge-Kutta method. """

    cdef float vk1, vk2, vk3, vk4

    # Compute change in velocity
    vk1 = dt*deceleration(K, m, rho_atm, v)
    vk2 = dt*deceleration(K, m, rho_atm, v + vk1/2.0)
    vk3 = dt*deceleration(K, m, rho_atm, v + vk2/2.0)
    vk4 = dt*deceleration(K, m, rho_atm, v + vk3)
    
    return (vk1/6.0 + vk2/3.0 + vk3/3.0 + vk4/6.0)/dt



@cython.cdivision(True) 
cpdef float luminousEfficiency(float vel):
    """ Compute the luminous efficienty in percent for the given velocity. 
    
    Arguments:
        vel: [float] Velocity (m/s).

    Return:
        tau: [float] Luminous efficiency (ratio).

    """

    return 0.7/100



@cython.cdivision(True) 
cpdef atmDensity(float h, np.ndarray[FLOAT_TYPE_t, ndim=1] dens_co):
    """ Calculates the atmospheric density in kg/m^3. 
    
    Arguments:
        h: [float] Height in meters.
        dens_co: [ndarray] Array of 6th order poly coeffs.

    Return:
        [float] Atmosphere density at height h (kg/m^3)

    """

    # # If the atmosphere dentiy interpolation is present, use it as the source of atm. density
    # if const.atm_density_interp is not None:
    #     return const.atm_density_interp(h)

    # # Otherwise, use the polynomial fit (WARNING: the fit is not as good as the interpolation!!!)
    # else:

    return (10**(dens_co[0] + dens_co[1]*h/1000.0 + dens_co[2]*(h/1000)**2 + dens_co[3]*(h/1000)**3 \
        + dens_co[4]*(h/1000)**4 + dens_co[5]*(h/1000)**5))*1000