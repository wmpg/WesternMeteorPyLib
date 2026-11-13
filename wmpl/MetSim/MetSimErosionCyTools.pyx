""" Fast cython functions for the MetSimErosion module. """


import cython
cimport cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, M_PI, M_PI_2, atan2, tanh, log, exp, log10, cos, pow


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

#         # Height of disruption (will be assigned when the disruption occurs)
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
cpdef double luminousEfficiency(int lum_eff_type, double lum_eff, double vel, double mass, double v_init=-1.0):
    """ Compute the luminous efficiency of the given type, velocity, and mass.
    
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
            8 - Pecina & Ceplecha (1983)
        lum_eff: [double] Value of the constant luminous efficiency (percent).
        vel: [double] Velocity (m/s).
        mass: [double] Mass (kg).
        v_init: [double] Pre-atmospheric velocity (m/s), only used by the Revelle & Ceplecha (2001)
            models (types 1-3). If <= 0 (default -1), the deceleration correction term
            0.26*ln(dv) + 0.0042*ln(dv)^3, where dv = (v_init - vel) in km/s, is not applied.
            The correction diverges as dv -> 0, so dv is floored at 0.1 km/s. The floor also
            covers vel >= v_init (instead of producing a NaN).

    Return:
        tau: [double] Luminous efficiency (ratio).

    """

    cdef double c1, c2, lv, decel, dv

    # Velocity difference (km/s) at which the Revelle & Ceplecha (2001) deceleration correction is
    # floored, keeping it bounded as it otherwise diverges to -inf as dv -> 0
    cdef double dv_min = 0.1

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

        lv = log(vel/1000.0)

        decel = 0.0

        # Deceleration correction using the velocity difference from the pre-atmospheric velocity
        if v_init > 0:

            dv = (v_init - vel)/1000.0

            # Hold the correction at its dv_min value below dv_min, where it diverges as dv -> 0.
            # This also covers vel >= v_init (physically inconsistent input, avoid log of a
            # non-positive number)
            if dv < dv_min:
                dv = dv_min

            decel = 0.26*log(dv) + 0.0042*log(dv)**3

        # Slow meteoroids
        if vel < 25372:
            return exp(
                c1
                - 10.307*lv
                + 9.781*lv**2
                - 3.0414*lv**3
                + 0.3213*lv**4
                + 1.15*tanh(0.38*log(mass))
                + decel
            )/100.0

        # Fast meteoroids
        else:
            return exp(
                c2
                + lv
                + 1.15*tanh(0.38*log(mass))
                + decel
            )/100.0

    # Borovicka et al. (2013) - Kosice
    elif lum_eff_type == 4:
        return (exp(-1.45 + log(vel/1000.0) + 0.35*tanh(0.38*log(mass))))/100.0

    # CAMO fast meteors - Subasinghe 2018 & Brown 2020
    elif lum_eff_type == 5:
        return exp(-12.59 + 5.58*log(vel/1000.0) - 0.17*log(vel/1000.0)**3 - 1.21*tanh(0.2*log(1e6*mass)))/100.0

    # Ceplecha & McCrosky (1976)
    elif lum_eff_type == 6:

        # Correction factor to convert from panchromatic magnitude to fraction definition
        # Assume a P0M = 1500 W
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

    # Pecina & Ceplecha (1983)
    elif lum_eff_type == 8:
        
        # P0M = 1500 W (conversion from panchromatic magnitude) and the 10^7 correction factor
        # This converts the CGS-based tau into fractions
        correction_factor = 1500.0*10000000.0
        
        # Convert m/s to km/s for the formula
        v_km_s = vel/1000.0
        lv = log10(v_km_s)
        
        # Equation (40) for v < 25.372 km/s
        if v_km_s < 25.372:
            tau = 10**(-12.834 
                       - 10.307*lv 
                       + 22.522*lv**2 
                       - 16.125*lv**3 
                       + 3.922*lv**4)
        
        # Linear log-relation for v >= 25.372 km/s
        else:
            tau = 10**(lv - 13.70)
            
        return tau*correction_factor



@cython.cdivision(True) 
cpdef double ionizationEfficiency(double vel):
    """ Compute the ionization efficienty in percent for the given velocity. Jones (1997) function.
    
    Arguments:
        vel: [double] Velocity (m/s).

    Return:
        tau: [double] Dimensionless ionization efficiency (ratio, from 0 to 1).

    """

    # Scale velocity to km/s
    vel = vel/1000

    return 10**(5.84 - 0.09*vel**0.5 - 9.56/(log10(vel)))



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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double tableDensity(double ht, const double[::1] table_ht, const double[::1] table_log10_rho) nogil:
    """ Interpolate log10 of the density linearly in height in a table, extrapolating the end segments. """

    cdef Py_ssize_t n = table_ht.shape[0]
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = n - 1
    cdef Py_ssize_t mid

    # Find the last table point at or below the height, keeping the end segments outside the table
    if ht >= table_ht[n - 1]:
        lo = n - 2

    else:
        while hi - lo > 1:
            mid = (lo + hi)//2
            if table_ht[mid] <= ht:
                lo = mid
            else:
                hi = mid

    return 10**(table_log10_rho[lo] + (ht - table_ht[lo])*(table_log10_rho[lo + 1] - table_log10_rho[lo]) \
        /(table_ht[lo + 1] - table_ht[lo]))



def checkTableShape(table_ht, table_log10_rho):
    """ Check that a density table has at least 2 points and as many densities as heights.

    Arguments:
        table_ht: [ndarray] Table heights in meters.
        table_log10_rho: [ndarray] log10 of the densities (kg/m^3) at the table heights.

    Return:
        None
    """

    if (len(table_ht) < 2) or (len(table_ht) != len(table_log10_rho)):
        raise ValueError("The density table needs at least 2 points and as many densities as heights, got " \
            "{:d} heights and {:d} densities".format(len(table_ht), len(table_log10_rho)))



cdef class AtmDensityTableCy:
    """ Atmosphere density table for the simulation loop, which holds the table so that a call does not
        convert it, and which the Cython code can evaluate without the Python overhead.
    """

    cdef const double[::1] table_ht
    cdef const double[::1] table_log10_rho

    def __init__(self, table_ht, table_log10_rho):
        """ Store the table as contiguous arrays.

        Arguments:
            table_ht: [ndarray] Table heights in meters, ascending, at least 2 points.
            table_log10_rho: [ndarray] log10 of the densities (kg/m^3) at the table heights.
        """

        table_ht = np.ascontiguousarray(table_ht, dtype=np.float64)
        table_log10_rho = np.ascontiguousarray(table_log10_rho, dtype=np.float64)
        checkTableShape(table_ht, table_log10_rho)

        self.table_ht = table_ht
        self.table_log10_rho = table_log10_rho


    cpdef double density(self, double ht):
        """ Calculates the atmospheric density in kg/m^3 at the given height in meters. """

        return tableDensity(ht, self.table_ht, self.table_log10_rho)



cpdef double atmDensityTable(double ht, const double[::1] table_ht, const double[::1] table_log10_rho):
    """ Calculates the atmospheric density in kg/m^3 from a tabulated profile, interpolating log10 of the
        density linearly in height, which is an exponential atmosphere between the table points. Outside the
        table, the end segments are extrapolated. The simulation loop uses AtmDensityTableCy instead, which
        does not convert the table on every call.

    Arguments:
        ht: [double] Height in meters.
        table_ht: [ndarray] Table heights in meters, ascending, at least 2 points.
        table_log10_rho: [ndarray] log10 of the densities (kg/m^3) at the table heights.

    Return:
        [double] Atmosphere density at height h (kg/m^3)

    """

    checkTableShape(table_ht, table_log10_rho)

    return tableDensity(ht, table_ht, table_log10_rho)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double polyDensity(double ht, const double[::1] dens_co) nogil:
    """ Compute the atmosphere density (kg/m^3) at the given height (m) from the polynomial of
        atmDensityPoly(), in C.
    """

    cdef double x = ht/1e6
    cdef double log10_rho = dens_co[6]
    cdef int i

    for i in range(5, -1, -1):
        log10_rho = log10_rho*x + dens_co[i]

    return pow(10.0, log10_rho)



@cython.cdivision(True)
cdef double pathDensity(double length, double h_init, double cos_zc, double r_earth, double h_drop, \
    AtmDensityTableCy atm_table, const double[::1] dens_co):
    """ Compute the atmosphere density at the given length along the straight path, dropped by gravity.

    Arguments:
        length: [double] Length along the path from the beginning of the simulation (m).
        h_init: [double] Height of the beginning of the simulation (m).
        cos_zc: [double] Cosine of the zenith angle of the path at the beginning.
        r_earth: [double] Radius of the Earth (m).
        h_drop: [double] Drop of the height due to gravity (m).
        atm_table: [AtmDensityTableCy] Density table, or None to use the polynomial.
        dens_co: [ndarray] Coefficients of the density polynomial, used without the table.

    Return:
        [double] Atmosphere density (kg/m^3).
    """

    cdef double r0 = h_init + r_earth
    cdef double ht = sqrt(r0*r0 - 2*length*cos_zc*r0 + length*length) - r_earth - h_drop

    if atm_table is not None:
        return atm_table.density(ht)

    return polyDensity(ht, dens_co)



@cython.cdivision(True)
cpdef tuple ablationStepRK4(double dt, double K, double sigma, double m, double v, double length, \
    double h_init, double zenith_angle, double r_earth, double h_drop, AtmDensityTableCy atm_table, \
    const double[::1] dens_co):
    """ Integrate the mass, the speed and the length of a fragment over one time step together with the
        4th order Runge-Kutta method, taking the atmosphere density at the height of every stage.

    Arguments:
        dt: [double] Time step (s).
        K: [double] Shape-density coefficient (m^2/kg^(2/3)).
        sigma: [double] Sum of the ablation and the erosion coefficients (s^2/m^2).
        m: [double] Mass (kg).
        v: [double] Speed (m/s).
        length: [double] Length along the path from the beginning of the simulation (m).
        h_init: [double] Height of the beginning of the simulation (m).
        zenith_angle: [double] Zenith angle of the path at the beginning (rad).
        r_earth: [double] Radius of the Earth (m).
        h_drop: [double] Drop of the height due to gravity, kept over the step (m).
        atm_table: [AtmDensityTableCy] Density table, or None to use the polynomial.
        dens_co: [ndarray] Coefficients of the density polynomial, used without the table.

    Return:
        (dm, dv, dl): [tuple] Changes of the mass (kg), the speed (m/s) and the length (m) over the step.
    """

    cdef double cos_zc = cos(zenith_angle)
    cdef double mk[4]
    cdef double vk[4]
    cdef double lk[4]
    cdef double m_stage = m
    cdef double v_stage = v
    cdef double l_stage = 0.0
    cdef double rho_atm
    cdef double dm, dv, dl
    cdef int i

    for i in range(4):

        # Take the state of the stage from the previous one, halfway for the middle stages
        if i == 1 or i == 2:
            m_stage = m + dt*mk[i - 1]/2.0
            v_stage = v + dt*vk[i - 1]/2.0
            l_stage = dt*lk[i - 1]/2.0

        elif i == 3:
            m_stage = m + dt*mk[2]
            v_stage = v + dt*vk[2]
            l_stage = dt*lk[2]

        # Do not let a stage ablate more mass than there is, or the drag reverse the motion
        if m_stage < 0:
            m_stage = 0

        if v_stage < 0:
            v_stage = 0

        rho_atm = pathDensity(length + l_stage, h_init, cos_zc, r_earth, h_drop, atm_table, dens_co)

        if m_stage > 0:
            mk[i] = massLoss(K, sigma, m_stage, rho_atm, v_stage)
            vk[i] = deceleration(K, m_stage, rho_atm, v_stage)

        else:
            mk[i] = 0
            vk[i] = 0

        lk[i] = v_stage

    dm = dt*(mk[0] + 2*mk[1] + 2*mk[2] + mk[3])/6.0
    dv = dt*(vk[0] + 2*vk[1] + 2*vk[2] + vk[3])/6.0
    dl = dt*(lk[0] + 2*lk[1] + 2*lk[2] + lk[3])/6.0

    # Do not ablate more mass than there is
    if dm < -m:
        dm = -m

    # Stop a fragment which the drag would stop within the step, e.g. a small grain deep in the atmosphere,
    #   where the step is too long for the explicit integration
    if dv < -v:
        dv = -v

    return dm, dv, dl
