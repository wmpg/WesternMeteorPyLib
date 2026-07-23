""" Fast cython functions for the MetSimErosion module. """


import cython
cimport cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, M_PI, M_PI_2, atan2, tanh, log, exp, log10, cos, fabs, fmax


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
            The correction diverges as dv -> 0, so for 0 < dv < 0.1 km/s it is linearly tapered
            to zero, and for vel >= v_init it is set to zero (instead of producing a NaN).

    Return:
        tau: [double] Luminous efficiency (ratio).

    """

    cdef double c1, c2, lv, decel, dv

    # Velocity difference (km/s) below which the Revelle & Ceplecha (2001) deceleration correction is
    # linearly tapered to zero, keeping it bounded as it otherwise diverges to -inf as dv -> 0
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

            if dv >= dv_min:
                decel = 0.26*log(dv) + 0.0042*log(dv)**3

            # Taper the correction linearly to zero below dv_min (it diverges as dv -> 0), and
            # disable it for vel >= v_init (physically inconsistent input, avoid log of a
            # non-positive number)
            elif dv > 0:
                decel = (dv/dv_min)*(0.26*log(dv_min) + 0.0042*log(dv_min)**3)

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



### Adaptive per-fragment sub-stepping (used only when const.adaptive_dt is True) ###


cdef inline double clampMassC(double dm, double m):
    """ Reproduce the "ablate at most the whole mass" clamp used by the fixed engine: if the mass loss
        would drive the mass below zero, cap it at exactly -m so the new mass floors at 0.

    Arguments:
        dm: [double] Proposed mass change over the (sub-)step (kg, normally negative).
        m: [double] Current fragment mass (kg).

    Return:
        dm_clamped: [double] dm, or -m if (m + dm) < 0.
    """

    if (m + dm) < 0:
        return -m
    return dm


@cython.cdivision(True)
cdef double heightCurvatureC(double h0, double zc, double l, double r_earth):
    """ Cython/scalar twin of heightCurvature() (MetSimErosion.py): height at a distance l along the
        trajectory, accounting for the Earth's curvature.

    Arguments:
        h0: [double] Initial height (m).
        zc: [double] Zenith angle (radians).
        l: [double] Distance travelled along the trajectory from the origin (m).
        r_earth: [double] Earth radius (m).

    Return:
        h: [double] Height at distance l (m), before the gravity drop is subtracted.
    """

    return sqrt((h0 + r_earth)*(h0 + r_earth) - 2*l*cos(zc)*(h0 + r_earth) + l*l) - r_earth


@cython.cdivision(True)
cdef double atmDensityPolyC(double ht, FLOAT_TYPE_t[:] dens_co):
    """ Cython/memoryview twin of atmDensityPoly() for use inside the sub-step loop (avoids the
        ndarray-typed cpdef boundary).

    Arguments:
        ht: [double] Height (m).
        dens_co: [memoryview of float64] 7 coefficients of the log10(density) height polynomial.

    Return:
        rho: [double] Atmospheric mass density at height ht (kg/m^3).
    """

    cdef double x = ht/1e6
    return 10**(dens_co[0] + dens_co[1]*x + dens_co[2]*x*x + dens_co[3]*x*x*x
                + dens_co[4]*x*x*x*x + dens_co[5]*x*x*x*x*x + dens_co[6]*x*x*x*x*x*x)


@cython.cdivision(True)
cdef void advanceVelPosC(double m, double v, double vv, double vh, double length, double grav,
                         double dm, double decel_rate, double h, double h_at,
                         double r_earth, double g0, double* out):
    """ Advance the velocity components, speed, along-track length, and gravity drop of one fragment by
        one sub-step of size h, reproducing the fixed engine's single-body update exactly (velocity is
        updated BEFORE the length; gravity only lowers the height, it does not enter the velocity
        magnitude). Used by the step-doubling stepper (adaptiveSingleBodyStep).

    Arguments:
        m: [double] Mass at the sub-step start (kg).
        v: [double] Speed at the sub-step start (m/s).
        vv: [double] Vertical velocity component (m/s, negative downward).
        vh: [double] Horizontal velocity component (m/s).
        length: [double] Along-track length at the sub-step start (m).
        grav: [double] Accumulated gravity drop so far (m).
        dm: [double] Mass change over this sub-step (kg, already clamped).
        decel_rate: [double] Drag deceleration dv/dt (m/s^2, <= 0).
        h: [double] Sub-step size (s).
        h_at: [double] Height at which to evaluate gravity/curvature for this sub-step (m).
        r_earth: [double] Earth radius (m).
        g0: [double] Surface gravitational acceleration (m/s^2).
        out: [double*] Output buffer (length 7), filled with
            [m_new, v_new, vv_new, vh_new, length_new, grav_new, went_up_flag].

    Return:
        None (results written into 'out').
    """

    cdef double gv, av, ah, vv_n, vh_n, v_n

    # Accelerating (decel_rate > 0) or already stopped -> stop the fragment (mirror the fixed-path
    #   stop branch in ablateAll)
    if (decel_rate > 0) or (v <= 0):
        out[0] = m + dm
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        out[4] = length
        out[5] = grav
        out[6] = 0.0
        return
    gv = g0/((1.0 + h_at/r_earth)*(1.0 + h_at/r_earth))
    av = -decel_rate*vv/v + vh*v/(r_earth + h_at)
    ah = -decel_rate*vh/v - vv*v/(r_earth + h_at)
    vv_n = vv - av*h
    vh_n = vh - ah*h
    v_n = sqrt(vv_n*vv_n + vh_n*vh_n)
    out[0] = m + dm
    out[1] = v_n
    out[2] = vv_n
    out[3] = vh_n
    out[4] = length + v_n*h            # length uses the UPDATED speed, as in the fixed advance
    out[5] = grav + 0.5*gv*h*h
    out[6] = 1.0 if vv_n > 0 else 0.0  # going up


@cython.cdivision(True)
cpdef adaptiveSingleBodyStep(double dt_macro, double K, double sigma, double erosion_coeff,
        int erosion_active, double m, double v, double vv, double vh, double length,
        double h_grav_drop_total, double h_init, double zenith_angle, double r_earth, double g0,
        FLOAT_TYPE_t[:] dens_co, double rtol, double atol_m, double atol_v, double m_kill,
        double dt_min, double dt_max, int max_substeps, double h_sub_init,
        double erosion_release_length, double erosion_release_vref):
    """ Advance ONE fragment across a full macro interval dt_macro using error-controlled adaptive
        sub-steps (step-doubling on the operator-split RK4), refreshing the atmosphere/height each
        sub-step. Reproduces the single-body advance of the fixed engine in the one-sub-step limit, but
        drives the local error below (rtol, atol). Grain generation, kill checks, disruption and complex
        fragmentation stay at the macro boundary and are handled by the caller (ablateAll).

        See adaptiveDP45Step for the (default) embedded Dormand-Prince variant with the same I/O.

    Arguments:
        dt_macro: [double] Macro (output) interval to advance over (s).
        K: [double] Shape-density coefficient (m^2/kg^(2/3)).
        sigma: [double] Ablation coefficient (s^2/m^2).
        erosion_coeff: [double] Erosion coefficient (s^2/m^2); used only if erosion_active.
        erosion_active: [int] 1 if this fragment is currently eroding (erosion_enabled and coeff > 0).
        m: [double] Mass at the interval start (kg).
        v: [double] Speed at the interval start (m/s).
        vv: [double] Vertical velocity component (m/s, negative downward).
        vh: [double] Horizontal velocity component (m/s).
        length: [double] Along-track length at the interval start (m).
        h_grav_drop_total: [double] Accumulated gravity drop at the interval start (m).
        h_init: [double] Initial simulation height (m).
        zenith_angle: [double] Entry zenith angle (radians).
        r_earth: [double] Earth radius (m).
        g0: [double] Surface gravitational acceleration (m/s^2).
        dens_co: [memoryview of float64] 7 atmosphere log10(density) polynomial coefficients.
        rtol: [double] Relative error tolerance on mass and speed.
        atol_m: [double] Absolute mass tolerance (kg).
        atol_v: [double] Absolute speed tolerance (m/s).
        m_kill: [double] Kill mass (kg); the drag mass is floored at this to keep m^(-1/3) finite.
        dt_min: [double] Minimum sub-step (s).
        dt_max: [double] Maximum sub-step (s).
        max_substeps: [int] Sub-step cap per macro interval (runaway guard).
        h_sub_init: [double] Warm-start sub-step from the previous macro step (s); <= 0 means "use dt_macro".
        erosion_release_length: [double] Along-track grain-release interval (m). For an eroding fragment
            the sub-step is capped at erosion_release_length/min(v, erosion_release_vref) so grains are
            shed ~every this many metres, making the grain-birth cadence independent of dt/rtol. Ignored
            for non-eroding fragments.
        erosion_release_vref: [double] Reference speed (m/s) above which the along-track cadence is frozen
            in TIME rather than distance: fragments faster than this cap their sub-step at
            erosion_release_length/erosion_release_vref instead of .../v, bounding the sub-step count for
            fast meteors (cost grows with v otherwise). Slower fragments keep the exact distance cadence.
            Set <= 0 to disable the cap (pure distance cadence, the original behaviour).

    Return:
        A 17-tuple:
            m: [double] Mass at the interval end (kg).
            v: [double] Speed at the interval end (m/s).
            vv: [double] Vertical velocity component at the end (m/s).
            vh: [double] Horizontal velocity component at the end (m/s).
            length: [double] Along-track length at the end (m).
            h_grav_drop_total: [double] Accumulated gravity drop at the end (m).
            h_new: [double] Height at the interval end (m).
            rho_final: [double] Atmospheric density at h_new, for the end-of-step dynamic pressure (kg/m^3).
            dm_abl_macro: [double] Unclamped ablation mass loss over the interval (kg, for luminosity/q).
            dm_ero_macro: [double] Unclamped erosion mass loss over the interval (kg).
            decel_return: [double] Macro-averaged DRAG dv/dt (m/s^2, negative), for the luminosity term.
            went_up: [int] 1 if the fragment turned upward during the interval.
            n_substeps: [int] Number of accepted sub-steps taken.
            h_sub_carry: [double] Warm-start sub-step to carry to the next macro step (s).
            runaway: [int] 1 if the max_substeps cap was hit.
            floor_accepts: [int] Sub-steps accepted at dt_min while still over tolerance (under-resolved).
            erosion_events: [list or None] Per-sub-step erosion shedding events, each a tuple
                (eroded_mass, h, v, vv, vh, length, h_grav_drop_total); None for non-eroding fragments.
    """

    cdef double t = 0.0
    cdef double v_cap
    cdef double h_sub, h_cur, rho_atm, rho_mid, rho_last
    cdef double dm_abl_big, dm_ero_big, dm_tot_big, decel_big
    cdef double dm_abl_1, dm_ero_1, dm_tot_1, decel_1
    cdef double dm_abl_2, dm_ero_2, dm_tot_2, decel_2
    cdef double m_h, v_h, vv_h, vh_h, len_h, grav_h, h_mid
    cdef double m_big, v_big, m_two, v_two, hh
    cdef double err_m, err_v, sc_m, sc_v, E, E_prev, fac
    cdef double dm_abl_macro = 0.0
    cdef double dm_ero_macro = 0.0
    cdef double dv_drag = 0.0          # accumulated DRAG-only speed change (excludes gravity/curvature)
    cdef double v_start = v
    cdef int n_substeps = 0
    cdef int went_up = 0
    cdef int runaway = 0
    cdef int at_floor
    cdef int was_clamped        # this sub-step was shortened to land on the macro boundary
    cdef int floor_accepts = 0  # sub-steps accepted only because at dt_min with E>1 (under-resolved)
    cdef double h_sub_carry     # controller's natural next step, kept unclamped by the macro boundary
    cdef double out1[7]
    cdef double out2[7]
    cdef double outb[7]
    cdef double h_new, decel_return, rho_final

    cdef double safety = 0.9
    cdef double facmin = 0.2
    cdef double facmax = 5.0

    # Per-sub-step erosion shedding events (only for eroding fragments), so that ablateAll can spawn
    #   grains at each sub-step's resolved state instead of dumping the whole macro interval's eroded
    #   mass at one point. Each entry: (eroded_mass, h, v, vv, vh, length, h_grav_drop_total). None for
    #   non-eroding fragments (the vast majority) to keep their fast path allocation-free.
    erosion_events = [] if erosion_active else None

    h_sub = h_sub_init
    if h_sub <= 0:
        h_sub = dt_macro
    if h_sub > dt_max:
        h_sub = dt_max
    if h_sub < dt_min:
        h_sub = dt_min
    E_prev = 1.0
    rho_last = 0.0
    h_sub_carry = h_sub

    while t < dt_macro:

        # Clamp the final sub-step so sub-steps sum to exactly dt_macro (no overshoot). Remember whether
        #   this step was shortened by the boundary so we don't persist the tiny clamped size as the
        #   warm-start for the next macro step (that would restart every macro step with a tiny sub-step).
        was_clamped = 0
        if t + h_sub > dt_macro:
            was_clamped = 1
            h_sub = dt_macro - t

        # Cap an eroding fragment's sub-step so grains are shed roughly every erosion_release_length
        #   metres of flight, making the grain-birth cadence (and hence the erosion light curve)
        #   independent of dt and the error tolerance. Not applied to non-eroding fragments or grains.
        #   Above erosion_release_vref the cadence is frozen in time (uses v_cap = vref, not v) so the
        #   sub-step count stays bounded for fast meteors; slower fragments keep the pure distance cadence.
        v_cap = v
        if (erosion_release_vref > 0) and (v > erosion_release_vref):
            v_cap = erosion_release_vref
        if erosion_active and (v > 0) and (erosion_release_length > 0) \
                and (h_sub*v_cap > erosion_release_length):
            h_sub = erosion_release_length/v_cap

        hh = 0.5*h_sub

        # Height and atmosphere at the CURRENT state (refresh -> shrinks the operator-split/frozen-rho error)
        h_cur = heightCurvatureC(h_init, zenith_angle, length, r_earth) - h_grav_drop_total
        rho_atm = atmDensityPolyC(h_cur, dens_co)
        rho_last = rho_atm

        # --- Big step (size h_sub) ---
        dm_abl_big = massLossRK4(h_sub, K, sigma, m, rho_atm, v)
        if erosion_active:
            dm_ero_big = massLossRK4(h_sub, K, erosion_coeff, m, rho_atm, v)
        else:
            dm_ero_big = 0.0
        dm_tot_big = clampMassC(dm_abl_big + dm_ero_big, m)
        # Floor the deceleration mass at m_kill: deceleration ~ m^(-1/3) diverges as m -> 0, and a grain
        #   at/under m_kill is treated as dead (killed at the macro boundary), so this only bounds the
        #   drag on an already-exhausted grain rather than letting it blow up.
        decel_big = decelerationRK4(h_sub, K, fmax(m, m_kill), rho_atm, v)
        advanceVelPosC(m, v, vv, vh, length, h_grav_drop_total, dm_tot_big, decel_big, h_sub, h_cur,
                       r_earth, g0, outb)
        m_big = outb[0]
        v_big = outb[1]

        # --- Two half steps (hh each; rho refreshed at the midpoint) ---
        dm_abl_1 = massLossRK4(hh, K, sigma, m, rho_atm, v)
        if erosion_active:
            dm_ero_1 = massLossRK4(hh, K, erosion_coeff, m, rho_atm, v)
        else:
            dm_ero_1 = 0.0
        dm_tot_1 = clampMassC(dm_abl_1 + dm_ero_1, m)
        decel_1 = decelerationRK4(hh, K, fmax(m, m_kill), rho_atm, v)
        advanceVelPosC(m, v, vv, vh, length, h_grav_drop_total, dm_tot_1, decel_1, hh, h_cur,
                       r_earth, g0, out1)
        m_h = out1[0]
        v_h = out1[1]
        vv_h = out1[2]
        vh_h = out1[3]
        len_h = out1[4]
        grav_h = out1[5]

        h_mid = heightCurvatureC(h_init, zenith_angle, len_h, r_earth) - grav_h
        rho_mid = atmDensityPolyC(h_mid, dens_co)

        dm_abl_2 = massLossRK4(hh, K, sigma, m_h, rho_mid, v_h)
        if erosion_active:
            dm_ero_2 = massLossRK4(hh, K, erosion_coeff, m_h, rho_mid, v_h)
        else:
            dm_ero_2 = 0.0
        dm_tot_2 = clampMassC(dm_abl_2 + dm_ero_2, m_h)
        decel_2 = decelerationRK4(hh, K, fmax(m_h, m_kill), rho_mid, v_h)
        advanceVelPosC(m_h, v_h, vv_h, vh_h, len_h, grav_h, dm_tot_2, decel_2, hh, h_mid,
                       r_earth, g0, out2)
        m_two = out2[0]
        v_two = out2[1]

        # --- Error estimate (step doubling, RK4 order p=4 -> denom 2^p - 1 = 15) ---
        err_m = fabs(m_two - m_big)/15.0
        err_v = fabs(v_two - v_big)/15.0
        sc_m = atol_m + rtol*fmax(fabs(m_two), fabs(m_big))
        sc_v = atol_v + rtol*fmax(fabs(v_two), fabs(v_big))
        E = sqrt(0.5*((err_m/sc_m)*(err_m/sc_m) + (err_v/sc_v)*(err_v/sc_v)))

        at_floor = 1 if h_sub <= dt_min*(1.0 + 1e-12) else 0

        if (E <= 1.0) or at_floor:

            # Count sub-steps accepted only because they hit the dt_min floor while still over tolerance
            #   (E > 1) - these are locally under-resolved and otherwise invisible (see runSimulation warn)
            if at_floor and (E > 1.0):
                floor_accepts += 1

            # Accept the two-half (more accurate) state; accumulate per-species mass loss (unclamped)
            dm_abl_macro += dm_abl_1 + dm_abl_2
            dm_ero_macro += dm_ero_1 + dm_ero_2
            # Accumulate the DRAG-only speed change over this sub-step (decel_1/decel_2 are the pure-drag
            #   dv/dt from decelerationRK4). This feeds the luminosity deceleration term - unlike the net
            #   (v_start - v), it excludes the gravity/curvature reallocation, matching the fixed path.
            dv_drag += (decel_1 + decel_2)*hh
            m = out2[0]
            v = out2[1]
            vv = out2[2]
            vh = out2[3]
            length = out2[4]
            h_grav_drop_total = out2[5]
            t += h_sub
            n_substeps += 1

            # Record the eroded mass shed on this sub-step, tagged with the fragment's just-advanced
            #   state, so ablateAll can spawn grains here (finely resolved along the trajectory) rather
            #   than dumping the whole macro interval's eroded mass at the end point.
            if erosion_active and ((dm_ero_1 + dm_ero_2) < 0):
                erosion_events.append((
                    -(dm_ero_1 + dm_ero_2),
                    heightCurvatureC(h_init, zenith_angle, length, r_earth) - h_grav_drop_total,
                    v, vv, vh, length, h_grav_drop_total))

            if m <= m_kill:        # grain exhausted -> freeze, killed at the macro boundary
                break
            if out2[6] > 0.5:      # turned upward mid-interval -> freeze, kill at macro boundary
                vv = 0.0
                went_up = 1
                break
            if v <= 0:             # stopped (accelerating/decel guard) -> freeze
                break
            if n_substeps >= max_substeps:
                runaway = 1
                break

            # PI step-size controller (k = p + 1 = 5)
            if E <= 0:
                E = 1e-10
            fac = safety*(E**(-0.7/5.0))*(E_prev**(0.4/5.0))
            E_prev = E
            if fac < facmin:
                fac = facmin
            if fac > facmax:
                fac = facmax
            h_sub = h_sub*fac
            if h_sub > dt_max:
                h_sub = dt_max
            if h_sub < dt_min:
                h_sub = dt_min

            # Persist the controller's natural step as the warm-start, but only from sub-steps that were
            #   NOT boundary-clamped, so the tiny final sub-step doesn't shrink the next macro step's seed
            #   (that would restart every macro step with a tiny sub-step). This cuts sub-steps ~2-5x with
            #   no effect on the visible light curve (only the faint, sub-detection tail shifts slightly).
            if not was_clamped:
                h_sub_carry = h_sub
        else:
            # Reject: shrink and retry the SAME sub-step (do not advance t)
            fac = safety*(E**(-1.0/5.0))
            if fac < facmin:
                fac = facmin
            h_sub = h_sub*fac
            if h_sub < dt_min:
                h_sub = dt_min

    # Height from the along-track length (matches the fixed path, MetSimErosion.py: heightCurvature minus
    #   the gravity drop). Note: on went_up we do NOT force h_new = 0 - the fixed path's frag.h = 0 on
    #   'going up' is immediately overwritten by this same recompute there, so forcing 0 would kill the
    #   fragment a tick earlier than fixed mode. went_up already zeroed vv, matching the fixed behaviour.
    h_new = heightCurvatureC(h_init, zenith_angle, length, r_earth) - h_grav_drop_total

    # Density for the end-of-interval dynamic pressure; fall back to the last in-loop value if h<=0
    if h_new > 0:
        rho_final = atmDensityPolyC(h_new, dens_co)
    else:
        rho_final = rho_last

    decel_return = dv_drag/dt_macro   # macro-averaged DRAG dv/dt (negative), matches decelerationRK4 sign

    return (m, v, vv, vh, length, h_grav_drop_total, h_new, rho_final,
            dm_abl_macro, dm_ero_macro, decel_return, went_up, n_substeps, h_sub_carry, runaway,
            floor_accepts, erosion_events)



@cython.cdivision(True)
cdef void _rhsDP(double m, double vv, double vh, double length, double h_grav_drop,
                 double K, double sigma, double erosion_coeff, int erosion_active, double m_kill,
                 double h_init, double zenith_angle, double r_earth, FLOAT_TYPE_t[:] dens_co,
                 double* dydt, double* extras):
    """ Coupled right-hand side for the Dormand-Prince stepper: derivatives of the state
        y = [m, vv, vh, length]. Mirrors the fixed model - mass loss ~ m^(2/3), drag ~ m^(-1/3) (with the
        drag mass floored at m_kill so it stays finite near exhaustion), and the vv/vh derivatives carry
        the Earth-curvature term. Gravity acts only on the height drop and is handled by the caller (not
        here). The height for the atmosphere lookup is derived from length and the (frozen) gravity drop.

    Arguments:
        m: [double] Mass (kg).
        vv: [double] Vertical velocity component (m/s).
        vh: [double] Horizontal velocity component (m/s).
        length: [double] Along-track length (m).
        h_grav_drop: [double] Gravity drop to subtract from the curvature height (m), frozen over the step.
        K: [double] Shape-density coefficient (m^2/kg^(2/3)).
        sigma: [double] Ablation coefficient (s^2/m^2).
        erosion_coeff: [double] Erosion coefficient (s^2/m^2); used only if erosion_active.
        erosion_active: [int] 1 if the fragment is eroding.
        m_kill: [double] Kill mass (kg); floor for the drag mass.
        h_init: [double] Initial simulation height (m).
        zenith_angle: [double] Entry zenith angle (radians).
        r_earth: [double] Earth radius (m).
        dens_co: [memoryview of float64] Atmosphere density polynomial coefficients.
        dydt: [double*] Output buffer (length 4): [dm/dt, dvv/dt, dvh/dt, dlength/dt].
        extras: [double*] Output buffer (length 4): [ablation_rate, erosion_rate, drag_decel, rho],
            used by the caller for the per-step luminosity/electron-density/grain-shedding diagnostics.

    Return:
        None (results written into 'dydt' and 'extras').
    """

    cdef double v, h, rho, decel, mm, mpos
    v = sqrt(vv*vv + vh*vh)
    h = heightCurvatureC(h_init, zenith_angle, length, r_earth) - h_grav_drop
    rho = atmDensityPolyC(h, dens_co)
    mpos = m if m > 0.0 else 0.0
    extras[0] = massLoss(K, sigma, mpos, rho, v)
    if erosion_active:
        extras[1] = massLoss(K, erosion_coeff, mpos, rho, v)
    else:
        extras[1] = 0.0
    mm = fmax(m, m_kill)
    decel = deceleration(K, mm, rho, v)
    extras[2] = decel
    extras[3] = rho
    dydt[0] = extras[0] + extras[1]
    if v > 0.0:
        dydt[1] = decel*vv/v - vh*v/(r_earth + h)
        dydt[2] = decel*vh/v + vv*v/(r_earth + h)
    else:
        dydt[1] = 0.0
        dydt[2] = 0.0
    dydt[3] = v


@cython.cdivision(True)
cpdef adaptiveDP45Step(double dt_macro, double K, double sigma, double erosion_coeff,
        int erosion_active, double m, double v, double vv, double vh, double length,
        double h_grav_drop_total, double h_init, double zenith_angle, double r_earth, double g0,
        FLOAT_TYPE_t[:] dens_co, double rtol, double atol_m, double atol_v, double m_kill,
        double dt_min, double dt_max, int max_substeps, double h_sub_init,
        double erosion_release_length, double erosion_release_vref):
    """ Advance ONE fragment across a full macro interval dt_macro, like adaptiveSingleBodyStep, but with
        an embedded Dormand-Prince RK45 pair (5th-order solution + 4th-order error estimate) on the
        COUPLED system y = [m, vv, vh, length] instead of step-doubling on the operator split. One
        embedded step yields both solutions in ~7 RHS evaluations (vs ~24 for step-doubling), and its
        higher order takes larger sub-steps, so it is ~3-4x faster at the same tolerance. The atmosphere
        density is refreshed at every RK stage (from that stage's height), removing the operator-split /
        frozen-density error; the gravity drop is added once per accepted sub-step (as in the fixed
        model). This is the default adaptive stepper (const.adaptive_high_order).

        Arguments and the 17-element return tuple are identical to adaptiveSingleBodyStep() - see its
        docstring for the full I/O description.
    """

    # Dormand-Prince (RK45) Butcher tableau
    cdef double a21 = 1.0/5
    cdef double a31 = 3.0/40, a32 = 9.0/40
    cdef double a41 = 44.0/45, a42 = -56.0/15, a43 = 32.0/9
    cdef double a51 = 19372.0/6561, a52 = -25360.0/2187, a53 = 64448.0/6561, a54 = -212.0/729
    cdef double a61 = 9017.0/3168, a62 = -355.0/33, a63 = 46732.0/5247, a64 = 49.0/176, a65 = -5103.0/18656
    cdef double a71 = 35.0/384, a73 = 500.0/1113, a74 = 125.0/192, a75 = -2187.0/6784, a76 = 11.0/84
    # 5th-order weights b = 7th stage row (FSAL); 4th-order embedded weights b*
    cdef double b1 = 35.0/384, b3 = 500.0/1113, b4 = 125.0/192, b5 = -2187.0/6784, b6 = 11.0/84
    cdef double bs1 = 5179.0/57600, bs3 = 7571.0/16695, bs4 = 393.0/640, bs5 = -92097.0/339200
    cdef double bs6 = 187.0/2100, bs7 = 1.0/40

    cdef double t = 0.0
    cdef double h_sub, hh, gv, h_cur, rho0, v_cur, v_cap
    cdef double y[4]
    cdef double yt[4]
    cdef double k1[4]
    cdef double k2[4]
    cdef double k3[4]
    cdef double k4[4]
    cdef double k5[4]
    cdef double k6[4]
    cdef double k7[4]
    # Extras per stage: [ablation_rate, erosion_rate, drag_decel, rho]
    cdef double e1[4]
    cdef double e2[4]
    cdef double e3[4]
    cdef double e4[4]
    cdef double e5[4]
    cdef double e6[4]
    cdef double e7[4]
    cdef double m_new, v_new, vv_new, vh_new, len_new
    cdef double abl_step, ero_step, drag_step, err_m, err_v, sc_m, sc_v, E, E_prev, fac
    cdef double dm_abl_macro = 0.0, dm_ero_macro = 0.0, dv_drag = 0.0
    cdef double h_new, decel_return, rho_final, rho_last = 0.0, h_sub_carry
    cdef int n_substeps = 0, went_up = 0, runaway = 0, at_floor, was_clamped, floor_accepts = 0
    cdef int c
    cdef double safety = 0.9, facmin = 0.2, facmax = 5.0

    erosion_events = [] if erosion_active else None

    y[0] = m
    y[1] = vv
    y[2] = vh
    y[3] = length

    h_sub = h_sub_init
    if h_sub <= 0:
        h_sub = dt_macro
    if h_sub > dt_max:
        h_sub = dt_max
    if h_sub < dt_min:
        h_sub = dt_min
    E_prev = 1.0
    h_sub_carry = h_sub

    while t < dt_macro:

        was_clamped = 0
        if t + h_sub > dt_macro:
            was_clamped = 1
            h_sub = dt_macro - t

        # Cap an eroding fragment's sub-step so grains are shed roughly every erosion_release_length
        #   metres of flight - a grain-birth cadence independent of dt and the error tolerance (see the
        #   step-doubling stepper for the rationale). Only for eroding fragments; grains do not erode.
        #   Above erosion_release_vref the cadence is frozen in time (v_cap = vref) to bound the sub-step
        #   count for fast meteors; slower fragments keep the pure distance cadence.
        v_cur = sqrt(y[1]*y[1] + y[2]*y[2])
        v_cap = v_cur
        if (erosion_release_vref > 0) and (v_cur > erosion_release_vref):
            v_cap = erosion_release_vref
        if erosion_active and (v_cur > 0) and (erosion_release_length > 0) \
                and (h_sub*v_cap > erosion_release_length):
            h_sub = erosion_release_length/v_cap

        # Height/gravity at the sub-step start (gravity drop uses the start-of-step g, as in the fixed model)
        h_cur = heightCurvatureC(h_init, zenith_angle, y[3], r_earth) - h_grav_drop_total
        gv = g0/((1.0 + h_cur/r_earth)*(1.0 + h_cur/r_earth))

        # --- 7 Dormand-Prince stages (h_grav_drop frozen across the sub-step) ---
        _rhsDP(y[0], y[1], y[2], y[3], h_grav_drop_total, K, sigma, erosion_coeff, erosion_active,
               m_kill, h_init, zenith_angle, r_earth, dens_co, k1, e1)
        rho0 = e1[3]
        for c in range(4):
            yt[c] = y[c] + h_sub*a21*k1[c]
        _rhsDP(yt[0], yt[1], yt[2], yt[3], h_grav_drop_total, K, sigma, erosion_coeff, erosion_active,
               m_kill, h_init, zenith_angle, r_earth, dens_co, k2, e2)
        for c in range(4):
            yt[c] = y[c] + h_sub*(a31*k1[c] + a32*k2[c])
        _rhsDP(yt[0], yt[1], yt[2], yt[3], h_grav_drop_total, K, sigma, erosion_coeff, erosion_active,
               m_kill, h_init, zenith_angle, r_earth, dens_co, k3, e3)
        for c in range(4):
            yt[c] = y[c] + h_sub*(a41*k1[c] + a42*k2[c] + a43*k3[c])
        _rhsDP(yt[0], yt[1], yt[2], yt[3], h_grav_drop_total, K, sigma, erosion_coeff, erosion_active,
               m_kill, h_init, zenith_angle, r_earth, dens_co, k4, e4)
        for c in range(4):
            yt[c] = y[c] + h_sub*(a51*k1[c] + a52*k2[c] + a53*k3[c] + a54*k4[c])
        _rhsDP(yt[0], yt[1], yt[2], yt[3], h_grav_drop_total, K, sigma, erosion_coeff, erosion_active,
               m_kill, h_init, zenith_angle, r_earth, dens_co, k5, e5)
        for c in range(4):
            yt[c] = y[c] + h_sub*(a61*k1[c] + a62*k2[c] + a63*k3[c] + a64*k4[c] + a65*k5[c])
        _rhsDP(yt[0], yt[1], yt[2], yt[3], h_grav_drop_total, K, sigma, erosion_coeff, erosion_active,
               m_kill, h_init, zenith_angle, r_earth, dens_co, k6, e6)
        for c in range(4):
            yt[c] = y[c] + h_sub*(a71*k1[c] + a73*k3[c] + a74*k4[c] + a75*k5[c] + a76*k6[c])
        _rhsDP(yt[0], yt[1], yt[2], yt[3], h_grav_drop_total, K, sigma, erosion_coeff, erosion_active,
               m_kill, h_init, zenith_angle, r_earth, dens_co, k7, e7)

        # 5th-order solution (= yt above, the 7th stage node) and error vs the 4th-order embedded
        m_new = yt[0]
        vv_new = yt[1]
        vh_new = yt[2]
        len_new = yt[3]
        v_new = sqrt(vv_new*vv_new + vh_new*vh_new)

        # Error estimate on mass and speed (difference of 5th- and 4th-order weights)
        err_m = fabs(h_sub*((b1-bs1)*k1[0] + (b3-bs3)*k3[0] + (b4-bs4)*k4[0] + (b5-bs5)*k5[0]
                            + (b6-bs6)*k6[0] + (0.0-bs7)*k7[0]))
        # Velocity error via the vv/vh error components projected onto speed
        err_v = fabs(h_sub*((b1-bs1)*k1[1] + (b3-bs3)*k3[1] + (b4-bs4)*k4[1] + (b5-bs5)*k5[1]
                            + (b6-bs6)*k6[1] + (0.0-bs7)*k7[1]))
        err_v += fabs(h_sub*((b1-bs1)*k1[2] + (b3-bs3)*k3[2] + (b4-bs4)*k4[2] + (b5-bs5)*k5[2]
                             + (b6-bs6)*k6[2] + (0.0-bs7)*k7[2]))
        sc_m = atol_m + rtol*fmax(fabs(m_new), fabs(y[0]))
        sc_v = atol_v + rtol*fmax(v_new, v)
        E = sqrt(0.5*((err_m/sc_m)*(err_m/sc_m) + (err_v/sc_v)*(err_v/sc_v)))

        at_floor = 1 if h_sub <= dt_min*(1.0 + 1e-12) else 0

        if (E <= 1.0) or at_floor:

            if at_floor and (E > 1.0):
                floor_accepts += 1

            # Per-step mass loss split (integral of each rate with the 5th-order weights)
            abl_step = h_sub*(b1*e1[0] + b3*e3[0] + b4*e4[0] + b5*e5[0] + b6*e6[0])
            ero_step = h_sub*(b1*e1[1] + b3*e3[1] + b4*e4[1] + b5*e5[1] + b6*e6[1])
            drag_step = h_sub*(b1*e1[2] + b3*e3[2] + b4*e4[2] + b5*e5[2] + b6*e6[2])
            dm_abl_macro += abl_step
            dm_ero_macro += ero_step
            dv_drag += drag_step

            # Commit the state; floor the mass at 0 and add the gravity drop for this sub-step
            y[0] = m_new if m_new > 0.0 else 0.0
            y[1] = vv_new
            y[2] = vh_new
            y[3] = len_new
            h_grav_drop_total += 0.5*gv*h_sub*h_sub
            rho_last = rho0
            t += h_sub
            n_substeps += 1

            # Shed the eroded mass at this sub-step's resolved state (see adaptiveSingleBodyStep)
            if erosion_active and (ero_step < 0):
                erosion_events.append((
                    -ero_step,
                    heightCurvatureC(h_init, zenith_angle, y[3], r_earth) - h_grav_drop_total,
                    sqrt(y[1]*y[1] + y[2]*y[2]), y[1], y[2], y[3], h_grav_drop_total))

            if y[0] <= m_kill:          # exhausted
                break
            if y[1] > 0.0:              # turned upward -> freeze, matches fixed (vv zeroed)
                y[1] = 0.0
                went_up = 1
                break
            v_new = sqrt(y[1]*y[1] + y[2]*y[2])
            if v_new <= 0.0:
                break
            if n_substeps >= max_substeps:
                runaway = 1
                break

            # PI step-size controller (order p=4 for the error estimate -> k=5)
            if E <= 0:
                E = 1e-10
            fac = safety*(E**(-0.7/5.0))*(E_prev**(0.4/5.0))
            E_prev = E
            if fac < facmin:
                fac = facmin
            if fac > facmax:
                fac = facmax
            h_sub = h_sub*fac
            if h_sub > dt_max:
                h_sub = dt_max
            if h_sub < dt_min:
                h_sub = dt_min
            if not was_clamped:
                h_sub_carry = h_sub
        else:
            fac = safety*(E**(-1.0/5.0))
            if fac < facmin:
                fac = facmin
            h_sub = h_sub*fac
            if h_sub < dt_min:
                h_sub = dt_min

    m = y[0]
    vv = y[1]
    vh = y[2]
    length = y[3]
    v = sqrt(vv*vv + vh*vh)
    h_new = heightCurvatureC(h_init, zenith_angle, length, r_earth) - h_grav_drop_total
    if h_new > 0:
        rho_final = atmDensityPolyC(h_new, dens_co)
    else:
        rho_final = rho_last
    decel_return = dv_drag/dt_macro

    return (m, v, vv, vh, length, h_grav_drop_total, h_new, rho_final,
            dm_abl_macro, dm_ero_macro, decel_return, went_up, n_substeps, h_sub_carry, runaway,
            floor_accepts, erosion_events)