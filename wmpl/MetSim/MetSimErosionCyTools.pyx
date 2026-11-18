""" Fast cython functions for the MetSimErosion module. """


import cython
cimport cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, M_PI, M_PI_2, atan2, tanh, log, exp, log10, cos, sin, pow


# Define cython types for numpy arrays
FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t


# Do not let the C compiler fuse a multiplication and an addition into one rounding (FMA, done by default by
#   GCC for arm64 and by Clang), so that the results are the same as of the same arithmetic in Python
cdef extern from *:
    """
    #if defined(__clang__)
    #pragma STDC FP_CONTRACT OFF
    #elif defined(__GNUC__)
    #pragma GCC optimize ("fp-contract=off")
    #endif
    """



cdef class FragmentCy:
    """ A fragment of the meteoroid: the main fragment, a fragment of a disruption or a fragmentation entry,
        or a set of n_grains equal grains. The numbers used in every time step are C fields, which the
        fragment loop reads without Python objects. Use the Python class Fragment below, which adds the
        'complex' attribute.
    """

    # Identifier of the fragment
    cdef public object id

    # Shape-density coeff
    cdef public double K

    # Initial fragment mass, and the instantaneous fragment mass (kg)
    cdef public double m_init, m

    # Density (kg/m^3)
    cdef public double rho

    # Ablation coefficient (s^2/m^2)
    cdef public double sigma

    # Velocity, and its vertical and horizontal components (m/s)
    cdef public double v, vv, vh

    # Total drop due to gravity (m)
    cdef public double h_grav_drop_total

    # Speed of the drop due to gravity across the path (m/s)
    cdef public double v_grav_drop

    # Length along the trajectory (m)
    cdef public double length

    # Height (m)
    cdef public double h

    # Luminous intensity (Watts)
    cdef public double lum

    # Electron line density
    cdef public double q

    # Dynamic pressure (Gamma = 1.0, Pa)
    cdef public double dyn_press

    # Drag coefficient
    cdef public double gamma

    # Zenith angle (radians)
    cdef public double zenith_angle

    # Erosion coefficient value
    cdef public double erosion_coeff

    # Indicates that this fragment was born from a disruption event and uses a fixed, disruption-
    #   specific erosion coefficient (const.disruption_erosion_coeff) instead of the ordinary height-
    #   dependent one, so it's excluded from the height-based erosion_coeff recompute
    cdef public bint disruption_child

    # Indicates that the one-time density/ablation coeff change at erosion_height_change has already
    #   been applied to this (main) fragment, so it isn't silently re-applied (and doesn't clobber a
    #   later manual sigma change, e.g. from a complex fragmentation "A" event) on every subsequent tick
    cdef public bint erosion_height_change_applied

    # Grain mass distribution index
    cdef public double erosion_mass_index

    # Mass range for grains (kg)
    cdef public double erosion_mass_min, erosion_mass_max

    cdef public bint erosion_enabled

    cdef public bint disruption_enabled

    cdef public bint active

    # Fractional numbers of grains which the fragment eroded into each mass bin and did not release yet,
    #   and the parameters of the bins (see erodeIntoGrains)
    cdef public object erosion_bin_counts, erosion_bin_params

    # Indicate that a fragmentation entry set the erosion coefficient, so the height does not change it
    cdef public bint erosion_coeff_fixed

    # Indicate that this is the main fragment
    cdef public bint main

    # Indicate that the fragment is a grain
    cdef public bint grain

    # Indicate that this is born out of complex fragmentation (the attribute 'complex' of Fragment, as
    #   'complex' is a keyword of C)
    cdef public bint is_complex

    # Identifier of the complex fragmentation entry
    cdef public object complex_id

    # Number of grains the fragment stands for, as it was set (an int, or a float for a fraction), and as a
    #   double for the computations, converted once as Python converts it in every product
    cdef object n_grains_obj
    cdef double n_grains_double

    # Other attributes, among them 'const', which is a keyword of Cython
    cdef dict __dict__



    def __init__(self):

        self.id = 0

        self.__dict__['const'] = None

        self.K = 0
        self.m_init = 0
        self.m = 0
        self.rho = 0
        self.sigma = 0
        self.v = 0
        self.vv = 0
        self.vh = 0
        self.h_grav_drop_total = 0
        self.v_grav_drop = 0
        self.length = 0
        self.lum = 0
        self.q = 0
        self.dyn_press = 0
        self.erosion_coeff = 0
        self.disruption_child = False
        self.erosion_height_change_applied = False
        self.erosion_mass_index = 2.5
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10
        self.erosion_enabled = False
        self.disruption_enabled = False
        self.active = False
        self.n_grains = 1
        self.erosion_bin_counts = None
        self.erosion_bin_params = None
        self.erosion_coeff_fixed = False
        self.main = False
        self.grain = False
        self.is_complex = False
        self.complex_id = None



    @property
    def n_grains(self):
        """ Number of grains the fragment stands for. """

        return self.n_grains_obj



    @n_grains.setter
    def n_grains(self, value):

        self.n_grains_obj = value
        self.n_grains_double = value



    def init(self, sim_const, m, rho, v_init, sigma, gamma, zenith_angle, erosion_mass_index, \
        erosion_mass_min, erosion_mass_max):

        self.__dict__['const'] = sim_const

        self.m = m
        self.m_init = m
        self.h = sim_const.h_init
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
        self.vv = -v_init*cos(zenith_angle)
        self.vh = v_init*sin(zenith_angle)

        self.active = True
        self.n_grains = 1



    def updateShapeDensityCoeff(self):
        """ Update the value of the shape-density coefficient. """

        self.K = self.gamma*self.__dict__['const'].shape_factor*pow(self.rho, -2/3.0)



    def toDict(self):
        """ Return all attributes of the fragment as a dictionary, as the __dict__ of the Python class it
            replaced, e.g. to save the fragment as JSON. The __dict__ of this class only has 'const' and the
            attributes added by callers.

        Return:
            [dict] Names and values of the attributes.
        """

        frag_dict = {
            'id': self.id,
            'const': self.__dict__.get('const'),
            'K': self.K,
            'm_init': self.m_init,
            'm': self.m,
            'rho': self.rho,
            'sigma': self.sigma,
            'v': self.v,
            'vv': self.vv,
            'vh': self.vh,
            'h_grav_drop_total': self.h_grav_drop_total,
            'v_grav_drop': self.v_grav_drop,
            'length': self.length,
            'lum': self.lum,
            'q': self.q,
            'dyn_press': self.dyn_press,
            'erosion_coeff': self.erosion_coeff,
            'disruption_child': self.disruption_child,
            'erosion_height_change_applied': self.erosion_height_change_applied,
            'erosion_mass_index': self.erosion_mass_index,
            'erosion_mass_min': self.erosion_mass_min,
            'erosion_mass_max': self.erosion_mass_max,
            'erosion_enabled': self.erosion_enabled,
            'disruption_enabled': self.disruption_enabled,
            'active': self.active,
            'n_grains': self.n_grains_obj,
            'erosion_bin_counts': self.erosion_bin_counts,
            'erosion_bin_params': self.erosion_bin_params,
            'erosion_coeff_fixed': self.erosion_coeff_fixed,
            'main': self.main,
            'grain': self.grain,
            'complex': self.is_complex,
            'complex_id': self.complex_id,
            'h': self.h,
            'gamma': self.gamma,
            'zenith_angle': self.zenith_angle
            }

        # Add the attributes added by callers
        frag_dict.update(self.__dict__)

        return frag_dict



    def spawn_child(self):
        """ Create a child of the Fragment instance. Copy over the reference to the shared 'const' object
        and copy values of all other attributes. Note: if a mutable attribute that is not shared across
        Fragment instances is added, this function will need to be revised.
        """

        cdef FragmentCy child = type(self).__new__(type(self))

        child.id = self.id
        child.K = self.K
        child.m_init = self.m_init
        child.m = self.m
        child.rho = self.rho
        child.sigma = self.sigma
        child.v = self.v
        child.vv = self.vv
        child.vh = self.vh
        child.h_grav_drop_total = self.h_grav_drop_total
        child.v_grav_drop = self.v_grav_drop
        child.length = self.length
        child.h = self.h
        child.lum = self.lum
        child.q = self.q
        child.dyn_press = self.dyn_press
        child.gamma = self.gamma
        child.zenith_angle = self.zenith_angle
        child.erosion_coeff = self.erosion_coeff
        child.disruption_child = self.disruption_child
        child.erosion_height_change_applied = self.erosion_height_change_applied
        child.erosion_mass_index = self.erosion_mass_index
        child.erosion_mass_min = self.erosion_mass_min
        child.erosion_mass_max = self.erosion_mass_max
        child.erosion_enabled = self.erosion_enabled
        child.disruption_enabled = self.disruption_enabled
        child.active = self.active
        child.n_grains_obj = self.n_grains_obj
        child.n_grains_double = self.n_grains_double
        child.erosion_coeff_fixed = self.erosion_coeff_fixed
        child.main = self.main
        child.grain = self.grain
        child.is_complex = self.is_complex
        child.complex_id = self.complex_id
        child.__dict__ = dict(self.__dict__)

        # The child starts without grains of its own to release, and does not share the list of the parent
        child.erosion_bin_counts = None
        child.erosion_bin_params = None

        return child



class Fragment(FragmentCy):
    """ A fragment of the meteoroid (see FragmentCy), with the attribute 'complex', which indicates that the
        fragment is born out of complex fragmentation.
    """



    def _getComplex(self):
        return self.is_complex



    def _setComplex(self, value):
        self.is_complex = value



    complex = property(_getComplex, _setComplex)



    def __setstate__(self, state):
        """ Restore a pickled fragment, also one pickled by an older version of wmpl, in which Fragment was a
            Python class with all attributes in its __dict__.
        """

        if isinstance(state, dict):
            for name, value in state.items():
                setattr(self, name, value)

        else:
            self.__setstate_cython__(state)



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



# Largest fraction of the mass or the speed which one Runge-Kutta step may change, above which the step is
#   split, so that a small grain whose mass or speed changes on a time scale shorter than the step is
#   integrated in substeps
cdef double RK4_MAX_CHANGE = 0.3

# Largest number of substeps a time step is split into
cdef int RK4_MAX_SUBSTEPS = 65536



@cython.cdivision(True)
cdef bint rk4Substep(double dt, double K, double sigma, double m, double v, double length, double h_init, \
    double cos_zc, double r_earth, double h_drop, AtmDensityTableCy atm_table, const double[::1] dens_co, \
    double ablation_fraction, int lum_eff_type, double lum_eff, double mu, double v_kill, double *dm, \
    double *dv, double *dl, double *de, double *dq):
    """ Integrate the mass, the speed and the length over one Runge-Kutta step, and check that the step
        resolves the motion.

    Arguments:
        dt: [double] Step (s).
        K: [double] Shape-density coefficient (m^2/kg^(2/3)).
        sigma: [double] Sum of the ablation and the erosion coefficients (s^2/m^2).
        m: [double] Mass (kg).
        v: [double] Speed (m/s).
        length: [double] Length along the path from the beginning of the simulation (m).
        h_init: [double] Height of the beginning of the simulation (m).
        cos_zc: [double] Cosine of the zenith angle of the path at the beginning.
        r_earth: [double] Radius of the Earth (m).
        h_drop: [double] Drop of the height due to gravity, kept over the step (m).
        atm_table: [AtmDensityTableCy] Density table, or None to use the polynomial.
        dens_co: [ndarray] Coefficients of the density polynomial, used without the table.
        ablation_fraction: [double] Fraction of the mass loss which is ablated, the rest is eroded.
        lum_eff_type: [int] Type of the luminous efficiency (see luminousEfficiency).
        lum_eff: [double] Constant luminous efficiency, used by the type 0 (%).
        mu: [double] Mean mass of an atom of the meteoroid (kg).
        v_kill: [double] Speed below which a fragment gives no light and no ionization (m/s).
        dm, dv, dl: [double *] Changes of the mass (kg), the speed (m/s) and the length (m) over the step.
        de, dq: [double *] Energy radiated over the step (J), and the electron line density multiplied by
            the step (s/m).

    Return:
        [bool] True if every stage has a positive mass and speed and the step changes the mass and the speed
            by at most RK4_MAX_CHANGE of their values.
    """

    cdef double mk[4]
    cdef double vk[4]
    cdef double lk[4]
    cdef double ek[4]
    cdef double qk[4]
    cdef double m_stage = m
    cdef double v_stage = v
    cdef double l_stage = 0.0
    cdef double rho_atm
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

        # A stage which ablates all of the mass or reverses the motion does not resolve the motion
        if (m_stage <= 0) or (v_stage <= 0):
            return False

        rho_atm = pathDensity(length + l_stage, h_init, cos_zc, r_earth, h_drop, atm_table, dens_co)

        mk[i] = massLoss(K, sigma, m_stage, rho_atm, v_stage)
        vk[i] = deceleration(K, m_stage, rho_atm, v_stage)
        lk[i] = v_stage

        # Light from the ablated mass and the deceleration by the air, and the ionization from the ablated
        #   mass, integrated with the motion, while the fragment is faster than the speed at which it is
        #   ended, below which the efficiencies are out of their range
        if v_stage >= v_kill:
            ek[i] = -luminousEfficiency(lum_eff_type, lum_eff, v_stage, m_stage) \
                *(ablation_fraction*mk[i]*v_stage*v_stage/2.0 + m_stage*v_stage*vk[i])
            qk[i] = -ionizationEfficiency(v_stage)*ablation_fraction*mk[i]/(mu*v_stage)

        else:
            ek[i] = 0.0
            qk[i] = 0.0

    dm[0] = dt*(mk[0] + 2*mk[1] + 2*mk[2] + mk[3])/6.0
    dv[0] = dt*(vk[0] + 2*vk[1] + 2*vk[2] + vk[3])/6.0
    dl[0] = dt*(lk[0] + 2*lk[1] + 2*lk[2] + lk[3])/6.0
    de[0] = dt*(ek[0] + 2*ek[1] + 2*ek[2] + ek[3])/6.0
    dq[0] = dt*(qk[0] + 2*qk[1] + 2*qk[2] + qk[3])/6.0

    return (-dm[0] <= RK4_MAX_CHANGE*m) and (-dv[0] <= RK4_MAX_CHANGE*v)



@cython.cdivision(True)
cpdef tuple ablationStepRK4(double dt, double K, double sigma, double m, double v, double length, \
    double h_init, double zenith_angle, double r_earth, double h_drop, AtmDensityTableCy atm_table, \
    const double[::1] dens_co, double ablation_fraction, int lum_eff_type, double lum_eff, double mu, \
    double v_kill):
    """ Integrate the mass, the speed and the length of a fragment over one time step together with the
        4th order Runge-Kutta method, taking the atmosphere density at the height of every stage, and add up
        the light and the ionization over the step. If a step changes the mass or the speed by more than
        RK4_MAX_CHANGE, e.g. for a small grain deep in the atmosphere, the step is split into substeps which
        are halved until they resolve the motion, and doubled again after a resolved substep, and the light
        and the ionization are added up over the substeps.

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
        ablation_fraction: [double] Fraction of the mass loss which is ablated, the rest is eroded.
        lum_eff_type: [int] Type of the luminous efficiency (see luminousEfficiency).
        lum_eff: [double] Constant luminous efficiency, used by the type 0 (%).
        mu: [double] Mean mass of an atom of the meteoroid (kg).
        v_kill: [double] Speed below which a fragment gives no light and no ionization (m/s).

    Return:
        (dm, dv, dl, energy, electrons): [tuple]
            - dm: [double] Change of the mass over the step (kg).
            - dv: [double] Change of the speed by the air over the step (m/s).
            - dl: [double] Change of the length over the step (m).
            - energy: [double] Energy radiated over the step (J), from the ablated mass and the deceleration
                by the air, integrated with the motion.
            - electrons: [double] Electron line density multiplied by the time step (s/m), integrated with
                the motion.
    """

    cdef double cos_zc = cos(zenith_angle)
    cdef double dt_min = dt/RK4_MAX_SUBSTEPS
    cdef double t = 0.0
    cdef double h = dt
    cdef double m_now = m
    cdef double v_now = v
    cdef double l_now = 0.0
    cdef double dm = 0.0
    cdef double dv = 0.0
    cdef double dl = 0.0
    cdef double energy = 0.0
    cdef double electrons = 0.0
    cdef double de = 0.0
    cdef double dq = 0.0
    cdef double m_mid, v_mid
    cdef double rho_atm
    cdef bint resolved

    if (m <= 0) or (v <= 0):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    while t < dt:

        # Do not step past the end of the time step
        if h > dt - t:
            h = dt - t

        resolved = rk4Substep(h, K, sigma, m_now, v_now, length + l_now, h_init, cos_zc, r_earth, h_drop, \
            atm_table, dens_co, ablation_fraction, lum_eff_type, lum_eff, mu, v_kill, &dm, &dv, &dl, &de, &dq)

        # Halve a substep which does not resolve the motion, down to the smallest substep
        if (not resolved) and (h > dt_min):
            h /= 2.0
            continue

        # The smallest substep can only fail to resolve the motion when the fragment is ablated or stopped
        #   within it, so take it with the rates at its start
        if not resolved:
            rho_atm = pathDensity(length + l_now, h_init, cos_zc, r_earth, h_drop, atm_table, dens_co)
            dm = h*massLoss(K, sigma, m_now, rho_atm, v_now)
            dv = h*deceleration(K, m_now, rho_atm, v_now)
            dl = h*v_now

        # Do not ablate more mass than there is or reverse the motion
        if (dm < -m_now) or (dm != dm):
            dm = -m_now

        if (dv < -v_now) or (dv != dv):
            dv = -v_now

        # The light and the ionization of the smallest substep are taken with the mass and the speed in its
        #   middle, from its changes
        if not resolved:
            m_mid = m_now + dm/2.0
            v_mid = v_now + dv/2.0
            de = 0.0
            dq = 0.0
            if v_mid >= v_kill:
                de = -luminousEfficiency(lum_eff_type, lum_eff, v_mid, m_mid) \
                    *(ablation_fraction*dm*v_mid*v_mid/2.0 + m_mid*v_mid*dv)
                dq = -ionizationEfficiency(v_mid)*ablation_fraction*dm/(mu*v_mid)

        energy += de
        electrons += dq

        m_now += dm
        v_now += dv
        l_now += dl
        t += h

        # Stop integrating a fragment which is ablated or stopped
        if (m_now <= 0) or (v_now <= 0):
            break

        # Try a longer substep after a resolved one
        if resolved:
            h *= 2.0

    return m_now - m, v_now - v, l_now, energy, electrons
