# Compute the powers of doubles with pow() of C, without the check of Python for a complex result
# cython: cpow=True
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



# Call pow() through a volatile exponent, which a C compiler cannot replace by x*x for the exponent 2, so that
#   the powers are the same doubles as x**y in Python, which calls pow() of the C library
cdef extern from *:
    """
    #include <math.h>
    static double wmplPyPow(double x, double y) {
        volatile double y_volatile = y;
        return pow(x, y_volatile);
    }
    """
    double wmplPyPow(double x, double y) nogil



def pythonPow(double x, double y):
    """ Compute x**y as the fragment loop does, the same as x**y of Python floats.

    Arguments:
        x: [float] Base.
        y: [float] Exponent.

    Return:
        [float] x**y.
    """

    return wmplPyPow(x, y)



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

    # Indicate that the fragment erodes into grains
    cdef public bint erosion_enabled

    # Indicate that the fragment is disrupted when the dynamic pressure exceeds the compressive strength
    cdef public bint disruption_enabled

    # Indicate that the fragment is still ablating
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

    # Last substep of the Runge-Kutta integration which resolved the motion and was not shortened by the end
    #   of a time step, from which the next time step starts (0 to start from the full time step)
    cdef public double rk4_substep

    # Number of grains the fragment stands for, as it was set (an int, or a float for a fraction), and as a
    #   double for the computations, converted once as Python converts it in every product
    cdef object n_grains_obj
    cdef double n_grains_double

    # Other attributes, among them 'const', which is a keyword of Cython
    cdef dict __dict__



    def __init__(self):
        """ Make a fragment with the default values of all attributes, which init() sets to the initial
            conditions of the simulation. The height, the drag coefficient and the zenith angle are 0 until
            init() sets them.
        """

        # Identifier of the fragment
        self.id = 0

        # Reference to the simulation constants, kept in the __dict__ as 'const' is a keyword of Cython (the
        #   __dict__ is reached through the Python object, as Cython 3.3 does not allow self.__dict__)
        (<object>self).__dict__['const'] = None

        # Shape-density coefficient, masses, density and ablation coefficient
        self.K = 0
        self.m_init = 0
        self.m = 0
        self.rho = 0
        self.sigma = 0

        # Velocity and its vertical and horizontal components
        self.v = 0
        self.vv = 0
        self.vh = 0

        # Drop due to gravity, and the length along the trajectory
        self.h_grav_drop_total = 0
        self.v_grav_drop = 0
        self.length = 0

        # Luminous intensity, electron line density and dynamic pressure
        self.lum = 0
        self.q = 0
        self.dyn_press = 0

        # Erosion coefficient
        self.erosion_coeff = 0

        # Start the Runge-Kutta integration from the full time step
        self.rk4_substep = 0

        # Indicators of how the erosion coefficient is set
        self.disruption_child = False
        self.erosion_height_change_applied = False

        # Grain mass distribution index and the mass range of the grains (kg)
        self.erosion_mass_index = 2.5
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10

        # The fragment does not erode, is not disrupted and is not active until it is set up
        self.erosion_enabled = False
        self.disruption_enabled = False
        self.active = False

        # A single body
        self.n_grains = 1

        # No grains eroded into the mass bins yet
        self.erosion_bin_counts = None
        self.erosion_bin_params = None

        # Type of the fragment, and the identifier of the fragmentation entry it is born out of
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
        """ Set the number of grains the fragment stands for.

        Arguments:
            value: [int or float] Number of grains, an int, or a float for a fraction.
        """

        # Keep the value as it was given, for the callers which format it as an integer
        self.n_grains_obj = value

        # Convert it once to a double for the computations of the fragment loop
        self.n_grains_double = value



    def init(self, sim_const, m, rho, v_init, sigma, gamma, zenith_angle, erosion_mass_index, \
        erosion_mass_min, erosion_mass_max):
        """ Set the initial conditions of the fragment at the beginning of the simulation, and activate it.

        Arguments:
            sim_const: [Constants object] Simulation constants, kept as the attribute 'const'.
            m: [float] Initial mass (kg).
            rho: [float] Density (kg/m^3).
            v_init: [float] Initial velocity (m/s).
            sigma: [float] Ablation coefficient (s^2/m^2).
            gamma: [float] Drag coefficient.
            zenith_angle: [float] Zenith angle of the trajectory (rad).
            erosion_mass_index: [float] Grain mass distribution index.
            erosion_mass_min: [float] Smallest grain mass (kg).
            erosion_mass_max: [float] Largest grain mass (kg).

        Return:
            None
        """

        # Keep the reference to the simulation constants
        (<object>self).__dict__['const'] = sim_const

        # Set the initial mass, height, density, velocity and the coefficients
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

        # Set the mass distribution of the grains
        self.erosion_mass_index = erosion_mass_index
        self.erosion_mass_min = erosion_mass_min
        self.erosion_mass_max = erosion_mass_max

        # Compute velocity components
        self.vv = -v_init*cos(zenith_angle)
        self.vh = v_init*sin(zenith_angle)

        # Activate the fragment as a single body
        self.active = True
        self.n_grains = 1



    def updateShapeDensityCoeff(self):
        """ Update the value of the shape-density coefficient. """

        # Compute the coefficient from the drag coefficient, the shape factor and the density
        self.K = self.gamma*(<object>self).__dict__['const'].shape_factor*pow(self.rho, -2/3.0)



    def toDict(self):
        """ Return all attributes of the fragment as a dictionary, as the __dict__ of the Python class it
            replaced, e.g. to save the fragment as JSON. The __dict__ of this class only has 'const' and the
            attributes added by callers.

        Return:
            frag_dict: [dict] Names and values of the attributes.
        """

        # Take the C fields in the order of the attributes of the Python class
        frag_dict = {
            'id': self.id,
            'const': (<object>self).__dict__.get('const'),
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
            'rk4_substep': self.rk4_substep,
            'h': self.h,
            'gamma': self.gamma,
            'zenith_angle': self.zenith_angle
            }

        # Add the attributes added by callers
        frag_dict.update((<object>self).__dict__)

        return frag_dict



    def spawn_child(self):
        """ Create a child of the Fragment instance. Copy over the reference to the shared 'const' object
            and copy values of all other attributes. Note: if a mutable attribute that is not shared across
            Fragment instances is added, this function will need to be revised.

        Return:
            child: [Fragment object] A new fragment of the same class with the values of this one.
        """

        # Make a new fragment of the same class, without setting the default values
        cdef FragmentCy child = type(self).__new__(type(self))

        # Copy the values of all C fields
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
        child.rk4_substep = self.rk4_substep
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

        # Share the reference to the constants, and copy the attributes added by callers
        (<object>child).__dict__.update((<object>self).__dict__)

        # The child starts without grains of its own to release, and does not share the list of the parent
        child.erosion_bin_counts = None
        child.erosion_bin_params = None

        return child



class Fragment(FragmentCy):
    """ A fragment of the meteoroid (see FragmentCy), with the attribute 'complex', which indicates that the
        fragment is born out of complex fragmentation.
    """



    def _getComplex(self):
        """ Return True if the fragment is born out of complex fragmentation. """

        return self.is_complex



    def _setComplex(self, value):
        """ Set if the fragment is born out of complex fragmentation.

        Arguments:
            value: [bool] True if the fragment is born out of complex fragmentation.
        """

        self.is_complex = value



    # Indicate that this is born out of complex fragmentation (kept in the C field is_complex)
    complex = property(_getComplex, _setComplex)



    def __reduce__(self):
        """ Pickle the fragment by the names and the values of all of its attributes (see toDict()), as the
            Python class it replaced was pickled. The state which Cython would pickle is the tuple of the C
            fields, with a checksum of their names which has to match when the fragment is loaded, so a
            fragment pickled before a field was added could not be loaded any more. A fragment pickled by
            names is loaded by any version of the class, and a field which it does not have keeps its default
            value.

        Return:
            (cls, args, state): [tuple] The class, no arguments for making a new fragment, and the dictionary
                of the attributes, which __setstate__() sets.
        """

        return (type(self), (), self.toDict())



    def __setstate__(self, state):
        """ Restore a pickled fragment, also one pickled by an older version of wmpl, in which Fragment was a
            Python class with all attributes in its __dict__.

        Arguments:
            state: [dict or tuple] The attributes of the fragment by name (see __reduce__()), which is also
                the __dict__ of a fragment pickled as a Python class, or the tuple of the C fields of a
                fragment pickled by Cython before __reduce__() was added.
        """

        # Set every attribute by its name
        if isinstance(state, dict):
            for name, value in state.items():
                setattr(self, name, value)

        # Restore the C fields and the __dict__ of a fragment pickled by Cython
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
cdef void ablationStepRK4C(double dt, double K, double sigma, double m, double v, double length, \
    double h_init, double cos_zc, double r_earth, double h_drop, AtmDensityTableCy atm_table, \
    const double[::1] dens_co, double ablation_fraction, int lum_eff_type, double lum_eff, double mu, \
    double v_kill, double *out, double *h_substep):
    """ Integrate the mass, the speed and the length of a fragment over one time step, as ablationStepRK4,
        but with the cosine of the zenith angle, starting from the given substep, and give the results through
        pointers, so that the fragment loop calls it without making Python objects.

    Arguments:
        dt: [double] Time step (s).
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
        out: [double *] Array of 5 doubles, set to the change of the mass (kg), the change of the speed
            (m/s), the change of the length (m), the radiated energy (J) and the electron line density
            multiplied by the time step (s/m), as returned by ablationStepRK4.
        h_substep: [double *] Substep to start with (s), the full time step if 0, set to the last substep
            which resolved the motion and was not shortened by the end of the time step (0 if there was
            none).

    Return:
        None
    """


    # Smallest substep, to which a substep is halved at most
    cdef double dt_min = dt/RK4_MAX_SUBSTEPS

    # Time integrated so far, and the substep, starting from the full time step
    cdef double t = 0.0
    cdef double h = dt

    # Mass, speed and length at the current time
    cdef double m_now = m
    cdef double v_now = v
    cdef double l_now = 0.0

    # Changes of the mass, the speed and the length over a substep
    cdef double dm = 0.0
    cdef double dv = 0.0
    cdef double dl = 0.0

    # Radiated energy and electrons over the step, and over a substep
    cdef double energy = 0.0
    cdef double electrons = 0.0
    cdef double de = 0.0
    cdef double dq = 0.0

    # Mass and speed in the middle of the smallest substep, and the air density at its start
    cdef double m_mid, v_mid
    cdef double rho_atm

    # Last substep which resolved the motion and was not shortened by the end of the time step
    cdef double h_resolved = 0.0

    # Indicate that a substep changed the mass and the speed by at most RK4_MAX_CHANGE, and that it was
    #   shortened to end at the end of the time step
    cdef bint resolved, shortened

    # Start from the substep which resolved the motion in the previous time step
    if (h_substep[0] > 0) and (h_substep[0] < dt):
        h = h_substep[0]

    # A fragment without mass or speed does not change
    if (m <= 0) or (v <= 0):
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        out[4] = 0.0
        return

    # Integrate the substeps until the end of the time step
    while t < dt:

        # Do not step past the end of the time step
        shortened = h > dt - t
        if shortened:
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

        # Add up the light and the ionization of the substep
        energy += de
        electrons += dq

        # Move the fragment to the end of the substep
        m_now += dm
        v_now += dv
        l_now += dl
        t += h

        # Remember the substep for the next time step
        if resolved and (not shortened):
            h_resolved = h

        # Stop integrating a fragment which is ablated or stopped
        if (m_now <= 0) or (v_now <= 0):
            break

        # Try a longer substep after a resolved one, if it changed the mass and the speed by at most half of
        #   RK4_MAX_CHANGE, as the change grows in proportion to the substep and a longer one would fail
        if resolved and (-dm <= 0.5*RK4_MAX_CHANGE*(m_now - dm)) and (-dv <= 0.5*RK4_MAX_CHANGE*(v_now - dv)):
            h *= 2.0

    # Give the substep to start the next time step with
    h_substep[0] = h_resolved

    # Give the changes over the time step
    out[0] = m_now - m
    out[1] = v_now - v
    out[2] = l_now
    out[3] = energy
    out[4] = electrons



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

    # Results of the step, written by the C function
    cdef double out[5]

    # Start from the full time step
    cdef double h_substep = 0.0

    # Integrate the step with the cosine of the zenith angle
    ablationStepRK4C(dt, K, sigma, m, v, length, h_init, cos(zenith_angle), r_earth, h_drop, atm_table, \
        dens_co, ablation_fraction, lum_eff_type, lum_eff, mu, v_kill, out, &h_substep)

    return out[0], out[1], out[2], out[3], out[4]



@cython.cdivision(True)
cpdef tuple ablationStepRK4Substep(double dt, double K, double sigma, double m, double v, double length, \
    double h_init, double zenith_angle, double r_earth, double h_drop, AtmDensityTableCy atm_table, \
    const double[::1] dens_co, double ablation_fraction, int lum_eff_type, double lum_eff, double mu, \
    double v_kill, double h_substep):
    """ Integrate a fragment over one time step as ablationStepRK4, starting from the given substep, as the
        fragment loop does with the substep it remembers for every fragment.

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
        h_substep: [double] Substep to start with (s), the full time step if 0.

    Return:
        (dm, dv, dl, energy, electrons, h_substep):
            - dm: [double] Change of the mass over the step (kg).
            - dv: [double] Change of the speed by the air over the step (m/s).
            - dl: [double] Change of the length over the step (m).
            - energy: [double] Energy radiated over the step (J).
            - electrons: [double] Electron line density multiplied by the time step (s/m).
            - h_substep: [double] Last substep which resolved the motion and was not shortened by the end of
                the time step (s), 0 if there was none.
    """

    # Results of the step, written by the C function
    cdef double out[5]

    # Integrate the step with the cosine of the zenith angle, starting from the given substep
    ablationStepRK4C(dt, K, sigma, m, v, length, h_init, cos(zenith_angle), r_earth, h_drop, atm_table, \
        dens_co, ablation_fraction, lum_eff_type, lum_eff, mu, v_kill, out, &h_substep)

    return out[0], out[1], out[2], out[3], out[4], h_substep



@cython.boundscheck(False)
@cython.wraparound(False)
def ablateAllCy(list fragments, const_obj, bint compute_wake, wake_heights_queue, \
    AtmDensityTableCy atm_table, const double[::1] dens_co, hooks):
    """ Perform single body ablation of all fragments over one time step, the loop over the fragments of
        MetSimErosion.ablateAll in Cython. The numbers of the fragments are read and written as C fields, and
        the time step is integrated without making Python objects. The rare events (the erosion into grains,
        the disruption, the fragmentation entries, the end of a fragment and the wake) are computed by the
        Python functions of the module 'hooks', looked up at every call, so that a function replaced in the
        module (e.g. by a test) is used.

    Arguments:
        fragments: [list] A list of Fragment instances.
        const_obj: [Constants object] Simulation constants.
        compute_wake: [bool] If True, the wake profile will be computed.
        wake_heights_queue: [list] A list of heights at which the wake should be computed (m), or None.
        atm_table: [AtmDensityTableCy] Atmosphere density table, or None to use the density polynomial.
        dens_co: [ndarray] The 7 coefficients of the density polynomial, used without the table.
        hooks: [module] The module MetSimErosion, which has the functions of the rare events.

    Return:
        (fragments, const_obj, luminosity_total, luminosity_main, luminosity_eroded, electron_density_total,
            tau_total, tau_main, tau_eroded, brightest_height, brightest_length, brightest_vel,
            leading_frag_height, leading_frag_length, leading_frag_vel, leading_frag_dyn_press,
            mass_total_active, main_mass, main_height, main_length, main_vel, main_dyn_press, wake):
            - fragments: [list] Fragments after the step, with the new ones and without the dead grains.
            - const_obj: [Constants object] Simulation constants, with the time of the step added.
            - luminosity_total: [float] Luminosity of all fragments (W).
            - luminosity_main: [float] Luminosity of the main fragment (W).
            - luminosity_eroded: [float] Luminosity of the eroded and disrupted fragments (W).
            - electron_density_total: [float] Electron line density of all fragments (1/m).
            - tau_total: [float] Luminous efficiency of all fragments, weighted by their luminosity.
            - tau_main: [float] Luminous efficiency of the main fragment.
            - tau_eroded: [float] Luminous efficiency of the eroded and disrupted fragments, weighted by their
                luminosity.
            - brightest_height: [float] Height of the brightest fragment (m).
            - brightest_length: [float] Length of the brightest fragment along the path (m).
            - brightest_vel: [float] Velocity of the brightest fragment (m/s).
            - leading_frag_height: [float] Height of the leading fragment (m), None if no fragment was
                active.
            - leading_frag_length: [float] Length of the leading fragment along the path (m), None if no
                fragment was active.
            - leading_frag_vel: [float] Velocity of the leading fragment (m/s), None if no fragment was
                active.
            - leading_frag_dyn_press: [float] Dynamic pressure on the leading fragment (Pa), None if no
                fragment was active.
            - mass_total_active: [float] Mass of all active fragments, also in their mass bins (kg).
            - main_mass: [float] Mass of the main fragment (kg).
            - main_height: [float] Height of the main fragment (m).
            - main_length: [float] Length of the main fragment along the path (m).
            - main_vel: [float] Velocity of the main fragment (m/s).
            - main_dyn_press: [float] Dynamic pressure on the main fragment (Pa).
            - wake: [Wake object] The wake, None if it was not computed in this step.
    """

    # Keep track of the total luminosity
    cdef double luminosity_total = 0.0

    # Keep track of the total luminosity weighted lum eff
    cdef double tau_total = 0.0

    # Keep track of the luminosity of the main fragment
    cdef double luminosity_main = 0.0

    # Keep track of the luminosity weighted lum eff of the main fragment
    cdef double tau_main = 0.0

    # Keep track of the luminosity of eroded and disrupted fragments
    cdef double luminosity_eroded = 0.0

    # Keep track of the luminosity weighted lum eff of eroded and disrupted fragments
    cdef double tau_eroded = 0.0

    # Keep track of the total electron density
    cdef double electron_density_total = 0.0

    # Keep track of parameters of the brightest fragment
    cdef double brightest_height = 0.0
    cdef double brightest_length = 0.0
    cdef double brightest_lum = 0.0
    cdef double brightest_vel = 0.0

    # Keep track of the the main fragment parameters
    cdef double main_mass = 0.0
    cdef double main_height = 0.0
    cdef double main_length = 0.0
    cdef double main_vel = 0.0
    cdef double main_dyn_press = 0.0

    # Total active mass (grain-weighted), accumulated in the loop over the fragments
    cdef double mass_total_active = 0.0

    # Fragments made in this step, the fragments active at its start, and the fragmentation entries by id
    cdef list frag_children_all = []
    cdef list fragments_active_at_tick_start
    cdef dict fragmentation_entry_by_id

    # Fragment in the loop, the leading fragment, and the indices in the fragments and in a light curve
    cdef FragmentCy frag
    cdef FragmentCy leading_frag = None
    cdef Py_ssize_t i, n_data

    # Results of the Runge-Kutta integration of a fragment, written by ablationStepRK4C
    cdef double rk4_out[5]

    # Substep of the Runge-Kutta integration which a fragment starts the step with
    cdef double h_substep

    # Values of a fragment in the step, as C numbers
    cdef double rho_atm, erosion_coeff, m_start, v_start, ablation_fraction, mass_loss_total, velocity_change
    cdef double length_change, energy_step, electrons_step, mass_loss_erosion, mass_loss_ablation, m_new
    cdef double deceleration_total, gv, cos_z, sin_z_sq, sin_z, accel_along, av, ah, tau, lum, q, dyn_press
    cdef bint release_grains

    # Constants which do not change within a time step, read once
    cdef double dt = const_obj.dt
    cdef double h_init = const_obj.h_init
    cdef double zenith_angle = const_obj.zenith_angle
    cdef double cos_zc = cos(zenith_angle)
    cdef double r_earth = const_obj.r_earth
    cdef double m_kill = const_obj.m_kill
    cdef double v_kill = const_obj.v_kill
    cdef double h_kill = const_obj.h_kill
    cdef double len_kill = const_obj.len_kill
    cdef int lum_eff_type = const_obj.lum_eff_type
    cdef double lum_eff = const_obj.lum_eff
    cdef double mu = const_obj.mu
    cdef double erosion_height_start = const_obj.erosion_height_start
    cdef bint erosion_on = const_obj.erosion_on
    cdef bint disruption_on = const_obj.disruption_on
    cdef bint fragmentation_on = const_obj.fragmentation_on
    cdef bint show_individual_lcs = const_obj.fragmentation_show_individual_lcs
    cdef double G0 = hooks.G0
    cdef object dens_co_poly = const_obj.dens_co

    # Distance of the beginning from the centre of the Earth times the cosine of the zenith angle, from which
    #   the local zenith angle of the path is computed
    cdef double r_cos_zc = (h_init + r_earth)*cos(zenith_angle)

    # Snapshot which fragments were active at the start of this tick, before any of them can be killed below,
    #   for the leading fragment, so a fragment that dies partway through this tick still counts with its
    #   just-computed position for the tick it died on
    fragments_active_at_tick_start = [frag_obj for frag_obj in fragments if (<FragmentCy?>frag_obj).active]

    # Index the complex fragmentation entries by id once, so that the light curve of a fragment finds its
    #   entry without a scan of all entries
    if show_individual_lcs:
        fragmentation_entry_by_id = {frag_entry.id: frag_entry for frag_entry \
            in const_obj.fragmentation_entries}

    # Go through all active fragments, also the ones added to the list in the loop, as a Python loop over the
    #   list would
    i = 0
    while i < len(fragments):

        # Take the fragment, which must be a Fragment (raises TypeError otherwise)
        frag = <FragmentCy?>fragments[i]
        i += 1

        # Skip the fragment if it's not active
        if not frag.active:
            continue

        # Get atmosphere density for the given height
        if atm_table is None:
            rho_atm = atmDensityPoly(frag.h, dens_co_poly)
        else:
            rho_atm = atm_table.density(frag.h)

        # Take the erosion coefficient of the fragment if it erodes
        if frag.erosion_enabled and (frag.erosion_coeff > 0):
            erosion_coeff = frag.erosion_coeff
        else:
            erosion_coeff = 0.0

        # Keep the mass and the speed at the beginning of the step for the luminous efficiency
        m_start = frag.m
        v_start = frag.v

        # Take the fraction of the mass loss which is ablated, the rest is eroded
        if (frag.sigma + erosion_coeff) > 0:
            ablation_fraction = frag.sigma/(frag.sigma + erosion_coeff)
        else:
            ablation_fraction = 1.0

        # Integrate the mass, the speed and the length over the step together, with the air density at the
        #   height of every stage, as the density changes by about 1% in a step of a fast meteor, and add up
        #   the radiated energy and the ionization over the step, starting from the substep which resolved the
        #   motion in the previous step
        h_substep = frag.rk4_substep
        ablationStepRK4C(dt, frag.K, frag.sigma + erosion_coeff, frag.m, frag.v, frag.length, h_init, \
            cos_zc, r_earth, frag.h_grav_drop_total, atm_table, dens_co, ablation_fraction, lum_eff_type, \
            lum_eff, mu, v_kill, rk4_out, &h_substep)

        # Remember the substep for the next time step of the fragment
        frag.rk4_substep = h_substep

        # Take the results of the integration from the output array
        mass_loss_total = rk4_out[0]
        velocity_change = rk4_out[1]
        length_change = rk4_out[2]
        energy_step = rk4_out[3]
        electrons_step = rk4_out[4]

        # Split the mass loss into erosion and ablation, which have the same form, so that a fragment which
        #   does not erode loses no mass to erosion
        if erosion_coeff > 0:
            mass_loss_erosion = mass_loss_total*erosion_coeff/(frag.sigma + erosion_coeff)
        else:
            mass_loss_erosion = 0.0

        mass_loss_ablation = mass_loss_total - mass_loss_erosion

        # Compute new mass
        m_new = frag.m + mass_loss_total

        # Take the average deceleration by the air over the step
        deceleration_total = velocity_change/dt

        # Compute g at given height (the powers are taken as in Python, see wmplPyPow)
        gv = G0/wmplPyPow(1 + frag.h/r_earth, 2.0)

        # Compute the local zenith angle of the straight path, at the distance of the point on the path from
        #   the centre of the Earth without the drop
        cos_z = (r_cos_zc - frag.length)/(frag.h + frag.h_grav_drop_total + r_earth)
        sin_z_sq = 1.0 - cos_z*cos_z
        sin_z = sqrt(sin_z_sq) if (sin_z_sq > 0) else 0.0

        # Gravity along the path changes the speed, and the drag is only the deceleration by the air
        accel_along = deceleration_total + gv*cos_z

        # Vertical and horizontal components of a
        av = -accel_along*frag.vv/frag.v + frag.vh*frag.v/(r_earth + frag.h)
        ah = -accel_along*frag.vh/frag.v - frag.vv*frag.v/(r_earth + frag.h)

        # Gravity across the path drops the meteoroid below it, and the drag slows the speed of the drop as it
        #   slows the meteoroid
        frag.v_grav_drop += (gv*sin_z + deceleration_total*frag.v_grav_drop/frag.v)*dt

        # Track the total drop of the height, the vertical part of the drop across the path
        frag.h_grav_drop_total += frag.v_grav_drop*sin_z*dt

        # Add the length gained from gravity along the path in this step
        length_change += 0.5*gv*cos_z*wmplPyPow(dt, 2.0)

        # Update the velocity
        frag.vv -= av*dt
        frag.vh -= ah*dt
        frag.v = sqrt(wmplPyPow(frag.vh, 2.0) + wmplPyPow(frag.vv, 2.0))

        # Only allow the meteoroid to go down, and stop the ablation if it stars going up
        if frag.vv > 0:

            frag.vv = 0

            # Setting the height to zero will stop the ablation during the if catch below
            frag.h = 0

        # Update length along the track
        frag.length += length_change

        # Update the mass
        frag.m = m_new

        # Compute the height taking the curvature of the Earth and the gravity drop into account (the same
        #   arithmetic as heightCurvature)
        frag.h = sqrt(wmplPyPow(h_init + r_earth, 2.0) - 2*frag.length*cos(zenith_angle)*(h_init + r_earth) \
            + wmplPyPow(frag.length, 2.0)) - r_earth
        frag.h -= frag.h_grav_drop_total

        # Take the luminous efficiency in the middle of the step, used to weight the efficiencies of the
        #   fragments in the output
        tau = luminousEfficiency(lum_eff_type, lum_eff, (v_start + frag.v)/2, (m_start + frag.m)/2)

        # Compute the luminosity for one grain/fragment, from the ablated mass and the deceleration by the
        #   air, as the energy radiated over the step (from its substeps) divided by the step
        lum = energy_step/dt

        # Compute the electron line density
        q = electrons_step/dt

        # Compute the total luminosity
        frag.lum = lum*frag.n_grains_double

        # Compute the total electron line density
        frag.q = q*frag.n_grains_double

        # Keep track of the total luminosity across all fragments
        luminosity_total += frag.lum

        # Keep track of the total number of produced electrons
        electron_density_total += frag.q

        # Keep track of the total luminosity weighted lum eff
        tau_total += tau*frag.lum

        # Compute aerodynamic loading on the grain (always assume Gamma = 1.0)
        dyn_press = 1.0*rho_atm*wmplPyPow(frag.v, 2.0)
        frag.dyn_press = dyn_press

        # Keep track of the parameters of the main fragment
        if frag.main:
            luminosity_main = frag.lum
            tau_main = tau
            main_mass = frag.m
            main_height = frag.h
            main_length = frag.length
            main_vel = frag.v
            main_dyn_press = dyn_press

        # Keep track of the brightest fragment (checked before the kill check below, so that a fragment
        #   counts also on the step on which it dies)
        if frag.lum > brightest_lum:
            brightest_lum = frag.lum
            brightest_height = frag.h
            brightest_length = frag.length
            brightest_vel = frag.v

        # If the fragment is done, stop ablating
        if (frag.m <= m_kill) or (frag.v < v_kill) or (frag.h < h_kill) or (frag.lum < 0) \
            or ((len_kill > 0) and (frag.length > len_kill)):

            # Grains released at the speed and the height of a fragment which is too slow, too low or too far
            #   would be killed in their first step, which they would take outside of the range of the
            #   luminous efficiency, so they are released only by a fragment which has run out of mass
            release_grains = (frag.v >= v_kill) and (frag.h >= h_kill) \
                and not ((len_kill > 0) and (frag.length > len_kill))

            # Release the mass eroded in this step, as the grains are made below only for living fragments
            if release_grains and frag.erosion_enabled and (abs(mass_loss_erosion) > 0):
                grain_children = hooks.erodeIntoGrains(const_obj, frag, abs(mass_loss_erosion))
                const_obj.n_active += len(grain_children)
                frag_children_all += grain_children

            # Deactivate the fragment, which releases the grains in its mass bins
            frag_children_all += hooks.killFragment(const_obj, frag, release_grains=release_grains)

            continue

        # For fragments born out of complex fragmentation, keep track of their luminosity
        if not frag.main:

            # Keep track of magnitudes of complex fragmentation fragments (the light curves of the
            #   individual entries are only computed if requested)
            if frag.is_complex:

                # Add the light of the fragment to the light curve of its fragmentation entry, of the grains
                #   or of the main fragments of the entry, at the current time
                if show_individual_lcs:

                    # Find the corresponding fragmentation entry
                    frag_entry = fragmentation_entry_by_id.get(frag.complex_id)

                    if frag_entry is not None:

                        # Take the light curve of the grains of the entry
                        if frag.grain:
                            time_data = frag_entry.grains_time_data
                            luminosity_data = frag_entry.grains_luminosity
                            tau_over_lum_data = frag_entry.grains_tau_over_lum

                        # Take the light curve of the main fragments of the entry
                        else:
                            time_data = frag_entry.main_time_data
                            luminosity_data = frag_entry.main_luminosity
                            tau_over_lum_data = frag_entry.main_tau_over_lum

                        n_data = len(time_data)
                        total_time = const_obj.total_time

                        # Add the current integration time, if the last time entry is not the current time
                        if (not n_data) or (total_time != time_data[n_data - 1]):
                            time_data.append(total_time)
                            luminosity_data.append(frag.lum)
                            tau_over_lum_data.append(tau*frag.lum)

                        # Add to the total luminosity at the current time step that's already been added
                        else:
                            luminosity_data[n_data - 1] += frag.lum
                            tau_over_lum_data[n_data - 1] += tau*frag.lum

            # Keep track of luminosity of eroded and disrupted fragments ejected directly from the main
            #   fragment (always tracked, also without the light curves of the individual entries)
            else:
                luminosity_eroded += frag.lum
                tau_eroded += tau*frag.lum

        # Check if the erosion should start, given the height, for the fragments not born out of complex
        #   fragmentation
        if (not frag.is_complex) and (frag.h < erosion_height_start) and frag.erosion_enabled and erosion_on:

            # Turn on the erosion of the fragment (disruption daughters keep the fixed, disruption-specific
            #   erosion coefficient they were assigned at birth, and the coefficient set by a fragmentation
            #   entry is kept too)
            if (not frag.disruption_child) and (not frag.erosion_coeff_fixed):
                frag.erosion_coeff = hooks.getErosionCoeff(const_obj, frag.h)

            # Update the main fragment physical parameters if it is changed after erosion coefficient change
            #   (only once, so that a later change of sigma, e.g. by a complex fragmentation "A" event, is
            #   kept)
            if frag.main and (not frag.erosion_height_change_applied) \
                and (const_obj.erosion_height_change >= frag.h):

                # Update the density
                frag.rho = const_obj.erosion_rho_change
                frag.updateShapeDensityCoeff()

                # Update the ablation coeff
                frag.sigma = const_obj.erosion_sigma_change

                frag.erosion_height_change_applied = True

        # Create grains for erosion-enabled fragments
        if frag.erosion_enabled:

            # Generate new grains if there is some mass to distribute
            if abs(mass_loss_erosion) > 0:

                grain_children = hooks.erodeIntoGrains(const_obj, frag, abs(mass_loss_erosion))
                const_obj.n_active += len(grain_children)
                frag_children_all += grain_children

                # Record physical parameters at the beginning of erosion for the main fragment
                if frag.main:
                    if const_obj.erosion_beg_vel is None:
                        const_obj.erosion_beg_vel = frag.v
                        const_obj.erosion_beg_mass = frag.m
                        const_obj.erosion_beg_dyn_press = dyn_press

                    # Record the mass when erosion is changed
                    elif (const_obj.erosion_height_change >= frag.h) \
                        and (const_obj.mass_at_erosion_change is None):

                        const_obj.mass_at_erosion_change = frag.m

        # Disrupt the fragment if the dynamic pressure exceeds its strength
        if frag.disruption_enabled and disruption_on:
            if dyn_press > const_obj.compressive_strength:

                # Break the fragment into fragments and grains, and keep them with the other children
                frag_children_all += hooks.disruptFragment(const_obj, frag)

        # Handle complex fragmentation and status changes of the main fragment
        if frag.main and fragmentation_on:

            # Apply the fragmentation entries whose height the main fragment has passed
            hooks.applyFragmentationEntries(const_obj, frag, fragments, frag_children_all, dyn_press)

        # If the fragment is done, stop ablating
        if frag.m <= m_kill:
            frag_children_all += hooks.killFragment(const_obj, frag)
            continue

        # Add the grain-weighted mass of the fragment which survived this tick to the total active mass (the
        #   killed and disrupted fragments skipped this above)
        mass_total_active += frag.m*frag.n_grains_double

        # Add the mass the fragment eroded into its mass bins and did not release as grains yet
        if frag.erosion_bin_counts is not None:
            mass_total_active += hooks.erosionBinMass(frag)

    # Track the leading fragment, the first of the longest among those active at the start of the tick (as
    #   max() of Python picks it), so that a fragment which died in this tick is still a candidate for it
    for frag_obj in fragments_active_at_tick_start:
        frag = <FragmentCy?>frag_obj
        if (leading_frag is None) or (frag.length > leading_frag.length):
            leading_frag = frag

    # Take the parameters of the leading fragment
    if leading_frag is not None:
        leading_frag_length = leading_frag.length
        leading_frag_height = leading_frag.h
        leading_frag_vel = leading_frag.v
        leading_frag_dyn_press = leading_frag.dyn_press

    else:
        leading_frag_length = None
        leading_frag_height = None
        leading_frag_vel = None
        leading_frag_dyn_press = None

    ### Compute the wake profile ###

    # Compute the wake around the leading fragment, if it is requested at this time step
    wake = hooks.computeWake(const_obj, fragments, leading_frag_length, leading_frag_height, compute_wake, \
        wake_heights_queue)

    ### ###

    # Add generated fragment children to the list of fragments
    fragments += frag_children_all

    # Add the grain-weighted mass of the fragments and grains made in this tick, which are all active
    for frag_obj in frag_children_all:
        mass_total_active += (<FragmentCy?>frag_obj).m*(<FragmentCy?>frag_obj).n_grains_double

    # Drop fully-ablated grains from the fragment list so that the next ticks do not scan them again, but
    #   keep the dead non-grains (the main fragment and the complex fragmentation daughters), which
    #   runSimulation() scans after the run for the main fragment and the final masses of the fragmentations
    fragments = [frag_obj for frag_obj in fragments \
        if (<FragmentCy?>frag_obj).active or (not (<FragmentCy?>frag_obj).grain)]

    # Increment the running time
    const_obj.total_time += const_obj.dt

    # Weigh the tau by luminosity
    if luminosity_total > 0:
        tau_total /= luminosity_total
    else:
        tau_total = 0

    if luminosity_eroded > 0:
        tau_eroded /= luminosity_eroded
    else:
        tau_eroded = 0

    return fragments, const_obj, luminosity_total, luminosity_main, luminosity_eroded, \
        electron_density_total, tau_total, tau_main, tau_eroded, brightest_height, brightest_length, \
        brightest_vel, leading_frag_height, leading_frag_length, leading_frag_vel, leading_frag_dyn_press, \
        mass_total_active, main_mass, main_height, main_length, main_vel, main_dyn_press, wake
