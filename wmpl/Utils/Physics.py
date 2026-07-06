""" Implementations of frequently used equations. """

from __future__ import print_function, division, absolute_import


import numpy as np
import scipy.integrate

# Import the correct scipy.integrate.simpson function
try:
    from scipy.integrate import simps as simpson
except ImportError:
    from scipy.integrate import simpson as simpson

from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect


# Mapping of luminous efficiency model names to the integer lum_eff_type codes used by
# wmpl.MetSim.MetSimErosionCyTools.luminousEfficiency().
LUM_EFF_MODELS = {
    'constant':          0,
    'rc2001':            1,  # ReVelle & Ceplecha (2001), Type I
    'rc2001_t1':         1,
    'rc2001_t2':         2,
    'rc2001_t3':         3,
    'borovicka2013':     4,  # Borovicka et al. (2013), Kosice
    'kosice':            4,
    'camo':              5,  # CAMO faint meteors (Subasinghe 2018 & Brown 2020)
    'ceplecha_mccrosky': 6,  # Ceplecha & McCrosky (1976)
    'cm1976':            6,
    'borovicka2020':     7,  # Borovicka et al. (2020), Two strengths
    'pecina_ceplecha':   8,  # Pecina & Ceplecha (1983)
    'pc1983':            8,
}


def dynamicPressure(lat, lon, height, jd, velocity, gamma=1.0):
    """ Calculate dynamic pressure at the given point on meteor's trajectory. 

    Either a single set of values can be given (i.e. every argument is a float number), or all arguments 
    must be numpy arrays.
        
    Arguments:
        lat: [float] Latitude of the meteor (radians).
        lon: [float] Longitude of the meteor (radians).
        height: [float] Height of the meteor (meters).
        jd: [float] Julian date of the meteor.
        velocity: [float] Velocity of the meteor (m/s).

    Keyword arguments:
        gamma: [flot] Drag coefficient. 1 by defualt.


    Return:
        dyn_pressure: [float] Dynamic pressure in Pascals.

    """

    # Get the atmospheric densities at every heights
    atm_dens = getAtmDensity_vect(lat, lon, height, jd)

    # Calculate the dynamic pressure
    dyn_pressure = atm_dens*gamma*velocity**2

    return dyn_pressure


def dynamicMass(bulk_density, lat, lon, height, jd, velocity, decel, gamma=1.0, shape_factor=1.21):
    """ Calculate dynamic mass at the given point on meteor's trajectory. 
    
    Either a single set of values can be given (i.e. every argument is a float number), or all arguments 
    must be numpy arrays.
        
    Arguments:
        bulk_density: [float] Bulk density of the meteoroid in kg/m^3.
        lat: [float] Latitude of the meteor (radians).
        lon: [flaot] Longitude of the meteor (radians).
        height: [float] Height of the meteor (meters).
        jd: [float] Julian date of the meteor.
        velocity: [float] Velocity of the meteor (m/s).
        decel: [float] Deceleration in m/s^2.

    Keyword arguments:
        gamma: [flot] Drag coefficient. 1 by defualt.
        shape_factor: [float] Shape factory for the body. 1.21 (sphere) by default. Other values:
            - sphere      = 1.21
            - hemisphere  = 1.92
            - cube        = 1.0
            - brick 2:3:5 = 1.55

    Return:
        dyn_mass: [float] Dynamic mass in kg.


    """

    # Calculate the atmosphere density at the given point
    atm_dens = getAtmDensity_vect(lat, lon, height, jd)

    # Calculate the dynamic mass
    dyn_mass = (1.0/(bulk_density**2))*((gamma*shape_factor*(velocity**2)*atm_dens)/decel)**3

    return dyn_mass


def calcIntensity(mag_abs, P_0m=840.0):
    """ Calculate the radiated power (intensity) from absolute magnitudes.

    Arguments:
        mag_abs: [float or ndarray] Absolute magnitudes.

    Keyword arguments:
        P_0m: [float] Power of a zero-magnitude meteor (W). See calcRadiatedEnergy() for details.

    Return:
        intens: [float or ndarray] Radiated power (W).

    """

    return P_0m*10**(-0.4*mag_abs)


def calcRadiatedEnergy(time, mag_abs, P_0m=840.0):
    """ Calculate the radiated energy given the light curve.

    Arguments:
        time: [ndarray] Time of individual magnitude measurement (s).
        mag_abs: [nadrray] Absolute magnitudes (i.e. apparent meteor magnitudes @100km).

    Keyword arguments:
        P_0m: [float] Power output of a zero absolute magnitude meteor. 840W by default, as that is the R
            bandpass for a T = 4500K black body meteor. See: Weryk & Brown, 2013 - "Simultaneous radar and 
            video meteors - II. Photometry and ionisation" for more details.

    Return:
        initens_int: [float] Total radiated energy (J).

    """


    # Calculate the intensities from absolute magnitudes
    intens = calcIntensity(mag_abs, P_0m=P_0m)

    # # Interpolate I
    # intens_interpol = scipy.interpolate.PchipInterpolator(time, intens)

    # x_data = np.linspace(np.min(time), np.max(time), 1000)
    # plt.plot(x_data, intens_interpol(x_data))
    # plt.scatter(time, intens/(velocity**2))
    # plt.show()

    # Integrate the interpolated I
    #intens_int = intens_interpol.integrate(np.min(time), np.max(time))
    intens_int = simpson(intens, x=time)

    return intens_int


def calcMass(time, mag_abs, velocity, tau=0.007, P_0m=840.0, lum_eff_mass=-1, v_init=None):
    """ Calculates the mass of a meteoroid from the time and absolute magnitude.

    Arguments:
        time: [ndarray] Time of individual magnitude measurement (s).
        mag_abs: [ndarray] Absolute magnitudes (i.e. apparent meteor magnitudes @100km).
        velocity: [float or ndarray] Average velocity of the meteor, or velocity at every point of the meteor
            in m/s.

    Keyword arguments:
        tau: [float or int or str] Luminous efficiency selector.
            - float: constant luminous efficiency as a ratio (not percent!). 0.007 (i.e. 0.7%) by default
              (Ceplecha & McCrosky, 1976).
            - int: velocity-dependent lum_eff_type code passed to
              wmpl.MetSim.MetSimErosionCyTools.luminousEfficiency() (0 - constant, 1/2/3 - ReVelle &
              Ceplecha 2001 Type I/II/III, 4 - Borovicka 2013, 5 - CAMO, 6 - Ceplecha & McCrosky 1976,
              7 - Borovicka 2020, 8 - Pecina & Ceplecha 1983). tau is then computed per velocity point.
            - str: model name mapped to the above codes (see LUM_EFF_MODELS), e.g. 'rc2001'.
        P_0m: [float] Power output of a zero absolute magnitude meteor. 840W by default, as that is the R
            bandpass for a T = 4500K black body meteor. See: Weryk & Brown, 2013 - "Simultaneous radar and
            video meteors - II. Photometry and ionisation" for more details.
            NOTE: the panchromatic models (RC2001, Borovicka, Pecina-Ceplecha) are paired with a V-band
            zero-magnitude power of P_0m = 1500 W in Borovicka et al. (2022); set P_0m accordingly when
            using those models.
        lum_eff_mass: [float] Meteoroid mass (kg) used to evaluate the mass-dependent luminous efficiency
            models. Ignored for a float tau. If -1 (default), the mass is iterated to self-consistency
            (compute mass -> recompute tau -> repeat until convergence).
        v_init: [float or None] Pre-atmospheric velocity (m/s) used by the ReVelle & Ceplecha (2001)
            deceleration correction in the luminous efficiency calculation. Only affects the RC2001
            models (tau = 1, 2, 3 or 'rc2001*'). Default is None.
            - None: the deceleration correction is not applied.
            - -1: estimate the pre-atmospheric velocity as the median velocity over the first 25% of
              points (robust to measurement noise). Requires a velocity array.
            - >0: use the supplied value as the pre-atmospheric velocity.
            The correction is linearly tapered to zero for velocities within 0.1 km/s of v_init and
            disabled for velocities at or above it (see luminousEfficiency() for details).
            NOTE: the velocity array should represent a fitted deceleration curve (i.e. a physically
            consistent monotonic velocity evolution) rather than individual noisy measurements.

    Return:
        mass: [float] Photometric mass of the meteoroid in kg.

    """

    # Theory:
    # I = P_0m*10^(-0.4*M_abs)
    #
    # General (velocity and/or tau vary along the trajectory):
    #   M = integral(2*I/(tau*v^2) dt)
    #
    # For constant velocity and luminous efficiency this reduces to:
    #   M = (2/(tau*v^2))*integral(I dt)
    #
    # If tau is constant but velocity varies along the trajectory:
    #   M = (2/tau)*integral(I/v^2 dt)

    if len(time) != len(mag_abs):
        raise ValueError("time and mag_abs must have the same length.")

    # If a velocity array is given, it must have the same length as the photometric measurements
    if np.ndim(velocity) > 0 and len(velocity) != len(time):
        raise ValueError("Velocity array length ({:d}) does not match the number of time/magnitude "
            "measurements ({:d}).".format(len(velocity), len(time)))

    if lum_eff_mass < 0 and lum_eff_mass != -1:
        raise ValueError("lum_eff_mass must be >= 0, or -1 for self-consistent iteration.")

    # Constant luminous efficiency (tau given directly as a ratio).
    if not isinstance(tau, str) and not isinstance(tau, (bool, int)):

        # Constant velocity
        if np.ndim(velocity) == 0:
            intens_int = calcRadiatedEnergy(time, mag_abs, P_0m=P_0m)
            return (2.0/(tau*float(velocity)**2))*intens_int

        # Velocity array
        else:
            intens = calcIntensity(mag_abs, P_0m=P_0m)
            vel_arr = np.asarray(velocity, dtype=float)
            dm_dt = 2.0*intens/(tau*vel_arr**2)
            return simpson(dm_dt, x=time)

    # Velocity-dependent model: resolve the lum_eff_type code.
    if isinstance(tau, str):
        key = tau.strip().lower()
        if key not in LUM_EFF_MODELS:
            raise ValueError("Unknown luminous efficiency model: {!r}".format(tau))
        lum_eff_type = LUM_EFF_MODELS[key]
    else:
        lum_eff_type = int(tau)

    # Compute the luminous efficiency using an analytical velocity-dependent function.
    import pyximport
    pyximport.install(setup_args={'include_dirs': [np.get_include()]})
    from wmpl.MetSim.MetSimErosionCyTools import luminousEfficiency

    # Radiated power at every measurement point
    intens = calcIntensity(mag_abs, P_0m=P_0m)


    # Convert velocity to an array for the velocity-dependent luminous efficiency calculations.
    vel_arr = np.atleast_1d(np.asarray(velocity, dtype=float))

    # Resolve the pre-atmospheric velocity used by the ReVelle & Ceplecha (2001) deceleration
    # correction (only affects the RC2001 models, types 1-3)
    if v_init is None:

        # Disable the deceleration correction
        v_init_eff = -1.0

    elif v_init == -1:

        # A scalar velocity carries no deceleration information to estimate v_init from
        if np.ndim(velocity) == 0:
            raise ValueError("v_init=-1 requires a velocity array to estimate the pre-atmospheric "
                "velocity.")

        # Estimate the pre-atmospheric velocity as the median velocity over the first 25% of points,
        # which is robust to measurement noise (unlike e.g. the maximum velocity)
        v_init_eff = float(np.median(vel_arr[:max(1, len(vel_arr)//4)]))

    elif v_init > 0:

        v_init_eff = float(v_init)

        # The deceleration correction is tapered/disabled for velocities within 0.1 km/s of v_init,
        # so warn if the given v_init leaves the fastest points effectively uncorrected
        if v_init_eff < np.max(vel_arr) + 100:
            print("WARNING: v_init = {:.1f} m/s is within 100 m/s of the maximum given velocity. "
                "The ReVelle & Ceplecha (2001) deceleration correction will be tapered or disabled "
                "at the fastest points.".format(v_init_eff))

    else:
        raise ValueError("v_init must be None, -1 (automatic estimate), or > 0, "
            "got {}".format(v_init))

    def _mass_for(mass_assumed):
        """ Compute the photometric mass given an assumed meteoroid mass for the lum. eff. models. """

        # Vectorize the scalar Cython call over the velocity points (lum_eff arg only used for type 0)
        tau_arr = np.array([luminousEfficiency(lum_eff_type, 0.7, v, mass_assumed, v_init_eff) for v in vel_arr])

        # Integrate the instantaneous mass-loss rate dm/dt = 2*I/(tau*v^2)
        dm_dt = 2.0*intens/(tau_arr*vel_arr**2)

        return simpson(dm_dt, x=time)


    # If the mass is given explicitly, evaluate the models at that mass.
    if lum_eff_mass >= 0:
        return _mass_for(lum_eff_mass)

    # Otherwise iterate to a self-consistent mass: the luminous efficiency depends on the mass, which
    # is the quantity being solved for. Convergence is measured as a fraction of the first mass
    # estimate, so it behaves consistently across orders of magnitude (e.g. ~1e-7 kg meteoroids) and
    # does not divide by an evolving (possibly tiny) iterate.
    rel_tol = 1e-4

    # Seed the iteration with the luminous efficiency evaluated at a mass of 1 kg. The mass term of
    # the models is bounded, so this seed lands within a small factor of the true mass at any scale.
    mass = _mass_for(1.0)
    mass_ref = abs(mass)

    for _ in range(50):
        mass_new = _mass_for(mass)
        if abs(mass_new - mass) < rel_tol*mass_ref:
            mass = mass_new
            break
        mass = mass_new
    else:
        print("WARNING: luminous efficiency self-consistency iteration did not converge "
            "after 50 iterations.")

    return mass


def calcKB(lat, lon, ht_beg, jd, v, zenith_angle, gmn_correction=False):
    """ Calculate the Celpecha (1958) KB parameter.
    
    Arguments:
        lat: [float] Latitude of the meteor (radians).
        lon: [float] Longitude of the meteor (radians).
        ht_beg: [float] Beginning height of the meteor (meters).
        jd: [float] Julian date of the meteor.
        v: [float] Velocity of the meteor (m/s).
        zenith_angle: [float] Zenith angle of the meteor (radians).

    Keyword arguments:
        gmn_correction: [bool] Apply the correction for GMN data. False by default.
            The correction is described in:
            Cordonnier et al. (2024) - Meteor persisent trains

    Return:
        kb: [float] KB parameter.
    
    """

    # Get the atmospheric density at the meteor begin height
    rho_a = getAtmDensity_vect(lat, lon, ht_beg, jd)

    # Convert atmospheric density from kg/m^3 to g/cm^3
    rho_a /= 1000.0

    # Convert speed from m/s to cm/s
    v *= 100.0

    # Calculate the KB parameter
    kb = np.log10(rho_a) + 2.5*np.log10(v) - 0.5*np.log10(np.cos(zenith_angle))

    # Apply the GMN correction
    if gmn_correction:
        kb -= 0.1

    return kb


def calcPf(p_max, zangle, mass, v_0):
    """ The Borovicka et al. (2022) Pf parameter calculation.
    
    Arguments:
        p_max: [float] Maximum dynamic pressure (Pa).
        zangle: [float] Zenith angle (radians).
        mass: [float] Mass of the meteoroid (kg).
        v_0: [float] Initial velocity of the meteoroid (m/s).

    Return:
        (pf, pf_category): [tuple]
            - pf: [float] Pf parameter.
            - pf_category: [int] Pf category (1 to 5).

    """

    ### Convert units to match the paper ###
    
    # Pressue in MPa
    p_max /= 1e6

    # Entry speed in km/s
    v_0 /= 1e3

    ### ###

    # Calculate the Pf parameter
    pf = 100*p_max/(np.cos(zangle)*mass**(1/3.0)*v_0**(3/2))

    # Determine the Pf category
    if 0.85 < pf:
        pf_category = 1
    elif 0.27 < pf <= 0.85:
        pf_category = 2
    elif 0.085 < pf <= 0.27:
        pf_category = 3
    elif 0.027 < pf <= 0.085:
        pf_category = 4
    else:
        pf_category = 5


    return pf, pf_category



if __name__ == "__main__":

    import datetime

    from wmpl.Utils.TrajConversions import datetime2JD

    # Test the dynamic pressure function
    lat = np.radians(43)
    lon = np.radians(-81)
    height = 97000.0 # m 
    jd = datetime2JD(datetime.datetime(2018, 6, 15, 7, 15, 0))
    velocity = 41500 #m/s

    print('Dynamic pressure:', dynamicPressure(lat, lon, height, jd, velocity), 'Pa')



    # Test the KB function using an example GMN meteor (lambda-Sculptorid at  2023-12-12 12:46:23.484238)
    lat = np.radians(-30.904705)
    lon = np.radians(115.970139)
    ht_beg = 92.8401 # km
    jd = 2460291.03221625
    v_init = 14.70719 # km/s
    zenith_angle = np.radians(90 - 75.07381)

    print('KB:', calcKB(lat, lon, ht_beg*1000, jd, v_init*1000, zenith_angle, gmn_correction=True))



    ### Test the Pf function ###

    print()
    print("Borovicka et al. (2022) Pf parameter calculation:")

    # Test cases for Pf calculation
    pf_table = [
        # Label,              Pmax (MPa), Entry angle (deg), mass (kg), v_0 (km/s)
        ["Tagish Lake - C2u",       1.20,             18.00,  56_000.0,     15.80],
        ["Winchcombe  - CM2",       0.59,             41.87,      13.0,     13.61],
        ["Krizevci    -  H6",       3.70,             65.71,      50.0,     18.21],
        ["Novo Mesto  -  L5",      11.25,             47.85,    2500.0,     21.11],
        ["   2023CX1  - L56",       3.80,             49.09,     660.0,     14.04],
        ["   2024BX1  - Aub",       2.17,             75.56,     140.0,     15.20],
    ]

    # Test the Pf function - this value should return 1
    print("Pf test (should return ~1):", calcPf(1e6, 0, 1.0, 21.5e3))

    # Print the Pf parameters for the test cases
    for label, p_max, entry_angle, mass, v_0 in pf_table:

        pf, pf_category = calcPf(p_max*1e6, np.radians(90 - entry_angle), mass, v_0*1e3)

        print("  - {:s}: Pf = {:.2f}, Pf category = {:d}".format(label, pf, pf_category))