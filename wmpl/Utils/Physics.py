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
    intens = P_0m*10**(-0.4*mag_abs)

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



def calcMass(time, mag_abs, velocity, tau=0.007, P_0m=840.0):
    """ Calculates the mass of a meteoroid from the time and absolute magnitude. 
    
    Arguments:
        time: [ndarray] Time of individual magnitude measurement (s).
        mag_abs: [nadrray] Absolute magnitudes (i.e. apparent meteor magnitudes @100km).
        velocity: [float or ndarray] Average velocity of the meteor, or velocity at every point of the meteor
            in m/s.

    Keyword arguments:
        tau: [float] Luminous efficiency (ratio, not percent!). 0.007 (i.e. 0.7%) by default (Ceplecha & McCrosky, 1976)
        P_0m: [float] Power output of a zero absolute magnitude meteor. 840W by default, as that is the R
            bandpass for a T = 4500K black body meteor. See: Weryk & Brown, 2013 - "Simultaneous radar and 
            video meteors - II. Photometry and ionisation" for more details.

    Return:
        mass: [float] Photometric mass of the meteoroid in kg.

    """

    # Theory:
    # I = P_0m*10^(-0.4*M_abs)
    # M = (2/tau)*integral(I/v^2 dt)

    # Compute the radiated energy
    intens_int = calcRadiatedEnergy(time, mag_abs, P_0m=P_0m)

    # Calculate the mass
    mass = (2.0/(tau*velocity**2))*intens_int

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
    ]

    # Test the Pf function - this value should return 1
    print("Pf test (should return ~1):", calcPf(1e6, 0, 1.0, 21.5e3))

    # Print the Pf parameters for the test cases
    for label, p_max, entry_angle, mass, v_0 in pf_table:

        pf, pf_category = calcPf(p_max*1e6, np.radians(90 - entry_angle), mass, v_0*1e3)

        print("  - {:s}: Pf = {:.2f}, Pf category = {:d}".format(label, pf, pf_category))