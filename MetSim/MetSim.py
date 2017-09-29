from __future__ import print_function, division, absolute_import


import math
import time

import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt

from Utils.AtmosphereDensity import getAtmDensity


# Verbose printing flags
DEBUG_ON = False
VERBOSE_STATE = False

# Data type for precision
#D_TYPE = np.longdouble
D_TYPE = float

### DEFINE CONSTANTS

# Stefan-Boltzmann constant (W/(m^2K^4))
SIGMA_B = 5.67036713e-8
# was: sigma_b

# Boltzmann constant (J/K)      
K_BOLTZMAN = 1.3806485279e-23
# was: k_B

# Standard atmospheric pressure
P_SUR = 101325.0
# was P_sur

# earth acceleration in m/s^2
g0 = 9.81

# Earth radius (m) at 43.930723 deg latitude
R_EARTH = 6367888.0
# was: re

# Specific gas constant in J/kg*K
R_GAS = 287.2
# was R_gas

# Proton rest mass
M_PROTON = 1.6726231e-27

###

class MeteorProperties(object):

    def __init__(self):

        zero = D_TYPE(0)

        # Time
        self.t = zero

        # Length along the train
        self.s = zero

        # Height
        self.h = zero

        # Velocity
        self.v = zero

        # Luminosity
        self.lum = zero

        # Ionization
        self.q_ed = zero

        self.p2 = zero

        # Mass
        self.m = zero

        self.vh = zero
        self.vv = zero

        # Temperature
        self.temp = zero

        self.m_kill = zero
        self.T_lim = zero
        self.T_int = zero

        # Meteoroid density
        self.rho  = zero
        self.Vtot = zero

        # Boiling point
        self.T_boil = zero

        # Melting point 
        self.T_fus = zero

        # Specific heat
        self.c_p = zero

        # Heat of ablation  
        self.q = zero

        # Drag coeficient
        self.Gamma = 1.0

        # Heat transfer coeficient
        self.Lambda = 0.5

        # coefficient of condensation
        self.psi = zero

        # Average molar mass of meteor
        self.m_mass = zero

        # Thermal conductivity of meteoroid (NOT USED IN THIS CODE)
        self.therm_cond = zero
        
        self.lum_eff = zero
        self.poros = zero
        self.T_sphere = zero

        # 1 if the particle should fragment now
        self.Fl_frag_cond = zero

        # 1 if the particle is still actively ablating
        self.Fl_ablate = zero



class MeteorConstants(object):

    def __init__(self):

        zero = D_TYPE(0)

        # Emissivity
        self.emiss = zero

        # Shape factor of meteor
        self.shape_fact = zero

        # Zenith angle in radians
        self.zr = zero
        
        # Time step for integration
        self.dt = zero
        
        # Normalize magnitude to this height
        self.h_obs = zero
        
        # Arbitrary angle for hypersonic flight
        self.sigma = zero

        # Ratio of specific heats
        self.kappa = zero
            
        self.c_s = zero

        # atmospheric temperature
        self.T_a = zero
        
        # coefficients for atm. density
        self.dens_co = []

        # Interpolated atmospheric density
        self.atm_density_interp = None
        
        # Coefficients for atm. pressure
        self.press_co = []
        
        # Total number of records
        self.nrec = zero
                        
        # Largest time reached
        self.t_max = zero
                        
        # Smallest height reached (by any particle)
        self.h_min = zero

        
        self.ion_eff = zero

         # Largest height for which atm fit is valid
        self.maxht_rhoa = zero

         # Smallest height for atm fit
        self.minht_rhoa = zero



def printd(*args):
    """ Special print function which will print the contect only is DEBUG_ON is true. """

    if DEBUG_ON:
        print(*args)



def printv(*args):
    """ Special print function which will print the contect only is VERBOSE_STATE is true. """

    if VERBOSE_STATE:
        print(*args)



def fnext(f, n=1):
    """ Skip n lines from the file f. """

    for i in range(n):
        f.readline()



def parseFloat(f, skip_next=True):
    """ Parse a float from the next line in the file f. """

    val = float(f.readline())
    val = D_TYPE(val)

    # SKip the next line
    if skip_next:
        fnext(f)

    return val



def loadInputs(file_name):
    """ Loads input parameters of a meteor. """

    # Init new MeteorProperties struct
    met = MeteorProperties()

    # Init new constsants struct
    consts = MeteorConstants()

    with open(file_name) as f:

        fnext(f, 2)

        ### Meteoroid properties

        # Mass (kg)
        met.m_init = parseFloat(f)

        # Density (kg/m^3)
        met.rho = parseFloat(f)

        # Initial porosity
        met.poros = parseFloat(f)

        # Heat ablation (J/kg)
        met.q = parseFloat(f)

        # Boiling point (K)
        met.T_boil = parseFloat(f)

        # Melting point (K)
        met.T_fus = parseFloat(f)

        # Specific heat (J/kg K)
        met.c_p = parseFloat(f)

        # Condensation coefficient psi
        met.psi = parseFloat(f)

        # Molar mass (atomic units)
        met.m_mass = parseFloat(f)*M_PROTON

        # Therm conductivity (J/ m s K)
        met.therm_cond = parseFloat(f)

        # Luminous efficiency
        met.lum_eff = parseFloat(f)

        # Porosity temperature reduction
        met.T_sphere = parseFloat(f)

        # Shape factor
        consts.shape_fact = parseFloat(f)

        # Emissivity
        consts.emiss = parseFloat(f)

        ###


        ### Initial conditions
        fnext(f)

        # Initial height (m)
        met.h_init = parseFloat(f)

        # Initial trail length (m)
        met.s_init = parseFloat(f)

        # Initital speed (m/s)
        met.v_init = parseFloat(f)

        # Zenith angle (radians)
        consts.zr = math.radians(parseFloat(f))

        # Initial temperature (K)
        met.met_temp_init = parseFloat(f)

        ###

        # Simulation parameters
        fnext(f)

        # Absolute magnitude distance (m)
        consts.h_obs = parseFloat(f)

        # Time step (s)
        consts.dt = parseFloat(f)

        fnext(f, 3)

        # Atmospheric density
        consts.dens_co = []
        for i in range(6):
            consts.dens_co.append(parseFloat(f, skip_next=False))


        fnext(f)

        # Heights between which the atmospheric density fit is good
        consts.maxht_rhoa = parseFloat(f, skip_next=False)
        consts.minht_rhoa = parseFloat(f)

        fnext(f, 5)

        # Atmospheric pressure model
        consts.press_co = []
        for i in range(6):
            consts.press_co.append(parseFloat(f, skip_next=False))


        return met, consts



def atmDensity(h, consts):
    """ Calculates the atmospheric density in kg/m^3. 
    
    Arguments:
        h: [float] Height in meters.

    Return:
        [float] Atmosphere density at height h (kg/m^3)

    """

    # If the atmosphere dentiy interpolation is present, use it as the source of atm. density
    if consts.atm_density_interp is not None:
        return consts.atm_density_interp(h)

    # Otherwise, use the polynomial fit (WARNING: the fit is not as good as the interpolation!!!)
    else:
        dens_co = consts.dens_co

        rho_a = (10**(dens_co[0] + dens_co[1]*h/1000.0 + dens_co[2]*(h/1000)**2 + dens_co[3]*(h/1000)**3 \
            + dens_co[4]*(h/1000)**4 + dens_co[5]*(h/1000)**5))*1000

        return rho_a



def scaleHeight(h, consts):
    """ Calculates the scale height. """

    return 2000.0/math.log(atmDensity(h - 2000.0, consts)/atmDensity(h, consts))



def massLoss(met, consts, rho_atm):
    """ Mass loss due to ablation. """

    # Evaporation, using the Clausius-Clapeyron equation for vapour pressure (external pressure neglected 
    # for now) and the Knudsen-Langmuir formula for evaporation rate.

    qb1 = consts.dt*consts.shape_fact*math.pow(met.m*(1 + met.poros)/met.rho, 2.0/3)*met.psi \
            *math.exp(met.q*met.m_mass/(K_BOLTZMAN*met.T_boil))*P_SUR \
            *math.exp(-met.q*met.m_mass/(K_BOLTZMAN*met.temp))/math.sqrt(2*math.pi*K_BOLTZMAN*met.temp/met.m_mass)

    if qb1/2 > met.m:
        qb1 = met.m*2

    qb2 = consts.dt*consts.shape_fact*math.pow((met.m - qb1/2.0)*(1 + met.poros)/met.rho, 2.0/3)*met.psi \
            *math.exp(met.q*met.m_mass/(K_BOLTZMAN*met.T_boil))*P_SUR \
            *math.exp(-met.q*met.m_mass/(K_BOLTZMAN*met.temp))/math.sqrt(2*math.pi*K_BOLTZMAN*met.temp/met.m_mass)

    if qb2/2 > met.m:
        qb2 = met.m*2
    
    qb3 = consts.dt*consts.shape_fact*math.pow((met.m - qb2/2.0)*(1 + met.poros)/met.rho, 2.0/3)*met.psi \
        *math.exp(met.q*met.m_mass/(K_BOLTZMAN*met.T_boil))*P_SUR \
        *math.exp(-met.q*met.m_mass/(K_BOLTZMAN*met.temp))/math.sqrt(2*math.pi*K_BOLTZMAN*met.temp/met.m_mass)

    if qb3 > met.m:
        qb3 = met.m
    
    qb4 = consts.dt*consts.shape_fact*math.pow((met.m - qb3)*(1 + met.poros)/met.rho, 2.0/3)*met.psi \
        *math.exp(met.q*met.m_mass/(K_BOLTZMAN*met.T_boil))*P_SUR \
        *math.exp(-met.q*met.m_mass/(K_BOLTZMAN*met.temp))/math.sqrt(2*math.pi*K_BOLTZMAN*met.temp/met.m_mass)

    # Mass loss in kg/s due to ablation
    mdot = (qb1/6.0 + qb2/3.0 + qb3/3.0 + qb4/6.0)/consts.dt

    # Make sure the mass loss really happens
    if mdot*consts.dt > met.m:
        mdot = met.m/consts.dt

    return mdot



def tempChange(met, consts, rho_atm, scale_ht, lam, m_dot):
    """ Calculates the change in temperature. 
        
        Change in temperature is calculated from the kinetic energy of air molecules, blackbody radiation and 
        mass ablation.
    
    """

    # The total thermal inertia
    sumcpm = met.c_p*met.m

    # All the energy lost to ablation
    sumqmdot = met.q*m_dot
    
    # Three terms: fraction 'lam' of kinetic energy of air, blackbody radiation, energy to ablate mass
    qc1 = consts.dt*(1/sumcpm)*(consts.shape_fact*math.pow(met.Vtot, 2.0/3)*lam*rho_atm*(met.v**3)/2.0 \
        - 4*SIGMA_B*consts.emiss*(met.temp**4 - consts.T_a**4)*math.pow(met.Vtot, 2.0/3) - sumqmdot)

    qc2 = consts.dt*(1/sumcpm)*(consts.shape_fact*math.pow(met.Vtot, 2.0/3)*lam*rho_atm*(met.v**3)/2.0 \
        - 4*SIGMA_B*consts.emiss*((met.temp + qc1/2.0)**4 - consts.T_a**4)*math.pow(met.Vtot, 2.0/3) \
        - sumqmdot)

    qc3 = consts.dt*(1/sumcpm)*(consts.shape_fact*math.pow(met.Vtot, 2.0/3)*lam*rho_atm*(met.v**3)/2.0 \
        - 4*SIGMA_B*consts.emiss*((met.temp + qc2/2.0)**4 - consts.T_a**4)*math.pow(met.Vtot, 2.0/3) \
        - sumqmdot)

    qc4 = consts.dt*(1/sumcpm)*(consts.shape_fact*math.pow(met.Vtot, 2.0/3)*lam*rho_atm*(met.v**3)/2.0 \
        - 4*SIGMA_B*consts.emiss*((met.temp + qc3)**4 - consts.T_a**4)*math.pow(met.Vtot, 2.0/3) - sumqmdot)


    T_dot = (qc1/6.0 + qc2/3.0 + qc3/3.0 + qc4/6.0)/consts.dt


    return T_dot


def lumIntensity(met, consts, m_dot, vdot):
    """ Calculate the ionization and luminous intensity. """

    # Ionization
    beta = 4.0e-18*(met.v**3.5)
    q_ed = beta*m_dot/(met.v*met.m_mass)

    # ORIGINAL CODE - QUESTION!!!! Shouldn't q be summed in the loop, not assigned????
    # q_ed=0;
    # for (i=0;i<met.ncmp;i++)
    #   q_ed=beta*m_dotcm[i]/(met.v*met.m_mass[i]);

    # Luminosity
    lum = (0.5*met.v**2)*m_dot*met.lum_eff*1e10/(consts.h_obs**2) \
        + met.lum_eff*met.m*met.v*vdot*1e10/(consts.h_obs**2)

    return q_ed, lum



def ablate(met, consts):
    """ Ablate the main mass. """

    # Calculate atmospheric parameters
    rho_atm = atmDensity(met.h, consts)
    scale_ht = scaleHeight(met.h, consts)

    printd('h', met.h)
    printd('rho_atm', rho_atm)
    printd('scale_ht', scale_ht)

    # Mean free path
    mfp = 10**(-7.1 - math.log10(rho_atm))

    # Knudsen number
    Kn = (mfp/math.pow((3*met.Vtot/(4*math.pi)), 1/3.0))

    printd('mfp', mfp)
    printd('Kn', Kn)

    # Separate fragments in the free molecular flow, transition and continuum flow regime
    # Gamma is the drag coefficient, Lambda the heat transfer coefficient
    # if (Kn >=10) # free molecular flow
    # {
    #   Gamma = 1.0
    #   Lambda = 0.5
    # }
    # else if (Kn>=1) # Transition flow
    # {
    #   Gamma = 1.0
    #   Lambda = 0.5
    # }
    # else # continuum flow
    # {
    #   Gamma = 1.0
    #   Lambda = 0.5
    # }

    # Drag coeficient
    #Gamma = 1.0

    # Heat transfer coeficient
    #Lambda = 0.5

    # Calculation of pressure acting on meteor: atmospheric pressure
    p1 = math.pow(10, (consts.press_co[0] + consts.press_co[1]*met.h/1000 
        + consts.press_co[2]*((met.h/1000)**2) + consts.press_co[3]*((met.h/1000)**3) 
        + consts.press_co[4]*((met.h/1000)**4) + consts.press_co[5]*((met.h/1000)**5)))

    # Calculation of Mach-number
    Ma = met.v/(math.sqrt(consts.kappa*R_GAS*consts.T_a))

    printd('p1', p1)
    printd('Ma', Ma)

    # The following calculation of pressure is valid only for continuum flow
    if Kn >= 1:

        # Sets pressure to 0 for transition and free molecular flow
        Ma = 0

    # calculation of pressure
    met.p2 = p1*((2*consts.kappa/(consts.kappa + 1))*Ma**2*((math.sin(consts.sigma))**2))

    printd('met.p2', met.p2)


    # Integrations are a 4th order Runge-Kutta
    m_dot = massLoss(met, consts, rho_atm);

    printd('m_dot', m_dot)

    # Anything smaller than this will ablate in less than one time step
    met.m_kill = m_dot*consts.dt; 

    # Anything smaller than mkill kg will not ablate, since it will cool faster from radiation
    if met.m_kill < consts.mkill:
        met.m_kill = consts.mkill

    # Calculate the change in temperature
    T_dot = tempChange(met, consts, rho_atm, scale_ht, met.Lambda, m_dot)

    printd('T_dot', T_dot)

    # Three terms: fraction Lambda of kinetic energy of air, blackbody radiation, energy to ablate mass
    dEair = (consts.shape_fact*(met.Vtot**(2.0/3))*met.Lambda*rho_atm*(met.v**3)/2.0)
    dErad = -4*SIGMA_B*consts.emiss*(met.temp**4 - consts.T_a**4)*met.Vtot**(2.0/3)
    dEdm = -met.q*m_dot

    printd('dEair', dEair)
    printd('dErad', dErad)
    printd('dEdm', dEdm)

    # Change in velocity
    # The change in velocity is calculated from the exchange of momentum with atmospheric molecules 
    # (McKinley 1961, Bronshten 1983)
    qa1 = consts.dt*met.Gamma*consts.shape_fact*(met.Vtot**(2/3.0))*rho_atm*(met.v**2)/met.m
    qa2 = consts.dt*met.Gamma*consts.shape_fact*(met.Vtot**(2/3.0))*rho_atm*((met.v + qa1/2.0)**2)/met.m
    qa3 = consts.dt*met.Gamma*consts.shape_fact*(met.Vtot**(2/3.0))*rho_atm*((met.v + qa2/2.0)**2)/met.m
    qa4 = consts.dt*met.Gamma*consts.shape_fact*(met.Vtot**(2/3.0))*rho_atm*((met.v + qa3)**2)/met.m

    # Decelaration in m/s**2
    a_current = (qa1/6.0 + qa2/3.0 + qa3/3.0 + qa4/6.0)/consts.dt

    printd('a_current', a_current)

    # g at height h in m/s^2
    gv = g0/((1 + met.h/R_EARTH)**2)

    # Vertical component of a
    av = -gv - a_current*met.vv/met.v + met.vh*met.v/(R_EARTH + met.h)

    # Horizontal component of a
    ah = -a_current*met.vh/met.v - met.vv*met.v/(R_EARTH + met.h)

    # Velocity magnitude
    v_dot = math.sqrt(av**2 + ah**2)

    printd('v_dot', v_dot)

    # Check to make sure dm is less than m, otherwise nonexistant mass will be ablated
    if m_dot*consts.dt > met.m:

        # Ablate only what's there
        m_dot = met.m/consts.dt


    # Update parameters
    met.s = met.s + met.v*consts.dt
    met.h = met.h + met.vv*consts.dt
    met.m = met.m - m_dot*consts.dt

    # Calculate the new volume
    # # Use component with largest volume
    # Vmax = 0
    # dV = 0
    # for (i=0;i<met.ncmp;i++)
    #     if (met.mcmp[i]/(met.rho_meteor[i]*(1-met.poros))>Vmax)
    #     {
    #         Vmax=met.mcmp[i]/(met.rho_meteor[i]*(1-met.poros));
    #         dVmax=m_dotcm[i]*cs.dt/(met.rho_meteor[i]*(1-met.poros)); 
    #     }

    dV = m_dot*consts.dt/(met.rho*(1 - met.poros))
    met.Vtot -= dV

    printd("met.Vtot", met.Vtot)

    # Calculate the new porosity
    smV = met.m/met.rho # Volume with no porosity
    met.poros = (met.Vtot - smV)/met.Vtot

    printd('met.poros', met.poros)


    # If temperature is high enough, the particle starts to consolidate into a sphere
    if (met.temp > met.T_sphere) and ((met.poros - 0.002) >= 0):
    
        met.poros -= 0.002
        met.Vtot = met.Vtot*(1 - (met.poros + 0.002))/(1 - met.poros)
    
    if met.poros < 0:
        if met.poros < -1e-7:
            print("negative porosity")

        met.poros = 0.0
    
    met.temp = met.temp + T_dot*consts.dt

    # This should not occur, but if negative temperatures are produced, they must be dealt with
    if met.temp < 100:
    
        met.temp = 100
        print("Numerical instability in temperature at ", met.h/1000, "km")
    

    met.vv = met.vv + av*consts.dt
    met.vh = met.vh + ah*consts.dt
    met.v = math.sqrt(met.vh*met.vh + met.vv*met.vv)
    met.t = met.t + consts.dt


    # The visual luminosity is the kinetic energy lost in this step times lum_eff, the coefficient of luminous intensity. It is also corrected to a range of 100 km.
    # Ionization produced in this step as well
    q_ed, lum = lumIntensity(met, consts, m_dot, v_dot)

    met.q_ed = q_ed
    met.lum = lum

    printd("q_ed", q_ed)
    printd("lum", lum)
    printd("met.m", met.m)


    ### ORIGINAL CODE: In original code, 'met' values are the old ones, as met is updated after this!!!
    # accelout << met.t << '  ' << a_current << ' ' << v_dot << ' ' << lum << '   ' << rho_atmo << '  ' << met.v
    #     << '    ' << m_dot << ' ' << met.m << ' ' << T_dot << ' ' << met.poros << ' ' << met.temp << '  ' << met.h/1000.0 
    #     << '    ' << dEair << ' ' << -dEdm << ' ' << -dErad << endl;

    # Check fragmentation condition and ablation condition
    # Will fragment before next step
    if met.temp >= met.T_lim:
        met.Fl_frag_cond = 1


    if (met.h < 85000) and (met.temp < (consts.T_a + 200)):
        met.m_kill = met.m


    if (met.m <= met.m_kill):

        # No longer ablates
        met.Fl_ablate = 0


    # # Write results
    # if ((met.h/1000>60) and (met.h/1000<200)):
    
    #     outstream << endl;
    #     outstream << met.t << ' ' ;
    #     outstream.precision(8);
    #     outstream << met.s/1000.0 << '  ' << met.h/1000 << '    ' << met.v/1000 << '    ';
    #     outstream << lum << '   ';
    #     outstream  << q_ed << ' ' << met.temp << '  ' << met.n_frag << '    ' << met.m;
    

    # Print out the current state
    printv(met.t, met.h, met.v, met.temp)

    if ((met.h/1000 > 60) and (met.t > consts.t_max)):
        consts.t_max = met.t

    if ((met.h/1000 > 60) and (met.h < consts.h_min)):
        consts.h_min = met.h

    printd(met.h)

    return met, consts



def runSimulation(met, consts, lat=None, lon=None, jd_ref=None):
    """ Runs meteor ablation simulation with the given initial parameters. The location and time can be 
        provided as well, which will be used for calculating the atmosphere density at the given average 
        geodetic coordinates and Julian date. 

    Arguments:
        met: [MeteorProperties object] Structure containing physical characteristics of the simulated meteor.
        consts: [MeteorConstants object] Structure containing simulation constants.

    Keyword arguments:
        lat: [float] Geodetic latitude of the mean meteor path (radians).
        lon: [float] Geodetic longitude of the mean meteor path (radians).
        jd_ref: [float] Mean Julian date of the meteor.
    
    Return:
        [list] A list containing simulation results:
            [time, heights, length, velocity, luminosity]

    """


    # If the location is given, do a 5 order fit on the atmosphere density
    if (lat is not None) and (lon is not None) and (jd_ref is not None):

        # Get a range of heights from 60 to 200 km, every 100 meters
        heights_fit = np.arange(60, 200, 0.1)*1000

        # Get atmosphere density at simulation heights
        atm_dens_list = []
        for ht in heights_fit:
            atm_dens = getAtmDensity(lat, lon, ht, jd_ref)
            atm_dens_list.append(atm_dens)

        atm_dens_list = np.array(atm_dens_list)

        # Interpolate the density
        consts.atm_density_interp = scipy.interpolate.PchipInterpolator(heights_fit, atm_dens_list)



    ### Calculate physical/other params

    # Volume
    met.vol = met.m_init/met.rho
    met.vol *= 1 + met.poros

    # Initial radius (assume sphere)
    met.r_init = ((3.0/4)*met.vol/math.pi)**(1.0/3)

    # ?
    met.vv_init = -met.v_init*math.cos(consts.zr)
    met.vh_init = met.v_init*math.sin(consts.zr)

    ###

    # Other parameters
    consts.T_a = 280
    consts.h_min = 200000
    consts.t_max = 0
    consts.kappa = 1.39


    # Currently, fits for pressure and density of the atmosphere are only valid between 200 and 60 km. 
    # Results may not be valid outside this range.
    if met.h_init > consts.maxht_rhoa:
        print("**** WARNING: Beginning height exceeds", consts.maxht_rhoa, "km ****")
        print("Results of this integration may not be valid")
        print("Proceeding with integration")


    # Final mass at which the iteration stops
    consts.mkill = 1e-14

    if (consts.mkill > met.m_init):
        consts.mkill = met.m_init


    # Assume a single-body meteor
    #frag = MeteorProperties()

    met.t = 0
    met.s = 0
    met.h = met.h_init
    met.v = met.v_init
    met.p2 = 0
    met.m = met.m_init
    met.vv = met.vv_init
    met.vh = met.vh_init
    met.temp = met.met_temp_init
    met.m_kill = consts.mkill
    met.T_int = met.met_temp_init

    met.Vtot = met.vol

    # 1 if the particle should fragment now
    met.Fl_frag_cond = 0

    # 1 if the particle is still actively ablating
    met.Fl_ablate = 1

    results_list = []

    # Run the ablation until the mass is too small
    while(met.Fl_ablate and (met.m > consts.mkill) and (met.h/1000 < 200)):

        printd("##########################")
        
        # Ablate the fragment
        met, consts = ablate(met, consts)

        # Add the current state to results
        results_list.append([met.t, met.h, met.s, met.v, met.lum])


    printv('Minimum height was', consts.h_min/1000, 'km')

    if consts.h_min < consts.minht_rhoa:
        print("**** WARNING: Final height of some fragments was below", consts.minht_rhoa, "km ****")
        print("Atmospheric model not valid in this range: results may not be valid")
        print("Proceeding to evaluate")


    return results_list


if __name__ == "__main__":

    import os
    import datetime
    
    from Utils.TrajConversions import datetime2JD


    # Name of input file for meteor parameters
    file_name = os.path.join('MetSim', 'Metsim0001_input.txt')

    # Set meteor average location and time
    lat = np.radians(43.937484)
    lon = np.radians(-81.645127)
    #jd = datetime2JD(datetime.datetime.now())
    jd = 2457955.794670294970

    # Load input meteor data
    met, consts = loadInputs(file_name)


    # Set drag coeficient
    met.Gamma = 1.0

    # Set heat transfer corficient (cometary)
    met.Lambda = 0.5



    t1 = time.clock()
    
    # Run the simulation
    results_list = runSimulation(met, consts, lat=lat, lon=lon, jd_ref=jd)

    print('Runtime:', time.clock() - t1)



    # Get the results
    results_list = np.array(results_list)
    time, height, trail, velocity, luminosity = results_list.T

    # Convert distance/height to km
    height = height/1000
    trail = trail/1000
    velocity = velocity/1000

    ### Calculate the lag

    # Fit a line to the first 10% of points
    part_len = int(0.1*len(time))
    
    fpart_t = time[:part_len]
    fpart_s = trail[:part_len]


    def line(x, m, l):

        return m*x + l

    line_params, _ = scipy.optimize.curve_fit(line, fpart_t, fpart_s)

    lag = trail - line(time, *line_params)

    ###

    def jacchia(t, a, b, c, k):

        return a + b*t + c*np.exp(k*t)


    ### Take the last 25% of the trail
    lquart_len = int(0.75*len(time))
    lquart_t = time[lquart_len:]
    lquart_s = trail[lquart_len:]
    lquart_lag = lag[lquart_len:]

    #p0 = [line_params[1], line_params[0], 0, 0]
    #jacchia_fit, _ = scipy.optimize.curve_fit(jacchia, lquart_t, lquart_s, maxfev=20000, p0=p0)
    #print(jacchia_fit)

    spline_fit = scipy.interpolate.CubicSpline(time, lag)

    fig, ax1 = plt.subplots()

    # Plot the original data
    ax1.scatter(lag, time, s=2, zorder=2, c='red')

    # Plot the spline data
    ax1.plot(spline_fit(time), time, zorder=1)

    #ax1.plot(jacchia(lquart_t, *jacchia_fit), lquart_t)

    ax1.set_xlabel('Length (km)')

    ax1.set_ylabel('Time')
    ax1.set_ylim([min(time), max(time)])
    ax1.invert_yaxis()

    ax2 = ax1.twinx()
    ax2.set_ylim(min(height), max(height))
    ax2.set_ylabel('Height (km)')

    ax1.grid()

    plt.show()



    ### MAGNITUDES ###

    # Calculate absolute magnitude (apparent @100km) from luminous intensity
    P_0m = 840.0
    magnitude = -2.5*np.log10(luminosity/P_0m)


    # Plot time vs. absolute magnitude
    plt.plot(magnitude, height, zorder=3)

    plt.ylabel('Height (km)')
    plt.xlabel('Absolute magnitude')

    #plt.gca().invert_yaxis()

    plt.legend()

    plt.grid()

    plt.show()


    ### VELOCITY ###

    plt.plot(velocity, height, zorder=3)

    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Height (km)')

    # plt.gca().invert_yaxis()

    plt.grid(color='0.9')

    ## TEST
    plt.xlim(15.0, 17.0)
    plt.ylim(75, 185)

    # Find the index of the closest height to ht_mark
    ht_mark = 87.620
    ht_indx = np.argmin(np.abs(height - ht_mark))

    # Plot a point at the velocity at the given heights
    ht_mark = height[ht_indx]
    vel_mark = velocity[ht_indx]
    plt.scatter(vel_mark, ht_mark, c='red', label='Height = {:.3f} km\nVelocity = {:.3f} km/s'.format(ht_mark, vel_mark), zorder=4)

    plt.legend()

    plt.savefig('vel_diff.png', dpi=300)

    plt.show()