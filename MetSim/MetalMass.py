""" Loading and calculating masses from METAL .met files. """

from __future__ import print_function, absolute_import, division


import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from Formats.Met import loadMet
from Utils.TrajConversions import unixTime2Date
from Utils.Math import averageClosePoints




def calcMass(time, mag_abs, velocity):
    """ Calculates the mass of a meteoroid from the time and absolute magnitude. 
    
    Arguments:
        time: [ndarray] time of individual magnitude measurement (s)
        mag_abs: [nadrray] absolute magnitudes (i.e. apparent meteor magnitudes @100km)
        velocity: [float or ndarray] average velocity of the meteor, or velocity at every point of the meteor
            in m/s

    Return:
        mass: [float] photometric mass of the meteoroid in kg

    """

    # Theory:
    # I = P_0m*10^(-0.4*M_abs)
    # M = (2/tau)*integral(I/v^2 dt)

    # Luminous efficiency = 0.7% (Ceplecha & McCrosky, 1976)
    tau = 0.7/100

    # Calculate the intensities from absolute magnitudes
    # The number P_0m = 840W is the power output for a zero absolute magnitude meteor in the R bandpass (we are
    # using stars in the R band for photometry), for T = 4500K.
    # Weryk & Brown, 2013 - "Simultaneous radar and video meteors - II. Photometry and ionisation"
    P_0m = 840.0
    intens = P_0m*10**(-0.4*mag_abs)

    # Interpolate I/v^2
    intens_interpol = scipy.interpolate.PchipInterpolator(time, intens)

    # x_data = np.linspace(np.min(time), np.max(time), 1000)
    # plt.plot(x_data, intens_interpol(x_data))
    # plt.scatter(time, intens/(velocity**2))
    # plt.show()

    # Integrate the interpolated I/v^2
    intens_int = intens_interpol.integrate(np.min(time), np.max(time))

    # Calculate the mass
    mass = (2.0/(tau*velocity**2))*intens_int

    return mass

    

def loadMetalMags(dir_path, file_name):
    """ Loads time and absolute magnitudes (apparent @100km) from the METAL .met file where the photometry has
        been done on a meteor.

    Arguments:
        dir_path: [str] path to the directory where the METAL .met file is located
        file_name: [str] name of the METAL .met file

    Return:
        [(time1, mag_abs1), (time2, mag_abs2),...]: [list of tuples of ndarrays] Time in seconds and absolute 
            magnitudes
    
    """

    # Load the METAL-style met file
    met = loadMet(dir_path, file_name)

    time_mags = []

    # Extract the time and absolute magnitudes from all sites
    for site in met.sites:

        # Extract time, range, and apparent magnitude
        data = np.array([[unixTime2Date(int(pick[23]), int(pick[24]), dt_obj=True), pick[31], pick[17]] for pick in met.picks[site]])

        # Remove rows with infinite magnitudes
        data = data[data[:, 2] != np.inf]
        
        # Split into time, range and apparent magnitude
        time, r, mag_app = data.T

        # Calculate absolute magnitude (apparent magnitude @100km range)
        mag_abs = mag_app + 5.0*np.log10(100.0/r.astype(np.float64))

        # Append the time and magnitudes for this site
        time_mags.append((site, time, mag_abs))


    return time_mags







if __name__ == '__main__':

    #dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MirfitPrepare/20161007_052749_met"
    #dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MetalPrepare/20161007_052346_met"

    dir_path = "../MetalPrepare/20170721_070420_met"

    file_name = 'state.met'

    fps = 80.0

    # Load magnitudes from the .met file
    time_mags = loadMetalMags(dir_path, file_name)

    print(time_mags)

    # Average velocity (m/s)
    #v_avg = 23541.10
    #v_avg = 26972.28
    v_avg = 15821.78

    ##########################################################################################################

    site_names = ['Tavistock', 'Elginfield']
    for i, (site, time, mag_abs) in enumerate(time_mags):
        plt.plot(time, mag_abs, label=site_names[i], zorder=3)

    plt.legend()
    plt.grid(color='0.9')
    plt.xlabel('Time')
    plt.ylabel('Absolute mangitude')

    # Set datetime format
    myFmt = mdates.DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(myFmt)
    
    plt.gca().invert_yaxis()

    plt.savefig('photometry.png', dpi=300)
    
    plt.show()



    # Combine absolute magnitude and time to one array
    time_data = []
    mag_abs_data = []

    for site, time, mag_abs in time_mags:

        # Add the measurements from every site to one list
        time_data = np.r_[time_data, time]
        mag_abs_data = np.r_[mag_abs_data, mag_abs]


    # Sort the measurements by time
    time_mag_data = np.c_[time_data, mag_abs_data]
    time_mag_data = time_mag_data[np.argsort(time_mag_data[:, 0])]
    time_data, mag_abs_data = time_mag_data.T

    # Normalize the time to the first time point and convert it to seconds (from datetime objects)
    time_data -= np.min(time_data)
    time_data = np.fromiter((time_diff.total_seconds() for time_diff in time_data), np.float64)


    # Plot raw absolute magnitudes
    plt.scatter(time_data, mag_abs_data, label='Measurements', s=5)


    # Average absolute magnitude values which are closer than half a frame
    time_data, mag_abs_data = averageClosePoints(time_data, mag_abs_data, delta=1.0/(2*fps))


    time_data = np.array(time_data)
    mag_abs_data = np.array(mag_abs_data)

    # Calculate the mass from magnitude
    mass = calcMass(time_data, mag_abs_data, v_avg)


    print('Mass', mass, 'kg')
    print('Mass', mass*1000, 'g')
    print('log10', np.log10(mass))


    # Calculate approx. sphere diameter
    dens = 1000 # kg/m^3

    vol = mass/dens
    d = 2*(3*vol/(4*np.pi))**(1/3.0)

    print('Diameter', d*1000, 'mm at', dens, 'kg/m^3')


    # Plot averaged data
    plt.plot(time_data, mag_abs_data, label='Averaged')

    plt.legend()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute mangitude')
    
    plt.gca().invert_yaxis()
    
    plt.show()