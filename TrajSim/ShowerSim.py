""" Given the parameters that describe a meteor shower, this code generates the shower meteors.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from Utils.PlotCelestial import CelestialPlot
from Utils.Math import sphericalToCartesian, cartesianToPolar, rotateVector




def sampleActivityModel(b, sol_max, n_samples=1):
    """ Drawing samples from a probability distribution representing activity of a meteor shower. The sampling
        is done using the Inverse transform sampling method. Activity model taken from: Jenniskens, P. (1994). 
        Meteor stream activity I. The annual streams. Astronomy and Astrophysics, 287., equation 8.

    Arguments:
        sol_max: [float] Solar longitude of the maximum shower activity (radians).
        b: [float] Slope of the activity profile.

    Keyword arguments:
        n_samples: [float] Number of samples to be drawn from the activity profile distribution.

    """

    y = np.random.uniform(0, 1, size=n_samples)

    # Draw samples from the inverted distribution
    samples = np.sign(np.random.uniform(-1, 1, size=n_samples))*np.log10(y)/b + np.degrees(sol_max)

    return np.radians(samples)%(2*np.pi)



def simulateMeteorShower(n_meteors, ra_g, ra_g_sigma, dec_g, dec_g_sigma, d_ra, d_dec, v_g, v_g_sigma, year, 
    month, sol_max, sol_slope, beg_height, beg_height_sigma):
    """ 

    Arguments:
        n_meteors: [int] Number of simulated meteor radiants to draw.
        ra_g: [float] Right ascension of centre of geocentric radiant (radians).
        ra_g_sigma: [float] R.A. standard deviation of the radiant (radians).
        dec_g: [float] Declination of centre of geocentric radians (radians).
        dec_g_sigma: [float] Declination standard deviation of the radiant (radians).
        d_ra: [float] R.A. radiant drift (radians of R.A. per radian of solar longitude).
        d_dec: [float] Dec radiant drift (radians of declination per radian of solar longitude).
        v_g: [float] Mean geocentric velocity (m/s).
        v_g_sigma: [float] Standard deviation of the geocentric velocity (m/s).
        year: [int] Year of the meteor shower.
        month: [int] Month of the meteor shower.
        sol_max: [float] Solar longitude of the maximum shower activity (radians).
        sol_slope: [float] Slope of the activity profile.
        beg_height - mean of Gaussian beginning height profile
        beg_height_sigma - stddev of beginning height profile

        """



    # Draw solar longitudes from the activity profile
    sol_data = sampleActivityModel(sol_slope, sol_max, n_samples=n_meteors)

    # Draw R.A. and Dec from a bivariate Gaussian centred at (0, 0)
    mean = (0, 0)
    cov = [[ra_g_sigma, 0], [0, dec_g_sigma]]
    ra_g_data, dec_g_data = np.random.multivariate_normal(mean, cov, n_meteors).T


    # Convert the angles to direction vectors
    radiant_vect = np.array(sphericalToCartesian(1, dec_g_data, ra_g_data)).T

    print(radiant_vect)

    # Rotate the radiants to the centre of the meteor shower radiant
    for i in range(len(radiant_vect)):

        # Rotate Dec
        radiant_vect[i] = rotateVector(radiant_vect[i], np.array([1, 0, 0]), dec_g)

        # Rotate R.A.
        radiant_vect[i] = rotateVector(radiant_vect[i], np.array([0, 0, 1]), ra_g)


    # Convert radiant vectors to RA, Dec
    dec_g_data, ra_g_data = np.array(cartesianToPolar(*radiant_vect.T))


    # Wrap around R.A. to be within 2pi
    ra_g_data = ra_g_data%(2*np.pi)

    # Wrap around Dec to bo within -pi, pi
    dec_g_data[dec_g_data >  np.pi] =  np.pi - dec_g_data[dec_g_data >  np.pi]
    dec_g_data[dec_g_data < -np.pi] = -np.pi + dec_g_data[dec_g_data < -np.pi]

    print(np.degrees(np.c_[ra_g_data, dec_g_data]))

    m = CelestialPlot(ra_g_data, dec_g_data)

    m.scatter(ra_g_data, dec_g_data)

    plt.show()


    #return ra_g_data, dec_g_data, v_g_data, sol_data, beg_height_data

    







if __name__ == "__main__":


    n_meteors = 100

    #ra_g = np.radians(113.0)
    ra_g = np.radians(0)
    ra_g_sigma = np.radians(1.5)

    #dec_g = np.radians(32.5)
    dec_g = np.radians(0)
    dec_g_sigma = np.radians(1.5)

    # Radiant drift in radians per radian of solar longitude solar longitude
    d_ra = 1.05
    d_dec = -0.17

    v_g = 33.5
    v_g_sigma = 0.5

    year = 2012
    month = 12

    sol_max = np.radians(261)
    sol_slope = 0.4

    beg_height = 100
    beg_height_sigma = 20


    simulateMeteorShower(n_meteors, ra_g, ra_g_sigma, dec_g, dec_g_sigma, d_ra, d_dec, v_g, v_g_sigma, year, 
    month, sol_max, sol_slope, beg_height, beg_height_sigma)
    

    # # Sample the activity profile    
    # activity = sampleActivityModel(sol_slope, sol_max, n_samples=1000)

    # plt.hist(activity, bins=50)

    # plt.xlim([0, 2*np.pi])

    # plt.show()