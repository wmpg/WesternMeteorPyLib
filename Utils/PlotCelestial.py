""" Plotting points on a celestial sphere. """

from __future__ import print_function, absolute_import, division

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


from Utils.Math import meanAngle



def calcAngDataLimits(ra_data, dec_data, border_ratio=0.1):
    """ Calculate the limits of the data, such that they confortably encompass the whole data. Used for 
        determining plot limits.

    """

    # Check if there are any points crossing the 0/360 branch cut
    if max(ra_data) - min(ra_data) > 180:
        ra_data[ra_data < 180] += 360

    ra_min = min(ra_data)
    ra_max = max(ra_data)

    if ra_min > ra_max:
        ra_max = max([ra_max, ra_min])
        ra_min = min([ra_max, ra_min])


    dec_min = min(dec_data)
    dec_max = max(dec_data)


    ### Make the limits rectangular ###
    ##########################################################################################################
    delta_ra = abs(ra_max - ra_min)
    delta_dec = abs(dec_max - dec_min)

    if delta_ra > delta_dec:

        delta_largest = delta_ra
        
        delta_diff = (delta_ra - delta_dec)/2
        
        dec_min -= delta_diff
        dec_max += delta_diff

    else:

        delta_largest = delta_dec

        delta_diff = (delta_dec - delta_ra)/2

        ra_min -= delta_diff
        ra_max += delta_diff

    ##########################################################################################################


    ### Add a 'buffer' to the limits ###
    ##########################################################################################################

    ra_min -= delta_largest*border_ratio
    ra_max += delta_largest*border_ratio
    dec_min -= delta_largest*border_ratio
    dec_max += delta_largest*border_ratio

    ##########################################################################################################



    ### Make sure the data is within the proper limits ###
    ##########################################################################################################
    if dec_min < -90:
        dec_min = -90

    if dec_max > 90:
        dec_max = 90

    ra_min = ra_min%360
    ra_max = ra_max%360

    ##########################################################################################################

    return ra_min, ra_max, dec_min, dec_max, delta_largest




class CelestialPlot(object):

    def __init__(self, ra_data, dec_data, projection='sinu', bgcolor='k'):
        """ Plotting on a celestial sphere.

        Arguments:
            ra_data: [ndarray] R.A. data in radians.
            dec_data: [ndarray] Declination data in radians.
    
        Keyword arguments:
            projection: [str] Projection type:
                - "sinu" (default) All-sky sinusoidal projection.
                - "stere" - Stereographic projection centered at the centroid of the given data.
            bgcolor: [str] Background color of the plot. Black by defualt.

        """



        if projection == 'stere':

            ### STEREOGRAPHIC PROJECTION ###

            # Calculate plot limits and the span of those limits in degrees
            ra_min, ra_max, dec_min, dec_max, deg_span = calcAngDataLimits(np.degrees(ra_data), np.degrees(dec_data))
            
            # Calculate centre of the projection
            ra_mean = np.degrees(meanAngle([np.radians(ra_min), np.radians(ra_max)]))
            dec_mean = np.degrees(meanAngle([np.radians(dec_min), np.radians(dec_max)]))


            # Init a new basemap plot
            self.m = Basemap(celestial=True, projection=projection, lat_0=dec_mean, lon_0=-ra_mean, 
                llcrnrlat=dec_min, urcrnrlat=dec_max, llcrnrlon=-ra_min, urcrnrlon=-ra_max)

            # Calculate the frequency of RA/Dec angle labels in degrees, so it labels every .5 division
            label_angle_freq = 10**(np.ceil(np.log10(deg_span)))

            # Change label frequency so the ticks are not too close
            if deg_span/label_angle_freq < 0.25:
                label_angle_freq *= 1

            elif deg_span/label_angle_freq < 0.5:
                label_angle_freq *= 2

            else:
                label_angle_freq *= 4

            # Have 20 ticks between the min and max of the current scale
            label_angle_freq /= 20


            # Draw Dec lines
            dec_lines = np.arange(np.floor(dec_min/label_angle_freq)*label_angle_freq, np.ceil(dec_max/label_angle_freq)*label_angle_freq, label_angle_freq)
            self.m.drawparallels(dec_lines, labels=[True, False, False, False], color='0.25')

            print(ra_min, ra_max)

            ra_min_round = np.floor(ra_min/label_angle_freq)*label_angle_freq
            ra_max_round = np.ceil(ra_max/label_angle_freq)*label_angle_freq

            # Draw RA lines
            if ra_min < ra_max:
                ra_labels = np.arange(ra_min_round, ra_max_round, label_angle_freq)

            else:
                ra_labels = np.arange(ra_min_round - 360, ra_max_round, label_angle_freq) + 360


            
            self.m.drawmeridians(ra_labels, labels=[False, False, False, True], color='0.25')

            plt.gca().tick_params(axis='x', pad=15)

            ##################################################################################################

        else:

            ### ALL-SKY SINOSOIDAL PROJECTION ###

            # Frequency of RA/Dec angle labels (deg)
            label_angle_freq = 15

            self.m = Basemap(celestial=True, projection=projection, lon_0=0)


            # Draw Dec lines
            dec_lines = np.arange(-90, 90, label_angle_freq)
            self.m.drawparallels(dec_lines, labels=[False, True, True, False], color='0.25')

            # Draw RA lines
            ra_labels = np.arange(0, 360, label_angle_freq)
            self.m.drawmeridians(ra_labels, labels=[True, False, False, True], color='0.25')

            ##################################################################################################



        # Set background color
        self.m.drawmapboundary(fill_color=bgcolor)

        # Turn off shortening numbers into offsets
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)



    def scatter(self, ra_data, dec_data, c=None, **kwargs):
        """ Scatter plot on the celestial sphere. 
    
        Arguments:
            ra_data: [ndarray] R.A. data in radians.
            dec_data: [ndarray] Declination data in radians.

        Keyword arguments:
            **kwargs: [dict] Any additional keyword arguments will be passes to matplotlib scatter.
        """

        # Convert angular coordinates to image coordinates
        x, y = self.m(np.degrees(ra_data), np.degrees(dec_data))

        self.m.scatter(x, y, c=c, zorder=3, **kwargs)



    def colorbar(self, **kwargs):
        """ Plot a colorbar. """

        cbar = self.m.colorbar(pad="5%", **kwargs)

        # Turn off using offsets
        cbar.formatter.set_useOffset(False)
        cbar.update_ticks()








if __name__ == "__main__":


    ra = np.random.normal(0, 0.1, 10)
    dec = np.random.normal(6.6516, 0.05, size=ra.shape)

    ra = np.radians(ra)
    dec = np.radians(dec)

    v = np.linspace(0, 10, 10)

    celes_plot = CelestialPlot(ra, dec, projection='stere')
    #celes_plot = CelestialPlot(ra, dec)

    celes_plot.scatter(ra, dec, c=v)

    celes_plot.colorbar(label='test')

    plt.title('Test')

    plt.tight_layout()

    plt.show()