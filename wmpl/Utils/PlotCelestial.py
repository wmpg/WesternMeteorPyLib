""" Plotting points on a celestial sphere. """

from __future__ import print_function, absolute_import, division

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


from wmpl.Utils.Math import meanAngle



def calcAngDataLimits(ra_data, dec_data, border_ratio=0.1):
    """ Calculate the limits of the data, such that they confortably encompass the whole data. Used for 
        determining plot limits.

    """

    # If there's only one point, add 2 fake measurements to be able to calculate limits
    if len(ra_data) == 1:

        ra_data = np.r_[ra_data, np.array([ra_data[0] - 1, ra_data[0] + 1])]
        dec_data = np.r_[dec_data, np.array([dec_data[0], dec_data[0]])]
        

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
    def __init__(self, ra_data, dec_data, projection='sinu', bgcolor='k', lon_0=0):
        """ Plotting on a celestial sphere.

        Arguments:
            ra_data: [ndarray] R.A. data in radians.
            dec_data: [ndarray] Declination data in radians.
    
        Keyword arguments:
            projection: [str] Projection type:
                - "sinu" (default) All-sky sinusoidal projection.
                - "stere" - Stereographic projection centered at the centroid of the given data.
            bgcolor: [str] Background color of the plot. Black by default.
            lon_0: [float] Longitude or RA of the centre of the plot in degrees. Only for allsky plots.

        """

        if projection == 'stere':

            ### STEREOGRAPHIC PROJECTION ###

            # Calculate plot limits and the span of those limits in degrees
            ra_min, ra_max, dec_min, dec_max, deg_span = calcAngDataLimits(np.degrees(ra_data), \
                np.degrees(dec_data))
            
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
            dec_lines = np.arange(np.floor(dec_min/label_angle_freq)*label_angle_freq, \
                np.ceil(dec_max/label_angle_freq)*label_angle_freq, label_angle_freq)

            self.m.drawparallels(dec_lines, labels=[True, False, False, False], color='0.25')

            ra_min_round = np.floor(ra_min/label_angle_freq)*label_angle_freq
            ra_max_round = np.ceil(ra_max/label_angle_freq)*label_angle_freq

            # Draw RA lines
            if ra_min < ra_max:
                ra_labels = np.arange(ra_min_round, ra_max_round, label_angle_freq)

            else:
                ra_labels = np.arange(ra_min_round - 360, ra_max_round, label_angle_freq) + 360
            

            # Calculate R.A. labeling precision
            ra_label_rounding = np.ceil(-np.log10(label_angle_freq))
            if ra_label_rounding < 1:
                ra_label_rounding = 0
            ra_label_rounding = int(ra_label_rounding)
            
            ra_label_format = u"%." + str(ra_label_rounding) + u"f\N{DEGREE SIGN}"

            ra_labels_handle = self.m.drawmeridians(ra_labels, labels=[False, False, False, True], color='0.25', \
                fmt=(lambda x: ra_label_format%(x%360)))

            # Rotate R.A. labels
            for m in ra_labels_handle:
                try:
                    ra_labels_handle[m][1][0].set_rotation(30)
                except:
                    pass

            plt.gca().tick_params(axis='x', pad=15)


            ##################################################################################################

        else:

            ### ALL-SKY SINOSOIDAL OR OTHER PROJECTION ###

            # Frequency of RA/Dec angle labels (deg)
            label_angle_freq = 15

            self.m = Basemap(celestial=True, projection=projection, lon_0=lon_0)


            # Draw Dec lines
            dec_lines = np.arange(-90, 90, label_angle_freq)
            self.m.drawparallels(dec_lines, labels=[False, True, True, False], color='0.25')

            # Draw RA lines
            ra_labels = np.arange(0, 360, label_angle_freq)
            self.m.drawmeridians(ra_labels, color='0.25')

            # Plot meridian labels
            for ra in np.arange(0, 360, 2*label_angle_freq):
                plt.annotate(np.str(ra), xy=self.m(ra, -5), xycoords='data', ha='center', color='0.25')

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

        # Do not plot anything if the inputs are None
        if (ra_data is None) or (dec_data is None):
            return None

        # Convert angular coordinates to image coordinates
        x, y = self.m(np.degrees(ra_data), np.degrees(dec_data))

        scat_handle = self.m.scatter(x, y, c=c, zorder=3, **kwargs)

        return scat_handle



    def colorbar(self, turn_off_offset=True, **kwargs):
        """ Plot a colorbar. """

        cbar = self.m.colorbar(pad="5%", **kwargs)


        if turn_off_offset:
            
            # Turn off using offsets
            cbar.formatter.set_useOffset(False)


        cbar.update_ticks()








if __name__ == "__main__":


    ra = np.random.normal(10, 0.5, 10)
    dec = np.random.normal(6.6516, 0.05, size=ra.shape)

    ra = np.radians(ra)
    dec = np.radians(dec)

    v = np.linspace(0, 10, 10)

    # Initi a celestial plot (the points have to be given to estimate the data range)
    celes_plot = CelestialPlot(ra, dec, projection='sinu')

    celes_plot.scatter(ra, dec, c=v)

    celes_plot.colorbar(label='test')

    plt.title('Test')

    plt.show()