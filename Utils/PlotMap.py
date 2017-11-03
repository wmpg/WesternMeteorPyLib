""" Functions for plotting ground maps. """

from __future__ import print_function, division, absolute_import

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from Utils.Math import meanAngle




class GroundMap(object):
    def __init__(self, lat_list, lon_list, border_size=50, bgcolor='0.2', parallel_step=1.0, \
        meridian_step=1.0, plot_scale=True):
        """ Inits a gnomonic Basemap plot around given latitudes and longitudes.

        Arguments:
            lon_list: [list] A list of longitudes (+E in radians) which will be fitted inside the plot.
            lat_list: [list] A list of latitudes (+N in radians) which will be fitted inside the plot.

        Keyword arguments:
            border_size: [float] Size (kilometers) of the empty space which will be added around the box which 
                fits all given geo coordinates.
            bgcolor: [str] Background color of the plot. Default is dark gray.
            parallel_step: [float] Steps of parallel lines in degrees. 1 deg by default.
            meridian_step: [float] Steps of meridian lines in degrees. 1 deg by default.
            plot_scale: [bool] Plot the map scale. True by default.
        """


        # Calculate the mean latitude and longitude by including station positions
        lat_mean = meanAngle([lat for lat in lat_list])
        lon_mean = meanAngle([lon for lon in lon_list])


        # Put coordinate of all sites and the meteor in the one list
        geo_coords = zip(lat_list, lon_list)

        # Find the maximum distance from the center to all stations and meteor points, this is used for 
        # scaling the finalground track plot
        
        max_dist = 0
        
        lat1 = lat_mean
        lon1 = lon_mean

        for lat2, lon2 in geo_coords:

            # Calculate the angular distance between two coordinates
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            a = np.sin(delta_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2)**2
            c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d = 6371000*c

            # Set the current distance as maximum if it is larger than the previously found max value
            if d > max_dist:
                max_dist = d


        # Add some buffer to the region size distance
        max_dist += 1000*border_size

        # Init the map
        self.m = Basemap(projection='gnom', lat_0=np.degrees(lat_mean), lon_0=np.degrees(lon_mean), \
            width=2*max_dist, height=2*max_dist, resolution='i')

        # Draw the coast boundary and fill the oceans with the given color
        self.m.drawmapboundary(fill_color=bgcolor)

        # Fill continents, set lake color same as ocean color
        self.m.fillcontinents(color='black', lake_color='0.2', zorder=1)

        # Draw country borders
        self.m.drawcountries(color='0.2')
        self.m.drawstates(color='0.15', linestyle='--')


        # Calculate the range of meridians and parallels to plot
        x_min, x_max = plt.gca().get_xlim()
        y_min, y_max = plt.gca().get_ylim()

        lon_min, lat_min = self.m(x_min, y_min, inverse=True)
        lon_max, lat_max = self.m(x_max, y_max, inverse=True)


        lon_min = np.floor(lon_min%360)
        lat_min = np.floor(lat_min)
        lon_max = np.ceil(lon_max%360)
        lat_max = np.ceil(lat_max)

        ### Check that the min and max are in the proper range

        # Handle the 0/360 boundary
        if lon_min > lon_max:
            lon_min = 0
            lon_max = 360

        else:

            if lon_min < 0:
                lon_min = 0

            if lon_max > 360:
                lon_max = 360

        if lat_min < -90:
            lat_min = -90

        if lat_max > 90:
            lat_max = 90


        ######


        # Make sure there are always at least 2 and at most 7 parallels and meridians in the plot
        # Use at most 10 iterations
        for _ in range(10):

            parallel_no = abs(lat_max - lat_min)/parallel_step

            if parallel_no > 7:
                parallel_step *= 2
                meridian_step *= 2

            elif parallel_no < 3:
                parallel_step /= 2
                meridian_step /= 2            

            else:
                break



        # Draw parallels
        parallels = np.arange(lat_min, lat_max, parallel_step)
        self.m.drawparallels(parallels, labels=[True, False, False, False], color='0.25')

        # Draw meridians
        meridians = np.arange(lon_min, lon_max, meridian_step)
        self.m.drawmeridians(meridians, labels=[False, False, False, True], color='0.25')


        # Turn off shortening numbers into offsets
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)



        if plot_scale:
            ## Plot the map scale
            
            # Get XY cordinate of the lower left corner
            ll_x, _ = plt.gca().get_xlim()
            ll_y, _ = plt.gca().get_ylim()

            # Move the label to fit in the lower left corner
            ll_x += 0.2*2*max_dist
            ll_y += 0.1*2*max_dist

            # Convert XY to latitude, longitude
            ll_lon, ll_lat = self.m(ll_x, ll_y, inverse=True)


            # Determine the scale size
            scale_size = int(np.log10(max_dist/2/1000))

            # Round to distance to the closest scale size
            scale_range = round(max_dist/2/1000/(10**scale_size), 0)*(10**scale_size)


            # Plot the scale
            self.m.drawmapscale(ll_lon, ll_lat, lon_mean, lat_mean, scale_range, barstyle='fancy', units='km', 
                fontcolor='0.5', zorder=3)


            # Reset the colour cycle
            plt.gca().set_prop_cycle(None)



    def scatter(self, lat_list, lon_list, **kwargs):
        """ Perform a scatter plot on the initialized map. 
        
        Arguments:
            lon_list: [list] A list of longitudes (+E in radians) to plot.
            lat_list: [list] A list of latitudes (+N in radians) to plot.

        """

        # Convert coordinates to map coordinates
        x, y = self.m(np.degrees(lon_list), np.degrees(lat_list))

        self.m.scatter(x, y, zorder=3, **kwargs)



    def plot(self, lat_list, lon_list, **kwargs):
        """ Plot a curve of given coordinates on the initialized map. 
        
        Arguments:
            lon_list: [list] A list of longitudes (+E in radians) to plot.
            lat_list: [list] A list of latitudes (+N in radians) to plot.

        """

        # Convert coordinates to map coordinates
        x, y = self.m(np.degrees(lon_list), np.degrees(lat_list))

        self.m.plot(x, y, zorder=3, **kwargs)




if __name__ == "__main__":


    # Generate some geo coords
    lat_list = np.linspace(np.radians(45), np.radians(46), 10)
    lon_list = np.linspace(np.radians(13), np.radians(30), 10)

    # Test the ground plot function
    m = GroundMap(lat_list, lon_list)

    m.scatter(lat_list, lon_list)

    m.plot(lat_list, lon_list)

    plt.show()