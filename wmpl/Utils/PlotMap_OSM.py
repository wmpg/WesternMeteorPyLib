""" Functions for plotting ground maps. """

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from wmpl.Utils.OSTools import importBasemap
Basemap = importBasemap()

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from wmpl.Utils.Math import meanAngle


class MapColorScheme(object):
    def __init__(self):
        """ Container for map color schemes. """

        self.dark()


    def dark(self):
        """ Dark color theme. """

        self.map_background = '0.2'
        self.continents = 'black'
        self.lakes = '0.2'
        self.countries = '0.2'
        self.states = '0.15'
        self.parallels = '0.25'
        self.meridians = '0.25'
        self.scale_bar_text = '0.5'


    def light(self):
        """ Light color theme. """

        self.map_background = '0.85'
        self.continents = '0.99'
        self.lakes = '0.8'
        self.countries = '0.8'
        self.states = '0.85'
        self.parallels = '0.7'
        self.meridians = '0.7'
        self.scale_bar_text = '0.2'



class OSMMap(object):
    def __init__(self, lat_list, lon_list, border_size=50, color_scheme='dark', parallel_step=1.0, \
        meridian_step=1.0, plot_scale=True, ax=None):
        """ Inits a gnomonic Basemap plot around given latitudes and longitudes.

        Arguments:
            lon_list: [list] A list of longitudes (+E in radians) which will be fitted inside the plot.
            lat_list: [list] A list of latitudes (+N in radians) which will be fitted inside the plot.

        Keyword arguments:
            border_size: [float] Size (kilometers) of the empty space which will be added around the box which 
                fits all given geo coordinates.
            color_scheme: [str] 'dark' or 'light'. Dark by default.
            parallel_step: [float] Steps of parallel lines in degrees. 1 deg by default.
            meridian_step: [float] Steps of meridian lines in degrees. 1 deg by default.
            plot_scale: [bool] Plot the map scale. True by default.
            ax: [matplotlib axis handle] Axis to use for plotting. None by default.

        """



        # Calculate the mean latitude and longitude by including station positions
        
        # remove coord duplicates
        (lats, lons) = np.unique(np.array([lat_list, lon_list]).T, axis=0).T  
        
        # calculate mean longitude and latitude and center the map
        lat_mean = meanAngle([lat for lat in lats])
        lon_mean = meanAngle([lon for lon in lons])

        # Put coordinate of all sites and the meteor in the one list
        geo_coords = zip(lat_list, lon_list)
      
        lat1 = lat_mean
        lon1 = lon_mean
        max_delta_lon = 0
        max_delta_lat = 0

        for lat2, lon2 in geo_coords:

            # Calculate the angular distance between two coordinates
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1

            # saving the max lat and lon distance
            if delta_lon > max_delta_lon:
                max_delta_lon = delta_lon
            if delta_lat > max_delta_lat:
                max_delta_lat = delta_lat
              
				
				# fix the longitude extent
        max_delta_lon = max_delta_lon * 2
        max_delta_lat = max_delta_lat * 2

        # Init the map
        request = cimgt.OSM()  
        self.ax = plt.axes(projection=request.crs)
        
				# calculate map extent
        lon_min = np.degrees(lon_mean - max_delta_lon)
        lon_max = np.degrees(lon_mean + max_delta_lon)
        lat_min = np.degrees(lat_mean - max_delta_lat)
        lat_max = np.degrees(lat_mean + max_delta_lat)

        #lon_min = np.floor(lon_min%360)
        #at_min = np.floor(lat_min)
        #lon_max = np.ceil(lon_max%360)
        #lat_max = np.ceil(lat_max)

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

        extent = [lon_min, lon_max, lat_min, lat_max]

				# setting map extent
        self.ax.set_extent(extent)
        
        # adding map with zoom ratio
        self.ax.add_image(request, 6)


    def scatter(self, lat_list, lon_list, **kwargs):
        """ Perform a scatter plot on the initialized map. 
        
        Arguments:
            lon_list: [list] A list of longitudes (+E in radians) to plot.
            lat_list: [list] A list of latitudes (+N in radians) to plot.

        """

        # Convert coordinates to map coordinates
        (x,y) = (np.degrees(lon_list), np.degrees(lat_list))   
        plt.scatter(x, y, **kwargs, transform=ccrs.PlateCarree())

        return 0



    def plot(self, lat_list, lon_list, **kwargs):
        """ Plot a curve of given coordinates on the initialized map. 
        
        Arguments:
            lon_list: [list] A list of longitudes (+E in radians) to plot.
            lat_list: [list] A list of latitudes (+N in radians) to plot.

        """
        # Convert coordinates to map coordinates
        (x, y) = (np.degrees(lon_list), np.degrees(lat_list))
        plt.plot(x,y, transform=ccrs.PlateCarree())

        return 0




if __name__ == "__main__":


    # Generate some geo coords
    lat_list = np.linspace(np.radians(37.4), np.radians(37.7), 10)
    lon_list = np.linspace(np.radians(-122.5), np.radians(-122.9), 10)
    

    # Test the ground plot function
    m = OSMMap(lat_list, lon_list, color_scheme='light')

    pl.scatter(lat_list, lon_list)

    pl.plot(lat_list, lon_list)

    plt.show()