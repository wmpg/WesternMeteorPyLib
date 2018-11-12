import sys

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

def readnCDF(file_name):
    """ reads netCDF file

    """
    ensemble_no = True

    dataset = Dataset(file_name, "r+", format="NETCDF4")
    # print(file_name)
    # print(dataset)
    # print(dataset.variables)

    lon = np.array(dataset.variables['longitude'][:])
    lat = np.array(dataset.variables['latitude'][:])

    level = np.array(dataset.variables['level'][:])
    #pressure 1 - 1000 hPa , non-linear
    
    time = np.array(dataset.variables['time'][:])
    #not known

    # time, (number), level, lat, lon
    T = np.array(dataset.variables['t'][0, 0, 0, 0])
    u = np.array(dataset.variables['u'][0, 0, 0, 0])
    v = np.array(dataset.variables['v'][0, 0, 0, 0])

    if ensemble_no == True:
        x = np.arange(0,10,1)
        y = np.arange(0,10,1)

        number = np.array(dataset.variables['number'])
        T = np.array(dataset.variables['t'][0, 0, 0, 0:10, 0:10])
        u = np.array(dataset.variables['u'][0, 0, 0, 0, 0])
        v = np.array(dataset.variables['v'][0, 0, 0, 0, 0])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.contourf(x, y, T, 100, cmap='inferno', alpha=1.0)
    plt.show()
    # # Conversions
    # temps = (consts.GAMMA*consts.R/consts.M_0*temperature[:])**0.5

    # # Magnitude of winds (m/s)
    # mags = np.sqrt(x_wind**2 + y_wind**2)

    # # Direction the winds are coming from, angle in radians from North due East
    # dirs = (np.arctan2(-y_wind, -x_wind))%(2*np.pi)*180/np.pi
    # dirs = wmpl.Supracenter.angleConv.angle2NDE(dirs)*np.pi/180

    # # Store data in a list of arrays
    # store_data = [latitude, longitude, temps, mags, dirs, height]

    # return store_data

if __name__ == '__main__':
    #readnCDF("/home/luke/Desktop/StubenbergReanalysis.nc")
    readnCDF("/home/luke/Desktop/StubenbergEnsembleMembers.nc")
    #readnCDF("/home/luke/Desktop/StubenbergEnsembleMean.nc")
    #readnCDF("/home/luke/Desktop/StubenbergEnsembleSpread.nc")
