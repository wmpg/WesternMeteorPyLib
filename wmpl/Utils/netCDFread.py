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
    #0 - 359.5 in steps of 0.5
    lat = np.array(dataset.variables['latitude'][:])
    #90 - -90 in steps of 0.5


    level = np.array(dataset.variables['level'][:])
    #pressure 1 - 1000 hPa , non-linear
    
    time = np.array(dataset.variables['time'][:])
    #not known


    if ensemble_no == True:


        number = np.array(dataset.variables['number'])
        T = np.array(dataset.variables['t'][0, :, :, 90, 90])
        u = np.array(dataset.variables['u'][0, :, :, 90, 90])
        v = np.array(dataset.variables['v'][0, :, :, 90, 90])

    else:
        # time, (number), level, lat, lon
        T = np.array(dataset.variables['t'][0, :, 90, 90])
        u = np.array(dataset.variables['u'][0, :, 90, 90])
        v = np.array(dataset.variables['v'][0, :, 90, 90])

    mag = np.sqrt(u**2 + v**2)
    print(np.flip(mag, axis=0))
    #keep this as a function later
    # dim = 10
    # x = np.arange(0,dim,1)
    # x = np.repeat(x, dim**2)
    # y = np.arange(0,dim,1)
    # y = np.repeat(y, dim)
    # y = np.tile(y, dim)
    # z = np.arange(0,dim,1)
    # z = np.tile(z, dim**2)
    # t = T.reshape(-1)

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # sc = ax1.scatter(x, y, z, c=t, cmap='inferno', alpha=1.0)
    # #a = plt.colorbar(sc, ax=ax1)
    # #plt.contourf(x, y, u, 100, cmap='inferno', alpha=1.0)
    # plt.show()
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
