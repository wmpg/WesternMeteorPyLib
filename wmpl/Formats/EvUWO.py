""" I/O functions for the UWO ev_* file format. """

from __future__ import print_function, division, absolute_import


import os

import numpy as np

from wmpl.Utils.TrajConversions import jd2Date, jd2UnixTime


def writeEvFile(dir_path, file_name, jdt_ref, station_id, lat, lon, ele, time_data, theta_data, phi_data):
    """ Write a UWO style ev_* file.

    Arguments:
        dir_path: [str] Path to the directory where the file will be saved.
        file_name: [str] Name of the file.
        jdt_ref: [float] Julian date for which the time in time_data is 0.
        station_id: [str] Name of the station
        lat: [float] Latitude +N of the station (radians).
        lon: [float] Longitude +E of the station, (radians).
        ele: [float] Height above sea level (meters).
        time_data: [list of floats] A list of times of observations in seconds, where t = 0s is at jdt_ref.
        theta_data: [list of floats]: A list of zenith angles of observations (radians)
        phi_data: [list of floats] A list of azimuth (+N of due E) of observations (radians).

    """

    # Convert Julian date to date string
    year, month, day, hour, minute, second, millisecond = jd2Date(jdt_ref)
    date_str = "{:4d}{:02d}{:02d} {:02d}:{:02d}:{:02d}.{:03d}".format(year, month, day, hour, minute, second, \
        int(millisecond))

    # Convert JD to unix time
    unix_time = jd2UnixTime(jdt_ref)


    with open(os.path.join(dir_path, file_name), 'w') as f:

        f.write("#\n")
        f.write("#   version : WMPL\n")
        f.write("#    num_fr : {:d}\n".format(len(time_data)))
        f.write("#    num_tr : 1\n")
        f.write("#      time : {:s} UTC\n".format(date_str))
        f.write("#      unix : {:f}\n".format(unix_time))
        f.write("#       ntp : LOCK 83613 1068718 130\n")
        f.write("#       seq : 0\n")
        f.write("#       mul : 0 [A]\n")
        f.write("#      site : {:s}\n".format(station_id))
        f.write("#    latlon : {:9.6f} {:+10.6f} {:.1f}\n".format(np.degrees(lat), np.degrees(lon), ele))
        f.write("#      text : WMPL generated\n")
        f.write("#    stream : KT\n")
        f.write("#     plate : none\n")
        f.write("#      geom : 0 0\n")
        f.write("#    filter : 0\n")
        f.write("#\n")
        #f.write("#  fr    time    sum     seq       cx       cy     th        phi      lsp     mag  flag   bak    max\n")
        f.write("#  fr    time    sum     seq       cx       cy     th        phi      lsp     mag  flag\n")


        for fr, (t, theta, phi) in enumerate(zip(time_data, theta_data, phi_data)):

            sum_ = 0
            seq = 0
            cx = 0.0
            cy = 0.0
            lsp = 0.0
            mag = 0.0
            flag = "0000"
            bak = 0.0
            max_ = 0.0

            f.write("{:5d} ".format(fr))
            f.write("{:7.3f} ".format(t))
            
            f.write("{:6d} ".format(sum_))
            f.write("{:7d} ".format(seq))
            
            f.write("{:8.3f} ".format(cx))
            f.write("{:8.3f} ".format(cy))
            
            f.write("{:9.5f} ".format(np.degrees(theta)))
            f.write("{:10.5f} ".format(np.degrees(phi)))

            f.write("{:7.3f} ".format(lsp))
            f.write("{:6.2f} ".format(mag))

            f.write("{:5s} ".format(flag))

            #f.write("{:6.2} ".format(bak))
            #f.write("{:6.2} ".format(max_))

            f.write("\n")



if __name__ == '__main__':


    import datetime

    import numpy as np

    from wmpl.Utils.TrajConversions import datetime2JD

    dir_path = '/home/dvida/Desktop'
    file_name = 'ev_test.txt'

    jdt_ref = datetime2JD(datetime.datetime.utcnow())

    station_id = "01"
    lat = np.radians(45.458)
    lon = np.radians(+18.5)
    ele = 90

    time_data = np.linspace(0, 2, 30)
    theta_data = np.linspace(np.pi/4, np.pi/2, 30)
    phi_data = np.linspace(np.pi, 3*np.pi/2, 30)

    writeEvFile(dir_path, file_name, jdt_ref, station_id, lat, lon, ele, time_data, theta_data, phi_data)