""" Configuration for running the library. It is loaded by several modules to read in the default settings.
"""


from __future__ import print_function, division, absolute_import

import os
import urllib

import time
import datetime




def loadLeapSeconds(leap_seconds_file):
    """ To calculate a proper dynamical time, records of leap seconds are needed. As they are not predictable
        and are announced by USNO, the data about them have to be pulled from the USNO website.
        
    Arguments:
        leap_seconds_file: [str] Path to the local leap seconds file.

    Return:
        leap_seconds: [list of tuples] A list of (jd, leap_seconds) pairs.

    """

    # URL where the leap seconds recored are kept
    usno_leap_url = "ftp://maia.usno.navy.mil/ser7/tai-utc.dat" 


    leap_file_loaded = False

    # Check if a local leap seconds file exists and see if it was downloaded within the last 24 hrs
    if os.path.isfile(leap_seconds_file):

        # Load the leap seconds info from file is the file is younger than 24 hrs
        if os.path.getmtime(leap_seconds_file) > (time.time() - 24*60*60):

            with open(leap_seconds_file) as f:
                leap_string = '\n'.join(f.readlines())

                leap_file_loaded = True

                print('Loaded leap seconds from a local file:', leap_seconds_file)
            


    if not leap_file_loaded:

        # Try reading the leap second file from URL
        try:
            leap_string = urllib.urlopen(usno_leap_url).read()
            print('Downloaded leap seconds from:', usno_leap_url)

            # Save the leap seconds file locally
            with open(leap_seconds_file, 'w') as f:
                f.write(leap_string)


        except:

            # If the URL download failed, use the local file if it exists
            if os.path.isfile(leap_seconds_file):

                with open(leap_seconds_file) as f:
                    leap_string = '\n'.join(f.readlines())

            # If the file does not exist, use the hardcoded defaults
            else:

                print('Leap second data could not be downloaded from', usno_leap_url, 'using default values...')

                # If the data cannot be loaded, use the hardcoded values
                leap_string = """ 1961 JAN  1 =JD 2437300.5  TAI-UTC=   1.4228180 S + (MJD - 37300.) X 0.001296 S
         1961 AUG  1 =JD 2437512.5  TAI-UTC=   1.3728180 S + (MJD - 37300.) X 0.001296 S
         1962 JAN  1 =JD 2437665.5  TAI-UTC=   1.8458580 S + (MJD - 37665.) X 0.0011232S
         1963 NOV  1 =JD 2438334.5  TAI-UTC=   1.9458580 S + (MJD - 37665.) X 0.0011232S
         1964 JAN  1 =JD 2438395.5  TAI-UTC=   3.2401300 S + (MJD - 38761.) X 0.001296 S
         1964 APR  1 =JD 2438486.5  TAI-UTC=   3.3401300 S + (MJD - 38761.) X 0.001296 S
         1964 SEP  1 =JD 2438639.5  TAI-UTC=   3.4401300 S + (MJD - 38761.) X 0.001296 S
         1965 JAN  1 =JD 2438761.5  TAI-UTC=   3.5401300 S + (MJD - 38761.) X 0.001296 S
         1965 MAR  1 =JD 2438820.5  TAI-UTC=   3.6401300 S + (MJD - 38761.) X 0.001296 S
         1965 JUL  1 =JD 2438942.5  TAI-UTC=   3.7401300 S + (MJD - 38761.) X 0.001296 S
         1965 SEP  1 =JD 2439004.5  TAI-UTC=   3.8401300 S + (MJD - 38761.) X 0.001296 S
         1966 JAN  1 =JD 2439126.5  TAI-UTC=   4.3131700 S + (MJD - 39126.) X 0.002592 S
         1968 FEB  1 =JD 2439887.5  TAI-UTC=   4.2131700 S + (MJD - 39126.) X 0.002592 S
         1972 JAN  1 =JD 2441317.5  TAI-UTC=  10.0       S + (MJD - 41317.) X 0.0      S
         1972 JUL  1 =JD 2441499.5  TAI-UTC=  11.0       S + (MJD - 41317.) X 0.0      S
         1973 JAN  1 =JD 2441683.5  TAI-UTC=  12.0       S + (MJD - 41317.) X 0.0      S
         1974 JAN  1 =JD 2442048.5  TAI-UTC=  13.0       S + (MJD - 41317.) X 0.0      S
         1975 JAN  1 =JD 2442413.5  TAI-UTC=  14.0       S + (MJD - 41317.) X 0.0      S
         1976 JAN  1 =JD 2442778.5  TAI-UTC=  15.0       S + (MJD - 41317.) X 0.0      S
         1977 JAN  1 =JD 2443144.5  TAI-UTC=  16.0       S + (MJD - 41317.) X 0.0      S
         1978 JAN  1 =JD 2443509.5  TAI-UTC=  17.0       S + (MJD - 41317.) X 0.0      S
         1979 JAN  1 =JD 2443874.5  TAI-UTC=  18.0       S + (MJD - 41317.) X 0.0      S
         1980 JAN  1 =JD 2444239.5  TAI-UTC=  19.0       S + (MJD - 41317.) X 0.0      S
         1981 JUL  1 =JD 2444786.5  TAI-UTC=  20.0       S + (MJD - 41317.) X 0.0      S
         1982 JUL  1 =JD 2445151.5  TAI-UTC=  21.0       S + (MJD - 41317.) X 0.0      S
         1983 JUL  1 =JD 2445516.5  TAI-UTC=  22.0       S + (MJD - 41317.) X 0.0      S
         1985 JUL  1 =JD 2446247.5  TAI-UTC=  23.0       S + (MJD - 41317.) X 0.0      S
         1988 JAN  1 =JD 2447161.5  TAI-UTC=  24.0       S + (MJD - 41317.) X 0.0      S
         1990 JAN  1 =JD 2447892.5  TAI-UTC=  25.0       S + (MJD - 41317.) X 0.0      S
         1991 JAN  1 =JD 2448257.5  TAI-UTC=  26.0       S + (MJD - 41317.) X 0.0      S
         1992 JUL  1 =JD 2448804.5  TAI-UTC=  27.0       S + (MJD - 41317.) X 0.0      S
         1993 JUL  1 =JD 2449169.5  TAI-UTC=  28.0       S + (MJD - 41317.) X 0.0      S
         1994 JUL  1 =JD 2449534.5  TAI-UTC=  29.0       S + (MJD - 41317.) X 0.0      S
         1996 JAN  1 =JD 2450083.5  TAI-UTC=  30.0       S + (MJD - 41317.) X 0.0      S
         1997 JUL  1 =JD 2450630.5  TAI-UTC=  31.0       S + (MJD - 41317.) X 0.0      S
         1999 JAN  1 =JD 2451179.5  TAI-UTC=  32.0       S + (MJD - 41317.) X 0.0      S
         2006 JAN  1 =JD 2453736.5  TAI-UTC=  33.0       S + (MJD - 41317.) X 0.0      S
         2009 JAN  1 =JD 2454832.5  TAI-UTC=  34.0       S + (MJD - 41317.) X 0.0      S
         2012 JUL  1 =JD 2456109.5  TAI-UTC=  35.0       S + (MJD - 41317.) X 0.0      S
         2015 JUL  1 =JD 2457204.5  TAI-UTC=  36.0       S + (MJD - 41317.) X 0.0      S
         2017 JAN  1 =JD 2457754.5  TAI-UTC=  37.0       S + (MJD - 41317.) X 0.0      S"""

        
    leap_data = []

    # Parse leap second data - the first column will be the Julian date when the leap seconds started, the
    # second column is the difference between TAI and UTC in seconds
    for line in leap_string.split('\n'):
        
        line = line.replace('\n', '').replace('\r', '').split()

        if line:

            jd = float(line[4])
            t_diff = float(line[6])

            leap_data.append([jd, t_diff])


    return leap_data






class ConfigStruct(object):

    def __init__(self):

        # Find the absolute path of the directory where this file is located
        abs_path = os.path.abspath(os.path.join(os.path.split(__file__)[0]))

        ### EPHEMERIDS

        # VSOP87 file location
        self.vsop_file = os.path.join(abs_path, 'share', 'VSOP87D.ear')

        # DE430 JPL ephemerids file location
        self.jpl_ephem_file = os.path.join(abs_path, 'share', 'de430.bsp')

        ###


        ### PARENT BODY ORBITAL ELEMENTS

        # JPL comets elements
        self.comets_elements_file = os.path.join(abs_path, 'share', 'ELEMENTS.COMET')

        # Asteroid elements
        self.asteroids_amors_file = os.path.join(abs_path, 'share', 'Amors.txt')
        self.asteroids_apollos_file = os.path.join(abs_path, 'share', 'Apollos.txt')
        self.asteroids_atens_file = os.path.join(abs_path, 'share', 'Atens.txt')

        ###

        # Leap seconds file
        self.leap_seconds_file = os.path.join(abs_path, 'share', 'tai-utc.dat')


        # Leap seconds data
        self.leap_seconds = loadLeapSeconds(self.leap_seconds_file)

        # Meteor simulation default parameters file
        self.met_sim_input_file = os.path.join(abs_path, 'MetSim', 'Metsim0001_input.txt')

        # DPI of saves plots
        self.plots_DPI = 300






# Init the configuration structure
config = ConfigStruct()


### MATPLOTLIB OPTIONS ###
import matplotlib

# Override default DPI for saving from the interactive window
matplotlib.rcParams['savefig.dpi'] = 300

##########################