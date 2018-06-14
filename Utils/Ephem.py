""" Functions for ephemerid calculations, sunrise, sunsert, etc. 
"""

from __future__ import print_function, absolute_import, division

import datetime

import numpy as np
import ephem



def astronomicalNight(night_dt, lat, lon, elevation):
    """ Calculate the beginning and the ending time of the night (when the sun is below 18 degrees of the 
        horizon at the given date and time.
    
    Arguments:
        night_dt: [datetime object] Date and time of the given night.
        lat: [float] Latitude +N of the observer (radians).
        lon: [float] Longitude +E of the observer (radians)
        elevation: [float] Observer's elevation above sea level (meters).
    
    Return:
        (night_start, night_end):
            - night_start: [datetime object] Start of the astronomical night.
            - night_end: [datetime object] End of the astronomical night.
    """

    # If the given date is before noon (meaning it is after midnight), use the reference date will be the 
    # previous day's noon
    if night_dt.hour < 12:
        night_dt = night_dt - datetime.timedelta(days=1)
        
    # Use the current nooon as the reference time for calculating next sunrise/sunset
    night_dt = datetime.datetime(night_dt.year, night_dt.month, night_dt.day, 12, 0, 0)

    # Convert geo coordinates to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    # Initialize the observer
    o = ephem.Observer()  
    o.lat = str(lat)
    o.long = str(lon)
    o.elevation = elevation
    o.date = night_dt

    # The astronomical night starts when the Sun is 18 degrees below the horizon
    o.horizon = '-18:00'

    # Calculate the locations of the Sun
    s = ephem.Sun()  
    s.compute()

    
    # Calculate the time of next sunrise and sunset (local time)    
    # night_start = ephem.localtime(o.next_setting(s))
    # night_end = ephem.localtime(o.next_rising(s))

    # Calculate the time of next sunrise and sunset (UTC)
    night_start = o.next_setting(s).datetime()
    night_end = o.next_rising(s).datetime()
    

    return night_start, night_end
    



if __name__ == "__main__":
        

    ### TEST ###

    night_dt = datetime.datetime(2012, 12, 11, 18, 0, 0)

    
    print(astronomicalNight(night_dt, np.radians(43), np.radians(-81), 324.0))
