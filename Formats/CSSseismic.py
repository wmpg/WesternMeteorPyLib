""" Seismic CSS3 data format. """

from __future__ import print_function, division, absolute_import

import os
import sys

import numpy as np

from Utils.TrajConversions import unixTime2Date


class LineReader(object):
    def __init__(self, line):
        """ Enables reading a string like a file. """

        self.count = 0
        self.line = line


    def read(self, n=None):
        """ Read n characters from the line. """

        old_count = self.count

        if n is None:
            self.count = len(self.line)
            return self.line[old_count:]

        self.count += n

        return self.line[old_count:self.count]




class WfdiscStruct(object):
    def __init__(self):
        """ Structure containing data read from a CSS3 type .wfdisc file. """

        self.sta = None
        self.chan = None
        self.time = None
        self.begin_time = None
        self.wfid = None
        self.chanid = None
        self.jdate = None
        self.endtime = None
        self.nsamp = None
        self.samprate = None
        self.calib = None
        self.calper = None
        self.instype = None
        self.segtype = None
        self.datatype = None
        self.clip = None
        self.dir = None
        self.dfile = None
        self.foff = None
        self.commid = None
        self.lddate = None



class SiteStruct(object):
    def __init__(self):
        """ Structure for the CSS3 type .site file. """

        self.sta = None
        self.ondate = None
        self.offdate = None
        self.lat = None
        self.lon = None
        self.elev = None
        self.staname = None
        self.statype = None
        self.refsta = None
        self.dnorth = None
        self.deast = None
        self.lddate = None



def readWfdisc(dir_path, file_name):
    """ Read a .wfdisc file. 

    More info: ftp://ftp.pmel.noaa.gov/newport/lau/tphase/data/css_wfdisc.pdf

    Arguments:
        dir_path: [str] Path to the directory where the .wfdisc file is located.
        file_name: [str] Name of the .wfdisc file.


    """

    
    wfdisc_list = []

    with open(os.path.join(dir_path, file_name)) as fid:

        for line in fid:

            w = WfdiscStruct()

            f = LineReader(line)
        
            w.sta = f.read(6).strip()
            w.chan = f.read(9).strip()

            # Start time of the record in seconds from 1970/1/1 00:00:00
            w.time = float(f.read(18))

            # Calculate the datetime of the beginning time
            ts = int(w.time)
            tu = int((w.time - int(w.time))*1000000)
            w.begin_time = unixTime2Date(ts, tu, dt_obj=True)

            w.wfid = int(f.read(9))
            w.chanid = int(f.read(9))

            w.jdate = int(f.read(9))
            w.endtime = float(f.read(19))

            # Total number of samples
            w.nsamp = int(f.read(9))

            # Samples per second
            w.samprate = float(f.read(11))

            w.calib = float(f.read(17))
            w.calper = float(f.read(17))

            w.instype = f.read(7)

            w.segtype = f.read(2).strip()

            w.datatype = f.read(3).strip()

            w.clip = f.read(2).strip()

            w.dir = f.read(65).strip()

            w.dfile = f.read(33).strip()

            # Record offset from the beginning of the file
            w.foff = f.read(11)

            w.commid = f.read(9)

            w.lddate = f.read().strip()


            wfdisc_list.append(w)

            # print(w.sta)
            # print(w.chan)
            # print(w.time)
            # print(w.wfid)
            # print(w.chanid)

            # print(w.jdate)
            # print(w.endtime)

            # print(w.nsamp)

            # print(w.segtype)

            # print(w.clip)

            # print(w.dir)

            # print(w.dfile)

            # print(w.foff)

            # print(w.lddate)

        return wfdisc_list



def readSite(dir_path, file_name):
    """ Read a CSS3 type .site file. 

    More info: ftp://ftp.pmel.noaa.gov/newport/lau/tphase/data/css_wfdisc.pdf

    Arguments:
        dir_path: [str] Path to the directory where the .site file is located.
        file_name: [str] Name of the .site file.

    """

    site_list = []

    with open(os.path.join(dir_path, file_name)) as fid:

        for line in fid:

            s = SiteStruct()

            f = LineReader(line)


            s.sta = f.read(6).strip()
            
            s.ondate = int(f.read(9))
            s.offdate = int(f.read(9))

            # Geo coordinates (radians and meters)
            s.lat = np.radians(float(f.read(10)))
            s.lon = np.radians(float(f.read(10)))
            s.elev = float(f.read(10))*1000

            s.staname = f.read(51).strip()
            s.statype = f.read(5).strip()
            s.refsta = f.read(7).strip()

            s.dnorth = float(f.read(10))
            s.deast = float(f.read(10))

            s.lddate = f.read().strip()


            site_list.append(s)

            # print(s.lat, s.lon, s.elev)
            # print(s.staname)
            # print(s.dnorth)
            # print(s.deast)

        return site_list




def readWdata(dir_path, file_name, begin_time, sps, nsamples=-1, datatype='s4', offset=0):
    """ Read the binary .w file containing waveform data.

    More info: ftp://ftp.pmel.noaa.gov/newport/lau/tphase/data/css_wfdisc.pdf

    Arguments:
        dir_path: [str] Path to the directory where the .w file is located.
        file_name: [str] Name of the .w file.
        begin_time: [datetime] Datetime of the beginning time.
        sps: [float] Samples per second.

    Keyword arguments:
        nsamples: [int] Number of samples to read. By default, all samples will be read.
        datatype: [str] Type of data in the .w file. 32 bit signed integer is used by default.

    """

    # Determine the proper data type
    # WARNING, for now, only 32 bit signed integers are supported!
    if datatype == 's4':
        data_type = np.dtype('>i4')

    else:
        print('ERROR!', datatype, 'data type not supported!')
        sys.exit()


    with open(os.path.join(dir_path, file_name)) as f:

        # Skip to the offset
        f.seek(int(offset/data_type.itemsize))

        # Read the rest of the data
        data = np.fromfile(f, dtype=data_type, count=nsamples)

        return data




def loadCSSseismicData(dir_path, site_file, wfdisc_file):
    """ Load the data from a CSS seismic data format and returns a list containing infromation about
        the stations and the waveform data itself. 
    
    Arguments:
        dir_path: [str] Path to the directory where the .w file is located.
        site_file: [str] Name of the .site file.
        wfdisc_file: [str] Name of the .wfdisc file.
    
    Return:
        [list] A list of (SiteStruct, WfdiscStruct, time_data, seismic_data) entries. The time is in seconds,
            referent to the beginning time defined in the Wfdisc object (begin_time).
    """

    # Load the .site file
    site_list = readSite(dir_path, site_file)

    # Load the .wfdisc file
    wfdisc_file = readWfdisc(dir_path, wfdisc_file)


    data_pairs = []

    # Pair sites with their wfdisc entries and load data from .w files
    for site in site_list:
        for w in wfdisc_file:

            # Find the proper pair for the site entry in the wfdisc file
            if site.sta == w.sta:

                # Load the raw data from the .w file
                seismic_data = readWdata(os.path.join(dir_path, w.dir), w.dfile, w.begin_time, w.samprate, \
                    nsamples=w.nsamp)

                # Create the time array
                time_data = np.arange(0, len(seismic_data)/w.samprate, 1/w.samprate)


                data_pairs.append([site, w, time_data, seismic_data])


    return data_pairs




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Test CSS file
    dir_path = "/local4/infrasound/Infrasound/Fireball/15-Sep-2007/seismic/UBh2"
    wfdisc_file = "UBh2.wfdisc"
    site_file = "UBh2.site"


    # Load the seismic data
    seismic_data = loadCSSseismicDate(dir_path, site_file, wfdisc_file)


    # Determine the earliest time from all beginning times
    ref_time = min([w.begin_time for _, w, _, _ in seismic_data])

    # Setup the plotting
    f, axes = plt.subplots(nrows=len(seismic_data), ncols=1, sharex=True)

    for i, entry in enumerate(seismic_data):

        # Select the current axis for plotting
        ax = axes[i]

        # Unpack the loaded seismic data
        site, w, time_data, waveform_data = entry


        # Calculate the difference from the referent time
        t_diff = (w.begin_time - ref_time).total_seconds()

        # Offset the time data to be in accordance with the referent time
        time_data += t_diff


        ax.plot(time_data, waveform_data, zorder=3)

        ax.grid(color='0.9')

    plt.subplots_adjust(hspace=0)
    plt.show()