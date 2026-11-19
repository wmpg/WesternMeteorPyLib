""" Tests of reading the header and the rows of ECSV files with loadECSVs(): the metadata keys are matched
    exactly and their values may have colons.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVLoader.py -v
"""

from __future__ import print_function, division, absolute_import

import json
import os
import tempfile

from wmpl.Formats.ECSV import loadECSVs


# Columns, station metadata and times of the files
COLUMNS = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image', 'mag_data']
STATION = [('obs_latitude', '45.0'), ('obs_longitude', '15.0'), ('obs_elevation', '200.0')]
TIMES = ["2024-01-01T00:00:{:09.6f}".format(10 + 0.04*i) for i in range(8)]



def writeECSV(path, meta, columns, times, with_mag=True):
    """ Write an ECSV file with the given metadata and a meteor moving across the sky.

    Arguments:
        path: [str] Path of the file.
        meta: [list] Metadata entries, as (key, value as written).
        columns: [list] Names of the columns.
        times: [list] Times of the rows, as written.

    Keyword arguments:
        with_mag: [bool] Write the magnitude column. True by default.
    """

    # Write the metadata
    lines = ["# %ECSV 0.9", "# ---", "# delimiter: ','", "# meta: !!omap"]
    lines += ["# - {{{:s}: {:s}}}".format(key, value) for key, value in meta]

    # Write the header and one row per time
    names = [name for name in columns if with_mag or (name != 'mag_data')]
    lines.append(",".join(names))

    for i, time in enumerate(times):

        values = {'datetime': time, 'ra': "0.0", 'dec': "0.0", 'azimuth': "{:.6f}".format(90.0 + 0.3*i), \
            'altitude': "{:.6f}".format(40.0 - 0.2*i), 'x_image': "{:.3f}".format(300.0 + 5*i), \
            'y_image': "{:.3f}".format(200.0 + 3*i), 'mag_data': "{:+.2f}".format(-2.0 - 0.1*i)}
        lines.append(",".join(values[name] for name in names))

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")



def testHeaderKeysAreMatchedExactlyAndValuesKeepColons():
    """ Check that the station is not taken from a key which only contains camera_id, and that a value with
        colons is kept whole.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        # A key which contains camera_id, and a Windows path of the image file
        path = os.path.join(tmp_dir, "a.ecsv")
        meta = STATION + [('camera_id', "'XX0001'"), ('camera_id_alias', "'WRONG'"), \
            ('image_file', "'C:\\data\\FF_XX0001_20240101_000010.fits'")]
        writeECSV(path, meta, COLUMNS, TIMES)

        _, meteor_list = loadECSVs([path], no_prepare=True)

    meteor = meteor_list[0]
    assert meteor.station_id == 'XX0001'
    ff_name = json.loads(meteor.ff_name.replace('\\', '/'))['ff_name']
    assert ff_name == 'C:/data/FF_XX0001_20240101_000010.fits'



if __name__ == "__main__":

    testHeaderKeysAreMatchedExactlyAndValuesKeepColons()

    print("All tests passed.")
