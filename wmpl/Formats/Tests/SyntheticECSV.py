""" ECSV files with several fragments, as in Appendix 4 of the GFE standard, shared by the tests of the
    fragments of ECSV files.
"""

from __future__ import print_function, division, absolute_import


# Metadata of a station, with STATION replaced by its code
META = """# %ECSV 0.9
# ---
# meta: !!omap
# - {obs_latitude: -43.6}
# - {obs_longitude: 172.1}
# - {obs_elevation: 100.0}
# - {camera_id: 'STATION'}
# - {image_file: 'FF_STATION.fits'}
"""

# Columns of the main fragment and of an additional one (with its number as a suffix)
MAIN = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image', 'mag_data']
FRAGMENT = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image']



def fragmentPoint(i, altitude):
    """ Make the time and position of a point on a frame.

    Arguments:
        i: [int] Frame number.
        altitude: [float] Altitude of the point on the first frame (deg).

    Return:
        [list] Values of the FRAGMENT columns, as strings.
    """

    return ['2026-01-30T10:25:36.{:06d}'.format(40000*i), '10.0', '-30.0', '{:.4f}'.format(120 + 0.5*i), \
        '{:.4f}'.format(altitude - 0.3*i), '{:.3f}'.format(100 + 10*i), '{:.3f}'.format(200 + 5*i)]



def writeFragmentECSV(path, station, fragments=True):
    """ Write an ECSV file with six frames of the main fragment, the third flagged as not to be used and the
        last one only measured for fragment 1, and fragment 1 on the last five frames.

    Arguments:
        path: [str] Path of the file.
        station: [str] Station code.

    Keyword arguments:
        fragments: [bool] Write the columns of fragment 1 and the flags. If False, the file has the format
            without them, with the six frames of the main fragment. True by default.

    Return:
        [str] Path of the file.
    """

    header = MAIN
    if fragments:
        header = MAIN + ['frame_number', 'flare', 'trajectory_use'] + [name + '1' for name in FRAGMENT]

    rows = []
    for i in range(7):

        if i < 6:
            main = fragmentPoint(i, 30.0) + ['{:.2f}'.format(2.0 - 0.1*i)]
        else:
            main = ['']*len(MAIN)

        if not fragments:
            if i < 6:
                rows.append(main)

            continue

        if i < 6:
            flags = [str(i), 'False', 'False' if (i == 2) else 'True']
        else:
            flags = [str(i), '', '']

        fragment = fragmentPoint(i, 29.0) if (i >= 2) else ['']*len(FRAGMENT)
        rows.append(main + flags + fragment)

    with open(path, 'w') as f:
        f.write(META.replace('STATION', station) + ",".join(header) + "\n")
        f.write("".join(",".join(row) + "\n" for row in rows))

    return str(path)
