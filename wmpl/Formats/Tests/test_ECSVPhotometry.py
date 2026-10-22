""" Tests that loadECSVs() gives no magnitude to the points SkyFit2 picked without photometry: it sets their
integrated pixel value to 1, which leaves their magnitude at the photometric offset.

Run under pytest, or directly:

    python -m wmpl.Formats.Tests.test_ECSVPhotometry
"""

import os
import tempfile

import numpy as np

from wmpl.Formats.ECSV import loadECSVs


# The header SkyFit2 (RMS) writes, with the station of the points
HEADER = """# %ECSV 0.9
# ---
# datatype:
# - {name: datetime, datatype: string}
# - {name: ra, unit: deg, datatype: float64}
# - {name: dec, unit: deg, datatype: float64}
# - {name: azimuth, datatype: float64}
# - {name: altitude, datatype: float64}
# - {name: x_image, unit: pix, datatype: float64}
# - {name: y_image, unit: pix, datatype: float64}
# - {name: integrated_pixel_value, datatype: int64}
# - {name: background_pixel_value, datatype: int64}
# - {name: saturated_pixels, datatype: bool}
# - {name: mag_data, datatype: float64}
# - {name: err_minus_mag, datatype: float64}
# - {name: err_plus_mag, datatype: float64}
# - {name: snr, datatype: float64}
# delimiter: ','
# meta: !!omap
# - {obs_latitude: 45.0}
# - {obs_longitude: 15.0}
# - {obs_elevation: 200.0}
# - {camera_id: 'XX0001'}
# schema: astropy-2.0
"""
COLUMNS = "datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value,background_pixel_value," \
    "saturated_pixels,mag_data,err_minus_mag,err_plus_mag,snr\n"

# Points: integrated pixel value and magnitude. The first three were picked without photometry
POINTS = [(1, 9.52), (1, 9.55), (1, 9.49), (52000, -2.10), (81000, -2.58), (120000, -3.01), (90000, -2.70),
    (40000, -1.82)]


def _writeECSV(dir_path, columns=COLUMNS, points=POINTS):
    """ Write an ECSV of a meteor moving across the sky at 25 frames per second. """

    rows = []
    for i, (intensity, mag) in enumerate(points):
        row = ["2024-01-01T00:00:{:09.6f}".format(10 + i*0.04), "{:10.6f}".format(100.0 + i*0.2),
            "{:+10.6f}".format(20.0 + i*0.1), "{:10.6f}".format(90.0 + i*0.3), "{:+10.6f}".format(40.0 - i*0.2),
            "{:9.3f}".format(300.0 + 5*i), "{:9.3f}".format(200.0 + 3*i), "{:10d}".format(intensity),
            "{:10d}".format(500), "False", "{:+7.2f}".format(mag), "-0.10", "+0.10",
            "{:10.2f}".format(0.0 if intensity <= 1 else 50.0)]

        # Without the photometry columns, as older files
        if "integrated_pixel_value" not in columns:
            row = row[:7] + [row[10]]

        rows.append(",".join(row))

    path = os.path.join(dir_path, "test.ecsv")
    with open(path, "w") as f:
        f.write(HEADER + columns + "\n".join(rows) + "\n")

    return path


def test_points_without_photometry_have_no_magnitude():

    with tempfile.TemporaryDirectory() as tmp_dir:
        _, meteor_list = loadECSVs([_writeECSV(tmp_dir)], no_prepare=True)

    mag_data = np.array(meteor_list[0].mag_data, dtype=np.float64)

    assert np.all(np.isnan(mag_data[:3]))
    assert np.allclose(mag_data[3:], [mag for _, mag in POINTS[3:]])


def test_files_without_the_photometry_columns_keep_their_magnitudes():

    columns = "datetime,ra,dec,azimuth,altitude,x_image,y_image,mag_data\n"
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, meteor_list = loadECSVs([_writeECSV(tmp_dir, columns=columns)], no_prepare=True)

    assert np.allclose(np.array(meteor_list[0].mag_data, dtype=np.float64), [mag for _, mag in POINTS])


if __name__ == "__main__":
    test_points_without_photometry_have_no_magnitude()
    test_files_without_the_photometry_columns_keep_their_magnitudes()
    print("All ECSVPhotometry checks passed.")
