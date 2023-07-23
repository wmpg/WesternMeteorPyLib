""" Measure meteor shower radiants and create a reference table with solar longitudes and ecliptic coordinates of the radiants that can be used for shower association."""

import os

import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.stats

from wmpl.Formats.WmplTrajectorySummary import loadTrajectorySummaryFast
from wmpl.Utils.Math import isAngleBetween, angleBetweenSphericalCoords

# Vectorize the isAngleBetween function
isAngleBetweenVec = np.vectorize(isAngleBetween, excluded=["left", "right"])


def lgDriftFunc(x, lg_ref, lg_drift):
    """ Linear function for SCE longitude. """

    return (lg_ref + lg_drift*x)%(360)

def lgDriftFuncMini(params, x, lg_meas):
    """ Linear function for SCE longitude. Used for scipy minimize."""
    
    lg_ref, lg_drift = params

    return (lg_ref + lg_drift*x)%(360) - lg_meas


def bgDriftFunc(x, bg_ref, bg_drift):
    """ Linear function for ecliptic latitude. """

    return bg_ref + bg_drift*x

def bgDriftFuncMini(params, x, bg_meas):
    """ Linear function for ecliptic latitude. Used for scipy minimize. """

    bg_ref, bg_drift = params
    return (bg_ref + bg_drift*x) - bg_meas


def fitRadiantDrift(sol_data, sce_lon_data, bg_data, vg_data, sol_lon_ref=None):
    """ Fit the drift of the radiant and velocity of the meteor shower.
    
    Arguments:
        sol_data: [pandas.Series] Solar longitude data (deg).
        sce_lon_data: [pandas.Series] SCE longitude data (deg).
        bg_data: [pandas.Series] Ecliptic latitude data (deg).
        vg_data: [pandas.Series] Geocentric velocity data (km/s).

    Keyword arguments:
        sol_lon_ref: [float] Reference solar longitude (deg). None by default, it will be computed from the 
            data as the circular median.

    Returns:
        sol_lon_ref: [float] Reference solar longitude (deg).
        lg_drift: [list] Radiant drift fit in SCE longitude [slope, intercept].
        lg_drift_std: [list] Radiant drift fit in SCE longitude standard deviation [slope, intercept].
        bg_drift: [list] Radiant drift fit in ecliptic latitude [slope, intercept].
        bg_drift_std: [list] Radiant drift fit in ecliptic latitude standard deviation [slope, intercept].
        vg_drift: [list] Velocity drift fit [slope, intercept].
        vg_drift_std: [list] Velocity drift fit standard deviation [slope, intercept].
    """

    if sol_lon_ref is None:

        # Compute a circular mean of the solar longitude
        sol_lon_mean = np.degrees(scipy.stats.circmean(np.radians(sol_data)))

        # Normalize the sol data to the circular mean
        sol_data_normed = (sol_data - sol_lon_mean + 180)%360 - 180

        # Compute the median and return back to the original reference frame
        sol_lon_ref = np.nanmedian(sol_data_normed) + sol_lon_mean

        # Round the reference solar longitude 0.1 deg
        sol_lon_ref = round(sol_lon_ref, 1)


    # Normalize the solar longitude to the reference solar longitude
    sol_normed = (sol_data - sol_lon_ref + 180)%360 - 180


    # Fit the radiant drift using scipy minimize and absolute errors not squared
    p0_lg = [np.mean(sce_lon_data), 0.0]
    p0_bg = [np.mean(bg_data), 0.0]
    lg_drift, _ = scipy.optimize.curve_fit(lgDriftFunc, sol_normed, sce_lon_data, p0=p0_lg)
    bg_drift, _ = scipy.optimize.curve_fit(bgDriftFunc, sol_normed, bg_data, p0=p0_bg)
    res = scipy.optimize.least_squares(lgDriftFuncMini, lg_drift, args=(sol_normed, sce_lon_data), 
                                       loss="soft_l1")
    lg_drift = res.x
    lg_cov = res.jac
    res = scipy.optimize.least_squares(bgDriftFuncMini, bg_drift, args=(sol_normed, bg_data), 
                                       loss="soft_l1")
    bg_drift = res.x
    bg_cov = res.jac


    # Fit the velocity drift
    p0_vg = [np.mean(vg_data), 0.0]
    vg_drift, _ = scipy.optimize.curve_fit(bgDriftFunc, sol_normed, vg_data, p0=p0_vg)
    res = scipy.optimize.least_squares(bgDriftFuncMini, vg_drift, args=(sol_normed, vg_data), 
                                       loss="soft_l1")
    vg_drift = res.x
    vg_cov = res.jac

    # Compute the standard deviation of the drift
    lg_drift_std = np.sqrt(np.diag(lg_cov))
    bg_drift_std = np.sqrt(np.diag(bg_cov))
    vg_drift_std = np.sqrt(np.diag(vg_cov))

    return sol_lon_ref, lg_drift, lg_drift_std, bg_drift, bg_drift_std, vg_drift, vg_drift_std


def dispersionFunction(sol, sol_ref_beg, disp_beg, sol_ref_mid, disp_mid, sol_ref_end, disp_end, 
                       dispersion_peak_factor=2.0):
    """ 
    A function modelling radiant dispersion. All sol before the sol_ref_beg are set to disp_beg, all sol 
    after sol_ref_end are set to disp_end, and all sol in between are interpolated linearly. 

    The dispersion at the peak is then multiplied by the dispersion_peak_factor, while the edge dispersions are 
    left unchanged. The idea is that the dispersion at the peak is measured the best, while the dispersion 
    at the edges have too much influence from the sporadics and might be inflated.

    Arguments:
        sol: [float] Solar longitude, independent variable (deg).
        sol_ref_beg: [float] Solar longitude of the beginning of the dispersion (deg).
        disp_beg: [float] Dispersion at the beginning of the dispersion (deg).
        sol_ref_mid: [float] Solar longitude of the peak of the dispersion (deg).
        disp_mid: [float] Dispersion at the peak of the dispersion (deg).
        sol_ref_end: [float] Solar longitude of the end of the dispersion (deg).
        disp_end: [float] Dispersion at the end of the dispersion (deg).

    Keyword arguments:
        dispersion_peak_factor: [float] Dispersion factor. The dispersion at the peak is multiplied by this factor
            while the edge dispersions are left unchanged. Default is 2.0.
    """
    
    # Compute the angular difference between the sol and the reference sol
    sol_diff = sol - sol_ref_mid
    sol_diff = (sol_diff + 180)%360 - 180

    # Compute the difference between the sol beg and the reference sol
    sol_ref_diff_beg = sol_ref_beg - sol_ref_mid
    sol_ref_diff_beg = (sol_ref_diff_beg + 180)%360 - 180

    # Compute the difference between the sol end and the reference sol
    sol_ref_diff_end = sol_ref_end - sol_ref_mid
    sol_ref_diff_end = (sol_ref_diff_end + 180)%360 - 180

    # Multipy the peak dispersion by the dispersion factor
    disp_mid = disp_mid*dispersion_peak_factor

    # Compute the dispersion for the given sol

    # For sol before the sol_ref_beg, set the dispersion to disp_beg
    if sol_diff < sol_ref_diff_beg:
        disp = disp_beg

    # For sol after the sol_ref_end, set the dispersion to disp_end
    elif sol_diff > sol_ref_diff_end:
        disp = disp_end

    # For sol between sol_ref_beg and sol_ref_mid, interpolate linearly between disp_beg and disp_mid
    elif sol_diff < 0:
        disp = disp_mid + (-sol_diff/sol_ref_diff_beg)*(disp_mid - disp_beg)

    # For sol between sol_ref_mid and sol_ref_end, interpolate linearly between disp_mid and disp_end
    else:
        disp = disp_mid + (sol_diff/sol_ref_diff_end)*(disp_end - disp_mid)

        

    # Return the dispersion multiplied by the dispersion factor
    return disp

# Vectorized version of the dispersion function (sol is a numpy array)
dispersionFunctionVec = np.vectorize(dispersionFunction, excluded=["sol_ref_beg", "disp_beg", "sol_ref_mid", 
                                                                   "disp_mid", "sol_ref_end", "disp_end", 
                                                                   "dispersion_peak_factor"])
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import scipy.optimize

    #################

    show_plots = False

    # Path to the raw trajectory summary file
    dir_path_traj_summary = "C:\\temp\\traj_summary"

    # Name of the trajectory summary file
    traj_summary_file_name = "traj_summary_all_20230518.txt"

    # Name of the pickle file for quick loading of the trajectory data
    quick_file_name = "traj_summary_all_20230518.pkl"

    # Plot directory
    plot_dir = "plots"

    # Minimum number of meteors per shower for the fit
    min_meteors_per_shower = 10

    ## DISPERSION ###

    # If there are less than this number of meteors in the edge bin, set a fixed dispersion
    min_meteors_edge_bin = 50
    fixed_edge_dispersion = 3.0

    # Minimum dispersion for the edges (deg)
    min_dispersion = 1.5

    # Minimum dispersion at the peak (deg)
    min_peak_dispersion = 2.0

    # Maximum dispersion at the peak (deg)
    max_peak_dispersion = 3.5

    ##

    # Shower association velocity percentage
    shower_assoc_vel_perc = 10

    # Solar longitude delta for sampling the table (deg)
    sol_delta = 0.2

    # Median dispersion mulitplier
    dispersion_peak_factor = 2.5

    # New lookup table file name
    new_table_file_name = "gmn_shower_table.txt"

    # Full parameter fit table file name
    full_param_fit_table_file_name = "gmn_shower_table_full_param_fit.txt"


    # Some showers have a very long period of activity and the linear drift doesn't work well for them
    # We need to split those showers into multiple parts and fit each part separately
    multi_part_showers = {}
    # Range of sols for each part
    multi_part_showers["KCG"] = [
        [90, 125],
        [125, 150],
        [150, 170],
    ]
    multi_part_showers["AUD"] = [
        [130, 148],
        [148, 165]
    ]
    # known_showers["ETA"] = [ 30  ,  46  ,  76  ]
    multi_part_showers["ETA"] = [
        [30, 57],
        [57, 77]
    ]


    # List of showers with measured activity periods
    known_showers = {}
        # Name, sol_beg, sol_peak, sol_end
    known_showers["CAP"] = [110  , 127.1, 135  ]
    known_showers["STA"] = [190  , 216  , 230  ]
    known_showers["GEM"] = [252  , 262  , 266  ]
    known_showers["SDA"] = [114  , 125.5, 145  ]
    known_showers["LYR"] = [ 30  ,  32.4,  34  ]
    known_showers["PER"] = [115  , 140.4, 150  ]
    known_showers["ORI"] = [190  , 209  , 225  ]
    known_showers["DRA"] = [194  , 195.5, 197  ]
    known_showers["QUA"] = [281  , 283.0, 285  ]
    # known_showers["KCG"] = [130  , 145  , 152.2]
    known_showers["LEO"] = [224  , 235  , 245  ]
    known_showers["URS"] = [269  , 270.4, 272  ]
    known_showers["HYD"] = [238  , 256  , 270  ]
    known_showers["NTA"] = [217  , 224.5, 241  ]
    known_showers["MON"] = [250  , 261  , 266  ]
    known_showers["LMI"] = [199  , 208.7, 221  ]
    #known_showers["ETA"] = [ 30  ,  46  ,  76  ]
    known_showers["SSG"] = [ 71  ,  86  ,  96  ]
    known_showers["ARI"] = [ 62  ,  78.5,  99  ]
    known_showers["PAU"] = [124  , 136  , 142  ]
    known_showers["AUR"] = [154.2, 158.6, 167.7]
    known_showers["SPE"] = [161.9, 166.8, 178.4]
    known_showers["AND"] = [215  , 239  , 249  ]
    known_showers["COM"] = [253  , 263  , 290  ]
    known_showers["GDR"] = [122  , 125.5, 128  ]
    known_showers["ERI"] = [118  , 137  , 145  ]
    known_showers["NOO"] = [233  , 245  , 255  ]
    known_showers["OCT"] = [191  , 192.5, 194  ]
    known_showers["AHY"] = [278  , 285  , 290  ]
    known_showers["AND"] = [215  , 245.9, 250  ]
    known_showers["CAM"] = [ 60  ,  63.5,  65  ]
    known_showers["TAH"] = [ 66.7,  69.4,  71  ]
    known_showers["JBO"] = [ 88  ,  92.4,  99  ]
    known_showers["ADC"] = [143  , 143.7, 144.5]
    known_showers["SOA"] = [194  , 198.6, 202  ]
    known_showers["JBO"] = [ 88  ,  90.2,  92  ]
    known_showers["ARI"] = [ 70  ,  78.5,  85  ]
    known_showers["HVI"] = [ 35  ,  40  ,  42  ]
    known_showers["LLY"] = [ 38  ,  44  ,  55  ]
    known_showers["ACE"] = [310  , 314.2, 320  ]
    known_showers["DSX"] = [182  , 189  , 194  ]
    known_showers["JRD"] = [ 72  ,  74  , 75   ]
    known_showers["NET"] = [223.5, 225.5, 227.5]
    known_showers["JEO"] = [ 80,    90.8,  94.0]


    # Parameters of additional showers
    additional_shower_params = {}
        # IAU code, IAU No, Sol beg, Sol peak, Sol end, SCElon ref, SCElon drift, BET ref, BET drift,  Vg ref, Vg drift,  disp
    additional_shower_params["BTU"] = {
             "BTU",    108,  352.15,   352.25,  352.52,    306.365,       -4.627, -76.865,    -1.433,  30.549,   -2.122,  1.95
    }
    additional_shower_params["OZP"] = {
             "OZP",   1131,  211.24,   211.35,  211.42,    211.911,        7.596,  13.290,    -4.576,  47.353,   22.628,  1.04
    }
    additional_shower_params["OCR"] = {
             "OCR",   1033,  290.61,   290.75,  290.96,    295.459,        1.043, -68.539,     1.889,  40.158,    6.532,  1.52
    }

    # Additional showers with measured activity periods (no need to sort)
    additional_shower_entries = [
        "352.1  306.842  -76.717   30.768  1.95  108 BTU",
        "352.3  305.917  -77.004   30.344  1.95  108 BTU",
        "211.2  211.079   13.790   44.877  1.04 1131 OZP",
        "185.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "185.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "186.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "186.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "187.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "187.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "188.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "188.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "189.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "189.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "190.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "190.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "191.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "191.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "192.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "192.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "193.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "193.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "194.0   77.70   -34.50    10.70   3.00 1130 ARD",
        "194.0   65.1    -26.6     11.00   3.00 1130 ARD",
        "290.6  295.308  -68.813   39.212  1.52 1033 OCR",
        "290.8  295.517  -68.435   40.518  1.52 1033 OCR"
    ]

    # Get a list of additional shower codes
    additional_shower_codes = [entry.split()[-1] for entry in additional_shower_entries]
    additional_shower_codes += additional_shower_params.keys()
    additional_shower_codes = list(set(additional_shower_codes))

    #####################


    # Load the data from disk
    data = loadTrajectorySummaryFast(dir_path_traj_summary, traj_summary_file_name, quick_file_name)

    # Compute the total angular error for LAMgeo and BETgeo
    data["RadiantError (deg)"] = np.degrees(angleBetweenSphericalCoords(
        np.radians(data["BETgeo (deg)"]), 
        np.radians(data["LAMgeo (deg)"]), 
        np.radians(data["BETgeo (deg)"] + data["+/- (sigma.3)"]),
        np.radians(data["LAMgeo (deg)"] + data["+/- (sigma.2)"]) 
        ))

    # Remove all data with the radiant error larger than 2 deg and the velocity error larger than 5%
    data = data[
          (data["RadiantError (deg)"] < 2.0) 
        & (data["+/- (sigma.4)"] < 0.05*data["Vgeo (km/s)"])
        ]

    # Compute the SCE longitude (LAMg - solar longitude)
    data["LAMsce (deg)"] = (data["LAMgeo (deg)"] - data["Sol lon (deg)"])%360


    # Print the columns
    print(data.columns)



    ### Get a list of unique shower codes and sort them in the ascending order according to the IAU number ###

    # Extract only the shower codes and IAU numbers
    shower_code_pairs = data[["IAU (code)", "IAU (No)"]]

    # Remove -1 from the shower numbers (sporadics)
    shower_code_pairs = shower_code_pairs[shower_code_pairs["IAU (No)"] != -1]

    # Select unique shower numbers and sort in the ascending order by the number
    shower_code_pairs = shower_code_pairs.drop_duplicates().sort_values(by=["IAU (No)"])

    # Get the shower codes and numbers
    shower_codes = shower_code_pairs["IAU (code)"].values
    shower_nums = shower_code_pairs["IAU (No)"].values

    print("Shower codes to analyze:")
    print(shower_codes)
    print(shower_nums)

    ## ###


    # # Manual list of showers to analyze
    # shower_codes = ["PER", "GEM", "QUA"]
    # shower_codes = ["QUA", "EVI", "LYR", "PER", "ORI", "KCG"]
    # shower_codes = ["ZCY", "EVI", "STA", "ORI", "PER"]
    # shower_codes = ["PER"]
    # shower_codes = ["XHE", "BCO", "UCE", "BAU", "ZCY", "EVI", "AUD", "NUE", "EDR"]
    # shower_codes = ["ETA"]



    # New shower table with (sol lon, SCE lon, ecl lat, vg, IAU No) sampled at sol_delta intervals for each shower
    new_table_values = []

    # List of showers with not enough data
    showers_not_enough_data = []

    # A table with the full drift characterization
    full_drift_table = []

    # Go through the showers and measure the radiants
    for shower_code in shower_codes:

        print()
        print()
        print("=====================================")

        # Print the shower code
        print("Shower code: {}".format(shower_code))
        print()

        # Select meteors belonging to the shower, as currently associated
        shower_data = data[data["IAU (code)"] == shower_code]


        # If the shower is a multi-part shower, split it into parts
        if shower_code in multi_part_showers:

            shower_parts = []

            for i, (sol_min, sol_max) in enumerate(multi_part_showers[shower_code]):

                # Select meteors belonging to the shower, as currently associated
                shower_data_part = shower_data[
                    (shower_data["Sol lon (deg)"] >= sol_min) 
                    & (shower_data["Sol lon (deg)"] < sol_max)
                    ]

                shower_parts.append([i + 1, shower_data_part])

        else:
            shower_parts = [[-1, shower_data]]



        # Go through the shower parts
        for part_num, shower_data in shower_parts:

            # If there are not enough meteors, skip the shower
            if len(shower_data) < min_meteors_per_shower:
                print("Not enough meteors, skipping the shower.")
                if shower_code not in additional_shower_codes:
                    showers_not_enough_data.append(shower_code)
                continue


            # Check if the shower is in the list of showers with known activity periods
            if shower_code in known_showers:

                # If the shower is in the list, take the known activity period
                sol_lon_min = known_showers[shower_code][0]
                sol_lon_ref = known_showers[shower_code][1]
                sol_lon_max = known_showers[shower_code][2]

            else:
                sol_lon_min = None
                sol_lon_ref = None
                sol_lon_max = None


            ### RADIANT DRIFT FIT ###

            # If the shower SCE longitude crosses the 0/360 deg boundary, shift the SCE longitude to the 
            # range [-180, 180]
            if shower_data["LAMsce (deg)"].max() - shower_data["LAMsce (deg)"].min() > 180:

                # Shift the SCE longitude to the range [-180, 180]
                shower_data["LAMsce (deg)"] = (shower_data["LAMsce (deg)"] + 180)%360 - 180

            # Fit the radiant drift
            sol_lon_ref, lg_drift, lg_drift_std, bg_drift, bg_drift_std, vg_drift, vg_drift_std = fitRadiantDrift(
                shower_data["Sol lon (deg)"], 
                shower_data["LAMsce (deg)"], 
                shower_data["BETgeo (deg)"], 
                shower_data["Vgeo (km/s)"],
                sol_lon_ref=sol_lon_ref
                )

            # Print drift
            print()
            print("DRIFT:")
            print("Lg - sol drift = {:.3f}{:+.3f}(sol - {:.3f})".format(lg_drift[0], lg_drift[1], sol_lon_ref))
            print("Bg       drift = {:.3f}{:+.3f}(sol - {:.3f})".format(bg_drift[0], bg_drift[1], sol_lon_ref))
            print("Vg       drift = {:.3f}{:+.3f}(sol - {:.3f})".format(vg_drift[0], vg_drift[1], sol_lon_ref))

            # Print drift std
            print()
            print("DRIFT STD:")
            print("Lg ref = {:.3f} +/- {:.3f}".format(lg_drift[0], lg_drift_std[0]))
            print("Lg slope = {:.3f} +/- {:.3f}".format(lg_drift[1], lg_drift_std[1]))
            print("Bg ref = {:.3f} +/- {:.3f}".format(bg_drift[0], bg_drift_std[0]))
            print("Bg slope = {:.3f} +/- {:.3f}".format(bg_drift[1], bg_drift_std[1]))
            print("Vg ref = {:.3f} +/- {:.3f}".format(vg_drift[0], vg_drift_std[0]))
            print("Vg slope = {:.3f} +/- {:.3f}".format(vg_drift[1], vg_drift_std[1]))

            ### ###


            ### Compute the rolling dispersion and the rolling count###

            # In steps of 0.1 deg solar longitude, compute the median offset from the drift-corrected radiant

            # Generate the steps
            if sol_lon_min is None or sol_lon_max is None:

                sol_lon_min = np.min(shower_data["Sol lon (deg)"])
                sol_lon_max = np.max(shower_data["Sol lon (deg)"])

                # Handle the case when the shower spans the 0 deg solar longitude
                if abs(sol_lon_max - sol_lon_min) > 180:

                    # Measure the begin and end of the shower only taking the part that is closer to the 0 deg
                    sol_lon_min = np.min(shower_data["Sol lon (deg)"][shower_data["Sol lon (deg)"] > 180])
                    sol_lon_max = np.max(shower_data["Sol lon (deg)"][shower_data["Sol lon (deg)"] < 180])

                # Round to sol_delta
                sol_lon_min = np.floor(sol_lon_min*(1/sol_delta))*sol_delta
                sol_lon_max = np.ceil(sol_lon_max*(1/sol_delta))*sol_delta


            # Generate the steps taking the shower span into account
            if abs(sol_lon_max - sol_lon_min) > 180:
                    
                    # If the shower spans the 0 deg solar longitude, generate the steps in two parts
                    sol_lon_steps = np.concatenate((
                        np.arange(sol_lon_min - 360, 0, sol_delta),
                        np.arange(0, sol_lon_max, sol_delta)
                        ))
                    
            else:
                sol_lon_steps = np.arange(sol_lon_min, sol_lon_max, sol_delta)

            print()
            print("Sol min = {:.1f}".format(sol_lon_min))
            print("Sol ref = {:.1f}".format(sol_lon_ref))
            print("Sol max = {:.1f}".format(sol_lon_max))



            # Compute the rolling parameters
            rolling_dispersion = []
            rolling_count = []
            for sol_lon_step in sol_lon_steps:

                # Select meteors within the step
                meteors_in_step = shower_data[np.abs((shower_data["Sol lon (deg)"] - sol_lon_step + 180)%360 - 180) < sol_delta/2]

                # Compute the location of the radiant at the step
                lg_step = lgDriftFunc((sol_lon_step - sol_lon_ref + 180)%360 - 180, *lg_drift)
                bg_step = bgDriftFunc((sol_lon_step - sol_lon_ref + 180)%360 - 180, *bg_drift)

                # Compute angular distances from the fitted radiant location
                ang_dist = np.degrees(angleBetweenSphericalCoords(
                    np.radians(bg_step), 
                    np.radians(lg_step), 
                    np.radians(meteors_in_step["BETgeo (deg)"]),
                    np.radians(meteors_in_step["LAMsce (deg)"])
                    ))

                # Compute the median angular distance
                rolling_dispersion.append(np.nanmedian(ang_dist))

                # Compute the number of meteors in the step
                rolling_count.append(len(meteors_in_step))
            
            # Convert to numpy arrays
            rolling_dispersion = np.array(rolling_dispersion)
            rolling_count = np.array(rolling_count)
            
            # Compute the median dispersion in +/- 1 deg solar longitude around the reference solar longitude, ignoring nan values
            dispersion_median = np.nanmedian(rolling_dispersion[np.abs((sol_lon_steps - sol_lon_ref + 180)%360 - 180) < 1])

            if np.isnan(dispersion_median):
                print("WARNING: dispersion_median is nan, setting to 1.0 deg")
                dispersion_median = 1.0

            sol_lon_steps_diff = (sol_lon_steps - sol_lon_ref + 180)%360 - 180

            # Compute the median dispersion from the beginning up to 1 deg sol before the reference solar longitude
            before_filter = sol_lon_steps_diff < -1
            dispersion_median_before_sol = np.nanmedian(sol_lon_steps_diff[before_filter]) + sol_lon_ref
            dispersion_median_before = np.nanmedian(rolling_dispersion[before_filter])

            # Compute the median dispersion from 1 deg sol after the reference solar longitude up to the end
            after_filter = sol_lon_steps_diff > 1
            dispersion_median_after_sol = np.nanmedian(sol_lon_steps_diff[after_filter]) + sol_lon_ref
            dispersion_median_after = np.nanmedian(rolling_dispersion[after_filter])

            # If any of the before or after dispersion medians are nan, set them to the reference dispersion
            if np.isnan(dispersion_median_before):
                dispersion_median_before = dispersion_median

            if np.isnan(dispersion_median_after):
                dispersion_median_after = dispersion_median

            # Limit the peak dispersion to the maximum dispersion
            uncorrected_peak_dispersion_max = max_peak_dispersion/dispersion_peak_factor
            uncorrected_peak_dispersion_min = min_peak_dispersion/dispersion_peak_factor
            if dispersion_median > uncorrected_peak_dispersion_max:
                dispersion_median = uncorrected_peak_dispersion_max

            elif dispersion_median < uncorrected_peak_dispersion_min:
                dispersion_median = uncorrected_peak_dispersion_min

            # Set a fixed dispersion for the edges if there are not enough meteors
            if np.sum(rolling_count[before_filter]) < min_meteors_edge_bin:
                dispersion_median_before = fixed_edge_dispersion

            else:

                # Set the minimum dispersion for the edges
                if dispersion_median_before < min_dispersion:
                    dispersion_median_before = min_dispersion


            if np.sum(rolling_count[after_filter]) < min_meteors_edge_bin:
                dispersion_median_after = fixed_edge_dispersion
            else:
                if dispersion_median_after < min_dispersion:
                    dispersion_median_after = min_dispersion

            # Handle the case when the shower spans the 0 deg solar longitude
            if (sol_lon_min > sol_lon_max) and (dispersion_median_before_sol > 180):
                dispersion_median_before_sol -= 360
                dispersion_median_after_sol -= 360
            

            # Store the parameters in a dispersion fit list
            dispersion_fit = [
                dispersion_median_before_sol, dispersion_median_before, 
                sol_lon_ref, dispersion_median, 
                dispersion_median_after_sol, dispersion_median_after
                ]

            print()
            print("Dispersion before the reference solar longitude: {:.3f}".format(dispersion_median_before))
            print("Dispersion +/- 1 deg around the reference solar longitude: {:.3f}".format(dispersion_median))
            print("Dispersion after the reference solar longitude: {:.3f}".format(dispersion_median_after))

            ### ###

            # If the shower is multi-part, don't compute the 95 percentile activity period but take the whole shower
            if part_num > -1:
                sol_lon_min_95 = np.min(shower_data["Sol lon (deg)"])
                sol_lon_max_95 = np.max(shower_data["Sol lon (deg)"])


            else:

                # Check if the shower is in the list with known activity periods
                if shower_code in known_showers:

                    # If the shower is in the list, take the known activity period
                    sol_lon_min_95 = sol_lon_min
                    sol_lon_max_95 = sol_lon_max


                # Compute the 95 percentile activity period
                else:

                    # Rotate the data to the reference solar longitude
                    sol_lon_rotated = (shower_data["Sol lon (deg)"] - sol_lon_ref + 180)%360 - 180

                    # Find the minimum and maximum solar longitude which encompass 95% of the meteors
                    sol_lon_min_95 = np.nanpercentile(sol_lon_rotated, 2.5)
                    sol_lon_max_95 = np.nanpercentile(sol_lon_rotated, 97.5)

                    # Derotate the solar longitudes
                    sol_lon_min_95 = (sol_lon_min_95 + sol_lon_ref)%360
                    sol_lon_max_95 = (sol_lon_max_95 + sol_lon_ref)%360

                    # Round the minimum and maximum solar longitudes to the nearest 0.1 deg (force the minimum lower and the maximum higher)
                    sol_lon_min_95 = np.floor(sol_lon_min_95*10)/10
                    sol_lon_max_95 = np.ceil(sol_lon_max_95*10)/10

                    # If the shower spans the 0 deg solar longitude, subtract 360 deg from the minimum solar longitude
                    if sol_lon_min_95 > sol_lon_max_95:
                        sol_lon_min_95 -= 360

            
            print()
            print("Sol min 95 = {:.1f}".format(sol_lon_min_95))
            print("Sol max 95 = {:.1f}".format(sol_lon_max_95))

            # Compute the velocity association threshold
            vel_radius = (shower_assoc_vel_perc/100.0)*shower_data["Vgeo (km/s)"].median()

            lg_aspect = 1/np.cos(np.radians(shower_data["BETgeo (deg)"]))

            # Compute the association radius using the dispersion function
            shower_assoc_dispersion = dispersionFunctionVec(shower_data["Sol lon (deg)"], *dispersion_fit, 
                                                            dispersion_peak_factor=dispersion_peak_factor)

            ### Select meteors inside the given association radius for the radians and the velocity and the new activity period ###

            # Compute the solar longitude difference to select the activity period
            sol_lon_min_95_diff = (sol_lon_min_95 - sol_lon_ref + 180)%360 - 180
            sol_lon_max_95_diff = (sol_lon_max_95 - sol_lon_ref + 180)%360 - 180
            data_sol_diff = (shower_data["Sol lon (deg)"] - sol_lon_ref + 180)%360 - 180

            # Compute the drift-corrected radiant location for each considered meteor
            data_lg_drift = lgDriftFunc(data_sol_diff, *lg_drift)
            data_bg_drift = bgDriftFunc(data_sol_diff, *bg_drift)
            data_vg_drift = bgDriftFunc(data_sol_diff, *vg_drift)

            # Compute the angular distances from the drift-corrected radiant location
            data_ang_dist = np.degrees(angleBetweenSphericalCoords(
                np.radians(data_bg_drift), 
                np.radians(data_lg_drift), 
                np.radians(shower_data["BETgeo (deg)"]),
                np.radians(shower_data["LAMsce (deg)"])
                ))

            # Select meteors within the association radius
            new_associated_data = shower_data[
                isAngleBetweenVec(
                    np.radians(sol_lon_min_95%360), 
                    np.radians(shower_data["Sol lon (deg)"]), 
                    np.radians(sol_lon_max_95%360))
                & (data_ang_dist < shower_assoc_dispersion)
                & (np.abs(shower_data["Vgeo (km/s)"] - data_vg_drift) < vel_radius)]
            
            ### ###
            
            
            print("Original association count = {}".format(len(shower_data)))
            print("New association count = {}".format(len(new_associated_data)))

            # If there are not enough meteors, skip the shower
            if len(new_associated_data) < min_meteors_per_shower:
                print("Not enough meteors, skipping the shower.")


                # Only add the shower to the list if it's not in additional showers
                if shower_code not in additional_shower_codes:
                    showers_not_enough_data.append(shower_code)

                continue

            # Re-fit the drift using the new association
            sol_lon_ref, lg_drift, lg_drift_std, bg_drift, bg_drift_std, vg_drift, vg_drift_std = fitRadiantDrift(
                new_associated_data["Sol lon (deg)"], 
                new_associated_data["LAMsce (deg)"], 
                new_associated_data["BETgeo (deg)"], 
                new_associated_data["Vgeo (km/s)"],
                sol_lon_ref=sol_lon_ref
                )
            

            # Generate values for the updated shower table
            new_shower_lookup_values = []
            for sol_lon_step in sol_lon_steps:
                if (sol_lon_step >= sol_lon_min_95) and (sol_lon_step <= sol_lon_max_95):

                    # Compute the location of the radiant at the step
                    lg_step = lgDriftFunc((sol_lon_step - sol_lon_ref + 180)%360 - 180, *lg_drift)
                    bg_step = bgDriftFunc((sol_lon_step - sol_lon_ref + 180)%360 - 180, *bg_drift)
                    vg_step = bgDriftFunc((sol_lon_step - sol_lon_ref + 180)%360 - 180, *vg_drift)

                    # Compute the radiant dispersion
                    dispersion_step = dispersionFunction(sol_lon_step, *dispersion_fit, 
                                                         dispersion_peak_factor=dispersion_peak_factor)

                    new_shower_lookup_values.append([
                        sol_lon_step%360, lg_step%360, bg_step, vg_step, dispersion_step, 
                        new_associated_data["IAU (No)"][0], shower_code])


            # Print the new shower lookup values
            # print("New shower lookup values:")
            new_shower_values_str = []
            for new_shower_lookup_value in new_shower_lookup_values:
                
                # Format sol lon to one decimal place, other coordinates to three decimal places, and keep the IAU code as an integer
                str_formatted = "{:5.1f} {:8.3f} {:8.3f} {:8.3f} {:5.2f} {:4d} {:3s}".format(*new_shower_lookup_value)

                # # Print the formatted string
                # print(str_formatted)

                # Add the formatted string to the list of new shower lookup values
                new_shower_values_str.append(str_formatted)
                new_table_values.append(str_formatted)


            ### PLOTTING ###


            # Generate indepentant varable data for the plot, taking the 0/360 boundary into account
            sol_lon_plot = np.linspace(sol_lon_min_95, sol_lon_max_95, 100)
            sol_lon_plot_normed = (sol_lon_plot - sol_lon_ref + 180)%360 - 180

            # Plot SCE longitude vs. solar longitude (top), and SCE latitude vs. solar longitude (bottom)
            fig, ((ax_lg, ax_bg), (ax_vel, ax_disp), (ax_count, ax_dens)) = plt.subplots(3, 2, figsize=(10, 8))

            # If the solar longitude crosses the 0/360 boundary, modify the solar longitude to make the plot nice
            if (sol_lon_min_95 < 0) and (sol_lon_ref > 180):
                sol_lon_ref -= 360

            if (abs(np.min(shower_data["Sol lon (deg)"]) - np.max(shower_data["Sol lon (deg)"])) > 180):
                sol_data = shower_data["Sol lon (deg)"].copy()
                sol_data[sol_data > 180] -= 360

            else:
                sol_data = shower_data["Sol lon (deg)"].copy()

            if (abs(np.min(sol_lon_plot) - np.max(sol_lon_plot)) > 180):
                sol_lon_plot[sol_lon_plot > 180] -= 360


            # Plot the data
            scatter = ax_lg.scatter(sol_data, shower_data["LAMsce (deg)"], s=1)
            ax_bg.scatter(sol_data, shower_data["BETgeo (deg)"], s=1)
            ax_vel.scatter(sol_data, shower_data["Vgeo (km/s)"], s=1)

            # Plot the fits
            lg_drift_plot = lgDriftFunc(sol_lon_plot_normed, *lg_drift)
            if abs(np.min(lg_drift_plot) - np.max(lg_drift_plot)) > 180:
                lg_drift_plot[lg_drift_plot > 180] -= 360
            ax_lg.plot(sol_lon_plot, lg_drift_plot, color="black", linewidth=0.5)
            ax_bg.plot(sol_lon_plot, bgDriftFunc(sol_lon_plot_normed, *bg_drift), color="black", linewidth=0.5)
            ax_vel.plot(sol_lon_plot, bgDriftFunc(sol_lon_plot_normed, *vg_drift), color="black", linewidth=0.5)


            # Compute the association radius using the dispersion function
            shower_assoc_dispersion_plot = dispersionFunctionVec(sol_lon_plot, *dispersion_fit, dispersion_peak_factor=dispersion_peak_factor)

            # Plot a 3 degree line around the drift to indicate the association radius
            lg_aspect = 1.0/np.cos(np.radians(bgDriftFunc(sol_lon_plot_normed, *bg_drift)))
            ax_lg.plot(sol_lon_plot, lg_drift_plot + shower_assoc_dispersion_plot*lg_aspect, color="red", linestyle="--")
            ax_lg.plot(sol_lon_plot, lg_drift_plot - shower_assoc_dispersion_plot*lg_aspect, color="red", linestyle="--")
            ax_bg.plot(sol_lon_plot, bgDriftFunc(sol_lon_plot_normed, *bg_drift) + shower_assoc_dispersion_plot, color="red", linestyle="--")
            ax_bg.plot(sol_lon_plot, bgDriftFunc(sol_lon_plot_normed, *bg_drift) - shower_assoc_dispersion_plot, color="red", linestyle="--")

            # Plot a +/- 10% line around the velocity drift to indicate the velocity association radius
            ax_vel.plot(sol_lon_plot, bgDriftFunc(sol_lon_plot_normed, *vg_drift) + vel_radius, color="red", linestyle="--")
            ax_vel.plot(sol_lon_plot, bgDriftFunc(sol_lon_plot_normed, *vg_drift) - vel_radius, color="red", linestyle="--")

            # Plot points between the 95% solar longitude range at each step
            sol_sample_points = []
            lg_sample_points = []
            bg_sample_points = []
            vg_sample_points = []
            for sol_lon_step in sol_lon_steps:
                if sol_lon_step >= sol_lon_min_95 and sol_lon_step <= sol_lon_max_95:

                    # Compute the location of the radiant at the step
                    lg_step = lgDriftFunc((sol_lon_step - sol_lon_ref + 180)%360 - 180, *lg_drift)
                    bg_step = bgDriftFunc((sol_lon_step - sol_lon_ref + 180)%360 - 180, *bg_drift)
                    vg_step = bgDriftFunc((sol_lon_step - sol_lon_ref + 180)%360 - 180, *vg_drift)

                    # Add the point to the sample points
                    sol_sample_points.append(sol_lon_step)
                    lg_sample_points.append(lg_step)
                    bg_sample_points.append(bg_step)
                    vg_sample_points.append(vg_step)

            if len(sol_sample_points) == 0:
                if shower_code not in additional_shower_codes:
                    showers_not_enough_data.append(shower_code)
                continue

            # Check if lg crosses the 0/360 boundary
            if abs(np.min(lg_sample_points) - np.max(lg_sample_points)) > 180:
                lg_sample_points = np.array(lg_sample_points)
                lg_sample_points[lg_sample_points > 180] -= 360

            # Plot the sample points
            ax_lg.scatter(sol_sample_points, lg_sample_points, color="red", s=2)
            ax_bg.scatter(sol_sample_points, bg_sample_points, color="red", s=2)
            ax_vel.scatter(sol_sample_points, vg_sample_points, color="red", s=2)



            # Plot the rolling dispersion
            ax_disp.plot(sol_lon_steps, rolling_dispersion, color="red")

            # Plot the median dispersions before, at the reference, and after the reference
            ax_disp.scatter(dispersion_median_before_sol, dispersion_median_before, color="black", s=15, zorder=10, marker="x", 
                            label='Sol = {:.2f}, Disp = {:.2f}'.format(dispersion_median_before_sol, dispersion_median_before))
            ax_disp.scatter(sol_lon_ref, dispersion_median, color="black", s=15, zorder=10, marker="o", 
                            label='Sol = {:.2f}, Disp = {:.2f}'.format(sol_lon_ref, dispersion_median))
            ax_disp.scatter(sol_lon_ref, dispersion_median*dispersion_peak_factor, color="black", s=15, zorder=10, marker="^", 
                            label='Sol = {:.2f}, Disp* = {:.2f}'.format(sol_lon_ref, dispersion_median*dispersion_peak_factor))
            ax_disp.scatter(dispersion_median_after_sol, dispersion_median_after, color="black", s=15, zorder=10, marker="s", 
                            label='Sol = {:.2f}, Disp = {:.2f}'.format(dispersion_median_after_sol, dispersion_median_after))

            # Plot the reference solar longitude
            ax_disp.axvline(sol_lon_ref, color="black", linestyle="dotted")
            ax_disp.legend()


            # Plot the number of meteors in each step
            ax_count.plot(sol_lon_steps, rolling_count, color="red")
            ax_count.axvline(sol_lon_ref, color="black", linestyle="dotted", label='Ref = {:.1f}'.format(sol_lon_ref))
            ax_count.legend()

            # Plot vertical lines at the 95% solar longitude range
            ax_count.axvline(sol_lon_min_95, color="black", linestyle="--", label='Min = {:.1f}'.format(sol_lon_min_95))
            ax_count.axvline(sol_lon_max_95, color="black", linestyle="--", label='Max = {:.1f}'.format(sol_lon_max_95))
            ax_count.legend()


            # Compute the extent for the hexbins
            extent = [
                shower_data["LAMsce (deg)"].min(), 
                shower_data["LAMsce (deg)"].max(), 
                shower_data["BETgeo (deg)"].min(), 
                shower_data["BETgeo (deg)"].max()
                ]

            # Plot a density plot of LAMsce vs. BETgeo
            gridsize = 25
            hbin = ax_dens.hexbin(shower_data["LAMsce (deg)"], shower_data["BETgeo (deg)"], 
                                  gridsize=gridsize, cmap='inferno', mincnt=0, extent=extent, bins='log')
            
            # Plot the sampled points
            ax_dens.scatter(lg_sample_points, bg_sample_points, color="red", s=2)

            # Create a hexbin using new_associated_data but using the same binning scheme as in hbin
            hbin_associated = ax_dens.hexbin(
                new_associated_data["LAMsce (deg)"], 
                new_associated_data["BETgeo (deg)"], 
                gridsize=gridsize, cmap='inferno', mincnt=0, extent=extent
                )

            # Get the associated hexbin bin coordiantes and counts
            hbin_associated_x = hbin_associated.get_offsets()[:,0]
            hbin_associated_y = hbin_associated.get_offsets()[:,1]
            hbin_associated_counts = hbin_associated.get_array()

            # Interpolate the hbin_associated data to get a smooth contour
            X, Y = np.meshgrid(hbin_associated_x, hbin_associated_y)
            Z = scipy.interpolate.griddata(
                (hbin_associated_x, hbin_associated_y), 
                hbin_associated_counts, 
                (X, Y), 
                method='linear')

            # Plot the contour at level 2 using the interpolated model
            ax_dens.contour(X, Y, Z, levels=[1], linewidths=0.5, colors='red', extent=extent)

            # Invert the x-axis
            ax_dens.invert_xaxis()

            # Remove hbin_associated as it was only used for binning
            hbin_associated.remove()

            # Set the axis labels
            ax_lg.set_ylabel("LAMsce (deg)")
            ax_lg.set_xlabel("Sol lon (deg)")

            ax_bg.set_ylabel("BETgeo (deg)")
            ax_bg.set_xlabel("Sol lon (deg)")

            ax_vel.set_ylabel("Vgeo (km/s)")
            ax_vel.set_xlabel("Sol lon (deg)")

            ax_disp.set_ylabel("Rolling dispersion (deg)")
            ax_disp.set_xlabel("Sol lon (deg)")

            ax_count.set_ylabel("Number of meteors")
            ax_count.set_xlabel("Sol lon (deg)")

            ax_dens.set_ylabel("BETgeo (deg)")
            ax_dens.set_xlabel("LAMsce (deg)")

            # Set title with the shower code
            plt.suptitle("Shower code: {}".format(shower_code))

            plt.tight_layout()

            plot_dir_path = os.path.join(dir_path_traj_summary, plot_dir)
            if not os.path.exists(plot_dir_path):
                os.makedirs(plot_dir_path)

            if part_num == -1:
                part_str = ""
            else:
                part_str = "_part{:1d}".format(part_num)

            plot_name = "{:04d}_{:s}{:s}".format(new_associated_data["IAU (No)"][0], shower_code, part_str)
            plot_path = os.path.join(plot_dir_path, plot_name + ".png")

            plt.savefig(plot_path, dpi=300)

            if show_plots:
                plt.show()
            
            plt.close(fig)



            # Save the tabular data in a txt file
            txt_path = os.path.join(plot_dir_path, plot_name + ".txt")
            with open(txt_path, 'w') as f:
                for str_formatted in new_shower_values_str:
                    f.write(str_formatted + "\n")


            # Save the complete information about the drift and dispersion in a table
            full_table = [
                new_associated_data["IAU (No)"][0], 
                shower_code + part_str.replace("part", ""), 
                sol_lon_min_95%360, 
                sol_lon_ref%360, 
                sol_lon_max_95%360, 
                lg_drift[0],
                lg_drift[1],
                bg_drift[0],
                bg_drift[1],
                vg_drift[0],
                vg_drift[1]
            ]
            full_table += dispersion_fit


            # Store the full shower data in a list
            full_drift_table.append(full_table)


    ### Save the data for all showers in a single file, but first sort by the solar longitude ###

    new_table_path = os.path.join(dir_path_traj_summary, new_table_file_name)

    # Add additional shower entries to the table
    new_table_values += additional_shower_entries

    # Sort the data by solar longitude, where the solar longitude is the first element in the string
    new_table_values = sorted(new_table_values, key=lambda x: float(x.split()[0]))

    # Save the data in a txt file
    with open(new_table_path, 'w') as f:

        # Write the header
        head = """# Shower lookup table measured using GMN data up to May 18, 2023
# Using the wmpl.Utils.ShowerTableMeasurement script in WMPL
# 
# The table is sampled at 0.2 deg solar longitude. The SCE (Sun-centered 
# ecliptic) longitude is the ecliptic longitude minus the solar 
# longitude. The "Disp" column is the measured dispersion which changes 
# over time. A meteor is considered a shower member if it has been observed 
# within 1 deg of solar longitude of a table entry, within 10% of Vg, and
# within the listed dispersion.
#
"""
        head += "# IAU showers not present in the table:\n# "
        # Limit the line length to 80 characters, break after commas
        showers_not_enough_data_str = ""
        current_line = ""
        for shower_code in showers_not_enough_data:

            if len(current_line + shower_code + ", ") > 77:
                showers_not_enough_data_str += current_line + "\n# "
                current_line = ""

            current_line += shower_code + ", "

        showers_not_enough_data_str += current_line

        head += showers_not_enough_data_str

        head += """
# 
# Sol  SCE lon  SCE lat       Vg  Disp  IAU IAU
# deg      deg      deg     km/s  deg    No  cd
# 
"""
        f.write(head)

        for str_formatted in new_table_values:
            f.write(str_formatted + "\n")


    # Save the full drift table in a txt file
    full_drift_table_path = os.path.join(dir_path_traj_summary, full_param_fit_table_file_name)
    with open(full_drift_table_path, 'w') as f:

        # Write the header (comma separated)
        f.write("IAU (No),Shower code,Sol lon min,Sol lon ref,Sol lon max,LG drift ref,LG drift slope,BG drift ref,BG drift slope,VG drift ref,VG drift slope,Disp min sol,Disp min,Disp ref sol,Disp ref,Disp max sol,Disp max\n")

        for entry in full_drift_table:
            
            (
                iau_no, shower_code, 
                sol_lon_min, sol_lon_ref, sol_lon_max, 
                lg_drift_ref, lg_drift_slope, 
                bg_drift_ref, bg_drift_slope, 
                vg_drift_ref, vg_drift_slope, 
                disp_min_sol, disp_min, disp_ref_sol, disp_ref, disp_max_sol, disp_max
            ) = entry

            f.write("{:4d}, {:5s}, {:5.1f}, {:5.1f}, {:5.1f}, {:7.3f}, {:6.3f}, {:7.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:5.1f}, {:6.3f}, {:5.1f}, {:6.3f}, {:5.1f}, {:6.3f}\n".format(
                iau_no, shower_code,
                sol_lon_min%360, sol_lon_ref%360, sol_lon_max%360,
                lg_drift_ref%360, lg_drift_slope,
                bg_drift_ref, bg_drift_slope,
                vg_drift_ref, vg_drift_slope,
                disp_min_sol, disp_min, disp_ref_sol, disp_ref, disp_max_sol, disp_max
            ))


    print()
    print("Done!")

    print()
    print("Showers with not enough data:")
    print(showers_not_enough_data)
