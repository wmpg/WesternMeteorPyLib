""" Takes a .vid file and the reduction, and creates video frames with marked position of the fragments. """

from __future__ import print_function, division, absolute_import

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

from Formats.Vid import readVid
from Formats.Met import loadMet
from Utils.OSTools import mkdirP
from Utils.Pickling import loadPickle
from Utils.TrajConversions import unixTime2Date, unixTime2JD


def markFragments(out_dir, vid, met, site_id, traj=None):
    """ Mark fragments on .vid file frames and save them as images. If the trajectory structure is given,
        the approximate height at every frame will be plotted on the image as well.
    
    Arguments:
        out_dir: [str] Path to the directory where the images will be saved.
        vid: [VidStruct object] vid object containing loaded video frames.
        met: [MetStruct object] met object containing picks.
        site_id: [str] ID of the site used for loading proper picks from the met object.

    Keyword arguments:
        traj: [Trajectory object] Optional trajectory object from which the height of the meteor at evey frame
            will be estimated and plotted on the image. None by default (no height will be plotted on the
            image).

    Return:
        None
    """


    # Make the output directory
    mkdirP(out_dir)

    # Extract site picks
    picks = np.array(met.picks[site_id])

    # Find unique fragments
    fragments = np.unique(picks[:, 1])


    # Generate a unique color for every fragment
    colors = plt.cm.rainbow(np.linspace(0, 1, len(fragments)))

    # Create a dictionary for every fragment-color pair
    colors_frags = {frag: color for frag, color in zip(fragments, colors)}



    ### Height fit ###

    height_interp = None

    # If the trajectory was given, do interpolation on time vs. height
    if traj is not None:


        jd_data = []
        height_data = []

        # Take all observed points on the trajectory
        for obs in traj.observations:

            for jd, height in zip(obs.JD_data, obs.model_ht):
                jd_data.append(jd)
                height_data.append(height)


        jd_data = np.array(jd_data)
        height_data = np.array(height_data)

        # Sort the points by Julian date
        jd_ht_data = np.c_[jd_data, height_data]
        jd_ht_data = jd_ht_data[np.argsort(jd_ht_data[:, 0])]
        jd_data, height_data = jd_ht_data.T

        # Initerpolate Julian date vs. heights
        height_interp = scipy.interpolate.PchipInterpolator(jd_data, height_data, extrapolate=True)

        # # Plot JD vs. height
        # plt.scatter(jd_data, height_data)

        # jd_plot = np.linspace(min(jd_data), max(jd_data), 1000)
        # plt.plot(jd_plot, height_interp(jd_plot))

        # plt.show()




    ##################


    frag_count = 1
    frag_dict = {}

    # Go through all frames
    for i, fr_vid in enumerate(vid.frames):

        if (i < 218) or (i > 512):
            continue

        # Plot the frame
        plt.imshow(fr_vid.img_data, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')


        # Convert the frame time from UNIX timestamp to human readable format
        timestamp = unixTime2Date(fr_vid.ts, fr_vid.tu, dt_obj=True).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] \
            + ' UTC'

        # Calculate the Julian date of every frame
        frame_jd = unixTime2JD(fr_vid.ts, fr_vid.tu)


        # Plot the timestamp
        plt.text(10, fr_vid.ht - 10, timestamp, ha='left', va='bottom', color='0.7', size=10)


        # If the trajectory was given, plot the meteor height at every frame
        if height_interp is not None:

            # Get the height at the given frame
            frame_height = height_interp(frame_jd)

            height_str = 'Height = {:7.3f} km'.format(frame_height/1000)

            # Plot the height
            plt.text(fr_vid.wid - 10, fr_vid.ht - 10, height_str, ha='right', va='bottom', color='0.7', size=10)


        # Hide axes
        plt.gca().set_axis_off()




        # Extract picks on this frame
        fr_picks = picks[picks[:, 0] == i]

        # Check if there are any picks on the this frame and plot them by fragment number
        if len(fr_picks):

            # Go through every pick
            for pick in fr_picks:

                # Choose the appropriate colour by fragment
                frag_color = colors_frags[pick[1]]

                # Extract pick coordinates
                cx = pick[2]
                cy = pick[3]

                # Extract fragment number
                frag = pick[1]

                # Assign this fragment a sequential number if it was not yet plotted
                if not frag in frag_dict:
                    frag_dict[frag] = frag_count
                    frag_count += 1

                
                # Look up the fragment sequential number
                frag_no = frag_dict[frag]

                # Move the markers a bit to the left/right, intermittently for every fragment
                if frag%2 == 0:
                    cx -= 10
                    txt_cx = cx - 5
                    marker = '>'
                    txt_align = 'right'
                else:
                    cx += 10
                    txt_cx = cx + 5
                    marker = '<'
                    txt_align = 'left'


                # Plot the pick
                plt.scatter(cx, cy - 1, c=frag_color, marker=marker, s=5)

                # Plot the fragment number
                plt.text(txt_cx, cy, str(int(frag_no)), horizontalalignment=txt_align, verticalalignment='center', color=frag_color, size=8)
        

        # Set limits
        plt.xlim([0, vid.wid])
        plt.ylim([vid.ht, 0])

        # Save the plot
        extent = plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.savefig(os.path.join(out_dir, str(i) + '.png'), transparent=True, bbox_inches=extent, pad_inches=0, dpi=300)

        plt.clf()
        #plt.show()


    # IMPORTANT!!! COPY THE OUTPUT OF THIS TO ProjectNarrowPicksToWideTraj!!!!
    # Print the fragment dictionary, where the original fragment IDs are mapped into sequential numbers
    print('FRAG DICT:', frag_dict)




if __name__ == "__main__":


    # Main directory
    dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MetalPrepare/20170721_070420_met"

    # Output directory
    out_dir = os.path.join(dir_path, 'marked_frames')

    # .met file containing narrow-field picks
    met_file = 'state_fragment_picks.met'

    # .vid file
    vid_file = os.path.join('cut_20170721_070418_01T', 'ev_20170721_070420A_01T.vid')

    # Trajectory file
    traj_file = os.path.join('Monte Carlo', '20170721_070419_mc_trajectory.pickle')


    ##########################################################################################################

    # Load the MET file
    met = loadMet(dir_path, met_file, mirfit=True)

    # Load the vid file
    vid = readVid(dir_path, vid_file)

    # Load the trajectory file
    traj = loadPickle(dir_path, traj_file)

    # ID of the site used for loading proper picks from the met object
    site_id = '1'

    # Generate images with marked fragments on them
    markFragments(out_dir, vid, met, site_id, traj=traj)