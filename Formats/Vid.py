""" Loading and handling *.vid files. """


import os
import numpy as np


class VidStruct(object):
    """ Structure for storing vid file info. """

    def __init__(self):

        self.frames = 0
        self.magic = 0
        self.seqlen = 0
        self.headlen = 0


def readVid(dir_path, file_name, vid_h=480, vid_w=640):
    """ Read in a *.vid file. 
    
    Arguments:
        dir_path: [str] path to the directory where the *.vid file is located
        file_name: [str] name of the *.vid file

    Kwargs:
        vid_h: [int] height of the frames in the vid file
        vif_w: [int] width of the frames in the vid file

    Return:
        [VidStruct object]
    """

    # Open the file for binary reading
    fid = open(os.path.join(dir_path, file_name), 'rb')

    # Init the vid struct
    vid = VidStruct()

    vid.magic = int(np.fromfile(fid, dtype=np.uint32, count = 1))

    # Size of one frame in bytes
    vid.seqlen = int(np.fromfile(fid, dtype=np.uint32, count = 1))

    # Header length in bytes
    vid.headlen = int(np.fromfile(fid, dtype=np.uint32, count = 1))

    # Reset the file pointer to the beginning
    fid.seek(0)
    
    vid.frames = []

    # Read in the frames
    while True:

        # Read one frame
        fr = np.fromfile(fid, dtype=np.uint16, count = vid.seqlen/2)

        # Check if the read array is of proper size (if not, it is EOF and break reading)
        if not (fr.shape[0] == vid.seqlen/2):
            break


        # Set the values of the first row to 0
        fr[:vid_h] = 0

        # Reshape the frame and add it to the frame list
        vid.frames.append(fr.reshape(vid_h, vid_w))

    fid.close()


    return vid



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Vid file path
    dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MirfitPrepare/20160927_030027_mir"
    file_name = "ev_20160927_030028A_01T.vid"

    # Read in the *.vid file
    vid = readVid(dir_path, file_name)

    frame_num = 125

    # Show one frame of the vid file
    plt.imshow(vid.frames[frame_num], cmap='gray', vmin=0, vmax=255)
    plt.show()


