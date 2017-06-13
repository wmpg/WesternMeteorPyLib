""" Loading and handling *.vid files. """


import os
import numpy as np


class VidStruct(object):
    """ Structure for storing vid file info. """

    def __init__(self):

        self.frames = 0

        self.magic = 0

        # Bytes for a single image
        self.seqlen = 0

        # Header length in bytes
        self.headlen = 0

        self.flags = 0
        self.seq = 0

        # UNIX time
        self.ts = 0
        self.tu = 0

        # Station number
        self.station_id = 0

        # Image dimensions in pixels
        self.wid = 0
        self.ht = 0

        # Image depth in bits
        self.depth = 0

        # Mirror pointing for centre of frame
        self.hx = 0
        self.hy = 0

        # Stream number
        self.str_num = 0
        self.reserved0 = 0

        # Exposure time in milliseconds
        self.exposure = 0

        self.reserved2 = 0

        self.text = 0


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

    vid.flags = int(np.fromfile(fid, dtype=np.uint32, count = 1))
    vid.seq = int(np.fromfile(fid, dtype=np.uint32, count = 1))

    # UNIX time
    vid.ts = int(np.fromfile(fid, dtype=np.int32, count = 1))
    vid.tu = int(np.fromfile(fid, dtype=np.int32, count = 1))

    # Station number
    vid.station_id = int(np.fromfile(fid, dtype=np.int16, count = 1))

    # Image dimensions
    vid.wid = int(np.fromfile(fid, dtype=np.int16, count = 1))
    vid.ht = int(np.fromfile(fid, dtype=np.int16, count = 1))

    # Image depth
    vid.depth = int(np.fromfile(fid, dtype=np.int16, count = 1))

    vid.hx = int(np.fromfile(fid, dtype=np.uint16, count = 1))
    vid.hy = int(np.fromfile(fid, dtype=np.uint16, count = 1))

    vid.str_num = int(np.fromfile(fid, dtype=np.uint16, count = 1))
    vid.reserved0 = int(np.fromfile(fid, dtype=np.uint16, count = 1))
    vid.exposure = int(np.fromfile(fid, dtype=np.uint32, count = 1))
    vid.reserved2 = int(np.fromfile(fid, dtype=np.uint32, count = 1))

    vid.text = np.fromfile(fid, dtype=np.uint8, count = 64).tostring().decode("ascii").replace('\0', '')

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
        fr[:vid.ht] = 0

        # Reshape the frame and add it to the frame list
        vid.frames.append(fr.reshape(vid.ht, vid.wid))

    fid.close()


    return vid



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Vid file path
    dir_path = "../../MirfitPrepare/20160929_050928_mir"
    file_name = "ev_20160929_050928A_01T.vid"

    # Read in the *.vid file
    vid = readVid(dir_path, file_name)

    frame_num = 125

    # Show one frame of the vid file
    plt.imshow(vid.frames[frame_num], cmap='gray', vmin=0, vmax=255)
    plt.show()


