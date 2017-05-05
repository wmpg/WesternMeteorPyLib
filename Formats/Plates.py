""" Loading and handling scale and exact plates. """


import os
import numpy as np


class AffPlate(object):
    """ AFF type plate structure. """

    def __init__(self):

        self.magic = 0
        self.info_len = 0

        self.r0 = 0
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

        self.sx = 0
        self.sy = 0
        self.phi = 0
        self.tx = 0
        self.ty = 0
        self.wid = 0
        self.ht = 0
        self.site = 0
        self.text = ''
        self.text_size = 256
        self.flags = 0


    def initM(self):
        """ Calculate the conversion matrix. """

        # Init the M matrix
        M = np.zeros((3,3))

        M[0,0] =  self.sx*np.cos(self.phi)
        M[0,1] = -self.sy*np.sin(self.phi)
        M[0,2] =  self.tx*M[0,0] + self.ty*M[0,1]

        M[1,0] =  self.sx*np.sin(self.phi)
        M[1,1] =  self.sy*np.cos(self.phi)
        M[1,2] =  self.tx*M[1,0] + self.ty*M[1,1]

        M[2,0] = 0
        M[2,1] = 0
        M[2,2] = 1

        self.M = M


    def __repr__(self):

        return "sx " + str(self.sx) + " sy " + str(self.sy) + " phi " + str(self.phi) + " tx " + \
            str(self.tx) + " ty " + str(self.ty) + " wid " + str(self.wid) + " ht " + str(self.ht) + \
            " site " + str(self.site) + " text " + str(self.text) + " text_size " + str(self.text_size) + \
            " flags " + str(self.flags)



class AstPlate(object):
    """ AST type plate structure. """

    def __init__(self):

        self.magic = 0
        self.info_len = 0
        self.star_len = 0
        self.stars = 0

        self.r0 = 0
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

        self.text = ''
        self.starcat = ''
        self.sitename = ''
        self.site = 0

        self.lat = 0.0
        self.lon = 0.0
        self.elev = 0.0
        self.ts = 0
        self.tu = 0

        self.th0 = 0.0
        self.phi0 = 0.0

        self.wid = 0
        self.ht = 0

        self.a = 0
        self.da = 0
        self.b = 0
        self.db = 0
        self.c = 0
        self.dc = 0
        self.d = 0
        self.dd = 0

        self.flags = 0


    def initM(self):
        """ Calculates the conversion matrix. """

        # Init the conversion matrix
        M = np.zeros((3,3))

        M[0,0] = -np.sin(self.phi0)
        M[1,0] =  np.cos(self.phi0)
        M[2,0] =  0.0

        M[0,1] = -np.cos(self.th0)*np.cos(self.phi0)
        M[1,1] = -np.cos(self.th0)*np.sin(self.phi0)
        M[2,1] =  np.sin(self.th0)

        M[0,2] =  np.sin(self.th0)*np.cos(self.phi0)
        M[1,2] =  np.sin(self.th0)*np.sin(self.phi0)
        M[2,2] =  np.cos(self.th0)

        self.M = M


    def __repr__(self):


        return "text " + str(self.text) + " starcat " + str(self.starcat) + " sitename " + \
            str(self.sitename) + " site " + str(self.site) + " lat " + str(self.lat) + " lon " + \
            str(self.lon) + " elev " + str(self.elev) + " ts " + str(self.ts) + " tu " + str(self.tu) + \
            " th0 " + str(self.th0) + " phi0 " + str(self.phi0) + " wid " + str(self.wid) + " ht " + \
            str(self.ht) + " a " + str(self.a) + " da " + str(self.da) + " b " + str(self.b) + " db " + \
            str(self.db) + " c " + str(self.c) + " dc " + str(self.dc) + " d " + str(self.d) + " dd " + \
            str(self.dd) + " flags " + str(self.flags)




def loadExact(dir_path, file_name):
    """ Loads an AST exact plate. 
    
    Arguments:
        dir_path: [str] path to the directory where the plate file is located
        file_name: [str] name of the plate file

    Return:
        [AstPlate object]
    """


    # Open the file for binary reading
    fid = open(os.path.join(dir_path, file_name), 'rb')

    # Init the plate struct
    exact = AstPlate()

    # Load header
    exact.magic = np.fromfile(fid, dtype=np.uint32, count = 1)
    exact.info_len = np.fromfile(fid, dtype=np.uint32, count = 1)

    # Star structure size in bytes
    exact.star_len = np.fromfile(fid, dtype=np.uint32, count = 1)

    # Number of stars
    exact.stars = np.fromfile(fid, dtype=np.int32, count = 1)

    # Reserved
    exact.r0 = np.fromfile(fid, dtype=np.int32, count = 1)
    exact.r1 = np.fromfile(fid, dtype=np.int32, count = 1)
    exact.r2 = np.fromfile(fid, dtype=np.int32, count = 1)
    exact.r3 = np.fromfile(fid, dtype=np.int32, count = 1)

    # Text description
    exact.text = np.fromfile(fid, dtype='|S'+str(256), count = 1)[0]

    # Name of catalogue
    exact.starcat = np.fromfile(fid, dtype='|S'+str(32), count = 1)[0]

    # Name of observing site
    exact.sitename = np.fromfile(fid, dtype='|S'+str(32), count = 1)[0]

    # Site geo coordinates
    exact.lat = np.fromfile(fid, dtype=np.float64, count = 1)[0]
    exact.lon = np.fromfile(fid, dtype=np.float64, count = 1)[0]
    exact.elev = np.fromfile(fid, dtype=np.float64, count = 1)[0]

    # UNIX time for fit
    exact.ts = np.fromfile(fid, dtype=np.int32, count = 1)
    exact.tu = np.fromfile(fid, dtype=np.int32, count = 1)

    # Centre of plate
    exact.th0 = np.fromfile(fid, dtype=np.float64, count = 1)[0]
    exact.phi0 = np.fromfile(fid, dtype=np.float64, count = 1)[0]

    # Original image size
    exact.wid = np.fromfile(fid, dtype=np.int32, count = 1)[0]
    exact.ht = np.fromfile(fid, dtype=np.int32, count = 1)[0]

    ### Fit parameters
    # x/y --> th
    exact.a = np.fromfile(fid, dtype=np.float64, count = 10)
    exact.da = np.fromfile(fid, dtype=np.float64, count = 10)

    # x/y --> phi
    exact.b = np.fromfile(fid, dtype=np.float64, count = 10)
    exact.db = np.fromfile(fid, dtype=np.float64, count = 10)

    # th/phi --> x
    exact.c = np.fromfile(fid, dtype=np.float64, count = 10)
    exact.dc = np.fromfile(fid, dtype=np.float64, count = 10)

    # th/phi --> y
    exact.d = np.fromfile(fid, dtype=np.float64, count = 10)
    exact.dd = np.fromfile(fid, dtype=np.float64, count = 10)
    
    # Fit flags
    exact.flags = np.fromfile(fid, dtype=np.uint32, count = 1)

    # Calculate the conversion matrix
    exact.initM()


    return exact



def plateExactMap(exact, x, y, reverse_map=False):
    """ Map the mirror encoder coordinates (Hx, Hy) to sky coordinates (theta, phi) given an 
        appropriate exact plate. If a reverse mapping is desired, set reverse_map=True.
        
    Arguments:
        scale: [AstPlate object] AST plate structure
        x: [float] input parameter 1 (Hx by default, theta if reverse_map=True)
        y: [float] input parameter 2 (Hy by default, phi if reverse_map=True)

    Kwargs:
        reverse_map: [bool] default False, if True, revese mapping is performed

    Return:
        [tuple of floats]: output parameters (theta, phi) by default, (Hx, Hy) if reverse_map=True

    """

    M = exact.M

    # Forward mapping
    if not reverse_map:

        # Normalize coordinates to 0
        x -= exact.wid/2.0
        y -= exact.ht/2.0

        a = exact.a
        b = exact.b

        # Project onto (p, q) plane
        p = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*y + a[5]*y**2 + a[6]*y**3 + a[7]*x*y + a[8]*x**2*y + \
            a[9]*x*y**2

        q = b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3 + b[4]*y + b[5]*y**2 + b[6]*y**3 + b[7]*x*y + b[8]*x**2*y + \
            b[9]*x*y**2

        # Calculate beta and gamma
        bet = np.arcsin(np.hypot(p, q))
        gam = np.arctan2(q, p)

        if bet > np.pi:
            return False

        # Init vector v
        v = np.zeros(3)

        v[0] = np.sin(bet)*np.cos(gam)
        v[1] = np.sin(bet)*np.sin(gam)
        v[2] = np.cos(bet)

        # Calculate vector u
        u = np.zeros(3)        

        u[0] = M[0,0]*v[0] + M[0,1]*v[1] + M[0,2]*v[2]
        u[1] = M[1,0]*v[0] + M[1,1]*v[1] + M[1,2]*v[2]
        u[2] = M[2,0]*v[0] + M[2,1]*v[1] + M[2,2]*v[2]

        # Convert to theta, phi
        th  = np.arctan2(np.hypot(u[0], u[1]), u[2])
        phi = np.arctan2(u[1], u[0])

        return th, phi


    # Reverse mapping
    else:

        th, phi = x, y
            
        c = exact.c
        d = exact.d

        # Calculate the reverse map matrix
        R = np.linalg.inv(M)

        # Init vector v
        v = np.zeros(3)

        v[0] = np.sin(th)*np.cos(phi)
        v[1] = np.sin(th)*np.sin(phi)
        v[2] = np.cos(th)

        # Calculate vector u
        u = np.zeros(3)     

        u[0] = R[0,0]*v[0] + R[0,1]*v[1] + R[0,2]*v[2]
        u[1] = R[1,0]*v[0] + R[1,1]*v[1] + R[1,2]*v[2]
        u[2] = R[2,0]*v[0] + R[2,1]*v[1] + R[2,2]*v[2]

        # Calculate beta and gamma
        bet = np.arctan2(np.hypot(u[0], u[1]), u[2])
        gam = np.arctan2(u[1], u[0])

        if bet > np.pi:
            return False

        # Project onto (p, q) plane
        p = np.sin(bet)*np.cos(gam)
        q = np.sin(bet)*np.sin(gam)

        u = c[0] + c[1]*p + c[2]*p**2 + c[3]*p**3 + c[4]*q + c[5]*q**2 + c[6]*q**3 + c[7]*p*q + c[8]*p**2*q + \
            c[9]*p*q**2

        v = d[0] + d[1]*p + d[2]*p**2 + d[3]*p**3 + d[4]*q + d[5]*q**2 + d[6]*q**3 + d[7]*p*q + d[8]*p**2*q + \
            d[9]*p*q**2

        # Calculate Hx, Hy
        x = u + exact.wid/2.0
        y = v + exact.ht/2.0

        return x, y



def loadScale(dir_path, file_name):
    """ Loads an AFF scale plate. 
    
    Arguments:
        dir_path: [str] path to the directory where the plate file is located
        file_name: [str] name of the plate file

    Return:
        [AffPlate object]
    """

    # Open the file for binary reading
    fid = open(os.path.join(dir_path, file_name), 'rb')

    # Init the plate struct
    scale = AffPlate()


    # Load file info
    scale.magic = int(np.fromfile(fid, dtype=np.uint32, count = 1))
    scale.info_len = int(np.fromfile(fid, dtype=np.uint32, count = 1))

    # Load reserved members
    scale.r0 = np.fromfile(fid, dtype=np.uint32, count = 1)
    scale.r1 = np.fromfile(fid, dtype=np.uint32, count = 1)
    scale.r2 = np.fromfile(fid, dtype=np.uint32, count = 1)
    scale.r3 = np.fromfile(fid, dtype=np.uint32, count = 1)

    # Load scaling terms
    scale.sx = np.fromfile(fid, dtype=np.float64, count = 1)
    scale.sy = np.fromfile(fid, dtype=np.float64, count = 1)

    # Load the rotation term
    scale.phi = np.fromfile(fid, dtype=np.float64, count = 1)[0]

    # Load the translation terms
    scale.tx = np.fromfile(fid, dtype=np.float64, count = 1)
    scale.ty = np.fromfile(fid, dtype=np.float64, count = 1)[0]

    # Load the image size parameters
    scale.wid = np.fromfile(fid, dtype=np.int32, count = 1)[0]
    scale.ht = np.fromfile(fid, dtype=np.int32, count = 1)[0]

    # Load the site number
    scale.site = np.fromfile(fid, dtype=np.uint32, count = 1)[0]

    # Load the descriptive comment
    scale.text = np.fromfile(fid, dtype='|S'+str(scale.text_size), count = 1)[0]

    # Load the flags
    scale.flags = np.fromfile(fid, dtype=np.uint32, count = 1)[0]

    # Calculate the conversion matrix
    scale.initM()


    return scale



def plateScaleMap(scale, x, y, reverse_map=False):
    """ Map the image delta coordinates (delta_Nx, delta_Ny) to encoder delta coordinates (Hu, Hv) given an 
        appropriate scale plate. If a reverse mapping is desired, set reverse_map=True.
        
    Arguments:
        scale: [AffPlate object] AFF plate structure
        x: [float] input parameter 1 (delta_Nx by default, Hu if reverse_map=True)
        y: [float] input parameter 2 (delta_Ny by default, Hv if reverse_map=True)

    Kwargs:
        reverse_map: [bool] default False, if True, revese mapping is performed

    Return:
        [tuple of floats]: output parameters (Hu, Hv) by default, (delta_Nx, delta_Ny) if reverse_map=True

    """

    M = scale.M

    # Run if doing the reverse mapping
    if reverse_map:

        # Calculate the reverse map matrix
        R = np.linalg.inv(M)

        # Reverse mapping
        px = R[0,0]*x + R[0,1]*y + R[0,2]
        py = R[1,0]*x + R[1,1]*y + R[1,2]

        return px, py


    else:

        # Forward mapping
        mx = M[0,0]*x + M[0,1]*y + M[0,2]
        my = M[1,0]*x + M[1,1]*y + M[1,2]

        return mx, my



if __name__ == "__main__":

    # Scale plate file path
    scale_dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MirfitPrepare/20160927_030027_mir"
    scale_file_name = "scale_01.aff"

    # Load the scale plate
    scale = loadScale(scale_dir_path, scale_file_name)

    print scale.M

    # Convert image (X, Y) to encoder (Hu, Hv)
    hu, hv = plateScaleMap(scale, 100, 100)
    print hu, hv

    # Reverse map (Hu, Hv) to (X, Y)
    x, y = plateScaleMap(scale, hu, hv, reverse_map=True)
    print x, y


    # Exact plate file path
    exact_dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MirfitPrepare/20160927_030027_mir"
    exact_file_name = "exact_01.ast"

    # Load the exact plate
    exact = loadExact(exact_dir_path, exact_file_name)

    print exact.M
    print 'Exact lat, lon {:>12.6f}, {:>12.6f}'.format(np.degrees(exact.lat), np.degrees(exact.lon)), exact.elev

    # Convert (Hx, Hy) to (theta, phi)
    theta, phi = plateExactMap(exact, 31997, 22290)

    print np.degrees(theta), np.degrees(phi)

    # Convert (theta, phi) to (Hx, Hy)
    print plateExactMap(exact, theta, phi, reverse_map=True)

    print 'exact test'
    print plateExactMap(exact, np.radians(55.438), np.radians(128.128), reverse_map=True)
