""" General mathematical conversions. 

"""

import numpy as np
import scipy.linalg
from numpy.core.umath_tests import inner1d


def vectNorm(vect):
    """ Convert a given vector to a unit vector. """

    return vect/vectMag(vect)



def vectMag(vect):
    """ Calculate the magnitude of the given vector. """

    return np.sqrt(inner1d(vect, vect))



def meanAngle(angles):
    """ Calculate the mean angle from a list of angles. 

    Arguments:
        angles: [list of floats] A list of angles (in radians) used to calculate their mean value.

    Return:
        [float] Mean value of the given angles (in radians).
    """

    angles = np.array(angles)

    return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))



def cartesianToPolar(x, y, z):
    """ Converts 3D cartesian coordinates to polar coordinates. 

    Arguments:
        x: [float] point x coordinate
        y: [float] point y coordinate
        z: [float] point z coordinate

    Return:
        (theta, phi): [float] polar angles in radians

    """

    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    return theta, phi



def polarToCartesian(theta, phi):
    """ Converts 3D spherical coordinates to 3D cartesian coordinates. 

    Arguments:
        theta: [float] Inclination in radians.
        phi: [float] Azimuth angle in radians.

    Return:
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartiesian coordinates.
    """


    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    return x, y, z



def sphericalToCartesian(r, theta, phi):
    """ Convert spherical coordinates to cartesian coordinates. 
        
    Arguments:
        r: [float] Radius
        theta: [float] Inclination in radians.
        phi: [float] Azimuth angle in radians.

    Return:
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartiesian coordinates.
    """

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x, y, z


def angleBetweenSphericalCoords(phi1, lambda1, phi2, lambda2):
    """ Calculates the angle between two points on a sphere. 
    
    Arguments:
        phi1: [float] Latitude 1 (radians).
        lambda1: [float] Longitude 1 (radians).
        phi2: [float] Latitude 2 (radians).
        lambda2: [float] Longitude 2 (radians).

    Return:
        [float] Angle between two coordinates (radians).
    """

    return np.arccos(np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(lambda2 - lambda1))






def rotateVector(vect, axis, theta):
    """ Rotate vector around the given axis for the given angle.

    Arguments:
        vect: [3 element ndarray] vector to be rotated
        axis: [3 element ndarray] axis of rotation
        theta: [float] angle of rotation in radians

    Return:
        [3 element ndarray] rotated vector

    """
    
    rot_M = scipy.linalg.expm3(np.cross(np.eye(3), axis/vectMag(axis)*theta))

    return np.dot(rot_M, vect)



def rotatePoint(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.

    Source: http://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python

    Arguments:
        origin: [tuple of floats] (x, y) pair of Cartesian coordinates of the origin
        point: [tuple of floats] (x, y) pair of Cartesian coordinates of the point
        angle: [float] angle of rotation in radians

    Return:
        (qx, qy): [tuple of floats] Cartesian coordinates of the rotated point
    """

    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle)*(px - ox) - np.sin(angle)*(py - oy)
    qy = oy + np.sin(angle)*(px - ox) + np.cos(angle)*(py - oy)

    return qx, qy



def getRotMatrix(v1, v2):
    """ Generates a rotation matrix which is used to rotate from vector v1 to v2. 

        Source: http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    
    """

    # Rotation vector
    w = np.cross(v1, v2)

    # Check if the vectors are 0, then return I matrix
    if np.sum(np.abs(w)) == 0:
        return np.eye(3)

    w = vectNorm(w)

    # Skew-symmetric cross-product matrix
    w_hat = np.array([[    0, -w[2],  w[1]],
                      [ w[2],     0, -w[0]],
                      [-w[1], w[0],     0]])

    # Rotation angle
    theta = np.arccos(np.dot(vectNorm(v1), vectNorm(v2)))

    # Construct the rotation matrix
    R = np.eye(3) + w_hat*np.sin(theta) + np.dot(w_hat, w_hat)*(1 - np.cos(theta))

    return R



def findClosestPoints(P, u, Q, v):
    """ Calculate coordinates of closest point of approach (CPA) between lines of sight (LoS) of two 
        observers.

        Source: http://geomalgorithms.com/a07-_distance.html

    Arguments:
        P: [3 element vector] position coordinates of the 1st observer
        u: [3 element vector] 1st observer's direction vector
        Q: [3 element vector] position coordinates of the 2nd observer
        v: [3 element vector] 2nd observer's direction vector

    Return:
        S: [3 element vector] point on the 1st observer's LoS closest to the 2nd observer's LoS
        T: [3 element vector] point on the 2nd observer's LoS closest to the 1st observer's LoS
        d: [float] distance between r1 and r2
        
    """

    # Calculate the difference in position between the observers
    w = P - Q

    # Calculate cosines of angles between various vectors
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    sc = (b*e - c*d)/(a*c - b**2)
    tc = (a*e - b*d)/(a*c - b**2)

    # Point on the 1st observer's line of sight closest to the LoS of the 2nd observer
    S = P + u*sc

    # Point on the 2nd observer's line of sight closest to the LoS of the 1st observer
    T = Q + v*tc

    # Calculate the distance between S and T
    d = np.linalg.norm(S - T)

    return S, T, d




def pointInsidePolygon(x, y, poly):
    """ Checks if the given point (x, y) is inside a polygon. 

    Source: http://www.ariel.com.au/a/python-point-int-poly.html

    Arguments:
        x: [float] point x coordinate
        y: [float] point y coordinate
        poly: [2D list] a list of (x, y) pairs which are vertices of a polygon

    Return:
        inside: [bool] True if point inside the polygon, False otherwise
    """

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]

    for i in range(n+1):

        p2x,p2y = poly[i % n]

        if y > min(p1y,p2y):

            if y <= max(p1y,p2y):

                if x <= max(p1x,p2x):

                    if p1y != p2y:
                        xinters = (y - p1y)*(p2x - p1x)/(p2y - p1y) + p1x

                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x,p1y = p2x,p2y

    return inside



if __name__ == "__main__":

    # Cx, Cy
    cx, cy = 229, 163

    # Rotated (true):
    # x, y - 233, 440

    # Rotated (calc):
    # x, y - 359, 128

    # Rot img size: 673, 767

    img0 = 480
    img1 = 640
    img_rot0 = 767
    img_rot1 = 673

    angle = -np.radians(69.59197792)

    cx_rot, cy_rot = rotatePoint((320, 240), (cx, cy), angle)

    y_diff = (img_rot0 - img0)/2
    x_diff = (img_rot1 - img1)/2

    print cx_rot + x_diff, cy_rot + y_diff