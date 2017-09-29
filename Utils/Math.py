""" General mathematical conversions, linear algebra functions, and geometrical functions. 

"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.linalg
import scipy.spatial
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
        x: [float] Px coordinate.
        y: [float] Py coordinate.
        z: [float] Pz coordinate.

    Return:
        (theta, phi): [float] Polar angles in radians (inclination, azimuth).

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



def pointInsideConvexHull(hull_vertices, point):
    """ Checks whether the given point is inside the 3D hull defined by given vertices. 
    
    Arguments:
        hull_vertices: [2D ndarray] 2D numpy array containing 3D coordinates (x, y, z) of hull vertices.
        point: [ndarray] (x, y, z) coordinates of the test point.

    Return:
        [bool]: True if point is inside the hull, False otherwise.

    """

    # Create a convex hull from the given hull vertices
    hull = scipy.spatial.ConvexHull(hull_vertices)

    # Create a new hull by adding the point we are checking
    new_hull = scipy.spatial.ConvexHull(np.concatenate((hull.points, [point])))

    # Check if the veretices have remained the same
    if np.array_equal(new_hull.vertices, hull.vertices): 
        return True
    
    return False



def samplePointsFromHull(hull_vertices, n_points):
    """ Randomly samples points inside the given convex hull. 
    
    Arguments:
        hull_vertices: [2D ndarray] 2D numpy array containing 3D coordinates (x, y, z) of hull vertices.
        n_points: [int] Number of points to sample from the hull.

    Return:
        samples_hull: [list] A list of points sampled from the hull.
    """


    # Find a rectangular cuboid which envelops the given hull
    min_point = np.array([ np.inf,  np.inf,  np.inf])
    max_point = np.array([-np.inf, -np.inf, -np.inf])

    for vert in hull_vertices:
        min_point = np.min([min_point, vert], axis=0)
        max_point = np.max([max_point, vert], axis=0)


    samples_hull = []


    while True:

        # Draw a sample from the rectangular cuboid
        x = np.random.uniform(min_point[0], max_point[0])
        y = np.random.uniform(min_point[1], max_point[1])
        z = np.random.uniform(min_point[2], max_point[2])

        point = np.array([x, y, z])

        # Check if the point is inside the hull
        if pointInsideConvexHull(hull_vertices, point):
            samples_hull.append(point)

        # Check if enough points were samples
        if len(samples_hull) == n_points:
            break


    return np.array(samples_hull)



def estimateHullOverlapRatio(hull1, hull2, niter=200, volume=False):
    """ Given two convex hulls, estimate their overlap ratio by randomly generating points inside the first
        and counting how many are inside the other. The ratio between the common and all points is the
        estimate of the overlap ratio.

    Arguments:
        hull1: [list] A list of points in the first convex hull.
        hull2: [list] A list of point in the second convex hull.

    Keyword arguments:
        niter: [int] Number of iterations for generating the points. 200 by default.
        volume: [bool] If True, the common volume between the cameras will be returned instead of the ratio.

    """

    inside_count = 0

    # Randomly generate a point inside the first convex hull
    test_points = samplePointsFromHull(hull1, niter)



    ## TEST
    inside_points = []
    ###

    # Do niter iterations
    for i in range(niter):

        # Check if the point is inside the other hull
        inside = pointInsideConvexHull(hull2, test_points[i])

        if inside:
            inside_count += 1

            ## TEST
            inside_points.append(test_points[i])


    ratio = float(inside_count)/niter



    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # colors = ['g', 'y']
    # for verts, color in zip([hull1, hull2], colors):

    #     hull = scipy.spatial.ConvexHull(verts)

    #     # Plot vertices
    #     ax.plot(verts.T[0], verts.T[1], verts.T[2], c=color)

    #     # Plot edges
    #     for simp in hull.simplices:

    #         # Cycle to the first coordinate
    #         simp = np.append(simp, simp[0])

    #         # Plot the edge
    #         ax.plot(verts[simp, 0], verts[simp, 1], verts[simp, 2], c=color)


    # # Plot convex hull1 vertices
    # #ax.scatter(hull1[:, 0], hull1[:, 1], hull1[:, 2], c='g')

    # # Plot convex hull2 vertices
    # #ax.scatter(hull2[:, 0], hull2[:, 1], hull2[:, 2], c='y')


    # # Plot all points
    # for pt in test_points:

    #     # Plot corner
    #     ax.scatter(*pt, c='b')

    # # Plot points inside both hulls
    # for pt in inside_points:

    #     # Plot corner
    #     ax.scatter(*pt, c='r')


    # # ax.set_xlim([-1, 1])
    # # ax.set_ylim([-1, 1])
    # # ax.set_zlim([-1, 1])
    # plt.show()




    # Return common volume if it was requested
    if volume:
        return ratio*scipy.spatial.ConvexHull(hull1).volume


    # Return ratio otherwise
    else:
        return ratio

        

def RMSD(x):
    """ Root-mean-square deviation of measurements vs. model. """

    return np.sqrt(np.sum(x**2)/len(x))
    


def averageClosePoints(x_array, y_array, delta, x_datetime=False):
    """ Finds if points have close values of x and averages all close values in y.

    Arguments:
        x_array: [list] A list of x values (must be of the same length as y_array!).
        y_array: [list] A list of y values (must be of the same length as x_array!).
        delta: [float] Threshold distance between two points in x to merge them. Can be sampling of data (e.g. 
            half the fps).

    Keyword arguments: 
        x_datatime: [bool] Should be True if X is an array of datetime objects. False by default.

    Return:
        x_final, y_final: [tuple of lists] Processed x and y arrays.

    """

    x_final = []
    y_final = []
    skip = 0

    # Step through all x values
    for i, x in enumerate(x_array):

        if skip > 0:
            skip -= 1
            continue


        # Calculate the difference between this point and all others
        diff = np.abs(x_array - x)

        # Convert the difference to seconds if X array elements are datetime
        if x_datetime:
            diff_iter = (time_diff.total_seconds() for time_diff in diff)
            diff = np.fromiter(diff_iter, np.float64, count=len(diff))


        # Count the number of close points to this element
        count = np.count_nonzero(diff < delta)

        # If there are more than one, average them and put them to the list
        if count > 1:

            skip += count - 1

            y = np.mean(y_array[i : i + count])


        # If there are no close points, add the current point to the list
        else:

            y = y_array[i]


        x_final.append(x)
        y_final.append(y)


    return x_final, y_final








if __name__ == "__main__":

    ### TESTS

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

    print(cx_rot + x_diff, cy_rot + y_diff)



    # Test hull functions
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    # Define hull points
    hull_vertices = np.array([
        [100, 100, 120],
        [ 98,  10, 120],
        [  5,   8, 120],
        [  3, 103, 120],

        [80, 78,  70],
        [81, 21,  70],
        [22, 23,  70],
        [19, 85, 70]])

    # Point to check
    point = np.array([10, 50, 80])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")


    print("Point", point, "inside the hull:", pointInsideConvexHull(hull_vertices, point))


    # Take samples from the hull
    samples_hull = samplePointsFromHull(hull_vertices, 100)

    # Plot point
    ax.scatter(*point, c='r')

    # Plot hull
    ax.scatter(hull_vertices[:, 0], hull_vertices[:, 1], hull_vertices[:, 2], c='b')

    # Plot samples
    ax.scatter(samples_hull[:, 0], samples_hull[:, 1], samples_hull[:, 2], c='g')

    plt.show()