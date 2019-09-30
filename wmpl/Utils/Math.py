""" General mathematical conversions, linear algebra functions, and geometrical functions. 

"""

from __future__ import division, print_function, absolute_import

import datetime
import pytz

import numpy as np
from numpy.core.umath_tests import inner1d
import scipy.linalg
import scipy.stats
import scipy.spatial
import scipy.optimize



### BASIC FUNCTIONS ###
##############################################################################################################

def lineFunc(x, m, k):
    """ A line function.

    Arguments:
        x: [float] Independant variable.
        m: [float] Slope.
        k: [float] Intercept.

    Return:
        y: [float] Line evaluation.
    """

    return m*x + k


##############################################################################################################



### VECTORS ###
##############################################################################################################


def vectNorm(vect):
    """ Convert a given vector to a unit vector. """

    return vect/vectMag(vect)



def vectMag(vect):
    """ Calculate the magnitude of the given vector. """

    return np.sqrt(inner1d(vect, vect))



def rotateVector(vect, axis, theta):
    """ Rotate vector around the given axis by a given angle.

    Arguments:
        vect: [3 element ndarray] vector to be rotated
        axis: [3 element ndarray] axis of rotation
        theta: [float] angle of rotation (radians)

    Return:
        [3 element ndarray] rotated vector

    """
    
    rot_M = scipy.linalg.expm(np.cross(np.eye(3), axis/vectMag(axis)*theta))

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


def angleBetweenVectors(a, b):
    """ Compute the angle between two vectors. 
    
    Arguments:
        a: [ndarray] First vector.
        b: [ndarray] Second vector.

    Return:
        [float] Angle between a and b (radians).
    """

    return np.arccos(np.dot(a, b)/(vectMag(a)*vectMag(b)))


def vectorFromPointDirectionAndAngle(pos, dir_hat, angle):
    """ Compute a new vector given an initial position, direction and and angle between the initial and the
    final position.

    See link for a detailed explanation: 
    https://math.stackexchange.com/questions/3297191/finding-the-direction-vector-magnitude-from-position-and-angle

    Arguments:
        pos: [ndarray] Initial position vector.
        dir_path: [ndarray] Direction unit vector.
        angle: [float] Angle between the inital and the final position vectors (radians).

    Return:
        [ndarray] Final position vector.
    """

    dir_hat = vectNorm(dir_hat)

    # Compute the scalar which will scale the unit direction vector
    beta = np.arccos(np.dot(dir_hat, pos)/vectMag(pos)) - angle
    k = vectMag(pos)*np.sin(angle)/np.sin(beta)

    return pos + k*dir_hat
    

##############################################################################################################




### POLAR/ANGULAR FUNCTIONS ###
##############################################################################################################


def meanAngle(angles):
    """ Calculate the mean angle from a list of angles. 

    Arguments:
        angles: [list of floats] A list of angles (in radians) used to calculate their mean value.

    Return:
        [float] Mean value of the given angles (in radians).
    """

    angles = np.array(angles)

    return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))




def normalizeAngleWrap(arr):
    """ Given an array with angles which possibly have values close to 0 and 360, normalize all angles
        to 0 (i.e. values > 180 will be negative).

    Arguments:
        arr: [ndarray] Array of angles (radians).


    Return:
        arr: [ndarray] Normalized array of angles (radians).

    """


    min_ang = np.min(arr)
    max_ang = np.max(arr)

    # Check if the data was wrapped
    if abs(max_ang - min_ang) >= np.pi:

        arr = np.array(arr)

        # Subtract 360 from large angles
        arr[arr >= np.pi] -= 2*np.pi


    return arr



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



def cartesianToSpherical(x, y, z):
    """ Convert cartesian coordinates to spherical coordinates. 
    
    Arguments:
        x: [float] 
        y: [float]
        z: [float]
    
    Return:
        r, theta, phi: [tuple of floats] Spherical coordinates (angles given in radians)    
            r - Radius (radians).
            theta - Elevation (radians).
            phi - Azimuth (radians).
    """

    # Radius
    r = np.sqrt(x**2 + y**2 + z**2)

    # Azimuith
    phi = np.arctan2(y, x)

    # Elevation
    theta = np.arccos(z/r)


    return r, theta, phi



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


##############################################################################################################




### 3D FUNCTIONS ###
##############################################################################################################


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



def lineAndSphereIntersections(centre, radius, origin, direction):
    """ Finds intersections between a sphere of given radius and coordiantes of the centre and a line
        defined by an origin and a direction vector.

    Source: http://www.lighthouse3d.com/tutorials/maths/ray-sphere-intersection/
    
    Argument:
        centre: [3 element ndarray] Coordinates of the centre of the sphere.
        radius: [float] Radius of the sphere.
        origin: [3 element ndarray] Coordinates of the origin of the line.
        direction: [3 element ndarray] 3D direction vector of the line

    Return:
        [list] A list of intersections, every intersections is a 3 element ndarray.
    """

    intersection_list = []


    # Make sure the direction is a unit vector
    direction = vectNorm(direction)

    # Vector pointing from the origin of the line to the centre of the sphere
    v = centre - origin

    # Projection of the centre on the line
    pc = origin + np.dot(direction, v)/vectMag(direction)*direction

    
    # No solutions
    if vectMag(centre - pc) > radius:

        # No intersection
        return intersection_list


    # Check if the line is only skimming the sphere
    elif vectMag(centre - pc) == radius:

        intersection_list.append(pc)

        return intersection_list


    # There are 2 solutions
    else:

        # Check if the sphere is behind the origin
        if np.dot(v, direction) < 0:

            # Distance from the projection to the intersection
            dist = np.sqrt(radius**2 - vectMag(pc - centre)**2)
            
            # Intersection at the front
            dil = dist - vectMag(pc - origin)
            intersection = origin + direction*dil

            intersection_list.append(intersection)


            # Intersection at the back
            dil = dist + vectMag(pc - origin)
            intersection = origin - direction*dil

            intersection_list.append(intersection)

            return intersection_list


        # If the sphere is in front of the origin
        else:


            # Distance from the projection to the intersection
            dist = np.sqrt(radius**2 - vectMag(pc - centre)**2)

            
            # Intersection at the front
            dil = vectMag(pc - origin) - dist
            intersection = origin + direction*dil

            intersection_list.append(intersection)

            # Intersection at the back
            dil = vectMag(pc - origin) + dist
            intersection = origin + direction*dil

            intersection_list.append(intersection)


            return intersection_list




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
    # inside_points = []
    ###

    # Do niter iterations
    for i in range(niter):

        # Check if the point is inside the other hull
        inside = pointInsideConvexHull(hull2, test_points[i])

        if inside:
            inside_count += 1

            ## TEST
            # inside_points.append(test_points[i])


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



##############################################################################################################


def RMSD(x, weights=None):
    """ Root-mean-square deviation of measurements vs. model. """

    if weights is None:
        weights = np.ones_like(x)

    return np.sqrt(np.sum(weights*x**2)/np.sum(weights))
    


def averageClosePoints(x_array, y_array, delta, x_datetime=False):
    """ Finds if points have similar sample values on the independant axis x (if they are within delta) and 
        averages all y values.

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



def movingAverage(arr, n=3):
    """ Perform a moving average on an array with the window size n.

    Arguments:
        arr: [ndarray] Numpy array of values.

    Keyword arguments:
        n: [int] Averaging window.

    Return:
        [ndarray] Averaged array. The size of the array is always by n-1 smaller than the input array.

    """

    ret = np.cumsum(arr, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:]/n


def subsampleAverage(arr, n=3):
    """ Average the given array and subsample it. 

    Source: https://stackoverflow.com/a/10847914/6002120
    
    Arguments:
        arr: [ndarray] Numpy array of values.

    Keyword arguments:
        n: [int] Averaging window.

    Return:
        [ndarray] Subsampled array.

    """

    end =  n*int(len(arr)/n)

    return np.mean(arr[:end].reshape(-1, n), 1)



def checkContinuity(sequence):
    """ Given a sequence of 1s and 0s, check if there is exactly one continuous sequence of 1s. 

    Arguments:
        sequence: [list] A list of 0s and 1s.

    Return:
        (status, first_index, last_index): [bool], [int], [int] True if the sequence contains exaclty one 
            continous string of 1s, False otherwise. first_index and last_index are indices of the beginning
            and the end of the sequence.

    Examples:

        Input: [1, 1, 1, 1, 1, 1]
        Output: True, 0, 5

        Input: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        Output: True, 1, 8
        
        Input: [0, 0, 0, 0, 0]
        Output: False, 0, 0

        Input: [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
        Output: False, 0, 0
    """

    sequence = np.array(sequence)

    # Compute element-wise differences
    diffs = sequence[:-1] - sequence[1:]

    # Check if the number of differences is 2 or less, indicating one continous sequence
    if np.count_nonzero(diffs) == 0:

        # Make sure there are more then one 1s in the sequence
        if np.count_nonzero(sequence == 1) > 1:
            return True, 0, len(sequence) - 1

    elif np.count_nonzero(diffs) == 1:

        # Find the index at which the change occurs
        change_indx = np.argwhere(diffs != 0)[0][0]

        # If the first part are zeros
        if diffs[change_indx] < 0:
            return True, change_indx + 1, len(sequence) - 1

        else:
            return True, 0, change_indx


    elif np.count_nonzero(diffs) == 2:

        # Check if that continous sequence is a sequence of ones, or a sequence of zeros
        first, last = np.argwhere(diffs != 0)

        if np.count_nonzero(sequence[first[0]+1:last[0]] == 1):
            return True, first[0] + 1, last[0]

    return False, 0, 0




def histogramEdgesEqualDataNumber(x, nbins):
    """ Given the data, divide the histogram edges in such a way that every bin has the same number of
        data points. 

        Source: https://stackoverflow.com/questions/37649342/matplotlib-how-to-make-a-histogram-with-bins-of-equal-area/37667480

    Arguments:
        x: [list] Input data.
        nbins: [int] Number of bins.

    """

    npt = len(x)
    return np.interp(np.linspace(0, npt, nbins + 1), np.arange(npt), np.sort(x))



### OPTIMIZATION ###
##############################################################################################################


def fitConfidenceInterval(x_data, y_data, conf=0.95, x_array=None, func=None):
    """ Fits a line to the data and calculates the confidence interval. 

    Source: https://gist.github.com/rsnemmen/f2c03beb391db809c90f

    Arguments:
        x_data: [ndarray] Independent variable data.
        y_data: [ndarray] Dependent variable data.

    Keyword arguments:
        conf: [float] Confidence interval (fraction, 0.0 to 1.0).
        x_array: [ndarray] Array for evaluating the confidence interval on. Usually used as an independant 
            variable array for plotting.
        func: [function] Function to fit. None by default, which means a line will be fit.


    Return:
        line_params, lcb, ucb, x_array:
            line_params: [list] Fitted line parameters.
            sd: [float] Standard deviation of the fit in Y.
            lcb: [ndarray] Lower bound values evaluated on x_array.
            ucb: [ndarray] Upper bound values evaluated on x_array.
            x_array: [ndarray] Independent variable array for lcb and ucb evaluation.
    """

    if func is None:
        func = lineFunc

    # Fit a line
    fit_params, _ = scipy.optimize.curve_fit(func, x_data, y_data)


    alpha = 1.0 - conf
    n = x_data.size

    # Array for evaluation the fit and the confidence interval on
    if x_array is None:
        x_array = np.linspace(x_data.min(), x_data.max(), 100)

    # Predicted values (best-fit model)
    y_array = func(x_array, *fit_params)

    # Auxiliary definitions
    sxd = np.sum((x_data - x_data.mean())**2)
    sx = (x_array - x_data.mean())**2

    # Quantile of Student's t distribution for p=1-alpha/2
    q = scipy.stats.t.ppf(1.0 - alpha/2.0, n-2)

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)  
    N = np.size(x_data)  
    sd = 1.0/(N - 2.0)*np.sum((y_data - func(x_data, *fit_params))**2)
    sd = np.sqrt(sd)

    # Confidence band
    dy = q*sd*np.sqrt(1 + 1.0/n + sx/sxd)
    
    # Upper confidence band
    ucb = y_array + dy    

    # Lower confidence band
    lcb = y_array - dy    


    return fit_params, sd, lcb, ucb, x_array


##############################################################################################################




### TIME FUNCTIONS ###
##############################################################################################################


def generateDatetimeBins(dt_beg, dt_end, bin_days=7, utc_hour_break=12):
    """ Given a beginning and end datetime, bin this time range into bins bin_days long. The bin edges will
        be at utc_hour_break UTC. 12:00 UTC is chosen because at that time it is midnight at the International
        Date Line, and it is very unlikely that there are any meteor cameras there.

    Arguments:
        dt_beg: [datetime] Begin datetime.
        dt_end: [datetime] End datetime.

    Keyword arguments:
        bin_days: [float] Length of bin in days.
        utc_hour_break: [float] UTC hour when the break in time will occur, i.e. this will be the edges of the
            time bins.

    Return:
        [list] A list of (bin_beg, bin_end) datetime pairs.
    """

    # Convert input times to UTC
    dt_beg = dt_beg.replace(tzinfo=pytz.utc)
    dt_end = dt_end.replace(tzinfo=pytz.utc)


    # Compute the total number of bins
    n_bins = np.ceil((dt_end - dt_beg).total_seconds()/(bin_days*86400))

    # Generate time bins
    time_bins = []
    for i in range(int(n_bins)):

        # Generate the bin beginning edge
        if i == 0:
            # For the first bin beginning, use the initial time
            bin_beg = dt_beg

        else:
            bin_beg = dt_beg + datetime.timedelta(days=i*bin_days)
            bin_beg = bin_beg.replace(hour=int(utc_hour_break), minute=0, second=0, microsecond=0)

        # Generate the bin ending edge
        bin_end = bin_beg + datetime.timedelta(days=bin_days)
        bin_end = bin_end.replace(hour=int(utc_hour_break), minute=0, second=0, microsecond=0)

        # Check that the ending bin is not beyond the end dt
        end_reached = False
        if bin_end > dt_end:
            bin_end = dt_end
            end_reached = True


        time_bins.append([bin_beg, bin_end])

        # Stop iterating if the end was reached
        if end_reached:
            break


    return time_bins




##############################################################################################################


if __name__ == "__main__":

    import sys
    import matplotlib.pyplot as plt

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




    # Test cartesian to spherical transform
    x = -3125996.00181 
    y = 5180605.28206 
    z = 4272017.20178

    print('X, Y, Z')
    print(x, y, z)

    # Convert to spherical
    r, theta, phi = cartesianToSpherical(x, y, z)

    print('r, phi, theta')
    print(r, phi, theta)

    # Convert back to cartesian
    x2, y2, z2 = sphericalToCartesian(r, theta, phi)

    print('Converted X, Y, Z')
    print(x2, y2, z2)




    # Generate fake data, fit a line and get confidence interval
    x_data = np.linspace(1, 10, 100)
    y_data = np.random.normal(0, 1, size=len(x_data))

    # Fit a line with the given confidence interval
    line_params, lcb, ucb, x_array = fitConfidenceInterval(x_data, y_data, conf=0.95)


    plt.scatter(x_data, y_data)
    plt.plot(x_array, lineFunc(x_array, *line_params))
    plt.plot(x_array, lcb)
    plt.plot(x_array, ucb)

    plt.plot()

    plt.show()



    sys.exit()

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