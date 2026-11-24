""" Synthetic meteors for the tests of the alpha-beta fits, integrated in time along a straight path,
independently of the alpha-beta functions which are tested.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.integrate

from wmpl.Utils.AlphaBeta import alphaBetaVelocityNormed, HT_NORM_CONST



def integrateAlphaBetaPath(time_data, v_init, alpha, beta, slope, ht_begin, rtol=1e-10, atol=1e-6):
    """ Integrate the distance along a straight path of a meteor with the alpha-beta velocity, in the
        exponential atmosphere of the model.

    Arguments:
        time_data: [ndarray] Increasing times at which the distance is computed, from the time of ht_begin
            (s).
        v_init: [float] Velocity at entry (m/s).
        alpha: [float] Alpha of the meteor.
        beta: [float] Beta of the meteor.
        slope: [float] Slope of the path to the horizontal (rad).
        ht_begin: [float] Height at the first time (m).

    Keyword arguments:
        rtol: [float] Relative tolerance of the integration. 1e-10 by default.
        atol: [float] Absolute tolerance of the integration (m). 1e-6 by default.

    Return:
        (length_data, ht_data):
            - length_data: [ndarray] Distance along the path at every time (m).
            - ht_data: [ndarray] Height at every time (m).
    """

    def _velocity(t, length):
        """ Velocity along the path at the given distance (m/s). """

        ht_normed = (ht_begin - length*np.sin(slope))/HT_NORM_CONST

        return v_init*alphaBetaVelocityNormed(ht_normed, alpha, beta)


    sol = scipy.integrate.solve_ivp(_velocity, (time_data[0], time_data[-1]), [0.0], t_eval=time_data, \
        rtol=rtol, atol=atol)
    length_data = sol.y[0]

    return length_data, ht_begin - length_data*np.sin(slope)
