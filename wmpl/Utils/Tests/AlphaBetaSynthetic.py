""" Synthetic meteors for the tests of the alpha-beta fits, integrated in time along a straight path,
independently of the alpha-beta functions which are tested.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.integrate

from wmpl.Utils.AlphaBeta import alphaBetaVelocityNormed, HT_NORM_CONST



def integrateAlphaBetaPath(time_data, v_init, alpha, beta, slope, ht_begin, rtol=1e-10, atol=1e-6, \
    knot_ht_normed=None, knot_log_mass=None):
    """ Integrate the distance along a straight path of a meteor with the alpha-beta velocity, in the
        exponential atmosphere of the model.

        With knots, the meteor loses mass at them and its alpha grows by exp(u/3), with u the logarithm of the
        ratio of the masses before and after. The normalized velocity v is then integrated in time with the
        distance, by dv/dy = alpha_j*v*exp(beta*(1 - v^2) - y) of the single body between the knots, from the
        velocity of the single body at ht_begin, above the first knot, and restarted at every knot with the
        next alpha.

    Arguments:
        time_data: [ndarray] Increasing times at which the distance is computed, from the time of ht_begin
            (s).
        v_init: [float] Velocity at entry (m/s).
        alpha: [float] Alpha of the meteor, before the first knot.
        beta: [float] Beta of the meteor.
        slope: [float] Slope of the path to the horizontal (rad).
        ht_begin: [float] Height at the first time (m).

    Keyword arguments:
        rtol: [float] Relative tolerance of the integration. 1e-10 by default.
        atol: [float] Absolute tolerance of the integration (m). 1e-6 by default.
        knot_ht_normed: [ndarray] Normalized heights of the knots, decreasing. None by default, for a single
            body.
        knot_log_mass: [ndarray] Logarithm u of the ratio of the masses before and after every knot. None by
            default.

    Return:
        (length_data, ht_data):
            - length_data: [ndarray] Distance along the path at every time (m).
            - ht_data: [ndarray] Height at every time (m).
    """

    def _velocity(t, length):
        """ Velocity along the path at the given distance (m/s). """

        ht_normed = (ht_begin - length*np.sin(slope))/HT_NORM_CONST

        return v_init*alphaBetaVelocityNormed(ht_normed, alpha, beta)


    if knot_ht_normed is None:

        sol = scipy.integrate.solve_ivp(_velocity, (time_data[0], time_data[-1]), [0.0], t_eval=time_data, \
            rtol=rtol, atol=atol)
        length_data = sol.y[0]

        return length_data, ht_begin - length_data*np.sin(slope)


    def _motion(t, state, alpha_segment):
        """ Time derivatives of the distance and of the normalized velocity. """

        length, v_normed = state
        ht_normed = (ht_begin - length*np.sin(slope))/HT_NORM_CONST
        ht_rate = -v_init*v_normed*np.sin(slope)/HT_NORM_CONST

        return [v_init*v_normed, alpha_segment*v_normed*np.exp(beta*(1.0 - v_normed**2) - ht_normed)*ht_rate]


    # Integrate every segment up to the next knot
    alpha_segments = alpha*np.exp(np.concatenate([[0.0], np.cumsum(np.asarray(knot_log_mass)/3.0)]))
    state = [0.0, float(np.squeeze(alphaBetaVelocityNormed(ht_begin/HT_NORM_CONST, alpha, beta)))]
    time_start = time_data[0]
    segments = []
    for j, alpha_segment in enumerate(alpha_segments):

        events = None
        if j < len(knot_ht_normed):

            def _aboveKnot(t, state, alpha_segment, ht_knot=knot_ht_normed[j]):
                """ Normalized height above the next knot. """

                return (ht_begin - state[0]*np.sin(slope))/HT_NORM_CONST - ht_knot


            _aboveKnot.terminal = True
            _aboveKnot.direction = -1
            events = _aboveKnot

        sol = scipy.integrate.solve_ivp(_motion, (time_start, time_data[-1]), state, args=(alpha_segment,), \
            events=events, dense_output=True, rtol=rtol, atol=atol)
        segments.append((time_start, sol.t[-1], sol.sol))

        # Stop at the last time, or continue from the knot
        if sol.status != 1:
            break

        time_start, state = sol.t[-1], sol.y[:, -1]

    length_data = np.zeros(len(time_data))
    for time_lower, time_upper, solution in segments:
        in_segment = (time_data >= time_lower) & (time_data <= time_upper)
        length_data[in_segment] = solution(time_data[in_segment])[0]

    return length_data, ht_begin - length_data*np.sin(slope)
