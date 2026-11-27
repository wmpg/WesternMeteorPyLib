""" MetSim's erosion model from the alpha-beta closed form of the ablation in the slant column, a faster
alternative to the time steps of MetSimErosion.runSimulation that returns its result columns as MetSim gives
them at a fine time step.

Between the events of a body (the start and the change of the erosion, the fragmentation entries, the
disruption, the end) its velocity and mass follow the alpha-beta solution of the single-body equations
(Gritsevich 2007, 2009) with mu = 2/3, and its light the closed-form light curve of Gritsevich & Koschny
(2011) with MetSim's luminous efficiency. The coordinate of the solution is the slant column, the air column
along the path, as in the fit of the entry lag in wmpl.Utils.AlphaBeta (AlphaBetaSlantColumn, WMPL pull
request #242), and the length and the time along the path follow from it. The grains are released in epochs
of the column, gravity is taken to first order around the closed form, and the light is summed exactly into
the bins of a 1 ms grid. The functions whose names start with nb are compiled with numba.

References:
    Gritsevich, M. I. (2007). Approximation of the observed motion of bolides by the analytical solution of
    the equations of meteor physics. Solar System Research, 41(6), 509-514.

    Gritsevich, M. I. (2009). Determination of parameters of meteor bodies based on flight observational
    data. Advances in Space Research, 44(3), 323-334.

    Gritsevich, M., & Koschny, D. (2011). Constraining the luminous efficiency of meteors. Icarus, 212(2),
    877-884.

"""

from __future__ import print_function, division, absolute_import

import copy

import numpy as np
import scipy.integrate

try:
    # If Numba is available, use the jit decorator with specified options
    from numba import njit

except ImportError:
    # If Numba is not available, define a no-op decorator that just returns the function unchanged
    def njit(*args, **kwargs):
        """ Return the function unchanged, with or without the decorator's options. """

        # Used without options, return the function itself
        if args and callable(args[0]):
            return args[0]

        # Used with options, return a decorator that changes nothing
        return lambda func: func

import wmpl.MetSim.MetSimErosion as MetSimErosion
from wmpl.MetSim.MetSimErosion import usesAtmDensityTable, atmDensityTableArrays
from wmpl.Utils.Math import expEi, expEiScalar


# Target spacing of the returned results, a whole fraction of MetSim's step (s)
OUTPUT_STEP = 0.001

# Least nodes of a grain, its log nodes per e-fold of the column, and their rounding step
GRAIN_NODES = 40
GRAIN_NODES_PER_FOLD = 12.0
GRAIN_NODE_STEP = 5

# Latest grain groups that give the grains leading once the bodies end
LEAD_GROUPS = 64

# Smallest ablation coefficient (s^2/m^2), as a body that does not ablate is the limit of a small one
SIGMA_MIN = 1e-16

# Gamma function of 5/3, for MetSim's gamma distribution of the grain diameters
GAMMA_5_3 = 0.9027452929509337

# Grain epochs per e-fold of the column, nodes of the main body, and the gravity restarts at a length gained
#   (m) or a relative change of the mass or velocity
ACCURACY = {'fast': {'epochs_per_fold': 10, 'n_main': 300, 'dl_restart': 50.0, 'grav_rel': 0.03}, \
    'precise': {'epochs_per_fold': 15, 'n_main': 600, 'dl_restart': 20.0, 'grav_rel': 0.01}}



### COMPILED KERNELS ###

@njit(cache=True)
def nbLumEff(lum_eff_type, lum_eff, vel, lv, lm):
    """ Compute the luminous efficiency as MetSimErosionCyTools.luminousEfficiency does, without v_init.

    Arguments:
        lum_eff_type: [int] Model of the luminous efficiency.
        lum_eff: [float] Constant luminous efficiency (percent), for type 0.
        vel: [float] Velocity (m/s).
        lv: [float] ln(vel/1000).
        lm: [float] ln(mass).

    Return:
        [float] Luminous efficiency.
    """

    # A constant efficiency
    if lum_eff_type == 0:
        return lum_eff/100.0

    # Revelle and Ceplecha (2001), types 1 to 3
    if (lum_eff_type == 1) or (lum_eff_type == 2) or (lum_eff_type == 3):
        # Coefficients of the type
        if lum_eff_type == 1:
            c1, c2 = 0.466, -1.538
        elif lum_eff_type == 2:
            c1, c2 = -0.955, -2.959
        else:
            c1, c2 = -2.670, -4.674

        # A polynomial in ln(v) below 25.4 km/s
        if vel < 25372:
            lv2 = lv*lv
            return np.exp(c1 - 10.307*lv + 9.781*lv2 - 3.0414*lv2*lv + 0.3213*lv2*lv2 \
                + 1.15*np.tanh(0.38*lm))/100.0

        # Proportional to the velocity above 25.4 km/s
        return np.exp(c2 + lv + 1.15*np.tanh(0.38*lm))/100.0

    # Borovicka et al. (2013), Kosice
    if lum_eff_type == 4:
        return np.exp(-1.45 + lv + 0.35*np.tanh(0.38*lm))/100.0

    # CAMO faint meteors, the mass in mg as ln(1e6) + ln(mass)
    if lum_eff_type == 5:
        return np.exp(-12.59 + 5.58*lv - 0.17*lv*lv*lv - 1.21*np.tanh(0.2*(lm + 13.815510557964274)))/100.0

    # Borovicka et al. (2020), two strengths
    if lum_eff_type == 7:
        # A polynomial in ln(v) below 25.4 km/s
        if vel < 25372:
            lv2 = lv*lv
            return np.exp(0.567 - 10.307*lv + 9.781*lv2 - 3.0414*lv2*lv + 0.3213*lv2*lv2 \
                + 0.3470*np.tanh(0.38*lm))/100.0

        # Proportional to the velocity above 25.4 km/s
        return np.exp(-1.4286 + lv + 0.347*np.tanh(0.38*lm))/100.0

    # Types 6 and 8 depend on the velocity only
    cf = 1500.0*10000000.0
    l10 = np.log10(vel/1000.0)

    # Ceplecha and McCrosky (1976), piecewise in log10(v)
    if lum_eff_type == 6:
        if vel <= 9300:
            return cf*10**(-12.75)
        elif vel <= 12500:
            return cf*10**(-15.60 + 2.92*l10)
        elif vel <= 17000:
            return cf*10**(-13.24 + 0.77*l10)
        elif vel <= 27000:
            return cf*10**(-12.50 + 0.17*l10)

        return cf*10**(-13.69 + 1.00*l10)

    # Pecina and Ceplecha (1983)
    if vel/1000.0 < 25.372:
        return cf*10**(-12.834 - 10.307*l10 + 22.522*l10**2 - 16.125*l10**3 + 3.922*l10**4)

    # Proportional to the velocity above 25.4 km/s
    return cf*10**(l10 - 13.70)



@njit(cache=True)
def nbDeltaEta(u, c0, beta, f_beta):
    """ Compute the column of air flown by a body from its start to u = (v/v_s)^2.

    Arguments:
        u: [float] Squared velocity relative to the start.
        c0: [float] m_s^(1/3)/(2 K) of the body.
        beta: [float] Mass loss parameter, sigma v_s^2/6.
        f_beta: [float] exp(-beta) Ei(beta).

    Return:
        [float] Column flown (kg/m^2).
    """

    return c0*(f_beta - np.exp(-beta*(1.0 - u))*expEiScalar(beta*u))



@njit(cache=True)
def nbUAtDeltaEta(d_eta, c0, beta, f_beta, s_start):
    """ Invert nbDeltaEta() with Newton steps on s = ln(beta u), from above.

    Arguments:
        d_eta: [float] Column flown (kg/m^2).
        c0: [float] m_s^(1/3)/(2 K) of the body.
        beta: [float] Mass loss parameter.
        f_beta: [float] exp(-beta) Ei(beta).
        s_start: [float] Starting s, not below the root.

    Return:
        (u, s): [tuple] u = (v/v_s)^2 at the column, and its s.
    """

    # Target of exp(-beta) Ei(beta u) at the column, for Newton's method in s = ln(beta u)
    target = f_beta - d_eta/c0

    # Start from s_start, at most at s = ln(beta) where u = 1
    s_max = np.log(beta)
    s = min(s_start, s_max)

    # Newton steps until exp(-beta) Ei(beta u) matches the target
    for _ in range(80):
        x = np.exp(s)
        g = np.exp(min(x - beta, 700.0))*expEiScalar(x) - target

        # Newton step in s, at most ln(beta)
        s_new = min(s - g*np.exp(min(beta - x, 700.0)), s_max)

        # Stop once the step is below 1e-12
        if abs(s_new - s) < 1e-12:
            s = s_new
            break

        s = s_new

    # Back to u, within [0, 1]
    u = np.exp(s)/beta
    if u > 1.0:
        u = 1.0
    if u < 0.0:
        u = 0.0

    return u, s



@njit(cache=True)
def nbUAtDeltaEtaArray(d_eta, m_s, K, beta):
    """ Invert the column flown for every element of 1D arrays, from the start of the body.

    Arguments:
        d_eta: [ndarray] Columns flown (kg/m^2).
        m_s: [ndarray] Masses at the start (kg).
        K: [ndarray] Shape-density coefficients.
        beta: [ndarray] Mass loss parameters.

    Return:
        [ndarray] u = (v/v_s)^2 at the columns.
    """

    # Every column from its own start
    out = np.empty(d_eta.shape[0])
    for i in range(d_eta.shape[0]):
        # Constant c0 of the body, and s at u = 1
        c0 = m_s[i]**(1/3.0)/(2*K[i])
        s_max = np.log(beta[i])

        # Start just above the root for a small column
        if d_eta[i]/c0 < 1e-3:
            s0 = min(s_max, np.log(beta[i]/(1 + d_eta[i]/c0)) + 1e-9)
        else:
            s0 = s_max

        # Invert the column with Newton steps from s0
        out[i], _ = nbUAtDeltaEta(d_eta[i], c0, beta[i], expEiScalar(beta[i]), s0)

    return out



@njit(cache=True)
def nbLogNodes(eta_s, m_s, K, beta, eta_targets):
    """ Find u at increasing target columns of every body, continuing Newton from the previous node.

    Arguments:
        eta_s: [ndarray] Columns at the start of the bodies (kg/m^2).
        m_s: [ndarray] Masses at the start (kg).
        K: [ndarray] Shape-density coefficients.
        beta: [ndarray] Mass loss parameters.
        eta_targets: [ndarray] Target columns, one row per body (kg/m^2).

    Return:
        [ndarray] u at the target columns.
    """

    # u at every target column, each from the last one
    n, k = eta_targets.shape
    out = np.empty((n, k))
    for r in range(n):
        # Constants of the body, and Newton from u = 1
        c0 = m_s[r]**(1/3.0)/(2*K[r])
        f_beta = expEiScalar(beta[r])
        s = np.log(beta[r])

        # Target columns of the body in increasing order
        for i in range(k):
            # A target at or before the start of the body keeps u = 1
            d = eta_targets[r, i] - eta_s[r]
            if d <= 0.0:
                out[r, i] = 1.0
                continue

            # Newton from the s of the previous node
            out[r, i], s = nbUAtDeltaEta(d, c0, beta[r], f_beta, s)

    return out



@njit(cache=True)
def nbInterpIdx(x, xp, j):
    """ Find the interval of an increasing table that holds x, searching from the index j.

    Arguments:
        x: [float] Value.
        xp: [ndarray] Increasing table.
        j: [int] Index to start from.

    Return:
        (j, w): [tuple] Index of the interval and the weight of its upper end.
    """

    # The ends of the table
    n = xp.shape[0]
    if x <= xp[0]:
        return 0, 0.0

    # Above the table, the upper end of the last interval
    if x >= xp[n - 1]:
        return n - 2, 1.0

    # Walk from the guess to the interval of x
    while (j < n - 2) and (xp[j + 1] <= x):
        j += 1

    # Walk down while the interval starts above x
    while (j > 0) and (xp[j] > x):
        j -= 1

    return j, (x - xp[j])/(xp[j + 1] - xp[j])



@njit(cache=True)
def nbEvalBodies(path_eta, path_l, path_rho, eta_s, t_s, m_s, v_s, K, sigma, sigma_eff, u, lum_eff_type, \
    lum_eff):
    """ Evaluate bodies at their nodes of u = (v/v_s)^2.

    Arguments:
        path_eta, path_l, path_rho: [ndarray] Column (kg/m^2), length (m) and air density (kg/m^3) of the
            path.
        eta_s, t_s, m_s, v_s: [ndarray] Column, time, mass and velocity at the start of the bodies.
        K, sigma, sigma_eff: [ndarray] Shape-density coefficient, ablation coefficient, and the ablation plus
            erosion coefficient of the bodies.
        u: [ndarray] Nodes, one row per body.
        lum_eff_type: [int] Model of the luminous efficiency.
        lum_eff: [float] Constant luminous efficiency (percent).

    Return:
        (eta, l, t, v, m, rho, e_rad): [tuple] Column, length, time, velocity, mass and air density at the
            nodes, and the energy radiated between nodes (J).
    """

    # Arrays at the nodes of every body
    n, k = u.shape
    eta = np.empty((n, k))
    l = np.empty((n, k))
    t = np.empty((n, k))
    v = np.empty((n, k))
    m = np.empty((n, k))
    rho = np.empty((n, k))
    e_rad = np.empty((n, k - 1))

    # Every body in turn
    for r in range(n):
        # Mass loss parameter and the constants of the closed form
        beta = sigma_eff[r]*v_s[r]**2/6.0
        c0 = m_s[r]**(1/3.0)/(2*K[r])
        f_beta = expEiScalar(beta)

        # Ablated share of the mass loss, and the kinetic energy at the start
        abl = sigma[r]/sigma_eff[r]
        e0 = m_s[r]*v_s[r]**2/2.0

        # Path index, and the loss and the work at the previous node
        j = 0
        loss_p = 0.0
        work_p = 0.0

        # Every node of the body in turn
        for i in range(k):
            # Column at the node from the closed form
            ui = u[r, i]
            eta[r, i] = eta_s[r] + nbDeltaEta(ui, c0, beta, f_beta)

            # Length and air density at the column, from the path
            j, w = nbInterpIdx(eta[r, i], path_eta, j)
            l[r, i] = path_l[j] + w*(path_l[j + 1] - path_l[j])
            rho[r, i] = path_rho[j] + w*(path_rho[j + 1] - path_rho[j])

            # Velocity and mass at the node
            v[r, i] = v_s[r]*np.sqrt(ui)
            ex = np.exp(-3*beta*(1.0 - ui))
            m[r, i] = m_s[r]*ex

            # Kinetic energy lost and the work of the drag since the start
            loss = e0*(1.0 - ui*ex)
            work = e0*(-np.expm1(-3*beta*(1.0 - ui)))/(3*beta)

            # Time and light of the interval
            if i == 0:
                # The first node at the start time of the body
                t[r, 0] = t_s[r]
            else:
                # Time from the mean of 1/v over the interval
                t[r, i] = t[r, i - 1] + (l[r, i] - l[r, i - 1])*0.5*(1.0/v[r, i] + 1.0/v[r, i - 1])

                # Velocity and mass in the middle of the interval
                v_mid = v_s[r]*np.sqrt(0.5*(ui + u[r, i - 1]))
                m_mid = np.sqrt(m[r, i]*m[r, i - 1])

                # Radiated energy from the ablated mass and the work of the drag
                tau = nbLumEff(lum_eff_type, lum_eff, v_mid, np.log(v_mid/1000.0), np.log(m_mid))
                e_rad[r, i - 1] = tau*(abl*((loss - loss_p) - (work - work_p)) + (work - work_p))

            # Carry the loss and the work to the next node
            loss_p = loss
            work_p = work

    return eta, l, t, v, m, rho, e_rad



@njit(cache=True)
def nbGravity(path_l, path_h, path_log_rho, path_dlnrho, h_init, zenith, r_earth, g0, lum_eff_type, lum_eff, \
    v_kill, t, l, v, m, rho, K, sigma, sigma_eff, e_rad):
    """ Integrate the gain of gravity along the path of every body to first order around the closed form, with
        Heun's method over the nodes, and scale the radiated energy to the moved state.

    Arguments:
        path_l, path_h, path_log_rho, path_dlnrho: [ndarray] Length (m), height (m), ln of the air density and
            its derivative along the path.
        h_init, zenith, r_earth: [float] Height of the start (m), zenith angle (rad) and Earth radius (m).
        g0: [float] Gravitational acceleration at the surface (m/s^2).
        lum_eff_type: [int] Model of the luminous efficiency.
        lum_eff: [float] Constant luminous efficiency (percent).
        v_kill: [float] Kill velocity (m/s).
        t, l, v, m, rho: [ndarray] Time, length, velocity, mass and air density at the nodes of the bodies.
        K, sigma, sigma_eff: [ndarray] Coefficients of the bodies.
        e_rad: [ndarray] Energy radiated between nodes, scaled in place (J).

    Return:
        (dl, dv, dm): [tuple] Gain of gravity in length (m), velocity (m/s) and mass (kg) at the nodes.
    """

    # Gains of gravity at the nodes of every body
    n, k = t.shape
    dl = np.zeros((n, k))
    dv = np.zeros((n, k))
    dm = np.zeros((n, k))

    # Geometry of the path for the local zenith angle
    r0 = h_init + r_earth
    r_cos = r0*np.cos(zenith)

    # Coefficients at the nodes of one body
    gc = np.empty(k)
    drag = np.empty(k)
    md = np.empty(k)
    dlr = np.empty(k)

    # Every body in turn
    for r in range(n):

        # Gravity, drag and mass loss at the nodes
        j = 0
        for i in range(k):
            # Height and d ln(rho)/dl at the node, from the path
            j, w = nbInterpIdx(l[r, i], path_l, j)
            h_c = path_h[j] + w*(path_h[j + 1] - path_h[j])
            dlr[i] = path_dlnrho[j] + w*(path_dlnrho[j + 1] - path_dlnrho[j])

            # Gravity along the path at the local zenith angle
            cos_z = (r_cos - l[r, i])/(h_c + r_earth)
            gc[i] = g0/(1 + h_c/r_earth)**2*cos_z

            # Drag deceleration and mass loss rate at the node
            drag[i] = K[r]*rho[r, i]*v[r, i]**2*m[r, i]**(-1/3.0)
            md[i] = -K[r]*sigma_eff[r]*rho[r, i]*v[r, i]**3*m[r, i]**(2/3.0)

        # Heun's method for the linearized equations of the gain
        for i in range(k - 1):
            # Time step and the gains at the start of the interval
            h = t[r, i + 1] - t[r, i]
            l0 = dl[r, i]
            v0 = dv[r, i]
            m0 = dm[r, i]

            # Rates of the velocity and mass gains at the start
            av0 = gc[i] - 2*drag[i]/v[r, i]*v0 - drag[i]*dlr[i]*l0 + drag[i]*m0/(3*m[r, i])
            am0 = 3*md[i]/v[r, i]*v0 + md[i]*dlr[i]*l0 + 2*md[i]/(3*m[r, i])*m0

            # Euler predictor at the end of the interval
            l1 = l0 + h*v0
            v1 = v0 + h*av0
            m1 = m0 + h*am0

            # Rates of the gains at the predicted end
            jn = i + 1
            av1 = gc[jn] - 2*drag[jn]/v[r, jn]*v1 - drag[jn]*dlr[jn]*l1 + drag[jn]*m1/(3*m[r, jn])
            am1 = 3*md[jn]/v[r, jn]*v1 + md[jn]*dlr[jn]*l1 + 2*md[jn]/(3*m[r, jn])*m1

            # Heun corrector with the mean of the two rates
            dl[r, jn] = l0 + 0.5*h*(v0 + v1)
            dv[r, jn] = v0 + 0.5*h*(av0 + av1)
            dm[r, jn] = m0 + 0.5*h*(am0 + am1)

        # Keep the change of the mass above -m/2
        for i in range(k):
            if dm[r, i] < -0.5*m[r, i]:
                dm[r, i] = -0.5*m[r, i]

        # Light of the moved state relative to the closed form
        jl = 0
        for i in range(k - 1):
            # Velocity, mass and length in the middle of the interval
            vm = 0.5*(v[r, i] + v[r, i + 1])
            mm = np.sqrt(m[r, i]*m[r, i + 1])
            lm = 0.5*(l[r, i] + l[r, i + 1])

            # The same moved by the gains, with floors on the velocity and mass
            vp = max(vm + 0.5*(dv[r, i] + dv[r, i + 1]), v_kill)
            mp = max(mm + 0.5*(dm[r, i] + dm[r, i + 1]), 1e-3*mm)
            lp = lm + 0.5*(dl[r, i] + dl[r, i + 1])

            # Log of the air density in the middle of the interval
            jl, w = nbInterpIdx(lm, path_l, jl)
            lr0 = path_log_rho[jl] + w*(path_log_rho[jl + 1] - path_log_rho[jl])

            # Log of the air density at the moved length
            j2, w2 = nbInterpIdx(lp, path_l, jl)
            lr1 = path_log_rho[j2] + w2*(path_log_rho[j2 + 1] - path_log_rho[j2])

            # Intensity of the closed form in the middle of the interval
            vm = max(vm, v_kill)
            i0 = nbLumEff(lum_eff_type, lum_eff, vm, np.log(vm/1000.0), np.log(mm))*vm**3*mm**(2/3.0) \
                *(sigma[r]*vm**2/2 + 1)

            # Intensity of the moved state, with the ratio of the air densities
            i1 = nbLumEff(lum_eff_type, lum_eff, vp, np.log(vp/1000.0), np.log(mp))*np.exp(lr1 - lr0) \
                *vp**3*mp**(2/3.0)*(sigma[r]*vp**2/2 + 1)

            # Scale the radiated energy by the ratio of the two
            if (i0 > 0) and np.isfinite(i1):
                e_rad[r, i] *= i1/i0

    return dl, dv, dm



@njit(cache=True)
def nbEndAtColumn(eta_stop, eta_s, m_s, K, beta, v_s, v_kill, m_kill):
    """ Find where a body ends: at the kill velocity or mass, or where it reaches a column.

    Arguments:
        eta_stop: [float] Column where the body stops (kg/m^2).
        eta_s, m_s, K, beta, v_s: [float] Column, mass, shape-density coefficient, mass loss parameter and
            velocity at the start of the body.
        v_kill, m_kill: [float] Kill velocity (m/s) and mass (kg).

    Return:
        (u_lo, u_end, reaches, by_mass): [tuple] Lowest u before the kill velocity or mass, u at the end,
            whether the body reaches eta_stop, and whether it ends by running out of mass.
    """

    # Constants of the closed form of the body
    c0 = m_s**(1/3.0)/(2*K)
    f_beta = expEiScalar(beta)

    # Values of u at the kill velocity and mass, the higher reached first
    u_v = (v_kill/v_s)**2
    u_m = 1.0 + np.log(m_kill/m_s)/(3*beta) if m_kill < m_s else 1.0
    u_lo = max(u_v, u_m)

    # Keep u_lo within [1e-12, 1]
    if u_lo < 1e-12:
        u_lo = 1e-12
    if u_lo > 1.0:
        u_lo = 1.0

    # Column flown to that u against the column to the stop
    d_lo = nbDeltaEta(u_lo, c0, beta, f_beta)
    d_stop = eta_stop - eta_s
    if d_stop < d_lo:
        # No column to fly, the body stops where it starts
        if d_stop <= 0.0:
            return u_lo, 1.0, True, False

        # The body reaches the stop first
        u, _ = nbUAtDeltaEta(d_stop, c0, beta, f_beta, np.log(beta))

        return u_lo, u, True, False

    # It ends at the kill velocity or mass first
    return u_lo, u_lo, False, u_m >= u_v



@njit(cache=True)
def nbEnvelopeSpan(ka, km, kb, t_prev, t_i, l_prev, l_i, v_prev, v_i, h_prev, h_i, lm_prev, lm_i, mult, \
    env_t0, env_dt, env_l, env_v, env_h, env_m, m_stride, env_smin, k_end):
    """ Raise the longest grain in flight to a grain linear in time over an interval, at the grid points ka
        to kb, with its velocity and its drop of gravity, and add the mass in flight of its group at every
        m_stride-th grid point from km to kb (see nbGrainPiece() for the arrays).
    """

    # Skip the grid times if the grain stays behind the bodies
    kp = ka - 1 if ka > 0 else 0
    if not ((kb <= k_end) and (l_i < env_smin[kp])):
        for kk in range(ka, kb + 1):
            # Length of the grain at the grid time, linear in time
            tk = env_t0 + kk*env_dt
            f = (tk - t_prev)/(t_i - t_prev) if t_i > t_prev else 1.0
            lk = l_prev + f*(l_i - l_prev)

            # Raise the envelope where this grain is the longest
            if lk > env_l[kk]:
                env_l[kk] = lk
                env_v[kk] = v_prev + f*(v_i - v_prev)
                env_h[kk] = h_prev + f*(h_i - h_prev)

    # The mass in flight on every m_stride-th grid point
    km = ((km + m_stride - 1)//m_stride)*m_stride
    for kk in range(km, kb + 1, m_stride):
        # Mass of the group at the grid time, log-linear in time
        tk = env_t0 + kk*env_dt
        f = (tk - t_prev)/(t_i - t_prev) if t_i > t_prev else 1.0
        env_m[kk] += mult*np.exp(lm_prev + f*(lm_i - lm_prev))



@njit(cache=True)
def nbGrainPiece(path_tab, eta_s, t_s, m_s, v_s, K, sigma, hd_s, vd_s, u_end, ratio, eta_s0, n_a, n_u, \
    lum_eff_type, lum_eff, mult, wa, wb, t_cut, out_ta, out_tb, out_e, pos, env_t0, env_dt, env_l, env_v, \
    env_h, env_m, m_stride, env_smin, k_end, g0):
    """ Evaluate one piece of a grain group from its start to u_end or to the time t_cut, writing the light of
        every interval from the position pos, and updating the longest grain in flight and the grain mass in
        flight on the envelope grid. The gain of gravity along the path moves the grain in the envelope, to
        first order around the closed form, and the grain carries its own drop of gravity across the path as
        MetSim's fragments do.

    Arguments:
        path_tab: [ndarray] Rows of the path, see nbGrainTracks().
        eta_s, t_s, m_s, v_s, K, sigma: [float] Column, time, mass, velocity, shape-density and ablation
            coefficients at the start of the piece.
        hd_s, vd_s: [float] Drop of gravity (m) and its velocity (m/s) at the start of the piece.
        u_end: [float] u = (v/v_s)^2 at the end of the grain.
        ratio, eta_s0: [float] Ratio of the columns the grain flies through, and the column of its first node.
        n_a, n_u: [int] Nodes uniform in the log of the column and uniform in u.
        lum_eff_type, lum_eff: [int, float] Luminous efficiency model and its constant value (percent).
        mult: [float] Number of grains in the group.
        wa, wb: [float] Release of the group before and after its time (s).
        t_cut: [float] Time of the next "A" event (s).
        out_ta, out_tb, out_e: [ndarray] Intervals of the light and their energies, written from pos.
        pos: [int] Position to write from.
        env_t0, env_dt: [float] Start (s) and spacing (s) of the envelope grid.
        env_l, env_v, env_h, env_m: [ndarray] Longest grain in flight (m), its velocity (m/s) and drop (m),
            and the mass in flight (kg) on the envelope grid, updated in place.
        m_stride: [int] Stride of the envelope grid where the mass in flight is summed.
        env_smin: [ndarray] Lowest leading body from a grid time on, while one flies (m).
        k_end: [int] Last grid index with a body in flight.
        g0: [float] Gravitational acceleration at the surface (m/s^2).

    Return:
        (pos, cut, eta, t, m, v, hd, vd): [tuple] Next position to write, whether the piece stopped at t_cut,
            and the column, time, mass, velocity, drop and velocity of the drop at its end, the column and
            velocity moved by gravity at a cut.
    """

    # Column and length along the path
    path_eta = path_tab[0]
    path_l = path_tab[1]

    # Mass loss parameter and the constants of the column flown
    beta = sigma*v_s**2/6.0
    c0 = m_s**(1/3.0)/(2*K)
    f_beta = expEiScalar(beta)

    # Nodes uniform in u and uniform in the log of the column
    k = n_a + n_u
    u = np.empty(k)

    # Nodes uniform in u, from 1 down to u_end
    for j in range(n_u):
        frac = j/(n_u - 1) if n_u > 1 else 0.0
        u[j] = 1.0 - (1.0 - u_end)*frac

    # Nodes at columns uniform in log, from the first node to the end
    for j in range(n_a):
        eta_j = eta_s0*ratio**(j/(n_a - 1) if n_a > 1 else 0.0)

        # Their u without the mass loss, kept between u_end and 1
        uj = np.exp(-(eta_j - eta_s)/c0)
        if uj > 1.0:
            uj = 1.0
        if uj < u_end:
            uj = u_end

        # Store it after the nodes uniform in u
        u[n_u + j] = uj

    # Sort the nodes by decreasing u
    u = -np.sort(-u)

    # Pin the first node at the start and the last at the end
    u[0] = 1.0
    u[k - 1] = u_end

    # State at the start of the piece
    e0 = m_s*v_s**2/2.0
    lm_s = np.log(m_s)
    lv_s = np.log(v_s/1000.0)

    # The last node and its path interval, at the start of the piece
    lm_prev = lm_s
    jp = 0
    t_prev = t_s
    l_prev = 0.0
    v_prev = v_s

    # Its energy lost, mass, u and column
    loss_p = 0.0
    m_prev = m_s
    u_prev = 1.0
    eta_prev = eta_s

    # Gain of gravity in length, velocity and relative mass, and the drop and its velocity
    dl_prev = 0.0
    dv_prev = 0.0
    dmu_prev = 0.0
    hd_prev = hd_s
    vd_prev = vd_s

    # Integrals of the forcing at the start, dv/v^2 and v_drop/v under the drag of the closed form
    gain_int = 0.0
    drop_int = vd_s/v_s

    # Gravity and air density terms at the last node
    gc_p = 0.0
    dr_p = 0.0
    gs_p = 0.0
    sz_p = 0.0

    # Go through the nodes, from the start of the piece to its end
    for jn in range(k):
        # Column of the node from its u
        ui = u[jn]
        eta_i = eta_s + nbDeltaEta(ui, c0, beta, f_beta)

        # Length along the path at that column
        jp, w = nbInterpIdx(eta_i, path_eta, jp)
        l_i = path_l[jp] + w*(path_l[jp + 1] - path_l[jp])

        # Velocity and log mass of the node
        v_i = v_s*np.sqrt(ui)
        lm_i = lm_s - 3*beta*(1.0 - ui)

        # Mass and kinetic energy lost at the node
        ex = np.exp(-3*beta*(1.0 - ui))
        m_i = m_s*ex
        loss = e0*(1.0 - ui*ex)

        # Gravity along and across the path, d ln(rho)/dl and the sine of the zenith angle at the node
        gc_i = 0.0
        dr_i = 0.0
        gs_i = 0.0
        sz_i = 0.0

        # Interpolate them along the path when gravity is on
        if g0 > 0:
            dr_i = path_tab[2, jp] + w*(path_tab[2, jp + 1] - path_tab[2, jp])
            gc_i = g0*(path_tab[3, jp] + w*(path_tab[3, jp + 1] - path_tab[3, jp]))
            gs_i = g0*(path_tab[4, jp] + w*(path_tab[4, jp + 1] - path_tab[4, jp]))
            sz_i = path_tab[5, jp] + w*(path_tab[5, jp + 1] - path_tab[5, jp])

        # The gains and the drop at the node, carried over from the last one
        dl_i = dl_prev
        dv_i = dv_prev
        dmu_i = dmu_prev
        hd_i = hd_prev
        vd_i = vd_prev

        # Time of the node, from the length and the mean of 1/v
        if jn == 0:
            t_i = t_s
        else:
            t_i = t_prev + (l_i - l_prev)*0.5*(1.0/v_i + 1.0/v_prev)

            # Gain of gravity and drop over the interval, with the mean drag of the interval and the
            #   couplings with the length and the mass taken at its start
            if (g0 > 0) and (t_i > t_prev):
                # Duration and mean drag of the interval
                h_t = t_i - t_prev
                drag_m = (v_prev - v_i)/h_t

                # Velocity of the drop, from the gravity across the path
                drop_int += 0.5*h_t*(gs_p/v_prev + gs_i/v_i)
                vd_i = v_i*drop_int

                # Drop at the node, with the sine of the zenith angle
                hd_i = hd_prev + 0.5*h_t*(vd_prev*sz_p + vd_i*sz_i)

                # Forcing along the path, with the drag changed by the mass and the air
                couple = drag_m*(dmu_prev/3.0 - 0.5*(dr_p + dr_i)*dl_prev)
                gain_int += 0.5*h_t*((gc_p + couple)/(v_prev*v_prev) + (gc_i + couple)/(v_i*v_i))

                # Gain in velocity and length at the node
                dv_i = v_i*v_i*gain_int
                dl_i = dl_prev + 0.5*h_t*(dv_prev + dv_i)

                # The relative mass, as the mass loss grows with the velocity and the air
                dmu_i = dmu_prev + (lm_i - lm_prev)*(1.5*(dv_prev + dv_i)/(v_prev + v_i) + 0.25*(dr_p \
                    + dr_i)*(dl_prev + dl_i) - dmu_prev/3.0)

                # Keep the relative mass at -0.5 or above
                if dmu_i < -0.5:
                    dmu_i = -0.5

            # End the piece at t_cut, at the length linear in time as the envelope takes it
            if t_i > t_cut:
                # Length at the cut, linear in time
                f = (t_cut - t_prev)/(t_i - t_prev) if t_i > t_prev else 1.0
                l_c = l_prev + f*(l_i - l_prev)

                # Column of the path at that length
                jc, wc = nbInterpIdx(l_c, path_l, jp)
                eta_c = path_eta[jc] + wc*(path_eta[jc + 1] - path_eta[jc])

                # Invert the closed form for u at the cut
                uc, _ = nbUAtDeltaEta(eta_c - eta_s, c0, beta, f_beta, np.log(beta))

                # Keep it within the interval
                if uc > u_prev:
                    uc = u_prev
                if uc < ui:
                    uc = ui

                # Energy lost at the cut
                exc = np.exp(-3*beta*(1.0 - uc))
                loss_c = e0*(1.0 - uc*exc)

                # Velocity at the mean u from the last node to the cut
                v_mid = v_s*np.sqrt(0.5*(uc + u_prev))

                # Mass at the cut
                m_c = m_s*exc
                lm_c = lm_s - 3*beta*(1.0 - uc)

                # Luminous efficiency at the mean velocity and log mass
                tau = nbLumEff(lum_eff_type, lum_eff, v_mid, lv_s + 0.5*np.log(0.5*(uc + u_prev)), \
                    0.5*(lm_c + lm_prev))

                # Light from the last node to the cut, widened by the release
                out_ta[pos] = t_prev - wa
                out_tb[pos] = t_cut + wb
                out_e[pos] = mult*tau*(loss_c - loss_p)
                pos += 1

                # The envelope and the mass in flight up to the cut
                n_env = env_l.shape[0]
                if n_env > 0:
                    # Grid points after the last node and up to the cut
                    km = max(int(np.floor((t_prev - env_t0)/env_dt)) + 1, 0)
                    kb = min(int(np.floor((t_cut - env_t0)/env_dt)), n_env - 1)

                    # Update the envelope at those grid points
                    nbEnvelopeSpan(km, km, kb, t_prev, t_i, l_prev + dl_prev, l_i + dl_i, v_prev + dv_prev, \
                        v_i + dv_i, hd_prev, hd_i, lm_prev, lm_i, mult, env_t0, env_dt, env_l, env_v, env_h, \
                        env_m, m_stride, env_smin, k_end)

                # The next piece starts from the state moved by gravity
                dl_c = dl_prev + f*(dl_i - dl_prev)
                dv_c = dv_prev + f*(dv_i - dv_prev)

                # Column of the cut moved by the gain in length
                if dl_c != 0.0:
                    jc, wc = nbInterpIdx(l_c + dl_c, path_l, jc)
                    eta_c = path_eta[jc] + wc*(path_eta[jc + 1] - path_eta[jc])

                return pos, True, eta_c, t_cut, m_c, v_s*np.sqrt(uc) + dv_c, hd_prev + f*(hd_i - hd_prev), \
                    vd_prev + f*(vd_i - vd_prev)

            # Luminous efficiency at the mean velocity and log mass
            v_mid = v_s*np.sqrt(0.5*(ui + u_prev))
            tau = nbLumEff(lum_eff_type, lum_eff, v_mid, lv_s + 0.5*np.log(0.5*(ui + u_prev)), \
                0.5*(lm_i + lm_prev))

            # Write the interval and its light energy
            out_ta[pos] = t_prev - wa
            out_tb[pos] = t_i + wb
            out_e[pos] = mult*tau*(loss - loss_p)
            pos += 1

            # Longest grain and mass in flight at the grid times of the interval
            n_env = env_l.shape[0]
            if (n_env > 0) and (t_i > env_t0):
                # Grid points from the last node to this one
                ka = max(int(np.ceil((t_prev - env_t0)/env_dt)), 0)
                km = max(int(np.floor((t_prev - env_t0)/env_dt)) + 1, 0)
                kb = min(int(np.floor((t_i - env_t0)/env_dt)), n_env - 1)

                # Update the envelope at those grid points
                nbEnvelopeSpan(ka, km, kb, t_prev, t_i, l_prev + dl_prev, l_i + dl_i, v_prev + dv_prev, \
                    v_i + dv_i, hd_prev, hd_i, lm_prev, lm_i, mult, env_t0, env_dt, env_l, env_v, env_h, \
                    env_m, m_stride, env_smin, k_end)

        # Carry the node to the next interval
        t_prev = t_i
        l_prev = l_i
        v_prev = v_i

        # Its energy lost, mass, u and column
        loss_p = loss
        m_prev = m_i
        lm_prev = lm_i
        u_prev = ui
        eta_prev = eta_i

        # Carry the gain of gravity and the drop
        dl_prev = dl_i
        dv_prev = dv_i
        dmu_prev = dmu_i
        hd_prev = hd_i
        vd_prev = vd_i

        # Carry the gravity and air density terms
        gc_p = gc_i
        dr_p = dr_i
        gs_p = gs_i
        sz_p = sz_i

    return pos, False, eta_prev, t_prev, m_prev, v_prev, hd_prev, vd_prev



@njit(cache=True)
def nbGrainEnd(path_eta, eta_kill, eta_s, m_s, v_s, K, sigma, v_kill, m_kill):
    """ Find the end of a grain, at the kill velocity, mass or column.

    Arguments:
        path_eta: [ndarray] Column of the path (kg/m^2).
        eta_kill: [float] Column at the kill height (kg/m^2).
        eta_s, m_s, v_s, K, sigma: [float] Column, mass, velocity, shape-density and ablation coefficients at
            the start of the grain.
        v_kill, m_kill: [float] Kill velocity (m/s) and mass (kg).

    Return:
        (u_end, ratio, eta_s0): [tuple] u at the end, the ratio of the columns the grain flies through, and
            the column of its first node.
    """

    # Constants of the closed form of the body
    beta = sigma*v_s**2/6.0
    c0 = m_s**(1/3.0)/(2*K)
    f_beta = expEiScalar(beta)

    # Lowest u from the kill velocity
    u_lo = (v_kill/v_s)**2

    # Raised to u at the kill mass when it is below the start mass
    if m_kill < m_s:
        u_m = 1.0 + np.log(m_kill/m_s)/(3*beta)
        if u_m > u_lo:
            u_lo = u_m

    # Keep the lowest u between 1e-12 and 1
    if u_lo < 1e-12:
        u_lo = 1e-12
    if u_lo > 1.0:
        u_lo = 1.0

    # The kill column first, or the kill velocity or mass
    d_lo = nbDeltaEta(u_lo, c0, beta, f_beta)
    d_stop = eta_kill - eta_s

    # End at the kill column if the grain reaches it first
    if d_stop < d_lo:
        # A grain starting at or past the kill column ends at once
        if d_stop <= 0.0:
            u_end = 1.0
        else:
            # Invert the closed form at the kill column
            u_end, _ = nbUAtDeltaEta(d_stop, c0, beta, f_beta, np.log(beta))
    else:
        # Otherwise at the kill velocity or mass
        u_end = u_lo

    # First log node, not before the second point of the path
    eta_s0 = max(eta_s, path_eta[1])

    # Column at the end of the grain
    eta_e = eta_s + nbDeltaEta(u_end, c0, beta, f_beta)

    return u_end, max(eta_e/eta_s0, 1.0), eta_s0



@njit(cache=True)
def nbGrainTracks(path_tab, eta_kill, v_kill, m_kill, lum_eff_type, lum_eff, g_m, g_n, g_eta, g_t, g_v, \
    g_sigma, g_K, g_wa, g_wb, g_hd, g_vd, g_a0, n_base, a_t, a_sigma, a_kratio, env_t0, env_dt, env_l0, \
    env_v0, env_smin, k_end, m_stride, g0):
    """ Evaluate every grain group through the "A" fragmentation entries after its release, which set sigma
        (if not NaN) and multiply K for every grain in flight. A grain is cut at the entry and goes on in a
        new closed form from its state.

    Arguments:
        path_tab: [ndarray] Rows of the path: column (kg/m^2), length (m), d ln(rho)/dl of the air density
            (1/m), the gravity along and across the path per unit of surface gravity, and the sine of the
            zenith angle.
        eta_kill, v_kill, m_kill: [float] Column at the kill height (kg/m^2), kill velocity (m/s) and mass
            (kg).
        lum_eff_type, lum_eff: [int, float] Luminous efficiency model and its constant value (percent).
        g_m, g_n, g_eta, g_t, g_v, g_sigma, g_K, g_wa, g_wb: [ndarray] Mass, number, column, time, velocity,
            ablation and shape-density coefficients of the groups, and their release before and after their
            time.
        g_hd, g_vd: [ndarray] Drop of gravity (m) and its velocity (m/s) of the groups at their release.
        g_a0: [ndarray] Index of the first "A" entry of every group.
        n_base: [int] Least number of nodes of a grain.
        a_t, a_sigma, a_kratio: [ndarray] Times of the "A" entries, their sigma, and their ratio of K.
        env_t0, env_dt: [float] Start (s) and spacing (s) of the envelope grid.
        env_l0, env_v0: [ndarray] Leading body (m) and its velocity (m/s) on the envelope grid.
        env_smin: [ndarray] Lowest leading body from a grid time on, while one flies (m).
        k_end: [int] Last grid index with a body in flight.
        m_stride: [int] Stride of the envelope grid where the mass in flight is summed.
        g0: [float] Gravitational acceleration at the surface (m/s^2).

    Return:
        (t_a, t_b, e, env_l, env_v, env_h, env_m, row_end): [tuple] Intervals of the light and their
            energies, the longest fragment in flight, its velocity and drop, and the grain mass in flight on
            the envelope grid, and the position after the intervals of every group.
    """

    # Sizes, and the outputs grown when needed
    path_eta = path_tab[0]
    ng = g_m.shape[0]
    n_half = n_base//2
    n_ev = a_t.shape[0]
    cap = ng*(1 + n_ev)*n_base + 1024
    out_ta = np.empty(cap)
    out_tb = np.empty(cap)
    out_e = np.empty(cap)

    # Envelope grid, starting from the leading body
    n_env = env_l0.shape[0]
    env_l = env_l0.copy()
    env_v = env_v0.copy()
    env_h = np.zeros(n_env)
    env_m = np.zeros(n_env)

    # End position of every group, and the write position
    row_end = np.empty(ng, np.int64)
    pos = 0

    # Go through the grain groups
    for i in range(ng):
        # State of the group at its release
        eta_s, t_s, m_s, v_s, K, sigma = g_eta[i], g_t[i], g_m[i], g_v[i], g_K[i], g_sigma[i]
        hd_s, vd_s = g_hd[i], g_vd[i]

        # Index of its first "A" entry
        ev = g_a0[i]

        # Every piece of the group, between its "A" entries
        while True:
            # End of the grains, and the ratio of the columns flown
            u_end, ratio, eta_s0 = nbGrainEnd(path_eta, eta_kill, eta_s, m_s, v_s, K, sigma, v_kill, m_kill)

            # Nodes in the log of the column, as many as needed
            need = GRAIN_NODES_PER_FOLD*np.log(ratio)
            if need <= n_half:
                n_a = n_half
            else:
                n_a = min(240, int(np.ceil(need/GRAIN_NODE_STEP))*GRAIN_NODE_STEP)

            # Nodes uniform in u, and the time of the next "A" entry
            n_u = max(n_base - n_half, n_a//2)
            t_cut = a_t[ev] if ev < n_ev else 1e300

            # Grow the outputs when a piece could pass their end
            if pos + n_a + n_u > out_e.shape[0]:
                # Arrays twice as large, with room for the piece
                new_cap = 2*out_e.shape[0] + n_a + n_u
                ta2 = np.empty(new_cap)
                tb2 = np.empty(new_cap)
                e2 = np.empty(new_cap)

                # Copy the intervals written so far into them
                ta2[:pos] = out_ta[:pos]
                tb2[:pos] = out_tb[:pos]
                e2[:pos] = out_e[:pos]
                out_ta, out_tb, out_e = ta2, tb2, e2

            # Evaluate the piece up to the next entry
            pos, cut, eta_c, t_c, m_c, v_c, hd_c, vd_c = nbGrainPiece(path_tab, eta_s, t_s, m_s, v_s, K, \
                sigma, hd_s, vd_s, u_end, ratio, eta_s0, n_a, n_u, lum_eff_type, lum_eff, g_n[i], g_wa[i], \
                g_wb[i], t_cut, out_ta, out_tb, out_e, pos, env_t0, env_dt, env_l, env_v, env_h, env_m, \
                m_stride, env_smin, k_end, g0)

            # Stop when the grains end before the next entry
            if not cut:
                break

            # Apply the "A" entry from the state at the cut
            if not np.isnan(a_sigma[ev]):
                sigma = a_sigma[ev]

            # Scale K by the ratio of the entry
            K = K*a_kratio[ev]

            # Next piece from the state at the cut, up to the next entry
            eta_s, t_s, m_s, v_s, hd_s, vd_s = eta_c, t_c, m_c, v_c, hd_c, vd_c
            ev += 1

            # Stop when the grains reach the kill velocity, mass or column
            if (v_s <= v_kill) or (m_s <= m_kill) or (eta_s >= eta_kill):
                break

        # End of the intervals of the group
        row_end[i] = pos

    return out_ta[:pos], out_tb[:pos], out_e[:pos], env_l, env_v, env_h, env_m, row_end



@njit(cache=True)
def nbGridIndex(t, t0, dt, n):
    """ Index of the first point of a uniform grid at or after the time t, clipped to [0, n]. """

    # Grid index at or after the time t
    i = int(np.ceil((t - t0)/dt))

    # Clip to index 0 before the grid
    if i < 0:
        return 0

    # Clip to index n past the grid
    if i > n:
        return n

    return i



@njit(cache=True)
def nbDepositLight(t_a, t_b, energy, spread, q, t0, dt, n):
    """ Sum the light of intervals exactly into the bins of a uniform time grid. An interval of a grain group
        is spread by the release of the group over an epoch, in the two halves of the epoch with (1 - q)/2 and
        (1 + q)/2 of the grains, each the convolution of the interval and the half epoch, a trapezoid. An
        interval of a body has no spread and q = 1.

    Arguments:
        t_a, t_b: [ndarray] Start and end of the intervals, the grains' widened by the release (s).
        energy: [ndarray] Energy of every interval (J).
        spread: [ndarray] Length of the epoch of every interval (s).
        q: [ndarray] Excess of grains released in the second half of the epoch.
        t0: [float] Time of the first grid point (s).
        dt: [float] Spacing of the grid (s).
        n: [int] Number of grid points.

    Return:
        [ndarray] Energy in each of the n - 1 bins (J).
    """

    # Steps of the linear and quadratic ramps of the cumulative energy
    a1 = np.zeros(n + 1)
    b1 = np.zeros(n + 1)
    a2 = np.zeros(n + 1)
    b2 = np.zeros(n + 1)
    c2 = np.zeros(n + 1)

    # Go through the intervals
    for k in range(t_a.shape[0]):
        e = energy[k]

        # Skip the intervals without energy
        if e == 0.0:
            continue

        # Start and width of the widened interval
        ta = t_a[k]
        width = t_b[k] - ta

        # Epoch, kept between zero and the widened interval
        sp = spread[k]
        if sp < 0.0:
            sp = 0.0
        if sp > width:
            sp = width

        # Interval without the release, and the half epoch
        sn = width - sp
        h = 0.5*sp

        # Energy of the grains released in each half
        e1 = 0.5*e*(1.0 - q[k])
        e2 = 0.5*e*(1.0 + q[k])

        # Two boxes when a ramp is under a thousandth of a bin
        if min(h, sn) < 1e-3*dt:
            # Each half as one box over the interval and the half epoch
            w = max(sn + h, 1e-12)

            # Linear ramps of the cumulative energy at the box edges
            for tt, s in ((ta, e1/w), (ta + w, -e1/w), (ta + h, e2/w), (ta + h + w, -e2/w)):
                # Add the ramp at the first grid point at or after the edge
                i = nbGridIndex(tt, t0, dt, n)
                a1[i] += s
                b1[i] += s*tt

        # Otherwise two trapezoids, as quadratic ramps with shared knots merged
        else:
            # Coefficients of the quadratic ramps of each half
            s1 = 0.5*e1/(h*sn)
            s2 = 0.5*e2/(h*sn)

            # Quadratic ramps of the cumulative energy at the knots
            for tt, s in ((ta, s1), (ta + sn, -s1), (ta + h, s2 - s1), (ta + sn + h, s1 - s2), (ta + sp, \
                -s2), (ta + sn + sp, s2)):
                # Add the ramp at the first grid point at or after the knot
                i = nbGridIndex(tt, t0, dt, n)
                a2[i] += s
                b2[i] += s*tt
                c2[i] += s*tt*tt

    # Differences of the cumulative energy at the grid points
    out = np.empty(n - 1)

    # Running sums of the steps of the ramps
    sa1 = 0.0
    sb1 = 0.0
    sa2 = 0.0
    sb2 = 0.0
    sc2 = 0.0

    # Cumulative energy at the last grid point
    prev = 0.0

    # Go through the grid points
    for j in range(n):
        # Steps of the linear ramps up to this point
        sa1 += a1[j]
        sb1 += b1[j]

        # Steps of the quadratic ramps up to this point
        sa2 += a2[j]
        sb2 += b2[j]
        sc2 += c2[j]

        # Cumulative energy of the ramps at the grid time
        T = t0 + j*dt
        cum = T*sa1 - sb1 + T*T*sa2 - 2.0*T*sb2 + sc2

        # Energy of the bin that ends at this point
        if j > 0:
            out[j - 1] = cum - prev
        prev = cum

    return out



### PATH AND BODIES ###

def airDensity(const, ht):
    """ Compute MetSim's air density at the given heights, vectorized: atmDensityTable() of the table when the
        constants use it, otherwise atmDensityPoly() of the polynomial.

    Arguments:
        const: [Constants object] Simulation constants.
        ht: [ndarray] Heights (m).

    Return:
        [ndarray] Air density (kg/m^3).
    """

    # MetSim's table, linear in the log of the density
    if usesAtmDensityTable(const):
        # Heights of the table and the log10 of its densities
        table_ht, table_lr = atmDensityTableArrays(const)

        # Linear in log10 of the density, extrapolating the end segments
        idx = np.clip(np.searchsorted(table_ht, ht, side='right') - 1, 0, len(table_ht) - 2)
        h0, h1 = table_ht[idx], table_ht[idx + 1]

        return 10**(table_lr[idx] + (ht - h0)*(table_lr[idx + 1] - table_lr[idx])/(h1 - h0))

    # MetSim's polynomial of the log of the density
    d = np.asarray(const.dens_co, dtype=np.float64)

    # Heights in units of 1000 km
    x = ht/1e6

    return 10**(d[0] + d[1]*x + d[2]*x**2 + d[3]*x**3 + d[4]*x**4 + d[5]*x**5 + d[6]*x**6)



class PathTable(object):
    def __init__(self, const, h_end=None, dl=50.0, drop_l=None, drop_h=None):
        """ MetSim's straight path from h_init in curved-Earth geometry (heightCurvature), lowered by the
            drop of gravity if given, with the air density and the slant column along it. The column of every
            interval is integrated as an exponential.

        Arguments:
            const: [Constants object] Simulation constants.

        Keyword arguments:
            h_end: [float] Height where the path ends, before a margin of 2 km (m). 2 km below h_kill by
                default.
            dl: [float] Spacing of the path (m). 50 m by default.
            drop_l: [ndarray] Lengths of the drop of gravity (m). None by default.
            drop_h: [ndarray] Drop of gravity at drop_l (m). None by default.
        """

        # Straight path from h_init down to h_end
        self.const = const
        r_earth = const.r_earth
        r0 = const.h_init + r_earth
        cz = np.cos(const.zenith_angle)

        # End of the path 2 km below the kill height by default
        if h_end is None:
            h_end = max(const.h_kill - 2000.0, 0.0)

        # Length of the path down to h_end
        disc = (cz*r0)**2 - (r0**2 - (h_end + r_earth)**2)

        # A path that climbs back up before the kill height is not supported
        disc_kill = (cz*r0)**2 - (r0**2 - (const.h_kill + r_earth)**2)
        if (cz <= 0) or (disc_kill <= 0):
            raise NotImplementedError("Paths that climb back up above the kill height (grazing meteors) " \
                "are not supported, use MetSimErosion")

        # Down to h_end, or to the lowest point of the path above it
        l_end = cz*r0 - np.sqrt(disc) if disc > 0 else cz*r0

        # Lengths every dl to 2 km past the end, and their heights
        self.l = np.arange(0.0, l_end + 2000.0 + dl, dl)
        self.h = np.sqrt(r0**2 - 2*self.l*cz*r0 + self.l**2) - r_earth

        # Arguments to rebuild the path
        self.args = {'h_end': h_end, 'dl': dl}

        # Cosine and sine of the local zenith angle along the path
        cos_z = (r0*cz - self.l)/(self.h + r_earth)
        self.sin_z = np.sqrt(np.clip(1 - cos_z**2, 0.0, None))

        # Components scaled by the inverse square of the geocentric distance
        self.g_cos = cos_z/(1 + self.h/r_earth)**2
        self.g_sin = self.sin_z/(1 + self.h/r_earth)**2

        # Drop of gravity along the path, zero without one
        self.h_drop = np.zeros_like(self.l) if drop_l is None else np.interp(self.l, drop_l, drop_h)

        # Air density and column along the whole path
        self.rho = np.zeros_like(self.l)
        self.eta = np.zeros_like(self.l)
        self.setAir(const, 0)



    def setAir(self, const, i0):
        """ Compute the reported heights, the air density and the slant column from the point i0 of the path
            on, the column of every interval exact for an exponential density between the points.
        """

        # Heights MetSim reports, lowered by the drop of gravity
        self.h_rep = self.h - self.h_drop

        # Air density at the reported heights
        self.rho[i0:] = airDensity(const, self.h_rep[i0:])

        # Column of every interval, exact for an exponential density
        log_ratio = np.log(self.rho[i0:-1]/self.rho[i0 + 1:])
        flat = np.abs(log_ratio) < 1e-12
        steps = np.diff(self.l[i0:])*np.where(flat, self.rho[i0:-1], \
            (self.rho[i0:-1] - self.rho[i0 + 1:])/np.where(flat, 1.0, log_ratio))

        # Column summed on from the point i0
        self.eta[i0 + 1:] = self.eta[i0] + np.cumsum(steps)

        # Log of the density and its slope along the path
        self.log_rho = np.log(self.rho)
        self.dlnrho_dl = np.gradient(self.log_rho, self.l)

        # Rows of the path for the grain kernels
        self.grain_tab = np.ascontiguousarray(np.vstack([self.eta, self.l, self.dlnrho_dl, self.g_cos, \
            self.g_sin, self.sin_z]))



    def lowered(self, const, drop_l, drop_h, l_from):
        """ Give a copy of the path with its drop of gravity replaced from the length l_from on, where the air
            is computed again.

        Arguments:
            const: [Constants object] Simulation constants.
            drop_l: [ndarray] Lengths of the drop of gravity (m).
            drop_h: [ndarray] Drop of gravity at drop_l (m).
            l_from: [float] Length from which the drop changes (m).

        Return:
            [PathTable object] The lowered path.
        """

        # Last point of the path before l_from, where the air is computed again
        i0 = max(int(np.searchsorted(self.l, l_from)) - 1, 0)

        # Shallow copy with the new drop from i0 on
        new = copy.copy(self)
        new.h_drop = self.h_drop.copy()
        new.h_drop[i0:] = np.interp(self.l[i0:], drop_l, drop_h)

        # Own density and column, computed again from i0
        new.rho = self.rho.copy()
        new.eta = self.eta.copy()
        new.setAir(const, i0)

        return new



    def etaAtHeight(self, ht):
        """ Column at a reported height (with the drop), as MetSim compares its heights. """

        return np.interp(ht, self.h_rep[::-1], self.eta[::-1])



# Paths without the drop of gravity, by geometry, atmosphere and kill height
PATH_CACHE = {}



def cachedPath(const):
    """ Make the path without the drop once for the geometry, the atmosphere and the kill height that a fit
        keeps.

    Arguments:
        const: [Constants object] Simulation constants.

    Return:
        [PathTable object] Path.
    """

    # Key of the geometry, the atmosphere and the kill height
    table = getattr(const, 'atm_table_log10_rho', None)
    key = (const.h_init, const.zenith_angle, const.r_earth, const.h_kill, getattr(const, \
        'atm_density_model', None), None if table is None else hash((np.asarray(table, \
        dtype=np.float64).tobytes(), np.asarray(getattr(const, 'atm_table_ht', ()), \
        dtype=np.float64).tobytes())), tuple(np.asarray(const.dens_co, dtype=np.float64).tolist()))

    # Build the path once, and empty the cache past 16 paths
    path = PATH_CACHE.get(key)
    if path is None:
        # Empty the cache when it is full
        if len(PATH_CACHE) > 16:
            PATH_CACHE.clear()

        # Build the path and keep it under its key
        path = PathTable(const)
        PATH_CACHE[key] = path

    # Give the path the constants of this run
    path.const = const

    return path



def rhoGrain(const):
    """ Density of the grains, raised to the density of the meteoroid as MetSim's runSimulation does. """

    return max(const.rho_grain, const.rho)



def sigmaOf(sigma):
    """ Take an ablation coefficient of at least SIGMA_MIN (s^2/m^2). """

    return max(float(sigma), SIGMA_MIN)



def killColumn(path, const):
    """ Column where bodies stop, at the kill height or at MetSim's kill length when it is set (kg/m^2). """

    # Column at the kill height
    eta_kill = path.etaAtHeight(const.h_kill)

    # Or at the kill length if it is set and comes first
    if getattr(const, 'len_kill', -1) > 0:
        eta_kill = min(eta_kill, float(np.interp(const.len_kill, path.l, path.eta)))

    return eta_kill



def deltaEta(u, m_s, K, beta):
    """ Compute the column of air flown by bodies from their start to u = (v/v_s)^2 (kg/m^2). """

    return m_s**(1/3.0)/(2*K)*(expEi(beta) - np.exp(-beta*(1 - u))*expEi(beta*u))



def uAtDeltaEta(d_eta, m_s, K, beta):
    """ Compute u = (v/v_s)^2 at the column flown by bodies, element-wise. The arrays broadcast. """

    # One body at many columns
    if (np.ndim(m_s) == 0) and (np.ndim(K) == 0) and (np.ndim(beta) == 0) and (np.ndim(d_eta) == 1):
        # Scalars repeated for every column
        n = len(d_eta)
        return nbUAtDeltaEtaArray(np.ascontiguousarray(d_eta, dtype=np.float64), np.full(n, float(m_s)), \
            np.full(n, float(K)), np.full(n, float(beta)))

    # Other shapes, broadcast to one another
    d_eta, m_s, K, beta = (np.array(a, dtype=np.float64) for a in np.broadcast_arrays( \
        *(np.asarray(a, dtype=np.float64) for a in (d_eta, m_s, K, beta))))

    # Kernel on the flat arrays, back in their shape
    shape = d_eta.shape

    return nbUAtDeltaEtaArray(d_eta.ravel(), m_s.ravel(), K.ravel(), beta.ravel()).reshape(shape)



class Bodies(object):
    def __init__(self, path, eta_s, t_s, m_s, v_s, K, sigma, erosion, u_end, n_log=0, n_u=40, eta_floor=None):
        """ Bodies with the same kind of start, evaluated on nodes of u = (v/v_s)^2, one row per body.

        Arguments:
            path: [PathTable object] Path.
            eta_s, t_s, m_s, v_s: [ndarray] Column (kg/m^2), time (s), mass (kg) and velocity (m/s) at the
                start.
            K, sigma, erosion: [ndarray] Shape-density, ablation and erosion coefficients.
            u_end: [ndarray] u at the end of every body.

        Keyword arguments:
            n_log: [int] Nodes uniform in the log of the column. 0 by default.
            n_u: [int] Nodes uniform in u. 40 by default.
            eta_floor: [float] Lowest column of the nodes in the log of the column (kg/m^2). The second point
                of the path by default.
        """

        # Rows of the bodies, the scalars spread to all of them
        arrays = [np.atleast_1d(np.asarray(a, dtype=np.float64)) for a in (eta_s, t_s, m_s, v_s, K, sigma, \
            erosion, u_end)]
        n = max(len(a) for a in arrays)
        (self.eta_s, self.t_s, self.m_s, self.v_s, self.K, self.sigma, self.erosion, u_end) = \
            ((a if len(a) == n else np.full(n, a[0])).reshape(n, 1) for a in arrays)

        # Ablation plus erosion, and the mass loss parameter
        self.sigma_eff = self.sigma + self.erosion
        self.beta = self.sigma_eff*self.v_s**2/6.0

        # Nodes uniform in u, from 1 down to u_end
        frac = np.linspace(0.0, 1.0, n_u)[None, :]
        parts = [1 - (1 - u_end)*frac]

        # Add nodes uniform in the log of the column when asked
        if n_log:
            # Range of the column, from the start or the floor to the end
            eta_end = self.eta_s + deltaEta(u_end, self.m_s, self.K, self.beta)
            eta_lo = np.maximum(self.eta_s, path.eta[1] if eta_floor is None else eta_floor)
            eta_end = np.maximum(eta_end, eta_lo)

            # Target columns uniform in the log
            log_eta = np.log(eta_lo) + (np.log(eta_end) - np.log(eta_lo))*np.linspace(0.0, 1.0, \
                n_log)[None, :]

            # u of every body at its target columns
            parts.append(nbLogNodes(self.eta_s[:, 0], self.m_s[:, 0], self.K[:, 0], self.beta[:, 0], \
                np.ascontiguousarray(np.exp(log_eta))))

        # Nodes in order, between u_end and 1
        u = np.concatenate(parts, axis=1)
        u = np.clip(u, u_end, 1.0)
        u = -np.sort(-u, axis=1)

        # First node at the start and last at the end
        u[:, 0] = 1.0
        u[:, -1] = u_end[:, 0]

        # Closed form at the nodes
        self.u = u
        self.evaluate(path)



    def evaluate(self, path):
        """ Evaluate the bodies at their nodes. """

        # Constants of the run, from the path
        const = path.const

        # State at the nodes and the energy radiated between them
        self.eta, self.l, self.t, self.v, self.m, self.rho, self.e_rad = nbEvalBodies(path.eta, path.l, \
            path.rho, self.eta_s[:, 0], self.t_s[:, 0], self.m_s[:, 0], self.v_s[:, 0], self.K[:, 0], \
            self.sigma[:, 0], self.sigma_eff[:, 0], np.ascontiguousarray(self.u), int(const.lum_eff_type), \
            float(const.lum_eff))



    def truncate(self, path, rows, u_cuts):
        """ End the bodies of the given rows at their u_cuts, moving the nodes below them there, and evaluate
            the bodies once.
        """

        # Nodes past the cut moved to it
        for row, u_cut in zip(rows, u_cuts):
            self.u[row] = np.maximum(self.u[row], u_cut)

            # Last node exactly at the cut
            self.u[row, -1] = u_cut

        # Closed form at the new nodes
        self.evaluate(path)



def uLimit(const, v_s, m_s, beta):
    """ Compute the lowest u = (v/v_s)^2 bodies reach before the kill velocity or the kill mass. """

    # u at the kill velocity and at the kill mass
    u_v = (const.v_kill/v_s)**2

    # u at the kill mass, 1 for bodies not heavier than it
    with np.errstate(divide='ignore'):
        u_m = np.where(const.m_kill < m_s, 1 + np.log(const.m_kill/m_s)/(3*beta), 1.0)

    return np.clip(np.maximum(u_v, u_m), 1e-12, 1.0)



def uEndAtColumn(eta_stop, eta_s, m_s, K, beta, u_lo):
    """ Compute u at which bodies reach the column eta_stop, or u_lo if they end before.

    Return:
        (u_end, reaches): [tuple] u at the end, and whether every body reaches eta_stop.
    """

    # Every body against its stop column
    eta_s, m_s, K, beta, u_lo = (np.atleast_1d(np.array(a, dtype=np.float64)) for a in \
        np.broadcast_arrays(eta_s, m_s, K, beta, u_lo))

    # Columns flown to u_lo and to the stop
    d_lo = deltaEta(u_lo, m_s, K, beta)
    d_stop = eta_stop - eta_s

    # Bodies that reach the stop before u_lo
    reaches = d_stop < d_lo

    # u at the stop where reached, u_lo elsewhere
    u_out = u_lo.copy()
    if np.any(reaches):
        u_out[reaches] = uAtDeltaEta(np.maximum(d_stop[reaches], 0.0), m_s[reaches], K[reaches], \
            beta[reaches])

    return u_out, reaches



def endsByMass(const, v_s, m_s, beta, reaches):
    """ Check whether bodies end by running out of mass, not at the kill velocity or height. """

    # u at the kill velocity and at the kill mass
    u_v = (const.v_kill/v_s)**2

    # u at the kill mass, 1 for bodies not heavier than it
    with np.errstate(divide='ignore'):
        u_m = np.where(const.m_kill < m_s, 1 + np.log(const.m_kill/m_s)/(3*beta), 1.0)

    return (~np.asarray(reaches)) & (np.asarray(u_m) >= np.asarray(u_v))



def bodyGravity(const, path, bodies, g0):
    """ Compute the gain of gravity along the path of every body to first order around the closed form. With
        the drag D = K rho v^2 m^(-1/3) and the mass loss mdot = -K sigma_eff rho v^3 m^(2/3) of the closed
        form,

            dv' = g cos z - (2 D/v) dv - D (dln rho/dl) dl + D dm/(3 m),
            dl' = dv,
            dm' = (3 mdot/v) dv + mdot (dln rho/dl) dl + (2 mdot/(3 m)) dm,

        integrated over the nodes, and the light of every interval scaled to the moved state. Sets bodies.dl,
        bodies.dv and bodies.dm.

    Arguments:
        const: [Constants object] Simulation constants.
        path: [PathTable object] Path.
        bodies: [Bodies object] Bodies.
        g0: [float] Gravitational acceleration at the surface (m/s^2).
    """

    # No gain without gravity
    if g0 == 0:
        # Zero gains of length, velocity and mass
        shape = bodies.t.shape
        bodies.dv = np.zeros(shape)
        bodies.dl = np.zeros(shape)
        bodies.dm = np.zeros(shape)
        return

    # Gains of the bodies from the compiled kernel
    bodies.dl, bodies.dv, bodies.dm = nbGravity(path.l, path.h, path.log_rho, path.dlnrho_dl, \
        float(const.h_init), float(const.zenith_angle), float(const.r_earth), float(g0), \
        int(const.lum_eff_type), float(const.lum_eff), float(const.v_kill), bodies.t, bodies.l, bodies.v, \
        bodies.m, bodies.rho, bodies.K[:, 0].copy(), bodies.sigma[:, 0].copy(), bodies.sigma_eff[:, \
        0].copy(), bodies.e_rad)



def dropTerms(const, path, lead, g0):
    """ Compute MetSim's drop of gravity across the path on the leading fragment, slowed by the drag as the
        meteoroid is, v_drop' = g sin z - (drag/v) v_drop and h_drop' = v_drop sin z.

    Arguments:
        const: [Constants object] Simulation constants.
        path: [PathTable object] Path without the drop.
        lead: [dict] Leading fragment, with its time, length, velocity, mass and K.
        g0: [float] Gravitational acceleration at the surface (m/s^2).

    Return:
        (h_drop, v_drop): [tuple] Drop (m) and velocity of the drop (m/s) at the times of the leading
            fragment.
    """

    # Drag and gravity across the path along the leading body
    t, l, v, m, K = lead['t'], lead['l'], lead['v'], np.maximum(lead['m'], 1e-30), lead['K']

    # Drag deceleration from the air density at the body
    rho = np.interp(l, path.l, path.rho)
    drag = K*rho*v**2*m**(-1/3.0)

    # Gravity and the sine of the zenith angle along it
    g, sin_z = pathGravity(const, path, l, g0)

    # Integral of drag/v, the exponent of the integrating factor
    psi = scipy.integrate.cumulative_trapezoid(drag/v, t, initial=0)

    # Velocity of the drop with the integrating factor exp(psi)
    v_drop = np.exp(-psi)*scipy.integrate.cumulative_trapezoid(g*sin_z*np.exp(psi), t, initial=0)

    return scipy.integrate.cumulative_trapezoid(v_drop*sin_z, t, initial=0), v_drop



def releaseDrop(path, drop, g):
    """ Take the drop of gravity of every grain group at its release from the path, and the velocity of the
        drop from the leading body at the time of the release, as MetSim's fragments inherit them.

    Arguments:
        path: [PathTable object] Path lowered by the drop of the bodies.
        drop: [tuple] Times (s) and lengths (m) of the leading body, its drop (m) and the velocity of its
            drop (m/s), or None without gravity.
        g: [dict] Grain groups, see grainGroups().

    Return:
        (hd, vd): [tuple] Drop (m) and its velocity (m/s) of every group at its release.
    """

    # No drop without gravity
    if drop is None:
        return np.zeros(len(g['m'])), np.zeros(len(g['m']))

    # Drop from the path at the release, its velocity from the leading body
    t_d, _, _, w_d = drop

    # Length of the path at the release of every group
    l_rel = np.interp(g['eta'], path.eta, path.l)

    return np.ascontiguousarray(np.interp(l_rel, path.l, path.h_drop)), \
        np.ascontiguousarray(np.interp(g['t'], t_d, w_d))



def pathGravity(const, path, l, g0):
    """ Compute the gravity and the sine of MetSim's local zenith angle at lengths along the straight path.

    Arguments:
        const: [Constants object] Simulation constants.
        path: [PathTable object] Path.
        l: [ndarray] Lengths along the path (m).
        g0: [float] Gravitational acceleration at the surface (m/s^2).

    Return:
        (g, sin_z): [tuple] Gravity (m/s^2) and the sine of the zenith angle at the lengths.
    """

    # Height of the straight path at the lengths
    h_c = np.interp(l, path.l, path.h)

    # Cosine and sine of the zenith angle at the lengths
    r0 = const.h_init + const.r_earth
    cos_z = (r0*np.cos(const.zenith_angle) - l)/(h_c + const.r_earth)
    sin_z = np.sqrt(np.clip(1 - cos_z**2, 0.0, None))

    return g0/(1 + h_c/const.r_earth)**2, sin_z



def pathPastBodies(const, path, drop, g, a_events, env_t0, env_dt, n_env, env_l0, env_v0, env_smin, k_end):
    """ Lower the path past the end of the bodies by the drop of gravity of the longest of the latest grain
        groups, which lead once the bodies end.

    Arguments:
        const: [Constants object] Simulation constants.
        path: [PathTable object] Path lowered by the drop of the bodies.
        drop: [tuple] Times (s) and lengths (m) of the leading body, its drop (m) and the velocity of its
            drop (m/s).
        g: [dict] Grain groups, see grainGroups().
        a_events: [list] The "A" entries.
        env_t0, env_dt: [float] Start (s) and spacing (s) of the envelope grid.
        n_env: [int] Points of the envelope grid.
        env_l0, env_v0: [ndarray] Leading body (m) and its velocity (m/s) on the envelope grid.
        env_smin: [ndarray] Lowest leading body from a grid time on, while one flies (m).
        k_end: [int] Last grid index with a body in flight.

    Return:
        [PathTable object] The path, lowered past the bodies where a grain flies there.
    """

    # The drop of the bodies, which the grains continue
    t_d, l_d, h_d, _ = drop

    # The LEAD_GROUPS latest grain groups by release time
    late = np.argsort(-g['t'], kind='stable')[:LEAD_GROUPS]
    sub = {key: np.ascontiguousarray(val[late]) for key, val in g.items()}

    # Track these groups through the A entries
    _, _, _, env_l, _, env_h, _, _ = trackGroups(path, const, sub, a_events, env_t0, env_dt, env_l0, \
        env_v0, env_smin, k_end, max(n_env, 1))

    # Grid points past the end of the bodies with a grain in flight
    t_env = env_t0 + env_dt*np.arange(n_env)
    past = (np.arange(n_env) > k_end) & np.isfinite(env_l) & (t_env > t_d[-1])

    # Path unchanged if no grain flies past the bodies
    if not np.any(past):
        return path

    # Keep the points ahead of all before them
    l_p, h_p = env_l[past], env_h[past]
    ahead = l_p > np.maximum.accumulate(np.concatenate([[l_d[-1]], l_p[:-1]]))
    l_p, h_p = l_p[ahead], h_p[ahead]

    # Path unchanged with fewer than two points
    if len(l_p) < 2:
        return path

    return path.lowered(const, np.concatenate([l_d, l_p]), np.concatenate([h_d, h_p]), l_d[-1])



def depositRamps(t_a, t_b, energy, t_grid):
    """ Sum energies spread uniformly over intervals exactly into the bins of a uniform time grid. """

    # A body's interval has no spread and q = 1
    n = len(t_a)
    return nbDepositLight(np.ascontiguousarray(t_a, dtype=np.float64), np.ascontiguousarray(t_b, \
        dtype=np.float64), np.ascontiguousarray(energy, dtype=np.float64), np.zeros(n), np.ones(n), \
        float(t_grid[0]), float(t_grid[1] - t_grid[0]), len(t_grid))



### GRAIN RELEASES ###

def massBins(const, mass, mass_index, m_min, m_max, whole=True):
    """ Compute the masses and numbers of the grains in MetSim's bins, of its power law or gamma distribution
        (generateFragments).

    Arguments:
        const: [Constants object] Simulation constants.
        mass: [float] Mass to divide (kg).
        mass_index: [float] Mass index of the power law.
        m_min, m_max: [float] Smallest and largest grain masses (kg).

    Keyword arguments:
        whole: [bool] Whole numbers, with the fraction moved to the next smaller bin as MetSim does without
            bin counts, instead of the expected numbers. True by default.

    Return:
        (m_bins, n): [tuple] Grain masses (kg) and their numbers.
    """

    # MetSim's bins, erosion_bins_per_10mass per decade of mass
    c = 10**(-1.0/const.erosion_bins_per_10mass)
    k = int(1 + np.log10(m_min/m_max)/np.log10(c))

    # Whether the grains follow the gamma distribution
    gamma_model = getattr(const, 'erosion_grain_distribution', 'powerlaw') == 'gamma'

    # MetSim's power law divides by zero when no bin fits between the masses
    if (k == 0) and (mass > 0) and (not gamma_model):
        raise ZeroDivisionError("No mass bin between the grain masses, as in MetSim's generateFragments")

    # No bins, or no mass to divide
    if (k <= 0) or (mass <= 0):
        return np.zeros(0), np.zeros(0)

    # Expected numbers of the gamma distribution or of the power law
    if gamma_model:
        # Bin masses and the expected numbers of the gamma distribution
        m_bins = m_max*c**np.arange(k)
        n_exp = gammaNumbers(const, mass, mass_index, m_min, m_max, m_bins, c)

    else:
        # Number of the largest grains
        if mass_index == 2:
            n0 = mass/(m_max*k)
        else:
            n0 = abs((mass/m_max)*(1 - c**(2 - mass_index))/(1 - c**((2 - mass_index)*k)))

        # Numbers of the power law in every bin
        m_bins = m_max*c**np.arange(k)
        n_exp = n0*(m_max/m_bins)**(mass_index - 1)

    # The expected numbers when asked for them
    if not whole:
        return m_bins, n_exp

    # Whole numbers, the rest moved to the next smaller bin
    n_out = np.zeros(k)
    leftover = 0.0
    for i in range(k):
        # Grains of the bin with the mass left by the larger bins
        n = n_exp[i] + leftover/m_bins[i]

        # Whole grains, the rest carried on as mass
        n_out[i] = np.floor(n)
        leftover = (n - n_out[i])*m_bins[i]

    # Bins with at least one grain
    keep = n_out > 0

    return m_bins[keep], n_out[keep]



def gammaNumbers(const, mass, mass_index, m_min, m_max, m_bins, c):
    """ Compute the numbers of grains in MetSim's bins for its gamma distribution of the diameters, whose
        scale is set by the mean mass of the power law with the mass index, as generateFragments does.

    Arguments:
        const: [Constants object] Simulation constants.
        mass: [float] Mass to divide (kg).
        mass_index: [float] Mass index of the power law that sets the mean.
        m_min, m_max: [float] Smallest and largest grain masses (kg).
        m_bins: [ndarray] Grain masses of the bins (kg).
        c: [float] Ratio of the masses of successive bins.

    Return:
        [ndarray] Expected number of grains in every bin.
    """

    # Log range of the grain masses, for the mean of the power law
    log_range = np.log(m_max/m_min)

    # Indices 1 and 2, where one of the integrals is a logarithm
    if mass_index == 1.0:
        m_mean = (m_max - m_min)/log_range
    elif mass_index == 2.0:
        m_mean = log_range/(1.0/m_min - 1.0/m_max)
    else:
        # Ratio of the integrals of mass and number otherwise
        a = 2 - mass_index
        b = 1 - mass_index
        m_mean = ((m_max**a - m_min**a)/a)/((m_max**b - m_min**b)/b)

    # Mean diameter and scale of the gamma distribution of the diameters
    d_mean = (6*m_mean/(np.pi*const.rho_grain))**(1/3)
    s_d = (d_mean*GAMMA_5_3)**3

    # Diameters of the bins and their numbers per unit diameter
    diam = (6*m_bins/(np.pi*const.rho_grain))**(1/3)
    n_d = (3*diam**2/s_d)*np.exp(-diam**3/s_d)

    # Numbers per unit mass, with dD/dm
    n_m = n_d*np.abs((1/3)*(6/(np.pi*const.rho_grain))**(1/3)*m_bins**(-2/3))

    # Scaled to the mass in the bins
    widths = m_bins*(1 - c)

    return n_m*mass/np.sum(n_m*widths*m_bins)*widths



def binFractions(const, mass_index, m_min, m_max):
    """ Compute the grain masses of MetSim's bins and the fraction of the mass in every bin. """

    # Expected numbers in the bins for a unit mass
    m_bins, n_exp = massBins(const, 1.0, mass_index, m_min, m_max, whole=False)

    return m_bins, n_exp*m_bins/np.sum(n_exp*m_bins)



def erosionEpochs(path, bodies, rows, n_epochs, mass_params, epochs_per_fold):
    """ Release the eroded mass of bodies as grain groups in epochs uniform in the log of the column.

    Arguments:
        path: [PathTable object] Path.
        bodies: [Bodies object] Eroding bodies.
        rows: [list] Rows of the bodies to release.
        n_epochs: [int] A quarter of the largest number of epochs of a body.
        mass_params: [tuple] Mass index and the smallest and largest grain masses (kg).
        epochs_per_fold: [int] Epochs per e-fold of the column.

    Return:
        [list] Grain groups, with their column, time, velocity, mass and release before and after their time.
    """

    # Epochs of every eroding row
    groups = []
    for r in rows:
        # Start of the body and its last column
        eta_s, m_s, K, beta = bodies.eta_s[r, 0], bodies.m_s[r, 0], bodies.K[r, 0], bodies.beta[r, 0]
        eta_end = bodies.eta[r, -1]

        # Lowest column of the epochs, skipping bodies that end before it
        lo = max(eta_s, path.eta[1])
        if eta_end <= lo:
            continue

        # Number of epochs, epochs_per_fold per e-fold, from 10 to 4 n_epochs
        n_ep = int(np.clip(epochs_per_fold*np.log(eta_end/lo), 10, 4*n_epochs))

        # Bounds of the epochs uniform in the log of the column, and their u
        log_ep = np.linspace(np.log(lo), np.log(eta_end), n_ep + 1)
        eta_ep = np.exp(log_ep)
        u_ep = uAtDeltaEta(eta_ep - eta_s, m_s, K, beta)

        # First and last bounds pinned to the ends of the body
        u_ep[0] = max(u_ep[0], bodies.u[r, 0]) if eta_s > 0 else 1.0
        u_ep[-1] = bodies.u[r, -1]

        # Eroded mass of every epoch, released at its middle
        m_ep = m_s*np.exp(-3*beta*(1 - u_ep))
        mass = -np.diff(m_ep)*bodies.erosion[r, 0]/bodies.sigma_eff[r, 0]

        # Middle of every epoch in the log of the column, and its u
        eta_mid = np.exp(0.5*(log_ep[1:] + log_ep[:-1]))
        u_mid = uAtDeltaEta(eta_mid - eta_s, m_s, K, beta)

        # Times of the bounds and of the middles
        t_ep = np.interp(eta_ep, bodies.eta[r], bodies.t[r])
        t_mid = np.interp(eta_mid, bodies.eta[r], bodies.t[r])

        # With the gain of gravity of the body
        dv_mid = np.interp(t_mid, bodies.t[r], bodies.dv[r])

        # Length of the middles with the gain, and their column
        l_mid = np.interp(eta_mid, bodies.eta[r], bodies.l[r]) + np.interp(t_mid, bodies.t[r], bodies.dl[r])
        eta_mid = np.interp(l_mid, path.l, path.eta)

        # The release of every epoch, at its middle
        groups.append({'eta': eta_mid, 't': t_mid, 'v': bodies.v_s[r, 0]*np.sqrt(u_mid) + dv_mid, \
            'mass': mass, 'sigma': np.full(n_ep, bodies.sigma[r, 0]), 'wa': t_mid - t_ep[:-1], \
            'wb': t_ep[1:] - t_mid, 'mass_params': mass_params})

    return groups



def releaseSlope(rel):
    """ Compute the relative slope of the release rate of a body's eroded mass within each of its epochs, from
        the mean rates of the neighbouring epochs (1/s).
    """

    # Times, eroded masses and widths of the epochs
    t, mass = rel['t'], rel['mass']
    width = rel['wa'] + rel['wb']

    # Zero slope for fewer than two epochs, or empty or unordered ones
    slope = np.zeros(len(t))
    if (len(t) < 2) or np.any(width <= 0) or np.any(mass <= 0) or np.any(np.diff(t) <= 0):
        return slope

    # Relative slope of the rate from the neighbouring epochs
    slope = np.gradient(np.log(mass/width), t)

    return np.clip(slope, -3.6/width, 3.6/width)



### MAIN BODY, FRAGMENTATION ENTRIES AND DISRUPTION ###

def mainEvents(const):
    """ List the heights at which the main body changes: the erosion phases before the fragmentation entries
        at the same height, as in MetSim's loop, and the entries in the order of the file.

    Arguments:
        const: [Constants object] Simulation constants.

    Return:
        [list] Events as (height, order, kind, entry).
    """

    # The start and the change of the erosion
    ev = []
    if const.erosion_on:
        h_es, h_ec = const.erosion_height_start, const.erosion_height_change

        # A change at or above the start height applies from the start
        if h_ec >= h_es:
            ev.append((h_es, -1, 'change', None))
        else:
            # Otherwise the start, then the change lower down
            ev.append((h_es, -2, 'start', None))
            ev.append((h_ec, -1, 'change', None))

    # The fragmentation entries applied downwards
    if getattr(const, 'fragmentation_on', False):
        for i, entry in enumerate(const.fragmentation_entries):
            # Entries applied only upwards are not supported
            if getattr(entry, 'upward_only', False):
                raise NotImplementedError("Fragmentation entries applied only upwards are not supported")

            # Event at the entry height, in the order of the file
            ev.append((float(entry.height), i, 'entry', entry))

    # From the top down, in the order of the entries at one height
    ev.sort(key=lambda e: (-e[0], e[1]))

    return ev



def massIndex(value, default=2.0):
    """ Mass index of an entry, 2 when it is not given. """

    return default if value is None else value



def endState(path, b, r, sigma):
    """ Take the state of a body at its last node, with the gain of gravity, where a body that runs out of
        mass releases what is left in its mass bins (MetSim's killFragment).

    Return:
        [dict] Column, time, velocity and sigma at the end.
    """

    # The last node moved by gravity
    dl = b.dl[r, -1]
    dv = b.dv[r, -1]

    # Drop a gain of gravity that is not finite or not small
    if not (np.isfinite(dl) and np.isfinite(dv) and (abs(dv) < 0.01*b.v[r, -1]) and (abs(dl) < 100.0)):
        dl, dv = 0.0, 0.0

    # Column at the moved length, with the time, velocity and sigma
    return {'eta': float(np.interp(b.l[r, -1] + dl, path.l, path.eta)), 't': float(b.t[r, -1]), \
        'v': float(b.v[r, -1] + dv), 'sigma': float(sigma)}



def gravityFar(b, r, dl_restart, grav_rel):
    """ Find the nodes of a body where the gain of gravity is no longer small: the length gained above
        dl_restart, or the mass or the velocity changed by more than grav_rel of their value.
    """

    return np.where((np.abs(b.dl[r]) > dl_restart) | (np.abs(b.dm[r]) > grav_rel*np.abs(b.m[r])) \
        | (np.abs(b.dv[r]) > grav_rel*np.abs(b.v[r])))[0]



def mainChain(path, const, settings, n_main, n_epochs, g0, with_grains=True):
    """ Follow the main body through its erosion phases and the fragmentation entries until it ends or
        disrupts.

    Arguments:
        path: [PathTable object] Path.
        const: [Constants object] Simulation constants.
        settings: [dict] Accuracy settings.
        n_main: [int] Largest number of nodes in the log of the column of a segment.
        n_epochs: [int] A quarter of the largest number of grain epochs of a segment.
        g0: [float] Gravitational acceleration at the surface (m/s^2).

    Keyword arguments:
        with_grains: [bool] Release the grains. True by default.

    Return:
        (segments, releases, seeds, a_events, disruption, body_end): [tuple] Segments of the main body,
            grain releases, seeds of the fragments, the "A" entries, the state at the disruption, and the ends
            of the grain bins.
    """

    # The events of the main body and the kill column
    shape = const.shape_factor
    eta_kill = killColumn(path, const)
    events = mainEvents(const)

    # State of the main body
    gamma = const.gamma
    rho = const.rho
    K, sigma, erosion = gamma*shape*rho**(-2/3.0), sigmaOf(const.sigma), 0.0

    # No erosion coefficient fixed by an entry yet
    erosion_fixed = False

    # Grain mass parameters of the erosion, and the index of their bins
    mass_params = (const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)
    bins = 0

    # Column, time, mass and velocity at the start
    eta_s, t_s, m_s, v_s = 0.0, 0.0, const.m_init, const.v_init

    # Results of the chain, and the index of the next event
    segments, releases, seeds, a_events = [], [], [], []
    body_end = {}
    disruption = None
    ev_i = 0
    restarts = 0

    # Grain density, no lower than the bulk density
    rho_grain = rhoGrain(const)

    # One segment of the closed form per stretch between events
    while True:

        # End the segment at the next event or the kill height
        eta_bound = path.etaAtHeight(events[ev_i][0]) if ev_i < len(events) else np.inf
        eta_stop = min(eta_bound, eta_kill)

        # Mass loss parameter with the erosion
        beta = (sigma + erosion)*v_s**2/6.0

        # End at the stop column, or at the kill velocity or mass
        _, u_e, reach, _ = nbEndAtColumn(float(eta_stop), float(eta_s), float(m_s), float(K), float(beta), \
            float(v_s), float(const.v_kill), float(const.m_kill))

        # As arrays of one body
        u_end = np.array([u_e])
        reaches = np.array([reach])

        # Restart where the gain of gravity is no longer small, up to 50 restarts
        restart = False
        if (g0 > 0) and (restarts < 50):
            # Coarse nodes of the segment with the gain of gravity
            coarse = Bodies(path, eta_s, t_s, m_s, v_s, K, sigma, erosion, u_end, n_log=60, n_u=20)
            bodyGravity(const, path, coarse, g0)

            # Nodes where the length, mass or velocity gained is no longer small
            far = gravityFar(coarse, 0, settings['dl_restart'], settings['grav_rel'])
            if len(far) and (far[0] < coarse.u.shape[1] - 2):
                # End the segment at the first far node, and restart there
                u_e = coarse.u[0, max(far[0], 1)]
                u_end = np.array([u_e])
                restart = True
                restarts += 1
                reaches = np.ones(1, dtype=bool)

        # End column of the segment and the e-folds it flies
        eta_end_seg = eta_s + nbDeltaEta(u_e, m_s**(1/3.0)/(2*K), beta, expEiScalar(beta))
        n_fold = np.log(max(eta_end_seg, path.eta[1])/max(eta_s, path.eta[1]))

        # Sixty log nodes per e-fold, from 40 to n_main
        n_log = int(np.clip(60*n_fold, 40, n_main))

        # The segment on these nodes, with the gain of gravity
        seg = Bodies(path, eta_s, t_s, m_s, v_s, K, sigma, erosion, u_end, n_log=n_log, n_u=max(n_log//2, 20))
        bodyGravity(const, path, seg, g0)

        # Disrupt where the dynamic pressure passes the strength
        if const.disruption_on:
            # Dynamic pressure at the nodes, with the gain of gravity
            p = seg.rho[0]*(seg.v[0] + seg.dv[0])**2

            # Nodes where the pressure is above the compressive strength
            above = np.where(p > const.compressive_strength)[0]
            if len(above):
                i = above[0]
                # Bisect the node interval for the crossing
                if i == 0:
                    # Disrupt at the start, already above the strength
                    u_cut = 1.0
                else:
                    # The crossing lies between node i and the node before
                    lo, hi = seg.u[0, i], seg.u[0, i - 1]

                    # Halve the bracket forty times
                    for _ in range(40):
                        mid = 0.5*(lo + hi)

                        # Column at the middle, and its velocity with the gain of gravity
                        eta_mid = eta_s + deltaEta(mid, m_s, K, beta)
                        v_mid = v_s*np.sqrt(mid) + np.interp(-mid, -seg.u[0], seg.dv[0])

                        # Keep the half of the bracket that holds the crossing
                        if np.interp(eta_mid, path.eta, path.rho)*v_mid**2 > const.compressive_strength:
                            lo = mid
                        else:
                            hi = mid

                    # Disrupt at the bisected crossing
                    u_cut = lo

                # End the segment at the disruption
                seg.truncate(path, [0], [u_cut])
                bodyGravity(const, path, seg, g0)

                # State at the disruption, with the gain of gravity
                l_act = seg.l[0, -1] + seg.dl[0, -1]
                disruption = {'eta': np.interp(l_act, path.l, path.eta), 't': seg.t[0, -1], 'm': seg.m[0, \
                    -1] + seg.dm[0, -1], 'v': seg.v[0, -1] + seg.dv[0, -1], 'K': K, 'sigma': sigma, \
                    'mass_params': mass_params, 'gamma': gamma}

        # Keep the segment, and the grains of its erosion
        segments.append(seg)
        if with_grains and (erosion > 0):
            for rel in erosionEpochs(path, seg, [0], n_epochs, mass_params, settings['epochs_per_fold']):
                # Tag the groups with the main body bins and grain K
                rel['body'] = ('main', bins)
                rel['K'] = np.full(len(rel['t']), gamma*shape*rho_grain**(-2/3.0))
                releases.append(rel)

        # Go on from the state that gravity moved
        if (disruption is None) and restart:
            l_act = seg.l[0, -1] + seg.dl[0, -1]
            eta_s, t_s, m_s, v_s = np.interp(l_act, path.l, path.eta), seg.t[0, -1], \
                seg.m[0, -1] + seg.dm[0, -1], seg.v[0, -1] + seg.dv[0, -1]
            continue

        # The main body ends here
        if (disruption is not None) or (not reaches[0]) or (eta_stop == eta_kill):
            # Whether it ends by the disruption or by running out of mass
            ends = (disruption is not None) or bool(endsByMass(const, v_s, m_s, beta, \
                reaches if eta_stop == eta_kill else np.zeros(1, dtype=bool))[0])

            # Only then is what is left in its bins released
            body_end[('main', bins)] = endState(path, seg, 0, sigma) if ends else False
            break

        # The state at the boundary, with the gain of gravity
        l_act = seg.l[0, -1] + seg.dl[0, -1]
        eta_s, t_s, m_s, v_s = np.interp(l_act, path.l, path.eta), seg.t[0, -1], seg.m[0, -1] + seg.dm[0, \
            -1], seg.v[0, -1] + seg.dv[0, -1]

        # Every event at this height
        h_now = events[ev_i][0]
        while (ev_i < len(events)) and (events[ev_i][0] == h_now):
            # Kind and entry of the event, then move to the next
            _, _, kind, entry = events[ev_i]
            ev_i += 1

            # The erosion starts
            if kind == 'start':
                # Erosion coefficient of the start, unless an entry fixed it
                if not erosion_fixed:
                    erosion = const.erosion_coeff
                continue

            # The erosion changes, with the density and sigma
            if kind == 'change':
                # Changed erosion coefficient, unless an entry fixed it
                if not erosion_fixed:
                    erosion = const.erosion_coeff_change

                # Changed density with its K, and the changed sigma
                rho = const.erosion_rho_change
                K = gamma*shape*rho**(-2/3.0)
                sigma = sigmaOf(const.erosion_sigma_change)
                continue

            # Change of the main body, with new grain bins when they change
            ft = entry.frag_type
            if ft == 'M':
                # Sigma and erosion of the entry, the erosion then kept fixed
                if entry.sigma is not None:
                    sigma = sigmaOf(entry.sigma)
                if entry.erosion_coeff is not None:
                    erosion = entry.erosion_coeff
                    erosion_fixed = True

                # Grain mass parameters of the entry, the old ones where not given
                new_params = (entry.mass_index if entry.mass_index is not None else mass_params[0], \
                    entry.grain_mass_min if entry.grain_mass_min is not None else mass_params[1], \
                    entry.grain_mass_max if entry.grain_mass_max is not None else mass_params[2])

                # Release what is left in the old bins, as MetSim does
                if new_params != mass_params:
                    body_end[('main', bins)] = {'eta': float(eta_s), 't': float(t_s), 'v': float(v_s), \
                        'sigma': float(sigma)}
                    bins += 1
                    mass_params = new_params

            # Fragments, eroding for EF
            elif ft in ('F', 'EF'):
                # Number of fragments, one when not given
                number = int(entry.number) if entry.number is not None else 1
                if number <= 0:
                    continue

                # Mass of every fragment, taken from the main body
                m_one = m_s*(entry.mass_percent/100.0)/number
                m_s = m_s - number*m_one
                if m_one > 0:
                    # Sigma of the fragments, that of the main body when not given
                    sig_c = sigmaOf(entry.sigma) if entry.sigma is not None else sigma

                    # EF fragments erode with the grain masses of the entry
                    if ft == 'EF':
                        er_c = entry.erosion_coeff if entry.erosion_coeff is not None else 0.0
                        params_c = (massIndex(entry.mass_index), entry.grain_mass_min, entry.grain_mass_max)
                    else:
                        # F fragments do not erode
                        er_c, params_c = 0.0, mass_params

                    # Seed of the fragments at the state of the main body
                    seeds.append({'eta': eta_s, 't': t_s, 'm': np.array([m_one]), 'v': v_s, 'K': K, \
                        'sigma': sig_c, 'erosion': er_c, 'n': np.array([float(number)]), \
                        'mass_params': params_c, 'ids': [('entry', id(entry))], 'gamma': gamma, \
                        'a_from': len(a_events)})

            # Dust released at once
            elif ft == 'D':
                # Dust mass, taken from the main body
                dust = m_s*(entry.mass_percent/100.0)
                m_s = m_s - dust
                if with_grains and (dust > 0):
                    # Whole grains of the dust in MetSim's bins
                    m_b, n_b = massBins(const, dust, massIndex(entry.mass_index), entry.grain_mass_min, \
                        entry.grain_mass_max, whole=True)
                    if len(m_b):
                        # One grain group at the state of the main body
                        releases.append({'body': ('dust', id(entry)), 'a_from': len(a_events), \
                            'm_bins': m_b, 'counts': n_b, \
                            'eta': np.array([eta_s]), 't': np.array([t_s]), 'v': np.array([v_s]), \
                            'sigma': np.array([sigma]), 'wa': np.zeros(1), 'wb': np.zeros(1), \
                            'K': np.array([gamma*shape*rho_grain**(-2/3.0)])})

            # A change of every fragment in flight
            elif ft == 'A':
                ratio = 1.0
                if entry.gamma is not None:
                    # New drag coefficient, with K scaled by its ratio
                    ratio = entry.gamma/gamma
                    gamma = entry.gamma
                    K = K*ratio

                # New sigma of the main body
                if entry.sigma is not None:
                    sigma = sigmaOf(entry.sigma)

                # Keep the time, sigma and K ratio for the fragments in flight
                a_events.append((t_s, None if entry.sigma is None else sigmaOf(entry.sigma), ratio))

        # A main body that the entries leave without mass ends here
        if m_s <= const.m_kill:
            body_end[('main', bins)] = {'eta': float(eta_s), 't': float(t_s), 'v': float(v_s), \
                'sigma': float(sigma)}
            break

    return segments, releases, seeds, a_events, disruption, body_end



def disruptionSeed(path, const, dis, with_grains=True):
    """ Make the fragments and grains of the disruption (MetSim's disruptFragment).

    Arguments:
        path: [PathTable object] Path.
        const: [Constants object] Simulation constants.
        dis: [dict] State of the main body at the disruption.

    Keyword arguments:
        with_grains: [bool] Release the grains. True by default.

    Return:
        (seed, releases): [tuple] Seed of the fragments, or None, and the grain release.
    """

    # Fragments of the disruption in MetSim's bins
    mass_frag = dis['m']*(1 - const.disruption_mass_grain_ratio)
    m_d, n_d = massBins(const, mass_frag, const.disruption_mass_index, \
        const.disruption_mass_min_ratio*mass_frag, const.disruption_mass_max_ratio*mass_frag) \
        if mass_frag > 0 else (np.zeros(0), np.zeros(0))

    # Mass in whole fragments, the rest of the mass goes to grains
    fragments_total = float(np.sum(m_d*n_d))

    # Empty outputs, and the grain density
    seed = None
    releases = []
    rho_grain = rhoGrain(const)

    # Seed when any bin holds a whole fragment
    if len(m_d):
        # With the erosion on, the daughters erode with the main body's K
        if const.erosion_on:
            K_d, erosion_d = dis['K'], const.disruption_erosion_coeff
        else:
            # Otherwise they do not erode and take the grain density
            K_d, erosion_d = dis['gamma']*const.shape_factor*rho_grain**(-2/3.0), 0.0

        # Daughters at the state of the main body at the disruption
        seed = {'eta': dis['eta'], 't': dis['t'], 'm': m_d, 'v': dis['v'], 'K': K_d, 'sigma': dis['sigma'], \
            'erosion': erosion_d, 'n': n_d.astype(float), 'mass_params': dis['mass_params'], \
            'ids': [('daughter', r) for r in range(len(m_d))], 'gamma': dis['gamma']}

    # The rest of the mass as grains
    mass_grain = dis['m'] - fragments_total
    if with_grains and (mass_grain > 0):
        # Whole grains in the grain bins of the main body
        idx, m_min, m_max = dis['mass_params']
        m_b, n_b = massBins(const, mass_grain, idx, m_min, m_max, whole=True)
        if len(m_b):
            # One grain group at the disruption
            releases.append({'body': 'disruption grains', 'm_bins': m_b, 'counts': n_b, \
                'eta': np.array([dis['eta']]), 't': np.array([dis['t']]), 'v': np.array([dis['v']]), \
                'sigma': np.array([dis['sigma']]), 'wa': np.zeros(1), 'wb': np.zeros(1), \
                'K': np.array([dis['gamma']*const.shape_factor*rho_grain**(-2/3.0)])})

    return seed, releases



def propagateFragments(path, const, seed, a_events, settings, n_main, n_epochs, g0, with_grains=True):
    """ Follow the fragments of a seed through the "A" entries after their birth, restarted from their actual
        state where the gain of gravity stops being small, as the main body is.

    Arguments:
        path: [PathTable object] Path.
        const: [Constants object] Simulation constants.
        seed: [dict] Fragments with the same start.
        a_events: [list] The "A" entries.
        settings: [dict] Accuracy settings.
        n_main: [int] Largest number of nodes in the log of the column of the main body.
        n_epochs: [int] A quarter of the largest number of grain epochs of the main body.
        g0: [float] Gravitational acceleration at the surface (m/s^2).

    Keyword arguments:
        with_grains: [bool] Release the grains. True by default.

    Return:
        (pieces, releases, body_end): [tuple] Bodies with their numbers and ids, the grain releases of the
            eroding ones, and the ends of their grain bins.
    """

    # Kill column, shape and grain density
    eta_kill = killColumn(path, const)
    shape = const.shape_factor
    rho_grain = rhoGrain(const)

    # The "A" entries after the seed, by list order or by time
    later = a_events[seed['a_from']:] if 'a_from' in seed else [e for e in a_events if e[0] > seed['t']]

    # Erosion of the seed, and the outputs
    erosion = seed['erosion']
    pieces, releases, body_end = [], [], {}

    # Queue of row sets, first all the fragments of the seed
    k0 = len(seed['m'])
    queue = [{'eta_s': np.full(k0, seed['eta']), 't_s': np.full(k0, seed['t']), 'm_s': np.array(seed['m'], \
        dtype=float), 'v_s': np.full(k0, seed['v']), 'K': np.full(k0, seed['K']), 'sigma': np.full(k0, \
        seed['sigma']), 'n': np.array(seed['n'], dtype=float), 'ids': list(seed['ids']), \
        'gamma': seed['gamma'], 'ev': 0, 'restarts': 0}]

    # Rows of fragments, from their start or from a cut
    while queue:
        # Take the next set of rows and its state
        q = queue.pop(0)
        eta_s, t_s, m_s, v_s, K, sigma, n, ids = (q[k] for k in ('eta_s', 't_s', 'm_s', 'v_s', 'K', 'sigma', \
            'n', 'ids'))

        # Skip a set with no rows left
        if not len(m_s):
            continue

        # The "A" entries at the start of the rows, as those listed after the entry that made them
        while (q['ev'] < len(later)) and (later[q['ev']][0] <= np.min(t_s)):
            # Scale gamma and K by the drag change of the entry
            _, sig_a, ratio = later[q['ev']]
            q['gamma'] = q['gamma']*ratio
            K = K*ratio

            # New sigma of every row when the entry gives one
            if sig_a is not None:
                sigma = np.full(len(m_s), sig_a)

            # Move to the next entry
            q['ev'] += 1

        # Mass loss parameter of every row with the erosion
        beta = (sigma + erosion)*v_s**2/6.0

        # End of every row, at the kill column or before it
        u_lo = uLimit(const, v_s, m_s, beta)
        u_end, reaches = uEndAtColumn(eta_kill, eta_s, m_s, K, beta, u_lo)

        # Rows that end by running out of mass
        ends_mass = endsByMass(const, v_s, m_s, beta, reaches)

        # Nodes of the rows with the gain of gravity
        b = Bodies(path, eta_s, t_s, m_s, v_s, K, sigma, erosion, u_end, n_log=max(n_main//2, 60), \
            n_u=max(n_main//4, 30), eta_floor=None)
        bodyGravity(const, path, b, g0)

        # Cut every row at the next "A" entry or gravity restart
        t_a = later[q['ev']][0] if q['ev'] < len(later) else np.inf

        # Rows cut at the entry and rows cut by gravity
        cut_a, cut_g = [], []
        for r in range(len(m_s)):
            u_cut, kind = None, None

            # Cut where the gain of gravity grows large, up to 50 restarts
            if g0 > 0 and q['restarts'] < 50:
                far = gravityFar(b, r, settings['dl_restart'], settings['grav_rel'])
                if len(far) and (far[0] < b.u.shape[1] - 2):
                    u_cut, kind = b.u[r, max(far[0], 1)], 'g'

            # Cut at the entry when the row flies across its time
            if b.t[r, 0] < t_a < b.t[r, -1]:
                u_a = np.interp(t_a, b.t[r], b.u[r])

                # Take the entry cut unless the gravity cut comes earlier
                if (u_cut is None) or (u_a > u_cut):
                    u_cut, kind = u_a, 'a'

            # Sort the row into the cuts of its kind
            if kind == 'a':
                cut_a.append((r, u_cut))
            elif kind == 'g':
                cut_g.append((r, u_cut))

        # End the cut rows at their cut
        if cut_a or cut_g:
            b.truncate(path, [r for r, _ in cut_a + cut_g], [u_c for _, u_c in cut_a + cut_g])
            bodyGravity(const, path, b, g0)

        # Keep the piece with the numbers and ids of its rows
        b.n = n[:, None]
        b.ids = ids
        pieces.append(b)

        # Release the bins of uncut eroding rows that run out of mass
        cut_set = {r for r, _ in cut_a + cut_g}
        for r in range(len(m_s)):
            if (r not in cut_set) and (erosion > 0):
                body_end[ids[r]] = endState(path, b, r, sigma[r]) if ends_mass[r] else False

        # Grains of the erosion of the rows
        if with_grains and (erosion > 0):
            for r in range(len(m_s)):
                for rel in erosionEpochs(path, b, [r], max(n_epochs//4, 10), seed['mass_params'], \
                    settings['epochs_per_fold']):
                    # Tag the groups with the row, its number and grain K
                    rel['body'] = ids[r]
                    rel['mult'] = float(n[r])
                    rel['K'] = np.full(len(rel['t']), q['gamma']*shape*rho_grain**(-2/3.0))
                    releases.append(rel)

        # Go on from the actual state of the cut rows
        for cuts, at_event in ((cut_a, True), (cut_g, False)):
            if not cuts:
                continue

            # State of the cut rows at their end
            rows = [r for r, _ in cuts]
            l_act = np.array([b.l[r, -1] + b.dl[r, -1] for r in rows])
            nxt = {'eta_s': np.interp(l_act, path.l, path.eta), 't_s': np.array([b.t[r, \
                -1] for r in rows]), 'm_s': np.array([b.m[r, -1] + b.dm[r, -1] for r in rows]), \
                'v_s': np.array([b.v[r, -1] + b.dv[r, -1] for r in rows]), 'K': K[rows].copy(), \
                'sigma': sigma[rows].copy(), 'n': n[rows], 'ids': [ids[r] for r in rows], \
                'gamma': q['gamma'], 'ev': q['ev'], 'restarts': q['restarts']}

            # Rows cut at the entry take its change of gamma and K
            if at_event:
                _, sig_a, ratio = later[q['ev']]
                nxt['gamma'] = q['gamma']*ratio
                nxt['K'] = nxt['K']*ratio

                # New sigma of the rows when the entry gives one
                if sig_a is not None:
                    nxt['sigma'] = np.full(len(rows), sig_a)

                # Move to the next entry
                nxt['ev'] = q['ev'] + 1
            else:
                # Count one more gravity restart
                nxt['restarts'] = q['restarts'] + 1

            # Rows that still fly
            keep = (nxt['v_s'] > const.v_kill) & (nxt['m_s'] > const.m_kill)

            # Keep only those rows in the next set
            for key in ('eta_s', 't_s', 'm_s', 'v_s', 'K', 'sigma', 'n'):
                nxt[key] = nxt[key][keep]
            nxt['ids'] = [i for i, k in zip(nxt['ids'], keep) if k]

            # Queue the set for its next stretch
            queue.append(nxt)

    return pieces, releases, body_end



def runBodies(path, const, settings, n_main, n_epochs, g0, with_grains):
    """ Follow the main body and all its fragments.

    Return:
        (segments, pieces, releases, a_events, body_end, dis): [tuple] Segments of the main body, the
            fragments, the grain releases, the "A" entries, the ends of the grain bins and the state at the
            disruption.
    """

    # The main body, then the disruption
    segments, releases, seeds, a_events, dis, body_end = mainChain(path, const, settings, n_main, \
        n_epochs, g0, with_grains=with_grains)

    # Fragments and grains of the disruption, if there is one
    if dis is not None:
        seed, rel_d = disruptionSeed(path, const, dis, with_grains=with_grains)

        # Add its grains to the releases, and its fragments to the seeds
        releases += rel_d
        if seed is not None:
            seeds.append(seed)

    # The fragments of every seed
    pieces = []
    for seed in seeds:
        p, rel, ends = propagateFragments(path, const, seed, a_events, settings, n_main, n_epochs, g0, \
            with_grains=with_grains)

        # Gather their pieces, grains and bin ends
        pieces += p
        releases += rel
        body_end.update(ends)

    return segments, pieces, releases, a_events, body_end, dis



### LIGHT AND LEADING FRAGMENT ###

def grainGroups(const, releases, body_end, a_events):
    """ Gather the grain groups of every release, as whole grains per body as MetSim counts them.

    Arguments:
        const: [Constants object] Simulation constants.
        releases: [list] Grain releases.
        body_end: [dict] End of the grain bins of every eroding body.
        a_events: [list] The "A" entries.

    Return:
        (g, bin_steps): [tuple] The mass (kg), number, column (kg/m^2), time (s), velocity (m/s), ablation
            coefficient (s^2/m^2), K, release before and after the time (s), release slope, first "A" entry
            and origin in a fragmentation entry (1) of every group (None without grains), and the mass left
            in the bins of every body.
    """

    # Fields of the groups, and the slope of every release
    keys = ('eta', 't', 'v', 'sigma', 'wa', 'wb', 'K', 'slope')
    cols = {key: [] for key in ('m', 'n', 'a_from', 'complex') + keys}
    for rel in releases:
        if 'slope' not in rel:
            rel['slope'] = np.zeros(len(rel['t'])) if 'counts' in rel else releaseSlope(rel)

    # Releases of every body
    groups = {}
    bin_steps = []
    for rel in releases:
        groups.setdefault(rel['body'], []).append(rel)

    # Groups of every body
    for body, rels in groups.items():
        # Number of bodies the releases stand for
        mult = rels[0].get('mult', 1.0)

        # Whether the grains come from a fragmentation entry
        cx = float(isinstance(body, tuple) and (body[0] in ('entry', 'dust')))

        # Releases with their own counts of grains, the dust and the disruption
        if 'counts' in rels[0]:
            for rel in rels:
                # A group for every mass bin, with its count of grains
                k = len(rel['m_bins'])
                cols['m'].append(rel['m_bins'])
                cols['n'].append(rel['counts']*mult)

                # First "A" entry if known, and the origin in a fragmentation entry
                cols['a_from'].append(np.full(k, float(rel.get('a_from', -1))))
                cols['complex'].append(np.full(k, cx))

                # The state of the release, repeated for every bin
                for key in keys:
                    cols[key].append(np.repeat(rel[key], k))
            continue

        # MetSim's bins of the erosion of the body
        m_bins, w_bins = binFractions(const, *rels[0]['mass_params'])

        # No grains without a bin between the grain masses
        if not len(m_bins):
            continue

        # Epochs of the body in time order
        cat = {key: np.concatenate([r[key] for r in rels]) for key in keys + ('mass',)}
        order = np.argsort(cat['t'], kind='stable')
        cat = {key: val[order] for key, val in cat.items()}

        # Whole grains of every bin, the fractions kept for the next epochs
        expected = cat['mass'][:, None]*w_bins[None, :]/m_bins[None, :]
        cumulative = np.cumsum(expected, axis=0)
        whole = np.floor(cumulative)
        counts = np.diff(whole, axis=0, prepend=0.0)

        # Where the body releases what is left in its bins, if it does
        end = body_end.get(body, False)

        # Mass left in the bins, part of the total active mass
        frac = ((cumulative - whole)*m_bins[None, :]).sum(axis=1)*mult
        bin_steps.append((cat['t'].copy(), frac, end['t'] if isinstance(end, dict) else cat['t'][-1]))

        # What is left in the bins leaves where the body ends
        if end:
            # The fraction of a grain left in every bin
            cols['m'].append(m_bins)
            cols['n'].append((cumulative[-1] - whole[-1])*mult)
            cols['a_from'].append(np.full(len(m_bins), -1.0))
            cols['complex'].append(np.full(len(m_bins), cx))

            # Released at the end of the body at once, with the last K
            for key, val in (('eta', end['eta']), ('t', end['t']), ('v', end['v']), ('sigma', end['sigma']), \
                ('wa', 0.0), ('wb', 0.0), ('K', cat['K'][-1]), ('slope', 0.0)):
                cols[key].append(np.full(len(m_bins), val))

        # Every bin of every epoch
        k = len(m_bins)
        cols['m'].append(np.tile(m_bins, len(cat['mass'])))
        cols['n'].append((counts*mult).ravel())

        # First "A" entry found from the time, and the origin
        cols['a_from'].append(np.full(k*len(cat['mass']), -1.0))
        cols['complex'].append(np.full(k*len(cat['mass']), cx))

        # The state of every epoch, repeated for every bin
        for key in keys:
            cols[key].append(np.repeat(cat[key], k))

    # No grains
    if not cols['m']:
        return None, bin_steps

    # Groups with grains
    g = {key: np.concatenate(val) for key, val in cols.items()}
    keep = g['n'] > 0
    g = {key: np.ascontiguousarray(val[keep], dtype=np.float64) for key, val in g.items()}

    # None of the groups has grains
    if not len(g['m']):
        return None, bin_steps

    # The first "A" entry of every group, after the entry that made it or after its release
    a_t = np.array([e[0] for e in a_events], dtype=np.float64)
    g['a0'] = np.ascontiguousarray(np.where(g['a_from'] >= 0, g['a_from'], np.searchsorted(a_t, g['t'], \
        side='right')), dtype=np.int64)

    return g, bin_steps



def trackGroups(path, const, g, a_events, env_t0, env_dt, env_l0, env_v0, env_smin, k_end, m_stride):
    """ Evaluate grain groups through the "A" entries with nbGrainTracks(), which gives the arguments and the
        return.
    """

    # The "A" entries as arrays
    a_t = np.array([e[0] for e in a_events], dtype=np.float64)
    a_sigma = np.array([np.nan if e[1] is None else e[1] for e in a_events], dtype=np.float64)
    a_ratio = np.array([e[2] for e in a_events], dtype=np.float64)

    # Evaluate the groups in the compiled kernel
    return nbGrainTracks(path.grain_tab, float(killColumn(path, const)), float(const.v_kill), \
        float(const.m_kill), int(const.lum_eff_type), float(const.lum_eff), g['m'], g['n'], g['eta'], \
        g['t'], g['v'], g['sigma'], g['K'], g['wa'], g['wb'], g['hd'], g['vd'], g['a0'], int(GRAIN_NODES), \
        a_t, a_sigma, a_ratio, float(env_t0), float(env_dt), env_l0, env_v0, env_smin, int(k_end), \
        int(m_stride), float(MetSimErosion.G0))



def grainLight(path, const, g, a_events, env_t0, env_dt, n_env, m_stride, env_l0, env_v0, env_smin, k_end):
    """ Evaluate the grain groups, their light and the longest grain in flight.

    Arguments:
        path: [PathTable object] Path.
        const: [Constants object] Simulation constants.
        g: [dict] Grain groups, see grainGroups(), or None.
        a_events: [list] The "A" entries.
        env_t0, env_dt: [float] Start (s) and spacing (s) of the envelope grid.
        n_env: [int] Points of the envelope grid.
        m_stride: [int] Stride of the envelope grid where the mass in flight is summed.
        env_l0, env_v0: [ndarray] Leading body (m) and its velocity (m/s) on the envelope grid.
        env_smin: [ndarray] Lowest leading body from a grid time on, while one flies (m).
        k_end: [int] Last grid index with a body in flight.

    Return:
        (deposits, deposits_entries, env_l, env_v, env_m): [tuple] Light intervals of the grains and those
            of the grains of fragmentation entries, the longest fragment in flight, its velocity and the grain
            mass in flight on the envelope grid.
    """

    # No grains
    if g is None:
        return [], [], np.full(n_env, -np.inf), np.zeros(n_env), np.zeros(n_env)

    # Evaluate every group
    t_a, t_b, e, env_l, env_v, _, env_m, row_end = trackGroups(path, const, g, a_events, env_t0, env_dt, \
        env_l0, env_v0, env_smin, k_end, m_stride)

    # Epoch of every interval and the grains of its second half
    row = np.repeat(np.arange(len(row_end)), np.diff(np.concatenate([[0], row_end])))
    spread = (g['wa'] + g['wb'])[row]
    q = np.clip(0.25*g['slope'][row]*spread, -0.9, 0.9)

    # Intervals of the grains of the fragmentation entries
    cx = g['complex'][row] > 0
    entries = [(t_a[cx], t_b[cx], e[cx], spread[cx], q[cx])] if np.any(cx) else []

    return [(t_a, t_b, e, spread, q)], entries, env_l, env_v, env_m



def envelopeStart(lead, env_t0, env_dt, n_env):
    """ Start the envelope of the longest grain from the leading body.

    Arguments:
        lead: [dict] Leading body, with its time (s), length (m) and velocity (m/s).
        env_t0, env_dt: [float] Start (s) and spacing (s) of the envelope grid.
        n_env: [int] Points of the envelope grid.

    Return:
        (env_l0, env_v0, env_smin, k_end): [tuple] Leading body (m) and its velocity (m/s) on the grid, the
            lowest leading body from a grid time on while one flies (m), and the last grid index with a body
            in flight.
    """

    # The leading body on the grid while it flies
    t_env = env_t0 + env_dt*np.arange(n_env)
    alive_b = t_env <= lead['t'][-1]

    # Length and velocity of the body, none after it ends
    env_l0 = np.where(alive_b, np.interp(t_env, lead['t'], lead['l']), -np.inf)
    env_v0 = np.where(alive_b, np.interp(t_env, lead['t'], lead['v']), 0.0)

    # Last grid index with the body in flight
    k_end = int(np.where(alive_b)[0][-1]) if np.any(alive_b) else -1

    # Lowest leading body from every grid time on, while it flies
    env_smin = np.full(n_env, -np.inf)
    if k_end >= 0:
        env_smin[:k_end + 1] = np.minimum.accumulate(env_l0[:k_end + 1][::-1])[::-1]

    return env_l0, env_v0, env_smin, k_end



def leadingTrack(segments, pieces):
    """ Track the leading fragment, the longest body in flight at every time, as MetSim's leading_frag.

    Arguments:
        segments: [list] Segments of the main body.
        pieces: [list] Fragments.

    Return:
        [dict] Time, length, velocity, mass and K of the leading fragment.
    """

    # Track of the main body
    t = np.concatenate([s.t[0] for s in segments])

    # Length and velocity with the gain of gravity
    l = np.concatenate([s.l[0] + s.dl[0] for s in segments])
    v = np.concatenate([s.v[0] + s.dv[0] for s in segments])

    # Mass and K at the nodes of every segment
    m = np.concatenate([s.m[0] for s in segments])
    K = np.concatenate([np.full(s.u.shape[1], s.K[0, 0]) for s in segments])

    # Nodes in time order
    order = np.argsort(t, kind='stable')
    t, l, v, m, K = (a[order] for a in (t, l, v, m, K))

    # Drop the repeated times
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, l, v, m, K = (a[keep] for a in (t, l, v, m, K))

    # Only the main body without fragments
    if not pieces:
        return {'t': t, 'l': l, 'v': v, 'm': m, 'K': K}

    # Longest body on a common time grid, the earlier one on ties
    rows = [(p, r) for p in pieces for r in range(p.t.shape[0])]
    t_end = max(max(p.t[r, -1] for p, r in rows), t[-1])
    t_g = np.unique(np.concatenate([t, np.linspace(t[0], t_end, 4000)]))

    # The main body leads while it flies
    alive = t_g <= t[-1]
    lead_l = np.full(len(t_g), -np.inf)
    lead_l[alive] = np.interp(t_g[alive], t, l)

    # Velocity, mass and K of the main body
    lead_v = np.interp(t_g, t, v)
    lead_m = np.interp(t_g, t, m)
    lead_k = np.interp(t_g, t, K)

    # Fragment rows take the lead where they are ahead
    for p, r in rows:
        # Grid points while the row flies
        i0 = np.searchsorted(t_g, p.t[r, 0], side='left')
        i1 = np.searchsorted(t_g, p.t[r, -1], side='right')
        if i1 <= i0:
            continue

        # Length of the row with gravity, and where it is ahead
        t_r = t_g[i0:i1]
        l_r = np.interp(t_r, p.t[r], p.l[r] + p.dl[r])
        ahead = l_r > lead_l[i0:i1]
        if not np.any(ahead):
            continue

        # The row leads at those points
        idx = i0 + np.where(ahead)[0]
        lead_l[idx] = l_r[ahead]

        # Its velocity, mass and K there
        lead_v[idx] = np.interp(t_r[ahead], p.t[r], p.v[r] + p.dv[r])
        lead_m[idx] = np.interp(t_r[ahead], p.t[r], p.m[r])
        lead_k[idx] = p.K[r, 0]

    # Grid times where a body flies
    ok = np.isfinite(lead_l)

    return {'t': t_g[ok], 'l': lead_l[ok], 'v': lead_v[ok], 'm': lead_m[ok], 'K': lead_k[ok]}



def simulate(const, settings, dt_out):
    """ Simulate the light on a uniform grid and the leading fragment.

    Arguments:
        const: [Constants object] Simulation constants.
        settings: [dict] Accuracy settings.
        dt_out: [float] Spacing of the light grid (s).

    Return:
        (t, lum, lead, info): [tuple] Times at the end of every bin (s), the light of every bin (W), the
            leading body, and the rest of the run (main body, fragments, grains in flight, path).
    """

    # MetSim's swap of the grain masses and raise of the grain density
    const = copy.copy(const)
    if const.erosion_mass_min > const.erosion_mass_max:
        const.erosion_mass_min, const.erosion_mass_max = const.erosion_mass_max, const.erosion_mass_min
    const.rho_grain = max(const.rho_grain, const.rho)

    # Path, gravity and the numbers of nodes and epochs
    path = cachedPath(const)
    g0 = MetSimErosion.G0
    n_main = settings['n_main']
    n_epochs = 80

    # A first pass for the drop of gravity on the leading fragment
    drop = None
    if g0 > 0:
        # Bodies with fewer nodes and without grains
        segments, pieces, _, _, _, _ = runBodies(path, const, settings, max(n_main//4, 60), n_epochs, g0, \
            False)

        # Drop of gravity and its velocity along the leading fragment
        lead = leadingTrack(segments, pieces)
        h_drop, v_drop = dropTerms(const, path, lead, g0)
        drop = (lead['t'], lead['l'], h_drop, v_drop)

        # Path lowered by the drop
        path = PathTable(const, drop_l=lead['l'], drop_h=h_drop, **path.args)

    # All the bodies and the releases of their grains
    segments, pieces, releases, a_events, body_end, dis = runBodies(path, const, settings, n_main, \
        n_epochs, g0, True)

    # Light of the segments of the main body
    deposits = [(s.t[0, :-1], s.t[0, 1:], s.e_rad[0]) for s in segments]

    # Light of the fragments, times their number in every row
    for p in pieces:
        deposits.append((p.t[:, :-1].ravel(), p.t[:, 1:].ravel(), (p.e_rad*p.n).ravel()))

    # Envelope of the longest grain on the output grid
    lead = leadingTrack(segments, pieces)
    env_t0 = 0.0
    env_step = dt_out

    # Grid points in MetSim's step, where the mass in flight is summed
    m_stride = max(1, int(round(const.dt/dt_out)))

    # Grid to 30 s past the end of the leading body
    n_env = int(np.ceil((lead['t'][-1] + 30.0)/env_step))

    # Start the envelope from the leading body
    env_l0, env_v0, env_smin, k_end = envelopeStart(lead, env_t0, env_step, n_env)

    # The grain groups, with the drop of gravity continued past the bodies along the grains that lead there
    g, bin_steps = grainGroups(const, releases, body_end, a_events)
    if g is not None:
        # Drop of gravity of every group at its release
        g['hd'], g['vd'] = releaseDrop(path, drop, g)

        # Path lowered past the bodies by the drop of the leading grains
        if drop is not None:
            path = pathPastBodies(const, path, drop, g, a_events, env_t0, env_step, n_env, env_l0, env_v0, \
                env_smin, k_end)

    # Light of the grains and the envelope
    grains = grainLight(path, const, g, a_events, env_t0, env_step, n_env, m_stride, env_l0, env_v0, \
        env_smin, k_end)

    # A longer envelope grid when grains fly past its end
    t_grains = max([float(np.max(d[1][d[2] > 0])) for d in grains[0] if np.any(d[2] > 0)] + [0.0])
    if t_grains > env_t0 + env_step*(n_env - 1):
        # Grid to 1 s past the last light of the grains, with the leading body
        n_env = int(np.ceil((t_grains + 1.0)/env_step))
        env_l0, env_v0, env_smin, k_end = envelopeStart(lead, env_t0, env_step, n_env)

        # Light of the grains on the longer grid
        grains = grainLight(path, const, g, a_events, env_t0, env_step, n_env, m_stride, env_l0, env_v0, \
            env_smin, k_end)

    # Light of the grains
    grain_deposits, entry_deposits, env_l, env_v, env_m = grains
    deposits += grain_deposits

    # Bodies and grains of the fragmentation entries, which MetSim leaves out of the eroded light
    for p in pieces:
        # Rows of the fragments made by the entries
        rows = [r for r, i in enumerate(p.ids) if isinstance(i, tuple) and (i[0] == 'entry')]
        if rows:
            # Light of those rows, times their number
            entry_deposits.append((p.t[rows, :-1].ravel(), p.t[rows, 1:].ravel(), (p.e_rad[rows] \
                *p.n[rows]).ravel()))

    # Sum the light into the bins of the output grid
    t_max = max(float(np.max(d[1][d[2] > 0])) if np.any(d[2] > 0) else 0.0 for d in deposits)
    t_grid = np.arange(0.0, t_max + 2*dt_out, dt_out)
    energy = np.zeros(len(t_grid) - 1)
    for d in deposits:
        # Grain intervals, spread over the epochs of their release
        if len(d) == 5:
            energy += nbDepositLight(*d, float(t_grid[0]), float(dt_out), len(t_grid))
        else:
            # Body intervals, without spread
            energy += depositRamps(*d, t_grid)

    # Light of every bin (W)
    lum = energy/dt_out

    # Heights of the leading body, and the longest grain
    lead['h'] = np.interp(lead['l'], path.l, path.h_rep)
    alive = np.isfinite(env_l)
    if np.any(alive):
        # The longest grain up to its last point in flight
        last = np.where(alive)[0][-1]
        env = {'t': env_t0 + env_step*np.arange(last + 1), 'l': env_l[:last + 1], 'v': env_v[:last + 1]}
    else:
        # No grain in flight
        env = None

    # Light of the main body
    main_energy = np.zeros(len(t_grid) - 1)
    for s in segments:
        main_energy += depositRamps(s.t[0, :-1], s.t[0, 1:], s.e_rad[0], t_grid)

    # Light of the fragmentation entries
    entry_energy = np.zeros(len(t_grid) - 1)
    for d in entry_deposits:
        # Grain intervals, spread over the epochs of their release
        if len(d) == 5:
            entry_energy += nbDepositLight(*d, float(t_grid[0]), float(dt_out), len(t_grid))
        else:
            # Body intervals, without spread
            entry_energy += depositRamps(*d, t_grid)

    # Track of the main body at its nodes
    main = {'t': np.concatenate([s.t[0] for s in segments]), 'l': np.concatenate([s.l[0] \
        + s.dl[0] for s in segments]), 'v': np.concatenate([s.v[0] + s.dv[0] for s in segments]), \
        'm': np.concatenate([s.m[0] for s in segments])}

    # The rest of the run, for the results
    info = {'disruption': dis, 'path': path, 'lum_main': main_energy/dt_out, 'main': main, \
        'lum_entries': entry_energy/dt_out, \
        'grain_lead': env, 'grain_mass': (env_t0, env_step*m_stride, env_m[::m_stride]), \
        'bin_steps': bin_steps, 'pieces': pieces, 'segments': segments}

    return t_grid[1:], lum, lead, info



### RESULTS AS METSIM'S ###

def setRunConstants(const, info, path, t, mt, ml, mv, mm, main_h, alive):
    """ Set the values MetSim records in the constants during a run, which the post-processing of
        DynestyMetSim reads: the main body at the end of its first eroding step (velocity, mass, dynamic
        pressure), its mass at the end of the first eroding step below erosion_height_change, the height
        where it ends and the lowest height it reaches, the height of the disruption and the length of the
        run.
    """

    # Start and change of the erosion, from the main body
    dt = const.dt
    const.erosion_beg_vel = const.erosion_beg_mass = const.erosion_beg_dyn_press = None
    const.mass_at_erosion_change = None

    # First segment of the main body that erodes
    seg_e = next((s for s in info['segments'] if s.erosion[0, 0] > 0), None)
    if seg_e is not None:
        # End of the first eroding step of MetSim
        t_rec = dt*np.floor(seg_e.t[0, 0]/dt + 1e-9) + dt
        if t_rec <= mt[-1]:
            # Velocity, mass and dynamic pressure of the main body there
            l_rec = np.interp(t_rec, mt, ml)
            const.erosion_beg_vel = float(np.interp(t_rec, mt, mv))
            const.erosion_beg_mass = float(np.exp(np.interp(t_rec, mt, np.log(np.clip(mm, 1e-300, None)))))
            const.erosion_beg_dyn_press = float(np.interp(l_rec, path.l, path.rho)*const.erosion_beg_vel**2)

            # The first eroding step below the change of the erosion
            h_c = const.erosion_height_change
            h_main = np.interp(ml, path.l, path.h_rep)
            below = np.where((h_main <= h_c) & (mt > t_rec))[0]
            if len(below):
                # Mass of the main body at the end of that step
                t_c = dt*np.floor(mt[below[0]]/dt + 1e-9) + dt
                if (t_c <= mt[-1]) and (np.interp(t_c, mt, mm) > 0):
                    const.mass_at_erosion_change = float(np.exp(np.interp(t_c, mt, np.log(np.clip(mm, \
                        1e-300, None)))))

    # Height where the main body ends, and the lowest
    const.main_mass_exhaustion_ht = float(np.interp(ml[-1], path.l, path.h_rep))
    const.main_bottom_ht = min(float(np.min(main_h[alive])) if np.any(alive) else const.h_init, \
        const.main_mass_exhaustion_ht)

    # Height of the disruption, None without one or with no mass for fragments
    dis = info['disruption']
    no_fragments = (dis is None) or (const.disruption_mass_grain_ratio >= 1)
    const.disruption_height = None if no_fragments else float(np.interp(np.interp(dis['eta'], path.eta, \
        path.l), path.l, path.h_rep))

    # Duration of the run
    const.total_time = float(t[-1])



def envelopeAt(t, env, step):
    """ Interpolate the longest fragment in flight at the given times from its envelope grid: linear between
        two samples in flight, the earlier sample until the next if that one has none, and -inf before a
        sample in flight, as the grains are released at the body, behind it.

    Return:
        (l_env, v_env): [tuple] Length (m) and velocity (m/s) of the longest fragment at the times.
    """

    # Longest grain at the times, from the grid
    n = len(t)
    l_env = np.full(n, -np.inf)
    v_env = np.zeros(n)

    # Number of samples, and the times in grid steps
    m = len(env['l'])
    x = (t - env['t'][0])/step

    # Times on the grid take the samples as they are
    k = np.rint(x).astype(np.int64)
    exact = (np.abs(x - k) < 1e-6) & (k >= 0) & (k < m)
    kc = np.clip(k, 0, m - 1)
    l_env[exact] = env['l'][kc][exact]
    v_env[exact] = env['v'][kc][exact]

    # Done when every time is on the grid
    if np.all(exact):
        return l_env, v_env

    # Linear between the grid points
    k0 = np.floor(x).astype(np.int64)
    f = x - k0
    inside = (k0 >= -1) & (k0 < m)

    # Samples before and after every time
    a = np.clip(k0, 0, m - 1)
    b = np.clip(k0 + 1, 0, m - 1)
    la, lb = env['l'][a], env['l'][b]

    # Linear where both samples are in flight
    both = ~exact & inside & (k0 >= 0) & (k0 + 1 < m) & np.isfinite(la) & np.isfinite(lb)
    l_env[both] = la[both] + f[both]*(lb[both] - la[both])
    v_env[both] = env['v'][a][both] + f[both]*(env['v'][b][both] - env['v'][a][both])

    # The earlier sample where the next one has none
    hold = ~exact & inside & ~both & (k0 >= 0) & np.isfinite(la)
    l_env[hold] = la[hold]
    v_env[hold] = env['v'][a][hold]

    return l_env, v_env



class MainFragment(object):
    def __init__(self, m, v, h):
        """ Final state of the main fragment, as MetSim's frag_main.

        Arguments:
            m: [float] Mass (kg).
            v: [float] Velocity (m/s).
            h: [float] Height (m).
        """

        # Mass, velocity and height at the end
        self.m = m
        self.v = v
        self.h = h



def runSimulation(const, compute_wake=False, accuracy='fast'):
    """ Run MetSim's erosion model from the alpha-beta closed forms. It returns the columns of
        MetSimErosion.runSimulation on MetSim's step split into steps near OUTPUT_STEP, with const.dt left at
        MetSim's step for the constants that are saved. The ionization and the tau columns are not computed
        (zero and NaN), and the brightest fragment is the leading one.

    Arguments:
        const: [Constants object] Simulation constants.

    Keyword arguments:
        compute_wake: [bool] Not supported, must be False. False by default.
        accuracy: [str] 'fast' or 'precise', see ACCURACY. 'fast' by default.

    Return:
        (frag_main, results_list, wake_results):
            - frag_main: [MainFragment object] Final state of the main fragment.
            - results_list: [ndarray] The 21 columns of MetSim's results, one row per time.
            - wake_results: [list] None for every row.
    """

    # No wake
    if compute_wake:
        raise NotImplementedError("The wake is not computed by the alpha-beta model")

    # The run on MetSim's step split into steps near OUTPUT_STEP
    settings = ACCURACY[accuracy]
    dt_step = const.dt
    n_sub = max(1, int(round(dt_step/OUTPUT_STEP)))

    # Light, leading fragment and the rest of the run on that grid
    t, lum, lead, info = simulate(const, settings, dt_step/n_sub)
    path = info['path']

    # Results while any fragment flies, led by the longest
    env = info['grain_lead']
    t_last = lead['t'][-1] if env is None else max(lead['t'][-1], env['t'][-1])

    # Times while a fragment flies, at least two
    keep = t <= t_last + 1e-9
    if np.sum(keep) < 2:
        keep[:2] = True

    # Light at the kept times
    t, lum = t[keep], lum[keep]
    info['lum_main'] = info['lum_main'][keep]
    info['lum_entries'] = info['lum_entries'][keep]
    n = len(t)

    # Leading body at the times while it flies
    body_alive = t <= lead['t'][-1]
    l_lead = np.where(body_alive, np.interp(t, lead['t'], lead['l']), -np.inf)
    v_lead = np.interp(t, lead['t'], lead['v'])

    # The envelope at the times, when there is one
    if env is not None:
        l_env, v_env = envelopeAt(t, env, env['t'][1] - env['t'][0] if len(env['t']) > 1 else dt_step/n_sub)

        # It leads where it is ahead of the body
        ahead = l_env > l_lead
        l_lead = np.where(ahead, l_env, l_lead)
        v_lead = np.where(ahead, v_env, v_lead)

    # Last state of the leading body where none flies
    bad = ~np.isfinite(l_lead)
    l_lead[bad] = lead['l'][-1]
    v_lead[bad] = lead['v'][-1]

    # Height and air density of the leading fragment
    h_lead = np.interp(l_lead, path.l, path.h_rep)
    rho = np.interp(l_lead, path.l, path.rho)

    # The main body in time order
    main = info['main']
    order = np.argsort(main['t'], kind='stable')
    mt, ml, mv, mm = (main[k][order] for k in ('t', 'l', 'v', 'm'))

    # Its length, velocity and mass at the times, zero after it ends as in MetSim
    alive = t <= mt[-1]
    main_l = np.where(alive, np.interp(t, mt, ml), 0.0)
    main_v = np.where(alive, np.interp(t, mt, mv), 0.0)
    main_m = np.where(alive, np.interp(t, mt, mm), 0.0)

    # Height and dynamic pressure of the main body
    main_h = np.where(alive, np.interp(main_l, path.l, path.h_rep), 0.0)
    main_p = np.where(alive, np.interp(main_l, path.l, path.rho)*main_v**2, 0.0)

    # Times of MetSim's steps, or all the times when there are fewer than two
    tc = t[n_sub - 1::n_sub] if n_sub > 1 else t
    if len(tc) < 2:
        tc = t

    # Mass of the fragments in flight
    rest = np.zeros(len(tc))
    for p in info['pieces']:
        for r in range(p.t.shape[0]):
            # MetSim's steps while the row flies
            i0 = np.searchsorted(tc, p.t[r, 0], side='left')
            i1 = np.searchsorted(tc, p.t[r, -1], side='right')
            if i1 > i0:
                # Mass of the row interpolated in the log, times its number
                m_r = np.clip(p.m[r] + p.dm[r], 1e-300, None)
                rest[i0:i1] += float(np.ravel(p.n[r])[0])*np.exp(np.interp(tc[i0:i1], p.t[r], np.log(m_r)))

    # Grain mass in flight from the envelope grid
    g_t0, g_dt, g_m = info['grain_mass']

    # Mass at the nearest grid point of every step
    k = np.rint((tc - g_t0)/g_dt).astype(np.int64)
    inside = (k >= 0) & (k < len(g_m))
    rest[inside] += g_m[k[inside]]

    # Mass left in the grain bins of the bodies until they end
    for t_ep, frac, t_stop in info['bin_steps']:
        # Steps from the first epoch to the end of the body
        i0 = np.searchsorted(tc, t_ep[0], side='left')
        i1 = np.searchsorted(tc, t_stop, side='left')
        if i1 > i0:
            # The mass left at the last epoch up to every step
            rest[i0:i1] += frac[np.clip(np.searchsorted(t_ep, tc[i0:i1], side='right') - 1, 0, len(frac) - 1)]

    # Add the main body on the same steps, continuous through fragmentations
    if len(tc) < len(t):
        # From MetSim's steps to the output grid
        m_tot = np.interp(t, tc, main_m[n_sub - 1::n_sub][:len(tc)] + rest)
    else:
        # Steps already on the output grid
        m_tot = main_m + rest

    # Values MetSim records in the constants
    setRunConstants(const, info, path, t, mt, ml, mv, mm, main_h, alive)

    # Light of the main body and of the fragmentation entries
    lum_main = info['lum_main']
    lum_entries = info['lum_entries']

    # The 21 columns of MetSim's results
    tau = np.full(n, np.nan)
    zeros = np.zeros(n)
    results = np.column_stack([t, lum, lum_main, lum - lum_main - lum_entries, zeros, tau, tau, tau, h_lead, \
        l_lead, v_lead, h_lead, l_lead, v_lead, rho*v_lead**2, m_tot, main_m, main_h, main_l, main_v, main_p])

    # The main fragment at its end
    frag_main = MainFragment(float(main_m[alive][-1]) if np.any(alive) else 0.0, float(main_v[alive][-1]) \
        if np.any(alive) else 0.0, float(main_h[alive][-1]) if np.any(alive) else 0.0)

    return frag_main, results, [None]*n
