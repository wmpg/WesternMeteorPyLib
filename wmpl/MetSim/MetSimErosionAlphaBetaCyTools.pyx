""" Fast Cython companion for MetSimErosionAlphaBeta.py's grain-population RK4 stepping.

Profiling (see the implementation plan's own write-up) found _stepGrainPopulationRK4()'s per-tick
loop - already vectorized across the whole active grain population via numpy - spends most of its
time not on the underlying arithmetic (cheap) but on repeated numpy-level dispatch/temporary-array
allocation: each tick calls ~15 separate numpy functions (_atmDensityPolyVec, _massLossRK4Vec,
_decelerationRK4Vec, several np.where, the luminosity/ionization formulas), each with real per-call
overhead, over up to ~1000-2000 ticks per simulation. This module fuses all of that per-tick, per-
active-grain physics into ONE typed C loop per tick - no intermediate numpy temporaries, no repeated
Python-level dispatch - while leaving the OUTER per-tick bookkeeping (spawn detection, alive/done
tracking, output accumulation) in MetSimErosionAlphaBeta.py exactly as validated, since that part
was never the bottleneck.

massLoss/deceleration/atmDensityPoly/ionizationEfficiency formulas are transcribed directly from
MetSimErosionCyTools.pyx (not re-derived, not imported - a cross-module cimport would need a .pxd
this project doesn't have, and importing the cpdef wrappers as plain Python callables would reintroduce
per-grain-per-tick Python call overhead, defeating the point) - see each helper's own docstring.
luminousEfficiency (9 branching physical models) is the one exception: for the default
lum_eff_type=0 (constant efficiency) it is trivial and inlined directly; every other type falls back
to calling the real MetSimErosionCyTools.luminousEfficiency() per grain, matching
_stepGrainPopulationRK4()'s own pre-existing "fast path type 0, Python-loop fallback otherwise"
convention - not optimized further here, since every scenario validated in this project uses the
default type.
"""

import cython
cimport cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, log10

from wmpl.MetSim.MetSimErosionCyTools import luminousEfficiency as _luminousEfficiencyPy


FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t


@cython.cdivision(True)
cdef inline double _atmDensityPoly(double ht, double* dens_co) nogil:
    """ Transcribed from MetSimErosionCyTools.atmDensityPoly() - identical formula. """

    cdef double x = ht/1e6
    return pow(10.0, dens_co[0] + dens_co[1]*x + dens_co[2]*x*x + dens_co[3]*x*x*x
        + dens_co[4]*x*x*x*x + dens_co[5]*x*x*x*x*x + dens_co[6]*x*x*x*x*x*x)


@cython.cdivision(True)
cdef inline double _massLoss(double K, double sigma, double m, double rho_atm, double v) nogil:
    """ Transcribed from MetSimErosionCyTools.massLoss(). """

    return -K*sigma*pow(m, 2.0/3.0)*rho_atm*v*v*v


@cython.cdivision(True)
cdef inline double _massLossRK4(double dt, double K, double sigma, double m, double rho_atm,
        double v) nogil:
    """ Transcribed from MetSimErosionCyTools.massLossRK4() - same RK4-of-the-mass-loss-ODE
    structure and negative-mass clamping at each sub-stage. """

    cdef double mk1, mk2, mk3, mk4

    mk1 = dt*_massLoss(K, sigma, m, rho_atm, v)
    if -mk1/2.0 > m:
        mk1 = -m*2.0

    mk2 = dt*_massLoss(K, sigma, m + mk1/2.0, rho_atm, v)
    if -mk2/2.0 > m:
        mk2 = -m*2.0

    mk3 = dt*_massLoss(K, sigma, m + mk2/2.0, rho_atm, v)
    if -mk3 > m:
        mk3 = -m

    mk4 = dt*_massLoss(K, sigma, m + mk3, rho_atm, v)

    return mk1/6.0 + mk2/3.0 + mk3/3.0 + mk4/6.0


@cython.cdivision(True)
cdef inline double _deceleration(double K, double m, double rho_atm, double v) nogil:
    """ Transcribed from MetSimErosionCyTools.deceleration(). """

    return -K*pow(m, -1.0/3.0)*rho_atm*v*v


@cython.cdivision(True)
cdef inline double _decelerationRK4(double dt, double K, double m, double rho_atm, double v) nogil:
    """ Transcribed from MetSimErosionCyTools.decelerationRK4(). """

    cdef double vk1, vk2, vk3, vk4

    vk1 = dt*_deceleration(K, m, rho_atm, v)
    vk2 = dt*_deceleration(K, m, rho_atm, v + vk1/2.0)
    vk3 = dt*_deceleration(K, m, rho_atm, v + vk2/2.0)
    vk4 = dt*_deceleration(K, m, rho_atm, v + vk3)

    return (vk1/6.0 + vk2/3.0 + vk3/3.0 + vk4/6.0)/dt


@cython.cdivision(True)
cdef inline double _ionizationEfficiency(double vel) nogil:
    """ Transcribed from MetSimErosionCyTools.ionizationEfficiency(), with the same floor at 1 m/s
    _ionizationEfficiencyVec() (MetSimErosionAlphaBeta.py) already applies, to keep log10() finite
    for a near-zero velocity - only ever called here for v_new > 0 (see the caller below), so this
    floor is a defensive match to that vectorized port's own convention rather than a live path. """

    cdef double vel_km = vel/1000.0
    if vel_km < 1e-3:
        vel_km = 1e-3

    return pow(10.0, 5.84 - 0.09*sqrt(vel_km) - 9.56/log10(vel_km))


@cython.boundscheck(False)
@cython.wraparound(False)
def stepGrainPopulationTick(
        np.ndarray[FLOAT_TYPE_t, ndim=1] h_a not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] v_a not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] m_a not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] vv_a not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] vh_a not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] length_a not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] hgd_a not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] curvature_offset_a not None,
        np.ndarray[np.uint8_t, ndim=1, cast=True] need_refresh not None,
        double K, double sigma, double dt, double cos_zenith, double sin_zenith,
        double r_earth, np.ndarray[FLOAT_TYPE_t, ndim=1] dens_co not None,
        double G0, double mu, double h_init,
        int lum_eff_type, double lum_eff,
        double m_kill, double v_kill, double h_kill, double len_kill):
    """ One GLOBAL TICK of RK4 physics for the whole currently-active grain-population subset,
    fused into a single typed loop - the direct Cython replacement for the ~15 separate numpy calls
    _stepGrainPopulationRK4() (MetSimErosionAlphaBeta.py) used to make per tick. Every input array
    is the ACTIVE-SUBSET state already gathered by the caller (idx = np.where(alive)[0]) - this
    function does not know about spawn/death bookkeeping at all, only "given N grains' current
    state, advance them one dt". Formulas and branch structure replicate
    _stepGrainPopulationRK4()'s own numpy version and, beneath that, _stepGrainRK4()'s scalar
    version EXACTLY (same operation order within float-precision reordering noise - see
    _stepGrainPopulationRK4()'s own docstring for the ~1e-11 relative tolerance already accepted
    between the numpy and scalar versions; this Cython version is held to the same bar).

    Arguments mirror _stepGrainPopulationRK4()'s own active-subset arrays 1:1, plus dt/geometry/
    atmosphere/luminous-efficiency scalars passed through instead of a Constants object (Cython
    cannot introspect an arbitrary Python object's attributes without per-access Python overhead,
    which would defeat the point of fusing this loop).

    Return:
        (v_new, m_new, h_new, lum, q, dyn_press, length_new, vv_new, vh_new, hgd_new,
        curvature_offset_new, kill): each an ndarray of the same length as the inputs. kill is
        uint8 (0/1) - the caller casts to bool.
    """

    cdef Py_ssize_t n = h_a.shape[0]
    cdef Py_ssize_t i

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] v_new = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] m_new = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] h_new = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] lum = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] q = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dyn_press = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] length_new = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] vv_new = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] vh_new = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] hgd_new = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] curvature_offset_new = np.empty(n, dtype=FLOAT_TYPE)
    cdef np.ndarray[np.uint8_t, ndim=1] kill = np.empty(n, dtype=np.uint8)

    cdef double dens_co_arr[7]
    for i in range(7):
        dens_co_arr[i] = dens_co[i]

    cdef double h_i, v_i, m_i, vv_i, vh_i, length_i, hgd_i, curv_i
    cdef double rho_atm, mass_loss_ablation, m_new_i, decel, gv, av, ah, vv_new_i, vh_new_i, v_new_i
    cdef double hgd_new_i, length_new_i, curv_new_i, tau, beta_ion, lum_i, q_i, dp_i
    cdef bint accelerating

    for i in range(n):

        h_i = h_a[i]; v_i = v_a[i]; m_i = m_a[i]
        vv_i = vv_a[i]; vh_i = vh_a[i]; length_i = length_a[i]; hgd_i = hgd_a[i]

        rho_atm = _atmDensityPoly(h_i, dens_co_arr)

        mass_loss_ablation = _massLossRK4(dt, K, sigma, m_i, rho_atm, v_i)
        m_new_i = m_i + mass_loss_ablation
        if m_new_i < 0.0:
            m_new_i = 0.0

        # decelerationRK4() is evaluated on the PRE-update mass m_i, not m_new_i - matches
        # _stepGrainRK4()/_stepGrainPopulationRK4()'s own exact ordering.
        decel = _decelerationRK4(dt, K, m_i, rho_atm, v_i)

        accelerating = decel > 0.0

        if accelerating:
            vv_new_i = 0.0
            vh_new_i = 0.0
            v_new_i = 0.0
            hgd_new_i = hgd_i
            decel = 0.0
        else:
            gv = G0/((1.0 + h_i/r_earth)*(1.0 + h_i/r_earth))
            av = -decel*vv_i/v_i + vh_i*v_i/(r_earth + h_i)
            ah = -decel*vh_i/v_i - vv_i*v_i/(r_earth + h_i)
            hgd_new_i = hgd_i + 0.5*gv*dt*dt
            vv_new_i = vv_i - av*dt
            vh_new_i = vh_i - ah*dt
            v_new_i = sqrt(vh_new_i*vh_new_i + vv_new_i*vv_new_i)
            if vv_new_i > 0.0:
                vv_new_i = 0.0

        length_new_i = length_i + v_new_i*dt

        if need_refresh[i]:
            curv_new_i = (sqrt((h_init + r_earth)*(h_init + r_earth)
                - 2.0*length_new_i*cos_zenith*(h_init + r_earth) + length_new_i*length_new_i)
                - r_earth) - (h_init - length_new_i*cos_zenith)
        else:
            curv_new_i = curvature_offset_a[i]

        h_new_i = h_init - length_new_i*cos_zenith + curv_new_i - hgd_new_i

        if lum_eff_type == 0:
            tau = lum_eff/100.0
        else:
            tau = _luminousEfficiencyPy(lum_eff_type, lum_eff, v_new_i, m_new_i)

        lum_i = -tau*((mass_loss_ablation/dt*v_new_i*v_new_i)/2.0 + m_new_i*v_new_i*decel)

        if v_new_i > 0.0:
            beta_ion = _ionizationEfficiency(v_new_i)
            q_i = -beta_ion*(mass_loss_ablation/dt)/(mu*v_new_i)
        else:
            q_i = 0.0

        dp_i = rho_atm*v_new_i*v_new_i

        v_new[i] = v_new_i; m_new[i] = m_new_i; h_new[i] = h_new_i
        lum[i] = lum_i; q[i] = q_i; dyn_press[i] = dp_i; length_new[i] = length_new_i
        vv_new[i] = vv_new_i; vh_new[i] = vh_new_i; hgd_new[i] = hgd_new_i
        curvature_offset_new[i] = curv_new_i

        if (m_new_i <= m_kill) or (v_new_i < v_kill) or (h_new_i < h_kill) or (lum_i < 0.0) \
                or ((len_kill > 0.0) and (length_new_i > len_kill)):
            kill[i] = 1
        else:
            kill[i] = 0

    return (v_new, m_new, h_new, lum, q, dyn_press, length_new, vv_new, vh_new, hgd_new,
        curvature_offset_new, kill)


@cython.cdivision(True)
cdef inline double _heightCurvatureCos(double h0, double cos_zc, double l, double r_earth) nogil:
    """ heightCurvature(), with cos(zenith_angle) precomputed once by the caller instead of
    recomputed every call - see MetSimErosionAlphaBeta.heightCurvature() for the reference
    formula this is transcribed from. """

    return sqrt((h0 + r_earth)*(h0 + r_earth) - 2.0*l*cos_zc*(h0 + r_earth) + l*l) - r_earth


@cython.boundscheck(False)
@cython.wraparound(False)
def stepGrainPopulationFull(
        np.ndarray[FLOAT_TYPE_t, ndim=1] m0 not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] v0 not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] t0 not None,
        np.ndarray[FLOAT_TYPE_t, ndim=1] length0 not None,
        double K, double sigma, double dt, double cos_zenith, double sin_zenith,
        double r_earth, np.ndarray[FLOAT_TYPE_t, ndim=1] dens_co not None,
        double G0, double mu, double h_init,
        int lum_eff_type, double lum_eff,
        double m_kill, double v_kill, double h_kill, double len_kill,
        long max_steps_per_grain):
    """ Full replacement for _stepGrainPopulationRK4()'s ENTIRE per-tick loop - not just the
    per-tick physics (stepGrainPopulationTick() above), but also the spawn-detection/active-set/
    output-accumulation bookkeeping that used to stay in Python/numpy. Profiling (see the
    implementation plan's own write-up) found that OUTER bookkeeping became the single largest
    remaining cost in this engine once the inner-tick physics was already fused into Cython (~27%
    of a whole benchmark scenario's wall-clock) - and a pure-numpy attempt to reduce it (a compact
    "currently active" array instead of full-population boolean scans) barely moved the needle
    (~1.06x on the largest real call measured), showing the real cost was per-tick PYTHON LOOP
    ITERATION COUNT itself (hundreds of ticks, each with ~10 small numpy calls carrying real fixed
    dispatch overhead), not the per-tick array size. This function eliminates that entirely: one
    Cython call replaces the WHOLE while-loop, with no Python-level statement executed per tick at
    all.

    Architecture:
    - Grains are pre-sorted by their own spawn tick (k_spawn) ONCE, via numpy's own argsort (a
      single O(N log N) call - not reimplemented in Cython, numpy's own sort is already optimal C
      code) - so per-tick spawn detection becomes an O(1)-amortized pointer advance through that
      sorted order (each grain visited exactly once across the whole run), not a per-tick scan.
    - The "currently active" population is a fixed-CAPACITY (size N, an exact upper bound - at
      most every grain can be active at once) set of C arrays, compacted IN PLACE each tick:
      grains that die are dropped by simply not copying their new state forward (a single forward
      pass with a read index i and a write index j <= i - see the loop body for why this has no
      aliasing hazard), no separate boolean-mask allocation needed at all.
    - Output rows (one per (grain, tick) pair actually computed, exactly matching
      _stepGrainPopulationRK4()'s own return contract) are accumulated into growable buffers:
      pre-allocated at a generous initial guess, DOUBLED (new array, old contents copied in) only
      on the rare occasion that guess is exceeded - the standard amortized-O(1)-per-element
      dynamic-array growth pattern, avoiding both a wasteful worst-case pre-allocation (N times
      max_steps_per_grain could be enormous) and a per-tick reallocation.

    Arguments: same physical/Constants scalars as stepGrainPopulationTick() above (Constants
    object not passed - see that function's own docstring for why) plus max_steps_per_grain (that
    function's own hardcoded int(10.0/dt) safety cap, computed by the caller once) and m0/v0/t0/
    length0 (one entry per grain, same convention as _stepGrainPopulationRK4()'s own arguments).

    Return:
        (global_tick_idx, v, m, h, lum, q, dyn_press, length, is_death_tick, grain_id): the SAME
        10 arrays _stepGrainPopulationRK4() itself computes directly - grain_id indexes into the
        ORIGINAL m0/v0/t0/length0 arrays (not into n_grains - the caller applies n_grains[grain_id]
        itself, matching _stepGrainPopulationRK4()'s own existing final-return convention, since
        n_grains is not needed by any of this function's own physics). Validated bit-for-bit-
        equivalent to _stepGrainPopulationRK4() by test_step_grain_population_rk4_matches_scalar_reference()
    (wmpl/MetSim/Tests/test_MetSimErosionAlphaBeta.py).
    """

    cdef Py_ssize_t N = m0.shape[0]

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] empty_f = np.empty(0, dtype=FLOAT_TYPE)
    cdef np.ndarray[np.int64_t, ndim=1] empty_i = np.empty(0, dtype=np.int64)
    cdef np.ndarray[np.uint8_t, ndim=1] empty_b = np.empty(0, dtype=np.uint8)

    if N == 0:
        return (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_b,
            empty_i)

    cdef double dens_co_arr[7]
    cdef Py_ssize_t di
    for di in range(7):
        dens_co_arr[di] = dens_co[di]

    cdef np.ndarray[np.int64_t, ndim=1] k_spawn = np.round(
        np.asarray(t0, dtype=FLOAT_TYPE)/dt).astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] spawn_order = np.argsort(k_spawn, kind="stable").astype(
        np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] k_spawn_sorted = k_spawn[spawn_order]

    cdef long k_min = <long>k_spawn.min()
    cdef long k_max = <long>k_spawn.max()
    cdef long k_deadline = k_max + max_steps_per_grain

    # Active-set state - fixed capacity N (an exact upper bound), compacted in place each tick.
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] a_h = np.empty(N, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] a_v = np.empty(N, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] a_m = np.empty(N, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] a_vv = np.empty(N, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] a_vh = np.empty(N, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] a_length = np.empty(N, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] a_hgd = np.empty(N, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] a_curv = np.empty(N, dtype=FLOAT_TYPE)
    cdef np.ndarray[np.int64_t, ndim=1] a_steps = np.empty(N, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] a_gid = np.empty(N, dtype=np.int64)
    cdef Py_ssize_t n_active = 0

    # Output buffers - growable (doubled on overflow, see this function's own docstring).
    cdef Py_ssize_t out_cap = N*8 if N*8 > 64 else 64
    cdef np.ndarray[np.int64_t, ndim=1] out_idx = np.empty(out_cap, dtype=np.int64)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] out_v = np.empty(out_cap, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] out_m = np.empty(out_cap, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] out_h = np.empty(out_cap, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] out_lum = np.empty(out_cap, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] out_q = np.empty(out_cap, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] out_dp = np.empty(out_cap, dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] out_len = np.empty(out_cap, dtype=FLOAT_TYPE)
    cdef np.ndarray[np.uint8_t, ndim=1] out_last = np.empty(out_cap, dtype=np.uint8)
    cdef np.ndarray[np.int64_t, ndim=1] out_gid = np.empty(out_cap, dtype=np.int64)
    cdef Py_ssize_t out_n = 0

    cdef Py_ssize_t spawn_ptr = 0
    cdef long k = k_min
    cdef Py_ssize_t i, j, gid
    cdef double h_i, v_i, m_i, vv_i, vh_i, length_i, hgd_i, curv_i, length_new_i
    cdef double rho_atm, mass_loss_ablation, m_new_i, decel, gv, av, ah, vv_new_i, vh_new_i, v_new_i
    cdef double hgd_new_i, curv_new_i, h_new_i, tau, beta_ion, lum_i, q_i, dp_i
    cdef bint accelerating, need_refresh_i, kill_i
    cdef long steps_i
    cdef Py_ssize_t new_cap
    cdef np.ndarray new_buf

    while k < k_deadline:
        k += 1

        while spawn_ptr < N and k_spawn_sorted[spawn_ptr] == k - 1:
            gid = spawn_order[spawn_ptr]
            spawn_ptr += 1

            length_i = length0[gid]
            curv_i = (_heightCurvatureCos(h_init, cos_zenith, length_i, r_earth)
                - (h_init - length_i*cos_zenith))

            a_v[n_active] = v0[gid]
            a_vv[n_active] = -v0[gid]*cos_zenith
            a_vh[n_active] = v0[gid]*sin_zenith
            a_m[n_active] = m0[gid]
            a_length[n_active] = length_i
            a_hgd[n_active] = 0.0
            a_curv[n_active] = curv_i
            a_h[n_active] = h_init - length_i*cos_zenith + curv_i
            a_steps[n_active] = 0
            a_gid[n_active] = gid
            n_active += 1

        if n_active == 0:
            if k > k_max:
                break
            continue

        if out_n + n_active > out_cap:
            new_cap = out_cap*2
            if new_cap < out_n + n_active:
                new_cap = out_n + n_active

            new_buf = np.empty(new_cap, dtype=np.int64)
            new_buf[:out_n] = out_idx[:out_n]
            out_idx = new_buf
            new_buf = np.empty(new_cap, dtype=FLOAT_TYPE)
            new_buf[:out_n] = out_v[:out_n]
            out_v = new_buf
            new_buf = np.empty(new_cap, dtype=FLOAT_TYPE)
            new_buf[:out_n] = out_m[:out_n]
            out_m = new_buf
            new_buf = np.empty(new_cap, dtype=FLOAT_TYPE)
            new_buf[:out_n] = out_h[:out_n]
            out_h = new_buf
            new_buf = np.empty(new_cap, dtype=FLOAT_TYPE)
            new_buf[:out_n] = out_lum[:out_n]
            out_lum = new_buf
            new_buf = np.empty(new_cap, dtype=FLOAT_TYPE)
            new_buf[:out_n] = out_q[:out_n]
            out_q = new_buf
            new_buf = np.empty(new_cap, dtype=FLOAT_TYPE)
            new_buf[:out_n] = out_dp[:out_n]
            out_dp = new_buf
            new_buf = np.empty(new_cap, dtype=FLOAT_TYPE)
            new_buf[:out_n] = out_len[:out_n]
            out_len = new_buf
            new_buf = np.empty(new_cap, dtype=np.uint8)
            new_buf[:out_n] = out_last[:out_n]
            out_last = new_buf
            new_buf = np.empty(new_cap, dtype=np.int64)
            new_buf[:out_n] = out_gid[:out_n]
            out_gid = new_buf

            out_cap = new_cap

        j = 0
        for i in range(n_active):

            h_i = a_h[i]; v_i = a_v[i]; m_i = a_m[i]
            vv_i = a_vv[i]; vh_i = a_vh[i]; length_i = a_length[i]; hgd_i = a_hgd[i]
            curv_i = a_curv[i]; steps_i = a_steps[i]

            rho_atm = _atmDensityPoly(h_i, dens_co_arr)

            mass_loss_ablation = _massLossRK4(dt, K, sigma, m_i, rho_atm, v_i)
            m_new_i = m_i + mass_loss_ablation
            if m_new_i < 0.0:
                m_new_i = 0.0

            decel = _decelerationRK4(dt, K, m_i, rho_atm, v_i)
            accelerating = decel > 0.0

            if accelerating:
                vv_new_i = 0.0
                vh_new_i = 0.0
                v_new_i = 0.0
                hgd_new_i = hgd_i
                decel = 0.0
            else:
                gv = G0/((1.0 + h_i/r_earth)*(1.0 + h_i/r_earth))
                av = -decel*vv_i/v_i + vh_i*v_i/(r_earth + h_i)
                ah = -decel*vh_i/v_i - vv_i*v_i/(r_earth + h_i)
                hgd_new_i = hgd_i + 0.5*gv*dt*dt
                vv_new_i = vv_i - av*dt
                vh_new_i = vh_i - ah*dt
                v_new_i = sqrt(vh_new_i*vh_new_i + vv_new_i*vv_new_i)
                if vv_new_i > 0.0:
                    vv_new_i = 0.0

            length_new_i = length_i + v_new_i*dt

            steps_i += 1
            need_refresh_i = steps_i >= 20

            if need_refresh_i:
                curv_new_i = (_heightCurvatureCos(h_init, cos_zenith, length_new_i, r_earth)
                    - (h_init - length_new_i*cos_zenith))
                steps_i = 0
            else:
                curv_new_i = curv_i

            h_new_i = h_init - length_new_i*cos_zenith + curv_new_i - hgd_new_i

            if lum_eff_type == 0:
                tau = lum_eff/100.0
            else:
                tau = _luminousEfficiencyPy(lum_eff_type, lum_eff, v_new_i, m_new_i)

            lum_i = -tau*((mass_loss_ablation/dt*v_new_i*v_new_i)/2.0 + m_new_i*v_new_i*decel)

            if v_new_i > 0.0:
                beta_ion = _ionizationEfficiency(v_new_i)
                q_i = -beta_ion*(mass_loss_ablation/dt)/(mu*v_new_i)
            else:
                q_i = 0.0

            dp_i = rho_atm*v_new_i*v_new_i

            kill_i = ((m_new_i <= m_kill) or (v_new_i < v_kill) or (h_new_i < h_kill)
                or (lum_i < 0.0) or ((len_kill > 0.0) and (length_new_i > len_kill)))

            out_idx[out_n] = k - 1
            out_v[out_n] = v_new_i
            out_m[out_n] = m_new_i
            out_h[out_n] = h_new_i
            out_lum[out_n] = lum_i
            out_q[out_n] = q_i
            out_dp[out_n] = dp_i
            out_len[out_n] = length_new_i
            out_last[out_n] = 1 if kill_i else 0
            out_gid[out_n] = a_gid[i]
            out_n += 1

            if not kill_i:
                a_h[j] = h_new_i; a_v[j] = v_new_i; a_m[j] = m_new_i
                a_vv[j] = vv_new_i; a_vh[j] = vh_new_i; a_length[j] = length_new_i
                a_hgd[j] = hgd_new_i; a_curv[j] = curv_new_i; a_steps[j] = steps_i
                a_gid[j] = a_gid[i]
                j += 1

        n_active = j

    return (out_idx[:out_n], out_v[:out_n], out_m[:out_n], out_h[:out_n], out_lum[:out_n],
        out_q[:out_n], out_dp[:out_n], out_len[:out_n], out_last[:out_n], out_gid[:out_n])
