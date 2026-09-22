""" Implementation of the Borovicka (2007) meteor erosion model with added disruption.

References:
    Borovička, J., Spurný, P., & Koten, P. (2007). Atmospheric deceleration and light curves of Draconid 
    meteors and implications for the structure of cometary dust. Astronomy & Astrophysics, 473(2), 661-672.

    Campbell-Brown, M. D., Borovička, J., Brown, P. G., & Stokan, E. (2013). High-resolution modelling of 
    meteoroid ablation. Astronomy & Astrophysics, 557, A41.

"""

from __future__ import print_function, division, absolute_import


import math

import numpy as np
import scipy.stats
import scipy.integrate
import scipy.special
import scipy.optimize
import scipy.interpolate

# LUT-accelerated inverse of Ei() - see VelocitySpline's own docstring for how this is used here.
# The ONLY piece reused from wmpl.Utils.AlphaBeta (deliberately not otherwise imported - see
# HT_NORM_CONST's own comment below): getDefaultInverseEiLUT() inverts scipy.special.expi() in a
# way that depends only on the value being inverted, not on (alpha, beta) or on which higher-level
# formula produced it - equally valid for this module's own generalized (finite-start) Delta(v_n)
# as for AlphaBeta.py's own vacuum-start one.
from wmpl.Utils.AlphaBeta import getDefaultInverseEiLUT


# Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from wmpl.MetSim.MetSimErosionCyTools import massLossRK4, decelerationRK4, luminousEfficiency, \
    ionizationEfficiency, atmDensityPoly
from wmpl.MetSim.MetSimErosionAlphaBetaCyTools import stepGrainPopulationTick, \
    stepGrainPopulationFull


### DEFINE CONSTANTS

# Earth acceleration in m/s^2 on the surface
G0 = 9.81

# Exponential atmosphere scale height (m) the analytic alpha-beta solution is derived under.
# Matches wmpl.Utils.AlphaBeta.HT_NORM_CONST (kept as an independent constant here, since this
# module does not import from wmpl.Utils.AlphaBeta - see the analytic-engine section below).
HT_NORM_CONST = 7160.0

# Sea-level reference air density (kg/m^3) used to anchor the atmosphere-equivalent-height map.
RHO_ATM_0 = 1.225

###


### ANALYTIC ALPHA-BETA ENGINE (fast, closed-form replacement for the RK4 integration below)
#
# See the implementation plan at
# /Users/eloy.peas/.claude/plans/converting-each-concurrent-eager-thimble.md for full derivations
# and validation results. Summary of the two corrections beyond wmpl.Utils.AlphaBeta.py's existing
# (alpha, beta) formulation, both required for this to exactly reproduce the RK4 model above:
#
# 1) mu = 2/3 (not 0): the RK4 model's ODEs (massLoss/deceleration in MetSimErosionCyTools.pyx),
#    with K = gamma*shape_factor*rho**(-2/3) held constant within a segment, correspond to
#    beta = sigma_eff * v_start**2 / 6 (NOT /2 - that's mu=0, a different, wrong physical
#    assumption for this model). Verified analytically (full Ei-substitution re-derivation) and
#    numerically (RK4 comparison): the /2 form diverges to a spurious full stop within ~10-15km of
#    the segment start; the /6 form tracks RK4 to <0.01% through most of the deceleration range.
#
# 2) Finite-start closed form: alphaBetaHeightNormed()/alphaBetaVelocityNormed() in AlphaBeta.py
#    assume the segment starts where the atmosphere is negligible (v_n=1 <=> height=+infinity).
#    True for the main fragment's first segment, false for every fragment/grain spawned mid-flight
#    (erosion, disruption, complex fragmentation) at a finite starting height. The generalized,
#    non-singular closed form below carries an explicit h_equiv_start term and reduces exactly to
#    the standard formula as h_equiv_start -> infinity (verified to 1e-12 m against
#    alphaBetaHeightNormed()/alphaBetaVelocityNormed() in that limit).
#
# 3) Atmosphere reconciliation: the closed form assumes a pure exponential atmosphere
#    (rho = RHO_ATM_0*exp(-h_equiv/HT_NORM_CONST)); atmDensityPoly() above is a real, non-
#    exponential (polynomial-log) atmosphere fit. The height variable used throughout this engine
#    is therefore not real height, but an "equivalent height" h_equiv obtained by matching
#    CUMULATIVE COLUMN DENSITY (integral of rho over height), not pointwise density value.
#    Pointwise matching (the approach used by wmpl.Utils.AlphaBeta.rescaleHeightToExponentialAtmosphere(),
#    which is adequate for that function's purpose of pre-processing data ahead of a numerical
#    curve fit) does NOT satisfy the chain rule the closed-form ODE solution needs to exactly
#    reproduce the true trajectory: dv/dh_equiv only equals the standard exponential-atmosphere
#    expression if d(h_equiv)/d(h_real) = rho_real(h_real)/rho_exp(h_equiv), which only holds when
#    h_equiv is defined via the column integral. Verified empirically: pointwise matching gave
#    velocity errors up to several percent (growing with depth into the trajectory, confirmed via
#    RK4 dt-refinement to not be a discretization artifact); column-density matching brought this
#    down to <0.01% through the great majority of the deceleration range in every test case.
#
# The column-density integral has no closed form for the degree-6 log-polynomial atmDensityPoly(),
# so it is precomputed once per Constants instance (per event/dens_co) on a grid via numerical
# quadrature, and evaluated cheaply afterwards through a monotonic spline (AtmEquivHeightMap).


def _atmColumnDensity(h, dens_co, h_ref):
    """ Cumulative atmospheric column density (kg/m^2) between height h and a fixed reference
    height h_ref (integral of atmDensityPoly from h to h_ref).

    Only ever used in differences (S(h) - S(h0)) by the analytic engine, so the arbitrary
    reference h_ref cancels out algebraically - it must simply be held fixed for an entire
    simulation/dens_co, and must not be lower than any height ever queried (to avoid needing to
    extrapolate atmDensityPoly's polynomial fit, which can behave wildly outside its fitted range).

    Arguments:
        h: [float] Height (m).
        dens_co: [ndarray] Atmosphere density polynomial coefficients (see atmDensityPoly()).
        h_ref: [float] Fixed reference height (m), >= every height ever queried for this dens_co.

    Return:
        [float] Column density from h to h_ref (kg/m^2).
    """

    return scipy.integrate.quad(atmDensityPoly, h, h_ref, args=(dens_co,), limit=200)[0]


class AtmEquivHeightMap(object):
    """ Precomputed, fast forward/inverse mapping between real height and the exponential-
    atmosphere-equivalent height (h_equiv) the analytic alpha-beta engine operates in, for one
    fixed atmosphere profile (dens_co). Built once per Constants instance/event and reused cheaply
    for every fragment and every query in that simulation.
    """

    def __init__(self, dens_co, h_ref, h_min=20000.0, n_grid=400):
        """
        Arguments:
            dens_co: [ndarray] Atmosphere density polynomial coefficients.
            h_ref: [float] Fixed column-density reference height (m) - must be >= the highest
                height ever queried (typically Constants.h_init) and within dens_co's valid range.

        Keyword arguments:
            h_min: [float] Lowest height (m) the map is built down to.
            n_grid: [int] Number of grid points used to build the interpolating spline.
        """

        self.dens_co = dens_co
        self.h_ref = h_ref

        # Build the grid finer near the top (where most of the trajectory is spent) using a
        # sqrt-spaced grid biased towards h_ref, but still covering h_min..h_ref densely enough
        # for cubic interpolation to be accurate.
        frac = np.linspace(0.0, 1.0, n_grid)
        h_grid = h_ref - (h_ref - h_min)*frac**1.5

        # Ensure the top of the grid is exactly h_ref (column density is exactly 0 there, so
        # h_equiv would be +infinity - handled as a special case, not included in the spline).
        h_grid = h_grid[h_grid < h_ref - 1e-6]
        h_grid = np.append(h_grid, h_ref - 1e-6)
        h_grid = np.unique(h_grid)[::-1]   # descending height, for readability only

        S_grid = np.array([_atmColumnDensity(h, dens_co, h_ref) for h in h_grid])

        # h_equiv such that RHO_ATM_0*HT_NORM_CONST*exp(-h_equiv/HT_NORM_CONST) = S(h)
        h_equiv_grid = -HT_NORM_CONST*np.log(S_grid/(RHO_ATM_0*HT_NORM_CONST))

        # Both arrays are monotonically increasing with h (S increases as h decreases, so
        # h_equiv increases as h increases) - sort ascending by h_real for the interpolators.
        order = np.argsort(h_grid)
        self._h_real_grid = h_grid[order]
        self._h_equiv_grid = h_equiv_grid[order]

        self._to_equiv = scipy.interpolate.PchipInterpolator(self._h_real_grid, self._h_equiv_grid,
            extrapolate=True)
        self._to_real = scipy.interpolate.PchipInterpolator(self._h_equiv_grid, self._h_real_grid,
            extrapolate=True)


    def toEquiv(self, h_real):
        """ Real height (m) -> equivalent height (m) the analytic engine operates in. """

        return self._to_equiv(h_real)


    def toReal(self, h_equiv):
        """ Equivalent height (m) -> real height (m). Only needed once, when producing final
        real-height output (the analytic engine's internal math only needs the forward direction).
        """

        return self._to_real(h_equiv)


def _deltaEi(beta, v_n):
    """ Delta(v_n) = Ei(beta) - Ei(beta*v_n^2), the core quantity of Gritsevich (2009) Eq. (7) -
    see wmpl.Utils.AlphaBeta.alphaBetaHeightNormed(). """

    return scipy.special.expi(beta) - scipy.special.expi(beta*np.asarray(v_n)**2)


def alphaFromPhysical(K, sin_slope, m_start, rho_atm_0=RHO_ATM_0):
    """ Ballistic coefficient alpha from physical parameters. Matches
    wmpl.Utils.AlphaBeta.alphaBetaMasses()'s alpha formula exactly when evaluated with this
    fragment's own K = gamma*shape_factor*rho**(-2/3) (NOT AlphaBeta.py's generic literature
    default gamma*shape_coeff=0.55 - using the real K reparameterizes the same physics exactly,
    rather than approximating it with a different default).

    Arguments:
        K: [float] Shape-density coefficient (m^2/kg^(2/3)), gamma*shape_factor*rho**(-2/3).
        sin_slope: [float] sin(entry slope from horizontal) = cos(zenith_angle).
        m_start: [float] Mass at the start of this segment (kg).

    Keyword arguments:
        rho_atm_0: [float] Sea-level reference air density (kg/m^3).

    Return:
        [float] alpha (dimensionless).
    """

    return K*rho_atm_0*HT_NORM_CONST/(sin_slope*m_start**(1/3.0))


def betaFromPhysical(sigma_eff, v_start):
    """ Mass-loss parameter beta from the physical ablation coefficient, under the mu=2/3
    (self-similar shrinking body) assumption that matches this model's constant-K-per-segment
    ODEs exactly - see the module-level notes above. NOT sigma_eff*v_start**2/2 (that is the
    mu=0 formula, a different physical assumption that does not match this model).

    Arguments:
        sigma_eff: [float] Effective ablation coefficient for this segment (s^2/m^2). Equal to
            sigma while not eroding, sigma + erosion_coeff while eroding (see module notes).
        v_start: [float] Velocity at the start of this segment (m/s).

    Return:
        [float] beta (dimensionless).
    """

    return sigma_eff*v_start**2/6.0


def hEquivFromVn(v_n, alpha, beta, h_equiv_start):
    """ Generalized (finite-start) form of Gritsevich (2009) Eq. (7): equivalent height as a
    function of normalized velocity v_n = v/v_start, for a segment that begins at v_n=1,
    h_equiv=h_equiv_start (rather than assuming h_equiv_start -> infinity). Reduces exactly to
    wmpl.Utils.AlphaBeta.alphaBetaHeightNormed()*HT_NORM_CONST as h_equiv_start -> infinity
    (verified to 1e-12 m).

    Arguments:
        v_n: [float or ndarray] Velocity normalized to this segment's v_start, in (0, 1).
        alpha: [float] Ballistic coefficient (see alphaFromPhysical()).
        beta: [float] Mass-loss parameter (see betaFromPhysical()).
        h_equiv_start: [float] Equivalent height (m) at the start of this segment (v_n=1).

    Return:
        [float or ndarray] Equivalent height (m).
    """

    delta = _deltaEi(beta, v_n)

    return -HT_NORM_CONST*np.log(np.exp(-h_equiv_start/HT_NORM_CONST) + delta*np.exp(-beta)/(2*alpha))


def vnFromHEquiv(h_equiv, alpha, beta, h_equiv_start, v_eps=1e-10, bracket=None, xtol=1e-6):
    """ Invert hEquivFromVn() for the normalized velocity, given an equivalent height. Non-singular
    at v_n=1 (h_equiv=h_equiv_start exactly there), unlike the standard (infinite-start) inverse.
    Values outside the invertible range are clipped to v_eps/(1 - v_eps), mirroring
    wmpl.Utils.AlphaBeta.alphaBetaVelocityNormed()'s convention.

    Arguments:
        h_equiv: [float] Equivalent height (m) to evaluate at.
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass-loss parameter.
        h_equiv_start: [float] Equivalent height (m) at the start of this segment (v_n=1).

    Keyword arguments:
        v_eps: [float] Bracket margin around (0, 1).
        bracket: [tuple or None] Precomputed (h_hi, h_lo) = (hEquivFromVn(1-v_eps, ...),
            hEquivFromVn(v_eps, ...)) for this (alpha, beta, h_equiv_start) triple. These only
            depend on the segment, not on h_equiv, so when calling this function many times for
            the same segment (as every real caller does - a segment is evaluated at many output
            times/heights), computing them once and passing them in avoids two wasted
            scipy.special.expi evaluations per call. Computed internally if not given.
        xtol: [float] Absolute tolerance on v_n passed to scipy.optimize.brentq. The model's own
            physical accuracy is at the ~0.1-1% level (see Tests/test_MetSimErosionAlphaBeta.py),
            so xtol far below that (default 1e-6) buys no real precision, only wasted brentq
            iterations - each of which costs two more scipy.special.expi evaluations.

    Return:
        [float] Normalized velocity v_n = v/v_start.
    """

    v_lo, v_hi = v_eps, 1.0 - v_eps

    def _g(v_n):
        return hEquivFromVn(v_n, alpha, beta, h_equiv_start) - h_equiv

    if bracket is None:
        h_hi = hEquivFromVn(v_hi, alpha, beta, h_equiv_start)
        h_lo = hEquivFromVn(v_lo, alpha, beta, h_equiv_start)
    else:
        h_hi, h_lo = bracket

    if h_equiv >= h_hi:
        return v_hi

    if h_equiv <= h_lo:
        return v_lo

    return scipy.optimize.brentq(_g, v_lo, v_hi, xtol=xtol)


def velocityBracket(alpha, beta, h_equiv_start, v_eps=1e-10):
    """ Precompute the (h_hi, h_lo) bracket for vnFromHEquiv()'s `bracket` argument, for a given
    segment (alpha, beta, h_equiv_start) - see vnFromHEquiv()'s docstring. Call once per segment,
    reuse for every query height/time against that segment.

    Return:
        (h_hi, h_lo): [tuple of float] Equivalent height at v_n=1-v_eps and v_n=v_eps.
    """

    h_hi = hEquivFromVn(1.0 - v_eps, alpha, beta, h_equiv_start)
    h_lo = hEquivFromVn(v_eps, alpha, beta, h_equiv_start)

    return h_hi, h_lo


class VelocitySpline(object):
    """ Fast inverse of hEquivFromVn() for one fixed (alpha, beta, h_equiv_start) segment:
    v_n(h_equiv). "Fixed segment" describes the INSTANCE's own scope, not shared state - each
    segment genuinely has its own (alpha, beta, h_equiv_start), so a "give me v_n at this h_equiv"
    question has a different answer per segment even when two segments share the underlying
    inversion MACHINERY. Two implementations, chosen automatically at construction time - see the
    PRIMARY path's own paragraph below for the important distinction this class name blurs: the
    process-wide LUT it delegates to is genuinely shared/built-once, but each VelocitySpline
    INSTANCE is still a new, per-segment object (now a near-free one - a few scalars, not a table):

    PRIMARY (beta below _LUT_BETA_MAX, the overwhelming majority of real segments): a genuinely
    CLOSED-FORM inversion via wmpl.Utils.AlphaBeta.getDefaultInverseEiLUT() - the SAME
    Ei(beta) - Ei(beta*v_n^2) relation hEquivFromVn() itself evaluates forward, solved for v_n
    using a process-wide, pre-tabulated inverse of scipy.special.expi() (built once per process,
    shared across every segment/fragment/simulation - see that function's own docstring) instead
    of either scipy.optimize.brentq (vnFromHEquiv(), ~50us/query) or this class's own former
    per-segment Chebyshev-grid-plus-PchipInterpolator spline (still used as the FALLBACK below,
    and still what built the accuracy numbers quoted in the rest of this docstring). Needs NO
    per-segment grid/spline construction at all - only two scalar scipy.special.expi() calls at
    __init__ (self._ei_beta, plus the h_lo/h_hi bracket via hEquivFromVn(), both O(1)) - directly
    removing the ~130-150us/segment PCHIP-fit cost profiled as a real bottleneck in Stage 7
    (`AnalyticTrajectory.__init__` dominating cProfile output via repeated per-segment spline
    construction), not just making each query faster. See _velocityNormedAtLUT()'s own docstring
    for the derivation and the same near-v_n=1 cancellation safeguard
    wmpl.Utils.AlphaBeta.alphaBetaVelocityNormedLUT() already uses.

    Measured, not assumed: in ISOLATION (test_velocity_spline_speedup()) this is a real, substantial
    win - 428x faster than vnFromHEquiv() at a realistic 100-query-point resolution (was 137.9x
    with the PCHIP-only implementation). But a clean, same-process, back-to-back A/B of the FULL
    runSimulation() pipeline (LUT path vs. the OLD PCHIP-only path, forced via _LUT_BETA_MAX) on
    both a single-body and this file's own "complex" scenario found the NET wall-clock difference
    is within measurement noise (ratio 0.98-1.005, no significant change either way) - velocity-
    spline construction was never the DOMINANT cost of a full simulation to begin with (a typical
    run builds dozens of AnalyticTrajectory/VelocitySpline instances - one per segment/daughter -
    not thousands; AtmEquivHeightMap's own atmosphere quadrature, the 4 OTHER PchipInterpolator
    fits per AnalyticTrajectory instance - time/gravity-drop, unrelated to this class - and grain-
    population stepping over thousands of grains all dominate instead). Kept anyway: it is a real,
    validated, mathematically-exact simplification (removes a whole category of per-segment work,
    not just makes it faster) that could matter more in a scenario with proportionally more
    segments/daughters than grains than anything benchmarked so far - just not, currently, the
    dominant lever for this engine's own overall speed. A one-time, ~5-6ms process-wide LUT build
    cost is paid once per process (getDefaultInverseEiLUT()'s own docstring), not per segment.

    FALLBACK (beta >= _LUT_BETA_MAX, a rare, extreme regime - e.g. a Stage 5 "A" event stacking a
    sigma change on top of an already-active erosion_coeff, confirmed to reach beta~38 in this
    engine's own test suite): the ORIGINAL Chebyshev-grid-plus-PchipInterpolator spline, unchanged
    below. Needed because Ei(beta) - Ei(beta*v_n^2) loses precision to direct float64 cancellation
    once Ei(beta) is many orders of magnitude larger than Ei(beta*v_n^2) - a limitation of the
    closed-form/LUT approach specifically (the PCHIP spline never evaluates Ei() at query time at
    all, so it has no such failure mode), not of this model's own physical accuracy. Validated to
    <0.002% velocity error (see the implementation plan / conversation log) across main-fragment,
    grain, fireball, and high-drag-daughter cases, including the deep low-velocity tail, at
    ~2us/query - vnFromHEquiv() remains the independent reference implementation BOTH paths above
    are validated against (see wmpl/MetSim/Tests/test_MetSimErosionAlphaBeta.py's
    test_velocity_spline_accuracy()), deliberately never itself switched to the LUT, so it stays a
    genuinely independent check.

    The sampling grid is deliberately non-uniform in v_n - Chebyshev-style spacing (dense at both
    v_n=1 and v_n=v_eps, sparser in between). A naive uniform grid under-resolves badly right near
    v_n=1 (segment start): hEquivFromVn()'s derivative is finite there (by construction - see the
    module notes on Correction 2), but small, which makes the *inverse* mapping locally
    ill-conditioned in that region even though the forward function itself is well-behaved. It
    also under-resolves the deep low-velocity tail if not specifically checked there (found by
    testing further into the tail than the first validation pass covered) - the symmetric
    Chebyshev grid fixes both without needing per-case tuning.

    A SECOND, more targeted gap was found and fixed directly against a complex-scenario comparison
    plot: even the Chebyshev grid's own closest point to v_n=1 (besides the exact endpoint itself)
    sits at 1-v_n ~ (pi/n_grid)^2/4 (~6e-5 for the default n_grid=200) - for a near-vacuum-start
    segment with a large alpha (confirmed on a real case: a main fragment's own first segment,
    alpha~85, h_equiv_start~2.9e5), hEquivFromVn()'s finite-but-steep derivative there means this
    ONE untabulated gap (1-v_n from 0 to ~6e-5) spans roughly 1.8e5 EQUIVALENT-HEIGHT units with no
    tabulated point inside it at all - confirmed directly, not assumed, by comparing this spline's
    own output against a high-precision (non-spline) quadrature reconstruction: up to ~0.15-0.2 m/s
    of spurious extra velocity loss within the first several milliseconds of a 16 km/s flight,
    invisible in overall velocity/height/mass (a ~1e-5 relative error, buried under the model's own
    already-accepted ~0.5-1% tolerance) but large enough, relative to the TRUE near-zero
    deceleration in that specific narrow regime, to show up as a real, visible spike in a
    frame-by-frame acceleration comparison. Fixed with a dedicated log-spaced refinement sub-grid
    in (1-v_n) covering exactly this gap (see below) - cheap (hEquivFromVn() is closed-form, no
    root-finding, so extra points cost almost nothing at construction time) and safe for every
    alpha (adds resolution that is simply redundant, never harmful, wherever the base grid was
    already adequate).

    A real regression was introduced and caught while building the fix above, worth recording: a
    first version also deduplicated NEAR-duplicate h_grid entries (a relative-threshold guard,
    reasoned to protect against the refinement points landing indistinguishably close to an
    existing point for small-alpha cases). This was WRONG - caught by the existing
    test_velocity_spline_accuracy() suite (a "shallow low-density" case went from passing to a 51%
    error) before shipping: for LARGE-alpha cases, the base Chebyshev grid alone already
    legitimately packs points as close as ~2e-9 apart in some regions (confirmed directly), and a
    single relative threshold applied grid-wide discarded HALF of a 260-point grid, destroying real
    resolution far from where the near-v_n=1 refinement was even trying to help. Only EXACT (zero-
    width) duplicates are removed now - PchipInterpolator has no actual difficulty with closely-
    spaced-but-distinct points, which the unmodified algorithm already relied on throughout.
    """

    # Above this beta, Ei(beta) - Ei(beta*v_n^2) starts losing real precision to float64
    # cancellation (Ei(beta) many orders of magnitude larger than Ei(beta*v_n^2)) - matches
    # wmpl.Utils.AlphaBeta's own documented "beta >~ 30" limit for this exact cancellation, with
    # margin (not pushed right up to the edge of what that module's own accuracy testing covers).
    _LUT_BETA_MAX = 25.0

    def __init__(self, alpha, beta, h_equiv_start, n_grid=200, v_eps=1e-6):
        """
        Arguments:
            alpha: [float] Ballistic coefficient (see alphaFromPhysical()).
            beta: [float] Mass-loss parameter (see betaFromPhysical()).
            h_equiv_start: [float] Equivalent height (m) at the start of this segment (v_n=1).

        Keyword arguments:
            n_grid: [int] Number of forward-evaluation grid points used to build the spline
                (before adding the near-v_n=1 refinement points - see class docstring). Unused
                when the LUT path is selected (beta < _LUT_BETA_MAX).
            v_eps: [float] Bracket margin around (0, 1), matching vnFromHEquiv()'s convention -
                queries at or beyond the resulting height bounds are clipped to v_eps/(1 - v_eps).
        """

        self.alpha = alpha
        self.beta = beta
        self.h_equiv_start = h_equiv_start
        self.v_eps = v_eps

        self._use_lut = beta < self._LUT_BETA_MAX

        if self._use_lut:
            self._ei_beta = float(scipy.special.expi(beta))
            self.h_hi = float(hEquivFromVn(1.0 - v_eps, alpha, beta, h_equiv_start))
            self.h_lo = float(hEquivFromVn(v_eps, alpha, beta, h_equiv_start))
            return

        # Chebyshev-style node spacing: dense at both u=0 (v_n=1) and u=1 (v_n=v_eps)
        u = np.linspace(0.0, 1.0, n_grid)
        v_n_grid = v_eps + (1.0 - v_eps)*(1.0 + np.cos(np.pi*u))/2.0

        # Near-v_n=1 refinement: log-spaced in (1-v_n), from float64's own useful precision floor
        # up to a comfortable overlap with the Chebyshev grid's own existing coverage there (~1e-2,
        # well inside where the Chebyshev spacing above is already dense) - see class docstring for
        # why this specific gap needs its own dedicated points rather than a bigger n_grid overall.
        v_n_grid = np.concatenate([v_n_grid, 1.0 - np.logspace(-13, -2, 60)])

        h_grid = hEquivFromVn(v_n_grid, alpha, beta, h_equiv_start)

        order = np.argsort(h_grid)
        h_sorted = h_grid[order]
        v_n_sorted = v_n_grid[order]

        # Guard against EXACT-duplicate h_grid entries only (e.g. a refinement v_n coinciding
        # bit-for-bit with a base-grid v_n) - PchipInterpolator only requires strictly increasing
        # x, and a zero-width gap would violate that. A first version of this guard used a
        # threshold relative to the grid's own overall span, reasoned to protect against "near"-
        # duplicates too - this was WRONG and a real regression, caught by the existing
        # test_velocity_spline_accuracy() suite before shipping: for large-alpha cases the base
        # Chebyshev grid alone already legitimately packs points as close as ~2e-9 apart in some
        # regions (confirmed directly, not assumed), and a global relative threshold discarded
        # HALF of a 260-point grid, destroying real, needed resolution far from where the
        # near-v_n=1 refinement above was even trying to help. PchipInterpolator has no actual
        # difficulty with closely-spaced-but-DISTINCT points (the unmodified algorithm already
        # relied on exactly that), so only true (gap == 0) duplicates need removing.
        keep = np.concatenate([[True], np.diff(h_sorted) > 0.0])
        self._h_grid = h_sorted[keep]
        self._v_n_grid = v_n_sorted[keep]

        # v_n is a monotonically increasing function of h_equiv (see hEquivFromVn()'s docstring),
        # so these are exactly the height bounds vnFromHEquiv() would clip against
        self.h_lo = self._h_grid[0]
        self.h_hi = self._h_grid[-1]

        self._spline = scipy.interpolate.PchipInterpolator(self._h_grid, self._v_n_grid,
            extrapolate=False)


    def velocityNormedAt(self, h_equiv):
        """ Normalized velocity v_n = v/v_start at the given equivalent height(s). Clipped to
        v_eps/(1 - v_eps) outside the built range, matching vnFromHEquiv()'s convention.

        Arguments:
            h_equiv: [float or ndarray] Equivalent height(s) (m) to evaluate at.

        Return:
            [float or ndarray] Normalized velocity v_n, same shape as h_equiv.
        """

        h_clipped = np.clip(h_equiv, self.h_lo, self.h_hi)

        if self._use_lut:
            return self._velocityNormedAtLUT(h_clipped)

        return self._spline(h_clipped)


    def _velocityNormedAtLUT(self, h_equiv):
        """ Closed-form v_n(h_equiv) via the process-wide inverse-Ei() LUT - the PRIMARY path (see
        class docstring), used whenever self.beta < _LUT_BETA_MAX. Inverts hEquivFromVn() exactly
        (h_equiv = -H*ln(exp(-h_equiv_start/H) + delta*exp(-beta)/(2*alpha)), delta = Delta(v_n) =
        Ei(beta) - Ei(beta*v_n^2)) algebraically for delta, then for v_n via the LUT:

            delta = (exp(-h_equiv/H) - exp(-h_equiv_start/H)) * 2*alpha*exp(beta)
            Ei(beta*v_n^2) = Ei(beta) - delta
            v_n = sqrt( Ei^-1(Ei(beta) - delta) / beta )

        Same structure, and the same near-v_n=1 cancellation safeguard, as
        wmpl.Utils.AlphaBeta.alphaBetaVelocityNormedLUT() - Ei(beta) can be many orders of
        magnitude larger than delta itself for v_n close to 1 (h_equiv close to h_equiv_start),
        which would lose precision computing Ei(beta) - delta directly; a local Taylor expansion
        of Ei around beta (exact to leading order there) is used instead whenever delta is
        negligible relative to Ei(beta). h_equiv is assumed already clipped to [h_lo, h_hi] by
        the caller (velocityNormedAt()), so no further clipping of the result is needed here.

        Arguments:
            h_equiv: [float or ndarray] Equivalent height(s) (m), already clipped to this
                segment's own [h_lo, h_hi].

        Return:
            [float or ndarray] Normalized velocity v_n, same shape as h_equiv.
        """

        scalar_input = np.isscalar(h_equiv) or np.ndim(h_equiv) == 0
        h = np.atleast_1d(np.asarray(h_equiv, dtype=np.float64))

        lut = getDefaultInverseEiLUT()

        with np.errstate(over='ignore'):

            delta = ((np.exp(-h/HT_NORM_CONST) - np.exp(-self.h_equiv_start/HT_NORM_CONST))
                *2*self.alpha*np.exp(self.beta))

            if self.beta >= 10.0:
                mask_asymptotic = delta < 1e-8*self._ei_beta
            else:
                mask_asymptotic = np.zeros_like(h, dtype=bool)

            x = np.empty_like(h)

            if np.any(mask_asymptotic):
                x[mask_asymptotic] = (self.beta
                    - delta[mask_asymptotic]*self.beta*np.exp(-self.beta))

            mask_lut = ~mask_asymptotic
            if np.any(mask_lut):
                x[mask_lut] = lut(self._ei_beta - delta[mask_lut])

            v_n = np.sqrt(np.clip(x, 0.0, None)/self.beta)

        # Exact boundary handling, matching alphaBetaVelocityNormedLUT()'s own convention of
        # clipping directly rather than trusting the closed-form computation exactly AT the
        # bracket edges: h is already clipped to [h_lo, h_hi] by the caller, so h<=h_lo/h>=h_hi
        # here means EXACTLY v_eps/1-v_eps by construction (h_lo/h_hi were themselves computed as
        # hEquivFromVn(v_eps/1-v_eps, ...)), not something to re-derive numerically. Needed for two
        # distinct reasons: (1) near v_n=v_eps, x=beta*v_eps^2 can fall below the shared LUT's own
        # tabulated floor (getDefaultInverseEiLUT()'s x_min=1e-10) for realistic beta - confirmed
        # directly, a case with beta=1.0 showed a 1153% relative error at h==h_lo before this fix;
        # (2) near v_n=1-v_eps, the LUT's own ~1e-6-level intrinsic approximation error (documented
        # in wmpl.Utils.AlphaBeta - negligible for interior queries against this model's own
        # ~0.5-1% physical accuracy) showed up as a v_n plateau just short of the true boundary
        # value for a whole cluster of near-v_n=1 queries that all clip to the same h_hi.
        v_n = np.where(h <= self.h_lo, self.v_eps, v_n)
        v_n = np.where(h >= self.h_hi, 1.0 - self.v_eps, v_n)

        if scalar_input:
            return float(v_n[0])

        return v_n


def massFromVelocityNormed(v_n, v_start, sigma_eff, m_start):
    """ Closed-form mass as a function of normalized velocity, exact for constant K (no erosion-
    vs-ablation split needed here - see the module notes on segments): derived directly from
    dm/dv = sigma_eff*m*v (dividing the mass-loss and drag ODEs), independent of atmosphere model
    or mu. Validated against RK4's own tracked mass to <0.3% (matching the model's established
    accuracy elsewhere).

    Arguments:
        v_n: [float or ndarray] Velocity normalized to v_start.
        v_start: [float] Velocity at the start of this segment (m/s).
        sigma_eff: [float] Effective ablation coefficient for this segment (s^2/m^2) - sigma alone
            while not eroding, sigma + erosion_coeff while eroding (see module notes). NOTE: once
            erosion arrives (Stage 3), luminosity/ionization must use sigma_own (ablation only),
            never sigma_eff - only the mass-budget calculation uses sigma_eff.
        m_start: [float] Mass at the start of this segment (kg).

    Return:
        [float or ndarray] Mass (kg), same shape as v_n.
    """

    return m_start*np.exp(sigma_eff*v_start**2*(np.asarray(v_n)**2 - 1.0)/2.0)


class AnalyticTrajectory(object):
    """ Full analytic trajectory for one segment of constant (K, sigma_eff): given physical
    starting conditions, provides fast v(t)/h_real(t)/m(t) evaluation. Built once per segment (two
    Chebyshev-tabulated splines - see VelocitySpline's docstring for why this shape of grid, and
    the module's "Query pattern" notes in the implementation plan for why building once and
    querying many times is the right shape for real usage), then every subsequent query is a cheap
    spline evaluation.

    Time (needed because results_list is indexed by time, one row per dt - see the plan's "Query
    pattern per fragment/segment" section) requires a second inversion beyond VelocitySpline's
    v_n(h_equiv): dh_real/dt = -v*sin(slope) is an exact kinematic relation (no atmosphere
    dependence at all), so t is obtained by integrating 1/v_n(h_real) over REAL height, not by
    integrating the atmosphere-dependent dv/dt directly over v_n. That distinction matters
    numerically, not just stylistically: integrating over v_n hits a genuine near-singularity for
    any segment starting close to vacuum (dt/dv_n can reach ~1e5 s right at v_n=1 for a segment
    starting at h_init=180km - the "coasting phase" before the atmosphere is dense enough to matter
    - and a per-point scipy.optimize.quad there was found to fail silently on ~2 of 30 sampled
    points, off by up to -55%, even though the surrounding points were fine). Integrating over real
    height instead avoids the density term entirely and was validated error-free (<0.5% velocity
    error at real dt=0.005s query ticks) across every case in this module's test suite, including
    the same near-vacuum-start segments that broke the v_n-integration approach.

    NOTE: this evaluates the physics (ablation, deceleration, hence also this class's own h_real)
    using a FLAT-EARTH path-length assumption throughout (dh_real/dx = -sin(slope), exactly linear
    - this is the alpha-beta closed form's native assumption, and matches Stage 1's atmosphere
    reconciliation, which is also built in flat-earth terms). Converting h_real here into the
    curved-Earth + gravity-drop corrected height MetSimErosion.py actually reports and uses for its
    OWN atmosphere lookups (heightCurvature(), MetSimErosion.py:804-818) is a separate, purely
    geometric post-processing step, not yet implemented in this class - see the implementation
    plan's "Secondary geometry note" for why this is deliberately decoupled rather than fed back
    into the atmosphere lookup, and for the quantification still owed before deciding whether an
    iterative refinement is needed for long/steep paths.
    """

    def __init__(self, K, sigma_eff, m_start, v_start, h_real_start, sin_slope, atm_map,
            t_start=0.0, n_grid=300, v_eps=1e-6, r_earth=6_371_008.7714, sim_dt=0.005,
            atm_height_fn=None, v_n_floor=0.02, h_real_floor=None):
        """
        Arguments:
            K: [float] Shape-density coefficient (m^2/kg^(2/3)).
            sigma_eff: [float] Effective ablation coefficient for this segment (s^2/m^2).
            m_start: [float] Mass at the start of this segment (kg).
            v_start: [float] Velocity at the start of this segment (m/s).
            h_real_start: [float] Real height at the start of this segment (m).
            sin_slope: [float] sin(entry slope from horizontal) = cos(zenith_angle).
            atm_map: [AtmEquivHeightMap] Atmosphere-equivalent-height map for this simulation.

        Keyword arguments:
            t_start: [float] Time at the start of this segment (s) - lets segments be chained.
            n_grid: [int] Number of grid points used to build both splines.
            v_eps: [float] Bracket margin around (0, 1) - see VelocitySpline.
            r_earth: [float] Earth radius (m), for the gravity-drop calculation (see
                gravityDropAt()) - matches Constants.r_earth.
            sim_dt: [float] const.dt of the simulation being replicated - NOT a step size used by
                any integration here (there is none), but gravityDropAt() must reproduce
                MetSimErosion.py's h_grav_drop_total EXACTLY, and that quantity is only well-
                defined relative to a chosen dt (see gravityDropAt()'s docstring for why).
            atm_height_fn: [callable or None] Optional h_real_array -> h_atm_array override for
                the height used ONLY when looking up atmosphere-equivalent height (h_equiv) to
                build the internal v_n/time tabulation - NOT for heightRealAt()/lengthAt(), which
                always stay in terms of the true flat-earth h_real this segment's kinematics are
                solved in. Defaults to None (identity: atmosphere is looked up at the flat height
                directly), matching the physics of a single, non-iterative build.

                This is the hook the iterative curvature/atmosphere refinement (see runSimulation()
                and the implementation plan) uses: it's not needed for correctness on its own, only
                to progressively feed a BETTER estimate of the true (curved-Earth + gravity-drop)
                height into the atmosphere lookup than the flat height alone would give, without
                changing what "h_real" or "length" mean for this segment (that would break the
                exact kinematic relation the closed form and time-quadrature depend on - see the
                class docstring's note on why curvature is kept as output-only post-processing).
            v_n_floor: [float] How far (in normalized velocity) the internal grid needs to extend,
                as a fraction of v_start. NOT v_eps: v_eps is a near-zero numerical safety margin
                for VelocitySpline's own construction; v_n_floor bounds how much of the segment's
                trajectory this class actually tabulates, and must be chosen from the caller's own
                physical stopping criteria (e.g. v_kill/v_start), not left at an arbitrary tiny
                default. Chasing v_n all the way to v_eps (effectively v=0) was tried and found to
                be actively harmful, not just wasteful: drag-limited deceleration never reaches v=0
                in finite time, so the time to reach a given v_n diverges as v_n->0, and a caller
                that (reasonably) tries to bound how far to search for that divergent quantity ends
                up needing atm_height_fn to extrapolate into a regime where it becomes unphysical
                (non-monotonic v_n vs h_real) - producing search results that don't converge between
                refinement passes even though the actual physical trajectory (v(t)/h(t) at any fixed
                t) converges cleanly. runSimulation() never needs v_n anywhere near that asymptote -
                it needs the trajectory resolved up to (and a little past) whichever physical kill
                condition (m_kill/v_kill/h_kill/len_kill) triggers first, exactly like the RK4
                model, which also never integrates to v=0.
            h_real_floor: [float or None] Optional hard lower bound (m) on how deep in FLAT real
                height the internal grid is ever tabulated, regardless of what v_n_floor would
                otherwise ask for. Needed because v_n_floor alone is not always a tight bound: for
                shallow/slowly-decelerating segments (small sin_slope, low sigma_eff), the flat
                path length needed to reach even a modest v_n_floor can run to hundreds of km,
                which can push heightCurvature() (in reportedHeightAt(), used one refinement pass
                later to build atm_height_fn) into a regime past where the curved-Earth law of
                cosines behaves monotonically - producing a corrected height ABOVE h_real_start,
                which then corrupts atm_height_fn for every subsequent lookup (found via a genuine
                crash on an 85-degree-entry case, not a hypothetical). Callers should pass a value
                derived from their own physical stopping height (e.g. const.h_kill, with a margin -
                see runSimulation()) so the grid never extends past where the real simulation could
                ever ask for data. None (default) disables the floor, matching prior behavior.
        """

        self.K = K
        self.sigma_eff = sigma_eff
        self.m_start = m_start
        self.v_start = v_start
        self.h_real_start = h_real_start
        self.sin_slope = sin_slope
        self.t_start = t_start
        self.atm_map = atm_map
        self.v_eps = v_eps
        self.r_earth = r_earth

        self.h_equiv_start = float(atm_map.toEquiv(h_real_start))
        self.alpha = alphaFromPhysical(K, sin_slope, m_start)
        self.beta = betaFromPhysical(sigma_eff, v_start)

        self.velocity_spline = VelocitySpline(self.alpha, self.beta, self.h_equiv_start,
            n_grid=n_grid, v_eps=v_eps)

        # h_real_end targets v_n_floor (a modest, physically-motivated bound - see v_n_floor's
        # docstring), NOT v_eps: v_eps only bounds VelocitySpline's OWN Chebyshev grid (a
        # near-singular numerical safety margin), unrelated to how far this class's trajectory
        # needs to be tabulated for real use.
        #
        # Uncorrected case (atm_height_fn is None): h_equiv at v_n_floor is EXACT and closed-form
        # (hEquivFromVn(), no root-finding needed) - unlike v_eps's implicit dependence on
        # VelocitySpline's own internal grid (velocity_spline.h_lo), which was a leftover
        # implementation detail this rewrite removes.
        h_equiv_floor = hEquivFromVn(v_n_floor, self.alpha, self.beta, self.h_equiv_start)
        self.h_real_end = float(atm_map.toReal(h_equiv_floor))

        # h_real_floor overrides v_n_floor whenever v_n_floor alone would reach deeper than the
        # caller's physical stopping height - see h_real_floor's docstring. Applied here too (not
        # just in the atm_height_fn branch below) because the OVEREXTENSION itself already exists
        # in this uncorrected pass; the atm_height_fn branch's own crash is a downstream symptom of
        # feeding this pass's un-floored reportedHeightAt() into the next pass's atm_height_fn.
        if h_real_floor is not None:
            self.h_real_end = max(self.h_real_end, h_real_floor)

        if atm_height_fn is not None:
            # Once atm_height_fn is in play, h_equiv_floor above is no longer self-consistent (the
            # corrected/curved height differs from the flat height at a given point, so the flat
            # depth needed to reach v_n_floor shifts too) - re-find h_real_end: walk downward in
            # FINE steps until the corrected v_n actually reaches v_n_floor, then refine with a
            # bracketed root-find on that narrow interval.
            #
            # Fine steps matter, not just a correctness nicety: atm_height_fn may need to
            # extrapolate beyond the previous pass's own grid range, and that extrapolation can
            # become unphysical (v_n stops being monotonic in h_real) far enough out - large steps
            # risk jumping straight over the real (narrow) crossing into that pathological region.
            # Targeting v_n_floor instead of v_eps keeps the crossing itself comfortably away from
            # where extrapolation is even needed in most cases, which is the actual fix; the fine
            # stepping is defense in depth, not a substitute for that.
            #
            # The search never steps below h_real_floor (when given): if v_n still hasn't reached
            # v_n_floor by the time the search reaches the floor, that means h_real_floor's own
            # physical condition (e.g. h_kill) would end the fragment first anyway - accept the
            # floor itself as h_real_end rather than continuing to chase v_n_floor into territory
            # the caller has already said it will never need.
            def _vn_minus_floor(h_real):
                h_equiv = atm_map.toEquiv(atm_height_fn(np.array([h_real]))[0])
                return float(self.velocity_spline.velocityNormedAt(h_equiv)) - v_n_floor

            span = h_real_start - self.h_real_end
            step = max(span*0.02, 100.0)

            h_b = self.h_real_end
            val_b = _vn_minus_floor(h_b)

            if val_b > 0:
                h_a, val_a = h_b, val_b
                hit_floor = False
                for _ in range(300):
                    h_a = h_b - step
                    if h_real_floor is not None and h_a <= h_real_floor:
                        h_a = h_real_floor
                        val_a = _vn_minus_floor(h_a)
                        hit_floor = True
                        break
                    val_a = _vn_minus_floor(h_a)
                    if val_a <= 0:
                        break
                    h_b, val_b = h_a, val_a
                else:
                    raise RuntimeError("AnalyticTrajectory: could not find where the "
                        "atm_height_fn-corrected v_n reaches v_n_floor within 300 fine-step "
                        "expansions.")

                if val_a <= 0:
                    self.h_real_end = float(scipy.optimize.brentq(_vn_minus_floor, h_a, h_b, xtol=1.0))
                else:
                    # hit_floor and v_n still above v_n_floor there - h_real_floor is the binding
                    # constraint, not v_n_floor.
                    assert hit_floor
                    self.h_real_end = h_a
            # else: the naive bound already has corrected v_n <= v_n_floor there too - a
            # conservative (possibly too-deep, never too-shallow) bound, fine to keep as-is.

        u = np.linspace(0.0, 1.0, n_grid)
        h_real_grid = self.h_real_end + (h_real_start - self.h_real_end)*(1.0 + np.cos(np.pi*u))/2.0
        h_real_grid = np.sort(h_real_grid)

        h_atm_grid = h_real_grid if atm_height_fn is None else atm_height_fn(h_real_grid)
        h_equiv_grid = atm_map.toEquiv(h_atm_grid)
        v_n_grid = self.velocity_spline.velocityNormedAt(h_equiv_grid)
        integrand_grid = -1.0/(v_n_grid*v_start*sin_slope)

        integrand_spline = scipy.interpolate.PchipInterpolator(h_real_grid, integrand_grid,
            extrapolate=True)
        antiderivative = integrand_spline.antiderivative()
        t_offset = antiderivative(h_real_start)

        t_grid = antiderivative(h_real_grid) - t_offset + t_start
        order = np.argsort(t_grid)
        self._t_grid = t_grid[order]
        self._v_n_grid = v_n_grid[order]
        self._h_real_grid = h_real_grid[order]

        # Time at v_n=v_eps (segment "end", essentially fully stopped) - asymptotically large
        # (drag-limited deceleration never quite reaches v=0 in finite time under this model), not
        # a physically meaningful "lifetime". Real kill conditions (m_kill/v_kill/h_kill/len_kill)
        # are always reached well before this bound.
        self.t_hi = self._t_grid[-1]

        # Cumulative gravity-drop since t_start: MetSimErosion.py accumulates
        # h_grav_drop_total += 0.5*g(h)*dt**2 every step (MetSimErosion.py:804-808). Summed over N
        # steps of size dt (T = N*dt), this is N*0.5*g_avg*dt**2 = 0.5*g_avg*dt*T - linear in T and
        # scaled by dt, NOT the textbook constant-acceleration drop 0.5*g*T**2 (that would be the
        # actual double integral of g over time, and is a DIFFERENT, dt-independent quantity - an
        # earlier version of this code used that by mistake, and it overshot the true value by
        # ~1700x on inspection: ~360 m vs ~0.21 m predicted vs. actual on an 8.57 s flight).
        # sum_i g_i*dt approximates the SINGLE integral of g over time, so the discrete sum overall
        # approximates 0.5*dt*integral(g dt), i.e. a single antiderivative scaled by 0.5*sim_dt,
        # not a double antiderivative - verified against the exact discrete sum to 5 decimal places.
        g_grid = G0/(1.0 + self._h_real_grid/r_earth)**2

        # v_n/h_real/g all share the exact same self._t_grid x-axis (by construction, above) - fit
        # ONE vector-valued PchipInterpolator (3 channels: v_n, h_real, g) instead of three separate
        # scalar ones. Profiling (see the implementation plan's own write-up) found PchipInterpolator
        # construction itself carries substantial FIXED per-call overhead (~75-84us, essentially
        # independent of n_grid - scipy/Python object-construction cost, not numerical work), so
        # three separate fits pay that fixed cost three times over for no reason once the x-axis is
        # already shared. extrapolate=True (matching the original g_spline's own setting, not the
        # original v_n/h_real splines' extrapolate=False) - safe uniformly because every query site
        # (velocityNormedAt/heightRealAt/gravityDropAt) already clips its own t to [t_start, t_hi]
        # BEFORE ever calling into this spline (confirmed by grep - _time_to_combined is never
        # accessed unclipped anywhere in this file), so extrapolate=False's out-of-range NaN
        # behavior was never actually exercised in practice.
        combined_grid = np.column_stack([self._v_n_grid, self._h_real_grid, g_grid])
        self._time_to_combined = scipy.interpolate.PchipInterpolator(self._t_grid, combined_grid,
            axis=0, extrapolate=True)
        combined_antideriv = self._time_to_combined.antiderivative()
        self._grav_drop_offset = combined_antideriv(t_start)[2]
        self._grav_drop_antideriv_combined = combined_antideriv
        self._grav_drop_scale = 0.5*sim_dt

        # Instrumentation: total number of individual (fragment, time) state queries against this
        # segment - see the implementation plan's "Query pattern per fragment/segment" section.
        # Counts query POINTS, not calls (a single batched/vectorized call with an array of times
        # counts once per element), since that's what the amortization argument for building this
        # spline in the first place is actually about.
        self.n_queries = 0


    def velocityNormedAt(self, t):
        """ Normalized velocity v_n = v/v_start at the given time(s) (s). Clipped to
        [t_start, t_hi] outside the built range. """

        self.n_queries += 1 if np.isscalar(t) else len(np.atleast_1d(t))

        t_clipped = np.clip(t, self.t_start, self.t_hi)

        return self._time_to_combined(t_clipped)[..., 0]


    def heightRealAt(self, t):
        """ Flat-earth-path real height (m) at the given time(s) (s) - see class docstring for why
        this is not yet the curved-Earth + gravity-drop corrected height MetSimErosion.py reports
        (use reportedHeightAt() for that).
        """

        t_clipped = np.clip(t, self.t_start, self.t_hi)

        return self._time_to_combined(t_clipped)[..., 1]


    def massAt(self, t):
        """ Mass (kg) at the given time(s) (s). """

        v_n = self.velocityNormedAt(t)

        return massFromVelocityNormed(v_n, self.v_start, self.sigma_eff, self.m_start)


    def lengthAt(self, t):
        """ Path length (m) traveled since this segment's start, at the given time(s) (s). Exact
        (not a spline lookup): dh_real/dx = -sin_slope is linear by construction of the flat-earth
        physics this segment is solved in, so length = (h_real_start - h_real(t))/sin_slope.
        """

        return (self.h_real_start - self.heightRealAt(t))/self.sin_slope


    def gravityDropAt(self, t):
        """ Cumulative gravity-drop (m) since t_start, at the given time(s) (s) - reproduces
        MetSimErosion.py's h_grav_drop_total exactly (a dt-scaled quantity, not the textbook
        constant-acceleration drop formula - see the construction in __init__() for why). """

        t_clipped = np.clip(t, self.t_start, self.t_hi)

        g_antideriv = self._grav_drop_antideriv_combined(t_clipped)[..., 2]
        return self._grav_drop_scale*(g_antideriv - self._grav_drop_offset)


def reportedHeightAt(traj, t, h_init, zenith_angle, length_start=0.0, grav_drop_start=0.0):
    """ Curved-Earth + gravity-drop corrected height (m) at the given time(s) - what
    MetSimErosion.py actually reports (heightCurvature() + accumulated gravity-drop,
    MetSimErosion.py:804-834), applied here as a post-processing step on top of an
    AnalyticTrajectory solved in flat-earth path-length coordinates (see the class docstring for
    why these are kept decoupled rather than fed back into the atmosphere lookup).

    Arguments:
        traj: [AnalyticTrajectory] The segment to report height for.
        t: [float or ndarray] Time(s) (s), in the same t_start-relative frame as traj.
        h_init: [float] Height at the start of the WHOLE simulation (m) - const.h_init, not
            necessarily this segment's own h_real_start (a fragment spawned mid-flight has
            h_real_start < h_init, but heightCurvature() is always measured from the simulation's
            initial height, matching how MetSimErosion.py's Fragment.spawn_child() inherits
            frag.length rather than resetting it - see MetSimErosion.py:373-384).
        zenith_angle: [float] Zenith angle at the start of the WHOLE simulation (radians) -
            const.zenith_angle.

    Keyword arguments:
        length_start: [float] Path length (m) already accumulated before this segment started
            (0.0 for a fragment born at the simulation's own start; a spawned fragment should pass
            its parent's length at the moment of spawning).
        grav_drop_start: [float] Gravity-drop (m) already accumulated before this segment started -
            0.0 for a fragment born at the simulation's own start (traj.gravityDropAt() alone is
            then the correct, complete total), nonzero for a later segment in a chain or a fragment
            spawned mid-flight, whose parent already accumulated some gravity-drop of its own
            before this segment/fragment began (MetSimErosion.py's frag.h_grav_drop_total is a
            single running total across a fragment's ENTIRE life, including whatever a parent had
            already accumulated at spawn_child() time - MetSimErosion.py:373-384 copies it
            verbatim, never resetting to 0). Omitting this (the previous behavior, kept as the
            default since it is exactly correct for any fragment's own FIRST segment) understates
            reported height by exactly this much for every later segment/spawned fragment - shown
            directly to be small in practice (under 1m even for a several-second first segment,
            since this term is dt-SCALED, see AnalyticTrajectory's own gravityDropAt() docstring)
            but not exactly zero, so segments/fragments starting well into a long flight should
            still pass their own true cumulative value here rather than relying on it being
            negligible.

    Return:
        [float or ndarray] Reported height (m), same shape as t.
    """

    length = length_start + traj.lengthAt(t)
    h_curved = heightCurvature(h_init, zenith_angle, length, traj.r_earth)

    return h_curved - (grav_drop_start + traj.gravityDropAt(t))

###


class Constants(object):
    def __init__(self):
        """ Constant parameters for the ablation modelling. """

        ### Simulation parameters ###

        # Time step
        self.dt = 0.005

        # Time elapsed since the beginning
        self.total_time = 0

        # Number of active fragments
        self.n_active = 0

        # Minimum possible mass for ablation (kg)
        self.m_kill = 1e-14

        # Minimum ablation velocity (m/s)
        self.v_kill = 3000

        # Minimum height (m)
        self.h_kill = 60000

        # Maximum length along the trajectory (m) after which the simulation will stop
        # -1 means no limit
        self.len_kill = -1000

        # Initial meteoroid height (m)
        self.h_init = 180000

        # Power of a 0 magnitude meteor
        self.P_0m = 840

        # Atmosphere density coefficients
        self.dens_co = np.array([6.96795507e+01, -4.14779163e+03, 9.64506379e+04, -1.16695944e+06, \
            7.62346229e+06, -2.55529460e+07, 3.45163318e+07])
        
        # Radius of the Earth (m)
        self.r_earth = 6_371_008.7714

        self.total_fragments = 0

        ### ###


        ### Wake parameters ###

        # PSF stddev (m)
        self.wake_psf_weights = [0.9, 0.1]
        self.wake_psf = [3.0, 20]

        # Wake extension from the leading fragment (m)
        self.wake_extension = 200

        # Specific heights at which the wake should be simulated (m)
        self.wake_heights = None

        ### ###



        ### Main meteoroid properties ###

        # Meteoroid bulk density (kg/m^3)
        self.rho = 1000

        # Initial meteoroid mass (kg)
        self.m_init = 2e-5

        # Initial meteoroid veocity (m/s)
        self.v_init = 23570

        # Shape factor (1.21 is sphere)
        self.shape_factor = 1.21

        # Main fragment ablation coefficient (s^2/km^2)
        self.sigma = 0.023/1e6

        # Zenith angle (radians)
        self.zenith_angle = math.radians(45)

        # Drag coefficient
        self.gamma = 1.0

        # Grain bulk density (kg/m^3)
        self.rho_grain = 3000


        # Luminous efficiency type (1 - 8, see luminousEfficiency function)
        self.lum_eff_type = 0

        # Constant luminous efficiency (percent)
        self.lum_eff = 0.7

        # Mean atomic mass of a meteor atom, kg (Jones 1997)
        self.mu = 23*1.66*1e-27

        ### ###


        ### Erosion properties ###

        # Toggle erosion on/off
        self.erosion_on = True


        # Bins per order of magnitude mass
        self.erosion_bins_per_10mass = 10
        
        # Height at which the erosion starts (meters)
        self.erosion_height_start = 102000

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = 0.33/1e6

        
        # Height at which the erosion coefficient changes (meters)
        self.erosion_height_change = 90000

        # Erosion coefficient after the change (s^2/m^2)
        self.erosion_coeff_change = 0.33/1e6

        # Density after erosion change (density of small chondrules by default)
        self.erosion_rho_change = 3700

        # Ablation coeff after erosion change
        self.erosion_sigma_change = self.sigma

        # Grain distribution model ('powerlaw' mass or 'gamma' diameters)
        self.erosion_grain_distribution = 'powerlaw'

        # Grain mass distribution index
        self.erosion_mass_index = 2.5

        # Mass range for grains (kg)
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10

        # [Analytic engine only, not present on MetSimErosion.Constants] Number of mass-uniform
        # epochs _spawnGrainsForSegment() splits each eroding segment's total eroded mass into (see
        # that function's docstring) - the main accuracy/speed knob for the erosion path of
        # runSimulation() (Stage 3d). Chosen from a direct, measured accuracy-vs-wall-clock sweep on
        # this file's own representative erosion-heavy test scenario (h_init=120000m, zenith=45deg,
        # m_init=0.5kg, v_init=16000m/s, rho=3300, sigma=0.015e-6, erosion_coeff=0.3e-6): 1000
        # gives a frame-averaged (30fps) worst-case light-curve error of 0.126 mag against
        # MetSimErosion.runSimulation() (matching Stage 3c's own validated number for the grain-
        # evolution mechanism in isolation) at a real 3.4x wall-clock speedup; 200 is ~14x faster
        # but the error grows to 0.58 mag (too coarse to trust for a real fit); 2000+ improves
        # accuracy further (0.061 mag) but the speedup shrinks to under 2x. 1000 is the default
        # because it is the smallest value tested that keeps the error at the level Stage 3c's
        # grain-evolution validation already accepted - not because it is provably optimal weighed
        # against any particular downstream fit's own noise floor. Lower for more speed at the cost
        # of light-curve smoothness; raise for more accuracy at the cost of wall-clock time.
        self.erosion_n_epochs = 1000

        # [Analytic engine only] Sub-spawns per epoch, uniform in TIME within that epoch's own real
        # span (see _spawnGrainsForSegment()'s docstring for the "pulse train" artifact this
        # targets). 1 was sufficient once _stepGrainRK4() (which replicates the reference's own
        # coarse-dt numerics, rather than assuming continuous-physics smoothness) became the active
        # grain-evolution mechanism - the artifact n_smear was originally built for was specific to
        # an earlier, since-abandoned closed-form grain trajectory approach (see the implementation
        # plan's Stage 3c write-up). Kept as a tunable rather than removed, since it is still a
        # valid (if currently unneeded) refinement.
        self.erosion_n_smear = 1

        # [Analytic engine only, not present on MetSimErosion.Constants] If True, every grain
        # population (plain erosion, disruption leftover mass, Stage 5 "D" dust) AND every
        # erosion-capable disruption/EF daughter's own spawned grains are evolved via
        # _stepGrainPopulationAnalytic() (a vectorized population wrapper around
        # _analyticGrainState() - the EXACT closed-form continuous-physics solution) instead of the
        # default _stepGrainRK4()/_stepGrainPopulationRK4() (which deliberately REPLICATES
        # MetSimErosion.py's own coarse dt=0.005 RK4 stepping, including its confirmed numerical
        # non-convergence for grain-scale masses - see _stepGrainRK4()'s own docstring). Default
        # False: matches every fragment/grain to the reference tool's own tick-by-tick numerical
        # output (the "tolerance-matched substitute" goal every other part of this engine is
        # validated against). True is a DIFFERENT acceptance criterion for grains specifically -
        # true continuous physics, not reference-numerical-matching - which necessarily diverges
        # from MetSimErosion.py's own output wherever the reference's own RK4 stepping was itself
        # inaccurate: confirmed directly (worst frame-averaged |dmag| across a whole simulation, not
        # just one grain in isolation) at ~0.59 mag for a representative plain-erosion scenario
        # (default mode: ~0.09 mag against the same reference) and ~0.96 mag for disruption+erosion
        # (default mode: ~1.06 mag - a real but PARTIAL improvement, not a fix - see
        # _runSimulationErosion()'s own "Stage 4" docstring section for the full, corrected account
        # of why that gap cannot be fully closed by a grain-evolution-mechanism change alone). Only
        # ever changes how SPAWNED GRAINS/daughters evolve - never the main fragment's or any
        # daughter's own segment dynamics, which are already the exact closed form either way.
        # _stepGrainPopulationAnalytic() itself is validated against a fine-resolution RK4 mirror
        # (true physics), not the reference tool - see test_step_grain_population_analytic_matches_scalar,
        # test_step_grain_population_analytic_accuracy_vs_fine_rk4_mirror,
        # test_run_simulation_analytic_mode_smoke_test, and
        # test_run_simulation_analytic_mode_disruption_gap_partially_improves in
        # wmpl/MetSim/Tests/test_MetSimErosionAlphaBeta.py. A first, per-grain-Python-call version
        # of this mode was measured to be SLOWER than the existing RK4-population mechanism (0.29x
        # on the plain-erosion scenario above) before being vectorized. A second version vectorized
        # the expensive grid construction but still looped per GRAIN (tens of thousands of Python-
        # level iterations) to extract each one's state - measured, directly, to still be slower
        # than the default mode (0.62x plain erosion, 0.51x disruption+erosion) despite that fix.
        # Root-caused via profiling (not assumed) to the per-grain loop's own iteration count, and
        # fixed by restructuring it to loop over GLOBAL TICKS instead (mirroring
        # _stepGrainPopulationRK4()'s own architecture) - see _stepGrainPopulationAnalytic()'s own
        # docstring for the full, three-version performance history. Current, final numbers: plain
        # erosion 8.71x vs the reference tool, 0.83x vs default mode; disruption+erosion 1.20x vs
        # the reference tool (now FASTER than the unmodified reference, unlike every earlier
        # version) and 0.79x vs default mode. Still short of matching the default RK4-population
        # mechanism outright - the largest remaining cost (_scatterArgmaxGroupby(), profiled
        # directly) is shared aggregation infrastructure common to every mode, not specific to
        # grain evolution itself. Opt into this mode for accuracy investigation; it is no longer a
        # clear performance loss versus the reference tool, but still trails the default mode.
        self.grain_evolution_analytic = False

        ###


        ### Disruption properties ###

        # Toggle disruption on/off
        self.disruption_on = True

        # Meteoroid compressive strength (Pa)
        self.compressive_strength = 2000

        # Height of disruption (will be assigned when the disruption occures)
        self.disruption_height = None

        # Erosion coefficient to use after disruption
        self.disruption_erosion_coeff = self.erosion_coeff

        # Disruption mass distribution index
        self.disruption_mass_index = 2.0


        # Mass ratio for disrupted fragments as the ratio of the disrupted mass
        self.disruption_mass_min_ratio = 1.0/100
        self.disruption_mass_max_ratio = 10.0/100

        # Ratio of mass that will disrupt into grains
        self.disruption_mass_grain_ratio = 0.25

        ### ###


        ### Complex fragmentation behaviour ###

        # Indicate if the complex fragmentation is used
        self.fragmentation_on = False

        # Track light curves of individual fragments
        self.fragmentation_show_individual_lcs = False

        # A list of fragmentation entries
        self.fragmentation_entries = []

        # Name of the fragmentation file
        self.fragmentation_file_name = "metsim_fragmentation.txt"

        ### ###


        ### Radar measurements ###

        # Height at which the electron line density is measured (m)
        self.electron_density_meas_ht = -1000

        # Measured electron line density (e-/m)
        self.electron_density_meas_q = -1

        ### ###


        
        ### OUTPUT PARAMETERS ###

        # Velocity at the beginning of erosion
        self.erosion_beg_vel = None

        # Mass at the beginning of erosion
        self.erosion_beg_mass = None

        # Dynamic pressure at the beginning of erosion
        self.erosion_beg_dyn_press = None

        # Mass of main fragment at erosion change
        self.mass_at_erosion_change = None

        # Energy received per unit cross section prior to to erosion begin
        self.energy_per_cs_before_erosion = None

        # Energy received per unit mass prior to to erosion begin
        self.energy_per_mass_before_erosion = None

        # Height at which the main mass was depleeted
        self.main_mass_exhaustion_ht = None

        # Bottom height that the main fragment reached
        self.main_bottom_ht = self.h_init

        ### ###


class Fragment(object):
    def __init__(self):

        self.id = 0

        self.const = None

        # Shape-density coeff
        self.K = 0

        # Initial fragment mass
        self.m_init = 0

        # Instantaneous fragment mass Mass (kg)
        self.m = 0

        # Density (kg/m^3)
        self.rho = 0

        # Ablation coefficient (s^2/m^2)
        self.sigma = 0

        # Velocity (m/s)
        self.v = 0

        # Velocity components (vertical and horizontal)
        self.vv = 0
        self.vh = 0

        # Total drop due to gravity (m)
        self.h_grav_drop_total = 0

        # Length along the trajectory
        self.length = 0

        # Luminous intensity (Watts)
        self.lum = 0

        # Electron line density
        self.q = 0

        # Dynamic pressure (Gamma = 1.0, Pa)
        self.dyn_press = 0

        # Erosion coefficient value
        self.erosion_coeff = 0

        # Grain mass distribution index
        self.erosion_mass_index = 2.5

        # Mass range for grains (kg)
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10


        self.erosion_enabled = False

        self.disruption_enabled = False

        self.active = False
        self.n_grains = 1

        # Indicate that this is the main fragment
        self.main = False

        # Indicate that the fragment is a grain
        self.grain = False

        # Indicate that this is born out of complex fragmentation
        self.complex = False

        # Identifier of the compex fragmentation entry
        self.complex_id = None


    def init(self, const, m, rho, v_init, sigma, gamma, zenith_angle, erosion_mass_index, erosion_mass_min, \
        erosion_mass_max):

        self.const = const

        self.m = m
        self.m_init = m
        self.h = const.h_init
        self.rho = rho
        self.v = v_init
        self.sigma = sigma
        self.gamma = gamma
        self.zenith_angle = zenith_angle

        # Compute shape-density coeff
        self.updateShapeDensityCoeff()

        self.erosion_mass_index = erosion_mass_index
        self.erosion_mass_min = erosion_mass_min
        self.erosion_mass_max = erosion_mass_max

        # Compute velocity components
        self.vv = -v_init*math.cos(zenith_angle)
        self.vh = v_init*math.sin(zenith_angle)

        self.active = True
        self.n_grains = 1

    def updateShapeDensityCoeff(self):
        """ Update the value of the shape-density coefficient. """

        self.K = self.gamma*self.const.shape_factor*self.rho**(-2/3.0)

    def spawn_child(self):
        """ Create a child of the Fragment instance. Copy over the reference to the shared 'const' object 
        and copy values of all other attributes. Note: if a mutable attribute that is not shared across 
        Fragment instances is added, this function will need to be revised. 
        """
        
        cls = self.__class__
        child = cls.__new__(cls)
                
        child.__dict__.update(self.__dict__)
    
        return child


class Wake(object):
    def __init__(self, const, frag_list, leading_frag_length, length_array):
        """ Container for the evaluated wake. 
        
        Arguments:
            const: [Constants]
            frag_list: [list of Fragment object] A list of active fragments visible in the wake.
            leading_frag_length: [float] Length from the beginning of the simulation of the leading fragment.
            length_array: [ndarray] An array of lengths (zero centered to the leading fragment) over which 
                the lag will be evaluated.
        """

        # Constants
        self.const = const

        # List of active fragments within the window
        self.frag_list = frag_list

        # Length of the leading fragment
        self.leading_frag_length = leading_frag_length

        # Array of lengths for plotting (independent variable)
        self.length_array = length_array

        # Length of visible fragments
        self.length_points = np.array([frag.length - self.leading_frag_length for frag in self.frag_list])

        # Luminosity of visible fragments
        self.luminosity_points = np.array([frag.lum for frag in self.frag_list])


        # Evalute the Gaussian at every fragment an add to the estimated wake
        self.wake_luminosity_profile = np.zeros_like(length_array)

        # If there is not entry for the wake PSF weights, initialize it
        if not hasattr(self.const, 'wake_psf_weights'):
            self.const.wake_psf_weights = np.ones_like(self.const.wake_psf)

        # Normalize the wake weights so they sum to 1
        self.const.wake_psf_weights = self.const.wake_psf_weights/np.sum(self.const.wake_psf_weights)
        
        for frag_lum, frag_len in zip(self.luminosity_points, self.length_points):

            for psf_m, psf_weight in zip(self.const.wake_psf, self.const.wake_psf_weights):
                self.wake_luminosity_profile += psf_weight*frag_lum*scipy.stats.norm.pdf(self.length_array, loc=frag_len, \
                    scale=psf_m)


def zenithAngleAtSimulationBegin(h0_sim, hb, zc, r_earth):
    """ Compute the meteor zenith angle at the beginning of the simulation, given the observed begin height
        and the observed zenith angle.

    Arguments:
        h0_sim: [float] Initial height of the simulation (m).
        hb: [float] Observed begin height (m).
        zc: [float] Observed zenith angle (radians).

    Returns:
        beta: [float] Zenith angle at the beginning of the simulation (radians).
        
    """

    beta = np.arcsin((hb + r_earth)/(h0_sim + r_earth)*np.sin(zc))

    return beta


def heightCurvature(h0, zc, l, r_earth):
    """ Compute the height at a given distance l from the origin, assuming a curved Earth.
    
    Arguments:
        h0: [float] Initial height (m).
        zc: [float] Zenith angle (radians).
        l: [float] Distance from the origin (m).
        r_earth: [float] Earth radius (m).

    Returns:
        h: [float] Height at distance l from the origin (m).
    """

    return np.sqrt((h0 + r_earth)**2 - 2*l*np.cos(zc)*(h0 + r_earth) + l**2) - r_earth


def generateFragments(const, frag_parent, eroded_mass, mass_index, mass_min, mass_max, 
                      keep_eroding=False, disruption=False, mass_model='powerlaw'):
    """ Given the parent fragment, fragment it into daughter fragments using either:
        - a power law mass distribution - appropriate for fragmentation of rock;
        - a gamma distribution - appropriate for spraying of droplets (iron meteoroids).

    Masses are binned and one daughter fragment may represent several fragments/grains, which is specified 
    with the n_grains atribute.

    Arguments:
        const: [object] Constants instance.
        frag_parent: [object] Fragment instance, the parent fragment.
        eroded_mass: [float] Mass to be distributed into daughter fragments. 
        mass_index: [float] Mass index to use to distribute the mass.
        mass_min: [float] Minimum mass bin (kg).
        mass_max: [float] Maximum mass bin (kg).

    Keyword arguments:
        keep_eroding: [bool] Whether the daughter fragments should keep eroding.
        disruption: [bool] Indicates that the disruption occured, uses a separate erosion parameter for
            disrupted daughter fragments.
        mass_model: [bool] Fragment mass distribution model to use. Options: 
            - 'powerlaw' (default) - a power law mass distribution, appropriate for fragmentation of rock.
            - 'gamma' - a gamma size distribution, appropriate for spraying of droplets (iron meteoroids).

    Return:
        frag_children: [list] A list of Fragment instances - these are the generated daughter fragments.

    """

    # Compute the mass bin coefficient
    mass_bin_coeff = 10**(-1.0/const.erosion_bins_per_10mass)

    # Compute the total number of mass bins across the specified mass range
    k = int(1 + math.log10(mass_min/mass_max)/math.log10(mass_bin_coeff))

    # Use the gamma distribution if specified (e.g. for iron meteoroids which spray droplets)
    if mass_model == 'gamma':

        # Compute the number of needed bins for the gamma distribution
        mass_bins = np.array([mass_max*(mass_bin_coeff**i) for i in range(k)])
        bin_widths = mass_bins*(1 - mass_bin_coeff)        

        # Compute the expected value from the mass power-law distribution 
        log_range = math.log(mass_max/mass_min)

        # The mass index has been adjusted to compute the peak of the gamma distribution
        # - For s = 1, the peak mass is the arithmetic mean of the min and max masses.
        # - For s = 2, the peak mass is the harmonic mean of the min and max masses.
        # - For other s, the peak mass is computed using the formula below
        if mass_index == 1.0:
            
            # For mass_index = 1, the peak is the arithmetic mean
            m_mean = (mass_max - mass_min)/log_range

        elif mass_index == 2.0:
            
            # For mass_index = 2, the peak is the harmonic mean
            m_mean = log_range/(1.0/mass_min - 1.0/mass_max)

        else:
            # For other mass indices, compute the mean using the formula, computing each step separatelly to save time
            a = 2 - mass_index
            b = 1 - mass_index
            m_max_a = mass_max**a
            m_min_a = mass_min**a
            m_max_b = mass_max**b
            m_min_b = mass_min**b

            num = (m_max_a - m_min_a)/a
            den = (m_max_b - m_min_b)/b
            m_mean = num/den

        # Convert mean mass to mean diameter using the formula for spherical grains
        D_mean = (6*m_mean/(math.pi*const.rho_grain))**(1/3)
        # print(f"Mean mass (kg): {m_mean:.4g}")
        # print(f"Mean diameter (µm): { D_mean*1e6:.4g}")

        # The gamma function value for 5/3, used in the gamma distribution
        gamma_5_3 = 0.90274529295093375313996375552960671484470367431640625  # gamma(5/3)
        s = (D_mean*gamma_5_3)**3

        # Compute the grain diameter and convert it to mass
        grain_diameter = (6*mass_bins/(math.pi*const.rho_grain))**(1/3)

        # Compute the number of grains in the bin for diameter distribution
        n_D = (3*grain_diameter **2/s)*np.exp(-grain_diameter **3/s)

        # Compute the derivative of the diameter with respect to mass
        dD_dm = (1/3)*(6/(math.pi*const.rho_grain))**(1/3)*mass_bins**(-2/3)

        # Compute the number of grains in the bin for unit mass distribution
        n_m_raw = n_D*np.abs(dD_dm)

        # Compute the mass per bin from the unit mass distribution
        mass_per_bin_raw = n_m_raw*bin_widths*mass_bins

        # Scale the number of grains in the bin to match the eroded mass
        scaling = eroded_mass/np.sum(mass_per_bin_raw)
        n_m_scaled = n_m_raw*scaling


    # Use the power-law mass distribution by default
    else:
        # Compute the number of the largest grains
        if mass_index == 2:
            n0 = eroded_mass/(mass_max*k)
        else:
            n0 = abs((eroded_mass/mass_max)*(1 - mass_bin_coeff**(2 - mass_index))/(1 - mass_bin_coeff**((2 - mass_index)*k)))


    # Go though every mass bin
    frag_children = []
    leftover_mass = 0
    for i in range(0, k):

        # Gamma size distribution (droplets)
        if mass_model == 'gamma':

            # Extract the mass of the grain in the bin
            m_grain = mass_bins[i]

            # Compute the number of grains in the bin
            n_grains_bin = n_m_scaled[i]*bin_widths[i] + leftover_mass/m_grain
            n_grains_bin_round = int(math.floor(n_grains_bin)) # int(expected_count)

            # Compute the leftover mass
            leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain

        # Power-law mass distribution
        else:
            # Compute the mass of all grains in the bin (per grain)
            m_grain = mass_max*mass_bin_coeff**i

            # Compute the number of grains in the bin
            n_grains_bin = n0*(mass_max/m_grain)**(mass_index - 1) + leftover_mass/m_grain
            n_grains_bin_round = int(math.floor(n_grains_bin))

            # Compute the leftover mass
            leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain

        # If there are any grains to erode, erode them
        if n_grains_bin_round > 0:

            # Init the new fragment with params of the parent
            frag_child = frag_parent.spawn_child()

            # Assign the number of grains this fragment stands for (make sure to preserve the previous value
            #   if erosion is done for more fragments)
            frag_child.n_grains *= n_grains_bin_round

            # Assign the grain mass
            frag_child.m = m_grain
            frag_child.m_init = m_grain

            frag_child.active = True
            frag_child.main = False
            frag_child.disruption_enabled = False

            # Indicate that the fragment is a grain
            if (not keep_eroding) and (not disruption):
                frag_child.grain = True

            # Set the erosion coefficient value (disable in grain, only larger fragments)
            if keep_eroding:
                frag_child.erosion_enabled = True

                # If the disruption occured, use a different erosion coefficient for daguhter fragments
                if disruption:
                    frag_child.erosion_coeff = const.disruption_erosion_coeff
                else:
                    frag_child.erosion_coeff = getErosionCoeff(const, frag_parent.h)

            else:
                # Compute the grain density and shape-density coeff
                frag_child.rho = const.rho_grain
                frag_child.updateShapeDensityCoeff()

                frag_child.erosion_enabled = False
                frag_child.erosion_coeff = 0


            # Give every fragment a unique ID
            frag_child.id = const.total_fragments
            const.total_fragments += 1

            frag_children.append(frag_child)


    return frag_children, const


def getErosionCoeff(const, h):
    """ Return the erosion coeff for the given height. """

    # Return the changed erosion coefficient
    if const.erosion_height_change >= h:
        return const.erosion_coeff_change

    # Return the starting erosion coeff
    elif const.erosion_height_start >= h:
        return const.erosion_coeff

    # If the height is above the erosion start height, return 0
    else:
        return 0


def killFragment(const, frag):
    """ Deactivate the given fragment and keep track of the stats. """

    frag.active = False
    const.n_active -= 1

    # Set the height when the main fragment was exhausted
    if frag.main:
        const.main_mass_exhaustion_ht = frag.h


def ablateAll(fragments, const, compute_wake=False, wake_heights_queue=None):
    """ Perform single body ablation of all fragments using the 4th order Runge-Kutta method. 

    Arguments:
        fragments: [list] A list of Fragment instances.
        const: [object] Constants instance.

    Keyword arguments:
        compute_wake: [bool] If True, the wake profile will be computed. False by default.
        wake_heights_queue: [list] A list of heights at which the wake should be computed. None by default.

    Return:
        ...
    """

    # Keep track of the total luminosity
    luminosity_total = 0.0

    # Keep track of the total luminosity weighted lum eff
    tau_total = 0.0

    # Keep track of the luminosity of the main fragment
    luminosity_main = 0.0

    # Keep track of the luminosity weighted lum eff of the main fragment
    tau_main = 0.0

    # Keep track of the luminosity of eroded and disrupted fragments
    luminosity_eroded = 0.0

    # Keep track of the luminosity weighted lum eff of eroded and disrupted fragments
    tau_eroded = 0.0

    # Keep track of the total electron density
    electron_density_total = 0.0

    # Keep track of parameters of the brightest fragment
    brightest_height = 0.0
    brightest_length = 0.0
    brightest_lum    = 0.0
    brightest_vel    = 0.0

    # Keep track of the the main fragment parameters
    main_mass = 0.0
    main_height = 0.0
    main_length = 0.0
    main_vel = 0.0
    main_dyn_press = 0.0

    frag_children_all = []

    # Go through all active fragments
    for frag in fragments:

        # Skip the fragment if it's not active
        if not frag.active:
            continue

        # Get atmosphere density for the given height
        rho_atm = atmDensityPoly(frag.h, const.dens_co)

        # Compute the mass loss of the fragment due to ablation
        mass_loss_ablation = massLossRK4(const.dt, frag.K, frag.sigma, frag.m, rho_atm, frag.v)

        # Compute the mass loss due to erosion
        if frag.erosion_enabled and (frag.erosion_coeff > 0):
            mass_loss_erosion = massLossRK4(const.dt, frag.K, frag.erosion_coeff, frag.m, rho_atm, frag.v)
        else:
            mass_loss_erosion = 0

        # Compute the total mass loss
        mass_loss_total = mass_loss_ablation + mass_loss_erosion

        # If the total mass after ablation in this step is below zero, ablate what's left of the whole mass
        if (frag.m + mass_loss_total) < 0:
            mass_loss_total = mass_loss_total + frag.m

        # Compute new mass
        m_new = frag.m + mass_loss_total

        # Compute change in velocity
        deceleration_total = decelerationRK4(const.dt, frag.K, frag.m, rho_atm, frag.v)

        # If the deceleration is negative (i.e. the fragment is accelerating), then stop the fragment
        if deceleration_total > 0:
            frag.vv = frag.vh = frag.v = 0
            deceleration_total = 0

        # Otherwise update the velocity
        else:

            # Compute g at given height
            gv = G0/((1 + frag.h/const.r_earth)**2)

            # ### Add velocity change due to Earth's gravity ###

            # # Vertical component of a
            # av = -gv - deceleration_total*frag.vv/frag.v + frag.vh*frag.v/(const.r_earth + frag.h)

            # # Horizontal component of a
            # ah = -deceleration_total*frag.vh/frag.v - frag.vv*frag.v/(const.r_earth + frag.h)

            # ### ###

            ### Compute deceleration without the effects of gravity (to reconstruct the initial velocity
            # without the gravity component)

            # Vertical component of a
            av = -deceleration_total*frag.vv/frag.v + frag.vh*frag.v/(const.r_earth + frag.h)

            # Horizontal component of a
            ah = -deceleration_total*frag.vh/frag.v - frag.vv*frag.v/(const.r_earth + frag.h)

            ###

            # Compute the drop due to gravity
            h_grav_drop = 0.5*gv*const.dt**2

            # Track the total drop due to gravity
            frag.h_grav_drop_total += h_grav_drop

            # Update the velocity
            frag.vv -= av*const.dt
            frag.vh -= ah*const.dt
            frag.v = math.sqrt(frag.vh**2 + frag.vv**2)

            # Only allow the meteoroid to go down, and stop the ablation if it stars going up
            if frag.vv > 0:

                frag.vv = 0

                # Setting the height to zero will stop the ablation during the if catch below
                frag.h = 0

        # Update length along the track
        frag.length += frag.v*const.dt

        # Update the mass
        frag.m = m_new

        # Old way of computing height which did not include the curvature of the Earth
        # frag.h = frag.h + frag.vv*const.dt

        # Compute the height taking the curvature of the Earth and the gravity drop into account
        frag.h = heightCurvature(const.h_init, const.zenith_angle, frag.length, const.r_earth)
        frag.h -= frag.h_grav_drop_total

        # Get the luminous efficiency
        tau = luminousEfficiency(const.lum_eff_type, const.lum_eff, frag.v, frag.m)

        # # Compute luminosity for one grain/fragment (without the deceleration term)
        # lum = -tau*((mass_loss_ablation/const.dt*frag.v**2)/2)

        # Compute luminosity for one grain/fragment (with the deceleration term)
        # NOTE: The deceleration term can sometimes be numerically unstable for some reason...
        lum = -tau*((mass_loss_ablation/const.dt*frag.v**2)/2 + frag.m*frag.v*deceleration_total)

        # Compute the electron line density
        beta = ionizationEfficiency(frag.v)
        q = -beta*(mass_loss_ablation/const.dt)/(const.mu*frag.v)

        # Compute the total luminosity
        frag.lum = lum*frag.n_grains

        # Compute the total electron line density
        frag.q = q*frag.n_grains

        # Keep track of the total luminosity across all fragments
        luminosity_total += frag.lum

        # Keep track of the total number of produced electrons
        electron_density_total += frag.q

        # Keep track of the total luminosity weighted lum eff
        tau_total += tau*frag.lum

        # Compute aerodynamic loading on the grain (always assume Gamma = 1.0)
        # dyn_press = frag.gamma*rho_atm*frag.v**2
        dyn_press = 1.0*rho_atm*frag.v**2
        frag.dyn_press = dyn_press

        # if frag.id == 0:
        #     print('----- id:', frag.id)
        #     print('t:', const.total_time)
        #     print('V:', frag.v/1000)
        #     print('H:', frag.h/1000)
        #     print('m:', frag.m)
        #     print('DynPress:', dyn_press/1000, 'kPa')

        # Keep track of the parameters of the main fragment
        if frag.main:
            luminosity_main = frag.lum
            tau_main = tau
            main_mass = frag.m
            main_height = frag.h
            main_length = frag.length
            main_vel = frag.v
            main_dyn_press = dyn_press

        # If the fragment is done, stop ablating
        if  (
            (frag.m <= const.m_kill) 
            or (frag.v < const.v_kill) 
            or (frag.h < const.h_kill) 
            or (frag.lum < 0)
            or ((const.len_kill > 0) and (frag.length > const.len_kill))
            ):

            killFragment(const, frag)

            # print('Killing', frag.id)
            continue

        # Keep track of the brightest fragment
        if frag.lum > brightest_lum:
            brightest_lum = lum
            brightest_height = frag.h
            brightest_length = frag.length
            brightest_vel = frag.v

        # For fragments born out of complex fragmentation, keep track of their luminosity and height
        if not frag.main:

            if const.fragmentation_show_individual_lcs: 

                # Keep track of magnitudes of complex fragmentation fragments
                if frag.complex:

                    # Find the corresponding fragmentation entry
                    frag_entry = next((x for x in const.fragmentation_entries if x.id == frag.complex_id), \
                        None)

                    if frag_entry is not None:

                        # Store luminosity of grains
                        if frag.grain:

                            add_new_entry = False

                            # Check if the last time entry corresponds to the current time, and add to it
                            if not len(frag_entry.grains_time_data):
                                add_new_entry = True
                            elif const.total_time != frag_entry.grains_time_data[-1]:
                                add_new_entry = True

                            # Add the current integration time
                            if add_new_entry:
                                frag_entry.grains_time_data.append(const.total_time)
                                frag_entry.grains_luminosity.append(frag.lum)
                                frag_entry.grains_tau_over_lum.append(tau*frag.lum)

                            # Add to the total luminosity at the current time step that's already been added
                            else:
                                frag_entry.grains_luminosity[-1] += frag.lum
                                frag_entry.grains_tau_over_lum[-1] += tau*frag.lum

                        # Store parameters of the main fragment
                        else:

                            add_new_entry = False

                            # Check if the last time entry corresponds to the current time, and add to it
                            if not len(frag_entry.main_time_data):
                                add_new_entry = True
                            elif const.total_time != frag_entry.main_time_data[-1]:
                                add_new_entry = True

                            # Add the current integration time
                            if add_new_entry:
                                frag_entry.main_time_data.append(const.total_time)
                                frag_entry.main_luminosity.append(frag.lum)
                                frag_entry.main_tau_over_lum.append(tau*frag.lum)

                            # Add to the total luminosity at the current time step that's already been added
                            else:
                                frag_entry.main_luminosity[-1] += frag.lum
                                frag_entry.main_tau_over_lum[-1] += tau*frag.lum

                # Keep track of luminosity of eroded and disrupted fragments ejected directly from the main
                #   fragment
                else:

                    luminosity_eroded += frag.lum
                    tau_eroded += tau*frag.lum

        # For non-complex fragmentation only: Check if the erosion should start, given the height,
        #   and create grains
        if (not frag.complex) and (frag.h < const.erosion_height_start) and frag.erosion_enabled \
            and const.erosion_on:

            # Turn on the erosion of the fragment
            frag.erosion_coeff = getErosionCoeff(const, frag.h)

            # Update the main fragment physical parameters if it is changed after erosion coefficient change
            if frag.main and (const.erosion_height_change >= frag.h):

                # Update the density
                frag.rho = const.erosion_rho_change
                frag.updateShapeDensityCoeff()

                # Update the ablation coeff
                frag.sigma = const.erosion_sigma_change

        # Create grains for erosion-enabled fragments
        if frag.erosion_enabled:

            # Generate new grains if there is some mass to distribute
            if abs(mass_loss_erosion) > 0:

                grain_children, const = generateFragments(const, frag, abs(mass_loss_erosion), \
                    frag.erosion_mass_index, frag.erosion_mass_min, frag.erosion_mass_max, \
                    keep_eroding=False, mass_model=const.erosion_grain_distribution)

                const.n_active += len(grain_children)
                frag_children_all += grain_children

                # print('Eroding id', frag.id)
                # print('Eroded mass: {:e}'.format(abs(mass_loss_erosion)))
                # print('Mass distribution:')
                # grain_mass_sum = 0
                # for f in frag_children:
                #     print('    {:d}: {:e} kg'.format(f.n_grains, f.m))
                #     grain_mass_sum += f.n_grains*f.m
                # print('Grain total mass: {:e}'.format(grain_mass_sum))

                # Record physical parameters at the beginning of erosion for the main fragment
                if frag.main:
                    if const.erosion_beg_vel is None:

                        const.erosion_beg_vel = frag.v
                        const.erosion_beg_mass = frag.m
                        const.erosion_beg_dyn_press = dyn_press

                    # Record the mass when erosion is changed
                    elif (const.erosion_height_change >= frag.h) and (const.mass_at_erosion_change is None):
                        const.mass_at_erosion_change = frag.m

        # Disrupt the fragment if the dynamic pressure exceeds its strength
        if frag.disruption_enabled and const.disruption_on:
            if dyn_press > const.compressive_strength:

                # Compute the mass that should be disrupted into fragments
                mass_frag_disruption = frag.m*(1 - const.disruption_mass_grain_ratio)

                fragments_total_mass = 0
                if mass_frag_disruption > 0:

                    # Disrupt the meteoroid into fragments
                    disruption_mass_min = const.disruption_mass_min_ratio*mass_frag_disruption
                    disruption_mass_max = const.disruption_mass_max_ratio*mass_frag_disruption

                    # Generate larger fragments, possibly assign them a separate erosion coefficient
                    frag_children, const = generateFragments(const, frag, mass_frag_disruption, \
                        const.disruption_mass_index, disruption_mass_min, disruption_mass_max, \
                        keep_eroding=const.erosion_on, disruption=True, 
                        mass_model=const.erosion_grain_distribution)

                    frag_children_all += frag_children
                    const.n_active += len(frag_children)

                    # Compute the mass that went into fragments
                    fragments_total_mass = sum([f.n_grains*f.m for f in frag_children])

                    # Assign the height of disruption
                    const.disruption_height = frag.h

                    print('Disrupting id', frag.id)
                    print('Height: {:.3f} km'.format(const.disruption_height/1000))
                    print('Disrupted mass: {:e}'.format(mass_frag_disruption))
                    print('Mass distribution:')
                    for f in frag_children:
                        print('{:4d}: {:e} kg'.format(f.n_grains, f.m))
                    print('Disrupted total mass: {:e}'.format(fragments_total_mass))

                # Disrupt a portion of the leftover mass into grains
                mass_grain_disruption = frag.m - fragments_total_mass
                if mass_grain_disruption > 0:
                    grain_children, const = generateFragments(const, frag, mass_grain_disruption, 
                        frag.erosion_mass_index, frag.erosion_mass_min, frag.erosion_mass_max, \
                        keep_eroding=False, mass_model=const.erosion_grain_distribution)

                    frag_children_all += grain_children
                    const.n_active += len(grain_children)

                # Deactive the disrupted fragment
                frag.m = 0
                killFragment(const, frag)

        # Handle complex fragmentation and status changes of the main fragment
        if frag.main and const.fragmentation_on:

            # Get a list of complex fragmentations that are still to do
            frags_to_do = [frag_entry for frag_entry in const.fragmentation_entries if not frag_entry.done]

            if len(frags_to_do):

                # Go through all fragmentations that needs to be performed
                for frag_entry in frags_to_do:

                    # Check if the height of the main fragment is right to perform the operation.
                    # Run if:
                    # (a) If the fireball is going down and the fragmentation is for the downward direction
                    # (b) If the fireball is going up and the fragmentation is for the upward direction
                    #     And the fireball started going up
                    if ( (not frag_entry.upward_only) and (frag.h < frag_entry.height) ) \
                    or ( 
                        frag_entry.upward_only 
                        and (frag.h > frag_entry.height) 
                        and (frag.h > const.main_bottom_ht)
                        ):
                        
                        parent_initial_mass = frag.m

                        # Change parameters of all fragments
                        if frag_entry.frag_type == "A":

                            for frag_tmp in (fragments + frag_children_all + [frag]):

                                # Update the ablation coefficient
                                if frag_entry.sigma is not None:
                                    frag_tmp.sigma = frag_entry.sigma

                                # Update the drag coefficient
                                if frag_entry.gamma is not None:
                                    frag_tmp.gamma = frag_entry.gamma
                                    frag_tmp.updateShapeDensityCoeff()

                        # Change the parameters of the main fragment
                        if frag_entry.frag_type == "M":

                            if frag_entry.sigma is not None:
                                frag.sigma = frag_entry.sigma

                            if frag_entry.erosion_coeff is not None:
                                frag.erosion_coeff = frag_entry.erosion_coeff

                            if frag_entry.mass_index is not None:
                                frag.erosion_mass_index = frag_entry.mass_index

                            if frag_entry.grain_mass_min is not None:
                                frag.erosion_mass_min = frag_entry.grain_mass_min

                            if frag_entry.grain_mass_max is not None:
                                frag.erosion_mass_max = frag_entry.grain_mass_max

                        # Create a new single-body or eroding fragment
                        if (frag_entry.frag_type == "F") or (frag_entry.frag_type == "EF"):

                            # Go through all new fragments
                            for frag_num in range(frag_entry.number):

                                # Mass of the new fragment
                                new_frag_mass = parent_initial_mass*(frag_entry.mass_percent/100.0)/frag_entry.number
                                frag_entry.mass = new_frag_mass*frag_entry.number

                                # Decrease the parent mass
                                frag.m -= new_frag_mass

                                # Create the new fragment
                                frag_new = frag.spawn_child()
                                frag_new.active = True
                                frag_new.main = False
                                frag_new.disruption_enabled = False

                                # Indicate that the fragments are born out of complex fragmentation
                                frag_new.complex = True

                                # Assign the complex fragmentation ID
                                frag_new.complex_id = frag_entry.id

                                # Assing the mass to the new fragment
                                frag_new.m = new_frag_mass

                                # Assign possible new ablation coeff to this fragment
                                if frag_entry.sigma is not None:
                                    frag_new.sigma = frag_entry.sigma

                                # If the fragment is eroding, set erosion parameters
                                if frag_entry.frag_type == "EF":
                                    frag_new.erosion_enabled = True

                                    frag_new.erosion_coeff = frag_entry.erosion_coeff

                                    frag_new.erosion_mass_index = frag_entry.mass_index
                                    frag_new.erosion_mass_min = frag_entry.grain_mass_min
                                    frag_new.erosion_mass_max = frag_entry.grain_mass_max

                                else:
                                    # Disable erosion for single-body fragments
                                    frag_new.erosion_enabled = False

                                # Add the new fragment to the list of childern
                                frag_children_all.append(frag_new)
                                const.n_active += 1

                        # Release dust
                        if frag_entry.frag_type == "D":

                            # Compute the mass of the dust
                            dust_mass = frag.m*(frag_entry.mass_percent/100.0)
                            frag_entry.mass = dust_mass

                            # Subtract from the parent mass
                            frag.m -= dust_mass

                            # Create the new fragment
                            frag_new = frag.spawn_child()
                            frag_new.active = True
                            frag_new.main = False
                            frag_new.disruption_enabled = False

                            # Indicate that the fragments are born out of complex fragmentation
                            frag_new.complex = True

                            # Assign the complex fragmentation ID
                            frag_new.complex_id = frag_entry.id

                            # Generate dust grains
                            grain_children, const = generateFragments(const, frag_new, dust_mass, \
                                frag_entry.mass_index, frag_entry.grain_mass_min, frag_entry.grain_mass_max, \
                                keep_eroding=False, mass_model=const.erosion_grain_distribution)

                            # Add fragments to the list
                            frag_children_all += grain_children
                            const.n_active += len(grain_children)

                        # Set the fragmentation as finished
                        frag_entry.done = True

                        # Set physical conditions at the moment of fragmentation
                        frag_entry.time = const.total_time
                        frag_entry.dyn_pressure = dyn_press
                        frag_entry.velocity = frag.v
                        frag_entry.parent_mass = parent_initial_mass

        # If the fragment is done, stop ablating
        if (frag.m <= const.m_kill):

            killFragment(const, frag)
            # print('Killing', frag.id)

            continue

    # Track the leading fragment length
    active_fragments = [frag for frag in fragments if frag.active]
    if len(active_fragments):
        leading_frag = max(active_fragments, key=lambda x: x.length)
        leading_frag_length    = leading_frag.length
        leading_frag_height    = leading_frag.h
        leading_frag_vel       = leading_frag.v
        leading_frag_dyn_press = leading_frag.dyn_press
    else:
        leading_frag_length    = None
        leading_frag_height    = None
        leading_frag_vel       = None
        leading_frag_dyn_press = None

    ### Compute the wake profile ###
    
    # If the specific wake heights are given, check if the current height is below the next wake height
    if (wake_heights_queue is not None) and (leading_frag_height is not None):

        # If there are any heights left in the queue
        if len(wake_heights_queue):
            
            # If the current height is below the next wake height, compute the wake
            if leading_frag_height <= wake_heights_queue[0]:
                compute_wake = True
                
                # Pop all heights that are above the current height (including the one we just passed)
                while len(wake_heights_queue) and (leading_frag_height <= wake_heights_queue[0]):
                    wake_heights_queue.pop(0)

            else:
                compute_wake = False
        
        else:
            compute_wake = False



    if compute_wake and (leading_frag_length is not None):

        # Evaluate the Gaussian from +3 sigma in front of the leading fragment to behind
        front_len = leading_frag_length + 3*const.wake_psf[0]
        back_len = leading_frag_length - const.wake_extension

        ### Compute the wake as convoluted luminosities with the PSF ###

        length_array = np.linspace(back_len, front_len, 500) - leading_frag_length

        frag_list = []

        for frag in fragments:

            # Take only those lengths inside the wake window
            if frag.active:
                if (frag.length > back_len) and (frag.length < front_len):
                    frag_list.append(frag.spawn_child())

        # Store evaluated wake
        wake = Wake(const, frag_list, leading_frag_length, length_array)

        ### ###

    else:
        wake = None

    ### ###

    # Add generated fragment children to the list of fragments
    fragments += frag_children_all

    # Compute the total mass of all active fragments
    active_fragments = [frag.m for frag in fragments if frag.active]
    if len(active_fragments):
        mass_total_active = np.sum(active_fragments)
    else:
        mass_total_active = 0.0

    # Increment the running time
    const.total_time += const.dt

    # Weigh the tau by luminosity
    if luminosity_total > 0:
        tau_total /= luminosity_total
    else:
        tau_total = 0

    if luminosity_eroded > 0:
        tau_eroded /= luminosity_eroded
    else:
        tau_eroded = 0

    return fragments, const, luminosity_total, luminosity_main, luminosity_eroded, electron_density_total, \
        tau_total, tau_main, tau_eroded, brightest_height, brightest_length, brightest_vel, \
        leading_frag_height, leading_frag_length, leading_frag_vel, leading_frag_dyn_press, \
        mass_total_active, main_mass, main_height, main_length, main_vel, main_dyn_press, wake


def _runSimulationRK4Reference(const, compute_wake=False):
    """ ORIGINAL RK4 implementation, kept as an internal reference for porting erosion/disruption/
    complex-fragmentation logic in Stages 3-5 (not part of the public API - see runSimulation()
    below for the analytic engine's actual entry point). Not called by anything in this module. """

    # Ensure that the grain mass min is smaller than the grain mass max
    if const.erosion_mass_min > const.erosion_mass_max:
        const.erosion_mass_min, const.erosion_mass_max = const.erosion_mass_max, const.erosion_mass_min

    ###


    if const.fragmentation_on:

        # Assign unique IDs to complex fragmentation entries
        for i, frag_entry in enumerate(const.fragmentation_entries):
            frag_entry.id = i

            # Reset output parameters for every fragmentation entry
            frag_entry.resetOutputParameters()


    fragments = []

    # Init the main fragment
    frag = Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, const.zenith_angle, \
        const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)
    frag.main = True
    
    # Erode the main fragment
    frag.erosion_enabled = True

    # Disrupt the main fragment
    frag.disruption_enabled = True

    fragments.append(frag)



    # Reset simulation parameters
    const.total_time = 0
    const.n_active = 1
    const.total_fragments = 1
    const.main_bottom_ht = const.h_init


    ###



    # Check that the grain density is larger than the bulk density, and if not, set the grain density
    #   to be the same as the bulk density
    if const.rho > const.rho_grain:
        const.rho_grain = const.rho


    # If the wake heights are given, sort them by height descending
    wake_heights_queue = None
    if (const.wake_heights is not None) and compute_wake:
        wake_heights_queue = sorted(const.wake_heights, reverse=True)


    # Run the simulation until all fragments stop ablating
    results_list = []
    wake_results = []
    while const.n_active > 0:

        # Ablate the fragments
        fragments, const, luminosity_total, luminosity_main, luminosity_eroded, electron_density_total, \
            tau_total, tau_main, tau_eroded, brightest_height, brightest_length, brightest_vel, \
            leading_frag_height, leading_frag_length, leading_frag_vel, leading_frag_dyn_press, \
            mass_total_active, main_mass, main_height, main_length, main_vel, main_dyn_press, \
            wake = ablateAll(fragments, const, compute_wake=compute_wake, wake_heights_queue=wake_heights_queue)
        
        # Track the bottom height of the main fragment
        if main_height > 0:
            const.main_bottom_ht = min(main_height, const.main_bottom_ht)

        # Store wake estimation results
        wake_results.append(wake)

        # Stack results list
        results_list.append([const.total_time, luminosity_total, luminosity_main, luminosity_eroded, \
            electron_density_total, tau_total, tau_main, tau_eroded, brightest_height, brightest_length, \
            brightest_vel, leading_frag_height, leading_frag_length, leading_frag_vel, \
            leading_frag_dyn_press, mass_total_active, main_mass, main_height, main_length, main_vel, \
            main_dyn_press])



    # Find the main fragment and return it with results
    frag_main = None
    for frag in fragments:
        if frag.main:
            frag_main = frag
            break


    ### Find the fragments born out of complex fragmentations and assign them to the fragmentation entries ###

    # Reset all fragment lists for entries
    for frag_entry in const.fragmentation_entries:
        frag_entry.fragments = []

    # Find fragments for every fragmentation
    for frag_entry in const.fragmentation_entries:
        for frag in fragments:
            if not frag.grain:
                if frag.complex_id is not None:
                    if frag_entry.id == frag.complex_id:

                        # Add fragment
                        frag_entry.fragments.append(frag)

                        # Compute the final mass of all fragments in this fragmentation after ablation stopped
                        final_mass = frag_entry.number*frag.m

                        # If the final mass is below a gram, assume it's zero
                        if final_mass < 1e-3:
                            final_mass = None

                        # Assign the final mass to the fragmentation entry
                        frag_entry.final_mass = final_mass


    ### ###


    return frag_main, results_list, wake_results


def _massLossRate(K, sigma, m, rho_atm, v):
    """ Pure-Python mirror of MetSimErosionCyTools.pyx's private massLoss() (dm/dt). That function
    is cdef (C-only), not callable from Python, so this is a small, exact reimplementation of the
    one-line formula rather than a modification of the existing Cython file (out of this task's
    scope - see the implementation plan's "Reuse plan" section). """

    return -K*sigma*m**(2/3.0)*rho_atm*v**3


def _decelerationRate(K, m, rho_atm, v):
    """ Pure-Python mirror of MetSimErosionCyTools.pyx's private deceleration() (dv/dt) - see
    _massLossRate()'s docstring for why this is a reimplementation, not an import. """

    return -K*m**(-1/3.0)*rho_atm*v**2


def _cumulativeErodedMass(v_n, v_start, sigma_eff, erosion_coeff, m_start):
    """ Closed-form cumulative mass (kg) lost to EROSION ONLY (not ablation) since the start of an
    eroding segment, as a function of normalized velocity - Stage 3's key closed-form building
    block (implementation plan's "Erosion grain spawning" section): since
    massLoss(K, sigma, ...) and massLoss(K, erosion_coeff, ...) share identical (K, m, v, rho_atm)
    inputs at every instant within a segment (both sigma and erosion_coeff are constant there by
    construction - that is what defines a segment boundary), massLoss's linearity in its
    coefficient argument means the erosion channel's share of any total mass lost is the constant
    ratio erosion_coeff/sigma_eff, at every point on the segment - no step-by-step tracking needed,
    just this ratio applied to the already-closed-form total mass lost
    (m_start - massFromVelocityNormed(v_n, ...)).

    Arguments:
        v_n: [float or ndarray] Velocity normalized to v_start.
        v_start: [float] Velocity at the start of this segment (m/s).
        sigma_eff: [float] sigma_own + erosion_coeff for this segment (s^2/m^2) - must be the SAME
            value the segment's AnalyticTrajectory was built with.
        erosion_coeff: [float] This segment's erosion coefficient (s^2/m^2).
        m_start: [float] Mass at the start of this segment (kg).

    Return:
        [float or ndarray] Cumulative eroded mass (kg, positive), same shape as v_n.
    """

    total_mass_lost = m_start - massFromVelocityNormed(v_n, v_start, sigma_eff, m_start)

    return (erosion_coeff/sigma_eff)*total_mass_lost


def _vnAtCumulativeErodedMass(target_eroded_mass, v_start, sigma_eff, erosion_coeff, m_start):
    """ Closed-form inverse of _cumulativeErodedMass(): the normalized velocity at which exactly
    target_eroded_mass has been eroded since the start of this segment. Derived by directly
    inverting massFromVelocityNormed() through _cumulativeErodedMass()'s formula (a single
    exp/log, not a root-find - see _cumulativeErodedMass()'s docstring for the derivation this
    inversion continues).

    Arguments:
        target_eroded_mass: [float or ndarray] Target cumulative eroded mass (kg).
        v_start, sigma_eff, erosion_coeff, m_start: [float] See _cumulativeErodedMass().

    Return:
        [float or ndarray] Normalized velocity v_n, same shape as target_eroded_mass.
    """

    ratio = erosion_coeff/sigma_eff

    return np.sqrt(1.0 + (2.0/(sigma_eff*v_start**2))
        *np.log(1.0 - np.asarray(target_eroded_mass)/(ratio*m_start)))


def _timeAtReportedHeight(traj, h_target, length_start, const, xtol=1e-3, grav_drop_start=0.0):
    """ Find the time (s) at which this segment's reported (curved-Earth + gravity-drop corrected)
    height crosses h_target - used to find erosion_height_start/erosion_height_change crossings
    (Stage 3) the same way the kill-search loop already finds h_kill crossings, just needed here
    as a segment-boundary lookup rather than a termination condition.

    Uses traj's own already-built Chebyshev grid (reportedHeightAt() at those points) for a coarse
    bracket, refined with brentq for precision - mirrors _estimateTimeAtHKill()'s approach, but
    exact rather than a sizing heuristic, since segment-transition state (v/m/h/length) needs to be
    continuous to the next segment.

    Assumes reported height is monotonically non-increasing over the segment - already guaranteed
    within _buildRefinedTrajectory()'s supported domain (it raises NotImplementedError otherwise).

    Arguments:
        traj: [AnalyticTrajectory] Already built and refined.
        h_target: [float] Reported height (m) to find the crossing time for.
        length_start: [float] Path length (m) accumulated before this segment started.
        const: [Constants] For h_init, zenith_angle.

    Keyword arguments:
        xtol: [float] brentq absolute tolerance on time (s).
        grav_drop_start: [float] See reportedHeightAt()'s own docstring - gravity-drop (m) already
            accumulated before this segment started.

    Return:
        [float or None] Time (s) of the crossing, or None if h_target is never reached within this
        segment's tabulated domain [traj.t_start, traj.t_hi] (the caller should then treat this
        segment as running to its own natural end instead).
    """

    t_grid = traj._t_grid
    h_grid = reportedHeightAt(traj, t_grid, const.h_init, const.zenith_angle,
        length_start=length_start, grav_drop_start=grav_drop_start)

    if (h_target < h_grid[-1]) or (h_target > h_grid[0]):
        return None

    # h_grid is descending (paired with ascending t_grid) - reverse both for np.interp's ascending-x
    # requirement.
    t_est = float(np.interp(h_target, h_grid[::-1], t_grid[::-1]))

    def _hDiff(t):
        return float(reportedHeightAt(traj, np.array([t]), const.h_init, const.zenith_angle,
            length_start=length_start, grav_drop_start=grav_drop_start)[0]) - h_target

    span = t_grid[-1] - t_grid[0]
    bracket_half_width = max(0.02*span, 1e-3)

    t_lo = max(t_est - bracket_half_width, t_grid[0])
    t_hi = min(t_est + bracket_half_width, t_grid[-1])
    f_lo, f_hi = _hDiff(t_lo), _hDiff(t_hi)

    expansions = 0
    while (f_lo*f_hi > 0) and (expansions < 20):
        bracket_half_width *= 2
        t_lo = max(t_est - bracket_half_width, t_grid[0])
        t_hi = min(t_est + bracket_half_width, t_grid[-1])
        f_lo, f_hi = _hDiff(t_lo), _hDiff(t_hi)
        expansions += 1

    if f_lo*f_hi > 0:
        # Could not bracket a sign change (e.g. h_target sits right at t_grid's own edge) - the
        # coarse interpolated estimate is the best available answer rather than failing outright.
        return t_est

    return float(scipy.optimize.brentq(_hDiff, t_lo, t_hi, xtol=xtol))


def _buildRefinedTrajectory(K, sigma_eff, m_start, v_start, h_real_start, sin_slope, atm_map,
        t_start, length_start, const, v_n_floor, h_real_floor, n_refine_passes=3,
        grav_drop_start=0.0):
    """ Build an AnalyticTrajectory for one segment (a main-fragment erosion phase or a grain) and
    run it through the iterative curvature/atmosphere refinement - generalizes Stage 2c's original
    runSimulation()-local closure (hardcoded to the whole simulation's t=0/h_init/v_init/m_init
    start) to an arbitrary finite-start segment, so the same machinery builds the main fragment's
    chained erosion segments AND every spawned grain's own trajectory (Stage 3).

    The physical reasoning is unchanged from Stage 2c: the first pass uses flat-earth height for
    atmosphere lookups, which can diverge from the true curved-Earth + gravity-drop height enough
    to matter for mass (an EXPONENTIAL function of velocity); each subsequent pass feeds the
    previous pass's own reportedHeightAt() back in as the atmosphere-lookup height, converging in a
    small fixed number of passes (flat and true height are exactly equal at t=t_start by
    construction, regardless of what t_start is).

    Arguments:
        K: [float] Shape-density coefficient for this segment (m^2/kg^(2/3)).
        sigma_eff: [float] Effective ablation coefficient DRIVING THIS SEGMENT'S DYNAMICS
            (sigma_own + erosion_coeff for an eroding main-fragment segment, sigma_own alone for a
            non-eroding segment or grain). NOT necessarily the same value _evaluateSegment() should
            be given for luminosity reconstruction - see that function's docstring.
        m_start, v_start, h_real_start: [float] State at the start of this segment.
        sin_slope: [float] sin(entry slope) = cos(zenith_angle) - shared by every segment/grain in
            a simulation (one straight line of sight throughout).
        atm_map: [AtmEquivHeightMap] Shared across the whole simulation (depends only on dens_co).
        t_start: [float] Time at the start of this segment (s), on the simulation's global clock.
        length_start: [float] Path length (m) already accumulated before this segment started -
            needed so reportedHeightAt()'s heightCurvature() call uses the TRUE cumulative length,
            not one reset to 0 at each segment boundary (matching how frag.length is never reset
            across a fragment's/child's lifetime in the original, MetSimErosion.py:373-384).
        const: [Constants] For h_init, zenith_angle, dt, r_earth.
        v_n_floor, h_real_floor: [float] See AnalyticTrajectory's own docstrings.
        n_refine_passes: [int] Number of iterative refinement passes (3 by default, Stage 2c's
            validated choice for the main fragment's first segment). Grains are typically much
            shorter-lived, so may tolerate fewer passes - not yet validated either way, so this
            defaults to the same value rather than assuming it's safe to cut corners.
        grav_drop_start: [float] See reportedHeightAt()'s own docstring - gravity-drop (m) already
            accumulated before this segment started, threaded into the internal atm_height_fn
            construction below (built from THIS segment's own reportedHeightAt(), which needs its
            true cumulative gravity-drop for the same reason the final reported height does -
            Stage 4's disruption daughters, which can inherit a substantial pre-spawn total from a
            long-lived parent, are what makes this matter beyond the sub-meter, dt-scaled effect
            already measured for a single long segment).
    Returns:
        (traj, atm_height_fn): the final (refined) AnalyticTrajectory and the atm_height_fn it was
        built with - both needed by _evaluateSegment() for the rate-term reconstruction.
    """

    traj = AnalyticTrajectory(K, sigma_eff, m_start, v_start, h_real_start, sin_slope, atm_map,
        t_start=t_start, sim_dt=const.dt, r_earth=const.r_earth, v_n_floor=v_n_floor,
        h_real_floor=h_real_floor)

    atm_height_fn = None

    for _ in range(n_refine_passes):
        traj_prev = traj
        h_atm_prev = reportedHeightAt(traj_prev, traj_prev._t_grid, const.h_init, const.zenith_angle,
            length_start=length_start, grav_drop_start=grav_drop_start)

        # See runSimulation()'s original closure (implementation plan, Stage 2c) for the full
        # derivation of why a reported height above h_init means this segment is not yet supported.
        if np.any(h_atm_prev > const.h_init):
            raise NotImplementedError(
                "Equivalent-atmosphere iterative refinement is currently only supported for "
                "trajectories whose reported (curved-Earth + gravity-drop) height stays at or "
                "below h_init throughout. This segment's reported height exceeds h_init "
                f"(max {np.max(h_atm_prev):.1f} m vs h_init={const.h_init:.1f} m), which happens "
                "for extremely grazing entries whose path length exceeds "
                "2*(h_init+r_earth)*cos(zenith_angle) - a real geometric property of "
                "heightCurvature(), not a bug. Not yet supported by the analytic engine; use "
                "MetSimErosion.runSimulation() for such near-tangential entries.")

        h_real_prev_asc = traj_prev._h_real_grid[::-1]
        h_atm_prev_asc = h_atm_prev[::-1]

        atm_height_interp = scipy.interpolate.interp1d(h_real_prev_asc, h_atm_prev_asc,
            kind="linear", fill_value="extrapolate", assume_sorted=True)

        def atm_height_fn(h_real_query, f=atm_height_interp):
            return f(h_real_query)

        traj = AnalyticTrajectory(K, sigma_eff, m_start, v_start, h_real_start, sin_slope, atm_map,
            t_start=t_start, sim_dt=const.dt, r_earth=const.r_earth, atm_height_fn=atm_height_fn,
            v_n_floor=v_n_floor, h_real_floor=h_real_floor)

    return traj, atm_height_fn


def _buildBatchedDaughterTrajectories(K, sigma_eff, masses, v_start, h_real_start, sin_slope,
        atm_map, t_start, length_start, const, v_n_floor, h_real_floor, grav_drop_start=0.0):
    """ Batched counterpart to _buildRefinedTrajectory(), for N daughters spawned from the SAME
    disruption event. Confirmed from source (generateFragments()'s spawn_child() dict-copy) that
    every such daughter shares the disrupting parent's own IDENTICAL (K, sigma_eff, v_start,
    h_real_start, t_start, length_start, grav_drop_start) - the mass-binning loop only ever
    overwrites frag_child.m per bin - so beta and h_equiv_start are shared too; only alpha (via
    mass, alpha ~ m^(-1/3)) differs per daughter.

    Only n_refine_passes=1 is supported (Stage 7's own validated default for daughters - see
    _buildDaughterFragmentSegments()'s own call site) - this function's own batching math (below)
    is only derived/validated for exactly one refinement pass; raises NotImplementedError for a
    zero-daughter batch, which the caller should never actually construct.

    Batching strategy (see the implementation plan's own write-up for the full derivation and the
    measured before/after numbers): AnalyticTrajectory.__init__ builds two PchipInterpolator fits
    per instance - an "integrand" spline (keyed on height, whose antiderivative gives t_grid) and a
    combined (v_n, h_real, gravity) spline (keyed on the resulting t_grid). Profiling found
    PchipInterpolator construction itself carries substantial FIXED per-call overhead, independent
    of n_grid - so N separate per-daughter fits pay that fixed cost N times over.

    The SECOND spline genuinely cannot be shared across daughters: its own x-axis (t_grid) differs
    in VALUE per daughter (different alpha -> different deceleration timing), not just by a linear
    rescale - so it stays one PchipInterpolator per daughter per pass, unavoidably.

    The FIRST (integrand) spline CAN be shared, via a change of variables: h_real as a function of
    the Chebyshev parameter u is h_real_end_i + (h_real_start - h_real_end_i)*s(u), where
    s(u) = (1+cos(pi*u))/2 is IDENTICAL for every daughter (h_real_start is shared; only the
    per-daughter h_real_end_i differs) - i.e. h_real is an AFFINE function of the shared s for every
    daughter, with dh_real/ds = (h_real_start - h_real_end_i), a per-daughter CONSTANT (not a
    function of s - no curvature/warping correction needed, unlike parametrizing by u directly,
    which would need an extra sin(pi*u) Jacobian term). This means the integrand, reweighted by that
    per-daughter constant Jacobian, can be tabulated as one (n_grid, n_daughters) array over the
    SHARED s-axis and fit with a SINGLE vector-valued PchipInterpolator (axis=0) - collapsing N
    per-daughter integrand fits into 1 shared one, per pass. Net: 4N PCHIP constructions (2
    passes x 2 fits x N daughters) drops to 2N+2 (2 passes x (1 shared + N per-daughter)).

    Arguments/Return: identical contract to calling _buildRefinedTrajectory() once per daughter
    with n_refine_passes=1 and masses[i] as m_start - returns a list of (traj, atm_height_fn) pairs,
    same order as `masses`, each traj a genuine AnalyticTrajectory instance (built via __new__ and
    direct attribute assignment, not __init__, to avoid redoing the now-batched construction work -
    every attribute _evaluateSegment()/reportedHeightAt()/etc. actually read is set explicitly
    below). Validated bit-for-bit-equivalent (to the same floating-point-reordering tolerance
    already accepted elsewhere in this file) to N separate _buildRefinedTrajectory() calls by
    test_build_batched_daughter_trajectories_matches_unbatched().
    """

    masses = np.asarray(masses, dtype=float)
    n_daughters = len(masses)

    if n_daughters == 0:
        return []

    v_eps = 1e-6
    n_grid = 300

    beta = betaFromPhysical(sigma_eff, v_start)
    h_equiv_start = float(atm_map.toEquiv(h_real_start))
    alpha_arr = alphaFromPhysical(K, sin_slope, masses)

    # Per-daughter VelocitySpline construction is NOT batched - already O(1) for beta < 25 (the
    # overwhelming majority of real cases) via the shared process-wide LUT (see that class's own
    # docstring - "needs NO per-segment grid/spline construction at all") - never the bottleneck
    # this function exists to address.
    velocity_splines = [VelocitySpline(float(alpha_arr[i]), beta, h_equiv_start)
        for i in range(n_daughters)]

    h_equiv_floor = hEquivFromVn(v_n_floor, alpha_arr, beta, h_equiv_start)
    h_real_end = np.array([float(atm_map.toReal(h_equiv_floor[i])) for i in range(n_daughters)])
    if h_real_floor is not None:
        h_real_end = np.maximum(h_real_end, h_real_floor)

    u = np.linspace(0.0, 1.0, n_grid)
    s = (1.0 + np.cos(np.pi*u))/2.0   # descending 1 (h_real_start) -> 0 (h_real_end), shared

    def _buildPass(atm_height_fns):
        scale = h_real_start - h_real_end   # (n_daughters,), positive: dh_real/ds per daughter
        h_real_grid_2d = h_real_end[None, :] + scale[None, :]*s[:, None]   # (n_grid, n_daughters)

        if atm_height_fns is None:
            h_atm_grid_2d = h_real_grid_2d
        else:
            h_atm_grid_2d = np.empty_like(h_real_grid_2d)
            for i in range(n_daughters):
                h_atm_grid_2d[:, i] = atm_height_fns[i](h_real_grid_2d[:, i])

        h_equiv_grid_2d = atm_map.toEquiv(h_atm_grid_2d)

        v_n_grid_2d = np.empty_like(h_real_grid_2d)
        for i in range(n_daughters):
            v_n_grid_2d[:, i] = velocity_splines[i].velocityNormedAt(h_equiv_grid_2d[:, i])

        integrand_h_2d = -1.0/(v_n_grid_2d*v_start*sin_slope)
        integrand_s_2d = integrand_h_2d*scale[None, :]

        # s (and everything built from it above) is descending - PchipInterpolator needs strictly
        # increasing x, so reverse once here (matches the ORIGINAL per-daughter code's own
        # ascending-h_real convention, just expressed via s instead).
        s_asc = s[::-1]
        integrand_s_2d_asc = integrand_s_2d[::-1, :]
        h_real_grid_2d_asc = h_real_grid_2d[::-1, :]
        v_n_grid_2d_asc = v_n_grid_2d[::-1, :]

        integrand_spline_batch = scipy.interpolate.PchipInterpolator(s_asc, integrand_s_2d_asc,
            axis=0, extrapolate=True)
        antideriv_batch = integrand_spline_batch.antiderivative()

        t_offset = antideriv_batch(1.0)   # s=1 <=> h_real_start, shared across every daughter
        t_grid_2d_asc = antideriv_batch(s_asc) - t_offset[None, :] + t_start

        trajs = []
        for i in range(n_daughters):
            order = np.argsort(t_grid_2d_asc[:, i])
            t_grid_i = t_grid_2d_asc[order, i]
            v_n_grid_i = v_n_grid_2d_asc[order, i]
            h_real_grid_i = h_real_grid_2d_asc[order, i]
            g_grid_i = G0/(1.0 + h_real_grid_i/const.r_earth)**2

            traj = AnalyticTrajectory.__new__(AnalyticTrajectory)
            traj.K = K
            traj.sigma_eff = sigma_eff
            traj.m_start = float(masses[i])
            traj.v_start = v_start
            traj.h_real_start = h_real_start
            traj.sin_slope = sin_slope
            traj.t_start = t_start
            traj.atm_map = atm_map
            traj.v_eps = v_eps
            traj.r_earth = const.r_earth
            traj.h_equiv_start = h_equiv_start
            traj.alpha = float(alpha_arr[i])
            traj.beta = beta
            traj.velocity_spline = velocity_splines[i]
            traj.h_real_end = float(h_real_end[i])
            traj._t_grid = t_grid_i
            traj._v_n_grid = v_n_grid_i
            traj._h_real_grid = h_real_grid_i
            traj.t_hi = float(t_grid_i[-1])

            combined_grid_i = np.column_stack([v_n_grid_i, h_real_grid_i, g_grid_i])
            traj._time_to_combined = scipy.interpolate.PchipInterpolator(t_grid_i, combined_grid_i,
                axis=0, extrapolate=True)
            combined_antideriv_i = traj._time_to_combined.antiderivative()
            traj._grav_drop_offset = combined_antideriv_i(t_start)[2]
            traj._grav_drop_antideriv_combined = combined_antideriv_i
            traj._grav_drop_scale = 0.5*const.dt
            traj.n_queries = 0

            trajs.append(traj)

        return trajs

    traj_list_prev = _buildPass(None)

    atm_height_fns = []
    for i in range(n_daughters):
        h_atm_prev = reportedHeightAt(traj_list_prev[i], traj_list_prev[i]._t_grid, const.h_init,
            const.zenith_angle, length_start=length_start, grav_drop_start=grav_drop_start)

        if np.any(h_atm_prev > const.h_init):
            raise NotImplementedError(
                "Equivalent-atmosphere iterative refinement is currently only supported for "
                "trajectories whose reported (curved-Earth + gravity-drop) height stays at or "
                "below h_init throughout - see _buildRefinedTrajectory()'s own identical guard.")

        h_real_prev_asc = traj_list_prev[i]._h_real_grid[::-1]
        h_atm_prev_asc = h_atm_prev[::-1]

        atm_height_interp = scipy.interpolate.interp1d(h_real_prev_asc, h_atm_prev_asc,
            kind="linear", fill_value="extrapolate", assume_sorted=True)
        atm_height_fns.append(atm_height_interp)

    # Once atm_height_fn is in play, each daughter's own pass-1 h_real_end is no longer self-
    # consistent (same reasoning as AnalyticTrajectory.__init__'s own atm_height_fn branch, which
    # this re-search is transcribed from verbatim, per-daughter - NOT batched, since it is an
    # iterative walk+brentq root-find, not a PchipInterpolator construction, so there is no fixed
    # per-call overhead here worth sharing; skipping this step entirely was a real bug caught by
    # test_build_batched_daughter_trajectories_matches_unbatched() before this function was ever wired
    # into the real pipeline - t_hi differed by up to 0.44s on a 10-daughter case).
    for i in range(n_daughters):
        atm_height_fn_i = atm_height_fns[i]
        velocity_spline_i = velocity_splines[i]

        def _vn_minus_floor(h_real, atm_height_fn_i=atm_height_fn_i, velocity_spline_i=velocity_spline_i):
            h_equiv = atm_map.toEquiv(atm_height_fn_i(np.array([h_real]))[0])
            return float(velocity_spline_i.velocityNormedAt(h_equiv)) - v_n_floor

        span = h_real_start - h_real_end[i]
        step = max(span*0.02, 100.0)

        h_b = h_real_end[i]
        val_b = _vn_minus_floor(h_b)

        if val_b > 0:
            h_a, val_a = h_b, val_b
            hit_floor = False
            for _ in range(300):
                h_a = h_b - step
                if h_real_floor is not None and h_a <= h_real_floor:
                    h_a = h_real_floor
                    val_a = _vn_minus_floor(h_a)
                    hit_floor = True
                    break
                val_a = _vn_minus_floor(h_a)
                if val_a <= 0:
                    break
                h_b, val_b = h_a, val_a
            else:
                raise RuntimeError(
                    "_buildBatchedDaughterTrajectories: could not find where the "
                    "atm_height_fn-corrected v_n reaches v_n_floor within 300 fine-step "
                    "expansions (daughter index {:d}).".format(i))

            if val_a <= 0:
                h_real_end[i] = float(scipy.optimize.brentq(_vn_minus_floor, h_a, h_b, xtol=1.0))
            else:
                assert hit_floor
                h_real_end[i] = h_a
        # else: the naive bound already has corrected v_n <= v_n_floor there too - keep as-is.

    traj_list_final = _buildPass(atm_height_fns)

    return [(traj_list_final[i], atm_height_fns[i]) for i in range(n_daughters)]


def _evaluateSegment(traj, atm_height_fn, t_ticks, K, sigma_own, v_start, length_start, const,
        grav_drop_start=0.0):
    """ Evaluate one segment's (main-fragment phase or grain) physical state and luminosity/
    ionization/dynamic-pressure at the given GLOBAL times t_ticks - generalizes what was originally
    inlined directly in Stage 2c's runSimulation() (see the implementation plan's Stage 2c write-up
    for the full derivation of the rate-term reconstruction) so the same logic is reused for the
    main fragment's chained erosion segments and every grain (Stage 3), rather than duplicated.

    Arguments:
        traj: [AnalyticTrajectory] Already built and refined (see _buildRefinedTrajectory()).
        atm_height_fn: [callable or None] The SAME atm_height_fn traj was built with (None only if
            traj was built with n_refine_passes=0, in which case flat height is used directly with
            a unit slope - an identity fallback, not yet exercised by any caller in this file).
        t_ticks: [ndarray] GLOBAL times (s) to evaluate at - must be within
            [traj.t_start, traj.t_hi] for meaningful (non-clipped) output; the caller is
            responsible for only requesting times within this segment's actual active lifetime.
        K: [float] This segment's shape-density coefficient.
        sigma_own: [float] This segment's OWN ablation coefficient - NOT sigma_eff. Luminosity/
            electron-density are driven ONLY by the ablation channel (mass_loss_ablation), never
            the erosion channel (MetSimErosion.py:844,848 use mass_loss_ablation, not
            mass_loss_total). This distinction was invisible in Stage 2c's non-eroding case (there
            sigma_own == sigma_eff), but for an eroding main-fragment segment
            sigma_eff = sigma_own + erosion_coeff, and using sigma_eff here would incorrectly
            attribute the erosion channel's mass loss to luminosity too.
        v_start: [float] This segment's OWN velocity normalization (AnalyticTrajectory normalizes
            internally by whatever v_start it was built with - not necessarily const.v_init for any
            segment after the first, or for any grain).
        length_start: [float] Path length (m) accumulated before this segment started - see
            _buildRefinedTrajectory()'s docstring.
        const: [Constants] For dens_co, dt, h_init, zenith_angle, lum_eff_type, lum_eff, mu.

    Keyword arguments:
        grav_drop_start: [float] See reportedHeightAt()'s own docstring - gravity-drop (m) already
            accumulated before this segment started.

    Returns:
        (v, m, length, h_reported, lum, q, dyn_press, tau): each an ndarray matching t_ticks.
    """

    dt = const.dt

    # Single combined (v_n, h_real) spline query per time point, reused for v/m/length/h_reported
    # below - separately calling velocityNormedAt()/massAt() (which itself calls
    # velocityNormedAt() again internally - see that method's own docstring/
    # test_analytic_trajectory_query_instrumentation(), deliberately unchanged)/lengthAt() (via
    # heightRealAt())/reportedHeightAt() (which calls lengthAt() AGAIN internally) used to make up
    # to 4 separate queries against the SAME underlying spline at the SAME t_ticks - profiling
    # found this a real, avoidable cost once VelocitySpline/heightRealAt/gravityDropAt were merged
    # into one combined interpolator (see AnalyticTrajectory.__init__'s own docstring): every one
    # of those 4 calls was ALREADY evaluating all 3 combined channels internally regardless of
    # which single value it needed. velocityNormedAt()/massAt()/heightRealAt()/lengthAt()/
    # reportedHeightAt() themselves are left untouched (other callers, and this class's own
    # n_queries instrumentation contract, still need them to behave exactly as documented) - this
    # function alone bypasses them via direct (but still fully public-contract-equivalent) access
    # to the same underlying spline, doing its own n_queries bookkeeping to match.
    t_clipped = np.clip(t_ticks, traj.t_start, traj.t_hi)
    traj.n_queries += 1 if np.isscalar(t_ticks) else len(np.atleast_1d(t_ticks))
    combined = traj._time_to_combined(t_clipped)
    v_n = combined[..., 0]
    h_real = combined[..., 1]

    v = v_n*v_start
    m = massFromVelocityNormed(v_n, traj.v_start, traj.sigma_eff, traj.m_start)
    length = length_start + (traj.h_real_start - h_real)/traj.sin_slope
    h_curved = heightCurvature(const.h_init, const.zenith_angle, length, traj.r_earth)
    h_reported = h_curved - (grav_drop_start + traj.gravityDropAt(t_ticks))

    # See the implementation plan's Stage 2c write-up for the full chain-rule derivation of why
    # the rate terms need atm_height_fn's local slope, not just its value.
    t_prev = t_ticks - dt
    t_prev_clipped = np.clip(t_prev, traj.t_start, traj.t_hi)
    traj.n_queries += 1 if np.isscalar(t_prev) else len(np.atleast_1d(t_prev))
    combined_prev = traj._time_to_combined(t_prev_clipped)
    v_n_prev = combined_prev[..., 0]
    h_flat_prev = combined_prev[..., 1]

    v_prev = v_n_prev*v_start
    m_prev = massFromVelocityNormed(v_n_prev, traj.v_start, traj.sigma_eff, traj.m_start)
    if atm_height_fn is None:
        h_atm_prev_pts = h_flat_prev
        atm_height_slope = np.ones_like(h_flat_prev)
    else:
        h_atm_prev_pts = atm_height_fn(h_flat_prev)
        _atm_slope_eps = 1.0  # meters
        atm_height_slope = ((atm_height_fn(h_flat_prev + _atm_slope_eps)
            - atm_height_fn(h_flat_prev - _atm_slope_eps))/(2.0*_atm_slope_eps))

    rho_atm_physical = np.array([atmDensityPoly(float(h_i), const.dens_co) for h_i in h_atm_prev_pts])
    rho_atm_rate = rho_atm_physical*atm_height_slope

    mass_loss_rate = _massLossRate(K, sigma_own, m_prev, rho_atm_rate, v_prev)
    deceleration_rate = _decelerationRate(K, m_prev, rho_atm_rate, v_prev)

    tau = np.array([luminousEfficiency(const.lum_eff_type, const.lum_eff, float(v_i), float(m_i))
        for v_i, m_i in zip(v, m)])
    beta_ion = np.array([ionizationEfficiency(float(v_i)) for v_i in v])

    lum = -tau*(mass_loss_rate*v**2/2.0 + m*v*deceleration_rate)
    q = -beta_ion*mass_loss_rate/(const.mu*v)
    # dyn_press is a STATE quantity (rho*v^2), not a derivative - uses the corrected density
    # WITHOUT the slope factor, see the implementation plan's Stage 2c write-up.
    dyn_press = 1.0*rho_atm_physical*v**2

    return v, m, length, h_reported, lum, q, dyn_press, tau


def _buildMainFragmentSegments(const, K, sin_slope, atm_map, h_real_floor):
    """ Build the main fragment's chain of AnalyticTrajectory segments, split at
    erosion_height_start/erosion_height_change (Stage 3) - mirrors ablateAll()'s own main-fragment
    branching (MetSimErosion.py:976-990): before erosion_height_start, sigma_eff = sigma_own (no
    erosion); between erosion_height_start and erosion_height_change, sigma_eff = sigma +
    erosion_coeff at the SAME (rho, K) as the pre-erosion segment; past erosion_height_change, rho/
    K/sigma_own/erosion_coeff all change together (a fresh (alpha, beta) computation - confirmed
    only frag.main gets this swap, MetSimErosion.py:983-990).

    If const.erosion_on is False, or erosion_height_start/erosion_height_change are never reached
    within the main fragment's own flight (e.g. it dies via v_kill/m_kill/h_kill/len_kill first),
    returns a shorter chain - the caller must not assume exactly 3 segments.

    v_n_floor is recomputed PER SEGMENT from that segment's own v_start (not reused from a single
    const.v_init-relative value): AnalyticTrajectory interprets v_n_floor relative to whatever
    v_start it's built with, so reusing a v_init-relative value for a later (slower-starting)
    segment would represent a DIFFERENT, and potentially insufficient, absolute velocity margin -
    exactly the kind of undershoot that caused the Stage 2e kill-search-exhaustion bug. h_real_floor
    is shared across all segments (it is defined relative to const.h_kill/const.h_init, both global,
    not segment-local).

    Arguments:
        const: [Constants]
        K: [float] Main fragment's shape-density coefficient for the PRE-erosion-change density
            (const.rho).
        sin_slope: [float] sin(entry slope) = cos(zenith_angle).
        atm_map: [AtmEquivHeightMap]
        h_real_floor: [float] See AnalyticTrajectory's own docstring - shared by every segment.

    Return:
        segments: [list of dict] Each has keys: traj, atm_height_fn, K, rho, gamma, sigma_own,
            sigma_eff, erosion_coeff, v_start, m_start, length_start, grav_drop_start, t_start,
            t_end, erosion_mass_index, erosion_mass_min, erosion_mass_max (rho/gamma/the mass-
            distribution trio always just const's own values here - only Stage 5's "M"/"A"-type
            fragmentation entries, handled in _buildMainFragmentSegmentsWithFragmentation(), ever
            change them mid-flight; rho/gamma are carried explicitly, not just folded into K,
            because Stage 5 type "A" retroactive re-splitting needs to change gamma alone and
            re-derive K from the UNCHANGED rho - see _applyRetroactiveResplitting()). t_end is None
            for the LAST segment in the chain (it runs to its own natural/kill-search-determined
            end); every earlier segment's t_end is an exact erosion_height_start/
            erosion_height_change crossing time.
    """

    def _vnFloorFor(v_start):
        return max(0.01, 0.5*const.v_kill/v_start)

    segments = []

    traj1, atm1 = _buildRefinedTrajectory(K, const.sigma, const.m_init, const.v_init, const.h_init,
        sin_slope, atm_map, t_start=0.0, length_start=0.0, const=const,
        v_n_floor=_vnFloorFor(const.v_init), h_real_floor=h_real_floor)

    seg1 = {"traj": traj1, "atm_height_fn": atm1, "K": K, "rho": const.rho, "gamma": const.gamma,
        "sigma_own": const.sigma, "sigma_eff": const.sigma, "erosion_coeff": 0.0,
        "v_start": const.v_init, "m_start": const.m_init, "length_start": 0.0,
        "grav_drop_start": 0.0, "t_start": 0.0, "t_end": None,
        "erosion_mass_index": const.erosion_mass_index,
        "erosion_mass_min": const.erosion_mass_min, "erosion_mass_max": const.erosion_mass_max}

    if not const.erosion_on:
        seg1["t_end"] = _findSegmentDeathTime(const, seg1)
        segments.append(seg1)
        return segments

    if const.erosion_height_start >= const.h_init:
        t_erosion_start = 0.0
    else:
        t_erosion_start = _timeAtReportedHeight(traj1, const.erosion_height_start, 0.0, const)

    if t_erosion_start is None:
        # erosion_height_start is never reached - the fragment dies first.
        seg1["t_end"] = _findSegmentDeathTime(const, seg1)
        segments.append(seg1)
        return segments

    if t_erosion_start > 0.0:
        seg1["t_end"] = t_erosion_start
        segments.append(seg1)
        v1_end = float(traj1.velocityNormedAt(t_erosion_start))*const.v_init
        m1_end = float(traj1.massAt(t_erosion_start))
        h1_end = float(traj1.heightRealAt(t_erosion_start))
        length1_end = float(traj1.lengthAt(t_erosion_start))
        # seg1's own grav_drop_start is 0.0, so its total contribution BY t_erosion_start is just
        # its own gravityDropAt() there - see reportedHeightAt()'s docstring for why this needs to
        # be threaded into segment 2 rather than implicitly reset to 0.
        grav_drop_1_end = float(traj1.gravityDropAt(t_erosion_start))
    else:
        # Erosion starts immediately at h_init - segment 1 has zero duration, discard it.
        v1_end, m1_end, h1_end, length1_end = const.v_init, const.m_init, const.h_init, 0.0
        grav_drop_1_end = 0.0

    erosion_coeff_2 = getErosionCoeff(const, const.erosion_height_start)
    sigma_eff_2 = const.sigma + erosion_coeff_2

    # n_refine_passes=1 (not the default 3): this segment starts MID-FLIGHT, not at h_init, so it
    # has much less accumulated flat-vs-curved divergence to correct for than segment 1's own first
    # (h_init-to-erosion_height_start) leg - the same Stage 7 reasoning as daughter segments (see
    # _buildDaughterFragmentSegments()'s own docstring), applied here to the main fragment's own
    # LATER segments specifically (segment 1 above keeps the original validated default).
    traj2, atm2 = _buildRefinedTrajectory(K, sigma_eff_2, m1_end, v1_end, h1_end, sin_slope,
        atm_map, t_start=t_erosion_start, length_start=length1_end, const=const,
        v_n_floor=_vnFloorFor(v1_end), h_real_floor=h_real_floor, grav_drop_start=grav_drop_1_end,
        n_refine_passes=1)

    seg2 = {"traj": traj2, "atm_height_fn": atm2, "K": K, "rho": const.rho, "gamma": const.gamma,
        "sigma_own": const.sigma, "sigma_eff": sigma_eff_2, "erosion_coeff": erosion_coeff_2,
        "v_start": v1_end, "m_start": m1_end, "length_start": length1_end,
        "grav_drop_start": grav_drop_1_end, "t_start": t_erosion_start, "t_end": None,
        "erosion_mass_index": const.erosion_mass_index,
        "erosion_mass_min": const.erosion_mass_min, "erosion_mass_max": const.erosion_mass_max}

    # erosion_height_change only takes effect if it is a LOWER height than erosion_height_start
    # (matching getErosionCoeff()'s own precedence) - otherwise segment 2 runs to its own end.
    if const.erosion_height_change >= const.erosion_height_start:
        seg2["t_end"] = _findSegmentDeathTime(const, seg2)
        segments.append(seg2)
        return segments

    t_erosion_change = _timeAtReportedHeight(traj2, const.erosion_height_change, length1_end, const,
        grav_drop_start=grav_drop_1_end)

    if t_erosion_change is None:
        seg2["t_end"] = _findSegmentDeathTime(const, seg2)
        segments.append(seg2)
        return segments

    seg2["t_end"] = t_erosion_change
    segments.append(seg2)

    v2_end = float(traj2.velocityNormedAt(t_erosion_change))*v1_end
    m2_end = float(traj2.massAt(t_erosion_change))
    h2_end = float(traj2.heightRealAt(t_erosion_change))
    # lengthAt() is LOCAL to traj2's own segment (measured from traj2's own h_real_start) - must
    # add seg2's length_start (the cumulative length already traveled before segment 2 began) to
    # get the TOTAL cumulative length, matching length1_end's convention above (correct there only
    # because segment 1 starts at the whole flight's own beginning, where local and cumulative
    # length coincide). Same reasoning for grav_drop_2_end below.
    length2_end = length1_end + float(traj2.lengthAt(t_erosion_change))
    grav_drop_2_end = grav_drop_1_end + float(traj2.gravityDropAt(t_erosion_change))

    K3 = const.gamma*const.shape_factor*const.erosion_rho_change**(-2/3.0)
    erosion_coeff_3 = const.erosion_coeff_change
    sigma_eff_3 = const.erosion_sigma_change + erosion_coeff_3

    traj3, atm3 = _buildRefinedTrajectory(K3, sigma_eff_3, m2_end, v2_end, h2_end, sin_slope,
        atm_map, t_start=t_erosion_change, length_start=length2_end, const=const,
        v_n_floor=_vnFloorFor(v2_end), h_real_floor=h_real_floor, grav_drop_start=grav_drop_2_end,
        n_refine_passes=1)

    seg3 = {"traj": traj3, "atm_height_fn": atm3, "K": K3, "rho": const.erosion_rho_change,
        "gamma": const.gamma, "sigma_own": const.erosion_sigma_change, "sigma_eff": sigma_eff_3,
        "erosion_coeff": erosion_coeff_3, "v_start": v2_end, "m_start": m2_end,
        "length_start": length2_end, "grav_drop_start": grav_drop_2_end,
        "t_start": t_erosion_change, "t_end": None, "erosion_mass_index": const.erosion_mass_index,
        "erosion_mass_min": const.erosion_mass_min, "erosion_mass_max": const.erosion_mass_max}
    segments.append(seg3)

    # Resolve the LAST segment's own natural end (the main fragment's death time) now, via the
    # same candidate-doubling kill search Stage 2c/2e built for the single-segment case (see
    # _findSegmentDeathTime()) - so every segment this function returns has BOTH t_start and t_end
    # set, and callers (grain-epoch placement, results_list assembly) never need to special-case a
    # None t_end.
    segments[-1]["t_end"] = _findSegmentDeathTime(const, segments[-1])

    return segments


def _applyFragmentationEntry(const, entry, rho, gamma, sigma_own, erosion_coeff,
        erosion_mass_index, erosion_mass_min, erosion_mass_max, m, v, h_reported, length, t,
        grav_drop, frag_daughters, frag_grain_specs, a_type_events):
    """ Apply one fragmentation entry's own effect at the instant the main fragment crosses its
    trigger height - the per-type switch mirrored directly from ablateAll()'s own complex-
    fragmentation block (MetSimErosion.py:1103-1213), called from
    _buildMainFragmentSegmentsWithFragmentation() once per entry, in the same order/timing
    ablateAll() itself applies them (height-descending, main-fragment-triggered only).

    Mutates the main fragment's own forward state (mass, rho/gamma pair, sigma_own, erosion_coeff,
    erosion_mass_index/min/max) for its NEXT segment, and appends any spawned daughters/grains to
    the caller's own accumulator lists (F/EF -> frag_daughters, D -> frag_grain_specs, A -> both a
    same-mechanism update here AND an entry in a_type_events for the aggregation level to
    retroactively re-derive any grain/daughter trajectory already alive at this instant - see
    _runSimulationErosion()'s own "Stage 5" docstring section for why that step cannot happen here,
    where only the main fragment's own state is available).

    F/EF do NOT use generateFragments() (confirmed from source, MetSimErosion.py:1135-1182): unlike
    erosion/disruption/dust-release, these create EXACTLY entry.number new fragments via a plain
    loop, each getting an EQUAL share (mass_percent/number) of the pre-event mass - no mass-binning
    power-law distribution involved. D DOES use generateFragments() (mass-binned, like any other
    grain population) - called immediately here (matching how disruption's own leftover-mass-to-
    grains spawn already works), not deferred.

    Arguments:
        const: [Constants]
        entry: [FragmentationEntry-like] Duck-typed - only attribute ACCESS needed (frag_type,
            sigma, gamma, erosion_coeff, mass_index, grain_mass_min, grain_mass_max, mass_percent,
            number, id) - this module never imports or constructs the class itself (neither does
            MetSimErosion.py - the caller, e.g. DynestyMetSim.py/GUI.py, builds
            const.fragmentation_entries before runSimulation() is ever called).
        rho, gamma: [float] Main fragment's CURRENT bulk density/drag coefficient - tracked
            separately (not just their product K) because type "A" can change gamma alone, and
            K = gamma*shape_factor*rho**(-2/3) must be rederived from the unchanged rho afterward.
        sigma_own, erosion_coeff: [float] Main fragment's current ablation/erosion coefficients.
        erosion_mass_index, erosion_mass_min, erosion_mass_max: [float] Main fragment's current
            grain-size-distribution parameters (only "M" can change these).
        m, v, h_reported, length, t, grav_drop: [float] Main fragment's exact state at the crossing
            instant (already computed by the caller via the segment's own trajectory).
        frag_daughters: [list, mutated in place] F/EF daughter specs, each a dict with keys K, rho,
            gamma, sigma, m_start, v_start, h_real_start, t_start, length_start, grav_drop_start,
            erosion_enabled, erosion_coeff_fixed, erosion_mass_index, erosion_mass_min,
            erosion_mass_max, n_grains, complex_id - consumed by _runSimulationErosion() to build
            each daughter's own segment chain (a FIXED, never-height-updated erosion_coeff for EF -
            confirmed from source: the height-based getErosionCoeff() auto-update is gated by
            `not frag.complex`, MetSimErosion.py:976, and every F/EF/D child gets `complex=True`
            unconditionally - a real, different behavior from disruption "fragment" daughters,
            which are NOT complex and DO get the height-based update).
        frag_grain_specs: [list, mutated in place] D-type dust grain_specs (same shape
            _spawnGrainsForSegment() produces: m, n_grains, sigma, rho, K, v, h_real, length, t).
        a_type_events: [list, mutated in place] One {"t", "sigma", "gamma"} entry per "A" trigger.

    Return:
        (m, rho, gamma, K, sigma_own, erosion_coeff, erosion_mass_index, erosion_mass_min,
        erosion_mass_max): the main fragment's own updated state for its next segment.
    """

    if entry.frag_type == "A":
        if entry.sigma is not None:
            sigma_own = entry.sigma
        if entry.gamma is not None:
            gamma = entry.gamma
        a_type_events.append({"t": t, "sigma": entry.sigma, "gamma": entry.gamma})

    elif entry.frag_type == "M":
        if entry.sigma is not None:
            sigma_own = entry.sigma
        if entry.erosion_coeff is not None:
            erosion_coeff = entry.erosion_coeff
        if entry.mass_index is not None:
            erosion_mass_index = entry.mass_index
        if entry.grain_mass_min is not None:
            erosion_mass_min = entry.grain_mass_min
        if entry.grain_mass_max is not None:
            erosion_mass_max = entry.grain_mass_max

    elif entry.frag_type in ("F", "EF"):
        parent_initial_mass = m
        d_sigma = entry.sigma if entry.sigma is not None else sigma_own
        d_erosion_enabled = (entry.frag_type == "EF")
        d_mass_index = (entry.mass_index if entry.mass_index is not None else 2.0)

        for _ in range(entry.number):
            new_frag_mass = parent_initial_mass*(entry.mass_percent/100.0)/entry.number
            m -= new_frag_mass

            frag_daughters.append({"K": gamma*const.shape_factor*rho**(-2/3.0), "rho": rho,
                "gamma": gamma, "sigma": d_sigma, "m_start": new_frag_mass, "v_start": v,
                "h_real_start": h_reported, "t_start": t, "length_start": length,
                "grav_drop_start": grav_drop, "erosion_enabled": d_erosion_enabled,
                "erosion_coeff_fixed": (entry.erosion_coeff if d_erosion_enabled else 0.0),
                "erosion_mass_index": d_mass_index, "erosion_mass_min": entry.grain_mass_min,
                "erosion_mass_max": entry.grain_mass_max, "n_grains": 1.0, "complex_id": entry.id})

        entry.mass = parent_initial_mass*(entry.mass_percent/100.0)

    elif entry.frag_type == "D":
        dust_mass = m*(entry.mass_percent/100.0)
        m -= dust_mass
        entry.mass = dust_mass

        mass_index = entry.mass_index if entry.mass_index is not None else 2.0
        parent = _makeVirtualParentFragment(const, dust_mass, v, h_reported, length, sigma_own,
            rho=rho)
        grain_children, _ = generateFragments(const, parent, dust_mass, mass_index,
            entry.grain_mass_min, entry.grain_mass_max, keep_eroding=False,
            mass_model=const.erosion_grain_distribution)

        for gc in grain_children:
            frag_grain_specs.append({"m": gc.m, "n_grains": gc.n_grains, "sigma": gc.sigma,
                "rho": gc.rho, "K": gc.K, "v": v, "h_real": h_reported, "length": length, "t": t})

    K = gamma*const.shape_factor*rho**(-2/3.0)

    return m, rho, gamma, K, sigma_own, erosion_coeff, erosion_mass_index, erosion_mass_min, \
        erosion_mass_max


def _buildMainFragmentSegmentsWithFragmentation(const, K_init, sin_slope, atm_map, h_real_floor):
    """ Stage 5: main-fragment segment builder generalizing _buildMainFragmentSegments() to also
    split at every (non-upward_only) fragmentation-entry height, interleaved with the erosion
    thresholds in true height order - called instead of _buildMainFragmentSegments() only when
    const.fragmentation_on is True (runSimulation() dispatch), so the erosion-only path stays
    completely untouched (zero added regression risk there).

    ALL FIVE entry types split the main fragment's own trajectory, not just M/A: F/EF/D all reduce
    frag.m directly (MetSimErosion.py:1145,1192), and the closed-form trajectory's own m_start is
    baked in at construction, so a new segment is unavoidable even when sigma/erosion_coeff/rho/
    gamma do not change. Every trigger height is therefore a segment boundary.

    upward_only fragmentation entries (MetSimErosion.py:1093-1098's second branch, triggered when
    frag.h > frag_entry.height AND frag.h > const.main_bottom_ht - i.e. REPORTED height has already
    dipped below frag_entry.height at some point and has since risen back above it) raise
    NotImplementedError explicitly, rather than being silently skipped or approximated with a
    heuristic. The original implementation plan recommended a heuristic trigger (e.g. near v_kill)
    for these - revisited once Stage 2c's own grazing-entry finding was in hand: reported height
    (heightCurvature() minus gravity-drop) is non-monotonic in path length by construction, with a
    genuine minimum at length = (h0+R)*cos(zenith) - the SAME l_min/l_return geometry Stage 2c's
    85-degree-grazing guard already targets (runSimulation() raises NotImplementedError there too,
    for the SAME underlying reason: AtmEquivHeightMap/VelocitySpline assume reported height is
    monotonic). For length to approach l_min within a fragment's own lifetime (needed for
    upward_only to ever fire at all) requires the same shallow/grazing/long-lived regime that
    guard already blocks - so any scenario able to reach an upward_only trigger would almost
    certainly have already hit that earlier guard first. A heuristic here would therefore be
    untestable against a real supported trajectory (nothing to validate it with) rather than
    genuinely useful - an honest, explicit failure is more useful than untested guesswork. Real
    support for either would need the same deferred architectural work (Stage 2c's own write-up:
    "AtmEquivHeightMap/VelocitySpline rebuilt to not assume reported height is monotonic").

    Arguments:
        const: [Constants]
        K_init: [float] Main fragment's initial shape-density coefficient (const.rho-based).
        sin_slope: [float] sin(entry slope) = cos(zenith_angle).
        atm_map: [AtmEquivHeightMap]
        h_real_floor: [float] See AnalyticTrajectory's own docstring.

    Return:
        (segments, frag_daughters, frag_grain_specs, a_type_events): segments is the same
        list-of-dict shape _buildMainFragmentSegments() returns (every entry fully resolved, t_end
        always set); frag_daughters/frag_grain_specs/a_type_events are exactly
        _applyFragmentationEntry()'s own accumulator lists, flattened across every entry that fired
        before the main fragment's own natural death.
    """

    def _vnFloorFor(v):
        return max(0.01, 0.5*const.v_kill/v)

    if const.fragmentation_on and any(entry.upward_only for entry in const.fragmentation_entries):
        raise NotImplementedError("fragmentation_entries with upward_only=True are not supported "
            "by the analytic engine - this condition only arises for shallow/grazing/long-lived "
            "trajectories the engine already can't build (see runSimulation()'s own 85-degree-"
            "grazing-entry guard and this function's own docstring). Use "
            "MetSimErosion.runSimulation() for upward_only fragmentation, or remove such entries.")

    triggers = []
    if const.erosion_on:
        if const.erosion_height_start < const.h_init:
            triggers.append((const.erosion_height_start, "erosion_start", None))
        if const.erosion_height_change < const.erosion_height_start:
            triggers.append((const.erosion_height_change, "erosion_change", None))
    if const.fragmentation_on:
        for entry in const.fragmentation_entries:
            if not entry.upward_only:
                triggers.append((entry.height, "frag", entry))

    # Descending height order - main only ever descends in this function's own scope (upward_only
    # entries are excluded above). Ties are resolved by Python's stable sort preserving the order
    # triggers were appended in above (erosion thresholds first, then fragmentation_entries in
    # their own list order) - matching ablateAll()'s own iteration order closely enough that a
    # genuine simultaneous-height tie (vanishingly unlikely with real-valued heights) would still
    # apply effects in a sensible, deterministic order.
    triggers.sort(key=lambda item: -item[0])

    segments = []
    frag_daughters = []
    frag_grain_specs = []
    a_type_events = []

    rho, gamma = const.rho, const.gamma
    K = K_init
    sigma_own = const.sigma
    erosion_coeff = 0.0
    erosion_mass_index = const.erosion_mass_index
    erosion_mass_min = const.erosion_mass_min
    erosion_mass_max = const.erosion_mass_max
    m, v, h, length, t, grav_drop = const.m_init, const.v_init, const.h_init, 0.0, 0.0, 0.0

    for trigger_height, kind, entry in triggers:

        if trigger_height >= h:
            # Already at/below this trigger's height - matches ablateAll()'s own strict
            # `frag.h < frag_entry.height` condition (a trigger exactly at the current height does
            # not yet fire), and correctly skips any entry whose height was already passed by an
            # EARLIER trigger's own mass-loss/deceleration before this one is reached.
            continue

        sigma_eff = sigma_own + erosion_coeff
        # n_refine_passes: 3 (the original validated default) only for the very first segment
        # (t==0.0, starting at h_init) - every later segment starts mid-flight, with much less
        # accumulated flat-vs-curved divergence to correct for, so 1 pass suffices (Stage 7 finding
        # - see _buildDaughterFragmentSegments()'s own docstring for the full reasoning, which
        # applies identically here now that this function can produce many more than 3 segments).
        traj, atm_fn = _buildRefinedTrajectory(K, sigma_eff, m, v, h, sin_slope, atm_map,
            t_start=t, length_start=length, const=const, v_n_floor=_vnFloorFor(v),
            h_real_floor=h_real_floor, grav_drop_start=grav_drop,
            n_refine_passes=(3 if t == 0.0 else 1))

        seg = {"traj": traj, "atm_height_fn": atm_fn, "K": K, "rho": rho, "gamma": gamma,
            "sigma_own": sigma_own, "sigma_eff": sigma_eff, "erosion_coeff": erosion_coeff,
            "v_start": v, "m_start": m, "length_start": length, "grav_drop_start": grav_drop,
            "t_start": t, "t_end": None, "erosion_mass_index": erosion_mass_index,
            "erosion_mass_min": erosion_mass_min, "erosion_mass_max": erosion_mass_max}

        t_cross = _timeAtReportedHeight(traj, trigger_height, length, const,
            grav_drop_start=grav_drop)

        if t_cross is None:
            # This trigger's height is never reached - the fragment dies first; no later trigger
            # (all at or below this one, by construction) can matter either.
            seg["t_end"] = _findSegmentDeathTime(const, seg)
            segments.append(seg)
            return segments, frag_daughters, frag_grain_specs, a_type_events

        seg["t_end"] = t_cross
        segments.append(seg)

        v = float(traj.velocityNormedAt(t_cross))*v
        m = float(traj.massAt(t_cross))
        h = float(traj.heightRealAt(t_cross))
        length = length + float(traj.lengthAt(t_cross))
        grav_drop = grav_drop + float(traj.gravityDropAt(t_cross))
        t = t_cross

        if kind == "erosion_start":
            erosion_coeff = getErosionCoeff(const, h)

        elif kind == "erosion_change":
            rho = const.erosion_rho_change
            sigma_own = const.erosion_sigma_change
            erosion_coeff = const.erosion_coeff_change
            K = gamma*const.shape_factor*rho**(-2/3.0)

        else:  # kind == "frag"
            parent_mass_at_trigger = m
            (m, rho, gamma, K, sigma_own, erosion_coeff, erosion_mass_index, erosion_mass_min,
                erosion_mass_max) = _applyFragmentationEntry(const, entry, rho, gamma, sigma_own,
                    erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, m, v, h,
                    length, t, grav_drop, frag_daughters, frag_grain_specs, a_type_events)
            entry.done = True
            entry.time = t
            entry.velocity = v
            entry.parent_mass = parent_mass_at_trigger
            entry.dyn_pressure = 1.0*atmDensityPoly(h, const.dens_co)*v**2

    sigma_eff = sigma_own + erosion_coeff
    # Same n_refine_passes reasoning as the in-loop build above - 3 only if this ended up being the
    # ONLY segment (no trigger ever fired, t is still 0.0, so this starts at h_init).
    traj, atm_fn = _buildRefinedTrajectory(K, sigma_eff, m, v, h, sin_slope, atm_map, t_start=t,
        length_start=length, const=const, v_n_floor=_vnFloorFor(v), h_real_floor=h_real_floor,
        grav_drop_start=grav_drop, n_refine_passes=(3 if t == 0.0 else 1))
    seg = {"traj": traj, "atm_height_fn": atm_fn, "K": K, "rho": rho, "gamma": gamma,
        "sigma_own": sigma_own, "sigma_eff": sigma_eff, "erosion_coeff": erosion_coeff, "v_start": v,
        "m_start": m, "length_start": length, "grav_drop_start": grav_drop, "t_start": t,
        "t_end": None, "erosion_mass_index": erosion_mass_index,
        "erosion_mass_min": erosion_mass_min, "erosion_mass_max": erosion_mass_max}
    seg["t_end"] = _findSegmentDeathTime(const, seg)
    segments.append(seg)

    return segments, frag_daughters, frag_grain_specs, a_type_events


def _findSegmentDeathTime(const, seg):
    """ Find the GLOBAL time (s, an exact multiple of const.dt) at which this segment's own state
    first satisfies a kill condition (m_kill/v_kill/h_kill/len_kill) - the candidate-doubling half
    of the kill search Stage 2c/2e built for the single whole-flight segment, generalized to work
    on any segment dict that starts partway through the simulation.

    Candidate ticks are placed on the GLOBAL dt-grid (dt, 2*dt, 3*dt, ... from the simulation's own
    t=0), not a grid re-based at this segment's own (generally non-dt-aligned, since segment
    boundaries come from continuous root-finding) start time - required so this segment's output
    rows line up with the rest of the simulation's results_list rows.

    Does NOT (yet) retry with a deeper h_real_floor if this segment's own tabulated domain (t_hi)
    turns out insufficient (Stage 2e's floor_extension mechanism) - erosion-affected segments
    ablate much faster than the non-eroding, long-surviving, shallow-angle cases that motivated
    that retry, so this gap is expected to be rare here, but has not been proven impossible - a
    targeted robustness follow-up if a real case ever needs it, not core physics.

    Arguments:
        const: [Constants]
        seg: [dict] A segment dict (from _buildMainFragmentSegments()/_buildDaughterFragmentSegments()),
            t_end not yet resolved. seg["grav_drop_start"] (see reportedHeightAt()'s docstring) is
            read if present, else treated as 0.0 (correct for a fragment's own first segment).

    Return:
        [float] The GLOBAL time (s) of the first kill condition.
    """

    traj = seg["traj"]
    v_start = seg["v_start"]
    length_start = seg["length_start"]
    grav_drop_start = seg.get("grav_drop_start", 0.0)
    dt = const.dt

    h_reported_grid = reportedHeightAt(traj, traj._t_grid, const.h_init, const.zenith_angle,
        length_start=length_start, grav_drop_start=grav_drop_start)
    t_est = float(np.interp(const.h_kill, h_reported_grid[::-1], traj._t_grid[::-1]))

    # First GLOBAL tick index (k such that t=k*dt) at or after this segment's own start - a small
    # epsilon avoids float round-off pushing an exact-boundary case to the next tick.
    k_start = int(math.ceil(seg["t_start"]/dt - 1e-9))
    n_candidate = max(int((t_est - seg["t_start"])/dt*1.5) + 20, 100)

    for _ in range(10):

        k_ticks = k_start + np.arange(n_candidate)
        t_ticks = k_ticks*dt

        v = traj.velocityNormedAt(t_ticks)*v_start
        m = traj.massAt(t_ticks)
        length = length_start + traj.lengthAt(t_ticks)
        h_reported = reportedHeightAt(traj, t_ticks, const.h_init, const.zenith_angle,
            length_start=length_start, grav_drop_start=grav_drop_start)

        kill_mask = ((m <= const.m_kill) | (v < const.v_kill) | (h_reported < const.h_kill)
            | ((const.len_kill > 0) & (length > const.len_kill)))

        if np.any(kill_mask):
            kill_idx = int(np.argmax(kill_mask))
            return float(t_ticks[kill_idx])

        if t_ticks[-1] >= traj.t_hi:
            raise RuntimeError(
                "_findSegmentDeathTime: this segment's own tabulated domain (bounded by "
                "v_n_floor/h_real_floor) does not reach a kill condition - Stage 2e's "
                "floor-extension retry is not yet implemented for erosion segments (see this "
                "function's docstring). Check Constants for an unusually long-surviving eroding "
                "fragment.")

        n_candidate *= 2

    raise RuntimeError("_findSegmentDeathTime: could not find a kill condition within 10 "
        "search-window expansions - check Constants for an unreachable "
        "m_kill/v_kill/h_kill configuration.")


def _buildDaughterFragmentSegments(const, K, sigma_own, m_start, v_start, h_real_start, sin_slope,
        atm_map, t_start, length_start, grav_drop_start, h_real_floor, rho=None, gamma=None,
        n_refine_passes=1):
    """ Build a disruption "fragment" daughter's own segment chain (Stage 4) - the daughter
    counterpart to _buildMainFragmentSegments(), needed because a daughter can be born ANYWHERE in
    the flight (frequently already below erosion_height_start, sometimes already below
    erosion_height_change too, since disruption typically happens well into an already-eroding
    flight), unlike the main fragment, which always starts at h_init above the whole erosion zone
    by physical construction.

    Genuinely simpler than the main-fragment case in one respect: daughters NEVER get the
    rho/K/sigma_own swap at erosion_height_change. MetSimErosion.py:983-990's swap is gated by
    `frag.main`, confirmed always False for a disruption daughter (generateFragments() sets
    frag_child.main=False unconditionally, MetSimErosion.py:1363) - K/sigma_own here stay CONSTANT
    across every phase of this function; only erosion_coeff (and therefore sigma_eff) changes
    between the erosion_height_start/erosion_height_change thresholds, via the same
    getErosionCoeff() lookup every erosion-enabled fragment uses (which already encodes the
    "erosion_height_change only takes effect if it is a LOWER height than erosion_height_start"
    precedence _buildMainFragmentSegments() checks for explicitly - here it falls out for free from
    _timeAtReportedHeight() correctly returning None for a threshold already passed).

    erosion_coeff is FIXED at const.disruption_erosion_coeff for this daughter's entire erosion-
    active life (both phases below), not the standard height-based getErosionCoeff() lookup every
    OTHER erosion-enabled fragment uses. This used to be documented here as a "negligible
    simplification" under the OLD reference behavior (MetSimErosion.py set a disruption daughter's
    erosion_coeff to const.disruption_erosion_coeff at spawn, but unconditionally OVERWROTE it with
    getErosionCoeff(const, frag.h) on the very next processed tick, with no "only if not yet set"
    guard - so disruption_erosion_coeff affected only a single tick's mass loss there). That
    reference behavior was ITSELF a bug, fixed upstream in MetSimErosion.py commit 6be7301 (a
    disruption_child flag now excludes these daughters from the height-based recompute entirely) -
    the reference now correctly keeps disruption_erosion_coeff for the daughter's WHOLE life, no
    longer a one-tick, negligible effect. This function was updated to match: confirmed directly
    (not assumed) that using the height-based value here (the OLD behavior) produced a genuine,
    compounding erosion_coeff mismatch (measured on a real case: 3.0e-7 here vs the reference's own
    3.3e-7 - a 10% difference, NOT the two constants coincidentally differing by rounding) that
    grew from a barely-visible mass error early in a daughter's life (~1% at 5% of its lifetime)
    into an over 2-orders-of-magnitude one by ~90% of its lifetime, and directly explained a
    previously-mysterious, perfectly consistent ~20-30ms LATE bias in every affected daughter's own
    death time (investigated at length - see _resolveSegmentChainDeathRegime()'s own docstring
    history - before this was found to be the true cause, not any inaccuracy in how that function
    resumes a closed-form segment via its own RK4 tail).

    Arguments:
        const: [Constants]
        K, sigma_own: [float] This daughter's shape-density/ablation coefficients - CONSTANT for
            its entire lifetime (see above; inherited from the disrupting parent at the instant of
            disruption, via the same spawn_child() dict-copy mechanism erosion grains use).
        m_start, v_start, h_real_start: [float] State at spawn (the disruption instant).
        sin_slope: [float] sin(entry slope) = cos(zenith_angle).
        atm_map: [AtmEquivHeightMap]
        t_start: [float] GLOBAL spawn time (s).
        length_start: [float] Cumulative path length (m) at spawn.
        grav_drop_start: [float] Cumulative gravity-drop (m) at spawn - see reportedHeightAt()'s
            docstring. NOT negligible here the way it is for tiny erosion grains (_stepGrainRK4()'s
            own docstring): a disruption daughter inherits its parent's ENTIRE pre-disruption
            accumulated total via spawn_child()'s dict copy (MetSimErosion.py:373-384, never reset
            to 0), which can be a non-trivial fraction of a meter to several meters for a
            long-lived parent, unlike the sub-tick lifetime of a typical erosion grain.
        h_real_floor: [float] See AnalyticTrajectory's own docstring.

    Keyword arguments:
        rho, gamma: [float or None] This daughter's own bulk density/drag coefficient - CONSTANT
            for its entire lifetime, same as K/sigma_own above (rho never changes for a non-main
            fragment, MetSimErosion.py:983's swap is gated by frag.main). Carried explicitly
            (rather than only folded into K) purely so the returned segment dicts can record them -
            Stage 5 type "A" retroactive re-splitting needs to change gamma alone and re-derive K
            from the UNCHANGED rho (see _applyRetroactiveResplitting()). Default to const.rho/
            const.gamma for backward compatibility with any caller that doesn't need this (every
            actual behavior in this function depends only on K/sigma_own, never on rho/gamma
            directly).
        n_refine_passes: [int] Passed through to every _buildRefinedTrajectory() call this function
            makes (1-3 of them, depending on regime) - defaults to 1, NOT
            _buildRefinedTrajectory()'s own default of 3. Stage 7 finding (cProfile, then directly
            measured): AnalyticTrajectory.__init__'s cost is dominated by fixed per-call scipy/
            PchipInterpolator overhead (~80us/call, essentially flat from n_grid=30 to 500 - NOT
            reducible by shrinking the grid), so the lever that actually matters is the NUMBER of
            AnalyticTrajectory builds, i.e. 1+n_refine_passes per segment. A disruption event alone
            can spawn 10+ daughters, each needing up to 3 of ITS OWN segments - at the previous
            default (3, i.e. 4 builds/segment), this measured as the dominant cost for disruption/
            complex-fragmentation scenarios, making this file's own engine SLOWER than the RK4
            reference for them (0.3-0.4x). Reduced to 1 (2 builds/segment, half the previous cost)
            specifically for daughters, not the main fragment's own segments (which keep the
            original validated default of 3 via _buildMainFragmentSegments()'s own call) - daughters
            have a much shorter remaining flight than the main fragment's own first segment (h_init
            to erosion_height_start), so the curvature/atmosphere divergence Stage 2c's own 3-pass
            choice was validated against (a multi-km-scale effect) should be proportionally smaller
            here. Validated, not assumed: the full existing test suite's own already-tight
            disruption/complex-fragmentation accuracy tolerances (test_run_simulation_disruption_only()
            etc.) pass unchanged at this value.

    Return:
        segments: [list of dict] Same shape as _buildMainFragmentSegments()'s own return (keys:
            traj, atm_height_fn, K, rho, gamma, sigma_own, sigma_eff, erosion_coeff, v_start,
            m_start, length_start, grav_drop_start, t_start, t_end) - 1 to 3 entries depending on
            which regime h_real_start starts in and whether erosion_on is even True.
    """

    def _vnFloorFor(v):
        return max(0.01, 0.5*const.v_kill/v)

    rho = const.rho if rho is None else rho
    gamma = const.gamma if gamma is None else gamma

    segments = []

    if (not const.erosion_on) or (h_real_start >= const.erosion_height_start):

        # Not eroding yet (or never will be) - pre-erosion phase first.
        traj_a, atm_a = _buildRefinedTrajectory(K, sigma_own, m_start, v_start, h_real_start,
            sin_slope, atm_map, t_start=t_start, length_start=length_start, const=const,
            v_n_floor=_vnFloorFor(v_start), h_real_floor=h_real_floor,
            grav_drop_start=grav_drop_start, n_refine_passes=n_refine_passes)

        seg_a = {"traj": traj_a, "atm_height_fn": atm_a, "K": K, "rho": rho, "gamma": gamma,
            "sigma_own": sigma_own, "sigma_eff": sigma_own, "erosion_coeff": 0.0,
            "v_start": v_start, "m_start": m_start, "length_start": length_start,
            "grav_drop_start": grav_drop_start, "t_start": t_start, "t_end": None}

        if not const.erosion_on:
            seg_a["t_end"] = _findSegmentDeathTime(const, seg_a)
            segments.append(seg_a)
            return segments

        t_cross = _timeAtReportedHeight(traj_a, const.erosion_height_start, length_start, const,
            grav_drop_start=grav_drop_start)

        if t_cross is None:
            seg_a["t_end"] = _findSegmentDeathTime(const, seg_a)
            segments.append(seg_a)
            return segments

        seg_a["t_end"] = t_cross
        segments.append(seg_a)

        v_b = float(traj_a.velocityNormedAt(t_cross))*v_start
        m_b = float(traj_a.massAt(t_cross))
        h_b = float(traj_a.heightRealAt(t_cross))
        length_b = length_start + float(traj_a.lengthAt(t_cross))
        grav_drop_b = grav_drop_start + float(traj_a.gravityDropAt(t_cross))
        t_b = t_cross

    else:
        # Already at or below erosion_height_start at spawn - skip the pre-erosion phase entirely.
        v_b, m_b, h_b, length_b, grav_drop_b, t_b = (v_start, m_start, h_real_start, length_start,
            grav_drop_start, t_start)

    # At or below erosion_height_start now (either just crossed above, or started there) - this
    # daughter is erosion-active from here to its own death, at a FIXED erosion_coeff (see this
    # function's own docstring for why - const.disruption_erosion_coeff, not the height-based
    # getErosionCoeff() lookup). Since erosion_coeff no longer depends on height for a daughter,
    # the erosion_height_change threshold that used to split this into two segments (phase b/c,
    # each with its own getErosionCoeff()-derived value) is now physically inert - both phases
    # would carry the IDENTICAL sigma_eff - so there is exactly one erosion-active segment here,
    # not up to two.
    erosion_coeff_b = const.disruption_erosion_coeff
    sigma_eff_b = sigma_own + erosion_coeff_b

    traj_b, atm_b = _buildRefinedTrajectory(K, sigma_eff_b, m_b, v_b, h_b, sin_slope, atm_map,
        t_start=t_b, length_start=length_b, const=const, v_n_floor=_vnFloorFor(v_b),
        h_real_floor=h_real_floor, grav_drop_start=grav_drop_b, n_refine_passes=n_refine_passes)

    seg_b = {"traj": traj_b, "atm_height_fn": atm_b, "K": K, "rho": rho, "gamma": gamma,
        "sigma_own": sigma_own, "sigma_eff": sigma_eff_b, "erosion_coeff": erosion_coeff_b,
        "v_start": v_b, "m_start": m_b, "length_start": length_b, "grav_drop_start": grav_drop_b,
        "t_start": t_b, "t_end": None}
    seg_b["t_end"] = _findSegmentDeathTime(const, seg_b)
    segments.append(seg_b)

    return segments


def _buildDaughterFragmentSegmentsBatch(const, K, sigma_own, masses, v_start, h_real_start,
        sin_slope, atm_map, t_start, length_start, grav_drop_start, h_real_floor, rho=None,
        gamma=None):
    """ Batched counterpart to _buildDaughterFragmentSegments(), for every daughter spawned by the
    SAME disruption event - confirmed from source (generateFragments()'s spawn_child() dict-copy)
    that these share K/sigma_own/v_start/h_real_start/t_start/length_start/grav_drop_start exactly
    (only mass differs per mass bin - see _buildBatchedDaughterTrajectories()'s own docstring for
    the full account of what that sharing buys). Returns a list of segment chains, one per
    daughter, in the same order as `masses` - each chain identical in shape/content to what N
    separate _buildDaughterFragmentSegments() calls would produce.

    Batches whichever of this function's own two possible segments is safe to batch:
    - The pre-erosion segment ("segment A"), when built at all, is ALWAYS batchable - its own build
      inputs are the shared disruption-instant state for every daughter, regardless of what happens
      to each daughter afterward.
    - The erosion-active segment ("segment B") is batchable ONLY when it starts from that SAME
      shared state too - true whenever every daughter skips segment A entirely (disruption already
      below erosion_height_start - confirmed the common case for every scenario actually
      benchmarked in this project), false when daughters build segment A first (each daughter's own
      crossing time/state into erosion_height_start then differs by mass, so segment B's own start
      state genuinely diverges per daughter) - falls back to N separate (unbatched)
      _buildRefinedTrajectory() calls for segment B in that specific case, identical to what
      _buildDaughterFragmentSegments() itself would do.

    Validated against N separate _buildDaughterFragmentSegments() calls (same inputs) by
    test_build_daughter_fragment_segments_batch_matches_unbatched().
    """

    masses = np.asarray(masses, dtype=float)
    n_daughters = len(masses)
    rho = const.rho if rho is None else rho
    gamma = const.gamma if gamma is None else gamma

    def _vnFloorFor(v):
        return max(0.01, 0.5*const.v_kill/v)

    segments_out = [[] for _ in range(n_daughters)]

    if (not const.erosion_on) or (h_real_start >= const.erosion_height_start):

        batch_a = _buildBatchedDaughterTrajectories(K, sigma_own, masses, v_start, h_real_start,
            sin_slope, atm_map, t_start, length_start, const, _vnFloorFor(v_start), h_real_floor,
            grav_drop_start=grav_drop_start)

        need_b = np.zeros(n_daughters, dtype=bool)
        b_state = [None]*n_daughters

        for i in range(n_daughters):
            traj_a, atm_a = batch_a[i]
            seg_a = {"traj": traj_a, "atm_height_fn": atm_a, "K": K, "rho": rho, "gamma": gamma,
                "sigma_own": sigma_own, "sigma_eff": sigma_own, "erosion_coeff": 0.0,
                "v_start": v_start, "m_start": float(masses[i]), "length_start": length_start,
                "grav_drop_start": grav_drop_start, "t_start": t_start, "t_end": None}

            if not const.erosion_on:
                seg_a["t_end"] = _findSegmentDeathTime(const, seg_a)
                segments_out[i].append(seg_a)
                continue

            t_cross = _timeAtReportedHeight(traj_a, const.erosion_height_start, length_start,
                const, grav_drop_start=grav_drop_start)

            if t_cross is None:
                seg_a["t_end"] = _findSegmentDeathTime(const, seg_a)
                segments_out[i].append(seg_a)
                continue

            seg_a["t_end"] = t_cross
            segments_out[i].append(seg_a)
            need_b[i] = True
            b_state[i] = (float(traj_a.velocityNormedAt(t_cross))*v_start,
                float(traj_a.massAt(t_cross)), float(traj_a.heightRealAt(t_cross)),
                length_start + float(traj_a.lengthAt(t_cross)),
                grav_drop_start + float(traj_a.gravityDropAt(t_cross)), t_cross)

        erosion_coeff_b = const.disruption_erosion_coeff
        sigma_eff_b = sigma_own + erosion_coeff_b
        for i in range(n_daughters):
            if not need_b[i]:
                continue
            v_b, m_b, h_b, length_b, grav_drop_b, t_b = b_state[i]
            traj_b, atm_b = _buildRefinedTrajectory(K, sigma_eff_b, m_b, v_b, h_b, sin_slope,
                atm_map, t_start=t_b, length_start=length_b, const=const,
                v_n_floor=_vnFloorFor(v_b), h_real_floor=h_real_floor, grav_drop_start=grav_drop_b,
                n_refine_passes=1)
            seg_b = {"traj": traj_b, "atm_height_fn": atm_b, "K": K, "rho": rho, "gamma": gamma,
                "sigma_own": sigma_own, "sigma_eff": sigma_eff_b, "erosion_coeff": erosion_coeff_b,
                "v_start": v_b, "m_start": m_b, "length_start": length_b,
                "grav_drop_start": grav_drop_b, "t_start": t_b, "t_end": None}
            seg_b["t_end"] = _findSegmentDeathTime(const, seg_b)
            segments_out[i].append(seg_b)

        return segments_out

    # Already at or below erosion_height_start at spawn for EVERY daughter (shared h_real_start) -
    # skip segment A entirely; segment B's own start state is shared too - fully batchable.
    erosion_coeff_b = const.disruption_erosion_coeff
    sigma_eff_b = sigma_own + erosion_coeff_b

    batch_b = _buildBatchedDaughterTrajectories(K, sigma_eff_b, masses, v_start, h_real_start,
        sin_slope, atm_map, t_start, length_start, const, _vnFloorFor(v_start), h_real_floor,
        grav_drop_start=grav_drop_start)

    for i in range(n_daughters):
        traj_b, atm_b = batch_b[i]
        seg_b = {"traj": traj_b, "atm_height_fn": atm_b, "K": K, "rho": rho, "gamma": gamma,
            "sigma_own": sigma_own, "sigma_eff": sigma_eff_b, "erosion_coeff": erosion_coeff_b,
            "v_start": v_start, "m_start": float(masses[i]), "length_start": length_start,
            "grav_drop_start": grav_drop_start, "t_start": t_start, "t_end": None}
        seg_b["t_end"] = _findSegmentDeathTime(const, seg_b)
        segments_out[i].append(seg_b)

    return segments_out


def _buildComplexFragmentDaughterSegments(const, K, sigma_own, erosion_coeff, m_start, v_start,
        h_real_start, sin_slope, atm_map, t_start, length_start, grav_drop_start, h_real_floor,
        erosion_mass_index=None, erosion_mass_min=None, erosion_mass_max=None, rho=None,
        gamma=None, n_refine_passes=1):
    """ Build a Stage 5 "F"/"EF" complex-fragmentation daughter's own segment chain - ALWAYS
    exactly ONE segment, unlike _buildDaughterFragmentSegments() (a Stage 4 disruption "fragment"
    daughter, which can have up to 3, one per erosion-threshold crossing).

    This simplification is exact, not approximate: an F daughter never erodes at all
    (erosion_enabled=False, so mass_loss_erosion is always 0 - MetSimErosion.py:754 gates the
    erosion mass-loss channel on frag.erosion_enabled), and an EF daughter erodes at a FIXED
    erosion_coeff for its ENTIRE remaining lifetime, never height-updated: every F/EF/D child gets
    frag_new.complex=True unconditionally (MetSimErosion.py:1145), and the standard height-based
    getErosionCoeff() auto-update every OTHER erosion-enabled fragment gets is gated by
    `not frag.complex` (MetSimErosion.py:976) - so there is no erosion_height_start/
    erosion_height_change branching to do here at all, unlike _buildDaughterFragmentSegments().
    Confirmed from source that grain-spawning itself is NOT gated by `not frag.complex` (only the
    erosion_coeff auto-update is, MetSimErosion.py:993's `if frag.erosion_enabled:` has no complex
    check) - so an EF daughter's single fixed-erosion_coeff segment still spawns grains normally
    via _spawnGrainsForSegment()/_spawnGrainSpecsForAllErodingSegments(), exactly like any other
    eroding segment.

    Arguments:
        const: [Constants]
        K, sigma_own: [float] This daughter's shape-density/ablation coefficients - constant for
            its entire lifetime (inherited from the parent fragment at the trigger instant, via
            _applyFragmentationEntry()'s own frag_daughters spec).
        erosion_coeff: [float] FIXED for this daughter's entire lifetime - 0.0 for an F daughter,
            entry.erosion_coeff for an EF daughter (frag_daughters' own "erosion_coeff_fixed" key).
        m_start, v_start, h_real_start: [float] State at spawn (the fragmentation trigger instant).
        sin_slope: [float] sin(entry slope) = cos(zenith_angle).
        atm_map: [AtmEquivHeightMap]
        t_start: [float] GLOBAL spawn time (s).
        length_start: [float] Cumulative path length (m) at spawn.
        grav_drop_start: [float] Cumulative gravity-drop (m) at spawn - inherited from the parent's
            own running total (spawn_child()'s dict copy, never reset to 0), same reasoning as
            _buildDaughterFragmentSegments()'s own grav_drop_start argument.
        h_real_floor: [float] See AnalyticTrajectory's own docstring.

    Keyword arguments:
        erosion_mass_index, erosion_mass_min, erosion_mass_max: [float or None] This daughter's own
            grain-size-distribution parameters (frag_daughters' own keys, sourced from the
            triggering entry's grain_mass_min/grain_mass_max/mass_index - NOT necessarily
            const's own values). Default to const's own values only if not given.
        rho, gamma: [float or None] This daughter's own bulk density/drag coefficient - constant
            for its entire lifetime (rho is never touched by F/EF; gamma only by a later Stage 5
            type "A" event). Carried explicitly so the returned segment dict can record them for
            _applyRetroactiveResplitting()'s own use - see _buildDaughterFragmentSegments()'s
            identical keyword arguments for the full reasoning. Default to const.rho/const.gamma.
        n_refine_passes: [int] Passed through to _buildRefinedTrajectory() - defaults to 1, not
            that function's own default of 3, for the same Stage 7 performance reasoning as
            _buildDaughterFragmentSegments()'s identical parameter (see its own docstring for the
            full account: AnalyticTrajectory.__init__'s cost is dominated by fixed per-call scipy
            overhead, not grid size, so halving the build COUNT per segment is what actually helps
            for the many-short-lived-daughters case complex fragmentation shares with disruption).

    Return:
        segments: [list of dict] Always exactly 1 entry, same dict shape as
            _buildMainFragmentSegments()'s own return (including rho/gamma/erosion_mass_index/
            min/max).
    """

    def _vnFloorFor(v):
        return max(0.01, 0.5*const.v_kill/v)

    rho = const.rho if rho is None else rho
    gamma = const.gamma if gamma is None else gamma

    sigma_eff = sigma_own + erosion_coeff
    traj, atm_fn = _buildRefinedTrajectory(K, sigma_eff, m_start, v_start, h_real_start, sin_slope,
        atm_map, t_start=t_start, length_start=length_start, const=const,
        v_n_floor=_vnFloorFor(v_start), h_real_floor=h_real_floor,
        grav_drop_start=grav_drop_start, n_refine_passes=n_refine_passes)

    seg = {"traj": traj, "atm_height_fn": atm_fn, "K": K, "rho": rho, "gamma": gamma,
        "sigma_own": sigma_own, "sigma_eff": sigma_eff, "erosion_coeff": erosion_coeff,
        "v_start": v_start, "m_start": m_start, "length_start": length_start,
        "grav_drop_start": grav_drop_start, "t_start": t_start, "t_end": None,
        "erosion_mass_index": const.erosion_mass_index if erosion_mass_index is None
            else erosion_mass_index,
        "erosion_mass_min": const.erosion_mass_min if erosion_mass_min is None
            else erosion_mass_min,
        "erosion_mass_max": const.erosion_mass_max if erosion_mass_max is None
            else erosion_mass_max}
    seg["t_end"] = _findSegmentDeathTime(const, seg)

    return [seg]


def _findDisruptionTime(const, segments):
    """ Find the GLOBAL time (s, an exact multiple of const.dt) at which the main fragment's own
    dynamic pressure first exceeds const.compressive_strength, scanning its segment chain in order
    - Stage 4's disruption trigger.

    dyn_press = rho_atm(h_reported(t))*v(t)^2 is not necessarily monotonic within a segment (as the
    body descends, atmosphere density rises while velocity falls, so dyn_press typically rises then
    falls - a well-known reason real dynamic pressure profiles peak partway through a flight) - this
    finds the FIRST tick where the threshold is exceeded, not the peak, exactly matching
    ablateAll()'s own per-tick check (`if dyn_press > const.compressive_strength:`, checked once per
    dt tick, triggering immediately the first time it holds - MetSimErosion.py:1027-1028). No
    continuous-time root-find/refinement is used (unlike _timeAtReportedHeight()'s erosion-boundary
    search): the reference itself only ever evaluates this check AT a dt tick, so the exact-matching
    answer is which TICK first satisfies it, not a finer-grained continuous crossing time - the same
    reasoning _findSegmentDeathTime() already uses for m_kill/v_kill/h_kill/len_kill.

    Arguments:
        const: [Constants]
        segments: [list of dict] The main fragment's FULL (natural-death) segment chain, from
            _buildMainFragmentSegments().

    Return:
        (t_disrupt, seg_idx): [tuple] GLOBAL time (s, a dt multiple) of the first tick where
        dyn_press exceeds compressive_strength, and the index (into segments) of the segment it
        falls within - or (None, None) if this never happens anywhere in the chain (including if
        the chain is empty, which should not occur in practice).
    """

    dt = const.dt

    for seg_idx, seg in enumerate(segments):

        k_start = int(math.ceil(seg["t_start"]/dt - 1e-9))
        k_end = int(math.floor(seg["t_end"]/dt + 1e-9))
        if k_end < k_start:
            continue

        t_ticks = np.arange(k_start, k_end + 1)*dt

        _, _, _, _, _, _, dyn_press, _ = _evaluateSegment(seg["traj"], seg["atm_height_fn"], t_ticks,
            seg["K"], seg["sigma_own"], seg["v_start"], seg["length_start"], const,
            grav_drop_start=seg.get("grav_drop_start", 0.0))

        over = dyn_press > const.compressive_strength
        if np.any(over):
            return float(t_ticks[int(np.argmax(over))]), seg_idx

    return None, None


def _stepGrainRK4(const, K, sigma, m0, v0, t0, length0):
    """ Evolve one grain-Fragment "bin" via EXACT per-tick RK4 stepping, replicating
    ablateAll()'s own discrete update sequence exactly (MetSimErosion.py:748-868) - NOT the
    closed-form AnalyticTrajectory machinery Stage 2/3 uses for the main fragment's own
    (long-lived) segments.

    Why grains need this and the main fragment does not: RK4's own fixed const.dt timestep is
    numerically UNRESOLVED for grains whose intrinsic dynamical timescale is comparable to or
    shorter than dt - confirmed directly: a grain near erosion_mass_min with a typical inherited
    sigma (matching Constants' own defaults, not an unusual test choice) can decelerate from full
    speed to below v_kill within ~4-5 ticks, with the FIRST single RK4 step alone already
    producing a ~37% velocity change. The closed-form solution (exact, continuous-time)
    legitimately disagrees with RK4's own (numerically inaccurate, but that is what the reference
    tool actually produces) answer in this regime, and this is not a negligible corner case:
    confirmed directly that even after frame-averaging at a realistic 30fps (matching
    DynestyMetSim.py's own integrateLuminosity(), which np.mean()s over all dt-samples in a
    1/fps window - not just a raw-dt peak comparison), the closed-form approach's light curve
    diverged from the reference's by up to 3.3 magnitudes (a ~20x brightness error) sustained over
    roughly the last second of a representative erosion-heavy test case - large enough to dominate
    any real fit, not something frame-averaging washes out.

    This is not a performance concern the way it would be for the main fragment: grains typically
    die within single-digit numbers of ticks (confirmed directly), so stepping RK4 tick-by-tick
    per grain is CHEAP (a handful of arithmetic operations) - likely cheaper, in fact, than
    building a full AnalyticTrajectory (VelocitySpline + iterative curvature refinement) for
    something this short-lived, not just "acceptably slow."

    Grains use FLAT (unaccumulated) gravity-drop starting from 0 at spawn, not inherited from the
    parent's own accumulated total (frag_child.h_grav_drop_total IS inherited via spawn_child()'s
    dict copy in the original) - grains live for so few ticks that re-deriving vs inheriting this
    term changes the result by a centimeter-scale amount at most (see Stage 2b's own gravity-drop
    magnitudes), not worth threading an extra parameter through for.

    Arguments:
        const: [Constants]
        K: [float] This grain's shape-density coefficient.
        sigma: [float] This grain's ablation coefficient (inherited from its parent segment).
        m0, v0: [float] Mass (kg) and speed (m/s, scalar) at spawn.
        t0: [float] GLOBAL spawn time (s) - MUST already be snapped to the nearest const.dt
            multiple by the caller (see _spawnGrainsForSegment), so this grain's own output ticks
            (t0+dt, t0+2dt, ...) land exactly on the same tick grid the rest of the simulation's
            output uses, with no interpolation needed at aggregation time.
        length0: [float] Cumulative path length (m) at spawn.

    Return:
        (t_arr, v_arr, m_arr, h_reported_arr, lum_arr, q_arr, dyn_press_arr, length_arr): ndarrays,
        one entry per GLOBAL dt-tick from t0+dt (the first tick after spawn) through this grain's
        own death tick (inclusive - matching the original's own convention of recording the state
        at the tick a kill condition is first satisfied, not the tick before). length_arr is
        additive (appended at the end, like Stage 2d's frag_main.n_queries) - needed by Stage 3d's
        results_list assembly for leading_frag_length/brightest_length, which are keyed on a
        fragment's own cumulative path length; every other field was already tracked internally for
        the height/curvature computation, so returning it too is free.
    """

    dt = const.dt
    zenith_angle = const.zenith_angle
    r_earth = const.r_earth
    cos_zenith = math.cos(zenith_angle)

    m, v = m0, v0
    vv = -v0*cos_zenith
    vh = v0*math.sin(zenith_angle)
    length = length0
    h_grav_drop_total = 0.0
    t = t0

    # heightCurvature() is a genuinely expensive-relative-to-everything-else-here Python function
    # (sqrt + several multiplications) - profiled directly as ~23% of this function's total
    # runtime, from being recomputed from scratch every tick, at realistic grain population sizes.
    # MOST grains die within a handful of ticks (confirmed directly), over which the curvature
    # CORRECTION (heightCurvature() minus the flat first-order term) barely changes (~1-5m over
    # 1000m of travel) - utterly negligible next to the ~10%-per-step gap this function already
    # accepts by design (see the docstring above). But a MINORITY of grains (spawned early/high, at
    # low ablation coefficient) survive for seconds and tens of km, over which this offset is NOT
    # negligible (confirmed directly: 31m at spawn growing to 414m after 74.6km of travel for one
    # such case - the offset's own rate of change grows with length, since heightCurvature() is
    # quadratic there). So the offset is refreshed periodically (every REFRESH_TICKS ticks, not
    # once and not every tick) - bounded staleness in TICKS translates to a roughly bounded
    # staleness in distance regardless of how long a given grain survives, while still cutting
    # heightCurvature() calls by ~REFRESH_TICKS-fold for the long-lived minority and to a single
    # call for the short-lived majority.
    REFRESH_TICKS = 20
    curvature_offset = (heightCurvature(const.h_init, zenith_angle, length, r_earth)
        - (const.h_init - length*cos_zenith))
    h = const.h_init - length*cos_zenith + curvature_offset - h_grav_drop_total

    t_list, v_list, m_list, h_list, lum_list, q_list, dp_list, len_list = (
        [], [], [], [], [], [], [], [])

    max_steps = int(10.0/dt)  # generous safety cap (10s) - grains should die within a handful of ticks

    for step_i in range(max_steps):

        rho_atm = atmDensityPoly(h, const.dens_co)

        mass_loss_ablation = massLossRK4(dt, K, sigma, m, rho_atm, v)
        m_new = m + mass_loss_ablation
        if m_new < 0:
            m_new = 0.0

        deceleration_total = decelerationRK4(dt, K, m, rho_atm, v)

        if deceleration_total > 0:
            vv = vh = v = 0.0
            deceleration_total = 0.0
        else:
            gv = G0/((1.0 + h/r_earth)**2)
            av = -deceleration_total*vv/v + vh*v/(r_earth + h)
            ah = -deceleration_total*vh/v - vv*v/(r_earth + h)
            h_grav_drop_total += 0.5*gv*dt**2
            vv -= av*dt
            vh -= ah*dt
            v = math.sqrt(vh**2 + vv**2)
            if vv > 0:
                vv = 0.0
                h = 0.0

        length += v*dt
        m = m_new
        if (step_i + 1) % REFRESH_TICKS == 0:
            curvature_offset = (heightCurvature(const.h_init, zenith_angle, length, r_earth)
                - (const.h_init - length*cos_zenith))
        h = const.h_init - length*cos_zenith + curvature_offset - h_grav_drop_total

        tau = luminousEfficiency(const.lum_eff_type, const.lum_eff, v, m)
        lum = -tau*((mass_loss_ablation/dt*v**2)/2.0 + m*v*deceleration_total)
        beta_ion = ionizationEfficiency(v)
        q = (-beta_ion*(mass_loss_ablation/dt)/(const.mu*v)) if v > 0 else 0.0
        dyn_press = 1.0*rho_atm*v**2

        t += dt


        t_list.append(t); v_list.append(v); m_list.append(m); h_list.append(h)
        lum_list.append(lum); q_list.append(q); dp_list.append(dyn_press); len_list.append(length)

        if ((m <= const.m_kill) or (v < const.v_kill) or (h < const.h_kill) or (lum < 0)
                or ((const.len_kill > 0) and (length > const.len_kill))):
            break

    else:
        raise RuntimeError("_stepGrainRK4: grain did not reach a kill condition within the 10s "
            "safety cap - check Constants for an unreachable m_kill/v_kill/h_kill configuration.")

    return (np.array(t_list), np.array(v_list), np.array(m_list), np.array(h_list),
        np.array(lum_list), np.array(q_list), np.array(dp_list), np.array(len_list))


def _findMassCrashOnset(m_arr, m_prev, ratio_threshold=0.5):
    """ Detect the onset of the closed-form segment model's own near-singular mass-blowup-near-
    death artifact (documented extensively in this file's Stage 4/9 write-ups, and precisely
    root-caused in the follow-up that added _stepErodingFragmentRK4Tail()/
    _resolveSegmentChainDeathRegime()): for a fragment with a large enough combined sigma_eff,
    the exact m(v) = m_start*exp(sigma_eff*(v^2-v_start^2)/2) relation this closed form solves
    is not itself singular (it has a finite, if possibly minuscule, floor as v->0), but its own
    STEEPNESS (d(ln m)/dv = sigma_eff*v) combined with a deceleration term that itself grows as
    m^(-1/3) creates a genuine positive-feedback runaway: velocity dropping makes mass drop,
    which makes deceleration grow, which drops velocity faster - a real, if idealized, feature of
    the underlying ODE system, not a numerical artifact of this file's own tabulation. Confirmed
    directly on a real disruption-daughter case (a leading_frag_length regression investigation):
    mass fell from 1.47e-7 kg to 3.2e-20 kg (7 orders of magnitude) over 4 consecutive GLOBAL
    ticks (0.02s), with velocity crashing from 13516 to 674 m/s in the same window - a physically
    implausible ~2.6 million m/s^2 deceleration no real body could sustain, and NOT what the
    reference tool's own coarse RK4 stepping produces for the equivalent object (confirmed
    directly: the reference's own analogous trajectory decelerates smoothly, 14313->12797->...->
    3108 m/s over 20 ticks/0.1s, right through the region the closed form collapses in).

    A ratio<0.5 threshold (mass more than halving in a SINGLE 0.005s tick) is a robust, cheap
    proxy for this runaway's onset, not an arbitrary cutoff: since dm/m = sigma_eff*v*dv over one
    tick, and dv itself is normally tiny (a small fraction of v) for the vast majority of a
    fragment's life, reaching ratio<0.5 in one tick requires deceleration ALREADY extreme enough
    that this is, in practice, only ever reached once the runaway has begun - confirmed never
    triggering (checked via the full existing 38+-test validation suite) on any of this file's
    already-validated non-pathological cases.

    Arguments:
        m_arr: [ndarray] Dense, tick-ordered mass values (kg) for one segment's own dense
            evaluation (e.g. from _evaluateFragmentSegments() applied to a single-segment chain).
        m_prev: [float] The mass (kg) at the tick immediately BEFORE m_arr[0] (i.e. the segment's
            own m_start) - checked too, so a crash landing on the very FIRST tick of a segment is
            still caught, not just crashes that onset after at least one good tick.
        ratio_threshold: [float] See above.

    Return:
        [int or None] The index INTO m_arr of the first tick whose mass ratio (vs the preceding
        tick, or vs m_prev for index 0) falls below ratio_threshold - i.e. the first tick that
        should NOT be trusted from the closed form. None if no such tick exists.
    """

    if len(m_arr) == 0:
        return None

    prev = np.concatenate([[m_prev], m_arr[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev > 0, m_arr/np.maximum(prev, 1e-300), 1.0)

    bad = np.where(ratio < ratio_threshold)[0]
    return int(bad[0]) if len(bad) else None


def _findSafeRestartTick(m_arr, m_prev, i_crash, safe_ratio=0.9):
    """ Given a crash tick already found by _findMassCrashOnset() (ratio_threshold=0.5), walk
    BACKWARD from it to find a tick whose OWN ratio (vs its own preceding tick) is comfortably
    above a much stricter safe_ratio - i.e. still outside the accelerating runaway that
    _findMassCrashOnset() only detects the LATE, already-extreme end of. Restarting the RK4 tail
    from here instead of from i_crash-1 directly (or from this segment's own start entirely - see
    below for why the latter is no longer necessary) keeps the tail SHORT (this file's own
    "eliminate O(timesteps) cost" goal) while avoiding a real, measured error: at i_crash-1
    specifically, the closed form's own mass can already differ from the true reference by a
    large factor even though its OWN tick-to-tick ratio hasn't yet crossed 0.5.

    History, for anyone revisiting this (see project memory / plan for the full account): an
    EARLIER version of this fix existed, was found insufficient, and was replaced with "always
    restart from the segment's own start" (full RK4 for the whole segment) - NOT because this
    backward-scan approach was wrong in principle, but because at the time, a SEPARATE, unrelated
    bug (_buildDaughterFragmentSegments() using the wrong, height-based erosion_coeff for a
    disruption daughter instead of the fixed const.disruption_erosion_coeff the reference actually
    uses - see that function's own docstring) was making the closed form diverge measurably from
    the very START of a daughter's life, not just near its own death - no restart-point choice
    within the segment could have compensated for that. With THAT bug fixed, the closed form was
    confirmed (not assumed) to track the reference closely again for the vast majority of a
    daughter's life (typically <1% mass error until the final ~50-100ms before its own true
    collapse) - restoring this narrower, cheaper backward-scan restart instead of paying full RK4
    for a fragment's entire life.

    Arguments:
        m_arr: [ndarray] Same dense mass array _findMassCrashOnset() was called with.
        m_prev: [float] Same m_prev _findMassCrashOnset() was called with.
        i_crash: [int] The index _findMassCrashOnset() returned (not None - callers only reach
            here once a crash was actually found).
        safe_ratio: [float] How close to 1.0 (no mass loss) a tick's own ratio must be to be
            trusted as an RK4-tail restart point.

    Return:
        [int or None] Index into m_arr of the latest tick at or before i_crash-1 whose own ratio
        is >= safe_ratio. None if no such tick exists in this array at all (the whole segment is
        already within the unsafe regime) - callers should fall back to the segment's OWN start in
        that case.
    """

    prev = np.concatenate([[m_prev], m_arr[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev > 0, m_arr/np.maximum(prev, 1e-300), 1.0)

    for j in range(i_crash - 1, -1, -1):
        if ratio[j] >= safe_ratio:
            return j

    return None


def _stepErodingFragmentRK4Tail(const, K, sigma_own, erosion_coeff, m0, v0, t0, length0,
        grav_drop0, erosion_mass_index, erosion_mass_min, erosion_mass_max, n_grains=1.0):
    """ Evolve ONE erosion-capable fragment (the main fragment's own final segment, or a
    daughter's) via EXACT per-tick RK4 stepping, replicating ablateAll()'s own discrete update
    sequence for a single (non-grain) fragment exactly (MetSimErosion.py:748-1024), including its
    OWN recursive per-tick erosion-channel grain-spawning - the RK4-tail counterpart to
    _stepGrainRK4() (which handles non-eroding, already-spawned grains), built specifically to fix
    the near-singular mass-blowup-near-death artifact _findMassCrashOnset() detects (see that
    function's own docstring for the full physical/numerical account).

    Deliberately NOT used for a fragment's ENTIRE life (that would reintroduce the exact
    O(timesteps x fragments) Python-loop cost this whole engine exists to eliminate - see this
    file's own implementation-plan "Feasibility verdict"): _resolveSegmentChainDeathRegime() only
    ever invokes this from the LAST safe tick before a detected crash onset, through to this
    fragment's own true death - by construction a short stretch (single-digit to low tens of
    ticks in every case measured so far), since the crash-onset condition is itself a proxy for
    "already very close to death". Every height-triggered transition (erosion_height_start/
    _height_change, complex-fragmentation entries, disruption) that could still lie ahead of a
    fragment is ALREADY resolved as an earlier segment boundary by the time this function is ever
    invoked (only ever called on a chain's own LAST segment, which by construction has no further
    such boundary before its own natural death) - so, unlike ablateAll() itself, this function
    only needs to check natural-death kill conditions each tick, not re-derive any of those other
    triggers.

    Arguments:
        const: [Constants]
        K: [float] This fragment's shape-density coefficient (already reflecting any Stage 5 "A"
            event that changed gamma/rho before this segment began - read directly off the
            segment dict, not re-derived from const.gamma).
        sigma_own: [float] This fragment's OWN ablation coefficient (radiates; excludes erosion).
        erosion_coeff: [float] This fragment's current erosion coefficient (0.0/non-positive if
            not erosion-enabled or erosion has not started - degrades this function to a pure
            per-tick ablation-only stepper with no grain-spawning, same physics core as
            _stepGrainRK4() then applies to a non-grain fragment instead).
        m0, v0: [float] Mass (kg) / speed (m/s) at the LAST safe (pre-crash) tick.
        t0: [float] GLOBAL time (s) at that tick - MUST already be on the const.dt grid (see
            _resolveSegmentChainDeathRegime() for how this is ensured, including the rare
            "crash onset on this segment's own very first tick" case).
        length0: [float] Cumulative path length (m) at that tick.
        grav_drop0: [float] This fragment's own ACCUMULATED gravity-drop (m) at that tick -
            reconstructed by the caller from heightCurvature(...) - h_reported, NOT restarted from
            0.0 the way _stepGrainRK4() does for freshly-spawned grains: this function resumes a
            fragment that may already be seconds/many km into its own life (main, or a long-lived
            daughter), where Stage 4's own finding (gravity-drop is genuinely cumulative and can
            reach tens of meters for long-lived fragments) applies, unlike a grain's own
            centimeter-scale total over its short remaining life.
        erosion_mass_index, erosion_mass_min, erosion_mass_max: [float] Grain-size-distribution
            parameters for this fragment's own tail-spawned grains (from the segment dict, falling
            back to const's own defaults - same convention _spawnGrainsForSegment() uses).
        n_grains: [float] This fragment's OWN parent multiplicity, same convention/mechanism as
            _spawnGrainsForSegment()'s own n_grains parameter - see _makeVirtualParentFragment()'s
            docstring. Defaults to 1.0 (correct for main); a daughter with its own n_grains>1 must
            pass it through here too, since this tail spawns grains from the SAME kind of virtual
            parent the closed-form epoch mechanism does.

    Return:
        (t_arr, v_arr, m_arr, h_reported_arr, lum_arr, q_arr, dyn_press_arr, length_arr, tau_arr,
        grain_specs): ndarrays, one entry per GLOBAL dt-tick from t0+dt through this fragment's own
        death tick (inclusive), plus grain_specs - a flat list of dicts (same shape
        _spawnGrainsForSegment() produces: m, n_grains, sigma, rho, K, v, h_real, length, t) from
        every tick this tail's own erosion channel produced eroded mass, ready to feed directly
        into _batchAndStepGrainSpecs() alongside disruption-leftover/D-dust grain_specs.
    """

    dt = const.dt
    zenith_angle = const.zenith_angle
    r_earth = const.r_earth
    cos_zenith = math.cos(zenith_angle)

    m, v = m0, v0
    vv = -v0*cos_zenith
    vh = v0*math.sin(zenith_angle)
    length = length0
    h_grav_drop_total = grav_drop0
    t = t0

    h = heightCurvature(const.h_init, zenith_angle, length, r_earth) - h_grav_drop_total

    t_list, v_list, m_list, h_list, lum_list, q_list, dp_list, len_list, tau_list = (
        [], [], [], [], [], [], [], [], [])
    grain_specs = []

    max_steps = int(10.0/dt)  # generous safety cap - this tail is only ever a short stretch

    for step_i in range(max_steps):

        rho_atm = atmDensityPoly(h, const.dens_co)

        mass_loss_ablation = massLossRK4(dt, K, sigma_own, m, rho_atm, v)
        mass_loss_erosion = (massLossRK4(dt, K, erosion_coeff, m, rho_atm, v)
            if erosion_coeff > 0 else 0.0)
        mass_loss_total = mass_loss_ablation + mass_loss_erosion
        # Simplified relative to ablateAll()'s own (MetSimErosion.py:763-764) near-zero clip -
        # matches _stepGrainRK4()'s own already-established precedent (clip to exactly 0.0) rather
        # than replicating that formula's own odd non-zero floor; only ever affects the last tick
        # or two before death, well within the near-death noise band already accepted throughout
        # this file.
        m_new = max(0.0, m + mass_loss_total)

        deceleration_total = decelerationRK4(dt, K, m, rho_atm, v)

        if deceleration_total > 0:
            vv = vh = v = 0.0
            deceleration_total = 0.0
        else:
            gv = G0/((1.0 + h/r_earth)**2)
            av = -deceleration_total*vv/v + vh*v/(r_earth + h)
            ah = -deceleration_total*vh/v - vv*v/(r_earth + h)
            h_grav_drop_total += 0.5*gv*dt**2
            vv -= av*dt
            vh -= ah*dt
            v = math.sqrt(vh**2 + vv**2)
            if vv > 0:
                vv = 0.0
                h = 0.0

        length += v*dt
        m = m_new
        h = heightCurvature(const.h_init, zenith_angle, length, r_earth) - h_grav_drop_total

        tau = luminousEfficiency(const.lum_eff_type, const.lum_eff, v, m)
        lum = -tau*((mass_loss_ablation/dt*v**2)/2.0 + m*v*deceleration_total)
        beta_ion = ionizationEfficiency(v)
        q = (-beta_ion*(mass_loss_ablation/dt)/(const.mu*v)) if v > 0 else 0.0
        dyn_press = 1.0*rho_atm*v**2

        t += dt

        t_list.append(t); v_list.append(v); m_list.append(m); h_list.append(h)
        lum_list.append(lum); q_list.append(q); dp_list.append(dyn_press); len_list.append(length)
        tau_list.append(tau)

        # Per-tick erosion-channel grain spawning, mirroring ablateAll()'s own unconditional
        # every-tick generateFragments() call during erosion (MetSimErosion.py:992-1024) exactly -
        # cheap and exact here specifically BECAUSE this tail is short by construction (see this
        # function's own docstring).
        if erosion_coeff > 0 and abs(mass_loss_erosion) > 0:
            h_real = const.h_init - length*cos_zenith
            parent = _makeVirtualParentFragment(const, m, v, h, length, sigma_own,
                erosion_mass_index=erosion_mass_index, erosion_mass_min=erosion_mass_min,
                erosion_mass_max=erosion_mass_max, n_grains=n_grains)
            frag_children, _ = generateFragments(const, parent, abs(mass_loss_erosion),
                erosion_mass_index, erosion_mass_min, erosion_mass_max,
                keep_eroding=False, mass_model=const.erosion_grain_distribution)
            for fc in frag_children:
                grain_specs.append({"m": fc.m, "n_grains": fc.n_grains, "sigma": fc.sigma,
                    "rho": fc.rho, "K": fc.K, "v": v, "h_real": h_real, "length": length, "t": t})

        if ((m <= const.m_kill) or (v < const.v_kill) or (h < const.h_kill) or (lum < 0)
                or ((const.len_kill > 0) and (length > const.len_kill))):
            break

    else:
        raise RuntimeError("_stepErodingFragmentRK4Tail: fragment did not reach a kill condition "
            "within the 10s safety cap - check Constants for an unreachable m_kill/v_kill/h_kill "
            "configuration.")

    return (np.array(t_list), np.array(v_list), np.array(m_list), np.array(h_list),
        np.array(lum_list), np.array(q_list), np.array(dp_list), np.array(len_list),
        np.array(tau_list), grain_specs)


def _resolveSegmentChainDeathRegime(const, segments, n_grains=1.0):
    """ Detect the closed-form near-singular mass-blowup-near-death artifact
    (_findMassCrashOnset()'s own docstring has the full account) in a fully-resolved segment
    chain's own LAST segment, and if found, replace it with a truncated version (t_end pulled
    back to the last trustworthy tick) carrying a "rk4_tail" continuation
    (_stepErodingFragmentRK4Tail()) that _evaluateFragmentSegments() knows how to splice in. A
    no-op (returns segments unchanged, no grain_specs) for the overwhelming majority of chains,
    which never enter this regime - confirmed via the full pre-existing 38+-test suite, none of
    which trigger this path.

    Must be called on a chain BEFORE anything else consumes it (_spawnGrainSpecsForAllErodingSegments()
    for grain-spawning epoch allocation, _evaluateFragmentSegments() for dense evaluation) - both
    need the (possibly truncated) t_end this function may produce, not the original
    _findSegmentDeathTime()-resolved one, to avoid double-counting eroded mass across the
    boundary between the closed-form prefix and the RK4 tail.

    Arguments:
        const: [Constants]
        segments: [list of dict] A fully-resolved segment chain (main's own, or one daughter's).

    Keyword arguments:
        n_grains: [float] This chain's own parent multiplicity, threaded through to
            _stepErodingFragmentRK4Tail()'s own grain-spawning - see
            _makeVirtualParentFragment()'s docstring for the full mechanism. Defaults to 1.0
            (main); a daughter with its own n_grains>1 must pass it through here.

    Return:
        (segments, tail_grain_specs): segments is the ORIGINAL list object if no crash was found,
        or a NEW list (sharing every earlier segment dict, with a new dict for the last one) if a
        tail was attached. tail_grain_specs is a flat list of grain_specs-shaped dicts (empty if no
        crash found) - the caller must fold these into whatever accumulator feeds
        _batchAndStepGrainSpecs() (the same one-shot mechanism disruption-leftover/D-dust grains
        already use), NOT _spawnGrainSpecsForAllErodingSegments() (which expects segment dicts).
    """

    if len(segments) == 0:
        return segments, []

    last_seg = segments[-1]
    tick_idx, v, m, length, h, _lum, _q, _dp, _tau, _nq, _tgs = _evaluateFragmentSegments(
        const, [last_seg])

    if len(m) == 0:
        return segments, []

    i_crash = _findMassCrashOnset(m, last_seg["m_start"])
    if i_crash is None:
        return segments, []

    # Restart the RK4 tail from a tick the closed form is STILL TRUSTWORTHY at - see
    # _findSafeRestartTick()'s own docstring for the full history (an earlier version of this
    # function restarted from i_crash-1 directly, found up to 4.7x over-massed there; a later
    # version, chasing a still-unexplained death-time bias, discarded the closed form for the
    # WHOLE segment - genuinely correct at the time, but only because a SEPARATE bug elsewhere
    # (_buildDaughterFragmentSegments() using the wrong erosion_coeff) was making the closed form
    # diverge from very early in a daughter's life; with that fixed, the narrower, cheaper
    # backward-scan restart below is correct again and avoids paying full per-tick RK4 for a
    # fragment's entire life). j_safe is None both when i_crash==0 (nothing before the segment's
    # own first tick to look at) and when no tick in this segment ever satisfies the stricter
    # safe_ratio (the whole segment is already within the unsafe regime) - both cases correctly
    # fall back to the segment's own start below.
    j_safe = _findSafeRestartTick(m, last_seg["m_start"], i_crash) if i_crash > 0 else None

    if j_safe is None:
        # This segment's own t_start generally is NOT on the const.dt grid (it comes from a
        # continuous root-find, e.g. _timeAtReportedHeight()) - snapped here to the nearest tick,
        # matching this file's own established precedent for exactly this kind of mismatch
        # (_spawnGrainsForSegment()'s own "snap epoch placement to the nearest real tick" comment).
        t0 = round(last_seg["t_start"]/const.dt)*const.dt
        v0 = last_seg["v_start"]
        m0 = last_seg["m_start"]
        length0 = last_seg["length_start"]
        grav_drop0 = last_seg.get("grav_drop_start", 0.0)
    else:
        t0 = float(tick_idx[j_safe] + 1)*const.dt
        v0 = float(v[j_safe])
        m0 = float(m[j_safe])
        length0 = float(length[j_safe])
        grav_drop0 = (heightCurvature(const.h_init, const.zenith_angle, length0, const.r_earth)
            - float(h[j_safe]))

    (t_tail, v_tail, m_tail, h_tail, lum_tail, q_tail, dp_tail, len_tail, tau_tail,
        grain_specs_tail) = _stepErodingFragmentRK4Tail(const, last_seg["K"],
        last_seg["sigma_own"], last_seg.get("erosion_coeff", 0.0), m0, v0, t0, length0,
        grav_drop0, last_seg.get("erosion_mass_index", const.erosion_mass_index),
        last_seg.get("erosion_mass_min", const.erosion_mass_min),
        last_seg.get("erosion_mass_max", const.erosion_mass_max), n_grains=n_grains)

    new_last_seg = dict(last_seg)
    new_last_seg["t_end"] = t0
    new_last_seg["rk4_tail"] = {"t": t_tail, "v": v_tail, "m": m_tail, "h": h_tail,
        "lum": lum_tail, "q": q_tail, "dyn_press": dp_tail, "length": len_tail, "tau": tau_tail,
        "grain_specs": grain_specs_tail}

    return segments[:-1] + [new_last_seg], grain_specs_tail


def _atmDensityPolyVec(ht, dens_co):
    """ Vectorized (numpy-array-capable) port of MetSimErosionCyTools.atmDensityPoly() - IDENTICAL
    formula, transcribed directly from that Cython source rather than re-derived, since
    atmDensityPoly() itself only accepts a scalar height (its ht argument is a typed C double, not
    an array) and is called millions of times by _stepGrainPopulationRK4()'s per-tick loop across a
    whole grain population - see that function's docstring for why a per-grain Python-level call
    into the scalar Cython function was measured to be too slow to use here. Validated directly
    against atmDensityPoly() to float precision during development (this specific vectorized port
    is exercised indirectly, not by a standalone dedicated test, by every erosion-scenario accuracy
    test in this file - e.g. test_run_simulation_erosion_accuracy()).
    """

    ht_scaled = np.asarray(ht, dtype=float)/1e6
    return 10**(dens_co[0] + dens_co[1]*ht_scaled + dens_co[2]*ht_scaled**2
        + dens_co[3]*ht_scaled**3 + dens_co[4]*ht_scaled**4 + dens_co[5]*ht_scaled**5
        + dens_co[6]*ht_scaled**6)


def _massLossRK4Vec(dt, K, sigma, m, rho_atm, v):
    """ Vectorized port of MetSimErosionCyTools.massLossRK4() - see _atmDensityPolyVec()'s
    docstring for why this exists alongside the scalar Cython original. Transcribed directly
    (same RK4-of-the-mass-loss-ODE structure, same negative-mass clamping at each sub-stage), not
    re-derived, to minimize the chance of a transcription error changing the physics.
    """

    def _massLoss(m_):
        return -K*sigma*np.power(m_, 2/3.0)*rho_atm*v**3

    mk1 = dt*_massLoss(m)
    mk1 = np.where(-mk1/2.0 > m, -m*2.0, mk1)

    mk2 = dt*_massLoss(m + mk1/2.0)
    mk2 = np.where(-mk2/2.0 > m, -m*2.0, mk2)

    mk3 = dt*_massLoss(m + mk2/2.0)
    mk3 = np.where(-mk3 > m, -m, mk3)

    mk4 = dt*_massLoss(m + mk3)

    return mk1/6.0 + mk2/3.0 + mk3/3.0 + mk4/6.0


def _decelerationRK4Vec(dt, K, m, rho_atm, v):
    """ Vectorized port of MetSimErosionCyTools.decelerationRK4() - see _atmDensityPolyVec()'s
    docstring for why this exists alongside the scalar Cython original.
    """

    def _deceleration(v_):
        return -K*np.power(m, -1/3.0)*rho_atm*v_**2

    vk1 = dt*_deceleration(v)
    vk2 = dt*_deceleration(v + vk1/2.0)
    vk3 = dt*_deceleration(v + vk2/2.0)
    vk4 = dt*_deceleration(v + vk3)

    return (vk1/6.0 + vk2/3.0 + vk3/3.0 + vk4/6.0)/dt


def _ionizationEfficiencyVec(vel):
    """ Vectorized port of MetSimErosionCyTools.ionizationEfficiency() - a simple closed-form
    algebraic formula (unlike luminousEfficiency(), no branching on a model type), so this is a
    direct, unconditional transcription - see _atmDensityPolyVec()'s docstring for why a vectorized
    port is needed alongside the scalar Cython original. vel is floored at 1 m/s (1e-3 km/s, well
    below const.v_kill's usual range) purely to keep log10() finite for a zero-velocity input (the
    scalar original has the same log10(0) singularity - it is just never called at vel=0 by any
    existing caller); callers that reach vel=0 (e.g. _stepGrainPopulationRK4()'s "accelerating"
    branch) always mask this value out of their own output anyway.
    """

    vel_km = np.maximum(np.asarray(vel, dtype=float), 1.0)/1000.0
    return 10**(5.84 - 0.09*vel_km**0.5 - 9.56/np.log10(vel_km))


def _stepGrainPopulationRK4(const, K, sigma, m0, v0, t0, length0, n_grains):
    """ Vectorized population version of _stepGrainRK4() - steps every grain in a POPULATION
    SHARING THE SAME (K, sigma) together, one Python-level loop iteration per GLOBAL dt-tick (not
    one per grain), each iteration a handful of vectorized numpy array operations across whichever
    grains are currently active. Every grain in an eroding segment DOES share the same (K, sigma):
    K depends only on (const.gamma, const.shape_factor, const.rho_grain), all simulation-wide
    constants (MetSimErosion.py's generateFragments():644-646 sets frag_child.rho = const.rho_grain
    unconditionally for plain erosion grains - not per-bin), and sigma is inherited unchanged from
    the segment's own sigma_own - only (m0, v0, t0, length0) vary per grain/bin.

    This exists because calling the scalar _stepGrainRK4() once per grain bin - the natural,
    already-validated (Stage 3c) approach - was measured directly to make the WHOLE erosion
    pipeline SLOWER than the RK4 reference it exists to replace: at n_epochs=1000 (needed for
    sub-0.15-mag frame-averaged light-curve accuracy - see runSimulation()'s erosion path
    docstring), a representative case spawned 34000 grain bins, and stepping each with its own
    Python-level _stepGrainRK4() call took 19.4s versus the reference's own 7.5s (0.4x "speedup" -
    i.e. a slowdown), even though _stepGrainRK4()'s CORE per-tick math is cheap - pure Python
    function-call and loop-setup overhead repeated 34000 times, not the arithmetic itself. This
    function fixes that by keeping the SAME per-tick algorithm (validated exhaustively in Stage 3c
    via _stepGrainRK4() and this file's tests) but restructuring the loop nesting: iterate over
    ticks, vectorize over grains - the "flat population table, evaluated via vectorized array math"
    architecture the implementation plan's own "Architecture" section calls for, applied to grains
    the same way Stage 1-3b already applies it to the main fragment's own segments.

    massLossRK4()/decelerationRK4()/atmDensityPoly() (MetSimErosionCyTools.pyx) are scalar-only
    Cython cpdef functions (typed C doubles, not arrays) - calling them per-grain-per-tick would
    just relocate the same overhead problem one level down. _massLossRK4Vec()/_decelerationRK4Vec()/
    _atmDensityPolyVec() are vectorized ports of their EXACT formulas (transcribed directly from the
    .pyx source, not re-derived), validated to match the scalar originals to float precision during
    development (exercised indirectly by every erosion-scenario accuracy test in this file).
    luminousEfficiency()/ionizationEfficiency() remain
    scalar-only Cython calls, looped in Python same as everywhere else in this file
    (_evaluateSegment(), _analyticGrainState()) - but only over the CURRENTLY ACTIVE subset at each
    tick, which shrinks rapidly (most grains die within a handful of ticks - confirmed repeatedly
    throughout this session), so the total (tick, active-grain) pair count stays far below
    N_grains x max_ticks.

    Arguments:
        const: [Constants]
        K, sigma: [float] Shared shape-density and ablation coefficients for this whole population.
        m0, v0, t0, length0: [ndarray] One entry per grain (mass, speed, GLOBAL spawn time, and
            cumulative path length at spawn) - see _stepGrainRK4()'s own docstring for units/
            conventions; t0 must already be snapped to the global const.dt grid, same requirement.
        n_grains: [ndarray] Number of physical grains each bin/row represents (Fragment.n_grains) -
            carried through only to be repeated into the output (this function does not itself
            weight lum/q/mass by it - callers follow the same n_grains-weighting convention
            documented in _runSimulationErosion()'s own docstring, points 1/2: lum/q and, since
            MetSimErosion.py commit 6be7301, mass_total_active too are all weighted by n_grains).

    Return:
        (global_tick_idx, v, m, h, lum, q, dyn_press, length, n_grains_out, is_death_tick,
        grain_id): flat ndarrays, one entry per (grain, tick) actually computed - i.e. the
        concatenation of what N separate _stepGrainRK4() calls would have returned, in the same
        per-grain order and convention (inclusive of each grain's own death tick), just without the
        per-call Python overhead. global_tick_idx is 0-INDEXED (matching a results_list row
        position, i.e. GLOBAL time = (global_tick_idx+1)*const.dt) - see runSimulation()'s erosion
        path for how this indexes into the shared output arrays. grain_id is an index into the
        ORIGINAL m0/v0/t0/length0/n_grains arrays, identifying which grain each row belongs to
        (needed since rows for different grains are interleaved by tick, not grouped by grain).
    """

    dt = const.dt
    cos_zenith = math.cos(const.zenith_angle)
    sin_zenith = math.sin(const.zenith_angle)
    # Explicit float64 + contiguity: stepGrainPopulationFull() (Cython) declares this argument as a
    # typed np.ndarray[float64] - const.dens_co is already a float64 ndarray in practice, but this
    # guarantees it regardless of caller.
    dens_co = np.ascontiguousarray(const.dens_co, dtype=np.float64)

    m0 = np.asarray(m0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    length0 = np.asarray(length0, dtype=float)
    n_grains = np.asarray(n_grains, dtype=float)

    if len(m0) == 0:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=np.int64)
        empty_b = np.array([], dtype=bool)
        return (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f,
            empty_b, empty_i)

    max_steps_per_grain = int(10.0/dt)  # matches _stepGrainRK4()'s own generous safety cap

    # Entire per-tick loop - spawn detection, active-set bookkeeping, AND the physics - fused into
    # one Cython call. Profiling (post the earlier stepGrainPopulationTick-only fusion, and post a
    # pure-numpy attempt at optimizing just this function's own outer bookkeeping - a "compact
    # active array instead of full-N boolean scans" rewrite that barely moved the needle, ~1.06x on
    # the largest real call measured) found the remaining dominant cost was PYTHON LOOP ITERATION
    # COUNT itself (hundreds of ticks, each running several small numpy statements with real fixed
    # dispatch overhead), not per-tick array size - so no amount of restructuring the array
    # bookkeeping in pure Python/numpy could remove it; only moving the loop itself into Cython
    # could. See MetSimErosionAlphaBetaCyTools.stepGrainPopulationFull()'s own docstring for the
    # full architecture (presorted spawn order, in-place active-set compaction, a growable output
    # buffer). Validated bit-for-bit-equivalent to the PRIOR (compact-active-array,
    # stepGrainPopulationTick-per-tick) implementation of this exact function by
    # test_step_grain_population_rk4_matches_scalar_reference().
    (idx, v, m, h, lum, q, dyn_press, length, is_death_tick, grain_id) = stepGrainPopulationFull(
        m0, v0, t0, length0, K, sigma, dt, cos_zenith, sin_zenith, const.r_earth, dens_co, G0,
        const.mu, const.h_init, const.lum_eff_type, const.lum_eff, const.m_kill, const.v_kill,
        const.h_kill, const.len_kill, max_steps_per_grain)

    return (idx, v, m, h, lum, q, dyn_press, length, n_grains[grain_id],
        is_death_tick.astype(bool), grain_id)


def _analyticGrainState(const, K, sigma, m0, v0, h0, t0, length0, atm_map, sin_slope, n_grid=41):
    """ Lightweight, direct closed-form evaluation of a grain's trajectory from spawn to death -
    NO AnalyticTrajectory/VelocitySpline/Chebyshev-grid machinery (superseded _stepGrainRK4() as
    Stage 3's grain evolution mechanism - see the implementation plan for the full account).

    AnalyticTrajectory's 300-point Chebyshev-grid + PchipInterpolator time quadrature was built and
    validated for main-body-scale trajectories spanning tens of km - it has a real, unresolved
    degeneracy for grains' much narrower (often under 1km) real-height range, confirmed directly on
    a real case where its own internal time quadrature diverged to t_hi=12457s for a grain that
    actually lives ~0.025s (h_real_end itself, from the closed-form hEquivFromVn(), was computed
    correctly - the bug was specifically in the grid-based antiderivative built on top of it).

    This function evaluates the EXACT closed-form alpha-beta solution directly instead, and was
    validated (see the implementation plan) to match a fine-resolution RK4 mirror to <0.3% for a
    representative extreme-alpha grain - confirming the closed form itself was never the problem.
    This deliberately does NOT replicate RK4's own coarse-dt numerical behavior the way
    _stepGrainRK4() does: the reference model's fixed dt=0.005 is itself numerically unresolved for
    these grains (confirmed directly: a single RK4 step can already overshoot ~37% of a grain's
    velocity), so true continuous-time physics necessarily leaves a real, understood ~10%-per-step
    gap against the reference's own answer - an accepted trade-off (physical accuracy over
    reference-numerical-matching), not a bug to chase further.

    Grains live over such a narrow real-height range that curvature and gravity-drop corrections
    are utterly negligible (curvature only matters over hundreds of km, per l_return; gravity-drop
    accumulates on the order of millimeters over a fraction of a second, per Stage 2b's own
    measurements over MUCH longer durations) - flat height is used directly as the reported height,
    with no reportedHeightAt()/atm_height_fn/curvature-Jacobian machinery.

    The one genuinely hard part - t as a function of h has no closed form, since it depends on the
    atmosphere's column density (same reason AnalyticTrajectory needs a quadrature at all) - is
    handled here with a SINGLE vectorized evaluation of the (positive-definite) integrand
    g(v_n) = m(v_n)^(1/3)/(K*rho(v_n)*v_n^2*v0) on a fixed, modest n_grid-point grid spanning
    [v_n_floor, 1], followed by one manual trapezoidal cumulative sum - not
    scipy.integrate.quad(), which was tried first and found to cost far more than the simple
    per-tick RK4 stepping it was meant to beat (each of n_grid-1 separate adaptive quad() calls
    re-evaluates its integrand many times internally; the vectorized version pays for the whole
    grid once). n_grid=41 was checked directly against both scipy.integrate.quad's answer and a
    fine-resolution RK4 mirror for representative grains without a meaningful accuracy change (see
    the implementation plan's convergence note).

    Arguments:
        const: [Constants]
        K, sigma: [float] This grain's shape-density and ablation coefficients.
        m0, v0, h0: [float] Mass (kg), speed (m/s), and FLAT real height (m) at spawn.
        t0: [float] GLOBAL spawn time (s) - should already be snapped to the nearest const.dt
            multiple by the caller (see _spawnGrainsForSegment), so output ticks land exactly on
            the simulation's own output grid.
        length0: [float] Cumulative path length (m) at spawn.
        atm_map: [AtmEquivHeightMap] Shared across the whole simulation.
        sin_slope: [float] sin(entry slope) = cos(zenith_angle).

    Keyword arguments:
        n_grid: [int] Number of (v_n, t) table points spanning [v_n_floor, 1] used to invert the
            time quadrature - grains' narrow v_n range needs far fewer than AnalyticTrajectory's
            300 (see docstring above).

    Return:
        (t_arr, v_arr, m_arr, h_arr, lum_arr, q_arr, dyn_press_arr, length_arr): ndarrays, one
        entry per GLOBAL dt-tick from t0+dt through this grain's own death tick (inclusive).
        length_arr is additive (appended at the end, matching _stepGrainRK4()'s own return
        contract) - already computed internally, just not previously returned.
    """

    dt = const.dt
    v_n_floor = max(0.01, 0.5*const.v_kill/v0)

    alpha = alphaFromPhysical(K, sin_slope, m0)
    beta = betaFromPhysical(sigma, v0)
    h_equiv_start = float(atm_map.toEquiv(h0))

    # Fixed grid, DESCENDING in v_n (1.0 -> v_n_floor). hEquivFromVn()/massFromVelocityNormed()/
    # atm_map.toReal() all accept array input; atmDensityPoly() is a scalar Cython function (not
    # vectorized), so it alone is looped - same as everywhere else it's called in this file.
    v_n_grid = np.linspace(1.0, v_n_floor, n_grid)
    h_equiv_grid = hEquivFromVn(v_n_grid, alpha, beta, h_equiv_start)
    h_real_grid = atm_map.toReal(h_equiv_grid)
    m_grid = massFromVelocityNormed(v_n_grid, v0, sigma, m0)
    rho_grid = np.array([atmDensityPoly(float(h), const.dens_co) for h in h_real_grid])

    # dt/dv_n = -g(v_n) (t increases as v_n decreases - see _massLossRate()/AlphaBeta.py's own
    # dv/dh derivation for where this integrand comes from); g is positive-definite, so the
    # trapezoidal rule on this DESCENDING grid (dv_n = -diff(v_n_grid), positive) gives a positive,
    # monotonically increasing t_grid directly - no sign juggling needed at the call site.
    g_grid = m_grid**(1.0/3.0)/(K*rho_grid*v_n_grid**2*v0)
    dv_n = -np.diff(v_n_grid)
    t_grid = np.concatenate(([0.0], np.cumsum(0.5*(g_grid[:-1] + g_grid[1:])*dv_n)))

    # Fully vectorized from here on (a per-tick Python loop calling np.interp with SCALAR inputs
    # was tried first and profiled directly: for grains living hundreds of ticks - the early,
    # high-altitude ones in a segment can survive seconds, not just the ~0.02s late/deep ones this
    # function was originally validated against - that meant up to ~10 MILLION individual np.interp
    # calls across a realistic grain population, each paying numpy's general-purpose per-call
    # overhead for a single scalar lookup. Batching all candidate ticks into one np.interp() call
    # each removes that overhead entirely - same pattern _evaluateSegment()/runSimulation()'s own
    # kill-search already use for the main fragment, just not yet applied here originally.
    n_candidate = max(int(t_grid[-1]/dt) + 3, 3)
    t_local = (np.arange(n_candidate) + 1)*dt
    t_local = t_local[t_local <= t_grid[-1]]

    if len(t_local) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([]))

    v_n = np.interp(t_local, t_grid, v_n_grid)
    v = v_n*v0
    m = np.interp(t_local, t_grid, m_grid)
    h_real = np.interp(t_local, t_grid, h_real_grid)
    length = length0 + (h0 - h_real)/sin_slope

    # Rate terms at the PREVIOUS tick, tau/current-state multipliers at the CURRENT tick - matching
    # the staggered convention used everywhere else in this file (see _evaluateSegment()'s
    # docstring for the full derivation/validation of why).
    t_prev_local = np.maximum(t_local - dt, 0.0)
    v_n_prev = np.interp(t_prev_local, t_grid, v_n_grid)
    m_prev = np.interp(t_prev_local, t_grid, m_grid)
    rho_atm_prev = np.interp(t_prev_local, t_grid, rho_grid)
    v_prev = v_n_prev*v0

    mass_loss_rate = _massLossRate(K, sigma, m_prev, rho_atm_prev, v_prev)
    deceleration_rate = _decelerationRate(K, m_prev, rho_atm_prev, v_prev)

    tau = np.array([luminousEfficiency(const.lum_eff_type, const.lum_eff, float(v_i), float(m_i))
        for v_i, m_i in zip(v, m)])
    beta_ion = np.array([ionizationEfficiency(float(v_i)) for v_i in v])

    lum = -tau*(mass_loss_rate*v**2/2.0 + m*v*deceleration_rate)
    q = -beta_ion*mass_loss_rate/(const.mu*v)
    dyn_press = 1.0*rho_atm_prev*v**2

    kill_mask = ((m <= const.m_kill) | (v < const.v_kill) | (h_real < const.h_kill)
        | ((const.len_kill > 0) & (length > const.len_kill)))

    if np.any(kill_mask):
        kill_idx = int(np.argmax(kill_mask))
        sl = slice(0, kill_idx + 1)
    else:
        # v_n_floor was reached before any kill condition fired (rare - v_n_floor is a generous,
        # physically-motivated margin past v_kill) - return the full grid rather than dropping the
        # tail silently.
        sl = slice(0, len(t_local))

    return (t0 + t_local[sl], v[sl], m[sl], h_real[sl], lum[sl], q[sl], dyn_press[sl], length[sl])


def _batchedBracket(x, xp_2d):
    """ Vectorized analog of the bracket-search half of np.interp(x[i], xp_2d[i], ...) applied
    independently per row i, for M rows sharing a common column count (n_grid) but each with its
    OWN strictly-increasing x-grid (xp_2d) - np.interp()/np.searchsorted() have no such "different
    reference curve per row" form (confirmed: this is why _stepGrainPopulationAnalytic() originally
    fell back to a per-grain Python loop calling np.interp() once per grain - see that function's
    own docstring for the full history).

    NOTE, since this is the other place in this file that "inverts" something for the "100%
    alpha-beta" grain-evolution path: this is NOT a candidate for
    wmpl.Utils.AlphaBeta.getDefaultInverseEiLUT() (see VelocitySpline's own docstring for where
    that LUT IS used). VelocitySpline inverts a genuinely closed-form relation
    (Ei(beta) - Ei(beta*v_n^2) = f(h_equiv)) that a tabulated inverse of Ei() can solve exactly.
    This function inverts TIME as a function of v_n - t(v_n) is only ever available via numerical
    quadrature (_stepGrainPopulationAnalytic()'s own t_grid, a cumulative trapezoidal sum - see its
    docstring), which has no closed form even with Ei^-1 available, LUT-based or otherwise - the
    bracket-search-plus-linear-interpolation done here IS the actual inversion method for that
    different problem, not a stand-in for one. Every caller here guarantees x[i] is within
    [xp_2d[i,0], xp_2d[i,-1]] (no extrapolation/clamping behavior implemented, unlike np.interp()'s
    own edge handling - not needed since callers only ever query in-range points here).

    Split from the actual value gather (_batchedGather(), below) because several y-arrays (e.g.
    v_n/m/h_real, all queried at the same x) share ONE bracket - computing it once and reusing it
    is a real, measured win over computing it redundantly per y-array (profiled directly: a first
    combined version recomputed this per value-array and cost noticeably more wall time for the
    same population than this split form).

    Returns (idx, frac) such that a value at x[i] can be reconstructed from ANY row-i-compatible
    y_2d via _batchedGather(y_2d, idx, frac).
    """

    n_grid = xp_2d.shape[1]
    idx = np.clip(np.sum(xp_2d <= x[:, None], axis=1) - 1, 0, n_grid - 2)
    rows = np.arange(xp_2d.shape[0])
    x_lo = xp_2d[rows, idx]
    x_hi = xp_2d[rows, idx + 1]
    frac = np.where(x_hi > x_lo, (x - x_lo)/np.where(x_hi > x_lo, x_hi - x_lo, 1.0), 0.0)

    return idx, frac


def _batchedGather(fp_2d, idx, frac):
    """ Applies a bracket (idx, frac) already computed by _batchedBracket() to a different row-
    aligned 2D value array - see that function's own docstring for why the two are split. Uses
    plain 2D fancy indexing (arr[rows, idx]), not np.take_along_axis(): measured directly to carry
    less per-call overhead for this shape of gather (single scalar column per row), which matters
    here since this runs once per (tick, value-array) pair across the whole simulation. """

    rows = np.arange(fp_2d.shape[0])
    y_lo = fp_2d[rows, idx]
    y_hi = fp_2d[rows, idx + 1]

    return y_lo + frac*(y_hi - y_lo)


def _stepGrainPopulationAnalytic(const, K, sigma, m0, v0, h0, t0, length0, n_grains, atm_map,
        sin_slope, n_grid=41):
    """ Vectorized population version of _analyticGrainState() - the "full analytic" counterpart
    to _stepGrainPopulationRK4(), matching its OWN flat-array output contract exactly (same
    (global_tick_idx, v, m, h, lum, q, dyn_press, length, n_grains_out, is_death_tick, grain_id)
    shape) so it can be swapped in at any call site with no downstream aggregation code changes -
    see const.grain_evolution_analytic's own docstring for why this exists and what it trades off
    (true continuous physics vs. matching MetSimErosion.py's own coarse-dt RK4 numerics).

    A first version of this function called _analyticGrainState() once per grain in a plain Python
    loop, per direct instruction ("test the scalar approach first, measure, then decide") rather
    than assuming vectorization was needed. Measured directly (not assumed): that version was
    SLOWER than the existing _stepGrainPopulationRK4() mechanism (0.29x on a representative erosion
    scenario, worse on disruption+erosion where many daughters each spawn their own grain
    population) - the exact same lesson Stage 3d already learned once for _stepGrainRK4() (per-
    grain Python-level calls into scipy/Cython-crossing code dominate, not the arithmetic itself).
    A second version vectorized the EXPENSIVE grid construction (every scipy/AtmEquivHeightMap/
    atmDensityPoly call) across the WHOLE population as 2D (N grains x n_grid) arrays, but still
    extracted each grain's own candidate-tick state via a per-GRAIN Python loop calling np.interp()
    - reasoned at the time to be unavoidable, since np.interp() has no "different reference curve
    per row" form. Profiled directly (Stage 9's own follow-up, not assumed): even with the
    expensive part vectorized, that per-grain loop still dominated wall time (a representative
    erosion scenario: 0.630s of 0.940s total inside this function, of which only 0.052s was
    np.interp()'s own C-level code - the rest was Python-level loop/call overhead, repeated once
    per grain, e.g. 16932 iterations for one segment's grain batch alone) - visibly SLOWER than
    _stepGrainPopulationRK4() (0.51-0.62x), which loops over GLOBAL TICKS instead of grains (~600-
    1000 iterations for that same segment, bounded by segment duration/dt, NOT by how many grains
    const.erosion_n_epochs happens to spawn).

    THIS version fixes that by restructuring the loop the same way: iterate over GLOBAL TICKS, not
    grains, mirroring _stepGrainPopulationRK4()'s own architecture exactly (same alive/done
    bookkeeping, same "record this tick for every currently-active grain, then check which ones
    just died" control flow). The one genuine obstacle - np.interp() has no batched-different-
    curve-per-row form - is worked around with _batchedBracket()/_batchedGather() (above): an
    explicit vectorized bracket-search (one O(M x n_grid) comparison per tick, M = active grain
    count at that tick, not N) applied once per DISTINCT query point (t_local and t_prev_local, not
    once per value array - v_n/m/h_real share one bracket, as do v_n_prev/m_prev/rho_prev - a real,
    measured win over recomputing it per array), applied to the SAME already-vectorized 2D grid
    arrays this function already built. This keeps the physics and the grid-construction step
    completely unchanged from the previous version - only the extraction loop's own dimension
    changed (ticks instead of grains), the same lesson now applied one level deeper.

    Result, measured end-to-end via runSimulation(grain_evolution_analytic=True) on the same two
    scenarios used throughout this history: plain erosion improved from 0.62x to 0.83x of
    _stepGrainPopulationRK4()'s own wall time (8.71x vs the reference tool, up from 6.31x);
    disruption+erosion improved from 0.51x to 0.79x of the default mode, and crucially from 0.75x
    to 1.20x of the reference tool - i.e. this mode now beats the unmodified reference on BOTH
    scenarios benchmarked, which it did not before this fix. Still short of matching
    _stepGrainPopulationRK4() itself: re-profiling after this fix (see const.grain_evolution_
    analytic's own docstring) found the largest single remaining cost is _scatterArgmaxGroupby()
    aggregating the resulting (grain, tick) output rows - a total row count measured similar
    between the two mechanisms (673880 analytic vs. 648267 RK4-population on the plain-erosion
    scenario, ~4% more, not the dominant factor), so this is shared aggregation infrastructure
    costing roughly the same regardless of mode, not something specific to this function - not
    chased further here.

    A subtlety in reproducing the OLD per-grain version's exact "is_death_tick" semantics: a grain
    can stop being recorded for two DIFFERENT reasons - (a) a kill condition (m_kill/v_kill/h_kill/
    len_kill) fires on its current tick, or (b) it simply runs out of tabulated grid (its NEXT
    candidate tick would exceed t_hi, the v_n_floor endpoint) with no kill condition ever firing.
    Both must be flagged as that grain's own final row - computed here via a one-tick lookahead
    (`(t_local + dt) > t_hi`) rather than only checking the current tick's own kill flag, which
    would silently miss case (b).

    Unlike _stepGrainPopulationRK4() (which derives each grain's own flat height fresh from its
    accumulated length via the curvature formula, needing no separate height input),
    _analyticGrainState()'s own math needs each grain's own FLAT height at spawn explicitly - h0
    must be supplied by the caller (the same h_real value _spawnGrainsForSegment()'s own
    grain_specs already carry, just not previously threaded through to grain stepping since
    _stepGrainPopulationRK4() never needed it).

    A grain whose entire life is shorter than one const.dt tick (a real, physically legitimate
    outcome under exact continuous-time physics) contributes NOTHING to the output, rather than a
    synthetic single-tick entry - _stepGrainPopulationRK4() cannot produce this (it structurally
    advances and records at least one tick per grain), a genuine, understood behavioral difference
    between the two mechanisms, not a bug.

    Arguments:
        const: [Constants]
        K, sigma: [float] Shared shape-density/ablation coefficients for this population (see
            _stepGrainPopulationRK4()'s own docstring for why every grain in one call shares them).
        m0, v0, h0, t0, length0: [ndarray] Mass (kg), speed (m/s), FLAT real height (m), GLOBAL
            spawn time (s, already snapped to a const.dt multiple), and cumulative path length (m)
            at spawn - one entry per grain.
        n_grains: [ndarray] Number of physical grains each bin/row represents.
        atm_map: [AtmEquivHeightMap] Shared across the whole simulation.
        sin_slope: [float] sin(entry slope) = cos(zenith_angle).

    Keyword arguments:
        n_grid: [int] Forward-evaluation grid points per grain - see _analyticGrainState()'s own
            n_grid docstring (this function reproduces its exact math, just batched across grains).

    Return:
        Same shape as _stepGrainPopulationRK4()'s own return - see that function's docstring.
    """

    empty_f = np.array([], dtype=float)
    empty_i = np.array([], dtype=np.int64)
    empty_b = np.array([], dtype=bool)
    empty_result = (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f,
        empty_f, empty_b, empty_i)

    N = len(np.atleast_1d(m0))
    if N == 0:
        return empty_result

    dt = const.dt
    m0 = np.asarray(m0, dtype=float); v0 = np.asarray(v0, dtype=float)
    h0 = np.asarray(h0, dtype=float); t0 = np.asarray(t0, dtype=float)
    length0 = np.asarray(length0, dtype=float); n_grains = np.asarray(n_grains, dtype=float)

    # --- Vectorized grid construction: ONE call each into every scipy/AtmEquivHeightMap/
    # atmDensityPoly function across all N grains simultaneously, as (N, n_grid) 2D arrays - this
    # is the part that was expensive when repeated N times in a Python loop. Mirrors
    # _analyticGrainState()'s own per-grain math exactly (see that function's docstring for the
    # physical derivation), just broadcast. ---
    v_n_floor_arr = np.maximum(0.01, 0.5*const.v_kill/v0)
    alpha_arr = alphaFromPhysical(K, sin_slope, m0)
    beta_arr = betaFromPhysical(sigma, v0)
    h_equiv_start_arr = atm_map.toEquiv(h0)

    u = np.linspace(0.0, 1.0, n_grid)
    v_n_grid = 1.0 + np.outer(v_n_floor_arr - 1.0, u)  # (N, n_grid), each row DESCENDING 1.0->floor

    h_equiv_grid = hEquivFromVn(v_n_grid, alpha_arr[:, None], beta_arr[:, None],
        h_equiv_start_arr[:, None])
    h_real_grid = atm_map.toReal(h_equiv_grid)
    m_grid = massFromVelocityNormed(v_n_grid, v0[:, None], sigma, m0[:, None])
    rho_grid = _atmDensityPolyVec(h_real_grid, const.dens_co)

    g_grid = m_grid**(1.0/3.0)/(K*rho_grid*v_n_grid**2*v0[:, None])
    dv_n = -np.diff(v_n_grid, axis=1)
    t_grid = np.concatenate([np.zeros((N, 1)),
        np.cumsum(0.5*(g_grid[:, :-1] + g_grid[:, 1:])*dv_n, axis=1)], axis=1)  # (N, n_grid)
    t_hi_arr = t_grid[:, -1]

    # --- Tick-loop extraction: loop over GLOBAL TICKS (bounded by segment duration/dt), mirroring
    # _stepGrainPopulationRK4()'s own architecture - see this function's own docstring for why this
    # replaced an earlier per-grain-loop version. Each iteration reads off every currently-active
    # grain's state via _batchedBracket()/_batchedGather() on the 2D grid rows built above. ---
    k_spawn_tick = np.round(t0/dt).astype(np.int64)
    n_local_ticks_bound = np.maximum((t_hi_arr/dt).astype(np.int64) + 3, 3)
    k_deadline = int(k_spawn_tick.max()) + int(n_local_ticks_bound.max()) + 1

    alive = np.zeros(N, dtype=bool)
    done = np.zeros(N, dtype=bool)

    out_idx, out_v, out_m, out_h, out_len = [], [], [], [], []
    out_vprev, out_mprev, out_rhoprev, out_gid, out_last = [], [], [], [], []

    k = int(k_spawn_tick.min())
    while k < k_deadline:
        k += 1

        newly_spawned = (~alive) & (~done) & (k_spawn_tick == k - 1)
        if np.any(newly_spawned):
            alive[newly_spawned] = True

        if not np.any(alive):
            # STRICT > , not >= - see _stepGrainPopulationRK4()'s own identical check for the full
            # account of why (a real, confirmed off-by-one this function originally copied from
            # there): a grain with k_spawn == k_spawn_tick.max() only becomes eligible in
            # newly_spawned once k reaches k_spawn_tick.max()+1, not k_spawn_tick.max() itself.
            if k > int(k_spawn_tick.max()):
                break
            continue

        idx = np.where(alive)[0]
        t_local = (k - k_spawn_tick[idx])*dt

        # Candidates whose CURRENT tick already exceeds their own t_hi (v_n_floor endpoint) are
        # silently dropped here, not recorded - matches the original per-grain version's own
        # `t_local = t_local[t_local <= t_hi_arr[i]]` trim exactly (that candidate was never
        # generated there either).
        in_range = t_local <= t_hi_arr[idx]
        if np.any(~in_range):
            out_of_range_idx = idx[~in_range]
            alive[out_of_range_idx] = False
            done[out_of_range_idx] = True

        if not np.any(in_range):
            if not np.any(alive):
                # STRICT > , not >= - see the same check above / _stepGrainPopulationRK4()'s own.
                if k > int(k_spawn_tick.max()):
                    break
            continue

        ridx = idx[in_range]
        t_local_r = t_local[in_range]
        t_prev_r = np.maximum(t_local_r - dt, 0.0)

        tg = t_grid[ridx]
        v_n_grid_r = v_n_grid[ridx]; m_grid_r = m_grid[ridx]

        idx_cur, frac_cur = _batchedBracket(t_local_r, tg)
        v_n = _batchedGather(v_n_grid_r, idx_cur, frac_cur)
        m = _batchedGather(m_grid_r, idx_cur, frac_cur)
        h_real = _batchedGather(h_real_grid[ridx], idx_cur, frac_cur)

        idx_prev, frac_prev = _batchedBracket(t_prev_r, tg)
        v_n_prev = _batchedGather(v_n_grid_r, idx_prev, frac_prev)
        m_prev = _batchedGather(m_grid_r, idx_prev, frac_prev)
        rho_atm_prev = _batchedGather(rho_grid[ridx], idx_prev, frac_prev)

        v = v_n*v0[ridx]
        v_prev = v_n_prev*v0[ridx]
        length = length0[ridx] + (h0[ridx] - h_real)/sin_slope

        kill = ((m <= const.m_kill) | (v < const.v_kill) | (h_real < const.h_kill)
            | ((const.len_kill > 0) & (length > const.len_kill)))
        # A grain's LAST recorded row is either where a kill condition fires, OR where the NEXT
        # candidate tick would already be out of range (no kill ever fires) - see this function's
        # own docstring for why both cases must be flagged, not kill alone.
        is_last = kill | ((t_local_r + dt) > t_hi_arr[ridx])

        out_idx.append(np.full(len(ridx), k - 1, dtype=np.int64))
        out_v.append(v); out_m.append(m); out_h.append(h_real); out_len.append(length)
        out_vprev.append(v_prev); out_mprev.append(m_prev); out_rhoprev.append(rho_atm_prev)
        out_gid.append(ridx); out_last.append(is_last)

        if np.any(kill):
            kill_idx = ridx[kill]
            alive[kill_idx] = False
            done[kill_idx] = True

        if not np.any(alive):
            # STRICT > , not >= - see the same check above / _stepGrainPopulationRK4()'s own.
            if k > int(k_spawn_tick.max()):
                break

    if not out_idx:
        return empty_result

    gidx = np.concatenate(out_idx)
    v = np.concatenate(out_v)
    m = np.concatenate(out_m)
    h_real = np.concatenate(out_h)
    length = np.concatenate(out_len)
    v_prev = np.concatenate(out_vprev)
    m_prev = np.concatenate(out_mprev)
    rho_atm_prev = np.concatenate(out_rhoprev)
    gid = np.concatenate(out_gid)
    last = np.concatenate(out_last)
    ng_out = n_grains[gid]

    # --- Rate-term/luminosity/ionization math, ONCE across every grain's every tick - matches
    # _analyticGrainState()'s own formulas exactly (staggered previous-tick rate terms, same
    # lum_eff_type==0 fast path _stepGrainPopulationRK4() already established for the same reason:
    # luminousEfficiency() has no general vectorized port and was profiled there as the dominant
    # remaining cost once the RK4 math itself was vectorized). ---
    mass_loss_rate = _massLossRate(K, sigma, m_prev, rho_atm_prev, v_prev)
    deceleration_rate = _decelerationRate(K, m_prev, rho_atm_prev, v_prev)

    if const.lum_eff_type == 0:
        tau = np.full(len(v), const.lum_eff/100.0)
    else:
        tau = np.array([luminousEfficiency(const.lum_eff_type, const.lum_eff, float(v_i),
            float(m_i)) for v_i, m_i in zip(v, m)])
    beta_ion = _ionizationEfficiencyVec(v)

    lum = -tau*(mass_loss_rate*v**2/2.0 + m*v*deceleration_rate)
    q = -beta_ion*mass_loss_rate/(const.mu*v)
    dyn_press = 1.0*rho_atm_prev*v**2

    return gidx, v, m, h_real, lum, q, dyn_press, length, ng_out, last, gid


def _makeVirtualParentFragment(const, m, v, h_reported, length, sigma, rho=None, gamma=None,
        erosion_mass_index=None, erosion_mass_min=None, erosion_mass_max=None, n_grains=1.0):
    """ Build a Fragment representing the main fragment's (or a disruption "fragment" daughter's)
    state at one grain-spawning epoch, or at the instant of disruption (Stage 3/4) - NOT a real
    simulated fragment, just enough of one for generateFragments()'s unchanged logic
    (spawn_child()'s __dict__ copy, updateShapeDensityCoeff()) to work correctly. Sets every
    attribute those two code paths touch, since Fragment() alone (unlike frag.init(), which is not
    used here) leaves gamma/zenith_angle/etc. unset.

    Arguments:
        const: [Constants]
        m: [float] Mass (kg) at this epoch.
        v: [float] Velocity (m/s) at this epoch.
        h_reported: [float] Reported (curved-Earth + gravity-drop) height (m) at this epoch.
        length: [float] Cumulative path length (m) at this epoch.
        sigma: [float] This segment's OWN ablation coefficient (sigma_own, not sigma_eff) - grains
            inherit this unchanged (MetSimErosion.py's generateFragments() does not modify
            frag_child.sigma in the non-keep_eroding branch - see the implementation plan's
            "sigma inheritance" note).

    Keyword arguments:
        rho: [float or None] This parent's OWN bulk density (kg/m^3), used to derive parent.K via
            updateShapeDensityCoeff() below. Defaults to const.rho (correct for every existing
            caller: plain erosion-grain spawning, where generateFragments()'s own
            "not keep_eroding" branch unconditionally overwrites frag_child.rho/K with
            const.rho_grain regardless of what the parent's own K was, MetSimErosion.py:1381-1383 -
            so the parent's rho/K value is inert for grains). NOT inert for Stage 4 disruption
            "fragment" daughters (keep_eroding=True): those inherit frag_parent's rho/K AS-IS via
            spawn_child()'s dict copy, with no override - so a disrupting fragment past
            erosion_height_change (whose OWN rho is const.erosion_rho_change, not const.rho) must
            pass that here explicitly, or its fragment daughters would silently inherit the wrong
            density/shape-density coefficient.
        gamma: [float or None] This parent's OWN drag coefficient, used together with rho to
            derive parent.K via updateShapeDensityCoeff() below. Defaults to const.gamma - correct
            UNLESS a Stage 5 type "A" fragmentation entry already changed the main fragment's (or a
            daughter's) own gamma before this call (e.g. "A" then a LATER disruption or F/EF/D
            trigger) - every caller spawning something from a fragment's CURRENT state after any
            possible "A" event must pass that fragment's own current gamma explicitly, mirroring
            how the reference always reads the live frag.gamma off the actual Fragment object being
            spawned from (MetSimErosion.py's generateFragments()/spawn_child()), never a fixed
            const.gamma.
        erosion_mass_index, erosion_mass_min, erosion_mass_max: [float or None] Grain-size-
            distribution parameters for whatever generateFragments() call this parent feeds into.
            Default to const's own values - correct for every caller except Stage 5's "M"-type
            fragmentation entries, which can change a main fragment's own copy of these mid-flight
            (MetSimErosion.py:1125-1132) for every SUBSEQUENT erosion epoch/grain spawn.
        n_grains: [float] This parent's OWN multiplicity - how many identical physical bodies this
            single Fragment object represents (Fragment.n_grains). Defaults to 1.0, correct for the
            main fragment (always a unique body) and for Stage 5 F/EF daughters (each spawned as
            its own separate Fragment, never mass-binned - see _applyFragmentationEntry()'s own "F"/
            "EF" handling). NOT 1.0 for a Stage 4 disruption "fragment" daughter whose OWN mass bin
            represents more than one identical fragment (generateFragments()'s power-law mass
            binning routinely produces n_grains>1 for its smaller bins, confirmed directly on this
            file's own complex-scenario test: bins with n_grains up to 9) - every caller spawning
            grains FROM such a daughter's own ongoing erosion must pass that daughter's own
            n_grains here explicitly. This mirrors a real, easy-to-miss mechanism in the reference:
            MetSimErosion.py's spawn_child() does a full __dict__ copy, so a grain-spawning call's
            own frag_child.n_grains STARTS at frag_parent.n_grains (not 1) before generateFragments()
            multiplies in its own per-bin count - i.e. a daughter representing N identical bodies
            correctly produces N times as much grain mass/light as a lone body would. This function
            hardcoded n_grains=1 here until that was found to be missing (see this file's own
            project memory / plan for the diagnosis): every disruption daughter with its own
            n_grains>1 was silently under-producing grain mass/luminosity by exactly that factor,
            confirmed as the dominant cause of a real, previously-unexplained ~2-3x total-luminosity
            undershoot in the ~0.1-0.3s immediately following a disruption event.

    Return:
        [Fragment]
    """

    # Fragment.__new__(Fragment), not Fragment() - Fragment.__init__() sets ~25 default attributes
    # that this function immediately overwrites almost all of anyway (this function's own docstring
    # already claims "sets every attribute [generateFragments()/spawn_child()] touch" - confirmed
    # directly by grepping every .lum/.q/.dyn_press/.h_grav_drop_total read in this file: none of
    # them are ever read off a grain_specs-sourced or disruption-daughter-sourced Fragment, only off
    # frag_main, which is built and populated separately). Skipping __init__() removes ~25 redundant
    # assignments per call - this function runs once per grain-spawning epoch (roughly a thousand
    # times per typical erosion-heavy simulation), so the redundant work was real, not theoretical.
    parent = Fragment.__new__(Fragment)
    parent.const = const
    parent.m = m
    parent.m_init = m
    parent.v = v
    parent.h = h_reported
    parent.length = length
    parent.rho = const.rho if rho is None else rho
    parent.sigma = sigma
    parent.gamma = const.gamma if gamma is None else gamma
    parent.zenith_angle = const.zenith_angle
    parent.vv = -v*math.cos(const.zenith_angle)
    parent.vh = v*math.sin(const.zenith_angle)
    parent.erosion_mass_index = (const.erosion_mass_index if erosion_mass_index is None
        else erosion_mass_index)
    parent.erosion_mass_min = (const.erosion_mass_min if erosion_mass_min is None
        else erosion_mass_min)
    parent.erosion_mass_max = (const.erosion_mass_max if erosion_mass_max is None
        else erosion_mass_max)
    parent.erosion_enabled = False
    parent.disruption_enabled = False
    parent.active = True
    parent.n_grains = n_grains
    parent.main = False
    parent.grain = False
    parent.complex = False
    parent.complex_id = None
    parent.updateShapeDensityCoeff()

    return parent


def _massBinGrains(const, eroded_mass, mass_index, mass_min, mass_max, mass_model="powerlaw"):
    """ Pure mass-binning arithmetic transcribed directly from generateFragments()'s own
    "else" (keep_eroding=False, disruption=False) branch - the same power-law/gamma mass-bin math,
    WITHOUT constructing any Fragment object at all.

    Exists because _spawnGrainsForSegment() (the dominant caller of generateFragments() by call
    volume - one call per (epoch, sub-spawn), n_epochs*n_smear times per eroding segment) only ever
    extracts 5 scalar fields (m, n_grains, sigma, rho, K) from each Fragment generateFragments()
    builds, immediately discarding the Fragment object itself - profiling found the FULL machinery
    behind that (Fragment.spawn_child()'s dict copy, updateShapeDensityCoeff()'s method call, all
    repeated once per non-empty mass bin, ~10-20 times per call) was real, avoidable overhead for a
    result that's fully determined by arithmetic alone, once traced through what generateFragments()
    ACTUALLY does in this specific (keep_eroding=False) branch:
    - rho is unconditionally overwritten to const.rho_grain (never inherited from the parent) -
      _makeVirtualParentFragment()'s own docstring already documents this as making the parent's own
      rho value "inert for grains".
    - K is then rederived via updateShapeDensityCoeff() from that fixed rho and the parent's own
      gamma - which for THIS caller is always const.gamma (no caller passes gamma= to
      _makeVirtualParentFragment() here) - making K = const.gamma*const.shape_factor*
      const.rho_grain**(-2/3) a SIMULATION-WIDE CONSTANT, not something that needs recomputing per
      grain bin at all.
    - sigma is untouched by this branch - stays exactly the parent's own sigma (sigma_own, passed
      through unchanged).
    - n_grains starts at the parent's own n_grains (via spawn_child()'s dict copy) and gets
      multiplied by this bin's own count - i.e. final n_grains = parent_n_grains*n_grains_bin_round,
      pure arithmetic once parent_n_grains is known.

    So the caller can skip Fragment construction ENTIRELY for this branch: this function returns
    just (m_grain, n_grains_bin) per non-empty bin, and the caller derives rho/K/sigma/n_grains
    itself (rho/K being simulation-wide constants it can compute ONCE, not per spawn event).

    Validated bit-for-bit-equivalent (m/n_grains per bin) to generateFragments()'s own output by
    test_mass_bin_grains_matches_generate_fragments().

    Return:
        (m_grains, n_grains_bin): [tuple of ndarray] One entry per non-empty mass bin, in the same
        order generateFragments() would produce them.
    """

    mass_bin_coeff = 10**(-1.0/const.erosion_bins_per_10mass)
    k = int(1 + math.log10(mass_min/mass_max)/math.log10(mass_bin_coeff))

    if mass_model == "gamma":
        mass_bins = np.array([mass_max*(mass_bin_coeff**i) for i in range(k)])
        bin_widths = mass_bins*(1 - mass_bin_coeff)
        log_range = math.log(mass_max/mass_min)

        if mass_index == 1.0:
            m_mean = (mass_max - mass_min)/log_range
        elif mass_index == 2.0:
            m_mean = log_range/(1.0/mass_min - 1.0/mass_max)
        else:
            a = 2 - mass_index
            b = 1 - mass_index
            m_max_a = mass_max**a
            m_min_a = mass_min**a
            m_max_b = mass_max**b
            m_min_b = mass_min**b
            num = (m_max_a - m_min_a)/a
            den = (m_max_b - m_min_b)/b
            m_mean = num/den

        D_mean = (6*m_mean/(math.pi*const.rho_grain))**(1/3)
        gamma_5_3 = 0.90274529295093375313996375552960671484470367431640625
        s = (D_mean*gamma_5_3)**3

        grain_diameter = (6*mass_bins/(math.pi*const.rho_grain))**(1/3)
        n_D = (3*grain_diameter**2/s)*np.exp(-grain_diameter**3/s)
        dD_dm = (1/3)*(6/(math.pi*const.rho_grain))**(1/3)*mass_bins**(-2/3)
        n_m_raw = n_D*np.abs(dD_dm)
        mass_per_bin_raw = n_m_raw*bin_widths*mass_bins
        scaling = eroded_mass/np.sum(mass_per_bin_raw)
        n_m_scaled = n_m_raw*scaling
    else:
        if mass_index == 2:
            n0 = eroded_mass/(mass_max*k)
        else:
            n0 = abs((eroded_mass/mass_max)*(1 - mass_bin_coeff**(2 - mass_index))
                /(1 - mass_bin_coeff**((2 - mass_index)*k)))

    m_out = []
    ng_out = []
    leftover_mass = 0.0

    for i in range(k):

        if mass_model == "gamma":
            m_grain = mass_bins[i]
            n_grains_bin = n_m_scaled[i]*bin_widths[i] + leftover_mass/m_grain
        else:
            m_grain = mass_max*mass_bin_coeff**i
            n_grains_bin = n0*(mass_max/m_grain)**(mass_index - 1) + leftover_mass/m_grain

        n_grains_bin_round = int(math.floor(n_grains_bin))
        leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain

        if n_grains_bin_round > 0:
            m_out.append(m_grain)
            ng_out.append(n_grains_bin_round)

    return np.array(m_out, dtype=float), np.array(ng_out, dtype=float)


def _spawnGrainsForSegment(const, seg, n_epochs, n_smear=1, n_grains=1.0):
    """ Stage 3: spawn grain-bins for one eroding main-fragment segment, at n_epochs COARSE points
    spaced uniformly in CUMULATIVE ERODED MASS (see _cumulativeErodedMass()'s docstring for why
    this is exact/closed-form) - replaces the original's continuous per-dt-tick grain spawning
    (MetSimErosion.py:992-1024: a fresh generateFragments() call every step erosion is active, one
    per dt=0.005s) with discrete batches, each representing the erosion that would have happened
    over many real ticks. Reuses generateFragments() completely unchanged (imported verbatim into
    this file, like the rest of the original's fragmentation bookkeeping - see the implementation
    plan's "Reuse plan").

    Each of the n_epochs mass-uniform epochs is further split into n_smear sub-spawns, placed
    UNIFORMLY IN TIME (not mass) within that epoch's own [t_epoch_start, t_epoch_end) span, each
    carrying epoch_mass_budget/n_smear. This is NOT equivalent to simply using n_epochs*n_smear
    mass-uniform epochs throughout: it specifically targets a real, measured failure mode where a
    single mass-uniform spawn point represents a batch of grains as being born at one instant, even
    though a whole epoch's worth of erosion actually happens continuously across that epoch's real
    time span. When individual grain lifetimes are much shorter than an epoch's own duration (found
    directly: a grain spawned late in a long, heavily-eroding segment can decelerate from full
    speed to below v_kill in ~0.02s, while that epoch spanned ~0.06s), treating the whole epoch as
    one delta-function spawn creates an artificial luminosity SPIKE at that instant rather than the
    smooth curve many staggered real spawns would produce - confirmed directly: peak-luminosity
    error dropping only 1780%->1443% when n_epochs alone was quadrupled (50->200) showed this isn't
    simple discretization noise that averages out with more coarse-grained mass-uniform points; it
    needed FINER TIME resolution specifically within each epoch's own (possibly very short, late in
    a segment) duration, which n_smear provides directly regardless of how short that epoch's own
    span is (unlike adding more globally mass-uniform epochs, which does not by itself guarantee
    tighter TIME spacing within any particular epoch).

    Arguments:
        const: [Constants]
        seg: [dict] One (fully resolved: t_start AND t_end both set) segment dict from
            _buildMainFragmentSegments() or _buildMainFragmentSegmentsWithFragmentation() (or a
            daughter's own _buildDaughterFragmentSegments() - erosion_mass_index/min/max fall back
            to const's own values there via seg.get(), correct since only "M"-type fragmentation
            entries - applied to the MAIN fragment only - ever change these mid-flight).
        n_epochs: [int] Number of COARSE epochs (mass-uniform) to split this segment's total eroded
            mass into.
        n_smear: [int] Number of sub-spawns per epoch (time-uniform within that epoch's own span).
            1 reproduces the original delta-function-per-epoch behavior exactly.
        n_grains: [float] This SEGMENT's OWN parent multiplicity - 1.0 for the main fragment
            (always a unique body) or a Stage 5 F/EF daughter (never mass-binned), but the actual
            multiplicity for a Stage 4 disruption "fragment" daughter whose own mass bin represents
            more than one identical body - see _makeVirtualParentFragment()'s own docstring for the
            full mechanism and why this must be threaded through explicitly rather than left at the
            default for such a daughter's own eroding segments.

    Return:
        grain_specs: [list of dict] One entry per (epoch, sub-spawn, mass-bin) grain-Fragment, each
            with keys: m (kg, per-grain mass), n_grains (int), sigma (s^2/m^2, inherited),
            rho (kg/m^3), K, v (m/s), h_real (m, FLAT height - for building this grain's own
            AnalyticTrajectory), length (m, cumulative), t (s, GLOBAL spawn time).
    """

    if seg["erosion_coeff"] <= 0:
        return []

    traj = seg["traj"]
    v_start = seg["v_start"]
    sigma_eff = seg["sigma_eff"]
    sigma_own = seg["sigma_own"]
    erosion_coeff = seg["erosion_coeff"]
    m_start = seg["m_start"]
    length_start = seg["length_start"]
    seg_erosion_mass_index = seg.get("erosion_mass_index", const.erosion_mass_index)
    seg_erosion_mass_min = seg.get("erosion_mass_min", const.erosion_mass_min)
    seg_erosion_mass_max = seg.get("erosion_mass_max", const.erosion_mass_max)

    v_n_end = float(traj.velocityNormedAt(seg["t_end"]))
    total_eroded_mass = float(_cumulativeErodedMass(v_n_end, v_start, sigma_eff, erosion_coeff,
        m_start))

    if total_eroded_mass <= 0:
        return []

    epoch_mass_budget = total_eroded_mass/n_epochs

    # Epoch BOUNDARIES (n_epochs+1 of them), not midpoints - needed to know each epoch's own real
    # time span for time-uniform sub-spawn placement below.
    target_masses_boundaries = np.arange(n_epochs + 1)*epoch_mass_budget
    # Clip the exact endpoint (target=total_eroded_mass) fractionally inward - the inversion
    # formula's log() is undefined exactly at the segment's own v_n_end (ratio=1.0 there).
    target_masses_boundaries[-1] = min(target_masses_boundaries[-1], total_eroded_mass*(1.0 - 1e-12))

    v_n_boundaries = _vnAtCumulativeErodedMass(target_masses_boundaries, v_start, sigma_eff,
        erosion_coeff, m_start)
    # t as a function of v_n, via this segment's own already-built (Chebyshev-dense) grid - v_n_grid
    # is descending (paired with ascending t_grid, matching AnalyticTrajectory's internal sort), so
    # reverse both for np.interp's ascending-x requirement.
    t_boundaries = np.interp(v_n_boundaries, traj._v_n_grid[::-1], traj._t_grid[::-1])

    sub_mass_budget = epoch_mass_budget/n_smear

    # Every (epoch, sub-spawn) shares the IDENTICAL sub_mass_budget/mass_index/mass_min/mass_max/
    # mass_model - none of those vary across this function's own double loop below - so the mass-
    # binning arithmetic itself only needs to run ONCE for this whole segment, not once per (epoch,
    # sub-spawn) as the original per-spawn generateFragments() call implied. rho/K are likewise
    # simulation-wide constants for plain erosion grains (see _massBinGrains()'s own docstring for
    # why - rho is always const.rho_grain, K is always derived from const.gamma, never from
    # anything segment- or spawn-specific) - computed once here too, not per bin.
    m_bins, ng_bins_raw = _massBinGrains(const, sub_mass_budget, seg_erosion_mass_index,
        seg_erosion_mass_min, seg_erosion_mass_max, mass_model=const.erosion_grain_distribution)
    ng_bins = ng_bins_raw*n_grains
    rho_grain = const.rho_grain
    K_grain = const.gamma*const.shape_factor*const.rho_grain**(-2/3.0)

    grain_specs = []

    for k in range(n_epochs):

        t_epoch_start, t_epoch_end = t_boundaries[k], t_boundaries[k + 1]
        j = np.arange(n_smear)
        t_subs = t_epoch_start + (j + 0.5)/n_smear*(t_epoch_end - t_epoch_start)

        # Snap to the nearest GLOBAL dt-tick: grains are RK4-stepped (_stepGrainRK4()) starting
        # from this spawn time, advancing by exactly dt each step - snapping here means those
        # steps land exactly on the simulation's own output tick grid (dt, 2*dt, 3*dt, ...), with
        # no interpolation needed when aggregating. This is also more physically apt than it might
        # look: in the original, a grain can only ever be born at one of these exact ticks anyway
        # (generateFragments() is only ever called from within ablateAll(), itself only ever
        # called at dt, 2*dt, ...), so snapping an epoch's own placement to the nearest real tick
        # is a faithful refinement, not an approximation on top of an approximation.
        t_subs = np.round(t_subs/const.dt)*const.dt

        # v_n at each TIME-uniform sub-point comes directly from the trajectory's own query
        # interface (no new inversion needed - unlike the epoch boundaries above, which had to go
        # the other way, mass -> v_n -> t). Re-queried at the SNAPPED time, not the original -
        # the segment's own closed-form trajectory remains exact to query at any point. Only v/
        # h_flat/length/t (this sub-spawn's own kinematic state) are actually used below - mass and
        # reported height were computed here in an earlier version purely to build a virtual parent
        # Fragment for generateFragments(), a step _massBinGrains() above has already made
        # unnecessary (see its own docstring - the mass-binning arithmetic never actually depended
        # on the parent's own m/h/v/length at all for this keep_eroding=False branch).
        # Single combined (v_n, h_real) spline query, reused for v/h_flat/length below - separately
        # calling velocityNormedAt()/heightRealAt()/lengthAt() (which itself calls heightRealAt()
        # AGAIN internally) used to query the same underlying spline up to 3 times for the same
        # t_subs - same redundancy, same fix, as _evaluateSegment()'s own (see that function's own
        # comment for the full reasoning).
        t_subs_clipped = np.clip(t_subs, traj.t_start, traj.t_hi)
        traj.n_queries += len(np.atleast_1d(t_subs))
        combined_subs = traj._time_to_combined(t_subs_clipped)
        v_n_subs = combined_subs[..., 0]
        h_flat_subs = combined_subs[..., 1]

        v_subs = v_n_subs*v_start
        length_subs = length_start + (traj.h_real_start - h_flat_subs)/traj.sin_slope

        for i in range(n_smear):

            v_i = float(v_subs[i])
            h_real_i = float(h_flat_subs[i])
            length_i = float(length_subs[i])
            t_i = float(t_subs[i])

            for mb, ngb in zip(m_bins, ng_bins):
                grain_specs.append({
                    "m": float(mb), "n_grains": float(ngb), "sigma": sigma_own, "rho": rho_grain,
                    "K": K_grain, "v": v_i, "h_real": h_real_i, "length": length_i, "t": t_i,
                })

    return grain_specs


def _scatterArgmaxGroupby(idx, key, n_ticks):
    """ Vectorized "groupby argmax": for each integer bucket in [0, n_ticks), find the position
    (into idx/key) of the candidate with the largest key value among all candidates sharing that
    bucket - the primitive _runSimulationErosion() needs to compute brightest_*
    (argmax over per-tick luminosity, across main fragment + every active grain) and
    leading_frag_* (argmax over per-tick length) without a Python loop over ticks or fragments.

    Implemented as two O(N) passes (no sort) rather than a Python groupby OR a lexsort: since idx
    is already a bounded integer bucket id (not an arbitrary sort key), the bucket-wise max can be
    found directly via np.maximum.at (one scatter-reduce pass), then every candidate whose own key
    equals its own bucket's max is a valid winner (a second, embarrassingly parallel pass, no
    sorting of the - possibly huge - candidate table at all).

    Superseded an earlier np.lexsort((key, idx)) implementation (sort by (idx, key) ascending, take
    the last entry of each contiguous idx-run) - correct, but paying an O(N log N) full sort of
    EVERY candidate (main + every daughter + every grain batch, at every tick each is alive) to
    answer a question that only needs O(N): profiling this project's own benchmark scenarios found
    this single function was the largest cost in the WHOLE engine (26% of total wall-clock on the
    complex scenario, 38% on a plain erosion scenario - ahead even of the grain-population RK4
    stepping loop), confirmed via cProfile before this rewrite, not assumed. Real, measured
    speedup: see this file's own profiling notes / the implementation plan's own write-up for
    before/after numbers.

    On an exact float tie for the bucket max (two or more candidates sharing the identical winning
    key value), this may pick a DIFFERENT winner than the old lexsort-based version's own "last in
    original candidate-list order" convention - both are valid per this function's own contract
    (some max-key candidate, not a specific tie-break rule) and test_scatter_argmax_groupby_correctness()
    checks the winning KEY VALUE achieved, not the row index, for exactly this reason. Ties are
    vanishingly rare in practice (independent physical fragments essentially never share a
    bit-exact luminosity/length), so this has no observable effect on any of this file's own
    accuracy-tolerance tests.

    Arguments:
        idx: [ndarray of int] Bucket (GLOBAL tick, 0-indexed) for each candidate.
        key: [ndarray] Value to maximize within each bucket (e.g. n_grains-weighted luminosity, or
            length) - same length as idx.
        n_ticks: [int] Number of buckets.

    Return:
        winner_pos: [ndarray of int, size n_ticks] For each bucket, the index INTO idx/key of the
            winning (max-key) candidate. -1 for buckets with no candidates at all - per this
            function's caller, this can only happen on the LAST tick of the whole simulation (see
            _runSimulationErosion()'s docstring for why, and for the documented simplification of
            reporting 0.0 rather than the reference's None in that one edge case).
    """

    winner_pos = np.full(n_ticks, -1, dtype=np.int64)

    if len(idx) == 0:
        return winner_pos

    best_key = np.full(n_ticks, -np.inf)
    np.maximum.at(best_key, idx, key)

    is_winner = key >= best_key[idx]
    winner_pos[idx[is_winner]] = np.flatnonzero(is_winner)
    return winner_pos


def _evaluateFragmentSegments(const, segments):
    """ Evaluate a fragment's FULL segment chain (the main fragment's own, from
    _buildMainFragmentSegments(), or a Stage 4 disruption "fragment" daughter's, from
    _buildDaughterFragmentSegments()) into dense per-GLOBAL-tick arrays spanning its own
    [first segment's t_start, last segment's t_end] - shared by _runSimulationErosion() for both,
    since once a fully-resolved segment chain exists, evaluating it is identical either way.

    Arguments:
        const: [Constants]
        segments: [list of dict] A fully-resolved segment chain (every t_end set - see
            _buildMainFragmentSegments()'s own guarantee).

    Return:
        (tick_idx, v, m, length, h, lum, q, dyn_press, tau, n_queries_total, tail_grain_specs):
        tick_idx is the 0-INDEXED GLOBAL row position for each array entry (GLOBAL time =
        (tick_idx+1)*const.dt, matching every other tick-indexed array in this file - e.g.
        _stepGrainPopulationRK4()'s own global_tick_idx); the rest (except the last two) are
        ndarrays of the same length, one entry per GLOBAL tick this fragment is active for (its
        own first tick after t_start through its own true death tick, inclusive - which can extend
        PAST the chain's own last segment's original t_end if that segment carries an "rk4_tail"
        - see _resolveSegmentChainDeathRegime()). n_queries_total sums traj.n_queries across every
        segment (Stage 2d instrumentation). tail_grain_specs is a flat list of grain_specs-shaped
        dicts from the tail's own per-tick erosion-channel spawning (empty unless the last segment
        carries an "rk4_tail") - the caller must fold these into the same accumulator that feeds
        _batchAndStepGrainSpecs().
    """

    # k_start/k_end must pick out exactly the same tick RANGE the per-segment mask below will
    # assign values into (mask: t_ticks > seg["t_start"] - EPS, t_ticks <= seg["t_end"] + EPS) -
    # a bare round(t_start/dt)+1 / round(t_end/dt) does NOT do this: rounding first discards which
    # side of the tick t_start/t_end actually falls on, so whenever the fractional part of
    # t_start/dt (or t_end/dt) is >= 0.5, round() silently rounds to the SAME tick floor()+1
    # would already give, adding a spurious extra +1 that SKIPS the true first/last tick entirely.
    # Confirmed as a real bug, not theoretical: for a Stage 5 "EF" daughter spawned at
    # t_start=1.94790... (t_start/dt=389.58), this skipped tick 390 (t=1.950) - the SAME tick the
    # main fragment's own chain already reflects the fragmentation event on - producing a spurious
    # one-tick gap in mass_total_active (daughters briefly uncounted) right at every F/EF
    # fragmentation event. floor()+eps below reproduces the mask's own boundary semantics exactly
    # (verified against a brute-force search across 200000 random boundary times, zero mismatches -
    # though that sweep used continuous random values and never happened to hit t_start=0.0 EXACTLY,
    # which is the main fragment's own t_start on every single run - see the max(..., 1) clamp
    # below for why that exact-zero case needed its own fix, found only after this shipped and
    # caused a real, visible bug: main_mass_col[tick_idx_main] = m_main silently wrapped around to
    # main_mass_col[-1] (numpy fancy-indexing treats -1 as "last element", not "invalid"), because
    # for t_start=0.0 this formula's own floor((0 - EPS)/dt) is -1, not 0 - k=0 is not a valid
    # GLOBAL TICK NUMBER (tick_idx = k-1 must be >= 0; k represents a 1-indexed tick, matching every
    # other tick-indexed convention in this file, e.g. t_ticks = (arange(n_ticks)+1)*dt elsewhere).
    EPS = 1e-9
    dt = const.dt
    k_start = max(1, int(math.floor((segments[0]["t_start"] - EPS)/dt)) + 1)

    # An "rk4_tail" attached to the last segment (_resolveSegmentChainDeathRegime()) can extend
    # this fragment's own true death PAST that segment's own (deliberately pulled-back) t_end - so
    # the tail's own last recorded GLOBAL time, not t_end, determines k_end whenever present.
    tail = segments[-1].get("rk4_tail")
    if tail is not None and len(tail["t"]) > 0:
        k_end = int(round(tail["t"][-1]/dt))
    else:
        k_end = int(math.floor((segments[-1]["t_end"] + EPS)/dt))

    if k_end < k_start:
        empty = np.array([])
        return (np.array([], dtype=np.int64), empty, empty, empty, empty, empty, empty, empty,
            empty, 0, [])

    t_ticks = np.arange(k_start, k_end + 1)*dt
    tick_idx = np.arange(k_start, k_end + 1) - 1

    n = len(t_ticks)
    v = np.zeros(n); m = np.zeros(n); length = np.zeros(n); h = np.zeros(n)
    lum = np.zeros(n); q = np.zeros(n); dp = np.zeros(n); tau = np.zeros(n)

    n_queries_total = 0
    assigned = np.zeros(n, dtype=bool)
    for seg in segments:
        mask = (t_ticks > seg["t_start"] - EPS) & (t_ticks <= seg["t_end"] + EPS) & (~assigned)
        if not np.any(mask):
            continue
        tq = t_ticks[mask]
        v_s, m_s, len_s, h_s, lum_s, q_s, dp_s, tau_s = _evaluateSegment(seg["traj"],
            seg["atm_height_fn"], tq, seg["K"], seg["sigma_own"], seg["v_start"],
            seg["length_start"], const, grav_drop_start=seg.get("grav_drop_start", 0.0))
        v[mask] = v_s; m[mask] = m_s; length[mask] = len_s; h[mask] = h_s
        lum[mask] = lum_s; q[mask] = q_s; dp[mask] = dp_s; tau[mask] = tau_s
        assigned |= mask
        n_queries_total += seg["traj"].n_queries

    tail_grain_specs = []
    if tail is not None:
        tail_mask = (t_ticks > segments[-1]["t_end"] + EPS) & (~assigned)
        v[tail_mask] = tail["v"]; m[tail_mask] = tail["m"]; length[tail_mask] = tail["length"]
        h[tail_mask] = tail["h"]; lum[tail_mask] = tail["lum"]; q[tail_mask] = tail["q"]
        dp[tail_mask] = tail["dyn_press"]; tau[tail_mask] = tail["tau"]
        assigned |= tail_mask
        tail_grain_specs = tail.get("grain_specs", [])

    return tick_idx, v, m, length, h, lum, q, dp, tau, n_queries_total, tail_grain_specs


def _erodingSegsAndMasses(segments):
    """ Filter a chain down to its own eroding segments (erosion_coeff > 0) and compute each one's
    own total eroded mass via the closed-form _cumulativeErodedMass() - the shared first half of
    what both _totalErodedMassAcrossChains() (the global pre-pass) and
    _spawnGrainSpecsForAllErodingSegments() (the actual per-chain spawn) need, factored out so the two
    can never silently compute this differently.

    Arguments:
        segments: [list of dict] A fully-resolved segment chain.

    Return:
        (eroding_segs, seg_masses): PARALLEL lists - eroding_segs is the erosion_coeff>0 subset of
        segments (possibly empty), seg_masses is each one's own total eroded mass (kg, >= 0.0).
    """

    eroding_segs = [seg for seg in segments if seg["erosion_coeff"] > 0]
    seg_masses = []
    for seg in eroding_segs:
        v_n_end = float(seg["traj"].velocityNormedAt(seg["t_end"]))
        seg_masses.append(max(0.0, float(_cumulativeErodedMass(v_n_end, seg["v_start"],
            seg["sigma_eff"], seg["erosion_coeff"], seg["m_start"]))))
    return eroding_segs, seg_masses


def _totalErodedMassAcrossChains(chains):
    """ Sum _erodingSegsAndMasses()'s own per-segment eroded mass across EVERY eroding segment of
    EVERY chain in a simulation (the main fragment's own chain, plus every daughter's own) - the
    global pre-pass _spawnGrainSpecsForAllErodingSegments() needs to allocate epoch budget across
    CHAINS, not just within one chain's own segments (see that function's own docstring for why).

    Arguments:
        chains: [list of list of dict] One fully-resolved segment chain per fragment (main's own,
            then one per daughter_fragments entry).

    Return:
        [float] Total eroded mass (kg) across every eroding segment of every chain, >= 0.0.
    """

    total = 0.0
    for segments in chains:
        _, seg_masses = _erodingSegsAndMasses(segments)
        total += sum(seg_masses)
    return total


def _spawnGrainSpecsForAllErodingSegments(const, segments, global_total_mass=None, n_grains=1.0):
    """ Call _spawnGrainsForSegment() for every eroding segment in a chain and return the raw,
    NOT-YET-STEPPED grain_specs (concatenated across every eroding segment in this one chain) - the
    per-fragment grain-spawning step _runSimulationErosion() needs once for the main fragment and
    once per Stage 4 disruption "fragment" daughter, factored out since both need exactly this and
    nothing else.

    Deliberately does NOT step the resulting population (that used to happen here, one
    _stepGrainPopulationRK4()/_stepGrainPopulationAnalytic() call per eroding segment - see the
    "one further step" paragraph below for why that changed): the caller is expected to concatenate
    this chain's own specs with every OTHER chain's own specs (plus any one-shot specs - disruption
    leftover, Stage 5 "D" dust, RK4-tail grain-spawning) into ONE flat list and hand the WHOLE thing
    to _batchAndStepGrainSpecs() once, which groups by (K, sigma) and steps each DISTINCT group as
    one population - collapsing what used to be one _stepGrainPopulationRK4() call per eroding
    segment into one call per distinct (K, sigma) actually present across the WHOLE simulation.

    Each eroding segment's own epoch budget is allocated PROPORTIONALLY to its own share of a TOTAL
    eroded mass, rather than every segment independently getting the full const.erosion_n_epochs -
    a real, measured Stage 7 finding, not a hypothetical: const.erosion_n_epochs=1000 was validated
    (Stage 3d) against a chain with 1-2 eroding segments, but Stage 5 fragmentation can split a
    chain into many MORE (often much shorter) eroding segments interleaved between trigger heights
    - a segment lasting 200ms between two closely-spaced triggers got the exact same 1000-epoch
    budget as a multi-second erosion phase, even though it represents a tiny fraction of the total
    eroded mass. Measured directly on a representative M+F+D scenario (cProfile, then a targeted
    candidate-array-size probe): 5-6 eroding segments instead of the validated 2 inflated the total
    grain population enough to produce 27 MILLION candidate rows into _scatterArgmaxGroupby() for
    only ~1000 output ticks - 15.5 of a 24s total runtime in that ONE (already Stage-6-validated-
    correct, not buggy) function alone, making this scenario 3x SLOWER than the RK4 reference it
    exists to replace.

    global_total_mass: [float or None] The denominator for the proportional-allocation formula
    below. Stage 7 only ever passed None here (falls back to THIS CALL's own chain-local total,
    i.e. summed across just this one chain's own eroding segments) - correct for a chain with
    MULTIPLE eroding segments of its own, but blind to every OTHER chain in the same simulation: a
    scenario combining disruption (spawning N daughters) with Stage 5 fragmentation (spawning more)
    called this function ONCE PER CHAIN, and each call's own chain-local total_mass only ever "saw"
    that one chain's own share - so N chains, each with a SINGLE eroding segment, each independently
    got the FULL const.erosion_n_epochs, with no cross-chain sharing at all. Measured directly on
    this file's own "complex scenario" (disruption + 2-phase erosion + EF/A/D fragmentation, 13
    chains total): 19 eroding segments, most with a single segment of their own (so Stage 7's own
    WITHIN-chain proportional allocation was a no-op for them), produced 2,755,342 total candidate
    rows for only 356 output ticks - _scatterArgmaxGroupby() alone cost 1.05s of a 4.19s total
    (called twice on that array), making this scenario 2-2.6x SLOWER than the RK4 reference despite
    the main-fragment-only cases this engine was originally benchmarked on being 3-10x faster.
    _runSimulationErosion() now computes _totalErodedMassAcrossChains() ONCE across every chain in
    the simulation and passes it here explicitly for every call (main's own and every daughter's),
    extending Stage 7's own "a segment representing a larger share of the total gets more epochs"
    principle from "within one chain" to "within the whole simulation" - a small daughter
    contributing a tiny fraction of the SIMULATION's total eroded mass now gets proportionally few
    epochs, same as a short segment within one chain already did. Kept optional (default None,
    chain-local fallback) rather than required, so existing direct callers of this function (e.g.
    test_run_simulation_erosion_mass_total_active_weighted(), which has no daughters to share a
    budget with) keep their own already-validated behavior unchanged.

    One further step, found by checking WHERE this file's own wall-clock cost actually went after
    the global_total_mass fix above (not assumed done once epochs were fixed): even with epochs
    correctly shared, this file's own "complex scenario" still called _stepGrainPopulationRK4() 19
    separate times (once per eroding segment) - checked directly whether those 19 calls' own (K,
    sigma) pairs actually differed enough to justify 19 separate populations: only 2 DISTINCT (K,
    sigma) pairs existed across all 19 (14 segments shared one, 5 shared the other - all these
    daughters' grains share the SAME K, since it only depends on const.rho_grain/gamma, and split
    into only 2 sigma groups matching before/after this scenario's own "A" event). Splitting this
    function into "spawn raw specs" (here) + a single shared _batchAndStepGrainSpecs() call in the
    caller (mirroring the ALREADY-EXISTING pattern that function already uses for one-shot D-dust/
    disruption-leftover grains) collapses those 19 calls down to 2, for the exact same total grain
    population and physics - purely a call-count reduction, matching the SAME "many small calls have
    real per-call overhead" lesson already learned twice in this file (Stage 3d for
    _stepGrainRK4()->_stepGrainPopulationRK4(), Stage 9 for the analytic grain mode's own
    per-grain->per-tick loop restructuring), just at the "many small populations" level this time
    instead of "many small grains" or "many small ticks".

    Arguments:
        const: [Constants]
        segments: [list of dict] A fully-resolved segment chain.

    Keyword arguments:
        n_grains: [float] This CHAIN's own parent multiplicity - see _makeVirtualParentFragment()'s
            own docstring for the full mechanism. Defaults to 1.0 (correct for the main fragment
            and Stage 5 F/EF daughters); a caller spawning from a Stage 4 disruption "fragment"
            daughter whose own mass bin represents more than one identical body must pass that
            daughter's own n_grains here explicitly, or the resulting grain population will be
            silently under-massed/under-luminous by exactly that factor.

    Return:
        grain_specs: [list of dict] Flat list of grain_specs-shaped dicts (m, n_grains, sigma, rho,
            K, v, h_real, length, t - the same shape _spawnGrainsForSegment()/
            _batchAndStepGrainSpecs() already use), concatenated across every eroding segment in
            this chain. Empty if this chain has no eroding segments.
    """

    MIN_EPOCHS = 20

    eroding_segs, seg_masses = _erodingSegsAndMasses(segments)
    if not eroding_segs:
        return []

    total_mass = sum(seg_masses) if global_total_mass is None else global_total_mass

    all_specs = []
    for seg, seg_mass in zip(eroding_segs, seg_masses):
        if total_mass > 0:
            n_epochs = max(MIN_EPOCHS, min(const.erosion_n_epochs,
                int(round(const.erosion_n_epochs*seg_mass/total_mass))))
        else:
            n_epochs = const.erosion_n_epochs

        all_specs.extend(_spawnGrainsForSegment(const, seg, n_epochs, n_smear=const.erosion_n_smear,
            n_grains=n_grains))

    return all_specs


def _batchAndStepGrainSpecs(const, grain_specs, atm_map, sin_slope):
    """ Group a flat list of one-shot (non-eroding, keep_eroding=False) grain_specs dicts - the
    shape a disruption's leftover-mass spawn or a Stage 5 "D" (dust release) trigger produce (keys:
    m, n_grains, sigma, rho, K, v, h_real, length, t) - by their exact (K, sigma) pair, then step
    each group through _stepGrainPopulationRK4() (or _stepGrainPopulationAnalytic(), if
    const.grain_evolution_analytic is True) as one population. atm_map/sin_slope are only actually
    used in analytic mode, but always required from the caller for a uniform signature.

    Grouping (rather than assuming the whole list shares one (K, sigma), as
    _runSimulationErosion()'s own disruption-leftover-grain handling could get away with before
    Stage 5) matters because grain_specs can now be a concatenation of grains from DIFFERENT
    generateFragments() calls with genuinely different (K, sigma) - e.g. two "D" triggers separated
    by an "A" trigger that changed sigma/gamma. _stepGrainPopulationRK4() assumes one shared
    (K, sigma) for its whole population (see its own docstring) - within any ONE
    generateFragments() call every grain bin shares an EXACTLY equal (K, sigma) float (both copied
    unchanged from the same parent Fragment via spawn_child(), MetSimErosion.py:373-384), so
    grouping by exact float equality (no tolerance) is safe and correctly reproduces "one population
    per generateFragments() call" without needing to track call boundaries explicitly.

    Arguments:
        const: [Constants]
        grain_specs: [list of dict] Each dict may carry an optional "complex" bool key (whether the
            fragment this grain was spawned from itself had frag.complex=True, i.e. a Stage 5 "D"
            dust release or a complex EF daughter's own erosion - see _runSimulationErosion()'s own
            lum_eroded comment for why this matters) - defaults to False if absent, matching every
            grain_specs producer that predates this distinction.

    Return:
        (grain_batches, batch_meta, batch_is_complex): grain_batches is [list] one
            _stepGrainPopulationRK4() return tuple per distinct (K, sigma) group found in
            grain_specs (possibly empty if grain_specs is empty); batch_meta is a PARALLEL list of
            the (K, sigma) pair each batch was built with - needed by Stage 5 type "A" retroactive
            re-splitting (_resplitAllGrainBatchesForEvents()) to know what a batch's own CURRENT
            (K, sigma) is. batch_is_complex is a PARALLEL list of boolean ndarrays, one per batch,
            aligned with that batch's own LOCAL grain_id (0..len(specs)-1) - a batch can mix
            complex and non-complex grains if they happen to share the same (K, sigma), so this is
            tracked per-grain, not per-batch.
    """

    if not grain_specs:
        return [], [], []

    groups = {}
    for g in grain_specs:
        groups.setdefault((g["K"], g["sigma"]), []).append(g)

    grain_batches = []
    batch_meta = []
    batch_is_complex = []
    for (K_g, sigma_g), specs in groups.items():
        m0_arr = np.array([g["m"] for g in specs])
        v0_arr = np.array([g["v"] for g in specs])
        t0_arr = np.array([g["t"] for g in specs])
        length0_arr = np.array([g["length"] for g in specs])
        n_grains_arr = np.array([g["n_grains"] for g in specs], dtype=float)
        if const.grain_evolution_analytic:
            h0_arr = np.array([g["h_real"] for g in specs])
            grain_batches.append(_stepGrainPopulationAnalytic(const, K_g, sigma_g, m0_arr, v0_arr,
                h0_arr, t0_arr, length0_arr, n_grains_arr, atm_map, sin_slope))
        else:
            grain_batches.append(_stepGrainPopulationRK4(const, K_g, sigma_g, m0_arr, v0_arr,
                t0_arr, length0_arr, n_grains_arr))
        batch_meta.append((K_g, sigma_g))
        batch_is_complex.append(np.array([bool(g.get("complex", False)) for g in specs]))

    return grain_batches, batch_meta, batch_is_complex


def _resplitDaughterAtTick(const, daughter, t_A, sigma_new, gamma_new, sin_slope, atm_map,
        h_real_floor):
    """ Stage 5 type "A", phase 1: truncate-and-continue ONE daughter_fragments entry's own segment
    chain at a single "A" trigger instant, IN PLACE (mutates daughter["segments"]) - matching
    ablateAll()'s own semantics of overwriting frag.sigma/frag.gamma for every fragment that exists
    at that tick (MetSimErosion.py:1103-1109), applied here to a fragment that is NOT the main one
    (the main fragment's own future is already handled inline while its chain is built - see
    _runSimulationErosion()'s own "phase 1" comment for the full reasoning on why this must run
    BEFORE any grain-spawning).

    Does nothing if the daughter was not yet spawned by t_A, or already dead by (or exactly at) t_A
    - checked via its own segments' [t_start, t_end] bounds - "A" can only ever affect an
    ALREADY-ALIVE fragment's own future, matching _resplitGrainBatchAtTick()'s identical convention
    for grains.

    Reuses the SAME regime-aware builder the daughter was originally constructed with, so its
    post-event future is computed by the exact same logic a freshly-spawned daughter in that state
    would use, not a special-cased approximation:
      - complex_id is None (a Stage 4 disruption "fragment" daughter, non-complex): continues via
        _buildDaughterFragmentSegments() - erosion_coeff keeps getting the normal height-based
        getErosionCoeff() auto-update after this point, exactly as before "A" fired (confirmed "A"
        never touches erosion_coeff, only sigma/gamma).
      - complex_id is not None (a Stage 5 F/EF daughter, complex=True): continues via
        _buildComplexFragmentDaughterSegments() with the SAME fixed erosion_coeff/grain-size
        parameters the daughter already had (complex fragments never get the height-based
        auto-update either before or after "A" - MetSimErosion.py:976's `not frag.complex` gate is
        untouched by "A").
    rho is never touched by "A" (only gamma) - K is re-derived from the daughter's own current rho
    (read off the segment containing t_A) and the new gamma.

    Arguments:
        const: [Constants]
        daughter: [dict, "segments" key mutated in place] One daughter_fragments entry.
        t_A: [float] GLOBAL time (s) of the "A" trigger (a continuous root-find result, same as
            every other fragmentation-entry trigger in this file - see
            _buildMainFragmentSegmentsWithFragmentation()'s own docstring for why this is kept
            consistent with erosion boundaries rather than made tick-exact like disruption).
        sigma_new, gamma_new: [float or None] The event's own sigma/gamma (None means "don't change
            this attribute for anyone" - the daughter keeps its own current value for that one).
        sin_slope, atm_map, h_real_floor: see _buildDaughterFragmentSegments()'s own arguments.
    """

    segments = daughter["segments"]

    if t_A < segments[0]["t_start"] - 1e-9:
        return
    if t_A >= segments[-1]["t_end"] - 1e-9:
        return

    seg_at = None
    keep_segments = []
    for s in segments:
        if s["t_start"] - 1e-9 <= t_A < s["t_end"] - 1e-9:
            seg_at = s
            break
        keep_segments.append(s)

    if seg_at is None:
        return

    v_A, m_A, len_A, h_A, _lum_A, _q_A, _dp_A, _tau_A = _evaluateSegment(seg_at["traj"],
        seg_at["atm_height_fn"], np.array([t_A]), seg_at["K"], seg_at["sigma_own"],
        seg_at["v_start"], seg_at["length_start"], const,
        grav_drop_start=seg_at.get("grav_drop_start", 0.0))
    v_A, m_A, len_A, h_A = float(v_A[0]), float(m_A[0]), float(len_A[0]), float(h_A[0])
    grav_drop_A = seg_at.get("grav_drop_start", 0.0) + float(seg_at["traj"].gravityDropAt(t_A))

    rho_current = seg_at["rho"]
    gamma_current = seg_at["gamma"] if gamma_new is None else gamma_new
    sigma_own_current = seg_at["sigma_own"] if sigma_new is None else sigma_new
    K_new = gamma_current*const.shape_factor*rho_current**(-2/3.0)

    seg_at_truncated = dict(seg_at)
    seg_at_truncated["t_end"] = t_A
    keep_segments.append(seg_at_truncated)

    if daughter["complex_id"] is None:
        continuation = _buildDaughterFragmentSegments(const, K_new, sigma_own_current, m_A, v_A,
            h_A, sin_slope, atm_map, t_start=t_A, length_start=len_A,
            grav_drop_start=grav_drop_A, h_real_floor=h_real_floor, rho=rho_current,
            gamma=gamma_current)
    else:
        continuation = _buildComplexFragmentDaughterSegments(const, K_new, sigma_own_current,
            seg_at["erosion_coeff"], m_A, v_A, h_A, sin_slope, atm_map, t_start=t_A,
            length_start=len_A, grav_drop_start=grav_drop_A, h_real_floor=h_real_floor,
            erosion_mass_index=seg_at.get("erosion_mass_index"),
            erosion_mass_min=seg_at.get("erosion_mass_min"),
            erosion_mass_max=seg_at.get("erosion_mass_max"), rho=rho_current, gamma=gamma_current)

    daughter["segments"] = keep_segments + continuation


def _resplitGrainBatchAtTick(const, batch, tick_idx_A, K_new, sigma_new, atm_map, sin_slope):
    """ Stage 5 type "A", phase 2: re-derive the portion of a grain population batch's own
    trajectory AFTER tick_idx_A for every grain in the batch that was STILL ALIVE at that tick.
    Grains that had ALREADY DIED (own last tick <= tick_idx_A) or were NOT YET SPAWNED (own first
    tick > tick_idx_A) are returned completely UNCHANGED - re-splitting only ever affects the
    mid-life portion of an already-alive grain's own trajectory, matching
    _resplitDaughterAtTick()'s identical convention for daughters.

    A grain's row AT exactly tick_idx_A (if it needs re-splitting) is KEPT as-is - it was already
    computed with the OLD (K, sigma) before the event took effect, matching ablateAll()'s own
    per-tick sequencing (the complex-fragmentation block mutates frag.sigma/frag.gamma AFTER that
    tick's own RK4 update has already been computed and recorded, so only the NEXT tick onward uses
    the new values). The new run is started from t0 = (tick_idx_A+1)*const.dt - the GLOBAL time
    that row represents - so it continues with no gap or double-step; gravity-drop/curvature-offset
    state resets to 0 at this new "spawn" instant, same as any other fresh grain-population call
    (consistent with Stage 3c's own established flat-gravity-drop simplification for grains).

    Dispatches to _stepGrainPopulationRK4() or _stepGrainPopulationAnalytic() for the continuation
    based on const.grain_evolution_analytic, matching whichever mechanism the ORIGINAL batch was
    itself built with (this flag is global for a whole simulation run, never mixed mid-run, so a
    batch's own `h` array is already in the right convention either way - REPORTED height for RK4
    mode, FLAT height for analytic mode - continuing with the SAME mechanism keeps that consistent;
    see _stepGrainPopulationAnalytic()'s own docstring for why it needs FLAT height specifically).

    Arguments:
        const: [Constants]
        batch: [tuple] One grain-population-stepping return tuple (RK4 or analytic shape - both
            identical).
        tick_idx_A: [int] 0-indexed GLOBAL tick the "A" event fires at (GLOBAL time =
            (tick_idx_A+1)*const.dt).
        K_new, sigma_new: [float] This population's own POST-event shape-density/ablation
            coefficients.
        atm_map, sin_slope: [AtmEquivHeightMap, float] Only actually used in analytic mode.

    Return:
        A new grain-population-stepping-shaped tuple, replacing `batch` (or `batch` itself,
        unchanged, if no grain in it needs re-splitting).
    """

    gidx, v, m, h, lum, q, dp, length, ng, last, gid = batch

    if len(gidx) == 0:
        return batch

    n_grain_slots = int(gid.max()) + 1

    first_tick = np.full(n_grain_slots, np.iinfo(np.int64).max, dtype=np.int64)
    last_tick = np.full(n_grain_slots, -1, dtype=np.int64)
    np.minimum.at(first_tick, gid, gidx)
    np.maximum.at(last_tick, gid, gidx)

    existed = first_tick <= tick_idx_A
    needs_resplit = np.zeros(n_grain_slots, dtype=bool)
    needs_resplit[existed] = last_tick[existed] > tick_idx_A

    if not np.any(needs_resplit):
        return batch

    row_needs_resplit = needs_resplit[gid]
    keep_mask = (~row_needs_resplit) | (gidx <= tick_idx_A)

    at_A_row = (gidx == tick_idx_A) & row_needs_resplit
    order = np.argsort(gid[at_A_row])
    resplit_ids_ordered = gid[at_A_row][order]
    v_A = v[at_A_row][order]
    m_A = m[at_A_row][order]
    h_A = h[at_A_row][order]
    length_A = length[at_A_row][order]
    ng_A = ng[at_A_row][order]

    t0_new = np.full(len(resplit_ids_ordered), (tick_idx_A + 1)*const.dt)

    if const.grain_evolution_analytic:
        (new_gidx, new_v, new_m, new_h, new_lum, new_q, new_dp, new_len, new_ng, new_last,
            new_gid_local) = _stepGrainPopulationAnalytic(const, K_new, sigma_new, m_A, v_A, h_A,
                t0_new, length_A, ng_A, atm_map, sin_slope)
    else:
        (new_gidx, new_v, new_m, new_h, new_lum, new_q, new_dp, new_len, new_ng, new_last,
            new_gid_local) = _stepGrainPopulationRK4(const, K_new, sigma_new, m_A, v_A, t0_new,
                length_A, ng_A)
    new_gid_mapped = resplit_ids_ordered[new_gid_local]

    return (np.concatenate([gidx[keep_mask], new_gidx]),
        np.concatenate([v[keep_mask], new_v]),
        np.concatenate([m[keep_mask], new_m]),
        np.concatenate([h[keep_mask], new_h]),
        np.concatenate([lum[keep_mask], new_lum]),
        np.concatenate([q[keep_mask], new_q]),
        np.concatenate([dp[keep_mask], new_dp]),
        np.concatenate([length[keep_mask], new_len]),
        np.concatenate([ng[keep_mask], new_ng]),
        np.concatenate([last[keep_mask], new_last]),
        np.concatenate([gid[keep_mask], new_gid_mapped]))


def _resplitAllGrainBatchesForEvents(const, a_type_events, grain_batches, grain_batch_meta,
        atm_map, sin_slope):
    """ Stage 5 type "A", phase 2 orchestrator: apply _resplitGrainBatchAtTick() to every grain
    batch for every "A" event, in time order (a_type_events is already time-ascending - see
    _buildMainFragmentSegmentsWithFragmentation()'s own trigger loop). A later event correctly
    builds on an earlier one's own effect: grain_batch_meta is updated after each event so the next
    one's "no override given" fallback (event["sigma"]/["gamma"] is None) resolves to whatever the
    PREVIOUS event left this population at, not the population's original pre-Stage-5 value.

    Arguments:
        const: [Constants]
        a_type_events: [list of dict] {"t", "sigma", "gamma"} entries.
        grain_batches, grain_batch_meta: [list, list] PARALLEL lists - see
            _batchAndStepGrainSpecs()'s own return for the exact shape/meaning. Every
            batch here must already have been spawned using the correct (post-phase-1) sigma_own -
            this function only re-splices already-alive grains' own remaining RK4 steps, it does
            NOT fix up spawn-time mass-binning (see _runSimulationErosion()'s own "phase 1"/"phase
            2" comments for why that split is necessary).

    Return:
        grain_batches: [list] A NEW list (same length/order as the input) - grain stepping is
            functional (_stepGrainPopulationRK4() always returns fresh arrays), so re-splitting
            naturally produces new tuples rather than mutating existing ones.
    """

    grain_batches = list(grain_batches)
    grain_batch_meta = list(grain_batch_meta)

    for event in a_type_events:
        if event["sigma"] is None and event["gamma"] is None:
            continue

        tick_idx_A = int(round(event["t"]/const.dt))

        for i, (batch, (K_old, sigma_old)) in enumerate(zip(grain_batches, grain_batch_meta)):
            sigma_new = sigma_old if event["sigma"] is None else event["sigma"]
            if event["gamma"] is None:
                K_new = K_old
            else:
                K_new = event["gamma"]*const.shape_factor*const.rho_grain**(-2/3.0)

            grain_batches[i] = _resplitGrainBatchAtTick(const, batch, tick_idx_A, K_new, sigma_new,
                atm_map, sin_slope)
            grain_batch_meta[i] = (K_new, sigma_new)

    return grain_batches


def _runSimulationErosion(const):
    """ erosion_on=True and/or disruption_on=True path of runSimulation() (called from there once
    fragmentation_on/compute_wake are confirmed False/off) - the main fragment's segment chain
    (Stage 3b, _buildMainFragmentSegments()) plus continuous grain spawning/stepping (Stage 3c,
    _spawnGrainsForSegment()/_stepGrainPopulationRK4()), plus disruption (Stage 4, see that
    section of this docstring below) - aggregated into the same results_list contract
    _runSimulationRK4Reference()/ablateAll() produce - built and validated by direct, repeated
    comparison against MetSimErosion.runSimulation() on this file's own representative scenarios
    (see the implementation plan's Stage 3d/Stage 4 write-ups for the full validation numbers), not
    assumed correct from the pieces being individually validated.

    This aggregation was reverse-engineered from ablateAll()'s actual source, not from what would
    seem like the "obviously correct" semantics - several real, non-obvious quirks were found this
    way and were reproduced here DELIBERATELY (not fixed), since the goal is a tolerance-matched
    substitute, not a corrected reimplementation. THREE of the four quirks originally documented in
    points 1-3 below were themselves later fixed upstream, in MetSimErosion.py commit 6be7301 ("Fix
    silent state-overwrite bugs in MetSimErosion.py") - this function was updated to match each one
    (see the reasoning/verification each point now documents), keeping the tolerance-matched-
    substitute goal intact against the NEW, corrected reference behavior rather than the old one:

    1) mass_total_active (results_list column 15) is weighted by frag.n_grains for every daughter/
       grain contribution (main fragment's own n_grains is always 1, so its own contribution is
       unaffected). Before commit 6be7301, the reference summed frag.m UNWEIGHTED
       (`active_fragments = [frag.m for frag in fragments if frag.active]`) even though
       lum_total/electron_density_total already weighted by n_grains (frag.lum = lum*frag.n_grains)
       - a real, confirmed asymmetry this function used to reproduce deliberately. That commit
       fixed it (`mass_total_active += frag.m*frag.n_grains`, matching lum_total's own convention)
       - see test_run_simulation_erosion_mass_total_active_weighted() for the current validation (it used
       to be named ...Unweighted and assert the opposite). The SEPARATE death-tick exclusion below
       (point 3) is UNCHANGED by this commit - confirmed directly against its own diff - and still
       applies on top of the weighting.

    2) lum_eroded/tau_eroded (columns 3/7) are now ALWAYS populated for non-complex (frag.complex is
       False) daughters/grains, regardless of const.fragmentation_show_individual_lcs. Before
       commit 6be7301, the "luminosity_eroded += frag.lum" line sat inside
       "if fragmentation_show_individual_lcs: if frag.complex: ... else: luminosity_eroded += ...",
       so by DEFAULT (flag False) these two columns stayed exactly 0.0 even while grains were
       actively radiating into lum_total/tau_total - a real quirk this function used to reproduce
       deliberately. That commit re-nested the eroded/disrupted aggregate tracking OUTSIDE the flag
       check (only the SEPARATE per-entry breakdown - still deliberately not implemented here, see
       the Stage 5 section below - stays gated by it), so lum_eroded/tau_eroded are now always
       populated for non-complex fragments. The "if frag.complex: (skip)" half of that same branch
       is UNCHANGED by the commit and was already being reproduced correctly here for daughters
       (Stage 4's own non-complex "fragment" daughters) once Stage 5 added complex F/EF/D entities -
       but was NOT yet correctly applied to grains spawned by a complex EF daughter's own ongoing
       erosion, nor by a Stage 5 "D" dust release (both inherit complex=True from their parent via
       spawn_child()'s dict copy, MetSimErosion.py:373-384, exactly like any other attribute) -
       fixed at the same time by tagging every grain_specs dict with a "complex" bool at its
       producer site and threading it through _batchAndStepGrainSpecs()'s own (K, sigma) batching
       (see that function's own docstring) - see
       test_run_simulation_erosion_lum_eroded_always_tracked()/
       test_run_simulation_erosion_lum_eroded_excludes_complex_fragments() for the current validation.

    3) brightest_*/leading_frag_* candidacy INCLUDES each fragment's own death tick as a valid
       candidate. Before commit 6be7301, the reference's brightest-tracking update sat AFTER the
       per-fragment kill-check `continue`, and its leading-fragment scan used a POST-loop
       `[frag for frag in fragments if frag.active]` filter (by which point a fragment that died
       THIS tick already had active=False) - so a fragment's own last recorded row could contribute
       to lum_total/mass_total_active while never being eligible to be "the brightest/leading
       fragment" that tick; for a genuinely single-body flight, this meant the reference's own last
       row reported brightest_height=0.0/leading_frag_height=None instead of the fragment's real
       final value - a real, confirmed reference quirk this function used to reproduce
       deliberately, and locked in by an earlier version of
       test_results_list_single_body_brightest_leading_equal_main(). That commit moved the brightest-
       tracking block before the kill-check, and the leading-fragment scan to a tick-start snapshot
       taken before any fragment active that tick can die - so a fragment's own death tick is now a
       valid candidate in both engines. Confirmed directly: a real single-body reference run's own
       last row now reports its actual final height/velocity. This function's own candidate-table
       construction was updated to match (no longer excludes each fragment's/grain's own last
       row) - see the current test_results_list_single_body_brightest_leading_equal_main() for the
       (inverted) validation.

    4) Simplification, not a match: leading_frag_* is None in the reference on the (rare) tick
       where literally every active fragment dies simultaneously (MetSimErosion.py:1240-1244) -
       proven analytically (see the implementation plan) that this can only ever be the LAST row of
       the whole simulation, never a middle one (once a tick empties n_active to 0, the reference's
       own `while const.n_active > 0` loop stops, so no further rows exist to show a mid-simulation
       gap) - this function reports 0.0 there instead of None/NaN, documented rather than silently
       differing. UNCHANGED by commit 6be7301 - now a genuinely rare edge case (point 3 above means
       most single-fragment-dies ticks are no longer affected at all), only relevant when the very
       last active fragment(s) in the whole simulation die on the same, final tick.

    5) Performance-driven architecture, not a semantics quirk: grain stepping uses
       _stepGrainPopulationRK4() (vectorized across every grain sharing one segment's (K, sigma) -
       true of every plain erosion grain, since K depends only on const.rho_grain and sigma is
       inherited unchanged from the segment), not a per-grain _stepGrainRK4() call. Measured
       directly: the per-grain-call approach made the WHOLE erosion pipeline SLOWER than the RK4
       reference it exists to replace at the epoch count needed for acceptable accuracy (0.4x
       "speedup" at const.erosion_n_epochs=1000 on this file's representative scenario) - pure
       Python function-call/loop-setup overhead repeated per grain bin (tens of thousands of them),
       not the underlying arithmetic. _stepGrainPopulationRK4() fixes this by keeping the exact same
       per-tick algorithm (validated bit-for-bit equal to _stepGrainRK4() - see this file's own
       tests) while restructuring the loop nesting (iterate ticks, vectorize grains); with it, the
       same scenario measures a real 3.4x wall-clock speedup at the validated default n_epochs.
       See Constants.erosion_n_epochs's own comment for the accuracy/speed tradeoff this default
       was chosen from, and Stage 7 in the implementation plan for further (Cython) headroom beyond
       this.

    Known, deliberately out-of-scope gaps (diagnostic-only Constants fields this function does not
    set, matching how Stage 5 itself is still NotImplementedError): erosion_beg_vel/
    erosion_beg_mass/erosion_beg_dyn_press/mass_at_erosion_change (MetSimErosion.py:1014-1024, set
    at the reference's own first per-tick erosion mass-loss event - this function's discrete-epoch
    spawning has no exact equivalent instant to match without approximation) and
    frag_main.h_grav_drop_total (accumulates PER-SEGMENT in this function's own AnalyticTrajectory-
    based segments, not carried cumulatively across segment boundaries the way the reference's
    running frag.h_grav_drop_total does - reportedHeightAt() already gets this right internally
    for frag_main.h itself, this gap is only in the raw h_grav_drop_total field, which nothing in
    this module reads back).

    ### Stage 4: disruption ###

    const.disruption_on may now also be True (routes here even with erosion_on=False - a fragment
    can disrupt without ever eroding). The main fragment's segment chain is scanned for the first
    GLOBAL TICK its own dynamic pressure exceeds const.compressive_strength (_findDisruptionTime() -
    a tick-exact match, not a continuous-time root-find, since ablateAll() itself only ever checks
    this once per dt tick, MetSimErosion.py:1027-1028) - not necessarily the peak dynamic pressure,
    which a descending body typically reaches only after decelerating for a while (density rising,
    velocity falling) - the FIRST crossing, matching the reference exactly. If found, the main
    chain is truncated there and generateFragments() (completely unchanged, reused exactly as
    Stage 3's grain spawning already does) spawns: (a) "fragment" daughters
    (const.disruption_mass_index/disruption_mass_min_ratio/disruption_mass_max_ratio applied to
    (1-disruption_mass_grain_ratio) of the disrupting fragment's mass, keep_eroding=const.erosion_on
    - so daughters keep eroding if the simulation itself has erosion_on=True) and (b) grains (the
    remaining leftover mass, keep_eroding=False, exactly like a plain erosion epoch spawn - see
    _makeVirtualParentFragment()'s rho= parameter for why the virtual parent must carry the
    disrupting fragment's ACTUAL rho, not always const.rho, since daughters inherit rho/K AS-IS via
    spawn_child(), unlike grains). Each eroding "fragment" daughter gets its OWN segment chain
    (_buildDaughterFragmentSegments() - genuinely simpler than the main fragment's own chain, since
    a daughter never gets the rho/K/sigma swap at erosion_height_change, MetSimErosion.py:983's
    `if frag.main` gate) and, recursively, its own grain population from its own eroding segments
    (_spawnGrainSpecsForAllErodingSegments(), the exact same helper the main fragment's own segments
    use) - true recursion is bounded to one level, since generateFragments() sets
    disruption_enabled=False unconditionally for every child it creates (MetSimErosion.py:1364),
    confirmed directly from source, so a daughter can never disrupt again itself.

    frag_main.m is forced to 0.0 if disruption occurred (MetSimErosion.py:1074, `frag.m = 0` set
    explicitly right before killFragment() - the ONLY attribute disruption zeroes; frag_main.v/h/
    length/lum/q/dyn_press all stay at their real pre-disruption values, matching how the SAME
    tick's results_list row - captured earlier in ablateAll()'s own per-tick sequence, before that
    zeroing - correctly shows the real pre-disruption mass in its own main_mass column).

    **Validated, and a real, precisely diagnosed remaining gap.** disruption_on=True with
    erosion_on=False (daughters do not keep eroding - simple ballistic/ablating bodies after the
    split) was validated end-to-end against MetSimErosion.runSimulation() and matches EXCELLENTLY:
    disruption height within 0.22m, main-fragment dynamics within 0.29m height / 0.002% velocity
    (first 90% of its own lifetime), frame-averaged (30fps) light curve within 0.01 mag, peak
    luminosity ratio 1.000x - see test_run_simulation_disruption_only() below.

    disruption_on=True WITH erosion_on=True (daughters keep eroding, matching the real combined use
    case in wmpl/Dynesty/priors/stony_meteoroid_eros+disruption.prior) matches similarly well for
    disruption height/main dynamics, but shows a SUBSTANTIALLY larger frame-averaged light-curve
    residual (up to ~1 magnitude, sustained over roughly a second right after disruption, for the
    representative scenario this was diagnosed on) than the erosion-only (no disruption) path's own
    0.06-0.28 mag range. Root-caused, not just observed: an eroding daughter's own mass, evaluated
    via its closed-form AnalyticTrajectory-based segment, can decay to an ESSENTIALLY ZERO value
    (confirmed directly: 1.4e-19 kg, many orders of magnitude below const.m_kill) exactly as its
    velocity approaches the segment's own v_n_floor - i.e. the closed form's own death point is
    governed by an extreme, near-singular mass-blowup regime, NOT a gentle approach to m_kill.
    Confirmed via a direct isolated-fragment comparison (spawning ablateAll() on a single hand-built
    Fragment matching one real daughter's exact spawn state, versus this function's own machinery
    on the identical state): total integrated luminosity (direct + all spawned grains) undershoots
    the reference by a factor of ~1.85x, and - critically - this ratio is COMPLETELY INSENSITIVE to
    const.erosion_n_epochs (identical to 4 significant figures from 1000 through 50000 epochs),
    ruling out epoch-count/discretization resolution as the cause and confirming a STRUCTURAL gap
    instead: mass-uniform epochs derived from a trajectory whose own mass decays near-singularly
    right at the death point place a disproportionate share of the eroded mass into epochs spawned
    very late in the daughter's life, when its OWN closed-form velocity is already near/at
    const.v_kill - grains born there are themselves born already at or past their own kill
    threshold, contributing negligible luminosity despite carrying real mass. This is the same
    underlying RK4-vs-closed-form non-convergence phenomenon _stepGrainRK4()'s own docstring
    documents for tiny erosion grains, now recognized to also apply to erosion-enabled DISRUPTION
    DAUGHTERS specifically near their own death, not just grains - the daughter's overall dynamics
    (validated separately, see _buildDaughterFragmentSegments()'s own docstring/tests) are NOT
    themselves wrong; the gap is specifically in how grain-spawning epochs derived from that
    trajectory behave near its most numerically extreme region.

    A full fix would mean extending _stepGrainRK4()/_stepGrainPopulationRK4()-style exact per-tick
    RK4 stepping (which already replicates the reference's own coarse-dt numerics on purpose,
    including this exact kind of non-convergence, rather than the more "correct" closed-form
    answer) to erosion-CAPABLE fragments, not just non-eroding grains - a genuinely new capability
    (mass loss split into ablation+erosion channels, per-tick erosion_coeff height updates, AND its
    own recursive grain-spawning from the RK4-tracked state) comparable in scope to Stage 3c's own
    body of work. Deliberately NOT attempted in this pass, given the scope already covered - scoped
    as a precise, diagnosed follow-up (see the implementation plan's Stage 4 write-up) rather than
    shipped silently or left as a vague "known issue". test_run_simulation_disruption_plus_erosion()
    below locks in the CURRENTLY measured residual with a loose, honest tolerance (not a tight one
    that would misrepresent this as solved) alongside tight checks on what IS solid here (disruption
    trigger height, mass budget, main-fragment dynamics).

    **Follow-up investigation, and a correction.** A later session revisited this gap after adding
    const.grain_evolution_analytic (see that flag's own docstring - an opt-in "full analytic" grain
    mode using _analyticGrainState()/_stepGrainPopulationAnalytic() instead of the RK4-replicating
    default). An isolated-daughter diagnostic initially suggested something far worse than described
    above - a BROAD divergence across most of a disruption daughter's own lifetime, not just its
    final tick - but this was traced to a bug in the DIAGNOSTIC SCRIPT itself, not this module: the
    script manually incremented const.total_time by const.dt after every ablateAll() call, on top of
    the increment ablateAll() already performs internally (MetSimErosion.py:1310), silently running
    the "reference" mirror at half the intended rate. With that fixed, the picture above is
    CONFIRMED, not overturned, and now more precisely localized: the daughter's own closed-form
    dynamics match the reference tool closely over the first 90%+ of its own lifetime (0.03%
    velocity error, <9% mass error - consistent with every other closed-form-vs-reference comparison
    in this file), and the real, remaining divergence is confined to the FINAL 1-2 ticks before
    death specifically, exactly where the mass-collapse described above happens. Switching grain
    evolution to const.grain_evolution_analytic=True recovers PART of the resulting light-curve
    deficit (worst frame-averaged |dmag| improves from ~1.06 to ~0.96 on this same scenario - see
    test_run_simulation_analytic_mode_disruption_gap_partially_improves()) but does not fully close it,
    since the daughter's own last-tick state - which any grain inherits its own spawn conditions
    from - is itself only ever resolved by either engine up to the same coarse dt grid; the full fix
    described above (exact RK4 stepping extended to erosion-capable fragments themselves) remains
    the only way to close this completely.

    **The full fix, implemented - real progress, plus a distinct, deeper gap found in the process.**
    A later session built the fix described above: _findMassCrashOnset() detects, on a chain's own
    LAST segment, the exact tick a fragment's mass ratio (vs the preceding tick) first drops below
    0.5 - a robust, cheap proxy for the runaway's onset (see that function's own docstring for why
    this threshold is not arbitrary), confirmed via direct trace on a real disruption daughter: mass
    fell from 1.47e-7 kg to 3.2e-20 kg (7 orders of magnitude) over 4 consecutive GLOBAL ticks
    (0.02s), velocity crashing from 13516 to 674 m/s in the same window - a ~2.6 million m/s^2
    deceleration no real body could sustain, and confirmed NOT something the reference tool's own
    coarse RK4 stepping produces for the equivalent object (its own analogous trajectory decelerates
    smoothly, 14313->12797->...->3108 m/s over 20 ticks/0.1s, right through the region the closed
    form used to collapse in). _resolveSegmentChainDeathRegime() truncates the closed-form segment
    back to the last trustworthy tick and hands off to _stepErodingFragmentRK4Tail() - a genuinely
    new per-tick RK4 stepper (mass loss split into ablation+erosion channels, plus its own recursive
    per-tick grain-spawning from the RK4-tracked state, mirroring ablateAll()'s exact sequence) - to
    finish that fragment's life exactly as the reference itself would. Deliberately NOT applied to a
    fragment's ENTIRE life (that would reintroduce the O(timesteps x fragments) cost this whole
    engine exists to eliminate): the crash-onset condition is itself a proxy for "already very close
    to death", so in every case measured so far the resulting RK4 tail is a short stretch (single-
    digit to low tens of ticks), not a wholesale return to per-tick stepping.

    This directly fixed the bug that prompted revisiting this gap: a disruption daughter hitting this
    exact crash inflated its own reported length right up to its (artificially early) death tick,
    producing a single ~67.6m BACKWARD jump in leading_frag_length in a complex-fragmentation
    scenario once that tick's candidacy was (correctly) excluded and a shorter grain took over as
    leader - confirmed eliminated (0 backward jumps > 1m, was 1) and the daughter's own tail now
    decelerates smoothly (matching the reference's own qualitative behavior) rather than crashing.
    Locked in by test_leading_frag_length_monotonic_after_crash_fix(), test_find_mass_crash_onset(), and
    test_step_eroding_fragment_rk4_tail_matches_reference() (the latter validated directly against an
    isolated ablateAll() mirror, not just indirectly through the full pipeline).

    It also measurably helps test_run_simulation_disruption_plus_erosion()'s own scenario: post-
    disruption integrated luminosity ratio improved from ~0.54-0.56x to ~0.624x. But its own WORST
    frame-averaged |dmag| did NOT improve (still ~1.06, unchanged to 3 significant figures) - traced
    directly (not assumed) rather than declared fixed: an isolated comparison of that scenario's own
    LARGEST daughter (m_start=0.026kg) against a full-life _stepErodingFragmentRK4Tail() mirror (not
    just its own final ticks) shows the closed form tracking that mirror closely over its ENTIRE
    life, never entering a crash regime at all for this daughter - yet mass_total_active still
    matches the reference to ~1.00-1.01x at every checked tick while lum_total undershoots by
    ~2.5-2.7x over an extended (~0.7s wide, not single-tick) window well BEFORE any daughter's own
    predicted death. This rules out the near-death-crash mechanism as the dominant cause for THIS
    scenario specifically (confirmed, not assumed: 10 of this scenario's 11 chains DO get an RK4
    tail attached, so the fix IS active - it just is not what is limiting this particular case) and
    points at a DIFFERENT, not-yet-root-caused mechanism, most likely in how the grain population
    spawned along the way (which dominates the light budget here - individual daughters' own direct
    light is only ~1000-1800 W each against a ~130000-345000 W total) reproduces the reference's own
    light output despite matching its mass budget almost exactly. test_run_simulation_disruption_plus_erosion()'s
    own tolerance is left as-is (still comfortably covers the measured ~1.06) rather than tightened,
    since the dominant mechanism it was guarding against has changed but the gap has not closed -
    see that test's own docstring for the precise, current framing. Flagged here as a genuine,
    separate open item, not silently left implicit.

    **The "distinct, deeper gap" above: root-caused and fixed.** Triggered by a separate user report
    on the complex (h_init=80km) scenario's own light curve (visibly wrong decay shape right after
    the disruption peak). Investigated by directly comparing a full snapshot of every active
    fragment's (v, m, lum) at a representative post-disruption tick, in both engines: daughters' own
    direct light matched the reference closely (even slightly ahead), but the GRAIN population
    (which dominates the light budget, matching the ~1000-1800 W vs ~130000-345000 W split noted
    above) was both ~4.2x under-massed and ~34% under-velocity at that instant. Ruled out epoch-
    count (tested to 80x the validated default budget - confirmed via a direct per-daughter epoch-
    count check that this really did increase resolution, 20->81 distinct spawn times, yet the
    deficit ratio barely moved) and erosion_n_smear (tested to 25x - same near-zero effect) as
    causes before finding the real one: _makeVirtualParentFragment() hardcoded n_grains=1 for the
    virtual parent every grain-spawning generateFragments() call uses - but a Stage 4 disruption
    "fragment" daughter's own mass bin routinely represents MORE than one identical physical body
    (generateFragments()'s power-law mass-binning produces n_grains>1 for its smaller bins,
    confirmed up to 9 on this exact scenario). The reference does not have this gap:
    MetSimErosion.py's spawn_child() does a full __dict__ copy, so a grain-spawning call's own
    frag_child.n_grains STARTS at frag_parent.n_grains (not 1) before generateFragments() multiplies
    in its own per-bin count - a daughter representing N identical bodies correctly produces N times
    as much grain mass/light as a lone body would; this function's own virtual parent never
    inherited that multiplier, silently under-producing grain mass/luminosity from every daughter
    with n_grains>1 by exactly that factor. Fixed by threading n_grains through
    _makeVirtualParentFragment() and every function that calls it on a daughter's own behalf
    (_spawnGrainsForSegment(), _spawnGrainSpecsForAllErodingSegments(),
    _stepErodingFragmentRK4Tail(), _resolveSegmentChainDeathRegime()), sourced from each daughter's
    own dict in this function. Main and Stage 5 F/EF daughters are unaffected (both always have
    n_grains=1 - the former by physical uniqueness, the latter because _applyFragmentationEntry()
    spawns each as its own separate Fragment rather than mass-binning them).

    This turned out to BE the exact "distinct, deeper, not-yet-root-caused mechanism" flagged above
    for test_run_simulation_disruption_plus_erosion()'s own scenario too, not just the complex scenario
    the report was about - confirmed directly, not assumed, by re-measuring both after the fix:
    test_run_simulation_complex_scenario_accuracy()'s own worst dmag improved 0.85 -> 0.76 (the small
    remainder is the already-documented epoch-allocation resolution tradeoff among that scenario's
    own 12, comparably small daughters, not this mechanism), while
    test_run_simulation_disruption_plus_erosion()'s own worst dmag - unchanged at ~1.06 through every
    earlier fix in this section - dropped to ~0.32, and its post-disruption integrated luminosity
    ratio (~0.54-0.624x across those same earlier partial fixes) is now ~0.998x. Both tests' own
    tolerances were tightened accordingly (1.3->1.0 and 1.5->0.5 respectively) rather than left at
    their old, now-stale loose values - see each test's own docstring for the full account.

    Arguments:
        const: [Constants] const.erosion_on and/or const.disruption_on must be True (checked by the
            caller) - at least one, not necessarily both.

    Return:
        (frag_main, results_list, wake_results): same contract as runSimulation()'s non-erosion
        path, with the same additive frag_main.n_queries extension (summed across the main
        fragment's own segments only - grains use _stepGrainPopulationRK4(), which does not build
        an AnalyticTrajectory at all, so they never contribute to this counter).
    """

    dt = const.dt
    K = const.gamma*const.shape_factor*const.rho**(-2/3.0)
    sin_slope = math.cos(const.zenith_angle)
    atm_map = AtmEquivHeightMap(const.dens_co, const.h_init)
    h_real_floor = const.h_kill - max(0.05*(const.h_init - const.h_kill), 5000.0)

    # Matches _runSimulationRK4Reference()'s own init (MetSimErosion.py:1370: main fragment is
    # "fragment 0"); _spawnGrainsForSegment() -> generateFragments() increments this for every
    # grain bin spawned below (MetSimErosion.py:653-654), exactly as the reference does.
    const.total_fragments = 1

    # Matches _runSimulationRK4Reference()'s own pre-loop adjustment (MetSimErosion.py:1378-1381):
    # grain density can never be LOWER than the bulk density, so if the caller's rho exceeds the
    # default/explicit rho_grain, rho_grain is silently raised to match - this directly changes
    # every grain's K (generateFragments() sets frag_child.rho = const.rho_grain unconditionally
    # for plain erosion grains), so skipping this produced a real, measured mass-budget mismatch
    # (up to ~9% on the main fragment's own final mass for const.rho=3300 vs the rho_grain default
    # of 3000) before this fix.
    if const.rho > const.rho_grain:
        const.rho_grain = const.rho

    # --- Stage 5: complex/discrete fragmentation entries - matches _runSimulationRK4Reference()'s
    # own init (MetSimErosion.py:2093-2100): assign each entry a unique id (frag_daughters'
    # "complex_id" and this function's own post-run bookkeeping below both key off this) and reset
    # its output parameters, gated by const.fragmentation_on so the non-fragmentation path (the
    # overwhelming majority of this file's own existing tests) is completely untouched. ---
    if const.fragmentation_on:
        for i, frag_entry in enumerate(const.fragmentation_entries):
            frag_entry.id = i
            frag_entry.resetOutputParameters()

        segments, frag_daughters, frag_grain_specs, a_type_events = \
            _buildMainFragmentSegmentsWithFragmentation(const, K, sin_slope, atm_map, h_real_floor)
    else:
        segments = _buildMainFragmentSegments(const, K, sin_slope, atm_map, h_real_floor)
        frag_daughters, frag_grain_specs, a_type_events = [], [], []

    # --- Stage 4: disruption - a ONE-TIME event on the main fragment only (confirmed daughters
    # never get disruption_enabled=True: generateFragments() sets it False unconditionally for
    # every child it creates, MetSimErosion.py:1364, and complex-fragmentation daughters get the
    # same treatment, MetSimErosion.py:1151,1198 - so this never recurses). If it triggers before
    # the main fragment's own natural death, truncate its segment chain there and spawn daughter
    # "fragments" (which may themselves keep eroding, recursively reusing this same segment-chain
    # + grain-spawning machinery) and/or grains from its state at that instant.
    daughter_fragments = []  # each: {"segments": [...], "n_grains": float, "complex_id": int/None}
    disruption_grain_specs = []  # grain_specs-shaped dicts from the one-time leftover-mass spawn
    t_disrupt = None

    if const.disruption_on:
        t_disrupt, seg_idx = _findDisruptionTime(const, segments)

        if t_disrupt is not None:
            segments = segments[:seg_idx + 1]
            seg_d = segments[-1]
            seg_d["t_end"] = t_disrupt

            v_d, m_d, length_d, h_d, lum_d, q_d, dp_d, tau_d = _evaluateSegment(seg_d["traj"],
                seg_d["atm_height_fn"], np.array([t_disrupt]), seg_d["K"], seg_d["sigma_own"],
                seg_d["v_start"], seg_d["length_start"], const,
                grav_drop_start=seg_d.get("grav_drop_start", 0.0))
            v_d, m_d = float(v_d[0]), float(m_d[0])
            length_d, h_d = float(length_d[0]), float(h_d[0])
            grav_drop_d = (seg_d.get("grav_drop_start", 0.0)
                + float(seg_d["traj"].gravityDropAt(t_disrupt)))

            const.disruption_height = h_d

            # rho/gamma at the disruption instant - read directly from the segment containing
            # t_disrupt (now that every segment dict carries its own rho/gamma - Stage 5), not
            # inferred from len(segments)==3: that heuristic assumed the ONLY way segments could
            # have more than 1 entry was the main fragment's own 3-segment erosion chain, which
            # stopped being true once Stage 5 fragmentation can add arbitrarily many segments before
            # a disruption ever triggers. Only differs from const.rho/const.gamma if the main
            # fragment had already crossed erosion_height_change (rho) or a prior "A" event already
            # fired (gamma) - see _makeVirtualParentFragment()'s own docstring for why both must be
            # threaded explicitly (daughter "fragments" inherit rho/K AS-IS, unlike grains).
            rho_d = seg_d["rho"]
            gamma_d = seg_d["gamma"]

            mass_frag_disruption = m_d*(1.0 - const.disruption_mass_grain_ratio)

            fragments_total_mass = 0.0
            if mass_frag_disruption > 0:
                disruption_mass_min = const.disruption_mass_min_ratio*mass_frag_disruption
                disruption_mass_max = const.disruption_mass_max_ratio*mass_frag_disruption

                parent = _makeVirtualParentFragment(const, m_d, v_d, h_d, length_d,
                    seg_d["sigma_own"], rho=rho_d, gamma=gamma_d)

                frag_children, const = generateFragments(const, parent, mass_frag_disruption,
                    const.disruption_mass_index, disruption_mass_min, disruption_mass_max,
                    keep_eroding=const.erosion_on, disruption=True,
                    mass_model=const.erosion_grain_distribution)

                fragments_total_mass = sum(fc.n_grains*fc.m for fc in frag_children)

                # Every daughter from this SAME generateFragments() call shares the disrupting
                # parent's own (K, sigma) exactly - confirmed from source: the keep_eroding=True,
                # disruption=True branch never overwrites frag_child.rho/K/sigma (that only happens
                # in the "not keep_eroding" grain branch) - only frag_child.m differs per mass bin.
                # _buildDaughterFragmentSegmentsBatch() exploits this to share the (fixed per-call-
                # overhead) trajectory-spline construction across every daughter here instead of
                # rebuilding it once per daughter - see that function's own docstring.
                d_masses = np.array([fc.m for fc in frag_children])
                d_segments_batch = _buildDaughterFragmentSegmentsBatch(const, frag_children[0].K,
                    frag_children[0].sigma, d_masses, v_d, h_d, sin_slope, atm_map,
                    t_start=t_disrupt, length_start=length_d, grav_drop_start=grav_drop_d,
                    h_real_floor=h_real_floor, rho=rho_d, gamma=gamma_d)

                for fc, d_segments in zip(frag_children, d_segments_batch):
                    daughter_fragments.append({"segments": d_segments,
                        "n_grains": float(fc.n_grains), "complex_id": None})

            mass_grain_disruption = m_d - fragments_total_mass
            if mass_grain_disruption > 0:
                parent = _makeVirtualParentFragment(const, m_d, v_d, h_d, length_d,
                    seg_d["sigma_own"], rho=rho_d, gamma=gamma_d)
                # erosion_mass_index/min/max inherited from the disrupting fragment - identical to
                # const.erosion_mass_index/min/max here (the main fragment never changes its own
                # copy) - use const directly, matching _spawnGrainsForSegment()'s own convention.
                grain_children, const = generateFragments(const, parent, mass_grain_disruption,
                    const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max,
                    keep_eroding=False, mass_model=const.erosion_grain_distribution)

                for gc in grain_children:
                    disruption_grain_specs.append({"m": gc.m, "n_grains": gc.n_grains,
                        "sigma": gc.sigma, "rho": gc.rho, "K": gc.K, "v": v_d, "h_real": h_d,
                        "length": length_d, "t": t_disrupt})

    # --- Stage 5: a disrupted-and-killed main fragment can never reach a later fragmentation-entry
    # trigger height (killFragment() deactivates it, MetSimErosion.py:1074-1075, and ablateAll()'s
    # own per-tick `if not frag.active: continue` then skips it in every later tick, including the
    # complex-fragmentation block) - but _buildMainFragmentSegmentsWithFragmentation() has no
    # knowledge of disruption (a wholly separate per-tick check, MetSimErosion.py:1027) and may
    # already have applied entries at heights the main fragment would never really have reached.
    # Filtered out here by each entry's own recorded trigger time, not by re-deriving anything -
    # cheap and exact, since every fragmentation/dust output dict already carries its own trigger
    # time (t_start for daughters, t for dust grains, t for A-type events).
    if t_disrupt is not None:
        frag_daughters = [fd for fd in frag_daughters if fd["t_start"] <= t_disrupt + 1e-9]
        frag_grain_specs = [gs for gs in frag_grain_specs if gs["t"] <= t_disrupt + 1e-9]
        a_type_events = [ev for ev in a_type_events if ev["t"] <= t_disrupt + 1e-9]

    # --- Stage 5: build each surviving F/EF complex-fragmentation daughter's own (always
    # single-segment - see _buildComplexFragmentDaughterSegments()'s own docstring) chain and fold
    # it into daughter_fragments alongside any Stage 4 disruption "fragment" daughters - the
    # aggregation below (results_list columns, brightest_*/leading_frag_* candidacy) treats every
    # daughter_fragments entry identically regardless of WHY it exists, so no separate aggregation
    # path is needed for complex-fragmentation-born fragments. ---
    for fd in frag_daughters:
        fd_segments = _buildComplexFragmentDaughterSegments(const, fd["K"], fd["sigma"],
            fd["erosion_coeff_fixed"], fd["m_start"], fd["v_start"], fd["h_real_start"], sin_slope,
            atm_map, t_start=fd["t_start"], length_start=fd["length_start"],
            grav_drop_start=fd["grav_drop_start"], h_real_floor=h_real_floor,
            erosion_mass_index=fd["erosion_mass_index"], erosion_mass_min=fd["erosion_mass_min"],
            erosion_mass_max=fd["erosion_mass_max"], rho=fd["rho"], gamma=fd["gamma"])
        daughter_fragments.append({"segments": fd_segments, "n_grains": float(fd["n_grains"]),
            "complex_id": fd["complex_id"]})

    # --- Stage 5 type "A", phase 1 of 2: retroactively truncate-and-continue every ALREADY-ALIVE
    # daughter's own segment chain at each "A" trigger instant - matching ablateAll()'s own
    # semantics (MetSimErosion.py:1103-1109: frag.sigma/frag.gamma overwritten for EVERY fragment
    # in `fragments + frag_children_all + [frag]`, main/daughters/grains alike). MUST run BEFORE
    # any grain-spawning below: a daughter's own eroding segments determine what sigma_own newly
    # SPAWNED grains get baked in with at birth (via _makeVirtualParentFragment()), not just how an
    # already-spawned grain's own RK4 stepping continues - spawning grains first and only
    # re-splicing mid-flight RK4 trajectories after the fact (phase 2, below, after grain spawning)
    # would leave any grain a daughter spawns AFTER t_A still using the PRE-event sigma_own at its
    # own birth, silently wrong. The main fragment's own segments need no equivalent step here:
    # they are already split at every "A" trigger height by construction
    # (_buildMainFragmentSegmentsWithFragmentation()'s own trigger loop), so every grain spawned
    # from them is already confined to one (pre- or post-event) segment with the correct sigma_own
    # baked in at spawn - see _resplitDaughterAtTick()'s own docstring for the per-daughter details.
    for event in a_type_events:
        if event["sigma"] is None and event["gamma"] is None:
            continue
        for daughter in daughter_fragments:
            _resplitDaughterAtTick(const, daughter, event["t"], event["sigma"], event["gamma"],
                sin_slope, atm_map, h_real_floor)

    # disrupted (used below to force frag_main.m to exactly 0.0, matching MetSimErosion.py:1074-
    # 1075) must be keyed on t_disrupt directly, NOT on daughter_fragments/grain-spec list lengths -
    # those now also carry Stage 5 complex-fragmentation entities, which do NOT zero the main
    # fragment's mass (F/EF/D reduce frag.m by the split-off amount only, already baked into the
    # main chain's own subsequent segments by _applyFragmentationEntry()).
    disrupted = t_disrupt is not None

    # --- Detect and fix the closed-form's own near-singular mass-blowup-near-death artifact
    # (_findMassCrashOnset()'s own docstring has the full physical/numerical account) in the main
    # fragment's own chain and every daughter's own chain, by attaching an exact per-tick RK4
    # continuation (_stepErodingFragmentRK4Tail()) for the short final stretch before death where
    # needed. MUST run before anything below consumes segments/daughter["segments"] - both grain-
    # spawning (epoch mass allocation) and dense evaluation need the (possibly truncated) t_end
    # this produces, not the original _findSegmentDeathTime()-resolved one. A no-op for the
    # overwhelming majority of chains, which never enter this regime.
    segments, tail_grain_specs = _resolveSegmentChainDeathRegime(const, segments)
    for gs in tail_grain_specs:
        gs["complex"] = False  # main fragment's own chain is never complex
    for daughter in daughter_fragments:
        daughter["segments"], d_tail_grain_specs = _resolveSegmentChainDeathRegime(const,
            daughter["segments"], n_grains=daughter["n_grains"])
        is_complex_daughter = daughter["complex_id"] is not None
        for gs in d_tail_grain_specs:
            gs["complex"] = is_complex_daughter
        tail_grain_specs.extend(d_tail_grain_specs)

    tick_idx_main, v_main, m_main, length_main, h_main, lum_main, q_main, dp_main, tau_main, \
        n_queries_total, _tgs_main = _evaluateFragmentSegments(const, segments)
    n_ticks_main = len(tick_idx_main)

    # --- Spawn every eroding segment's grains (main fragment's own, plus every eroding disruption
    # "fragment"/complex-fragmentation daughter's own) as RAW SPECS (not yet stepped), then batch
    # EVERYTHING in this simulation - those specs, plus disruption's leftover-mass spawn, every
    # Stage 5 "D" (dust release) trigger, and every RK4-tail's own per-tick erosion-channel
    # spawning - through ONE _batchAndStepGrainSpecs() call, grouped by exact (K, sigma). This
    # collapses what used to be one _stepGrainPopulationRK4() call per eroding segment (19 on this
    # file's own "complex scenario") down to one call per DISTINCT (K, sigma) actually present
    # across the whole simulation (2, on that same scenario - see
    # _spawnGrainSpecsForAllErodingSegments()'s own docstring, "One further step", for the measured
    # numbers) - a real, measured further speedup on top of the epoch-sharing fix below, for the
    # exact same total grain population and physics (purely a call-count reduction). grain_batch_meta
    # is a list PARALLEL to grain_batches (one (K, sigma) pair per batch) - needed only by Stage 5
    # type "A" retroactive re-splitting below; every other consumer of grain_batches ignores it.
    #
    # global_total_mass is computed ONCE, across every chain (main's own + every daughter's own),
    # and passed to EVERY _spawnGrainSpecsForAllErodingSegments() call below - extends Stage 7's own
    # within-one-chain proportional epoch allocation to across-the-whole-simulation (see that
    # function's own docstring for global_total_mass, and the real numbers that made this
    # necessary: 13 chains, 19 eroding segments, 2.75 MILLION candidate rows for only 356 output
    # ticks before this fix, on this file's own "complex scenario"). ---
    global_total_mass = _totalErodedMassAcrossChains(
        [segments] + [d["segments"] for d in daughter_fragments])

    eroding_grain_specs = _spawnGrainSpecsForAllErodingSegments(const, segments,
        global_total_mass=global_total_mass)
    for gs in eroding_grain_specs:
        gs["complex"] = False  # main fragment's own chain is never complex
    for daughter in daughter_fragments:
        d_specs = _spawnGrainSpecsForAllErodingSegments(const, daughter["segments"],
            global_total_mass=global_total_mass, n_grains=daughter["n_grains"])
        is_complex_daughter = daughter["complex_id"] is not None
        for gs in d_specs:
            gs["complex"] = is_complex_daughter
        eroding_grain_specs.extend(d_specs)

    # disruption_grain_specs: disruption only ever fires on the main fragment (never recurses onto
    # daughters, see the comment above where daughter_fragments is initialized), and main is never
    # complex, so this leftover-mass spawn is never complex either.
    for gs in disruption_grain_specs:
        gs["complex"] = False
    # frag_grain_specs: Stage 5 "D" (dust release) - every D-type virtual parent gets complex=True
    # unconditionally (MetSimErosion.py:1253, inherited by its grain children via spawn_child()'s
    # dict copy), so these are always complex.
    for gs in frag_grain_specs:
        gs["complex"] = True

    grain_batches, grain_batch_meta, batch_is_complex = _batchAndStepGrainSpecs(const,
        eroding_grain_specs + disruption_grain_specs + frag_grain_specs + tail_grain_specs,
        atm_map, sin_slope)

    # --- Stage 5 type "A", phase 2 of 2: retroactively re-splice every already-SPAWNED grain's own
    # mid-flight RK4 trajectory (main's own, every daughter's own, and the one-shot disruption-
    # leftover/D-type dust populations) at each "A" trigger instant - orthogonal to phase 1 above
    # (which fixed daughter SEGMENT continuation/spawn-time correctness): every grain batch here was
    # already spawned using the correct (post-phase-1) sigma_own, but a grain ALIVE AT t_A when it
    # was spawned still needs its own remaining RK4 steps re-derived with the new (K, sigma) from
    # that tick on - see _resplitGrainBatchAtTick()'s own docstring. No-op if a_type_events is
    # empty, which covers every non-Stage-5 and non-"A" Stage 5 scenario at zero added cost beyond
    # the check itself. ---
    grain_batches = _resplitAllGrainBatchesForEvents(const, a_type_events, grain_batches,
        grain_batch_meta, atm_map, sin_slope)

    # --- Evaluate every disruption "fragment" daughter's own dense per-tick state (same evaluation
    # _evaluateFragmentSegments() already gives the main fragment above). ---
    daughter_evals = []
    for daughter in daughter_fragments:
        tick_idx_d, v_d_arr, m_d_arr, len_d_arr, h_d_arr, lum_d_arr, q_d_arr, dp_d_arr, \
            tau_d_arr, n_q_d, _tgs_d = _evaluateFragmentSegments(const, daughter["segments"])
        n_queries_total += n_q_d
        daughter_evals.append((tick_idx_d, v_d_arr, m_d_arr, len_d_arr, h_d_arr, lum_d_arr,
            q_d_arr, dp_d_arr, tau_d_arr, daughter["n_grains"]))

    # --- Stage 5 lifecycle bookkeeping: populate each fragmentation entry's own .fragments/
    # .final_mass, matching _runSimulationRK4Reference()'s own post-run block (MetSimErosion.py:
    # 2180-2204/GUI.py's FragmentationEntry) - only F/EF daughters ever populate this (D-type dust
    # bins are grains, excluded by the reference's own `if not frag.grain` check there; M/A entries
    # never spawn a tracked fragment at all). daughter_fragments/daughter_evals are built in lock-
    # step above (one _evaluateFragmentSegments() call per daughter_fragments entry, in order), so
    # zipping them is a safe, exact pairing - not re-deriving anything. ---
    if const.fragmentation_on:
        for frag_entry in const.fragmentation_entries:
            frag_entry.fragments = []

        for daughter, d_eval in zip(daughter_fragments, daughter_evals):
            if daughter["complex_id"] is None:
                continue

            frag_entry = next((x for x in const.fragmentation_entries
                if x.id == daughter["complex_id"]), None)
            if frag_entry is None:
                continue

            tick_idx_d, _v_d, m_d_arr = d_eval[0], d_eval[1], d_eval[2]
            m_start_d = daughter["segments"][0]["m_start"]
            final_m = float(m_d_arr[-1]) if len(tick_idx_d) > 0 else m_start_d

            frag_stub = Fragment()
            frag_stub.id = None
            frag_stub.main = False
            frag_stub.grain = False
            frag_stub.complex = True
            frag_stub.complex_id = daughter["complex_id"]
            frag_stub.m = final_m
            frag_stub.m_init = m_start_d
            frag_entry.fragments.append(frag_stub)

            # Matches MetSimErosion.py:2196-2201: assumes every daughter from the same entry ends
            # with (approximately) equal mass - true here by construction, since F/EF daughters all
            # start with an EQUAL share of the parent's mass (_applyFragmentationEntry()) and share
            # identical physical parameters/atmosphere, so this is recomputed identically (not
            # meaningfully overwritten) for every matching daughter, not just the last one.
            final_mass = frag_entry.number*final_m
            if final_mass < 1e-3:
                final_mass = None
            frag_entry.final_mass = final_mass

    max_grain_idx = -1
    for batch in grain_batches:
        if len(batch[0]):
            max_grain_idx = max(max_grain_idx, int(batch[0].max()))
    for (tick_idx_d, *_rest) in daughter_evals:
        if len(tick_idx_d):
            max_grain_idx = max(max_grain_idx, int(tick_idx_d.max()))
    n_ticks = max(n_ticks_main, max_grain_idx + 1)

    def _binAdd(target, idx, values):
        """ target[idx] += values, correctly handling repeated indices (idx routinely has many
        grains/rows sharing the same GLOBAL tick) - via np.bincount(), replacing what used to be a
        direct np.add.at() call at each of this block's own 12 call sites. Both give the same
        scatter-sum result, but np.bincount() is numpy's own purpose-built C routine for exactly
        this "sum values into integer buckets" operation, while np.add.at() is a general (and, per
        numpy's own documentation, deliberately UNBUFFERED/slower) ufunc.at() fallback that has to
        support arbitrary ufuncs, not just addition - profiling found those 12 np.add.at() calls a
        real, measurable cost (see the implementation plan's own write-up). np.maximum.at() calls
        elsewhere in this file are NOT touched here - bincount has no max-reduction equivalent, so
        those still need ufunc.at()'s own general mechanism. """

        if len(idx) == 0:
            return
        target += np.bincount(idx, weights=values, minlength=len(target))

    # --- Allocate and fill the global (sum-aggregated) columns. ---
    lum_total = np.zeros(n_ticks)
    lum_main_col = np.zeros(n_ticks)
    edens_total = np.zeros(n_ticks)
    tau_total_num = np.zeros(n_ticks)
    tau_main_col = np.zeros(n_ticks)
    lum_eroded_raw = np.zeros(n_ticks)
    tau_eroded_num = np.zeros(n_ticks)
    mass_total_active = np.zeros(n_ticks)
    main_mass_col = np.zeros(n_ticks)
    main_h_col = np.zeros(n_ticks)
    main_len_col = np.zeros(n_ticks)
    main_v_col = np.zeros(n_ticks)
    main_dp_col = np.zeros(n_ticks)

    lum_total[tick_idx_main] += lum_main
    lum_main_col[tick_idx_main] = lum_main
    edens_total[tick_idx_main] += q_main
    tau_total_num[tick_idx_main] += tau_main*lum_main
    tau_main_col[tick_idx_main] = tau_main
    # mass_total_active EXCLUDES each fragment's own death/disruption tick - a real, confirmed
    # asymmetry with lum_total/edens_total/tau_total_num above (which correctly include it) -
    # unaffected by the n_grains-weighting fix below (mass_total_active's death-tick exclusion is
    # unchanged upstream, confirmed directly against the fixed reference's own diff). Traced
    # directly to MetSimErosion.py's own per-tick sequence: luminosity_total += frag.lum (:857)
    # happens INSIDE the per-fragment loop, before that fragment's own kill-check/continue
    # (:888-900) or disruption block (:1026-1075) - so a dying/disrupting fragment's CURRENT-tick
    # luminosity always lands in the running total before anything deactivates it.
    # mass_total_active, by contrast, is accumulated only for fragments that reach the END of their
    # own per-tick processing still active (killed/disrupted ones hit a 'continue' first) - a
    # dying/disrupting fragment's own current-tick mass is EXCLUDED, regardless of whether frag.m
    # itself was ever explicitly zeroed. Confirmed empirically against the reference: its own
    # mass_total_active at a real disruption tick matches ONLY the newly-spawned daughters, not
    # main's own pre-disruption mass - main_mass_col (a separate, dedicated column) correctly still
    # shows that same pre-disruption value at that tick, an intentionally different convention from
    # mass_total_active. main fragment's own n_grains is always 1, so no weighting needed here -
    # see the daughter/grain loops below for where n_grains weighting actually applies.
    mass_total_active[tick_idx_main[:-1]] += m_main[:-1]
    main_mass_col[tick_idx_main] = m_main
    main_h_col[tick_idx_main] = h_main
    main_len_col[tick_idx_main] = length_main
    main_v_col[tick_idx_main] = v_main
    main_dp_col[tick_idx_main] = dp_main

    # Candidate table for brightest_*/leading_frag_* - INCLUDES each fragment's own death tick as a
    # valid candidate (matching the reference's own fix moving its brightest-tracking block before
    # the per-fragment kill-check, and its leading-fragment scan to a tick-start snapshot taken
    # before any fragment active that tick can die - both confirmed directly against the fixed
    # reference: a single non-eroding body's own last row now reports its real final
    # height/velocity there, not 0.0). Main fragment first, then every disruption "fragment"
    # daughter (n_grains-weighting for lum applies to daughters too - a daughter bin can represent
    # multiple identical physical fragments, exactly like a grain bin), then every grain batch.
    cand_idx_parts = [tick_idx_main] if n_ticks_main > 0 else []
    cand_lumw_parts = [lum_main] if n_ticks_main > 0 else []
    cand_h_parts = [h_main] if n_ticks_main > 0 else []
    cand_len_parts = [length_main] if n_ticks_main > 0 else []
    cand_v_parts = [v_main] if n_ticks_main > 0 else []
    cand_dp_parts = [dp_main] if n_ticks_main > 0 else []

    # --- Fold in every disruption "fragment" daughter's own contribution - same n_grains-weighted
    # sum treatment as a grain batch, just from a dense segment-chain evaluation instead of
    # _stepGrainPopulationRK4(). Note (documented simplification, matching the same convention
    # already used for grain spawning): a daughter's contribution starts at its first evaluated
    # tick (t_disrupt+dt), not t_disrupt itself, even though the reference's own mass_total_active
    # would include its spawn-instant mass one tick earlier (fragments_all is extended before
    # mass_total_active is computed in ablateAll(), MetSimErosion.py:1300-1307) - a single-tick,
    # one-time effect, accepted for the same reason a grain's own spawn-tick mass is already not
    # separately special-cased. ---
    for daughter, (tick_idx_d, v_d_arr, m_d_arr, len_d_arr, h_d_arr, lum_d_arr, q_d_arr, dp_d_arr,
            tau_d_arr, ng_d) in zip(daughter_fragments, daughter_evals):
        if len(tick_idx_d) == 0:
            continue

        is_complex_d = daughter["complex_id"] is not None
        lum_w_d = lum_d_arr*ng_d

        _binAdd(lum_total, tick_idx_d, lum_w_d)
        _binAdd(edens_total, tick_idx_d, q_d_arr*ng_d)
        # Excludes this daughter's own death/disruption tick (unchanged upstream - see the main
        # fragment's own mass_total_active comment above), now weighted by n_grains to match the
        # reference's own fix (a daughter bin can represent multiple identical physical fragments).
        _binAdd(mass_total_active, tick_idx_d[:-1], m_d_arr[:-1]*ng_d)

        tau_lum_w_d = tau_d_arr*lum_w_d
        _binAdd(tau_total_num, tick_idx_d, tau_lum_w_d)
        # lum_eroded/tau_eroded EXCLUDE complex (Stage 5 F/EF) daughters entirely, regardless of
        # fragmentation_show_individual_lcs - matches ablateAll()'s own
        # `if frag.complex: ... else: luminosity_eroded += ...` branching (MetSimErosion.py:955-
        # 972): a complex fragment never reaches the lum_eroded accumulation at all: that flag only
        # gates the SEPARATE per-entry breakdown this module deliberately doesn't implement (see
        # this function's own docstring).
        if not is_complex_d:
            _binAdd(lum_eroded_raw, tick_idx_d, lum_w_d)
            _binAdd(tau_eroded_num, tick_idx_d, tau_lum_w_d)

        cand_idx_parts.append(tick_idx_d)
        cand_lumw_parts.append(lum_w_d)
        cand_h_parts.append(h_d_arr)
        cand_len_parts.append(len_d_arr)
        cand_v_parts.append(v_d_arr)
        cand_dp_parts.append(dp_d_arr)

    for (gidx, v_g, m_g, h_g, lum_g, q_g, dp_g, len_g, ng_g, last_g, gid_g), is_complex_arr in zip(
            grain_batches, batch_is_complex):
        if len(gidx) == 0:
            continue

        lum_w = lum_g*ng_g

        _binAdd(lum_total, gidx, lum_w)
        _binAdd(edens_total, gidx, q_g*ng_g)
        # Excludes each grain's own death tick (eligible = ~last_g, computed here so both this and
        # the candidate-table use below share it) - unchanged upstream, see the main fragment's own
        # mass_total_active comment above - now weighted by n_grains to match the reference's own
        # fix.
        eligible = ~last_g
        _binAdd(mass_total_active, gidx[eligible], m_g[eligible]*ng_g[eligible])

        if const.lum_eff_type == 0:
            tau_g = np.full(len(v_g), const.lum_eff/100.0)
        else:
            tau_g = np.array([luminousEfficiency(const.lum_eff_type, const.lum_eff, float(v_i),
                float(m_i)) for v_i, m_i in zip(v_g, m_g)])
        tau_lum_w = tau_g*lum_w

        _binAdd(tau_total_num, gidx, tau_lum_w)
        # lum_eroded/tau_eroded EXCLUDE complex-sourced grains - Stage 5 "D" dust release, or
        # grains spawned by a complex EF daughter's own ongoing erosion, both inherit complex=True
        # from their parent via spawn_child()'s dict copy (MetSimErosion.py:373-384) exactly like
        # any other attribute - see _batchAndStepGrainSpecs()'s own docstring for how this is
        # tracked per-grain through the (K, sigma) batching.
        not_complex = ~is_complex_arr[gid_g]
        _binAdd(lum_eroded_raw, gidx[not_complex], lum_w[not_complex])
        _binAdd(tau_eroded_num, gidx[not_complex], tau_lum_w[not_complex])

        cand_idx_parts.append(gidx)
        cand_lumw_parts.append(lum_w)
        cand_h_parts.append(h_g)
        cand_len_parts.append(len_g)
        cand_v_parts.append(v_g)
        cand_dp_parts.append(dp_g)

    # lum_eroded/tau_eroded are now always populated (matching the reference's own fix removing the
    # fragmentation_show_individual_lcs gate on this aggregate total - that flag now only gates a
    # separate per-entry breakdown this module doesn't implement) - the complex-fragment exclusion
    # in the daughter/grain loops above applies regardless of the flag, matching the reference.
    lum_eroded_col = lum_eroded_raw
    tau_eroded_col = np.where(lum_eroded_raw > 0,
        tau_eroded_num/np.maximum(lum_eroded_raw, 1e-300), 0.0)

    tau_total_col = np.where(lum_total > 0, tau_total_num/np.maximum(lum_total, 1e-300), 0.0)

    cand_idx = np.concatenate(cand_idx_parts) if cand_idx_parts else np.array([], dtype=np.int64)
    cand_lumw = np.concatenate(cand_lumw_parts) if cand_lumw_parts else np.array([])
    cand_h = np.concatenate(cand_h_parts) if cand_h_parts else np.array([])
    cand_len = np.concatenate(cand_len_parts) if cand_len_parts else np.array([])
    cand_v = np.concatenate(cand_v_parts) if cand_v_parts else np.array([])
    cand_dp = np.concatenate(cand_dp_parts) if cand_dp_parts else np.array([])

    brightest_winner = _scatterArgmaxGroupby(cand_idx, cand_lumw, n_ticks)
    leading_winner = _scatterArgmaxGroupby(cand_idx, cand_len, n_ticks)

    def _gather(arr, winner):
        out = np.zeros(n_ticks)
        valid = winner >= 0
        out[valid] = arr[winner[valid]]
        return out

    brightest_height = _gather(cand_h, brightest_winner)
    brightest_length = _gather(cand_len, brightest_winner)
    brightest_vel = _gather(cand_v, brightest_winner)

    leading_height = _gather(cand_h, leading_winner)
    leading_length = _gather(cand_len, leading_winner)
    leading_vel = _gather(cand_v, leading_winner)
    leading_dp = _gather(cand_dp, leading_winner)

    t_ticks = (np.arange(n_ticks) + 1)*dt

    results_list = [
        [t_ticks[i], lum_total[i], lum_main_col[i], lum_eroded_col[i], edens_total[i],
         tau_total_col[i], tau_main_col[i], tau_eroded_col[i], brightest_height[i],
         brightest_length[i], brightest_vel[i], leading_height[i], leading_length[i],
         leading_vel[i], leading_dp[i], mass_total_active[i], main_mass_col[i], main_h_col[i],
         main_len_col[i], main_v_col[i], main_dp_col[i]]
        for i in range(n_ticks)
    ]

    has_main_row = n_ticks_main > 0

    frag_main = Fragment()
    frag_main.const = const
    frag_main.id = 0
    frag_main.K = segments[-1]["K"]
    frag_main.m_init = const.m_init
    # Disruption sets frag.m = 0 explicitly right before killFragment() (MetSimErosion.py:1074-1075)
    # - unlike a natural death, where the fragment's own last real mass is left in place. This is
    # the ONLY attribute disruption zeroes; v/h/length/lum/q/dyn_press below stay at their real
    # pre-disruption values (matching m_main[-1] etc., which are NOT zeroed - those still feed
    # main_mass_col's own last row correctly, exactly as the reference's results_list row for that
    # same tick shows the real pre-disruption mass, captured earlier in ablateAll()'s own per-tick
    # sequence than the frag.m=0 assignment - see this function's docstring).
    frag_main.m = 0.0 if disrupted else (float(m_main[-1]) if has_main_row else const.m_init)
    # 3 segments only happens once erosion_height_change was reached (see
    # _buildMainFragmentSegments()) - the only point at which rho actually changes.
    frag_main.rho = const.erosion_rho_change if len(segments) == 3 else const.rho
    frag_main.sigma = segments[-1]["sigma_own"]
    frag_main.v = float(v_main[-1]) if has_main_row else const.v_init
    frag_main.zenith_angle = const.zenith_angle
    frag_main.vv = -frag_main.v*math.cos(const.zenith_angle)
    frag_main.vh = frag_main.v*math.sin(const.zenith_angle)
    frag_main.length = float(length_main[-1]) if has_main_row else 0.0
    frag_main.h = float(h_main[-1]) if has_main_row else const.h_init
    frag_main.lum = float(lum_main[-1]) if has_main_row else 0.0
    frag_main.q = float(q_main[-1]) if has_main_row else 0.0
    frag_main.dyn_press = float(dp_main[-1]) if has_main_row else 0.0
    frag_main.erosion_enabled = False
    frag_main.disruption_enabled = False
    frag_main.active = False
    frag_main.n_grains = 1
    frag_main.main = True
    frag_main.grain = False
    frag_main.complex = False
    frag_main.complex_id = None
    frag_main.n_queries = n_queries_total

    const.total_time = float(t_ticks[-1])
    const.n_active = 0
    const.main_bottom_ht = float(np.min(h_main)) if has_main_row else const.h_init
    const.main_mass_exhaustion_ht = frag_main.h

    wake_results = [None]*len(results_list)

    return frag_main, results_list, wake_results


def runSimulation(const, compute_wake=False):
    """ Run the ablation simulation using the analytic alpha-beta engine (AnalyticTrajectory).

    STAGE 3 SCOPE: erosion_on may now be True (Stage 3, see _runSimulationErosion()) or False
    (Stage 2, single non-eroding body, handled inline below) - disruption_on and fragmentation_on
    must still both be False (Stages 4/5 of the implementation plan add those; this raises
    NotImplementedError otherwise rather than silently producing wrong results). compute_wake is
    Stage 8.

    Architecture note: unlike _runSimulationRK4Reference()/ablateAll() above, there is no
    per-timestep loop here. AnalyticTrajectory already has the fragment's entire trajectory in
    closed form the moment it's built, so the whole dt-spaced output grid is evaluated in one
    vectorized pass - looping per tick and querying the trajectory once per iteration would forfeit
    most of the speedup this engine exists for (see VelocitySpline's measured timings).

    Matches _runSimulationRK4Reference()'s exact output conventions, verified against the original
    source rather than assumed:
      - results_list rows correspond to t = dt, 2*dt, 3*dt, ... (const.total_time is incremented
        BEFORE being recorded each step in the original, so t=0 itself is never a row - only used
        as the trajectory's starting condition).
      - The main_height/leading_frag_height/etc. columns report the curved-Earth + gravity-drop
        corrected height (reportedHeightAt()), matching frag.h - not the flat-earth heightRealAt()
        AnalyticTrajectory solves the physics in.
      - Atmosphere density for the rate terms (mass_loss_rate, deceleration_rate - see below) uses
        the SAME atm_height_fn-corrected height the final (iteratively-refined) trajectory was
        actually built with, multiplied by that correction's local slope (a chain-rule/Jacobian
        term this trajectory's time domain being parametrized by flat height, not corrected height,
        makes necessary - see the rate-term computation below for the full derivation). dyn_press
        uses the corrected density WITHOUT the slope factor, since it is an instantaneous state
        quantity (rho*v^2), not a derivative reconstruction. See the implementation plan's Stage 2c
        write-up for the full derivation and validation numbers (peak-luminosity error -0.09% to
        -3.06% across the validated geometry grid, down from 211.6% before this fix).
      - The luminosity/ionization/dynamic-pressure formulas mix a rate term (mass_loss_ablation,
        deceleration_total - computed from the PREVIOUS tick's state in the original's RK4
        stepping, before frag.m/frag.v/frag.h are updated, MetSimErosion.py:1339,1358 vs.
        :1399-1415) with position/velocity/mass terms (CURRENT tick, post-update). An initial
        version of this function evaluated every term at the same current tick, assuming the
        difference would be a small, bounded, dt-order effect - checked directly against a real
        run rather than left as an assumption, and that assumption was wrong: because the body is
        always descending, "current" is always denser than "previous", making it a systematic (not
        random) bias that grew to ~1.6% even in an otherwise well-matched case. Fixed by evaluating
        the rate terms at t-dt, matching the original exactly (verified: 0.34-1.63% -> 0.01-0.04%).

    Arguments:
        const: [Constants]

    Keyword arguments:
        compute_wake: [bool] Not yet supported (Stage 8) - must be False.

    Return:
        (frag_main, results_list, wake_results): same contract as _runSimulationRK4Reference(),
        with one additive extension: frag_main.n_queries (int) - the number of individual
        (fragment, time) state queries this run made against the final AnalyticTrajectory (see
        AnalyticTrajectory.n_queries's own comment) - see the implementation plan's "Query pattern
        per fragment/segment" section for why this is worth tracking (it's the number that
        determines whether building VelocitySpline per segment actually pays off, not a guess).
        Purely additive - ignored by any caller that doesn't know about it, so this doesn't affect
        drop-in compatibility with _runSimulationRK4Reference()'s Fragment objects.
    """

    if compute_wake:
        raise NotImplementedError("compute_wake=True is not yet supported by the analytic engine "
            "(Stage 8 of the implementation plan).")

    # disruption_on/fragmentation_on route here too (Stages 4/5), even with erosion_on=False - a
    # fragment can disrupt or complex-fragment without ever continuously eroding, and
    # _runSimulationErosion() handles all three independently (each guarded by its own
    # const.erosion_on/const.disruption_on/const.fragmentation_on check internally). Stage 5's
    # type "A" (retroactive re-splitting of already-alive grains/daughters) and upward_only entries
    # are not yet implemented - see _buildMainFragmentSegmentsWithFragmentation()'s and
    # _applyFragmentationEntry()'s own docstrings for the precise current scope.
    if const.erosion_on or const.disruption_on or const.fragmentation_on:
        return _runSimulationErosion(const)

    K = const.gamma*const.shape_factor*const.rho**(-2/3.0)
    sin_slope = math.cos(const.zenith_angle)

    atm_map = AtmEquivHeightMap(const.dens_co, const.h_init)

    # How far (in normalized velocity) AnalyticTrajectory needs to tabulate this fragment's
    # trajectory - see v_n_floor's docstring for why this must be a physically-motivated bound
    # (comfortably past the real stopping condition) rather than an arbitrary near-zero epsilon.
    v_n_floor = max(0.01, 0.5*const.v_kill/const.v_init)

    # Hard backstop on top of v_n_floor - see h_real_floor's docstring. For shallow/slowly-
    # decelerating segments (small sin_slope, low sigma_eff) v_n_floor alone can demand hundreds of
    # km of flat path length, which pushes reportedHeightAt()'s curvature formula into a
    # non-monotonic regime and corrupts the iterative refinement below (found via a genuine crash
    # on an 85-degree-entry case). const.h_kill is the real physical stopping height the fragment
    # can never survive past regardless of v_n - the margin below it is generous (not tight)
    # because this is a coarse backstop, not the actual kill check (that happens later, against
    # h_reported, at full precision).
    h_real_floor = const.h_kill - max(0.05*(const.h_init - const.h_kill), 5000.0)

    def _buildTrajectory(h_real_floor):
        """ Thin wrapper around the general _buildRefinedTrajectory() (Stage 3), fixing this
        function's segment to the whole simulation's own start (t=0, h_init, v_init, m_init,
        length_start=0) - see _buildRefinedTrajectory()'s docstring for the full reasoning. """

        return _buildRefinedTrajectory(K, const.sigma, const.m_init, const.v_init, const.h_init,
            sin_slope, atm_map, t_start=0.0, length_start=0.0, const=const, v_n_floor=v_n_floor,
            h_real_floor=h_real_floor)

    def _estimateTimeAtHKill(traj):
        """ Rough estimate of when REPORTED height crosses h_kill, used only to size the initial
        candidate-tick search window below - not correctness-critical, since the retry loop
        handles any misestimate by doubling/rebuilding. Must use reported height (matching the
        actual kill_mask condition, h_reported < const.h_kill, further below), not flat height:
        flat height can cross h_kill well before reported height does for shallow, long-surviving
        trajectories (reportedHeightAt()'s curvature-inflation term lags flat height's descent -
        see the floor_extension comment below), which would otherwise systematically
        under-estimate the time needed. Confirmed harmless in practice even before this fix (traced
        directly on a case that needed floor_extension retries: the under-estimate never bottlenecked
        anything, since n_candidate's own 1.5x margin plus the retry loop already covered the gap) -
        fixed anyway to remove a real height/reported-height unit mismatch, not because it was
        causing wrong output.
        """
        h_reported_grid = reportedHeightAt(traj, traj._t_grid, const.h_init, const.zenith_angle)
        return float(np.interp(const.h_kill, h_reported_grid[::-1], traj._t_grid[::-1]))

    dt = const.dt
    traj, atm_height_fn = _buildTrajectory(h_real_floor)

    # Estimate an upper bound on how many output ticks are needed (from AnalyticTrajectory's own
    # already-built grid - a cheap lookup, not a new computation), then expand if the fragment
    # turns out to survive longer than the estimate. A generous margin, not precision - the real
    # kill check below is exact.
    t_est = _estimateTimeAtHKill(traj)
    n_candidate = max(int(t_est/dt*1.5) + 20, 100)

    # Outer loop: if the trajectory's own tabulated domain (bounded by v_n_floor/h_real_floor)
    # turns out not to reach far enough, rebuild with a substantially deeper h_real_floor and
    # retry, rather than only ever expanding n_candidate against the SAME (clipped) trajectory -
    # found to be a real, non-extreme gap: reportedHeightAt()'s curvature-inflation term (its
    # quadratic-in-length correction relative to flat height) can make reported height lag flat
    # height's descent enough that the initial fixed-margin h_real_floor undershoots for fragments
    # that survive long, moderately-shallow (roughly 70-80 degree) flights - confirmed directly on
    # a 75deg/dense/low-ablation case that needed flat height to reach ~14km before reported height
    # dropped to a 60km h_kill, versus the initial floor's ~52km. Growing the extra depth below
    # h_kill MULTIPLICATIVELY (not additively) because that curvature-inflation term scales with
    # length^2, so a fixed-size extension could need many retries to catch up.
    MAX_FLOOR_EXTENSIONS = 6
    kill_idx = None

    for floor_extension in range(MAX_FLOOR_EXTENSIONS + 1):

        if floor_extension > 0:
            extra_depth = max(0.05*(const.h_init - const.h_kill), 5000.0)*(4**floor_extension)
            h_real_floor = const.h_kill - extra_depth
            traj, atm_height_fn = _buildTrajectory(h_real_floor)
            t_est = _estimateTimeAtHKill(traj)
            n_candidate = max(int(t_est/dt*1.5) + 20, 100)

        hit_ceiling = False

        for _ in range(10):

            t_ticks = (np.arange(n_candidate) + 1)*dt

            v = traj.velocityNormedAt(t_ticks)*const.v_init
            m = traj.massAt(t_ticks)
            length = traj.lengthAt(t_ticks)
            h_reported = reportedHeightAt(traj, t_ticks, const.h_init, const.zenith_angle)

            kill_mask = ((m <= const.m_kill) | (v < const.v_kill) | (h_reported < const.h_kill)
                | ((const.len_kill > 0) & (length > const.len_kill)))

            if np.any(kill_mask):
                kill_idx = int(np.argmax(kill_mask))
                break

            if t_ticks[-1] >= traj.t_hi:
                # Queries past t_hi are clipped to the same endpoint value forever (see
                # AnalyticTrajectory.velocityNormedAt()/heightRealAt()) - further n_candidate
                # doubling against this same trajectory cannot find a kill condition that isn't
                # already present at t_hi. Break out to rebuild with a deeper floor instead.
                hit_ceiling = True
                break

            n_candidate *= 2

        if kill_idx is not None:
            break

        if not hit_ceiling:
            raise RuntimeError("Could not find a kill condition within 10 search-window "
                "expansions of a fixed trajectory domain - check Constants for an unreachable "
                "m_kill/v_kill/h_kill configuration.")

        # else: hit_ceiling - loop continues to the next (deeper) floor_extension.

    else:
        raise RuntimeError(
            f"Could not find a kill condition even after extending the trajectory's tabulated "
            f"domain {MAX_FLOOR_EXTENSIONS} times - check Constants for an unreachable "
            "m_kill/v_kill/h_kill configuration.")

    sl = slice(0, kill_idx + 1)
    t_ticks = t_ticks[sl]

    # v/m/length/h_reported were already computed once inside the kill-search loop above (needed
    # there to evaluate kill_mask at each candidate window size), but _evaluateSegment() is called
    # again here for the FINAL, sliced t_ticks rather than reusing/re-slicing those arrays - this
    # also gives lum/q/dyn_press/tau in the same call, and (from Stage 3 onward) is the exact same
    # function every erosion segment and grain uses, so there is only one place this logic lives.
    # See _evaluateSegment()'s docstring for the full rate-term/luminosity derivation (staggered
    # previous-tick rate terms, the atm_height_fn chain-rule slope, sigma_own vs sigma_eff, etc.).
    v, m, length, h_reported, lum, q, dyn_press, tau = _evaluateSegment(traj, atm_height_fn, t_ticks,
        K, const.sigma, const.v_init, 0.0, const)

    # Single fragment: total == main, leading == brightest == main at EVERY row including the last
    # one - main's own death tick still counts as a valid brightest_*/leading_frag_* candidate now,
    # matching the reference's own fix (its brightest-tracking block moved before the per-fragment
    # kill-check, and its leading-fragment scan moved to a tick-start snapshot taken before any
    # fragment active that tick can die - confirmed directly: a real single-body run's own last row
    # now reports its real final height/velocity there, not 0.0/None as before that fix).
    zeros = np.zeros_like(lum)
    brightest_h = h_reported
    brightest_len = length
    brightest_v = v
    leading_h = h_reported
    leading_len = length
    leading_v = v
    leading_dp = dyn_press

    results_list = [
        [t_ticks[i], lum[i], lum[i], zeros[i], q[i], tau[i], tau[i], zeros[i], brightest_h[i],
         brightest_len[i], brightest_v[i], leading_h[i], leading_len[i], leading_v[i],
         leading_dp[i], m[i], m[i], h_reported[i], length[i], v[i], dyn_press[i]]
        for i in range(len(t_ticks))
    ]

    frag_main = Fragment()
    frag_main.const = const
    frag_main.id = 0
    frag_main.K = K
    frag_main.m_init = const.m_init
    frag_main.m = float(m[-1])
    frag_main.rho = const.rho
    frag_main.sigma = const.sigma
    frag_main.v = float(v[-1])
    frag_main.zenith_angle = const.zenith_angle
    frag_main.vv = -frag_main.v*math.cos(const.zenith_angle)
    frag_main.vh = frag_main.v*math.sin(const.zenith_angle)
    frag_main.h_grav_drop_total = float(traj.gravityDropAt(t_ticks[-1]))
    frag_main.length = float(length[-1])
    frag_main.h = float(h_reported[-1])
    frag_main.lum = float(lum[-1])
    frag_main.q = float(q[-1])
    frag_main.dyn_press = float(dyn_press[-1])
    frag_main.erosion_enabled = False
    frag_main.disruption_enabled = False
    frag_main.active = False
    frag_main.n_grains = 1
    frag_main.main = True
    frag_main.grain = False
    frag_main.complex = False
    frag_main.complex_id = None
    # Stage 2d instrumentation - see runSimulation()'s docstring "Return:" section and
    # AnalyticTrajectory.n_queries's own comment. Only the FINAL trajectory (after iterative
    # refinement and any floor_extension rebuilds) contributes: discarded intermediate builds only
    # ever get queried through heightRealAt()/lengthAt()/gravityDropAt() during refinement (for
    # reportedHeightAt()), never through velocityNormedAt() - see reportedHeightAt()'s own
    # implementation - so they never touch this counter regardless.
    frag_main.n_queries = traj.n_queries

    const.total_time = float(t_ticks[-1])
    const.n_active = 0
    const.total_fragments = 1
    const.main_bottom_ht = float(np.min(h_reported))

    wake_results = [None]*len(results_list)

    return frag_main, results_list, wake_results


def energyReceivedBeforeErosion(const, lam=1.0):
    """ Compute the energy the meteoroid receive prior to erosion, assuming no major mass loss occured. 
    
    Arguments:
        const: [Constants]

    Keyword arguments:
        lam: [float] Heat transfter coeff. 1.0 by default.

    Return:
        (es, ev):
            - es: [float] Energy received per unit cross-section (J/m^2)
            - ev: [float] Energy received per unit mass (J/kg).

    """

    # Integrate atmosphere density from the beginning of simulation to beginning of erosion.
    dens_integ = scipy.integrate.quad(atmDensityPoly, const.erosion_height_start, const.h_init, \
        args=(const.dens_co))[0]

    # Compute the energy per unit cross-section
    es = 1/2*lam*(const.v_init**2)*dens_integ/np.cos(const.zenith_angle)

    # Compute initial shape-density coefficient
    k = const.gamma*const.shape_factor*const.rho**(-2/3.0)

    # Compute the energy per unit mass
    ev = es*k/(const.gamma*const.m_init**(1/3.0))

    return es, ev


if __name__ == "__main__":

    import matplotlib.pyplot as plt


    from wmpl.Utils.AtmosphereDensity import fitAtmPoly
    from wmpl.Utils.TrajConversions import date2JD


    # Show wake
    show_wake = False


    # Init the constants
    const = Constants()

    # Fit atmosphere density polynomial for the given location and time on Earth, and the range of simulation 
    # heights
    const.dens_co = fitAtmPoly(
        np.radians(45.3), # lat +N
        np.radians(18.1), # lon +E
         70000, # height_min in m
        180000, # height_max in m (this needs to be the same as the beginning of simulation)
        date2JD(2020, 4, 20, 16, 15, 0) # Julian date
        )


    ### Set some physical parameters of the meteoroid ###

    # Set the power of a zero magnitude meteor for silicon sensors (W)
    const.P_0m = 1210

    # Initial mass (kg)
    const.m_init = 1e-5

    # Bulk density (kg/m^3)
    const.rho = 300

    # Grain density (kg/m^3)
    const.rho_grain = 3000

    # Initial velocity (m/s)
    const.v_init = 45000

    # Ablation coefficient (kg/MJ or s^2/km^2)
    const.sigma = 0.023/1e6

    # Zenith angle = 90 - elevation angle
    const.zenith_angle = math.radians(45)

    # Grain bulk density (kg/m^3) - used for erosion, 3000 is used for faint meteors and 3500 for fireballs
    const.rho_grain = 3000

    # Luminous efficiency type (5 for faint meteors, 7 for fireballs)
    const.lum_eff_type = 5



    # Toggle erosion on/off
    const.erosion_on = True

    # Bins per order of magnitude mass (2 is enough for firebals and a large range of masses, 
    # 5 for fainter meteors)
    const.erosion_bins_per_10mass = 5
    
    # Height at which the erosion starts (meters)
    const.erosion_height_start = 102000

    # Erosion coefficient (kg/MJ or s^2/km^2)
    const.erosion_coeff = 0.33/1e6

   
    # Height at which the erosion coefficient changes - for no change is should be below the end height 
    # (meters)
    const.erosion_height_change = 0


    # Grain mass distribution index
    const.erosion_mass_index = 2.0

    # Mass range for grains (kg)
    const.erosion_mass_min = 1.0e-11
    const.erosion_mass_max = 5.0e-10

    # Disable disruption
    const.disruption_on = False

    ### ###



    # Run the ablation simulation
    frag_main, results_list, wake_results = runSimulation(const, compute_wake=show_wake)



    ### ANALYZE RESULTS ###


    # System limiting magnitude (used for plotting the wake)
    lim_mag = 6.0

    # Unpack the results
    results_list = np.array(results_list).astype(np.float64)
    time_arr, luminosity_arr, luminosity_main_arr, luminosity_eroded_arr, electron_density_total_arr, \
        tau_total_arr, tau_main_arr, tau_eroded_arr, brightest_height_arr, brightest_length_arr, \
        brightest_vel_arr, leading_frag_height_arr, leading_frag_length_arr, leading_frag_vel_arr, \
        leading_frag_dyn_press_arr, mass_total_active_arr, main_mass_arr, main_height_arr, main_length_arr, \
        main_vel_arr, main_dyn_press_arr = results_list.T


    # Calculate absolute magnitude (apparent @100km) from given luminous intensity
    abs_magnitude = -2.5*np.log10(luminosity_arr/const.P_0m)

    # plt.plot(abs_magnitude, brightest_height_arr/1000)
    # plt.gca().invert_xaxis()
    # plt.show()

    plt.plot(time_arr, abs_magnitude)
    plt.gca().invert_yaxis()

    plt.xlabel("Time (s)")
    plt.ylabel("Absolulte magnitude")

    plt.show()



    # Plot mass loss
    plt.plot(time_arr, 1000*mass_total_active_arr)
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (g)")
    plt.show()



    # Plot length vs time
    plt.plot(brightest_length_arr[:-1]/1000, brightest_height_arr[:-1]/1000, label='Brightest bin')
    plt.plot(leading_frag_length_arr[:-1]/1000, leading_frag_height_arr[:-1]/1000, label='Leading fragment')

    plt.ylabel("Height (km)")
    plt.xlabel("Length (km)")

    plt.legend()


    plt.show()


    # Plot the wake animation
    if show_wake and wake_results:
        
        plt.ion()
        fig, ax = plt.subplots(1,1)

        # Determine the plot upper limit
        max_lum_wake = max([max(wake.wake_luminosity_profile) for wake in wake_results if wake is not None])

        

        for wake, abs_mag in zip(wake_results, abs_magnitude):

            if wake is None:
                continue

            # Skip points below the limiting magnitude
            if (abs_mag > lim_mag) or np.isnan(abs_mag):
                continue

            plt.cla()
                
            # Plot the wake profile
            ax.plot(wake.length_array, wake.wake_luminosity_profile)

            # Plot the location of grains
            ax.scatter(wake.length_points, wake.luminosity_points/10, c='k', s=10*wake.luminosity_points/np.max(wake.luminosity_points))

            plt.ylim([0, max_lum_wake])

            plt.pause(2*const.dt)

            fig.canvas.draw()

        plt.ioff()
        plt.clf()
        plt.close()
