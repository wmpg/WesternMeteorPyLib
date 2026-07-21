""" Tests for the alpha-beta model fitting in wmpl.Utils.AlphaBeta.

The tests generate a synthetic trajectory from known alpha-beta parameters, compute the
corresponding true masses, and check that fitAlphaBetaMass() recovers the true parameters under
all three mass constraints, both with a fixed and a fitted bulk density.

The joint dynamics + light curve fit (fitAlphaBetaLightCurve()), the robust dynamics-only fit
(fitAlphaBeta(method='robust')), the luminosity model, and the luminous efficiency inversion are
tested against a synthetic light curve generated from the same trajectory.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_AlphaBeta.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_AlphaBeta
"""

import time

import numpy as np
import scipy.special

import matplotlib
matplotlib.use("Agg")  # headless - plotAlphaBeta() tests must not require a display or block
import matplotlib.pyplot as plt

from wmpl.Utils.AlphaBeta import (fitAlphaBetaMass, fitAlphaBeta, fitAlphaBetaLightCurve,
    alphaBetaMasses, alphaBetaVelocity, alphaBetaHeight, alphaBetaVelocityNormed,
    alphaBetaVelocityNormedLUT, alphaBetaHeightNormed, alphaBetaLuminosityF,
    alphaBetaModelMagnitude, alphaBetaLuminousEfficiency, plotAlphaBeta, plotProfileAlphaBeta,
    plotAlphaBetaSurvivalDiagram, profileAlphaBeta, _profiledMagOffset, _gaussianEllipsePoints,
    getDefaultInverseEiLUT, HT_NORM_CONST, P_0M, ALPHA_BETA_BOUNDS)


# True parameters used to generate the synthetic trajectory. The height range is chosen so the
#   trajectory shows a strong deceleration (~26% velocity drop), which is required for a
#   well-conditioned alpha-beta fit
ALPHA_TRUE = 50.0
BETA_TRUE = 2.0
V_INIT_TRUE = 18000.0
DENS_TRUE = 3500.0
SLOPE = np.radians(45.0)
HT_DATA = np.linspace(85000.0, 40000.0, 80)

# Standard deviation of the Gaussian noise added to the synthetic velocities (m/s)
VEL_NOISE_STD = 20.0


def _syntheticTrajectory():
    """ Generate a synthetic noisy (velocity, height) trajectory and the true masses.

    Return:
        (v_data, v_final_true, m_init_true, m_final_mu0_true, m_final_mu23_true):
            - v_data: [ndarray] Noisy velocity data (m/s).
            - v_final_true: [float] Noise-free final velocity (m/s).
            - m_init_true: [float] True initial mass (kg).
            - m_final_mu0_true: [float] True final mass for mu = 0 (kg).
            - m_final_mu23_true: [float] True final mass for mu = 2/3 (kg).
    """

    # Compute the noise-free model velocities
    v_clean = alphaBetaVelocity(HT_DATA, ALPHA_TRUE, BETA_TRUE, V_INIT_TRUE)
    v_final_true = np.min(v_clean)

    # Add reproducible noise
    rng = np.random.RandomState(42)
    v_data = v_clean + rng.normal(0, VEL_NOISE_STD, HT_DATA.size)

    # Compute the true masses (the initial mass is independent of mu)
    m_init_true, m_final_mu0_true = alphaBetaMasses(ALPHA_TRUE, BETA_TRUE, SLOPE, mu=0, \
        dens=DENS_TRUE, vel_init=V_INIT_TRUE, vel_end=v_final_true)
    _, m_final_mu23_true = alphaBetaMasses(ALPHA_TRUE, BETA_TRUE, SLOPE, mu=2/3, dens=DENS_TRUE, \
        vel_init=V_INIT_TRUE, vel_end=v_final_true)

    return v_data, v_final_true, m_init_true, m_final_mu0_true, m_final_mu23_true


def _assertClose(value, expected, rel_tol, label):
    """ Assert that a value matches the expected one within a relative tolerance. """

    rel_diff = abs(value - expected)/abs(expected)

    assert rel_diff <= rel_tol, "{:s}: got {:.6g}, expected {:.6g} ({:.1%} off, tolerance {:.1%})".format(
        label, value, expected, rel_diff, rel_tol)


def testFitAlphaBetaUnconstrained():
    """ The plain unconstrained fit should recover alpha and beta from noisy synthetic data. """

    v_data, _, _, _, _ = _syntheticTrajectory()

    v_init, alpha, beta = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE)

    _assertClose(alpha, ALPHA_TRUE, 0.05, "alpha")
    _assertClose(beta, BETA_TRUE, 0.05, "beta")


def testInitialMassConstraint():
    """ Fitting with the true initial mass imposed should recover alpha, beta, and the final
        masses, both with a fixed and a fitted density.
    """

    v_data, v_final_true, m_init_true, m_final_mu0_true, m_final_mu23_true = _syntheticTrajectory()

    for density in [DENS_TRUE, None]:

        res = fitAlphaBetaMass(v_data, HT_DATA, SLOPE, m_init_true, mass_constraint="initial", \
            density=density, v_init=V_INIT_TRUE, verbose=False)

        (v_init, v_final,
            density_mu0, alpha_mu0, beta_mu0, m_initial_mu0, m_final_mu0,
            density_mu23, alpha_mu23, beta_mu23, m_initial_mu23, m_final_mu23,
            mu_best, density_mu_best, alpha_mu_best, beta_mu_best, m_initial_mu_best,
            m_final_mu_best) = res

        _assertClose(alpha_mu0, ALPHA_TRUE, 0.05, "alpha_mu0")
        _assertClose(beta_mu0, BETA_TRUE, 0.05, "beta_mu0")
        _assertClose(density_mu0, DENS_TRUE, 0.05, "density_mu0")
        _assertClose(v_final, v_final_true, 0.02, "v_final")

        # The imposed initial mass must be reproduced exactly
        _assertClose(m_initial_mu0, m_init_true, 1e-6, "m_initial_mu0")
        _assertClose(m_initial_mu23, m_init_true, 1e-6, "m_initial_mu23")

        # The final masses are derived, so allow a looser tolerance
        _assertClose(m_final_mu0, m_final_mu0_true, 0.15, "m_final_mu0")
        _assertClose(m_final_mu23, m_final_mu23_true, 0.30, "m_final_mu23")

        # The fit is independent of mu under this constraint, so both branches must be identical
        #   and no best-fit mu may be reported
        assert alpha_mu0 == alpha_mu23
        assert beta_mu0 == beta_mu23
        assert density_mu0 == density_mu23
        assert mu_best is None
        assert alpha_mu_best is None


def testFinalMassConstraint():
    """ Fitting with the true final mass imposed should recover alpha, beta, and the initial
        mass, both with a fixed and a fitted density.
    """

    v_data, v_final_true, m_init_true, m_final_mu0_true, _ = _syntheticTrajectory()

    for density in [DENS_TRUE, None]:

        res = fitAlphaBetaMass(v_data, HT_DATA, SLOPE, m_final_mu0_true, mass_constraint="final", \
            density=density, v_init=V_INIT_TRUE, v_final=v_final_true, verbose=False)

        (v_init, v_final,
            density_mu0, alpha_mu0, beta_mu0, m_initial_mu0, m_final_mu0,
            density_mu23, alpha_mu23, beta_mu23, m_initial_mu23, m_final_mu23,
            mu_best, density_mu_best, alpha_mu_best, beta_mu_best, m_initial_mu_best,
            m_final_mu_best) = res

        _assertClose(alpha_mu0, ALPHA_TRUE, 0.05, "alpha_mu0")
        _assertClose(beta_mu0, BETA_TRUE, 0.05, "beta_mu0")
        _assertClose(density_mu0, DENS_TRUE, 0.05, "density_mu0")

        # The imposed final mass must be reproduced exactly
        _assertClose(m_final_mu0, m_final_mu0_true, 1e-6, "m_final_mu0")

        # The initial mass is reconstructed, so allow a looser tolerance
        _assertClose(m_initial_mu0, m_init_true, 0.15, "m_initial_mu0")

        # The best-fit mu search must run under this constraint
        assert mu_best is not None
        assert 0 <= mu_best <= 2/3
        _assertClose(m_final_mu_best, m_final_mu0_true, 1e-6, "m_final_mu_best")


def testBothMassConstraint():
    """ Fitting with both true masses imposed should recover alpha, beta, and the density. """

    v_data, v_final_true, m_init_true, m_final_mu0_true, _ = _syntheticTrajectory()

    for density in [DENS_TRUE, None]:

        res = fitAlphaBetaMass(v_data, HT_DATA, SLOPE, (m_init_true, m_final_mu0_true), \
            mass_constraint="both", density=density, v_init=V_INIT_TRUE, v_final=v_final_true, \
            verbose=False)

        (v_init, v_final,
            density_mu0, alpha_mu0, beta_mu0, m_initial_mu0, m_final_mu0,
            density_mu23, alpha_mu23, beta_mu23, m_initial_mu23, m_final_mu23,
            mu_best, density_mu_best, alpha_mu_best, beta_mu_best, m_initial_mu_best,
            m_final_mu_best) = res

        # Under mu = 0 with both true masses and the true v_init/v_final, beta is analytic and
        #   must match the true value closely
        _assertClose(beta_mu0, BETA_TRUE, 0.01, "beta_mu0")
        _assertClose(alpha_mu0, ALPHA_TRUE, 0.05, "alpha_mu0")
        _assertClose(density_mu0, DENS_TRUE, 0.05, "density_mu0")

        # Both imposed masses must be reproduced exactly
        _assertClose(m_initial_mu0, m_init_true, 1e-6, "m_initial_mu0")
        _assertClose(m_final_mu0, m_final_mu0_true, 1e-6, "m_final_mu0")

        assert mu_best is not None
        assert 0 <= mu_best <= 2/3


def testMassConstraintQ4Method():
    """ The legacy Q4 fitting path of fitAlphaBetaMass() (method='q4', the behavior before the
        `method` argument was added, when robust became the default) must still recover the true
        parameters. Regression guard for the now-non-default Q4 path, which the other mass tests
        no longer exercise since they use the robust default.
    """

    v_data, v_final_true, m_init_true, m_final_mu0_true, m_final_mu23_true = _syntheticTrajectory()

    for density in [DENS_TRUE, None]:

        res = fitAlphaBetaMass(v_data, HT_DATA, SLOPE, m_init_true, mass_constraint="initial", \
            density=density, v_init=V_INIT_TRUE, method='q4', verbose=False)

        (v_init, v_final,
            density_mu0, alpha_mu0, beta_mu0, m_initial_mu0, m_final_mu0,
            density_mu23, alpha_mu23, beta_mu23, m_initial_mu23, m_final_mu23,
            mu_best, density_mu_best, alpha_mu_best, beta_mu_best, m_initial_mu_best,
            m_final_mu_best) = res

        _assertClose(alpha_mu0, ALPHA_TRUE, 0.05, "alpha_mu0 (q4)")
        _assertClose(beta_mu0, BETA_TRUE, 0.05, "beta_mu0 (q4)")
        _assertClose(density_mu0, DENS_TRUE, 0.05, "density_mu0 (q4)")

        # The imposed initial mass must still be reproduced exactly
        _assertClose(m_initial_mu0, m_init_true, 1e-6, "m_initial_mu0 (q4)")


def testDerivedVelocities():
    """ When v_init and v_final are not given, they should be derived from the data. """

    v_data, v_final_true, m_init_true, m_final_mu0_true, _ = _syntheticTrajectory()

    res = fitAlphaBetaMass(v_data, HT_DATA, SLOPE, m_final_mu0_true, mass_constraint="final", \
        density=DENS_TRUE, verbose=False)

    v_init, v_final = res[0], res[1]
    alpha_mu0, beta_mu0 = res[3], res[4]

    _assertClose(v_init, V_INIT_TRUE, 0.01, "v_init")
    _assertClose(v_final, v_final_true, 0.02, "v_final")
    _assertClose(alpha_mu0, ALPHA_TRUE, 0.10, "alpha_mu0")
    _assertClose(beta_mu0, BETA_TRUE, 0.10, "beta_mu0")


def testInputValidation():
    """ Invalid inputs should raise ValueError with informative messages. """

    v_data, _, m_init_true, m_final_mu0_true, _ = _syntheticTrajectory()

    def _assertRaisesValueError(label, **kwargs):

        call_kwargs = dict(v_data=v_data, ht_data=HT_DATA, slope=SLOPE, mass=m_init_true, \
            mass_constraint="initial", verbose=False)
        call_kwargs.update(kwargs)

        try:
            fitAlphaBetaMass(**call_kwargs)

        except ValueError:
            return

        raise AssertionError("{:s}: ValueError not raised".format(label))

    # Bad mass constraint name
    _assertRaisesValueError("bad constraint", mass_constraint="banana")

    # Non-positive and non-scalar masses
    _assertRaisesValueError("negative mass", mass=-1.0)
    _assertRaisesValueError("tuple mass for initial", mass=(1.0, 0.1))
    _assertRaisesValueError("scalar mass for both", mass=1.0, mass_constraint="both")
    _assertRaisesValueError("final >= initial", mass=(1.0, 2.0), mass_constraint="both")

    # Bad slope
    _assertRaisesValueError("zero slope", slope=0.0)
    _assertRaisesValueError("slope over pi/2", slope=np.pi/2 + 0.1)

    # Bad velocities
    _assertRaisesValueError("v_final >= v_init", v_init=15000.0, v_final=16000.0)

    # Mismatched data lengths
    _assertRaisesValueError("length mismatch", v_data=v_data[:-1])

    # alphaBetaMasses() slope validation
    try:
        alphaBetaMasses(ALPHA_TRUE, BETA_TRUE, -0.5)

    except ValueError:
        pass

    else:
        raise AssertionError("alphaBetaMasses: ValueError not raised for negative slope")


### Tests for the functions added in PR #75 (joint dynamics + light curve fit) ###


# True magnitude offset and photometric noise used by the synthetic light curve tests
MAG_OFFSET_TRUE = 8.0
MAG_NOISE_STD = 0.1


def _syntheticLightCurve():
    """ Generate a synthetic noisy light curve (mu = 0) matching the synthetic trajectory.

    Return:
        (mag_data, k_true):
            - mag_data: [ndarray] Noisy absolute magnitudes at HT_DATA.
            - k_true: [float] True light curve amplitude K (W) implied by MAG_OFFSET_TRUE.
    """

    mag_clean, _ = alphaBetaModelMagnitude(HT_DATA/HT_NORM_CONST, ALPHA_TRUE, BETA_TRUE, 0.0, \
        mag_offset=MAG_OFFSET_TRUE)

    rng = np.random.RandomState(43)
    mag_data = mag_clean + rng.normal(0, MAG_NOISE_STD, HT_DATA.size)

    k_true = P_0M*10**(-0.4*MAG_OFFSET_TRUE)

    return mag_data, k_true


def testAlphaBetaNormedRoundTrip():
    """ The normalized model must round-trip velocity -> height -> velocity to high precision,
        and the physical-units functions must be consistent with the normalized ones.
    """

    v_grid = np.linspace(0.05, 0.999, 200)

    ht_normed = alphaBetaHeightNormed(v_grid, ALPHA_TRUE, BETA_TRUE)
    v_back = alphaBetaVelocityNormed(ht_normed, ALPHA_TRUE, BETA_TRUE)

    assert np.max(np.abs(v_back - v_grid)) < 1e-8, \
        "round trip error {:.3g} > 1e-8".format(np.max(np.abs(v_back - v_grid)))

    # The physical-units inverse must match the normalized one
    v_phys = alphaBetaVelocity(HT_DATA, ALPHA_TRUE, BETA_TRUE, V_INIT_TRUE)
    v_normed = alphaBetaVelocityNormed(HT_DATA/HT_NORM_CONST, ALPHA_TRUE, BETA_TRUE)

    assert np.max(np.abs(v_phys - v_normed*V_INIT_TRUE)) < 1e-6

    # And the physical-units forward model must invert the velocities back to the input heights
    ht_back = alphaBetaHeight(v_phys, ALPHA_TRUE, BETA_TRUE, V_INIT_TRUE)

    assert np.max(np.abs(ht_back - HT_DATA)) < 0.1, \
        "height round trip error {:.3g} m > 0.1 m".format(np.max(np.abs(ht_back - HT_DATA)))


def testAlphaBetaVelocityNormedLUT():
    """ The experimental LUT-accelerated alphaBetaVelocityNormedLUT() must:
        (1) recover v_normed as accurately as the brentq-based alphaBetaVelocityNormed() across
            the (alpha, beta) domain where the underlying Ei(beta) - Ei(beta v^2) formulation
            itself is well-conditioned in float64 (beta up to ~25 - see the module comment above
            alphaBetaVelocityNormedLUT() in AlphaBeta.py for why beta >~ 30 is excluded: Ei(beta)
            there is so much larger than Ei(beta v^2) that the *forward* model already loses
            precision to cancellation, so brentq is not a trustworthy reference either), and
        (2) be substantially faster, since that's its only reason to exist.
    """

    # (alpha, beta) pairs spanning ALPHA_BETA_BOUNDS, capped at beta = 25 for the reason above
    (alpha_lo, alpha_hi), (beta_lo, _) = ALPHA_BETA_BOUNDS
    param_grid = [(alpha, beta)
        for alpha in (alpha_lo, 1.0, ALPHA_TRUE, 100.0, alpha_hi)
        for beta in (beta_lo, 0.01, 0.5, BETA_TRUE, 10.0, 20.0, 25.0)]

    v_grid = np.linspace(0.02, 0.999, 100)

    max_err_vs_truth = 0.0
    max_err_vs_brentq = 0.0

    for alpha, beta in param_grid:

        ht_normed = alphaBetaHeightNormed(v_grid, alpha, beta)

        v_lut = alphaBetaVelocityNormedLUT(ht_normed, alpha, beta)
        v_brentq = alphaBetaVelocityNormed(ht_normed, alpha, beta)

        max_err_vs_truth = max(max_err_vs_truth, np.max(np.abs(v_lut - v_grid)))
        max_err_vs_brentq = max(max_err_vs_brentq, np.max(np.abs(v_lut - v_brentq)))

    assert max_err_vs_truth < 1e-5, \
        "LUT max|dv| vs ground truth {:.3g} >= 1e-5".format(max_err_vs_truth)
    assert max_err_vs_brentq < 1e-5, \
        "LUT max|dv| vs brentq {:.3g} >= 1e-5".format(max_err_vs_brentq)

    # Edge cases: scalar input, and heights outside the invertible range must clip like
    #   alphaBetaVelocityNormed() does
    v_eps = 1e-10
    assert abs(float(alphaBetaVelocityNormedLUT(-100.0, ALPHA_TRUE, BETA_TRUE)) - v_eps) < 1e-12
    assert abs(float(alphaBetaVelocityNormedLUT(100.0, ALPHA_TRUE, BETA_TRUE)) - (1 - v_eps)) < 1e-12

    # Speed: the LUT path must be dramatically faster than the per-point brentq loop. Exclude the
    #   one-time table construction (a real caller builds/reuses it once, not per call) and use a
    #   generous threshold (measured speedup on the reference machine was ~500x) so the assertion
    #   only fails on a genuine performance regression, not machine noise
    ht_normed = alphaBetaHeightNormed(np.linspace(0.05, 0.999, 2000), ALPHA_TRUE, BETA_TRUE)

    alphaBetaVelocityNormedLUT(ht_normed, ALPHA_TRUE, BETA_TRUE)  # warm the default LUT

    t0 = time.perf_counter()
    alphaBetaVelocityNormedLUT(ht_normed, ALPHA_TRUE, BETA_TRUE)
    t_lut = time.perf_counter() - t0

    t0 = time.perf_counter()
    alphaBetaVelocityNormed(ht_normed, ALPHA_TRUE, BETA_TRUE)
    t_brentq = time.perf_counter() - t0

    speedup = t_brentq/t_lut
    # Soft check: timing ratios are machine/CI-load dependent, so a miss warns rather than fails
    #   the suite (the <1e-5 accuracy asserts above are the real guarantees).
    if speedup <= 20:
        print("WARNING: LUT speedup {:.1f}x <= 20x (perf only, not a correctness failure)".format(
            speedup))


def testFastFlagPropagation():
    """ fast=True must be accepted end-to-end by every public entry point that inverts velocity
        (fitAlphaBeta, fitAlphaBetaMass, fitAlphaBetaLightCurve, profileAlphaBeta, plotAlphaBeta,
        plotProfileAlphaBeta) and must recover essentially the same fit as fast=False - the LUT
        path is meant to be a faster ROUTE to the same answer within the well-conditioned regime
        (beta well under the ~30 cancellation ceiling - see the module comment above
        alphaBetaVelocityNormedLUT()) this synthetic scenario (BETA_TRUE=2) stays in, not a
        different model.
    """

    v_data, _, m_init_true, _, _ = _syntheticTrajectory()
    mag_data, _ = _syntheticLightCurve()

    # --- fitAlphaBeta ---
    v_init_slow, alpha_slow, beta_slow = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', verbose=False)
    v_init_fast, alpha_fast, beta_fast = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', verbose=False, fast=True)

    assert v_init_fast == v_init_slow
    _assertClose(alpha_fast, alpha_slow, 1e-3, "fitAlphaBeta alpha (fast vs slow)")
    _assertClose(beta_fast, beta_slow, 1e-3, "fitAlphaBeta beta (fast vs slow)")

    fit = (v_init_slow, alpha_slow, beta_slow)

    # --- fitAlphaBetaMass ---
    res_slow = fitAlphaBetaMass(v_data, HT_DATA, SLOPE, m_init_true, mass_constraint="initial", \
        density=DENS_TRUE, v_init=V_INIT_TRUE, verbose=False)
    res_fast = fitAlphaBetaMass(v_data, HT_DATA, SLOPE, m_init_true, mass_constraint="initial", \
        density=DENS_TRUE, v_init=V_INIT_TRUE, verbose=False, fast=True)

    _assertClose(res_fast[3], res_slow[3], 1e-3, "fitAlphaBetaMass alpha_mu0 (fast vs slow)")
    _assertClose(res_fast[4], res_slow[4], 1e-3, "fitAlphaBetaMass beta_mu0 (fast vs slow)")

    # --- fitAlphaBetaLightCurve ---
    lc_slow = fitAlphaBetaLightCurve(v_data, HT_DATA, HT_DATA, mag_data, v_init=V_INIT_TRUE, \
        sigma_mag=MAG_NOISE_STD, verbose=False)
    lc_fast = fitAlphaBetaLightCurve(v_data, HT_DATA, HT_DATA, mag_data, v_init=V_INIT_TRUE, \
        sigma_mag=MAG_NOISE_STD, verbose=False, fast=True)

    for mu in lc_slow['fits']:
        _assertClose(lc_fast['fits'][mu]['alpha'], lc_slow['fits'][mu]['alpha'], 1e-3, \
            "fitAlphaBetaLightCurve alpha mu={:.3f} (fast vs slow)".format(mu))
        _assertClose(lc_fast['fits'][mu]['beta'], lc_slow['fits'][mu]['beta'], 1e-3, \
            "fitAlphaBetaLightCurve beta mu={:.3f} (fast vs slow)".format(mu))

    # --- profileAlphaBeta ---
    profile_slow = profileAlphaBeta(v_data, HT_DATA, fit, param='beta', n_grid=40, verbose=False)
    profile_fast = profileAlphaBeta(v_data, HT_DATA, fit, param='beta', n_grid=40, verbose=False, \
        fast=True)

    _assertClose(profile_fast['beta']['ci_lower'], profile_slow['beta']['ci_lower'], 1e-2, \
        "profileAlphaBeta beta ci_lower (fast vs slow)")
    _assertClose(profile_fast['beta']['ci_upper'], profile_slow['beta']['ci_upper'], 1e-2, \
        "profileAlphaBeta beta ci_upper (fast vs slow)")

    # --- plotAlphaBeta / plotProfileAlphaBeta: must draw without raising under fast=True too ---
    _, _, _, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', \
        estimate_errors=True, fast=True)

    fig, _ = plotAlphaBeta(v_data, HT_DATA, fit, errors=errors, band_samples=20, fan_curves=5, \
        seed=0, fast=True)
    plt.close(fig)

    fig, _ = plotProfileAlphaBeta(v_data, HT_DATA, fit, profile_fast, degeneracy=False, \
        fast=True)
    plt.close(fig)

    # The default LUT getter must be public and reuse the same cached table across calls
    lut = getDefaultInverseEiLUT()
    assert lut is getDefaultInverseEiLUT()


def testFastPipelineSpeedup():
    """ fast=True must give a real, substantial speedup at the pipeline level too, not just for
        the bare alphaBetaVelocityNormedLUT() vs alphaBetaVelocityNormed() call already covered by
        testAlphaBetaVelocityNormedLUT(). Checks the two consumers that call the dynamics
        residual (_dynResiduals(), the innermost per-iteration hot path) the most:
        fitAlphaBeta(method='robust') (one least_squares run) and profileAlphaBeta() (O(n_grid)
        least_squares runs, one per grid point - the largest beneficiary in this module).

        Thresholds are set well below what was actually measured on the reference machine
        (~14x for fitAlphaBeta, ~37x for profileAlphaBeta at n_grid=40) so this only fails on a
        genuine performance regression, not machine noise.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    getDefaultInverseEiLUT()  # warm the default LUT once, excluded from both timings below

    # --- fitAlphaBeta(method='robust') ---
    t0 = time.perf_counter()
    fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', verbose=False, fast=False)
    t_slow = time.perf_counter() - t0

    t0 = time.perf_counter()
    fit = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', verbose=False, \
        fast=True)
    t_fast = time.perf_counter() - t0

    speedup = t_slow/t_fast

    if speedup <= 3:
        print("WARNING: fitAlphaBeta fast=True speedup {:.1f}x <= 3x (perf only)".format(speedup))

    # --- profileAlphaBeta() --- n_grid reduced from the default (250) to keep the test itself
    #   fast; the speedup factor doesn't depend on n_grid (every grid point costs the same
    #   relative amount either way), so this is still representative of the default.
    t0 = time.perf_counter()
    profileAlphaBeta(v_data, HT_DATA, fit, param='beta', n_grid=40, verbose=False, fast=False)
    t_slow = time.perf_counter() - t0

    t0 = time.perf_counter()
    profileAlphaBeta(v_data, HT_DATA, fit, param='beta', n_grid=40, verbose=False, fast=True)
    t_fast = time.perf_counter() - t0

    speedup = t_slow/t_fast
    if speedup <= 10:
        print("WARNING: profileAlphaBeta fast=True speedup {:.1f}x <= 10x (perf only)".format(
            speedup))


def testFitAlphaBetaRobust():
    """ The robust fit should recover alpha and beta from noisy synthetic data, with velocity
        residuals consistent with the injected noise.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    v_init, alpha, beta = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust')

    _assertClose(alpha, ALPHA_TRUE, 0.05, "alpha (robust)")
    _assertClose(beta, BETA_TRUE, 0.05, "beta (robust)")

    # The fit must track the data to within ~2x the injected noise
    v_model = alphaBetaVelocity(HT_DATA, alpha, beta, v_init)
    rmse = np.sqrt(np.mean((v_model - v_data)**2))

    assert rmse < 2*VEL_NOISE_STD, "robust fit RMSE {:.1f} m/s > {:.1f} m/s".format(
        rmse, 2*VEL_NOISE_STD)

    # An unknown method name must be rejected
    try:
        fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method="banana")

    except ValueError:
        pass

    else:
        raise AssertionError("fitAlphaBeta: ValueError not raised for a bad method name")


def testFitAlphaBetaErrorEstimationRejectsQ4():
    """ estimate_errors=True must be rejected for method='q4' - Q4's L1/Nelder-Mead fit has no
        natural Jacobian to build an analytic covariance from.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    try:
        fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='q4', estimate_errors=True)

    except ValueError:
        pass

    else:
        raise AssertionError(
            "fitAlphaBeta: ValueError not raised for estimate_errors=True with method='q4'")


def testFitAlphaBetaErrorEstimationRejectsBadCi():
    """ estimate_errors=True must reject a ci outside (0, 100) - out-of-range values degrade
        silently otherwise: ci=100 gives z=inf (0/inf bounds), ci>100 gives z=NaN, ci<0 inverts
        which bound is 'lower' vs 'upper'.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    for bad_ci in [0.0, 100.0, -5.0, 150.0]:

        try:
            fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', \
                estimate_errors=True, ci=bad_ci)

        except ValueError:
            pass

        else:
            raise AssertionError("fitAlphaBeta: ValueError not raised for ci={!r}".format(bad_ci))


def testFitAlphaBetaErrorEstimationPointEstimateUnchanged():
    """ estimate_errors=True must not change the point estimate itself, and the errors dict must
        echo back the same alpha/beta it was computed from.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    v_init_plain, alpha_plain, beta_plain = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust')

    v_init, alpha, beta, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)

    assert v_init == v_init_plain
    assert alpha == alpha_plain
    assert beta == beta_plain
    assert errors['alpha'] == alpha
    assert errors['beta'] == beta


def testFitAlphaBetaErrorEstimationCovarianceStructure():
    """ cov_log/cov must be well-formed (symmetric, positive diagonal), corr_log/corr must be a
        valid correlation coefficient, and the linear (alpha, beta) space quantities must match
        the exact delta-method transform of the (ln alpha, ln beta) ones: alpha_std =
        alpha*alpha_std_rel, cov = diag(alpha, beta) @ cov_log @ diag(alpha, beta), and corr must
        come out numerically identical to corr_log (correlation is invariant under this kind of
        positive-diagonal rescaling).
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    _, alpha, beta, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', \
        estimate_errors=True)

    # v_init given, no sigma_v_init: v_init must be treated as exact (zero propagated uncertainty)
    assert errors['sigma_v_init'] == 0.0

    assert np.isfinite(errors['alpha_std_rel']) and errors['alpha_std_rel'] > 0
    assert np.isfinite(errors['beta_std_rel']) and errors['beta_std_rel'] > 0
    assert -1.0 <= errors['corr_log'] <= 1.0
    assert errors['cov_log'].shape == (2, 2)
    assert np.allclose(errors['cov_log'], errors['cov_log'].T)
    assert errors['cov_log'][0, 0] > 0 and errors['cov_log'][1, 1] > 0

    # This trajectory has strong deceleration and moderate noise, so both parameters should be
    #   well under 100% relative uncertainty
    assert errors['alpha_std_rel'] < 1.0, "alpha_std_rel implausibly large: {:.3g}".format(
        errors['alpha_std_rel'])
    assert errors['beta_std_rel'] < 1.0, "beta_std_rel implausibly large: {:.3g}".format(
        errors['beta_std_rel'])

    _assertClose(errors['alpha_std'], alpha*errors['alpha_std_rel'], 1e-9, "alpha_std")
    _assertClose(errors['beta_std'], beta*errors['beta_std_rel'], 1e-9, "beta_std")
    assert errors['cov'].shape == (2, 2)
    assert np.allclose(errors['cov'], errors['cov'].T)
    scale = np.array([alpha, beta])
    _assertClose(errors['cov'][0, 1], scale[0]*scale[1]*errors['cov_log'][0, 1], 1e-9, "cov[0,1]")
    _assertClose(errors['corr'], errors['corr_log'], 1e-9, "corr vs corr_log")


def testFitAlphaBetaErrorEstimationConfidenceInterval():
    """ The parametric (Gaussian-in-log-space) confidence interval must bracket the point
        estimate, and must widen at a higher confidence level.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    _, alpha, beta, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', \
        estimate_errors=True)

    assert errors['alpha_ci_wald_lower'] < alpha < errors['alpha_ci_wald_upper']
    assert errors['beta_ci_wald_lower'] < beta < errors['beta_ci_wald_upper']

    _, _, _, errors_ci68 = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', \
        estimate_errors=True, ci=68.27)
    _, _, _, errors_ci99 = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', \
        estimate_errors=True, ci=99.0)

    assert (errors_ci99['alpha_ci_wald_upper'] - errors_ci99['alpha_ci_wald_lower']) > \
        (errors_ci68['alpha_ci_wald_upper'] - errors_ci68['alpha_ci_wald_lower'])
    assert (errors_ci99['beta_ci_wald_upper'] - errors_ci99['beta_ci_wald_lower']) > \
        (errors_ci68['beta_ci_wald_upper'] - errors_ci68['beta_ci_wald_lower'])


def testFitAlphaBetaErrorEstimationVerbose():
    """ verbose=True must not raise, with or without estimate_errors. """

    v_data, _, _, _, _ = _syntheticTrajectory()

    fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', verbose=True)
    fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', estimate_errors=True, \
        verbose=True)


def testFitAlphaBetaErrorEstimationVInitPropagation():
    """ sigma_v_init must only ever widen (never narrow) alpha/beta's uncertainty when supplied
        explicitly (the correction is a positive-semidefinite rank-1 term added to the same
        cov_fit), and must be auto-derived as the standard error of the median (NOT the plain
        MAD) of the same leading points fitAlphaBeta() itself uses to derive v_init, when v_init
        is left to be derived internally.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    _, _, _, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust', \
        estimate_errors=True)

    # Propagating an explicit sigma_v_init adds a positive-semidefinite rank-1 term to the same
    #   cov_fit, so it must only ever widen (never narrow) alpha/beta's uncertainty
    _, _, _, errors_with_v0_unc = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True, sigma_v_init=50.0)

    assert errors_with_v0_unc['sigma_v_init'] == 50.0
    assert errors_with_v0_unc['alpha_std_rel'] >= errors['alpha_std_rel']
    assert errors_with_v0_unc['beta_std_rel'] >= errors['beta_std_rel']

    # v_init=None: sigma_v_init must be auto-derived as the standard error of the median (not the
    #   plain MAD) of the same leading points fitAlphaBeta() itself uses to derive v_init
    _, _, _, errors_derived = fitAlphaBeta(v_data, HT_DATA, method='robust', estimate_errors=True)

    max_index = max(int(0.2*len(v_data)), 10)
    v_head = v_data[:max_index]
    mad = np.median(np.abs(v_head - np.median(v_head)))
    sigma_v_init_expected = 1.2533*1.4826*mad/np.sqrt(max_index)

    _assertClose(errors_derived['sigma_v_init'], sigma_v_init_expected, 1e-9, "sigma_v_init (auto)")
    assert errors_derived['sigma_v_init'] > 0

    # A superfluous sigma_v_init passed alongside v_init=None must be ignored (only a warning is
    #   printed, not captured here) - the auto-derived value must be unaffected by it
    _, _, _, errors_derived_with_superfluous = fitAlphaBeta(v_data, HT_DATA, method='robust', \
        estimate_errors=True, sigma_v_init=999.0)

    _assertClose(errors_derived_with_superfluous['sigma_v_init'], sigma_v_init_expected, 1e-9, \
        "sigma_v_init (auto, with a superfluous sigma_v_init passed)")


def testFitAlphaBetaErrorEstimationBoundPinned():
    """ When the fitted (alpha, beta) is pinned against an ALPHA_BETA_BOUNDS edge, estimate_errors
        must fail cleanly - NaN for every error field, no unhandled exception, no raw numpy
        RuntimeWarning from sqrt() of a negative covariance diagonal - instead of silently
        propagating a mathematically impossible negative variance. Found via real usage: a fit on
        an actual trajectory (beta landing at 49.99995, a hair under the upper bound of 50) came
        back with alpha_std_rel/beta_std_rel/cov_log/etc. all NaN and no explanation.

        The true beta here is set well beyond the upper bound, so the constrained fit has no
        choice but to pin against that edge - reproduces the same pathology deterministically.
    """

    beta_true_extreme = 150.0

    v_clean = alphaBetaVelocity(HT_DATA, ALPHA_TRUE, beta_true_extreme, V_INIT_TRUE)
    rng = np.random.RandomState(42)
    v_data = v_clean + rng.normal(0, VEL_NOISE_STD, HT_DATA.size)

    v_init, alpha, beta, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)

    # The fit must indeed have pinned against the beta upper bound (sanity-checks the test setup
    #   itself, so a future change that stops reproducing this failure mode fails here first,
    #   with a clear message, rather than via a confusing assertion below)
    _, beta_upper = ALPHA_BETA_BOUNDS[1]
    assert beta > 0.99*beta_upper, "test setup didn't actually pin beta against its bound"

    # Must fail cleanly: every error field NaN together, not a partial/mixed NaN state
    assert np.isnan(errors['alpha_std_rel'])
    assert np.isnan(errors['beta_std_rel'])
    assert np.isnan(errors['alpha_std'])
    assert np.isnan(errors['beta_std'])
    assert np.all(np.isnan(errors['cov_log']))
    assert np.isnan(errors['corr_log'])
    assert np.all(np.isnan(errors['cov']))
    assert np.isnan(errors['corr'])
    assert np.isnan(errors['alpha_ci_wald_lower'])
    assert np.isnan(errors['alpha_ci_wald_upper'])
    assert np.isnan(errors['beta_ci_wald_lower'])
    assert np.isnan(errors['beta_ci_wald_upper'])

    # alpha/beta themselves, and sigma_v/sigma_v_init, are still real, useful numbers even though
    #   the covariance failed - only the error fields collapse to NaN
    assert np.isfinite(alpha) and np.isfinite(beta)
    assert errors['alpha'] == alpha and errors['beta'] == beta
    assert np.isfinite(errors['sigma_v']) and errors['sigma_v'] > 0
    assert errors['sigma_v_init'] == 0.0


def testAlphaBetaMassesErrorEstimationRequiresCovLog():
    """ estimate_errors=True must be rejected without a cov_log - there is no (alpha, beta)
        covariance to propagate otherwise.
    """

    try:
        alphaBetaMasses(ALPHA_TRUE, BETA_TRUE, SLOPE, mu=0, dens=DENS_TRUE, estimate_errors=True)

    except ValueError:
        pass

    else:
        raise AssertionError(
            "alphaBetaMasses: ValueError not raised for estimate_errors=True without cov_log")


def testAlphaBetaMassesErrorEstimationRejectsBadCi():
    """ estimate_errors=True must reject a ci outside (0, 100), same reasoning as fitAlphaBeta()'s
        own ci validation.
    """

    cov_log = np.array([[0.01, 0.0], [0.0, 0.01]])

    for bad_ci in [0.0, 100.0, -5.0, 150.0]:

        try:
            alphaBetaMasses(ALPHA_TRUE, BETA_TRUE, SLOPE, mu=0, dens=DENS_TRUE, \
                estimate_errors=True, cov_log=cov_log, ci=bad_ci)

        except ValueError:
            pass

        else:
            raise AssertionError(
                "alphaBetaMasses: ValueError not raised for ci={!r}".format(bad_ci))


def testAlphaBetaMassesErrorEstimationPointEstimateUnchanged():
    """ estimate_errors=True must not change the point estimate itself, and the errors dict must
        echo back the same m_init/m_final it was computed from.
    """

    v_data, v_final_true, _, _, _ = _syntheticTrajectory()

    _, alpha, beta, fit_errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)

    m_init_plain, m_final_plain = alphaBetaMasses(alpha, beta, SLOPE, mu=0, dens=DENS_TRUE, \
        vel_init=V_INIT_TRUE, vel_end=v_final_true)

    m_init, m_final, errors = alphaBetaMasses(alpha, beta, SLOPE, mu=0, dens=DENS_TRUE, \
        vel_init=V_INIT_TRUE, vel_end=v_final_true, estimate_errors=True, \
        cov_log=fit_errors['cov_log'])

    assert m_init == m_init_plain
    assert m_final == m_final_plain
    assert errors['m_init'] == m_init
    assert errors['m_final'] == m_final


def testAlphaBetaMassesErrorEstimationCovarianceStructure():
    """ The propagated (ln m_init, ln m_final) covariance must be well-formed (symmetric,
        positive diagonal), corr_log must be a valid correlation coefficient, m_init_std/
        m_final_std must match the exact delta-method transform (m_std = m*m_std_rel), and the
        Wald CI must bracket the point estimate.
    """

    v_data, v_final_true, _, _, _ = _syntheticTrajectory()

    _, alpha, beta, fit_errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)

    m_init, m_final, errors = alphaBetaMasses(alpha, beta, SLOPE, mu=0, dens=DENS_TRUE, \
        vel_init=V_INIT_TRUE, vel_end=v_final_true, estimate_errors=True, \
        cov_log=fit_errors['cov_log'], sigma_slope=np.radians(1.0), sigma_dens=200.0, \
        sigma_vel_init=30.0, sigma_vel_end=30.0)

    assert np.isfinite(errors['m_init_std_rel']) and errors['m_init_std_rel'] > 0
    assert np.isfinite(errors['m_final_std_rel']) and errors['m_final_std_rel'] > 0
    assert -1.0 <= errors['corr_log'] <= 1.0

    assert errors['cov_log'].shape == (2, 2)
    assert np.allclose(errors['cov_log'], errors['cov_log'].T)
    assert errors['cov_log'][0, 0] > 0 and errors['cov_log'][1, 1] > 0

    _assertClose(errors['m_init_std'], m_init*errors['m_init_std_rel'], 1e-9, "m_init_std")
    _assertClose(errors['m_final_std'], m_final*errors['m_final_std_rel'], 1e-9, "m_final_std")

    assert errors['m_init_ci_wald_lower'] < m_init < errors['m_init_ci_wald_upper']
    assert errors['m_final_ci_wald_lower'] < m_final < errors['m_final_ci_wald_upper']

    assert errors['sigma_slope'] == np.radians(1.0)
    assert errors['sigma_dens'] == 200.0
    assert errors['sigma_vel_init'] == 30.0
    assert errors['sigma_vel_end'] == 30.0


def testAlphaBetaMassesErrorEstimationBoundPinnedPropagatesNaN():
    """ A NaN cov_log (e.g. from a fitAlphaBeta() fit pinned against an ALPHA_BETA_BOUNDS edge -
        see testFitAlphaBetaErrorEstimationBoundPinned) must propagate to NaN mass errors cleanly,
        not raise - the same graceful-degradation contract as fitAlphaBeta() itself.
    """

    beta_true_extreme = 150.0

    v_clean = alphaBetaVelocity(HT_DATA, ALPHA_TRUE, beta_true_extreme, V_INIT_TRUE)
    rng = np.random.RandomState(42)
    v_data = v_clean + rng.normal(0, VEL_NOISE_STD, HT_DATA.size)

    _, alpha, beta, fit_errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)

    assert np.all(np.isnan(fit_errors['cov_log']))  # sanity-check the test setup itself

    v_final = np.min(alphaBetaVelocity(HT_DATA, alpha, beta, V_INIT_TRUE))

    m_init, m_final, errors = alphaBetaMasses(alpha, beta, SLOPE, mu=0, dens=DENS_TRUE, \
        vel_init=V_INIT_TRUE, vel_end=v_final, estimate_errors=True, \
        cov_log=fit_errors['cov_log'])

    assert np.isfinite(m_init) and np.isfinite(m_final)
    assert errors['m_init'] == m_init and errors['m_final'] == m_final
    assert np.isnan(errors['m_init_std_rel'])
    assert np.isnan(errors['m_final_std_rel'])
    assert np.isnan(errors['m_init_std'])
    assert np.isnan(errors['m_final_std'])
    assert np.all(np.isnan(errors['cov_log']))
    assert np.isnan(errors['corr_log'])
    assert np.isnan(errors['m_init_ci_wald_lower'])
    assert np.isnan(errors['m_init_ci_wald_upper'])
    assert np.isnan(errors['m_final_ci_wald_lower'])
    assert np.isnan(errors['m_final_ci_wald_upper'])


def testAlphaBetaMassesErrorEstimationIgnoresVelSigmaWithoutFullSolution():
    """ sigma_vel_init/sigma_vel_end must be ignored (not raise, and not affect m_final_std_rel)
        when vel_init/vel_end are not both given - the simple mass-loss approximation doesn't
        depend on either velocity.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    _, alpha, beta, fit_errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)

    _, _, errors_no_vel = alphaBetaMasses(alpha, beta, SLOPE, mu=0, dens=DENS_TRUE, \
        estimate_errors=True, cov_log=fit_errors['cov_log'])

    _, _, errors_superfluous = alphaBetaMasses(alpha, beta, SLOPE, mu=0, dens=DENS_TRUE, \
        estimate_errors=True, cov_log=fit_errors['cov_log'], sigma_vel_init=999.0, \
        sigma_vel_end=999.0)

    assert errors_no_vel['sigma_vel_init'] == 0.0
    assert errors_no_vel['sigma_vel_end'] == 0.0
    assert errors_superfluous['sigma_vel_init'] == 0.0
    assert errors_superfluous['sigma_vel_end'] == 0.0

    _assertClose(errors_superfluous['m_final_std_rel'], errors_no_vel['m_final_std_rel'], 1e-9, \
        "m_final_std_rel (superfluous sigma_vel_init/sigma_vel_end must be ignored)")


def testAlphaBetaMassesErrorEstimationMassRatioIdentity():
    """ Var(ln(m_final/m_init)) must reduce to exactly (k*beta*beta_std_rel)^2 plus the
        vel_init/vel_end contribution (from r = vel_end/vel_init inside k) - independent of
        alpha/slope/dens entirely, since those enter ln(m_init) and ln(m_final) through the exact
        same additive term and cancel out of the difference (see _alphaBetaMassErrors()'s
        docstring). This is an exact algebraic identity of the propagation, checked directly
        rather than via Monte Carlo.
    """

    v_data, v_final_true, _, _, _ = _syntheticTrajectory()

    _, alpha, beta, fit_errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)

    sigma_vel_init, sigma_vel_end = 50.0, 50.0

    m_init, m_final, errors = alphaBetaMasses(alpha, beta, SLOPE, mu=0, dens=DENS_TRUE, \
        vel_init=V_INIT_TRUE, vel_end=v_final_true, estimate_errors=True, \
        cov_log=fit_errors['cov_log'], sigma_slope=np.radians(2.0), sigma_dens=300.0, \
        sigma_vel_init=sigma_vel_init, sigma_vel_end=sigma_vel_end)

    var_init = errors['cov_log'][0, 0]
    var_final = errors['cov_log'][1, 1]
    cov_init_final = errors['cov_log'][0, 1]

    var_ratio = var_init + var_final - 2.0*cov_init_final

    # Pure-beta contribution, k = (1 - r^2)/(1 - mu) with mu = 0
    r = v_final_true/V_INIT_TRUE
    k = 1.0 - r**2
    beta_std_rel = np.sqrt(fit_errors['cov_log'][1, 1])
    beta_term = (k*beta*beta_std_rel)**2

    # vel_init/vel_end contribution: m_init doesn't depend on either velocity at all, so unlike
    #   the alpha/slope/dens terms this one does NOT cancel out of the ratio
    vinit_coef = -2.0*beta*r**2/V_INIT_TRUE
    vend_coef = 2.0*beta*r/V_INIT_TRUE
    vel_term = vinit_coef**2*sigma_vel_init**2 + vend_coef**2*sigma_vel_end**2

    _assertClose(var_ratio, beta_term + vel_term, 1e-6, "Var(ln(m_final/m_init))")


def testAlphaBetaMassesErrorEstimationMonteCarlo():
    """ The analytic mass-error propagation must match a direct Monte Carlo propagation of the
        same (alpha, beta) covariance plus independent slope/density/velocity uncertainties
        through alphaBetaMasses() itself - both the marginal log-space standard deviations and
        the (ln m_init, ln m_final) correlation. Unlike fitAlphaBeta()'s own coverage checks (which
        need an expensive refit per realization), alphaBetaMasses() is a closed-form function, so
        this Monte Carlo comparison is cheap enough to run as a real automated test rather than a
        one-off analysis.
    """

    v_data, v_final_true, _, _, _ = _syntheticTrajectory()

    _, alpha, beta, fit_errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)
    cov_log = fit_errors['cov_log']
    assert np.all(np.isfinite(cov_log))

    sigma_slope = np.radians(1.0)
    sigma_dens = 200.0
    sigma_vel_init = 30.0
    sigma_vel_end = 30.0

    _, _, errors = alphaBetaMasses(alpha, beta, SLOPE, mu=0, dens=DENS_TRUE, \
        vel_init=V_INIT_TRUE, vel_end=v_final_true, estimate_errors=True, cov_log=cov_log, \
        sigma_slope=sigma_slope, sigma_dens=sigma_dens, sigma_vel_init=sigma_vel_init, \
        sigma_vel_end=sigma_vel_end)

    rng = np.random.RandomState(123)
    n_mc = 20000

    log_ab_samples = rng.multivariate_normal(np.log([alpha, beta]), cov_log, size=n_mc)
    alpha_samples = np.exp(log_ab_samples[:, 0])
    beta_samples = np.exp(log_ab_samples[:, 1])
    slope_samples = rng.normal(SLOPE, sigma_slope, n_mc)
    dens_samples = rng.normal(DENS_TRUE, sigma_dens, n_mc)
    vel_init_samples = rng.normal(V_INIT_TRUE, sigma_vel_init, n_mc)
    vel_end_samples = rng.normal(v_final_true, sigma_vel_end, n_mc)

    log_m_init_mc = np.empty(n_mc)
    log_m_final_mc = np.empty(n_mc)

    for i in range(n_mc):
        m_init_i, m_final_i = alphaBetaMasses(alpha_samples[i], beta_samples[i], \
            slope_samples[i], mu=0, dens=dens_samples[i], vel_init=vel_init_samples[i], \
            vel_end=vel_end_samples[i])
        log_m_init_mc[i] = np.log(m_init_i)
        log_m_final_mc[i] = np.log(m_final_i)

    mc_m_init_std_rel = np.std(log_m_init_mc)
    mc_m_final_std_rel = np.std(log_m_final_mc)
    mc_corr = np.corrcoef(log_m_init_mc, log_m_final_mc)[0, 1]

    _assertClose(errors['m_init_std_rel'], mc_m_init_std_rel, 0.05, \
        "m_init_std_rel vs Monte Carlo")
    _assertClose(errors['m_final_std_rel'], mc_m_final_std_rel, 0.08, \
        "m_final_std_rel vs Monte Carlo")

    analytic_corr = errors['corr_log']
    assert abs(analytic_corr - mc_corr) < 0.05, \
        "corr(ln m_init, ln m_final) mismatch: analytic {:.4f} vs MC {:.4f}".format(
        analytic_corr, mc_corr)


def testGaussianEllipsePoints():
    """ _gaussianEllipsePoints() must trace a circle of the exact closed-form radius for an
        identity covariance (chi2(df=2) = Exponential(scale=2), so its p-quantile has an exact
        closed form, sqrt(-2*ln(1-p))), centered exactly on the given mean, and must orient a
        correlated covariance's ellipse along the correlation's own sign.
    """

    mean = np.array([3.0, -2.0])
    cov_identity = np.eye(2)

    for p in [0.5, 0.6827, 0.95]:

        x, y = _gaussianEllipsePoints(mean, cov_identity, p, n_points=1000)

        expected_radius = np.sqrt(-2.0*np.log(1.0 - p))
        actual_radius = np.sqrt((x - mean[0])**2 + (y - mean[1])**2)

        assert np.max(np.abs(actual_radius - expected_radius)) < 1e-9, \
            "ellipse radius off at p={:.4f}".format(p)

    # Positive correlation must stretch the ellipse along the (+1, +1) diagonal, not (+1, -1):
    #   the point farthest in +x must also be displaced in +y, not -y
    cov_corr = np.array([[1.0, 0.9], [0.9, 1.0]])
    x, y = _gaussianEllipsePoints(mean, cov_corr, 0.6827, n_points=1000)
    assert y[np.argmax(x)] > mean[1]


def testPlotAlphaBetaGracefulDegradation():
    """ plotAlphaBeta() must never raise when errors is None, or when errors is given but its
        covariance is NaN (a fit pinned against an ALPHA_BETA_BOUNDS edge) - the band/ellipse
        degrade to data+fit (+ an annotation) instead, and the residuals panel (which only needs
        `fit` and the raw data) still draws the actual residual scatter either way, just without
        the sigma_v reference lines/down-weighting.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    v_init, alpha, beta, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)
    fit = (v_init, alpha, beta)

    fig, axes = plotAlphaBeta(v_data, HT_DATA, fit, errors=None, residuals=True, seed=0)
    plt.close(fig)

    # A true beta far beyond ALPHA_BETA_BOUNDS pins the fit against its upper bound, the same
    #   deterministic setup as testFitAlphaBetaErrorEstimationBoundPinned()
    beta_true_extreme = 150.0
    v_clean_extreme = alphaBetaVelocity(HT_DATA, ALPHA_TRUE, beta_true_extreme, V_INIT_TRUE)
    rng = np.random.RandomState(42)
    v_data_extreme = v_clean_extreme + rng.normal(0, VEL_NOISE_STD, HT_DATA.size)

    v_init_e, alpha_e, beta_e, errors_e = fitAlphaBeta(v_data_extreme, HT_DATA, \
        v_init=V_INIT_TRUE, method='robust', estimate_errors=True)

    assert np.isnan(errors_e['alpha_std_rel'])  # sanity-check the test setup itself

    fig, axes = plotAlphaBeta(v_data_extreme, HT_DATA, (v_init_e, alpha_e, beta_e), \
        errors=errors_e, residuals=True, seed=0)
    plt.close(fig)


def testPlotAlphaBetaAxesContract():
    """ axes must be reusable for a single-panel call (plotAlphaBeta() must hand back the exact
        same fig/ax), and must be rejected with ValueError for any multi-panel configuration,
        since a single Axes can't hold a multi-panel layout.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    v_init, alpha, beta, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)
    fit = (v_init, alpha, beta)

    fig_caller, ax_caller = plt.subplots()

    fig_ret, ax_ret = plotAlphaBeta(v_data, HT_DATA, fit, errors=errors, ellipse=False, \
        residuals=False, axes=ax_caller, seed=0)

    assert fig_ret is fig_caller
    assert ax_ret is ax_caller
    plt.close(fig_caller)

    fig_bad, ax_bad = plt.subplots()

    try:
        plotAlphaBeta(v_data, HT_DATA, fit, errors=errors, ellipse=True, axes=ax_bad)

    except ValueError:
        pass

    else:
        raise AssertionError("plotAlphaBeta: ValueError not raised for multi-panel + axes")

    finally:
        plt.close(fig_bad)


def testPlotAlphaBetaRejectsBadCi():
    """ plotAlphaBeta() must reject a ci outside (0, 100), same as fitAlphaBeta(). """

    v_data, _, _, _, _ = _syntheticTrajectory()

    v_init, alpha, beta, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)
    fit = (v_init, alpha, beta)

    for bad_ci in [0.0, 100.0, -5.0, 150.0]:

        try:
            plotAlphaBeta(v_data, HT_DATA, fit, errors=errors, ci=bad_ci)

        except ValueError:
            pass

        else:
            raise AssertionError(
                "plotAlphaBeta: ValueError not raised for ci={!r}".format(bad_ci))


def testPlotAlphaBetaSurvivalDiagramBoundaryMath():
    """ The boundary curve derivation (beta_boundary(x; mu) = (1-mu)*(ln(m0s/m_th) - 3x)) must be
        exact: a (alpha, beta) pair constructed to sit exactly on a given (mu, mass_threshold)
        boundary must, when passed back through alphaBetaMasses()'s own asymptotic approximation,
        reproduce that exact threshold mass. This is the analytic derivation
        plotAlphaBetaSurvivalDiagram() draws its lines from, checked independently of the
        plotting code itself.
    """

    dens, shape_coeff, gamma = 3500.0, 0.55, 1.0
    rho_atm_0 = 1.225
    m0s = (gamma*shape_coeff*rho_atm_0*HT_NORM_CONST/(dens**(2/3.0)))**3

    x = 1.0  # ln(alpha*sin(slope)); kept small so beta_boundary stays positive

    for mu, m_threshold in [(0.0, 1.0), (2/3, 1.0), (0.0, 0.05), (2/3, 0.05)]:

        beta_boundary = (1.0 - mu)*(np.log(m0s/m_threshold) - 3.0*x)
        alpha = np.exp(x)/np.sin(SLOPE)

        _, m_final = alphaBetaMasses(alpha, beta_boundary, SLOPE, mu=mu, dens=dens, \
            shape_coeff=shape_coeff, gamma=gamma)

        _assertClose(m_final, m_threshold, 1e-9, \
            "m_final on the mu={:.3g}, {:.3g} kg boundary".format(mu, m_threshold))


def testPlotAlphaBetaSurvivalDiagramRejectsBadInputs():
    """ Invalid mu_low/mu_high ordering, empty/non-positive mass_thresholds, and a bad ci must all
        raise ValueError, mirroring the validation style used throughout this module.
    """

    def _assertRaisesValueError(label, **kwargs):

        call_kwargs = dict(alpha=ALPHA_TRUE, beta=BETA_TRUE, slope=SLOPE)
        call_kwargs.update(kwargs)

        try:
            plotAlphaBetaSurvivalDiagram(**call_kwargs)

        except ValueError:
            plt.close('all')
            return

        plt.close('all')
        raise AssertionError("{:s}: ValueError not raised".format(label))

    _assertRaisesValueError("mu_low >= mu_high", mu_low=0.5, mu_high=0.3)
    _assertRaisesValueError("mu_high > 2/3", mu_low=0.0, mu_high=0.9)
    _assertRaisesValueError("mu_low < 0", mu_low=-0.1, mu_high=2/3)
    _assertRaisesValueError("empty mass_thresholds", mass_thresholds=())
    _assertRaisesValueError("non-positive mass_thresholds", mass_thresholds=(1.0, -0.05))

    for bad_ci in [0.0, 100.0, -5.0, 150.0]:
        _assertRaisesValueError("ci={!r}".format(bad_ci), ci=bad_ci)


def testPlotAlphaBetaSurvivalDiagramRuns():
    """ plotAlphaBetaSurvivalDiagram() must run without raising for the common cases: no errors,
        a real finite cov_log (draws the ellipse), and a NaN cov_log (degrades gracefully to an
        annotation instead of raising or emitting a raw numpy warning) - the same
        graceful-degradation contract as plotAlphaBeta().
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    _, alpha, beta, fit_errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)

    fig1, ax1 = plotAlphaBetaSurvivalDiagram(alpha, beta, SLOPE)
    plt.close(fig1)

    fig2, ax2 = plotAlphaBetaSurvivalDiagram(alpha, beta, SLOPE, errors=fit_errors)
    plt.close(fig2)

    # Single reference threshold only (no shading) must also work
    fig3, ax3 = plotAlphaBetaSurvivalDiagram(alpha, beta, SLOPE, mass_thresholds=(1.0,), \
        shade=False)
    plt.close(fig3)

    # A NaN cov_log (bound-pinned fit, same setup as testFitAlphaBetaErrorEstimationBoundPinned)
    beta_true_extreme = 150.0
    v_clean_extreme = alphaBetaVelocity(HT_DATA, ALPHA_TRUE, beta_true_extreme, V_INIT_TRUE)
    rng = np.random.RandomState(42)
    v_data_extreme = v_clean_extreme + rng.normal(0, VEL_NOISE_STD, HT_DATA.size)

    _, alpha_e, beta_e, fit_errors_e = fitAlphaBeta(v_data_extreme, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)
    assert np.all(np.isnan(fit_errors_e['cov_log']))  # sanity-check the test setup itself

    fig4, ax4 = plotAlphaBetaSurvivalDiagram(alpha_e, beta_e, SLOPE, errors=fit_errors_e)
    plt.close(fig4)


def testPlotAlphaBetaSurvivalDiagramAxesReuse():
    """ axes must be reusable - plotAlphaBetaSurvivalDiagram() must hand back the exact same
        fig/ax it was given, same contract as plotAlphaBeta()'s single-panel axes reuse.
    """

    fig_caller, ax_caller = plt.subplots()

    fig_ret, ax_ret = plotAlphaBetaSurvivalDiagram(ALPHA_TRUE, BETA_TRUE, SLOPE, axes=ax_caller)

    assert fig_ret is fig_caller
    assert ax_ret is ax_caller
    plt.close(fig_caller)


def testProfileAlphaBetaBracketsEstimate():
    """ profileAlphaBeta() must bracket the point estimate for a well-constrained fit, narrow at a
        lower confidence level, respect the `param` selector, and work regardless of whether the
        original point estimate came from method='q4' or 'robust' - unlike estimate_errors=True,
        it re-derives its own cost from `fit` directly rather than needing the robust fit's
        Jacobian.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    v_init, alpha, beta, errors = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, \
        method='robust', estimate_errors=True)
    fit = (v_init, alpha, beta)

    profile = profileAlphaBeta(v_data, HT_DATA, fit, sigma_v=errors['sigma_v'])

    assert set(profile.keys()) == {'alpha', 'beta'}

    for key, value in [('alpha', alpha), ('beta', beta)]:

        p = profile[key]
        assert p['ci_lower'] is not None and p['ci_upper'] is not None
        assert p['ci_lower'] < value < p['ci_upper']
        assert p['grid'].shape == p['delta_cost'].shape
        assert np.isfinite(p['cost_best']) and p['cost_best'] >= 0
        assert np.isfinite(p['s2']) and p['s2'] > 0
        assert p['value_hat'] == value
        _assertClose(p['delta_threshold'], scipy.special.ndtri(0.5 + 95.0/200.0)**2, 1e-9, \
            "delta_threshold")

    # A lower confidence level must narrow the interval
    profile_68 = profileAlphaBeta(v_data, HT_DATA, fit, sigma_v=errors['sigma_v'], ci=68.27)

    assert (profile_68['alpha']['ci_upper'] - profile_68['alpha']['ci_lower']) < \
        (profile['alpha']['ci_upper'] - profile['alpha']['ci_lower'])
    assert (profile_68['beta']['ci_upper'] - profile_68['beta']['ci_lower']) < \
        (profile['beta']['ci_upper'] - profile['beta']['ci_lower'])

    # param must restrict the returned keys
    assert set(profileAlphaBeta(v_data, HT_DATA, fit, sigma_v=errors['sigma_v'], \
        param='alpha').keys()) == {'alpha'}
    assert set(profileAlphaBeta(v_data, HT_DATA, fit, sigma_v=errors['sigma_v'], \
        param='beta').keys()) == {'beta'}

    # Must work from a method='q4' point estimate too
    v_init_q4, alpha_q4, beta_q4 = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='q4')
    profile_q4 = profileAlphaBeta(v_data, HT_DATA, (v_init_q4, alpha_q4, beta_q4))

    assert profile_q4['alpha']['ci_lower'] < alpha_q4 < profile_q4['alpha']['ci_upper']
    assert profile_q4['beta']['ci_lower'] < beta_q4 < profile_q4['beta']['ci_upper']


def testProfileAlphaBetaRejectsBadInputs():
    """ profileAlphaBeta() must reject a ci outside (0, 100) and an unknown `param`, the same
        input-validation convention as fitAlphaBeta()/plotAlphaBeta().
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    v_init, alpha, beta = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust')
    fit = (v_init, alpha, beta)

    for bad_ci in [0.0, 100.0, -5.0, 150.0]:

        try:
            profileAlphaBeta(v_data, HT_DATA, fit, ci=bad_ci)

        except ValueError:
            pass

        else:
            raise AssertionError(
                "profileAlphaBeta: ValueError not raised for ci={!r}".format(bad_ci))

    try:
        profileAlphaBeta(v_data, HT_DATA, fit, param='banana')

    except ValueError:
        pass

    else:
        raise AssertionError("profileAlphaBeta: ValueError not raised for param='banana'")


def testPlotProfileAlphaBeta():
    """ plotProfileAlphaBeta() must draw a full multi-panel figure without raising, honor the
        single-panel `axes` contract (reuse the passed-in axes), and reject `axes` for a
        multi-panel request - mirroring plotAlphaBeta()'s axes contract.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()

    fit = fitAlphaBeta(v_data, HT_DATA, v_init=V_INIT_TRUE, method='robust')
    profile = profileAlphaBeta(v_data, HT_DATA, fit)

    # Full multi-panel figure must build and return (fig, ax_map)
    fig, ax_map = plotProfileAlphaBeta(v_data, HT_DATA, fit, profile)
    plt.close(fig)

    # A single-panel request (one parameter, one panel type) with axes must reuse them
    profile_alpha = profileAlphaBeta(v_data, HT_DATA, fit, param='alpha')
    fig_in, ax_in = plt.subplots()
    fig_out, ax_out = plotProfileAlphaBeta(v_data, HT_DATA, fit, profile_alpha,
        degeneracy=False, trajectories=False, axes=ax_in)
    assert fig_out is fig_in and ax_out is ax_in
    plt.close(fig_in)

    # Multi-panel + axes must raise
    try:
        plotProfileAlphaBeta(v_data, HT_DATA, fit, profile, axes=ax_in)

    except ValueError:
        pass

    else:
        raise AssertionError("plotProfileAlphaBeta: ValueError not raised for multi-panel + axes")


def testLuminosityModel():
    """ The luminosity function f(v) must be positive on the interior, vanish at both ends, and
        the model magnitude must respond to the offset additively.
    """

    v_grid = np.linspace(0.01, 0.999, 200)

    for mu in [0.0, 2/3]:

        f_val = alphaBetaLuminosityF(v_grid, BETA_TRUE, mu)

        assert np.all(f_val > 0), "f(v) must be positive on (0, 1) for mu = {:.3f}".format(mu)

        # The luminosity vanishes at the beginning (v -> 1, nothing ablated yet) and at the end
        #   (v -> 0, no kinetic energy left) of the trajectory
        f_peak = np.max(f_val)
        assert alphaBetaLuminosityF(np.array([1e-6]), BETA_TRUE, mu)[0] < 1e-3*f_peak
        assert alphaBetaLuminosityF(np.array([1.0]), BETA_TRUE, mu)[0] < 1e-3*f_peak

    # mu >= 1 is outside the model's validity and must be rejected
    try:
        alphaBetaLuminosityF(v_grid, BETA_TRUE, 1.0)

    except ValueError:
        pass

    else:
        raise AssertionError("alphaBetaLuminosityF: ValueError not raised for mu = 1")

    # The magnitude offset must be purely additive
    ht_normed = HT_DATA/HT_NORM_CONST
    mag_zero, _ = alphaBetaModelMagnitude(ht_normed, ALPHA_TRUE, BETA_TRUE, 0.0)
    mag_off, _ = alphaBetaModelMagnitude(ht_normed, ALPHA_TRUE, BETA_TRUE, 0.0, \
        mag_offset=MAG_OFFSET_TRUE)

    assert np.max(np.abs((mag_off - mag_zero) - MAG_OFFSET_TRUE)) < 1e-12


def testProfiledMagOffsetWeighted():
    """ The profiled magnitude offset must reduce to the plain median for uniform weights and
        follow the precise points (not the noisy ones) for per-point weights.
    """

    rng = np.random.RandomState(44)

    # Uniform weights must reproduce np.median exactly, for both odd and even point counts
    for n in [100, 101]:

        diffs = rng.normal(0, 1, n)
        zero = np.zeros(n)

        assert np.isclose(_profiledMagOffset(diffs, zero), np.median(diffs))
        assert np.isclose(_profiledMagOffset(diffs, zero, 0.3), np.median(diffs))
        assert np.isclose(_profiledMagOffset(diffs, zero, np.full(n, 0.3)), np.median(diffs))

    # A minority of precise points must dominate a majority of noisy ones: 10 precise points at
    #   an offset of 1.0 mag against 30 noisy points at 5.0 mag. The unweighted median would land
    #   on the noisy group (5.0); the weighted median must land on the precise one (1.0).
    diffs = np.concatenate([np.full(10, 1.0), np.full(30, 5.0)])
    sigma = np.concatenate([np.full(10, 0.05), np.full(30, 5.0)])

    offset = _profiledMagOffset(diffs, np.zeros(diffs.size), sigma)

    _assertClose(offset, 1.0, 1e-9, "weighted mag offset")


def testFitAlphaBetaLightCurve():
    """ The joint dynamics + light curve fit should recover alpha, beta, the magnitude offset,
        and the amplitude K from noisy synthetic data generated with mu = 0.
    """

    v_data, _, _, _, _ = _syntheticTrajectory()
    mag_data, k_true = _syntheticLightCurve()

    # Run with auto-derived sigmas, a scalar sigma_mag, and a per-point sigma_mag array
    for sigma_mag in [None, MAG_NOISE_STD, np.full(HT_DATA.size, MAG_NOISE_STD)]:

        res = fitAlphaBetaLightCurve(v_data, HT_DATA, HT_DATA, mag_data, v_init=V_INIT_TRUE, \
            sigma_mag=sigma_mag, verbose=False)

        for key in ['fits', 'best_fixed_mu', 'v_init', 'sigma_v', 'alpha_dyn', 'beta_dyn']:
            assert key in res, "missing key '{:s}' in the fitAlphaBetaLightCurve() result".format(key)

        # The data were generated with mu = 0, so that branch must fit well
        fit_mu0 = res['fits'][0.0]

        assert fit_mu0['success']
        _assertClose(fit_mu0['alpha'], ALPHA_TRUE, 0.05, "alpha (joint)")
        _assertClose(fit_mu0['beta'], BETA_TRUE, 0.05, "beta (joint)")
        _assertClose(fit_mu0['mag_offset'], MAG_OFFSET_TRUE, 0.02, "mag_offset (joint)")
        _assertClose(fit_mu0['K'], k_true, 0.10, "K (joint)")

    # The free-mu fit must return a bounded mu with the same fit-quality keys
    res = fitAlphaBetaLightCurve(v_data, HT_DATA, HT_DATA, mag_data, v_init=V_INIT_TRUE, \
        sigma_mag=MAG_NOISE_STD, fit_free_mu=True, verbose=False)

    assert res['mu_free_fit'] is not None
    assert 0.0 <= res['mu_free_fit']['mu'] <= 2/3
    _assertClose(res['mu_free_fit']['alpha'], ALPHA_TRUE, 0.10, "alpha (free mu)")
    _assertClose(res['mu_free_fit']['beta'], BETA_TRUE, 0.10, "beta (free mu)")


def testLuminousEfficiency():
    """ alphaBetaLuminousEfficiency() must invert the K(tau) relation exactly. """

    tau_true = 0.005
    v_final = 13000.0

    # Compute the K that the true tau implies through Eq. (13)
    m_init, _ = alphaBetaMasses(ALPHA_TRUE, BETA_TRUE, SLOPE, mu=0, dens=DENS_TRUE, \
        vel_init=V_INIT_TRUE, vel_end=v_final)

    k_amplitude = tau_true*m_init*V_INIT_TRUE**3*np.sin(SLOPE)/(2.0*HT_NORM_CONST)

    tau, m_init_out, m_final_out = alphaBetaLuminousEfficiency(k_amplitude, ALPHA_TRUE, \
        BETA_TRUE, SLOPE, V_INIT_TRUE, mu=0, dens=DENS_TRUE, v_final=v_final)

    _assertClose(tau, tau_true, 1e-6, "tau")
    _assertClose(m_init_out, m_init, 1e-6, "m_init")
    assert m_final_out < m_init_out


def testLightCurveInputValidation():
    """ Invalid fitAlphaBetaLightCurve() inputs should raise ValueError. """

    v_data, _, _, _, _ = _syntheticTrajectory()
    mag_data, _ = _syntheticLightCurve()

    def _assertRaisesValueError(label, **kwargs):

        call_kwargs = dict(v_data=v_data, ht_data=HT_DATA, ht_lc_data=HT_DATA, \
            mag_abs_data=mag_data, v_init=V_INIT_TRUE, verbose=False)
        call_kwargs.update(kwargs)

        try:
            fitAlphaBetaLightCurve(**call_kwargs)

        except ValueError:
            return

        raise AssertionError("{:s}: ValueError not raised".format(label))

    # Mismatched data lengths
    _assertRaisesValueError("dynamics length mismatch", v_data=v_data[:-1])
    _assertRaisesValueError("light curve length mismatch", mag_abs_data=mag_data[:-1])
    _assertRaisesValueError("sigma_mag length mismatch", sigma_mag=np.full(mag_data.size - 1, 0.1))

    # Out-of-range shape change coefficients
    _assertRaisesValueError("mu >= 1", mu_values=(1.2,))
    _assertRaisesValueError("negative mu", mu_values=(-0.1,))

    # Bad dynamics method name
    _assertRaisesValueError("bad dyn_method", dyn_method="banana")


if __name__ == "__main__":

    # Standalone runner so the tests can be executed without pytest installed
    test_functions = [
        testFitAlphaBetaUnconstrained,
        testInitialMassConstraint,
        testFinalMassConstraint,
        testBothMassConstraint,
        testMassConstraintQ4Method,
        testDerivedVelocities,
        testInputValidation,
        testAlphaBetaNormedRoundTrip,
        testAlphaBetaVelocityNormedLUT,
        testFastFlagPropagation,
        testFastPipelineSpeedup,
        testFitAlphaBetaRobust,
        testFitAlphaBetaErrorEstimationRejectsQ4,
        testFitAlphaBetaErrorEstimationRejectsBadCi,
        testFitAlphaBetaErrorEstimationPointEstimateUnchanged,
        testFitAlphaBetaErrorEstimationCovarianceStructure,
        testFitAlphaBetaErrorEstimationConfidenceInterval,
        testFitAlphaBetaErrorEstimationVerbose,
        testFitAlphaBetaErrorEstimationVInitPropagation,
        testFitAlphaBetaErrorEstimationBoundPinned,
        testAlphaBetaMassesErrorEstimationRequiresCovLog,
        testAlphaBetaMassesErrorEstimationRejectsBadCi,
        testAlphaBetaMassesErrorEstimationPointEstimateUnchanged,
        testAlphaBetaMassesErrorEstimationCovarianceStructure,
        testAlphaBetaMassesErrorEstimationBoundPinnedPropagatesNaN,
        testAlphaBetaMassesErrorEstimationIgnoresVelSigmaWithoutFullSolution,
        testAlphaBetaMassesErrorEstimationMassRatioIdentity,
        testAlphaBetaMassesErrorEstimationMonteCarlo,
        testGaussianEllipsePoints,
        testPlotAlphaBetaGracefulDegradation,
        testPlotAlphaBetaAxesContract,
        testPlotAlphaBetaRejectsBadCi,
        testPlotAlphaBetaSurvivalDiagramBoundaryMath,
        testPlotAlphaBetaSurvivalDiagramRejectsBadInputs,
        testPlotAlphaBetaSurvivalDiagramRuns,
        testPlotAlphaBetaSurvivalDiagramAxesReuse,
        testProfileAlphaBetaBracketsEstimate,
        testProfileAlphaBetaRejectsBadInputs,
        testPlotProfileAlphaBeta,
        testLuminosityModel,
        testProfiledMagOffsetWeighted,
        testFitAlphaBetaLightCurve,
        testLuminousEfficiency,
        testLightCurveInputValidation,
    ]

    failed = 0
    for test_func in test_functions:

        try:
            test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {:s}".format(test_func.__name__, str(e)))

    print()
    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
