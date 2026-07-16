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

import numpy as np

from wmpl.Utils.AlphaBeta import (fitAlphaBetaMass, fitAlphaBeta, fitAlphaBetaLightCurve,
    alphaBetaMasses, alphaBetaVelocity, alphaBetaHeight, alphaBetaVelocityNormed,
    alphaBetaHeightNormed, alphaBetaLuminosityF, alphaBetaModelMagnitude,
    alphaBetaLuminousEfficiency, _profiledMagOffset, HT_NORM_CONST, P_0M)


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
        testDerivedVelocities,
        testInputValidation,
        testAlphaBetaNormedRoundTrip,
        testFitAlphaBetaRobust,
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
