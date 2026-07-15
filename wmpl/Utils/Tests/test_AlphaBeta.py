""" Tests for the alpha-beta model fitting in wmpl.Utils.AlphaBeta.

The tests generate a synthetic trajectory from known alpha-beta parameters, compute the
corresponding true masses, and check that fitAlphaBetaMass() recovers the true parameters under
all three mass constraints, both with a fixed and a fitted bulk density.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_AlphaBeta.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_AlphaBeta
"""

import numpy as np

from wmpl.Utils.AlphaBeta import fitAlphaBetaMass, fitAlphaBeta, alphaBetaMasses, alphaBetaVelocity


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


if __name__ == "__main__":

    # Standalone runner so the tests can be executed without pytest installed
    test_functions = [
        testFitAlphaBetaUnconstrained,
        testInitialMassConstraint,
        testFinalMassConstraint,
        testBothMassConstraint,
        testDerivedVelocities,
        testInputValidation,
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
