""" Tests for the REBOUND close-encounter, divergence and dynamical-classification numerics.

These cover the parts that are pure numerics and need no integration, so they run without REBOUND
or REBOUNDx installed, plus the Hill-radius table's internal consistency.
"""

import os
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_numerics", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _mkrow(t_internal, pos_x, a=1.0):
    """ Build one output row of the form [time, state_vect_hel, orbit, planet_dists]. """

    orbit = SimpleNamespace(a=a, e=0.1, inc=0.0, Omega=0.0, omega=0.0, f=0.0)

    return [t_internal, [pos_x, 0.0, 0.0, 0.0, 0.0, 0.0], orbit, {}]


def _series(n_samples, sep_func, a_func=None, day_step=1.0, helio_r=1.0):
    """ Build a nominal run and a set of realizations whose separation follows sep_func(day).

    The nominal solution sits at a realistic heliocentric distance, because the compactness test
    that bounds the fit window compares the ensemble spread against that distance.
    """

    year = 2*np.pi
    times = [k*day_step/365.25*year for k in range(n_samples)]

    nominal = [_mkrow(t, helio_r) for t in times]

    outputs_mc = {}
    for i in range(6):
        rows = []
        for k, t in enumerate(times):
            day = k*day_step
            scale = 1.0 + 0.1*i
            a_val = 1.0 if a_func is None else 1.0 + a_func(day)*scale
            rows.append(_mkrow(t, helio_r + sep_func(day)*scale, a=a_val))
        outputs_mc["mc%d" % i] = rows

    return nominal, outputs_mc


### Close-encounter detection from the tracked minima ###

def testEncounterFlaggedOnlyInsideThreshold(reb):
    """ A body is reported only when its closest approach is inside n_hill Hill radii. """

    hill_earth = reb.HILL_RADII_AU["Earth"]

    min_dist = {"Earth": 2.0*hill_earth, "Jupiter": 100.0}
    min_time = {"Earth": -5.0, "Jupiter": -10.0}

    encounters = reb.encountersFromMinDistances(min_dist, min_time, n_hill=3.0)

    assert [e["body"] for e in encounters] == ["Earth"]
    assert encounters[0]["n_hill"] == pytest.approx(2.0)
    assert encounters[0]["time_days"] == -5.0

    # Tightening the threshold below the approach must drop it
    assert reb.encountersFromMinDistances(min_dist, min_time, n_hill=1.0) == []


def testEncountersSortedByClosenessAndUntrackedSkipped(reb):
    """ Encounters are ordered by closeness in Hill radii, and untracked bodies are ignored. """

    min_dist = {
        "Earth": 2.5*reb.HILL_RADII_AU["Earth"],
        "Luna": 0.5*reb.HILL_RADII_AU["Luna"],
        "Mars": 1.5*reb.HILL_RADII_AU["Mars"],
        "Venus": None,
    }
    min_time = {"Earth": -1.0, "Luna": -2.0, "Mars": -3.0, "Venus": None}

    encounters = reb.encountersFromMinDistances(min_dist, min_time, n_hill=3.0)

    assert [e["body"] for e in encounters] == ["Luna", "Mars", "Earth"]
    assert "Venus" not in [e["body"] for e in encounters]


### Closest approach inside one integrator step ###

def _hyperbolicFlyby(t, mu, q, e):
    """ Two-body hyperbolic flyby (periapsis q at t = 0): relative position and velocity at t. """

    a = q/(e - 1)
    n = np.sqrt(mu/a**3)
    M = n*t

    # Solve the hyperbolic Kepler equation e*sinh(F) - F = M
    F = np.arcsinh(M/e)
    for _ in range(50):
        F -= (e*np.sinh(F) - F - M)/(e*np.cosh(F) - 1)

    F_dot = n/(e*np.cosh(F) - 1)
    k = np.sqrt(e**2 - 1)
    r = np.array([a*(e - np.cosh(F)), a*k*np.sinh(F), 0.0])
    v = np.array([-a*np.sinh(F)*F_dot, a*k*np.cosh(F)*F_dot, 0.0])

    return r, v


def testHermiteFindsStraightLineMinimumInsideALongStep(reb):
    """ A step spanning the whole flyby: the step ends are far off, the interpolant is exact. """

    b = np.array([0.0, 1.0, 0.0])
    v = np.array([2.0, 0.0, 0.0])
    t0, t1 = -3.0, 5.0

    d, t = reb.hermiteClosestApproach(t0, b + v*t0, v, t1, b + v*t1, v)

    assert d == pytest.approx(1.0, rel=1e-12)
    assert t == pytest.approx(0.0, abs=1e-12)

    # The step-end samples alone overestimate the minimum by a factor of ~6
    assert min(np.linalg.norm(b + v*t0), np.linalg.norm(b + v*t1)) > 6.0


def testHermiteWorksForBackwardSteps(reb):
    """ Backward integration steps (t1 < t0) give the same minimum and time. """

    b = np.array([0.0, 0.0, 0.5])
    v = np.array([-1.0, 1.0, 0.0])
    t0, t1 = 4.0, -2.5

    d, t = reb.hermiteClosestApproach(t0, b + v*t0, v, t1, b + v*t1, v)

    assert d == pytest.approx(0.5, rel=1e-12)
    assert t == pytest.approx(0.0, abs=1e-12)


def testHermiteReturnsTheStepEndWhenThereIsNoMinimumInside(reb):
    """ A receding step has no interior minimum, so the closer step end is returned. """

    b = np.array([0.0, 1.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    t0, t1 = 1.0, 3.0

    d, t = reb.hermiteClosestApproach(t0, b + v*t0, v, t1, b + v*t1, v)

    assert d == pytest.approx(np.sqrt(2.0), rel=1e-12)
    assert t == t0


@pytest.mark.parametrize("ecc, tol", [(1000.0, 1e-4), (20.0, 5e-3), (3.0, 2e-2)])
def testHermiteOnAKeplerianFlyby(reb, ecc, tol):
    """ On a real (bent) hyperbolic flyby spanned by a step of ~1.6 encounter times, the refined
    minimum is close to the true periapsis and far better than the step-end samples. Deflection
    makes the interpolant slightly underestimate it; the error shrinks as the flyby gets straighter.
    """

    mu, q = 1.0, 1.0
    tau = q/np.sqrt(mu*(ecc + 1)/q)
    t0, t1 = -0.7*tau, 0.9*tau

    r0, v0 = _hyperbolicFlyby(t0, mu, q, ecc)
    r1, v1 = _hyperbolicFlyby(t1, mu, q, ecc)

    d, t = reb.hermiteClosestApproach(t0, r0, v0, t1, r1, v1)

    assert abs(d/q - 1) < tol
    assert abs(t/tau) < 2*tol
    assert min(np.linalg.norm(r0), np.linalg.norm(r1))/q - 1 > 0.1

### MEGNO verdict ###

def _megnoSeries(y_func, years=1000.0, a_au=1.0, n=200):
    """ A synthetic MEGNO series <Y>(t) over the given span, with a constant semi-major axis. """

    t_days = np.linspace(years*365.25/n, years*365.25, n)
    return list(t_days), [y_func(t/365.25) for t in t_days], [a_au]*n


def testMegnoConvergedToTwoIsRegular(reb):
    """ A series settling at 2 (with a small decaying wiggle) is regular. """

    v = reb.classifyMegno(*_megnoSeries(lambda t: 2.0 + 0.3*np.exp(-t/50.0)*np.cos(t)))

    assert v["status"] == "regular"
    assert v["lyapunov_time_years"] is None
    assert v["n_orbits"] == pytest.approx(1000.0)


def testMegnoGrowingLinearlyIsChaoticWithLyapunovTime(reb):
    """ <Y> ~ (lambda/2) t is chaotic, and the fitted Lyapunov time is 1/lambda. """

    lyapunov_time = 80.0
    v = reb.classifyMegno(*_megnoSeries(lambda t: 2.0 + 0.5*t/lyapunov_time))

    assert v["status"] == "chaotic"
    assert v["lyapunov_time_years"] == pytest.approx(lyapunov_time, rel=1e-6)


def testMegnoTendingToZeroIsPeriodic(reb):
    """ A bounded deviation (<Y> -> 0) is a stable periodic orbit, not a converged one. """

    v = reb.classifyMegno(*_megnoSeries(lambda t: 0.4 + 0.1*np.sin(t)))

    assert v["status"] == "periodic"


def testMegnoInBetweenIsNotConverged(reb):
    """ A series sitting at 2.3 is neither converged to 2 nor clearly chaotic. """

    v = reb.classifyMegno(*_megnoSeries(lambda t: 2.3))

    assert v["status"] == "not_converged"


def testMegnoNeedsEnoughOrbits(reb):
    """ Fewer than MEGNO_MIN_ORBITS orbital periods give no verdict, even for a clean series. """

    # a = 10 AU: a 31.6-year period, so 300 years is ~9.5 orbits
    v = reb.classifyMegno(*_megnoSeries(lambda t: 2.0, years=300.0, a_au=10.0))

    assert v["status"] == "too_short"
    assert v["n_orbits"] == pytest.approx(300.0/10.0**1.5)


def testMegnoNotMeaningfulForUnboundOrbits(reb):
    """ A hyperbolic orbit (negative a) gets no MEGNO verdict. """

    v = reb.classifyMegno(*_megnoSeries(lambda t: 2.0, a_au=-3.0))

    assert v["status"] == "unbound"
    assert v["n_orbits"] is None


def testMegnoWithTooFewOutputsGivesNoVerdict(reb):
    """ Fewer than 8 outputs make the last-quarter mean meaningless, so there is no verdict. """

    v = reb.classifyMegno(*_megnoSeries(lambda t: 2.0, n=5))

    assert v["status"] == "too_short"
    assert v["n_orbits"] is None, "the orbit count is not computed without a verdict"


def testMegnoWithNoOutputsIsHandled(reb):
    """ An empty series (the object escaped before the first output) must not raise. """

    v = reb.classifyMegno([], [], [])

    assert v["status"] == "too_short"
    assert v["final"] is None
    assert v["last_quarter"] is None


### MEGNO report lines ###

def _megnoResult(reb, y_func, **kwargs):
    """ A MEGNO result dict with its verdict, in the form reboundSimulate stores it. """

    times_days, megno, a_au = _megnoSeries(y_func, **kwargs)

    return {"times_days": times_days, "megno": megno, "a_au": a_au, "escaped": False,
            "verdict": reb.classifyMegno(times_days, megno, a_au)}


def testMegnoReportStatesWhetherItConvergesToTwo(reb):
    """ A regular orbit is reported as converging to 2, a chaotic one as not, with a Lyapunov time. """

    regular = reb._megnoReportLines(_megnoResult(reb, lambda t: 2.0))
    chaotic = reb._megnoReportLines(_megnoResult(reb, lambda t: 2.0 + 0.5*t/80.0))

    assert any("Converges to 2: YES" in line for line in regular)
    assert not any("Lyapunov" in line for line in regular)

    assert any("Converges to 2: NO" in line for line in chaotic)
    assert any("Lyapunov time ~ 80 years" in line for line in chaotic)


def testMegnoReportGivesNoVerdictForAnUnboundOrbit(reb):
    """ The statuses without a verdict are reported as such, not as "does not converge". """

    lines = reb._megnoReportLines(_megnoResult(reb, lambda t: 2.0, a_au=-3.0))

    assert any("No verdict" in line for line in lines)
    assert not any("Converges to 2" in line for line in lines)


def testMegnoReportHandlesAnEmptySeries(reb):
    """ An empty series is reported in one line instead of raising on a missing value. """

    megno = {"times_days": [], "megno": [], "a_au": [], "escaped": True,
             "verdict": reb.classifyMegno([], [], [])}
    lines = reb._megnoReportLines(megno)

    assert any("no MEGNO values" in line for line in lines)


### Earth-departure gating ###

def testDepartureIndexFoundAndGatingExcludesTheStart(reb):
    """ The object starts at the Earth, so only a genuine later approach may be reported. """

    gate = 3.0*reb.HILL_RADII_AU["Earth"]

    bodies = ["Sun", "Mercury", "Venus", "Earth", "Luna", "Mars",
              "Jupiter", "Saturn", "Uranus", "Neptune"]

    # Starts at the Earth, leaves, then comes back inside the gate
    earth_track = [1e-5, 0.5*gate, 2.0*gate, 4.0*gate, 0.5*gate, 4.0*gate]

    rows = []
    for k, d_earth in enumerate(earth_track):
        dists = {b: 10.0 for b in bodies}
        dists["Earth"] = d_earth
        dists["Luna"] = 5.0
        rows.append([k*0.5, None, None, dists])

    idx = reb.findEarthDepartureIndex(rows, n_hill=3.0)
    assert idx == 2, "departure is the first sample beyond the gate"

    encounters = reb.detectCloseEncounters(rows, n_hill=3.0)
    earth = [e for e in encounters if e["body"] == "Earth"]

    assert earth, "the genuine return must be reported"
    assert earth[0]["index"] == 4, "the reported approach must be the return, not the start"


def testNoDepartureMeansOnlyTheMoonIsChecked(reb):
    """ If the object never leaves the Earth's neighbourhood, only the Moon may be reported. """

    bodies = ["Sun", "Mercury", "Venus", "Earth", "Luna", "Mars",
              "Jupiter", "Saturn", "Uranus", "Neptune"]

    rows = []
    for k in range(5):
        dists = {b: 10.0 for b in bodies}
        dists["Earth"] = 1e-4
        dists["Luna"] = 0.5*reb.HILL_RADII_AU["Luna"]
        rows.append([k*0.5, None, None, dists])

    assert reb.findEarthDepartureIndex(rows, n_hill=3.0) is None
    assert {e["body"] for e in reb.detectCloseEncounters(rows, n_hill=3.0)} == {"Luna"}


### Divergence / Lyapunov estimate ###

def testExponentialDivergenceRecoversLyapunovTime(reb):
    """ A synthetic exponentially diverging ensemble must give back its own e-folding time. """

    for t_true in (20.0, 50.0, 150.0):

        nominal, mc = _series(300, lambda day: 1e-8*np.exp(day/t_true))
        result = reb.estimateLyapunovFromMC(nominal, mc)

        assert result["growth"] == "exponential"
        assert result["lyapunov_time_days"] == pytest.approx(t_true, rel=1e-3)


def testLinearDivergenceIsNotCalledChaotic(reb):
    """ Linear (regular) growth must not be reported as an exponential divergence. """

    nominal, mc = _series(300, lambda day: 1e-8*(1.0 + day))
    result = reb.estimateLyapunovFromMC(nominal, mc)

    assert result["growth"] == "linear"
    assert result["lyapunov_time_days"] is None


def testTruncatedRealizationDoesNotDestroyTheEstimate(reb):
    """ One realization ending early (e.g. an impact) must not shorten the whole analysis. """

    nominal, mc = _series(200, lambda day: 1e-8*(1.0 + day))

    # One realization stops after three samples
    mc["mc0"] = mc["mc0"][:3]

    result = reb.estimateLyapunovFromMC(nominal, mc)

    assert result is not None, "a single short realization must not void the estimate"
    assert result["n_truncated"] == 1
    assert result["n_samples_used"] > 150, "the rest of the ensemble must still be used"


def testFitFallsBackToTheCompactWindow(reb):
    """ A run whose cloud saturates is fitted over its compact leading window, not rejected. """

    # Separation blows past 10% of the 1 AU heliocentric distance partway through
    nominal, mc = _series(400, lambda day: 1e-4*np.exp(day/40.0), day_step=1.0)
    result = reb.estimateLyapunovFromMC(nominal, mc)

    assert result is not None
    assert result["truncated_at_saturation"], "the fit must stop at saturation"
    assert result["fit_window_days"] < result["total_span_days"]
    assert result["growth"] == "exponential"
    assert result["lyapunov_time_days"] == pytest.approx(40.0, rel=1e-2)


def testElementDivergenceSeparatesPhaseDriftFromChaos(reb):
    """ A constant spread in a means regular motion, even when positions diverge strongly. """

    # Positions spread linearly (phase drift), but the semi-major axes stay put
    nominal, mc = _series(300, lambda day: 1e-6*(1.0 + day), a_func=lambda day: 1e-3)
    result = reb.estimateLyapunovFromMC(nominal, mc)

    element = result["element_divergence"]
    assert element is not None
    assert element["growth"] == "regular"
    assert element["sigma_a_growth_factor"] == pytest.approx(1.0, abs=1e-6)


def testElementDivergenceDetectsGrowingSpreadInA(reb):
    """ An exponentially growing spread in a is reported as an exponential divergence. """

    nominal, mc = _series(300, lambda day: 1e-6*(1.0 + day),
                          a_func=lambda day: 1e-6*np.exp(day/60.0))
    result = reb.estimateLyapunovFromMC(nominal, mc)

    element = result["element_divergence"]
    assert element["growth"] == "exponential"
    assert element["lyapunov_time_days"] == pytest.approx(60.0, rel=1e-2)


def testTooFewRealizationsGivesNoEstimate(reb):
    """ A divergence estimate needs an ensemble, not a single realization. """

    nominal, mc = _series(50, lambda day: 1e-8*(1.0 + day))
    single = {"mc0": mc["mc0"]}

    assert reb.estimateLyapunovFromMC(nominal, single) is None
    assert reb.estimateLyapunovFromMC([], mc) is None


### Tisserand parameter ###

@pytest.mark.parametrize("name,a,e,inc_deg,expected", [
    ("2P/Encke", 2.215, 0.848, 11.78, 3.03),
    ("1P/Halley", 17.834, 0.967, 162.26, -0.61),
    ("Ceres", 2.766, 0.0785, 10.59, 3.31),
])
def testTisserandMatchesPublishedValues(reb, name, a, e, inc_deg, expected):
    """ The Tisserand parameter must reproduce the accepted values for well-known objects. """

    t_j = reb.tisserandParameterJupiter(a, e, np.radians(inc_deg))

    assert t_j == pytest.approx(expected, abs=0.02), name


def testTisserandUndefinedForUnboundOrbits(reb):
    """ The parameter is not defined for an unbound orbit, and must not be invented. """

    assert reb.tisserandParameterJupiter(2.0, 1.2, 0.0) is None
    assert reb.tisserandParameterJupiter(-3.0, 0.5, 0.0) is None
    assert "undefined" in reb.tisserandClass(None)


def testTisserandClassBoundaries(reb):
    """ The conventional class boundaries sit at T_J = 2 and T_J = 3. """

    assert "asteroidal" in reb.tisserandClass(3.5)
    assert "Jupiter-family" in reb.tisserandClass(2.5)
    assert "Halley" in reb.tisserandClass(1.5)


### Radiation-pressure beta ###

def testRadiationBetaMatchesTheAnalyticFormula(reb):
    """ Beta must follow 5.7425e-4/(rho*s) and reproduce the known micron-grain value. """

    # A 1 micron grain at 3000 kg/m^3 has beta ~ 0.19
    assert reb.radiationPressureBeta(1e-6, 3000.0) == pytest.approx(0.1914, rel=1e-3)

    # A 1 cm meteoroid is essentially unaffected
    assert reb.radiationPressureBeta(1e-2, 3000.0) == pytest.approx(1.914e-5, rel=1e-3)

    # Beta scales inversely with both size and density
    assert reb.radiationPressureBeta(2e-6, 3000.0) == pytest.approx(
        reb.radiationPressureBeta(1e-6, 3000.0)/2)
    assert reb.radiationPressureBeta(1e-6, 6000.0) == pytest.approx(
        reb.radiationPressureBeta(1e-6, 3000.0)/2)


def testRadiationBetaRejectsUnphysicalInput(reb):
    """ A non-positive size or density is an error, not a silent NaN. """

    for radius, density in [(0.0, 3000.0), (-1e-6, 3000.0), (1e-6, 0.0), (1e-6, -10.0)]:
        with pytest.raises(ValueError):
            reb.radiationPressureBeta(radius, density)


### Hill-radius table ###

def testHillRadiiAreSelfConsistent(reb):
    """ The tabulated Hill radii must match r_H = a*(m/(3*M_sun))**(1/3) for the code's own masses. """

    semi_major_axes = {
        "Mercury": 0.387098, "Venus": 0.723332, "Earth": 1.000000, "Mars": 1.523679,
        "Jupiter": 5.204267, "Saturn": 9.582017, "Uranus": 19.229411, "Neptune": 30.103658,
    }

    for body, a in semi_major_axes.items():
        mass = reb.reboundBodyMassSolar(reb._EPHEM_MASS_NAIF[body])
        expected = a*(mass/3.0)**(1.0/3.0)

        assert reb.HILL_RADII_AU[body] == pytest.approx(expected, rel=1e-3), body

    # The Moon's value is relative to the Earth, not the Sun
    m_moon = reb.reboundBodyMassSolar(301)
    m_earth = reb.reboundBodyMassSolar(399)
    a_moon = 384400.0/149597870.7

    assert reb.HILL_RADII_AU["Luna"] == pytest.approx(
        a_moon*(m_moon/(3.0*m_earth))**(1.0/3.0), rel=1e-3)


def testMoonEncounterIsGeometricallyReachableOnlyNearEarth(reb):
    """ A lunar encounter can only happen well inside the Earth-departure gate.

    This is why the Moon is tracked over the whole integration while every other body is gated to
    after the departure: if the Moon were gated the same way, a lunar encounter would be impossible
    to detect.
    """

    moon_apogee_au = 405500.0/149597870.7
    lunar_reach = moon_apogee_au + 3.0*reb.HILL_RADII_AU["Luna"]
    departure_gate = 3.0*reb.HILL_RADII_AU["Earth"]

    assert lunar_reach < departure_gate

    # The starting point at the Earth can never lie inside the Moon's detection sphere
    moon_perigee_au = 362600.0/149597870.7
    earth_radius_au = 6371.0/149597870.7

    assert (moon_perigee_au - earth_radius_au) > 3.0*reb.HILL_RADII_AU["Luna"]
