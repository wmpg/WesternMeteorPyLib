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

    days_per_unit = np.sqrt(149597870700.0**3/1.3271244004193938e20)/86400.0
    times = [k*day_step/days_per_unit for k in range(n_samples)]

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


def testEncountersFromMinDistancesIncludesTheSun(reb):
    """ The Sun is gated on SUN_ENCOUNTER_AU, carries no Hill radius, and sorts by its own threshold.

    The per-particle "encounters" diagnostic already lists Sun passages, so the list rebuilt from
    the closest-approach dictionaries must agree with it rather than silently drop the Sun.
    """

    min_dist = {"Sun": 0.5*reb.SUN_ENCOUNTER_AU, "Earth": 2.5*reb.HILL_RADII_AU["Earth"],
                "Luna": 0.9*reb.HILL_RADII_AU["Luna"]}
    min_time = {"Sun": -40.0, "Earth": -1.0, "Luna": -2.0}

    encounters = reb.encountersFromMinDistances(min_dist, min_time, n_hill=3.0)

    # Luna at 0.3 of its threshold, the Sun at 0.5 of its own, the Earth at 0.83 of its own
    assert [e["body"] for e in encounters] == ["Luna", "Sun", "Earth"]

    sun = encounters[1]
    assert sun["min_dist_au"] == pytest.approx(0.5*reb.SUN_ENCOUNTER_AU)
    assert sun["time_days"] == -40.0
    assert (sun["hill_radius_au"] is None) and (sun["n_hill"] is None)

    # Outside the fixed distance the Sun is not an encounter, whatever n_hill is
    min_dist["Sun"] = 1.5*reb.SUN_ENCOUNTER_AU
    assert "Sun" not in [e["body"] for e in reb.encountersFromMinDistances(min_dist, min_time, n_hill=30.0)]


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
    assert v["lyapunov_method"] == "megno_second_half"


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



### Lyapunov time from ln|delta| ###


def _megnoFromLnDelta(t_years, ln_delta):
    """ <Y> from ln|delta| with the integrals of computeMegno (midpoint rule), starting at t = 0. """

    t = np.concatenate([[0.0], t_years])
    l = np.concatenate([[ln_delta[0]], ln_delta])
    I = np.concatenate([[0.0], np.cumsum(0.5*(t[1:] + t[:-1])*np.diff(l))])
    Y = np.where(t > 0, 2*I/np.where(t > 0, t, 1), 0.0)
    J = np.concatenate([[0.0], np.cumsum(0.5*(Y[1:] + Y[:-1])*np.diff(t))])

    return (J/np.where(t > 0, t, 1))[1:]


def _lnDeltaSeries(reb, ln_func, years=1000.0, a_au=1.0, n=2000):
    """ Times (days), <Y>, a and ln|delta| for a given ln|delta|(t in years). """

    t_years = np.linspace(years/n, years, n)
    ln_delta = np.array([ln_func(t) for t in t_years])

    return list(t_years*365.25), list(_megnoFromLnDelta(t_years, ln_delta)), [a_au]*n, list(ln_delta)


def testLyapunovTimeComesFromTheGrowthOfLnDelta(reb):
    """ A deviation that grows like a regular one (ln t) and then exponentially from 300 years on: the
    Lyapunov time is that of the exponential phase, in every window of it. """

    tau = 40.0
    times, megno, a, ln_delta = _lnDeltaSeries(
        reb, lambda t: np.log(min(t, 300.0)) + (max(t - 300.0, 0.0)/tau))

    v = reb.classifyMegno(times, megno, a, ln_delta=ln_delta)

    assert v["status"] == "chaotic"
    assert v["lyapunov_method"] == "ln_delta"
    assert v["lyapunov_time_years"] == pytest.approx(tau, rel=0.02)
    assert v["lyapunov_time_range_years"][0] == pytest.approx(tau, rel=0.05)
    assert v["lyapunov_time_range_years"][1] == pytest.approx(tau, rel=0.05)
    assert v["lyapunov_onset_years"] > 300.0


def testRegularDeviationGrowthIsRegular(reb):
    """ ln|delta| growing as ln t (a regular orbit) gives <Y> -> 2 and no Lyapunov time. """

    v = reb.classifyMegno(*_lnDeltaSeries(reb, lambda t: np.log(t))[:3],
                          ln_delta=_lnDeltaSeries(reb, lambda t: np.log(t))[3])

    assert v["status"] == "regular"
    assert v["lyapunov_time_years"] is None


def testNoLyapunovTimeFromAShortGrowthPhase(reb):
    """ Exponential growth over fewer than LYAPUNOV_MIN_ORBITS periods is not timed. """

    t = np.linspace(1.0, 1000.0, 2000)
    period = 11.9
    ln_delta = np.log(t) + np.maximum(t - 1000.0 + 0.5*reb.LYAPUNOV_MIN_ORBITS*period, 0.0)/5.0
    megno = 2.0 + np.maximum(t - 900.0, 0.0)

    assert reb.lyapunovFromDeviation(t, ln_delta, megno, period) is None

    # Over enough periods it is timed
    ln_delta = np.log(t) + np.maximum(t - 500.0, 0.0)/5.0
    fit = reb.lyapunovFromDeviation(t, ln_delta, 2.0 + np.maximum(t - 500.0, 0.0), period)
    assert fit["lyapunov_time_years"] == pytest.approx(5.0, rel=0.01)


### Predictability horizon from the clones ###


def _cloneOutputs(n_clones=40, n_out=200, scramble_at=None, seed=0, e_scramble_at=None):
    """ Nominal and clone outputs whose a (and e) keep the order of their start until scramble_at,
    and are shuffled among the clones after it. Time step 1 year (backward). """

    rng = np.random.default_rng(seed)
    day = 365.25
    times = [-k*day for k in range(n_out)]

    def rows(a, e):
        out = []
        for k, t in enumerate(times):
            orbit = SimpleNamespace(a=a[k], e=e[k], inc=0.1 + 0.001*a[k], Omega=0.0, omega=0.0, f=0.0)
            out.append([t*2*np.pi/365.25, None, orbit, {"Earth": 0.0 if k == 0 else 1.0}])
        return out

    a0 = 2.0 + 0.01*rng.normal(size=n_clones)
    e0 = 0.6 + 0.001*rng.normal(size=n_clones)
    a = np.tile(a0, (n_out, 1)) + 1e-5*rng.normal(size=(n_out, n_clones))
    e = np.tile(e0, (n_out, 1))
    if scramble_at is not None:
        for k in range(scramble_at, n_out):
            a[k] = rng.permutation(a0)
    if e_scramble_at is not None:
        for k in range(e_scramble_at, n_out):
            e[k] = rng.permutation(e0)

    nominal = rows(np.full(n_out, 2.0), np.full(n_out, 0.6))
    clones = {"obj_MC_{:d}".format(j): rows(a[:, j], e[:, j]) for j in range(n_clones)}

    return nominal, clones


def testHorizonIsWhereTheClonesLoseTheOrderOfTheirStart(reb):

    nominal, clones = _cloneOutputs(scramble_at=60, e_scramble_at=120)

    h = reb.predictabilityHorizon(nominal, clones)

    assert h["n_clones"] == 40
    assert abs(h["elements"]["a"]["horizon_days"])/365.25 == pytest.approx(59, abs=1)
    assert abs(h["elements"]["e"]["horizon_days"])/365.25 == pytest.approx(119, abs=1)

    # The horizon of the run is the earliest one
    assert abs(h["horizon_days"])/365.25 == pytest.approx(59, abs=1)
    assert abs(h["elements"]["a"]["rank_corr_end"]) < 0.5

    lines = reb._predictabilityReportLines(h, 199*365.25)
    assert any("Beyond ~59 years" in line for line in lines)


def testHorizonNotReachedOnAShortRun(reb):
    """ The default run, 60 days: the clones keep their order, so there is no horizon. """

    nominal, clones = _cloneOutputs(n_out=60)

    h = reb.predictabilityHorizon(nominal, clones)

    assert h["horizon_days"] is None
    assert all(el["horizon_days"] is None for el in h["elements"].values())
    assert h["elements"]["a"]["rank_corr_end"] > 0.99

    lines = reb._predictabilityReportLines(h, 60.0)
    assert any("Not reached within the 60 days" in line for line in lines)


def testHorizonNeedsEnoughClonesOverTheWholeSpan(reb):

    nominal, clones = _cloneOutputs(n_clones=reb.PREDICTABILITY_MIN_CLONES - 1)
    assert reb.predictabilityHorizon(nominal, clones) is None

    # Clones that ended early (impacted) are left out
    nominal, clones = _cloneOutputs(n_clones=reb.PREDICTABILITY_MIN_CLONES + 2)
    for name in list(clones)[:3]:
        clones[name] = clones[name][:50]
    assert reb.predictabilityHorizon(nominal, clones) is None


def testHorizonReportsTheFirstEncounters(reb):
    """ The time of each clone's first close encounter, without the Sun passages. """

    nominal, clones = _cloneOutputs(n_clones=20)
    diag = {}
    for j, name in enumerate(clones):
        encounters = [{"body": "Sun", "time_days": -1.0}]
        if j < 10:
            encounters += [{"body": "Earth", "time_days": -365.25*(10 + j)},
                           {"body": "Luna", "time_days": -365.25*(100 + j)}]
        diag[name] = {"encounters": encounters}

    fe = reb.predictabilityHorizon(nominal, clones, clone_diag=diag)["first_encounter_days"]

    assert fe["fraction"] == pytest.approx(0.5)
    assert fe["p50"]/365.25 == pytest.approx(14.5, abs=0.01)
    assert fe["p05"]/365.25 >= 10.0


### Figure of the orbit ###


def _plotOutputs(n_out=300, years=100.0, direction=-1, n_clones=0, node_start=355.0, seed=1,
                 truncate=None):
    """ Synthetic outputs for plotReboundOrbits: the node crosses 360 deg, the object leaves the Earth
    at the third output, and the clones scatter around the nominal solution. """

    rng = np.random.default_rng(seed)
    t_days = direction*np.linspace(0, years*365.25, n_out)

    # REBOUND time unit in days (DAYS_PER_SIM_TIME)
    days_per_unit = np.sqrt(149597870700.0**3/1.3271244004193938e20)/86400.0

    def rows(da, dnode, length=n_out):
        out = []
        for k in range(length):
            frac = k/(n_out - 1)
            orbit = SimpleNamespace(a=2.0 + da*frac, e=0.6, inc=np.radians(1.0),
                                    Omega=np.radians((node_start + 10.0*frac + dnode*frac) % 360.0),
                                    omega=np.radians(100.0), f=0.0)
            earth = 1e-5 if k < 2 else 1.0
            out.append([t_days[k]/days_per_unit, None, orbit, {"Earth": earth, "Sun": 1.0}])
        return out

    nominal = rows(0.0, 0.0)
    clones = {"obj_MC_{:d}".format(j): rows(0.1*rng.normal(), rng.normal(), length=(
        truncate if (truncate is not None and j == 0) else n_out)) for j in range(n_clones)}

    return nominal, clones


@pytest.fixture
def agg():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
    yield
    plt.close("all")


def testForwardRunShowsTheWholeSpan(reb, agg):
    """ The x axis covers the whole run forward in time, not only the departure from the Earth (the
    old figure set the shared x limit to the departure time). """

    nominal, clones = _plotOutputs(direction=1, n_clones=5)
    fig, axes = reb.plotReboundOrbits(nominal, clones, [])

    lo, hi = axes["a"].get_xlim()
    assert lo == pytest.approx(0.0) and hi == pytest.approx(100.0)
    assert axes["a"].get_xlabel() == "" and axes["varpi"].get_xlabel() == "Time [years]"


def testShortRunIsInDays(reb, agg):

    nominal, clones = _plotOutputs(years=60/365.25, n_clones=3)
    _, axes = reb.plotReboundOrbits(nominal, clones, [])

    assert axes["varpi"].get_xlabel() == "Time [days]"
    assert axes["a"].get_xlim()[0] == pytest.approx(-60.0)


def testManyClonesAreBandsAndTheAxesCoverThem(reb, agg):
    """ With many clones they are drawn as percentile bands, the y axis covers the 2.5-97.5% band,
    and only a sample of clones is drawn individually. """

    nominal, clones = _plotOutputs(n_clones=200)
    _, axes = reb.plotReboundOrbits(nominal, clones, [])

    ax = axes["a"]
    assert len(ax.collections) >= 2

    final_a = np.array([rows[-1][2].a for rows in clones.values()])
    lo, hi = ax.get_ylim()
    assert lo <= np.percentile(final_a, 2.5) and hi >= np.percentile(final_a, 97.5)

    thin = [line for line in ax.get_lines() if line.get_linewidth() < 0.5]
    assert len(thin) == reb.PLOT_SAMPLE_CLONES


def testAnglesAreUnwrapped(reb, agg):
    """ A node crossing 360 deg is drawn as a continuous line, and the clones follow it. """

    nominal, clones = _plotOutputs(n_clones=30, node_start=355.0)
    _, axes = reb.plotReboundOrbits(nominal, clones, [])

    nominal_line = [line for line in axes["node"].get_lines() if line.get_linewidth() > 1][0]
    y = np.array(nominal_line.get_ydata(), dtype=float)
    y = y[np.isfinite(y)]
    assert np.max(np.abs(np.diff(y))) < 1.0
    assert 0.0 <= y[0] < 360.0 and y[-1] > 360.0

    # The longitude of perihelion starts within 0-360 deg as well
    varpi_line = [line for line in axes["varpi"].get_lines() if line.get_linewidth() > 1][0]
    v = np.array(varpi_line.get_ydata(), dtype=float)
    assert 0.0 <= v[np.isfinite(v)][0] < 360.0


def testElementsBeforeTheDepartureAreLeftOut(reb, agg):

    nominal, clones = _plotOutputs(n_clones=0)
    _, axes = reb.plotReboundOrbits(nominal, clones, [])

    y = np.array([line for line in axes["a"].get_lines() if line.get_linewidth() > 1][0].get_ydata(),
                 dtype=float)
    assert np.all(np.isnan(y[:2])) and np.isfinite(y[2])


def testFigureWithoutClonesOrWithAShortClone(reb, agg, tmp_path):
    """ Without clones the last panel is the heliocentric distance; a clone that ended early (an
    impact) is padded, and the figure is saved. """

    nominal, _ = _plotOutputs()
    _, axes = reb.plotReboundOrbits(nominal, {}, [], file_path=str(tmp_path/"a.png"))
    assert axes["predictability"].get_ylabel() == "heliocentric distance [AU]"
    assert (tmp_path/"a.png").exists()

    nominal, clones = _plotOutputs(n_clones=25, truncate=50)
    encounters = [{"body": "Earth", "min_dist_au": 0.01, "time_days": -365.25*30},
                  {"body": "Jupiter", "min_dist_au": 0.9, "time_days": -365.25*60}]
    _, axes = reb.plotReboundOrbits(nominal, clones, encounters, file_path=str(tmp_path/"b.png"))
    labels = axes["encounters"].get_legend_handles_labels()[1]
    assert "encounter: Earth" in labels and "encounter: Jupiter" in labels
