# Classifying a single orbit

**`wmpl.Utils.OrbitClassification`** answers a different question from the
[dissimilarity criteria](OrbitSimilarity.md): not "are these two orbits related?" but "what kind of
orbit is this one?" — asteroidal or cometary, Jupiter family or Halley type, Centaur or
transneptunian, the meteoroid orbital class of Borovička et al. (2005), and for a near-Earth orbit,
Atira, Aten, Apollo or Amor.

## Table of Contents

- [Quick start](#quick-start)
- [The Tisserand parameter](#the-tisserand-parameter)
- [Orbital classes](#orbital-classes)
- [One-parameter criteria](#one-parameter-criteria)
- [Two-parameter cometary tests](#two-parameter-cometary-tests)
- [The Tancredi scheme](#the-tancredi-scheme)
- [Supporting machinery](#supporting-machinery)
- [Cost](#cost)
- [Verification status](#verification-status)

---

## Quick start

```python
import numpy as np
from wmpl.Utils.OrbitClassification import calcTisserand, classifyTancrediComet, isCometaryQi

# 2P/Encke: a, e, i with the inclination in radians
a, e, i = 2.2155, 0.8482, np.radians(11.78)

print("T_J   =", calcTisserand(a, e, i))          # 3.0253
print("class =", classifyTancrediComet(a, e, i))  # 'jupiter family'
print("cometary by Q-i?", isCometaryQi(a, e, i))  # False
```

All angles are in radians. Distances are in AU unless a function says otherwise.

Encke is a good first example because it lands between the two schemes. Its Tisserand parameter is
3.0253, which is **above** the classical cut of 3 and would make it asteroidal, but **below** the
Tancredi cut of 3.05, which calls it a Jupiter-family orbit. It is a real comet, so the finer cut is
the one that gets it right — and the Q-i test disagrees with both, because Encke's aphelion falls
short of 4.6 AU. Do not expect these criteria to agree.

---

## The Tisserand parameter

`calcTisserand(a, e, i, a_planet=A_JUPITER)` is the standard discriminant. It is very nearly
conserved through a close encounter with the planet, so it survives exactly the events that scramble
the orbital elements.

| T_J | Conventional reading |
| :--- | :--- |
| > 3 | Asteroidal, decoupled from Jupiter |
| 2 to 3 | Jupiter-family-comet-like |
| < 2 | Halley type or long period |

**It returns `nan` for an unbound orbit** (`e >= 1`), where the parameter is not defined. Every
comparison against `nan` is False, so test for it rather than relying on a chain of comparisons.

**There is a second implementation in the repository.**
`wmpl.Rebound.REBOUND.tisserandParameterJupiter` computes the same quantity for scalars, returns
`None` instead of `nan` for an unbound orbit, and is what the REBOUND report prints; the report's
orbital class comes from `classifyBorovicka`, below. Use whichever suits, but do not mix their
outputs: they use slightly different values for Jupiter's semi-major axis (5.204267 AU against the
5.20336 AU of Tancredi's tables), which shifts T_J by about 1e-4 — far below any class boundary, but
enough to make two numbers disagree in the fourth decimal.

---

## Orbital classes

Class labels and flags that need only the **perihelion distance, eccentricity and inclination**,
which is what a trajectory solution gives directly. They take `q` rather than `a` because the
perihelion distance exists for every orbit, a parabolic one included, where the semi-major axis does
not. All accept scalars or numpy arrays; a scalar gives a plain `str`.

```python
import numpy as np
from wmpl.Utils.OrbitClassification import (classifyNEO, isArjuna, classifyBorovicka,
    isEarthCrossing, isProgrograde)

q, e, i = 0.97, 0.03, np.radians(1.5)

classifyNEO(q, e)            # 'Apollo'
isArjuna(q, e, i)            # True
classifyBorovicka(q, e, i)   # 'asteroidal-chondritic'
isEarthCrossing(q, e)        # True
```

`python -m wmpl.Utils.OrbitClassification` prints all of them for an orbit, with `--quiet` giving
`KEY VALUE` lines.

### Near-Earth classes: `classifyNEO(q, e)`

| Class | Criterion |
| :--- | :--- |
| Atira | a < 1 AU and Q < 0.983 AU: entirely inside the Earth's orbit |
| Aten | a < 1 AU and Q ≥ 0.983 AU |
| Apollo | a ≥ 1 AU and q < 1.017 AU |
| Amor | a ≥ 1 AU and 1.017 ≤ q < 1.3 AU |
| non-NEO | a ≥ 1 AU and q ≥ 1.3 AU |

0.983 and 1.017 AU are the Earth's perihelion and aphelion distances. An orbit at exactly a = 1 AU
goes with the larger orbits so that no semi-major axis falls between the classes. An unbound orbit
(`e >= 1`) returns `'unbound'`: these are classes of bound asteroid orbits, and a hyperbolic
meteoroid has none.

### Arjuna: `isArjuna(q, e, i)`

Arjuna is **not** a fifth class beside the four above, because its members are also Atens, Apollos
or Amors. It is a subset picked out by Earth-like elements, so it is a flag to read alongside
`classifyNEO`. An orbit is in the Arjuna domain if

> 0.985 < a < 1.013 AU, e < 0.1, i < 8.56°

with strict comparisons, as in de la Fuente Marcos & de la Fuente Marcos (2015, Astron. Nachr. 336,
5, section 2.1). The paper gives these as *approximate*, derived from the short-term dynamics of four
objects, and the real criterion is episodes of 1:1 resonance with the Earth. Falling inside the box
makes an orbit a candidate, not a confirmed member.

### Meteoroid orbital classes: `classifyBorovicka(q, e, i, ecliptic_shower=False)`

The five orbital classes of Borovička et al. (2005), the scheme commonly used in meteor studies
(Borovička et al. 2005; Kikwaya et al. 2011; Vojáček et al. 2019), with the limits as Vojáček et al.
(2019, section 3.3) give them:

| Class | Criterion |
| :--- | :--- |
| `'sun-approaching'` | q < 0.2 AU |
| `'ecliptic-shower'` | member of an ecliptic shower, such as the Taurids |
| `'halley-type'` | T_J < 2, or 2 < T_J < 3 and i > 45° |
| `'jupiter-family'` | 2 < T_J < 3, i < 45° and Q > 4.5 AU |
| `'asteroidal-chondritic'` | T_J > 3, or Q < 4.5 AU |

The Sun-approaching limit is where the meteoroids lose their sodium: almost all of those inside
q = 0.2 AU are Na-free or Na-poor (Vojáček et al. 2019, section 3.3.2).

**The definitions overlap, so the classes are tested in the order of the table and the first that
holds is taken.**

- An orbit with q < 0.2 AU is Sun-approaching whatever its other elements, as fig. 4 of Vojáček et
  al. (2019) draws it. 3200 Phaethon (T_J = 4.51) and 96P/Machholz (T_J = 1.95) are both
  Sun-approaching.
- An orbit with T_J < 3 and i > 45° that stays inside Q = 4.5 AU fits both the Halley-type and the
  asteroidal-chondritic definitions. The published list does not say which wins; the order makes it
  Halley type.

**The ecliptic-shower class is not a function of the elements.** Its members, the Taurids and other
showers with orbits near the boundary between the asteroids and the Jupiter-family comets, are
defined by membership of the shower, which has to come from a shower association: pass
`ecliptic_shower=True`, or a boolean array. Without it a Taurid orbit (q = 0.34 AU, e = 0.85,
i = 5.4°, so T_J = 2.99 and Q = 4.19 AU) is asteroidal-chondritic. The command line has no shower
association and never gives this class.

The published limits are strict inequalities, which leaves the limits themselves unassigned. Here
T_J = 2 and i = 45° do not make an orbit Halley type, and T_J = 3 and Q = 4.5 AU do not make it
asteroidal-chondritic. T_J is computed from `q` and `e` so that it stays finite for an unbound orbit,
which has no aphelion and so is never asteroidal-chondritic by Q.

**The asteroidal limit of T_J = 3 differs from `classifyTancrediComet`**, which puts it at 3.05, so
an orbit with T_J between 3 and 3.05 — 2P/Encke at 3.0253 — is `'asteroidal-chondritic'` here and
`'jupiter family'` there. Encke's aphelion of 4.09 AU also puts it inside the asteroidal limit here.

References:

- Borovička, J., Koten, P., Spurný, P., Boček, J., Štork, R. (2005). A survey of meteor spectra and
  orbits: evidence for three populations of Na-free meteoroids. *Icarus* 174, 15–30.
- Kikwaya, J.-B., Campbell-Brown, M., Brown, P. G. (2011). Bulk density of small meteoroids.
  *A&A* 530, A113.
- Vojáček, V., Borovička, J., Koten, P., Spurný, P., Štork, R. (2019). Properties of small
  meteoroids studied by meteor video observations. *A&A* 621, A68.
- Subasinghe, D., Campbell-Brown, M. (2018). Luminous efficiency estimates of meteors II.
  Application to Canadian Automated Meteor Observatory meteor events. *AJ*,
  doi:10.3847/1538-3881/aaa3e0. Table 3 gives the same limits.

### Flags

| Function | True when |
| :--- | :--- |
| `isSunApproaching(q, q_limit=0.2)` | q < 0.2 AU, the Sun-approaching class of Borovička et al. (2005) |
| `isProgrograde(i)` | i < 90° |
| `isRetrograde(i)` | i > 90° |
| `isEarthCrossing(q, e)` | q ≤ 1 AU ≤ Q |

An orbit at exactly 90° is neither prograde nor retrograde.

`isEarthCrossing` tests the **heliocentric distance range only**: the orbit passes through the
distances at which the Earth moves, with the Earth's orbit taken as a circle of 1 AU. It does not
say the two orbits intersect in space, which depends on their orientation — use `calcMOID` for that.
An unbound orbit has no aphelion and crosses if its perihelion is inside 1 AU.

---

## One-parameter criteria

Each returns a number; the cometary side of the cut is noted.

| Function | Returns | Cometary when |
| :--- | :--- | :--- |
| `calcKresakK(a, e)` | K | K > 0 |
| `calcKresakP(a, e)` | P, in years | P > 2.5 |
| `calcAphelionDistance(a, e)` | Q, in AU | Q > 4.6 |
| `calcOrbitalEnergy(a)` | energy, AU²/day² | > −5.28e−5 |

These are the quantities the two-parameter tests below are built on. On their own they misclassify
freely, which is why the paper that collected them pairs each with the inclination.

---

## Two-parameter cometary tests

Each returns a boolean and takes `(a, e, i)`, so the four can be called interchangeably. An orbit
counts as cometary if the one-parameter criterion says so **or** if the inclination exceeds 75°.

| Function | Pairs the inclination with |
| :--- | :--- |
| `isCometaryQi` | Aphelion distance |
| `isCometaryEi` | Orbital energy |
| `isCometaryKi` | Kresák K |
| `isCometaryPi` | Kresák P |

**Prefer `isCometaryQi` and `isCometaryEi`.** Of the five criteria the source paper examined, those
two came out most reliable.

`isCometaryEi` ignores `e` entirely — the energy depends only on `a` — and keeps it in the signature
purely so the four are interchangeable.

All four are array-safe.

---

## The Tancredi scheme

A finer classification that asks not just whether an orbit *could* reach Jupiter but whether it
actually does. Applying a Tisserand cut alone to the asteroid population returns thousands of
candidates, nearly all on stable orbits; this scheme adds two conditions that filter them.

### `classifyTancrediComet(a, e, i)`

Returns `'halley'`, `'jupiter family'`, `'asteroidal orbit'`, `'centaur'` or `'unclassified'`.

The splits are at T = 2 and **T = 3.05** rather than 3 — the upper limit is raised because Jupiter's
orbit is not circular and because encounters out to a few Hill radii still perturb an orbit, so
objects a little above 3 can still be Jupiter-dominated. The intervals are half open, so an orbit
landing exactly on a limit is classified rather than falling through.

An unbound orbit returns `'unclassified'`, deliberately rather than by accident.

### `classifyTancrediAsteroid(a, e, i, moid_jupiter_hill, moid_giants_min_hill)`

Returns `'aco jupiter family'`, `'aco halley'`, `'centaur'`, `'transneptunian'` or `'asteroid'`.
"ACO" is an asteroid in a cometary orbit.

The two extra arguments are minimum orbital intersection distances, **in units of the relevant
planet's Hill radius, not AU**. They are arguments rather than computed internally so that a study
can supply values from whatever ephemeris it uses. `calcGiantPlanetMOIDs` computes them from J2000
mean elements and returns a dictionary keyed by planet, so a caller passes its `'jupiter'` entry and
the smallest of its values:

```python
from wmpl.Utils.OrbitClassification import calcGiantPlanetMOIDs, classifyTancrediAsteroid

moids = calcGiantPlanetMOIDs(a, e, i, node, peri)

label = classifyTancrediAsteroid(a, e, i, moids['jupiter'], min(moids.values()))
```

### `isTancrediResonanceProtected(a, e, moid_jupiter_hill)`

An orbit can have a tiny minimum distance to Jupiter's orbit and still never approach the planet,
because a mean-motion resonance keeps the two apart in phase. The Hildas are the standard example.
This is what stops the scheme from calling every resonant asteroid a comet.

The libration width is computed from the eccentricity for the five inner resonances and taken as a
fixed value for the three outer ones, where the expansion of the disturbing function is no longer
reliable.

---

## Supporting machinery

These exist to serve the Tancredi scheme but are usable on their own.

| Function | What it does |
| :--- | :--- |
| `calcMOID(a1, e1, i1, O1, w1, a2, e2, i2, O2, w2)` | Minimum orbital intersection distance between two orbits, in AU. It is `wmpl.Utils.MOID.calcMOID`, vectorised: the elements can be arrays |
| `calcGiantPlanetMOIDs(a, e, i, node, peri)` | The MOID with each giant planet, in that planet's Hill radii |
| `calcHillRadius(a_planet, mass_ratio)` | Hill radius of a planet, circular-orbit form |
| `calcResonanceSemiMajorAxis(p, p_plus_q)` | Centre of a mean-motion resonance |
| `calcResonanceWidth(p, p_plus_q, e)` | Half width of a resonance in semi-major axis |
| `calcMaxPerihelionForTisserand(T)` | Largest perihelion available at a given T, in units of the planet's semi-major axis |
| `calcMinMOIDForTisserand(T)` | Edge of the forbidden region in the T-against-MOID plane |
| `calcLaplaceCoefficient`, `calcLaplaceDerivative`, `calcDisturbingFunctionTerm` | The expansion of the disturbing function that the resonance width is built on |

Three things to watch:

- **`calcMOID` takes the semi-major axis first**, where every criterion in
  [`Dcriteria`](OrbitSimilarity.md) takes the perihelion distance. The signatures are otherwise the
  same shape, so a `q` passed by mistake is silently accepted.
- **`calcMaxPerihelionForTisserand` and `calcMinMOIDForTisserand` work in units of the planet's
  semi-major axis**, not AU, and assume a coplanar orbit.
- **The `HILL_RADIUS_*` constants are the paper's quoted values**, not output of `calcHillRadius`,
  so the classification reproduces the paper exactly. The two agree to about 2e-3 AU.

---

## Cost

Most of this module is arithmetic, but the MOID is not.

`calcMOID` is `wmpl.Utils.MOID.calcMOID`: a scan of one orbit, the closest point of the other for each
sample, and a two-dimensional Newton refinement of the deepest minima, vectorised over any number of
orbit pairs at ~30-40 µs per pair (see the module for the method and its validation).
`calcGiantPlanetMOIDs` calls it once per giant planet.

It replaced the implementation of Tancredi (2014, appendix A): Powell's method restarted from four
points, ~20 ms per pair. That one is kept as `wmpl.Utils.MOID.calcMOIDTancredi` for comparison. Both
agree for an orbit against a planet, but for two eccentric orbits Powell's method can stop at a local
minimum (in 1 of 80 random pairs, 0.0260 AU instead of 0.0206 AU).

The Laplace coefficients are evaluated by quadrature rather than from a truncated series, so they
stay exact as the semi-major axis ratio approaches 1. They are cached, since only a handful of
resonances are ever needed.

---

## Verification status

| Item | Status |
| :--- | :--- |
| Tisserand parameter, Kresák K and P, aphelion, orbital energy | **Verified** — standard definitions, and the Encke and Phaethon values reproduce published figures |
| The two-parameter cometary tests | **Verified** against the limits as published |
| Atira, Aten, Apollo, Amor limits | **Verified** against the standard definitions (Earth's perihelion 0.983 and aphelion 1.017 AU, q < 1.3 AU for near-Earth) and one known object of each class |
| Arjuna limits | **Verified** against the text of de la Fuente Marcos & de la Fuente Marcos (2015), section 2.1. The paper calls them approximate |
| Borovička et al. (2005) classes | **Verified indirectly** — the 2005 paper (Icarus 174, 15) is paywalled and Kikwaya et al. (2011) could not be read, but Vojáček et al. (2019, section 3.3), co-authored by Borovička, give the five classes and their limits, and table 3 of Subasinghe & Campbell-Brown (2018) gives the same. Fig. 4 of Vojáček et al. (2019) puts q < 0.2 AU first; the rest of the order of the overlapping classes follows the published list. Checked against one known body of each class and the two Halley-type iron meteoroids of Vojáček et al. (2019, table 1) |
| Flags | **Verified** — direct definitions |
| Tancredi class structure and the 3.05 limit | **Verified indirectly** — the primary paper (Icarus 234, 66) is paywalled and the author's own site no longer resolves, but a 2025 paper co-authored by Tancredi restates the class definitions and the 3.05 cut verbatim |
| Tancredi Hill radii, table 1 resonances and widths, appendix A and B formulae | **Unverified** — the primary source could not be read. The implementation reproduces the tabulated resonance centres and widths to the printed precision, which is strong but indirect evidence |

Related: [orbit dissimilarity criteria](OrbitSimilarity.md) and
[choosing a threshold](DThresholds.md).
