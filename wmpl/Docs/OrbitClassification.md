# Classifying a single orbit

**`wmpl.Utils.OrbitClassification`** answers a different question from the
[dissimilarity criteria](OrbitSimilarity.md): not "are these two orbits related?" but "what kind of
orbit is this one?" — asteroidal or cometary, Jupiter family or Halley type, Centaur or
transneptunian.

## Table of Contents

- [Quick start](#quick-start)
- [The Tisserand parameter](#the-tisserand-parameter)
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
`None` instead of `nan` for an unbound orbit, and is what the REBOUND report prints. Its companion
`tisserandClass` gives three classes split at T = 3. Use whichever suits, but do not mix their
outputs: they use slightly different values for Jupiter's semi-major axis (5.204267 AU against the
5.20336 AU of Tancredi's tables), which shifts T_J by about 1e-4 — far below any class boundary, but
enough to make two numbers disagree in the fourth decimal.

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
| `calcMOID(a1, e1, i1, O1, w1, a2, e2, i2, O2, w2)` | Minimum orbital intersection distance between two orbits, in AU |
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

`calcMOID` minimises the separation over two eccentric anomalies with Powell's method, restarted
from four points because the surface has several local minima. `calcGiantPlanetMOIDs` calls it four
times, once per giant planet. Classifying a large asteroid catalogue this way is a real computation,
not a lookup — budget for it, or supply MOIDs computed in bulk elsewhere.

The Laplace coefficients are evaluated by quadrature rather than from a truncated series, so they
stay exact as the semi-major axis ratio approaches 1. They are cached, since only a handful of
resonances are ever needed.

---

## Verification status

| Item | Status |
| :--- | :--- |
| Tisserand parameter, Kresák K and P, aphelion, orbital energy | **Verified** — standard definitions, and the Encke and Phaethon values reproduce published figures |
| The two-parameter cometary tests | **Verified** against the limits as published |
| Tancredi class structure and the 3.05 limit | **Verified indirectly** — the primary paper (Icarus 234, 66) is paywalled and the author's own site no longer resolves, but a 2025 paper co-authored by Tancredi restates the class definitions and the 3.05 cut verbatim |
| Tancredi Hill radii, table 1 resonances and widths, appendix A and B formulae | **Unverified** — the primary source could not be read. The implementation reproduces the tabulated resonance centres and widths to the printed precision, which is strong but indirect evidence |

Related: [orbit dissimilarity criteria](OrbitSimilarity.md) and
[choosing a threshold](DThresholds.md).
