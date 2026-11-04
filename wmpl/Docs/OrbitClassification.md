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
- [Jupiter-family dynamics](#jupiter-family-dynamics)
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

## Jupiter-family dynamics

The criteria above read one orbit. Shober et al. (2024, A&A 687, A181) asked instead how the orbit
*behaves*: integrate clones of it 10 000 years back and count how many raise their perihelion above
2.5 AU, as Jupiter-family comets do and asteroids on comet-like orbits rarely do. `--dynamics` runs
this through [`wmpl.Rebound.REBOUND`](REBOUND.md), on a trajectory pickle or, further below, on published
elements:

```
python -m wmpl.Utils.OrbitClassification traj.pickle --dynamics              # adapted mode
python -m wmpl.Utils.OrbitClassification traj.pickle --dynamics shober2024   # the paper's setup
```

Give `--dynamics` after the pickle: it takes an optional mode, and would read the path as one.

| | `adapted` (default) | `shober2024` |
| :--- | :--- | :--- |
| Clones | from the state-vector covariance of the Monte Carlo solution | a, e, i, node, peri drawn independently from their 1-sigma; without Monte Carlo, the values the paper assumed for MORP (0.01 AU, 0.001, 0.1°, 0.01°, 0.01°) |
| How many | batches of 20 until the class is settled, at most 200 | 1000, all of them |
| Start | the meteoroid's state in the atmosphere | each clone's orbit without the Earth's attraction, 3 Earth Hill radii out, at the geocentric speed it had there |
| Forces | Sun, planets, Moon, the Sun's post-Newtonian correction (`--gr`), Sun J2, Earth J2/J4, radiation optional (`--beta`, `--radius`) | Sun, planets, Moon, Newtonian (Earth J2/J4 stay) |
| Integrator | IAS15 (`--integrator`) | IAS15 |
| Outputs | every 50 years | every 10 years |

Both report the assumptions first, then f_q and f_a (the fractions of time beyond q = 2.5 AU and
a = 7.37 AU, Fernández et al. 2014), the number of clones that reached q > 2.5 AU, split by whether a
Jupiter encounter inside 1 Hill radius came first, and the fraction P with its Wilson interval.
Two verdicts follow: the paper's rule (P > 0.5) and a class from the interval, which stays
`undetermined` while the interval straddles 0.5. `--dynamics_save` writes all of it to JSON.

Three choices are not in the paper:

- **Batches** (adapted mode). Each batch is a look at P that could stop the run on a fluke, so
  each uses the interval at 1 - 0.05/n_looks (Bonferroni over the possible looks).
- **Clones that leave the 1000 AU volume** are told apart by their eccentricity after the epoch:
  bound at their last output, they keep their orbit; bound and then hyperbolic, they were thrown out
  and count as beyond every limit; hyperbolic throughout, they are left out and reported, also if the
  run ends before they reach 1000 AU.
- **Clones that fail to integrate** are in none of the counts and are reported, with `n_failed` in
  the result. If a whole batch fails, or every clone of `shober2024`, the run stops with an error: that
  says nothing about the orbit.
- **The start of the shober2024 clones.** The elements are those of the orbit without the
  Earth's attraction, placed where the meteoroid met the Earth. A clone started there would feel the
  Earth a second time; one started at a distance d with the speed at infinity leaves it with v²
  short by 2GM/d. In a Sun-and-Earth test on the Orionid example, that left a short by 0.014 AU from 3 Hill
  radii; with the speed it had at d, by 2.7e-4 AU. The paper starts its clones at the beginning of
  the fireball observation, and does not say how it treats the Earth's attraction there.

**Cost.** With the setup of `shober2024`, 360 clones of 21P took a mean of 102 s each over 10 000
years, on 10 cores of a machine that had other work. That makes about 30 CPU-hours for the 1000
clones on such an orbit, and minutes for an orbit whose class settles in one batch.

**From published elements.** `shober2024` also runs without a pickle, from the elements and the time of the
fireball, here those of Table 2 of Madiedo et al. (2014):

```
python -m wmpl.Utils.OrbitClassification -a 4.3 -e 0.79 -i 7.1 -p 40.95 -n 203.945 --dynamics shober2024 \
    --date "2013-04-13 21:43:55.4" --sigma_a 0.7 --sigma_e 0.04 --sigma_i 0.1 --sigma_node 1e-4 --sigma_peri 0.03
```

A list of elements does not say where on the orbit the meteoroid was, but a fireball is at the Earth, so the
clones start at the node the Earth was at that moment (`wmpl.Utils.OrbitFromElements`). The report gives that
node, its distance from the Earth and the geocentric speed there, to compare with the paper. A node more than
0.05 AU from the Earth is refused, as the elements or the date are then wrong: published fireballs give 0.0004 to
0.012 AU. `--max_node_miss` raises the limit. Without the five `--sigma` options, the clones take the
uncertainties the paper assumed for MORP. The adapted mode needs a pickle, since it takes the clones from the
full covariance.

a and e are drawn independently, as the paper does, which spreads the perihelion of the clones far beyond its
measured error when a is poorly known, since a and e are strongly correlated. `--sigma_q`, in place of
`--sigma_e`, draws q and takes e = 1 - q/a. This departs from the setup of the paper, and the report says so.

For the fireball of Trigo-Rodriguez et al. (2009), whose q is known to 0.0006 AU, a and e drawn independently spread
the perihelion of the clones from 0.67 to 1.66 AU, and give P = 10.0% [6.6, 14.9]. Respecting q, two runs of 200 clones
gave 4.0% [2.0, 7.7] (q held at its value) and 7.5% [4.6, 12.0] (`--sigma_q`), 5.8% [3.9, 8.5] together. The difference
from the independent draws is about 1.8 sigma: it points to an inflated fraction, but this does not settle it.

**Figure.** `--dynamics_plot FILE` saves a figure of the run, with the format given by the extension. Over the
time before the fireball, it draws the perihelion distance and the semi-major axis of the clones, as the band and
median of all of them, with up to 40 of those that reach q > 2.5 AU one by one, and the fraction that have reached
it so far with its interval. That last panel shows what P alone does not: whether it has settled by the end of the
integration. With `--dynamics_checkpoint`, a run already finished draws the figure again without integrating.

**Checkpoint.** A run this long can be cut. `--dynamics_checkpoint FILE` saves each clone of `shober2024`,
or each batch of the adapted mode, as soon as it ends:

```
python -m wmpl.Utils.OrbitClassification traj.pickle --dynamics shober2024 --dynamics_checkpoint run.pkl
```

Run the same command again to go on from there. The clones are the same, with the seed kept in the file if
none is given, so the result does not depend on the interruption. A batch or a clone cut halfway starts
again, and so does a clone that failed. A file from a run with other parameters, or one that is not a
checkpoint, is refused and left as it is. The verdict settings (`confidence`, `encounter_hill`) are not
part of the comparison, since they change nothing in the integration. The file takes about 60 kB per clone
with the outputs of `shober2024`, and stays in place at the end: delete it once you have the result. It is
a pickle, so read only your own.

After Ctrl-C the run stops once the clones already handed to the processes have ended, which is up to a few
clones' time. What they integrated meanwhile is not saved. If the process is killed outright, its workers
end the clone they are on before exiting.

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
| Jupiter-family dynamics | **Not yet compared with the paper.** The metrics are tested on synthetic ensembles and both modes ran end to end with stand-in planets; the paper's Tables 1 and 2 need DE430 and its data |

Related: [orbit dissimilarity criteria](OrbitSimilarity.md) and
[choosing a threshold](DThresholds.md).
