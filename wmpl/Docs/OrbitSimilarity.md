# Orbit dissimilarity criteria

**`wmpl.Utils.Dcriteria`** measures how similar two orbits are with a single number, the *D* value.
Below a threshold the two are called associated. Every criterion here answers that question
differently, and the choice matters more than it looks: they disagree about which pairs are close,
they are not on a common scale, and most of them are not distances in the mathematical sense.

## Table of Contents

- [Choosing a criterion](#choosing-a-criterion)
- [The full list](#the-full-list)
- [Criteria on orbital elements](#criteria-on-orbital-elements)
- [Criteria on geocentric quantities](#criteria-on-geocentric-quantities)
- [Criteria for the Taurid Complex](#criteria-for-the-taurid-complex)
- [Pitfalls](#pitfalls)
- [Worked examples](#worked-examples)
- [Verification status](#verification-status)

---

## Choosing a criterion

Start here if you only want one answer:

| If you want | Use | Why |
| :--- | :--- | :--- |
| What everyone else uses, for comparability | `calcDSH` | The 1963 original, in about half of all published shower searches. Its faults are well known and well documented. |
| A criterion that is actually a metric | `calcRho2` | Satisfies the triangle inequality, defined for circular orbits, and the only family here with a proper mathematical footing. |
| To compare streams whose nodes and apsides have circulated | `calcRho5` | The minimum of rho_2 over both nodes and both arguments of perihelion, so precession does not separate related orbits. |
| To avoid the semi-major axis, which carries the velocity error | `calcDN`, `calcDR`, `calcDX` | Built from the observed radiant and speed rather than from derived elements. |
| To weight each element by how much a real stream disperses in it | `calcDVJopek` | Weights are measured dispersions, not guesses. |

**If you are searching for a new shower, the threshold matters as much as the criterion.** See
[the threshold manual](DThresholds.md); the traditional Southworth & Hawkins threshold is now known
to be far too permissive.

---

## The full list

Angles are in radians throughout. `q` is the perihelion distance in AU, `e` the eccentricity, `i`
the inclination, `O` the longitude of the ascending node, `w` the argument of perihelion.

| Function | Inputs | Metric? | Published threshold | Source |
| :--- | :--- | :--- | :--- | :--- |
| `calcDSH` | q, e, i, O, w | No | 0.20 at N = 360, scaled | Southworth & Hawkins (1963) |
| `calcDD` | q, e, i, O, w | No | none stated | Drummond (1981) |
| `calcDH` | q, e, i, O, w | No | none stated | Jopek (1993) |
| `calcRho1` | q, e, i, O, w, L | **Yes** | none published | Kholshevnikov et al. (2016) eq. 11 |
| `calcRho2` | q, e, i, O, w, L | **Yes** | none published | Kholshevnikov et al. (2016) eq. 15 |
| `calcRho5` | q, e, i (O, w ignored) | **Yes** | none published | Kholshevnikov et al. (2016) eq. 22 |
| `calcC` | q, e, i, O, w | - | none published | Neslusan (2002) |
| `calcDB` | e, i, O, w | No | < 1.0, or < 1.5 | Jenniskens (2008) eq. 8 |
| `calcDT` | q, e, i | No | < 0.3, or < 0.6 | Jenniskens (2008) eqs 13-14 |
| `calcDN` | ra, dec, sol, vg | No | 0.20 (reported, not derived) | Valsecchi et al. (1999) eq. 5 |
| `calcDR` | ra, dec, sol, vg | No | none; D_R <= D_N | Valsecchi et al. (1999) |
| `calcDX` | ra, dec, sol, vg | No | <= 0.15 to merge | Rudawska et al. (2015) eq. 2 |
| `calcDV` | Lh, Bh, sol, Vh | No | none | Unpublished (Vida) |
| `calcDVJopek` | q, e, i, O, w | No | `DV_THRESHOLDS`, 99 % level | Jopek et al. (2008) eq. 7 |
| `calcDACS` | **a**, e, i | No | 0.15 core, ~0.2 | Asher, Clube & Steel (1993) eq. 2 |
| `calcDSAC` | q, e, i | No | 0.15 | Steel, Asher & Clube (1991) eq. 1 |

Note `calcDACS` takes the **semi-major axis** where everything else takes the perihelion distance.

---

## Criteria on orbital elements

### D_SH, D_D, D_H - the classical family

`calcDSH` is the original. `calcDD` (Drummond) makes every term dimensionless and linear in [0, 1],
at the cost of accuracy when the perihelion distance is small. `calcDH` (Jopek) is a deliberate mix
of the two. All three combine an eccentricity term, a perihelion term, the mutual inclination and
an apsidal term built on Pi_21, the difference in longitude of perihelion measured from the mutual
node.

**None of the three is a metric.** They can violate the triangle inequality, so "A is close to B and
B is close to C" does not constrain how far A is from C. That is not a curiosity: cluster-finding
algorithms generally assume it.

### rho_1, rho_2, rho_5 - the Kholshevnikov metrics

These *are* metrics. They are built from the difference of the angular momentum vectors and of the
eccentricity vectors, so they satisfy the triangle inequality and stay finite for a circular orbit,
where `calcDD` raises `ZeroDivisionError`.

- **rho_1** mixes a length term and a dimensionless term, made commensurate by the scale length `L`.
- **rho_2** puts both terms in the same units, which is the cleaner construction.
- **rho_5** is rho_2 minimised over both nodes and both arguments of perihelion, so it measures how
  close two orbits could be brought by precession alone. Use it when the nodes and apsides have had
  time to circulate; it ignores `O` and `w`, which are kept in the signature only so it can be
  substituted for `calcDSH` without changing the call.

`L` defaults to 1 AU. **It sets the numerical scale**, so a threshold from one choice of `L` means
nothing under another, and no D_SH or D_D threshold carries over.

### C - Neslusan

The length of the difference of the two specific angular momentum vectors. It compares orbital
planes and sizes only and says nothing about the apsidal orientation.

**Scale caveat:** this module builds the vector as `sqrt(p)`, so C carries units of sqrt(AU).
`calcDVJopek` builds the same vector as `k*sqrt(p)` with the Gaussian constant included. The
convention the original uses could not be established from an accessible copy, so do not compare a C
value from here against one quoted in a paper without checking. It does not affect a search that
uses the break-point method rather than a fixed threshold.

### D_B and D_T - Jenniskens

Both are built on quantities that survive secular perturbation. `calcDB` uses three invariants:
`(1-e^2)cos^2(i)`, `e^2(0.4 - sin^2(i) sin^2(w))` and `w + O`, each divided by its measured
dispersion. `calcDT` is simply the difference of the two Tisserand parameters, written in `q` and
`e` rather than `a` because the observational errors in those are smaller - and because that form
stays finite at `e = 1`, where the `a` form gives `nan`.

They are meant to be used together: `D_T < 0.3` with `D_B < 1.0` for parent bodies and siblings,
`D_T < 0.6` with `D_B < 1.5` to also catch bodies related through an earlier fragmentation.

### D_V - Jopek, Rudawska & Bartczak

Compares the vectorial elements - angular momentum, eccentricity vector, energy - with weights that
are **measured dispersions within real streams** rather than chosen. `calcDVWeights` derives them so
that a pair differing by twice the dispersion in one element contributes exactly 1.

The weights depend on how old the stream is; `DV_DISPERSIONS` is keyed by age in years, plus
`'sporadic'` for the background. The default is 4000 yr, the value the paper used.

---

## Criteria on geocentric quantities

The semi-major axis carries the full weight of the velocity uncertainty, which is why these exist.

- **`calcDN`** compares the geocentric velocity vector in Opik-like variables and takes the smaller
  of two branches for the angular terms.
- **`calcDR`** is `calcDN` with the terms that are *not* invariant under the circulation of the
  argument of perihelion dropped. It is therefore always <= D_N, and is a necessary but not a
  sufficient condition for association. **Scalars only** - it shares `calcVgComponents` with
  `calcDN`, which is written with the `math` module.
- **`calcDX`** compares solar longitude, right ascension, declination and geocentric speed directly,
  with published weights (0.17, 1.20, 1.20, 0.20). The radiant terms are scaled by the velocity
  difference, so the criterion is **not invariant to the unit of the speed** - the paper tabulates
  km/s, which is what is assumed. It is also not quite symmetric: the right-ascension term uses the
  declination of the first orbit and the velocity term its speed.

---

## Criteria for the Taurid Complex

`calcDACS` and `calcDSAC` are the same expression with a different first term, and neither carries a
node or longitude term - deliberately, because the Taurid Complex has been dispersed in longitude of
perihelion, so a longitude term suited to a narrow stream would swamp the sum.

- **`calcDSAC`** uses the perihelion distance. Use it for meteoroids, whose `q` is better determined.
- **`calcDACS`** uses the semi-major axis divided by 3 AU. Use it for asteroids, whose `a` is well
  determined.

The reference orbit of the complex core is available as `TC_REFERENCE_A`, `TC_REFERENCE_E`,
`TC_REFERENCE_INCL` and `TC_REFERENCE_Q`.

**One caveat that will otherwise cost you an afternoon:** the paper does not feed observed elements
into the criterion. It first adjusts the inclination and eccentricity to their secular minima using
Brouwer (1947) theory. These functions apply no such adjustment, so passing observed elements will
not reproduce the paper's table 1.

---

## Pitfalls

**Thresholds do not transfer between criteria.** At a fixed sample size the D_D threshold is about
2.4 times smaller than the D_SH one. Using a D_SH threshold with D_D will associate almost nothing;
the reverse will associate almost everything.

**Thresholds do not transfer between datasets either.** The video thresholds are about 1.15 times
the NEA ones at the same sample size.

**Most of these are not metrics.** Only the rho family satisfies the triangle inequality. If your
clustering algorithm assumes a metric space, it is not getting one from `calcDSH`.

**Circular orbits break `calcDD`** with a `ZeroDivisionError`, because its eccentricity term divides
by `e1 + e2`. The rho family handles them.

**A self-comparison does not return exactly zero.** The square of the criterion cancels to about
1e-16 and the square root lifts that to about 1e-8. Compare against a tolerance, not against zero.

**Seven of these criteria have no published threshold at all.** No default is supplied, because one
invented here would be indistinguishable from a real one a year later.

---

## Worked examples

**Comparing two orbits with several criteria.**

```python
import numpy as np
from wmpl.Utils.Dcriteria import calcDSH, calcDD, calcDH, calcRho2, calcRho5

# q, e, i, node, argument of perihelion - angles in radians
orbit1 = (0.622, 0.795, np.radians(65.82), np.radians(98.56), np.radians(13.28))
orbit2 = (0.615, 0.798, np.radians(65.03), np.radians(97.78), np.radians(15.55))

for name, func in (("D_SH", calcDSH), ("D_D", calcDD), ("D_H", calcDH),
                   ("rho_2", calcRho2), ("rho_5", calcRho5)):
    print("{:6s} = {:.4f}".format(name, func(*(orbit1 + orbit2))))
```

**Using a criterion that tolerates circular orbits.**

```python
from wmpl.Utils.Dcriteria import calcDD, calcRho2

circular1 = (1.0, 0.0, 0.0, 0.0, 0.0)
circular2 = (1.1, 0.0, 0.1, 0.0, 0.0)

print(calcRho2(*(circular1 + circular2)))   # fine

try:
    calcDD(*(circular1 + circular2))
except ZeroDivisionError:
    print("D_D is undefined for two circular orbits")
```

**Weighting by how much a real stream disperses.**

```python
from wmpl.Utils.Dcriteria import calcDVJopek, calcDVWeights, DV_THRESHOLDS

# Weights for a 4000 year old stream, the paper's own choice
w_h, w_e, w_energy = calcDVWeights(epoch=4000)

d_value = calcDVJopek(*(orbit1 + orbit2), w_h=w_h, w_e=w_e, w_E=w_energy)

# Threshold at the 99% level for a smallest accepted group of 10
print(d_value, "against", DV_THRESHOLDS[10])
```

---

## Verification status

Each formula below was checked against its primary source where one could be reached. This is
recorded so that a later reader knows which are confirmed and which are taken on the implementer's
word.

| Criterion | Status | Source consulted |
| :--- | :--- | :--- |
| rho_1, rho_2, rho_5 | **Verified** | Kholshevnikov et al. (2016), MNRAS 462, 2275, eqs 11, 15, 22 |
| D_SH, D_D, D_H | **Verified** | Courtot, Shober & Vaubaillon (2026), arXiv:2507.19075, eqs 1-3 |
| D_R, D_N | **Verified** | same review, eq. 5 |
| D_B | **Verified** | same review, eq. 8 |
| D_T | **Verified** | same review, section 4 |
| D_V (Jopek) | **Verified**, including the 1.5x on h_Z and 2x on the energy term | same review, eq. 7 |
| D_X | **Verified**, and a transcription error in the right-ascension term was found and fixed | same review, eq. 6 |
| D_ACS, D_SAC | **Verified**, including the 3 AU scale, which is printed in eq. 2 | Asher, Clube & Steel (1993), MNRAS 264, 93, ADS scan |
| C | **Unverified** - the source is a 2000 workshop proceedings chapter behind a paywall. The scale convention is open; see the caveat above. | - |

Related: [orbit classification](OrbitClassification.md) for criteria that take a single orbit, and
[choosing a threshold](DThresholds.md).
