# Choosing a D-criterion threshold

**`wmpl.Utils.Dthresholds`** supplies the other half of a stream search. A
[dissimilarity criterion](OrbitSimilarity.md) gives a number; the threshold decides what counts as
associated. Get the threshold wrong and the criterion cannot save you.

## Table of Contents

- [Read this first](#read-this-first)
- [The four methods](#the-four-methods)
- [Random pairing](#random-pairing)
- [The break point](#the-break-point)
- [Reliability against shuffled samples](#reliability-against-shuffled-samples)
- [The historical D_r threshold](#the-historical-d_r-threshold)
- [Worked examples](#worked-examples)
- [Verification status](#verification-status)

---

## Read this first

The traditional threshold, `0.20*(360/N)**0.25`, is **far too permissive for pair searches**.
Jopek & Bronikowska measured the probability of finding a coincidental similar pair at that
threshold as **1.0** for bolide samples of 200 to 800 orbits. In other words, two "similar" orbits
are certain to be found whether or not any are genuinely related. It runs 7 to 9 times looser than
the threshold that delivers a 1 % chance of coincidence.

It is still provided as `thresholdDr`, because a great deal of published work used it and you need
it to reproduce that work. Do not use it to claim a new association.

Use `thresholdRandomPairing` instead.

---

## The four methods

| Function | Needs | Gives |
| :--- | :--- | :--- |
| `thresholdRandomPairing(n_orbits, d_criterion, population)` | Sample size, criterion name, population | Threshold at a 1 % chance of a coincidental pair |
| `thresholdBreakPoint(d_values, ...)` | The D values against a mean orbit | Where the stream ends and the background begins |
| `thresholdReliability(orbit_params, d_func, ...)` | The sample and the criterion | Threshold at a stated reliability, from shuffled samples |
| `thresholdDr(n_orbits)` | Sample size | The historical Southworth & Hawkins value |

The first three are independent of which criterion you use — they take it as a callable, or are
selected by name. Only `thresholdRandomPairing` is tied to specific criteria, because its numbers
were measured per criterion and per population.

---

## Random pairing

`thresholdRandomPairing(n_orbits, d_criterion='D_SH', population='bolides')`

The recommended default. It returns the threshold at which the probability of a coincidental pair is
0.01, from `D_c = A*N**(-b)` with coefficients measured on synthetic samples that reproduce the
orbital distribution of the real one.

Both dependencies are substantial and neither is optional:

- At a fixed sample size the **D_D** thresholds are about **2.4 times smaller** than the D_SH ones.
- The **video** thresholds are about **1.15 times larger** than the NEA ones.

Available combinations, in `RANDOM_PAIRING_COEFFS`:

| Criterion | Populations |
| :--- | :--- |
| `'D_SH'`, `'D_H'`, `'D_D'` | `'bolides'`, `'NEAs'`, `'radar'`, `'video'` |

An unpublished combination raises `ValueError` rather than falling back to a default. That is
deliberate: a made-up threshold is indistinguishable from a measured one once it is in a table.

**Validity range.** The bolide coefficients were fitted over 200 to 1000 orbits, the rest over 1000
to 16000, which the authors expect to extend to about 50000. **Nothing is clamped**, so a call
outside those ranges returns an extrapolation without complaint.

**The synthetic samples matter.** The coefficients come from samples that reproduce the observed
orbital distributions, including the Earth-crossing condition. Drawing the elements uniformly
instead inflates the threshold by roughly a factor of two, so these numbers are specific to the
method that produced them.

---

## The break point

`thresholdBreakPoint(d_values, n_bins=100, d_max=None)`

Instead of a fixed threshold, look at how many orbits lie within a distance D of a stream's mean
orbit. The count rises steeply while the stream is being counted, then flattens into the much
shallower rise of the sporadic background. The D at which the slope drops separates the two.

The break is taken as the bin of strongest negative curvature of the cumulative distribution.

**The limitation is important:** the break only exists when the stream is a large enough fraction of
the sample to produce a visible change of slope. A sample that is almost all sporadic has no break —
and this function will still return its best candidate. Plot the distribution before believing the
number.

---

## Reliability against shuffled samples

`thresholdReliability(orbit_params, d_func, reliability=0.99, n_trials=100, random_state=None)`

Derives the threshold from synthetic samples that share the marginal distribution of each variable
with the real one but carry none of its correlations — which is what shuffling each variable
independently produces. The threshold is the largest value at which, in the requested fraction of
those samples, no pair is associated. A group found below it is therefore not reproducible by
chance at that reliability.

Two things to know:

- **Only pairs are considered**, so this is the threshold for a group of two. A larger minimum group
  size needs a clustering algorithm, which is a separate concern.
- **The cost grows as `n_trials` times the square of the sample size.** A few tens of orbits over a
  hundred trials is quick. A few thousand orbits is not, and wants a chunked or pre-filtered search.

Pass `random_state` if you intend to quote the result.

---

## The historical D_r threshold

`thresholdDr(n_orbits)` returns `0.20*(360/N)**0.25`, equivalently `0.8712*N**(-0.25)`. The
reference value of 0.20 applies to a sample of 360 orbits, and the formula was calibrated on 359
Super-Schmidt photographic meteors.

Keep it for reproducing published work. See [Read this first](#read-this-first) for why not to use
it for anything else.

---

## Worked examples

**The recommended route.**

```python
from wmpl.Utils.Dthresholds import thresholdRandomPairing, thresholdDr

n_orbits = 5000

recommended = thresholdRandomPairing(n_orbits, d_criterion='D_SH', population='video')
historical = thresholdDr(n_orbits)

print("random pairing (1% coincidence): {:.4f}".format(recommended))
print("Southworth & Hawkins:            {:.4f}".format(historical))
print("the old threshold is {:.1f}x looser".format(historical/recommended))
```

**Why the criterion has to match the threshold.**

```python
from wmpl.Utils.Dthresholds import thresholdRandomPairing

for criterion in ('D_SH', 'D_H', 'D_D'):
    print("{:5s} {:.5f}".format(criterion,
        thresholdRandomPairing(10000, d_criterion=criterion, population='NEAs')))
```

**Deriving a threshold for a criterion with no published one.**

```python
import numpy as np
from wmpl.Utils.Dcriteria import calcRho2
from wmpl.Utils.Dthresholds import thresholdReliability

# One row per orbit, in the order calcRho2 takes them: q, e, i, node, peri
orbit_params = np.array([
    [0.62, 0.79, 1.15, 1.72, 0.23],
    [0.61, 0.80, 1.14, 1.71, 0.27],
    # ... the rest of the sample
    ])

threshold = thresholdReliability(orbit_params, calcRho2, reliability=0.99, n_trials=100,
    random_state=42)

print("rho_2 below {:.4f} is not reproducible by chance at the 99% level".format(threshold))
```

---

## Verification status

| Item | Status |
| :--- | :--- |
| `thresholdRandomPairing` coefficients, all 12 pairs | **Verified** against Jopek & Bronikowska (2017), eqs 14-16 and table 9 (eqs 17-25), read from arXiv:1609.03968 |
| Validity ranges and the 1 % probability level | **Verified** in the same source |
| `thresholdDr` and the warning about it | **Verified** — the paper prints it as its eq. 1 and states it corresponds to a coincidence probability of 1 for bolide samples, and that it "should not be used" |
| `thresholdBreakPoint` | **Unverified** — the source is paywalled. The implementation is a reasonable reading of the method as described elsewhere |
| `thresholdReliability` | **Unverified** — same |

One caveat carried from the source: the coefficients were obtained with a particular synthetic-orbit
generation method (method E for the meteoroid populations, method D for the NEAs). The paper shows a
uniform-element method gives thresholds different by about a factor of two, so these numbers are
method-specific as well as criterion- and population-specific.

Related: [orbit dissimilarity criteria](OrbitSimilarity.md) and
[orbit classification](OrbitClassification.md).
