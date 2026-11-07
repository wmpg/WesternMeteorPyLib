# Source regions in NEOMOD

**`wmpl.Utils.NEOMODSourceRegions`** gives the probability that a body on a given orbit came from each
of the 12 source regions of the NEOMOD model of near-Earth objects (Nesvorný et al. 2023, 2024a, 2024b):
the ν6 secular resonance, the 3:1, 5:2, 7:3, 8:3, 9:4, 11:5 and 2:1 resonances with Jupiter, the weak
resonances of the inner belt, the Hungarias, the Phocaeas and the Jupiter-family comets. It takes the
orbit from a trajectory pickle, from elements typed on the command line, or from every output of a
REBOUND integration (`REBOUND.py --neomod`).

The values are those of the authors' own tool, `give_me_prob.f`, which reads them from the table
distributed with the NEOMOD Simulator. The authors mean the table to be used this way: "Researchers
interested in the probability that a specific NEO evolved from a particular source can obtain this
information from the ASCII table that is available along with the Simulator."

## Table of Contents

- [Quick start](#quick-start)
- [NEOMOD, NEOMOD2 or NEOMOD3?](#neomod-neomod2-or-neomod3)
- [The size of the body](#the-size-of-the-body)
- [Within the uncertainty of the orbit](#within-the-uncertainty-of-the-orbit)
- [Along a REBOUND integration](#along-a-rebound-integration)
- [What the probabilities assume](#what-the-probabilities-assume)
- [The model file](#the-model-file)
- [How it was checked](#how-it-was-checked)
- [References](#references)

---

## Quick start

```
python -m wmpl.Utils.NEOMODSourceRegions path/to/20210228_215416_trajectory.pickle
python -m wmpl.Utils.NEOMODSourceRegions path/to/trajectory.pickle --clones 1000 --seed 1
python -m wmpl.Utils.NEOMODSourceRegions --a 2.586 --e 0.618 --i 0.46 --size 28 19m 17
python -m wmpl.Rebound.REBOUND path/to/trajectory.pickle --days 365250 --mc 100 --neomod
```

The first run downloads the model file, 62 MB, once (see [The model file](#the-model-file)). The
published orbit of Winchcombe, at three sizes:

```
Diameter 19 m: H = 26.31
Source regions in NEOMOD2 (Nesvorny et al. 2024), Model136:
  Orbit: a = 2.5860 AU, e = 0.6180, i = 0.460 deg, q = 0.9879 AU
  Bin:   a 2.5-2.6 AU, e 0.60-0.64, i 0-4 deg
  H:     H 27.75-28.00 (H = 28.00), 1323 model NEOs in the bin
         H 26.25-26.50 (H = 26.31), 252.9 model NEOs in the bin
         H 17.00-17.25 (H = 17.00), 0.0333 model NEOs in the bin

  Source              H = 28.00    H = 26.31    H = 17.00
  3:1                     0.808        0.794        0.619
  nu6                     0.148        0.129        0.037
  8:3                     0.003        0.015        0.115
  5:2                     0.037        0.041        0.084
  inner weak res.         0.003        0.012        0.079
  11:5                    0.001        0.007        0.055
  ...
```

| Option | Description |
| :--- | :--- |
| `TRAJ_PATH` | Trajectory pickle to take the orbit from. Omit it to give the orbit by hand. |
| `--q`, `--a`, `--e`, `--i` | The orbit by hand: the perihelion distance or the semi-major axis (AU), the eccentricity and the inclination (deg). |
| `--size H\|MASS\|DIAMETER [...]` | The size of the body: its H, its mass ending in `kg` or `g`, or its diameter ending in `m`. Several values give a column each. See [The size of the body](#the-size-of-the-body). |
| `--clones N`, `--seed S` | The mean and the 95% interval of the probabilities over N clones of the orbit, drawn from the trajectory's covariance. See [Within the uncertainty of the orbit](#within-the-uncertainty-of-the-orbit). |
| `--model_file FILE` | A model file to read instead of the downloaded one, e.g. `input_neomod3.dat` from the authors' page. |
| `--save FILE` | Save the results to a JSON file. |

---

## NEOMOD, NEOMOD2 or NEOMOD3?

For source regions they are the same. The NEOMOD Simulator page distributes three model files,
`input_model.dat`, `input_neomod2.dat` and `input_neomod3.dat`. Below their 13-line headers, the three
hold the same table, byte for byte (566 828 bins): the best fit of NEOMOD2, "Model136". The headers
differ only in the distribution of sizes, which the source probabilities do not use:

| File | Header | Use |
| :--- | :--- | :--- |
| `input_model.dat` | NEOMOD2's distribution in H, 15 < H < 28 | `neomod_simulator.f` |
| `input_neomod2.dat` | the same, extended to 9 < H < 33 | `neomod2_simulator.f`, `magmod_simulator.f` |
| `input_neomod3.dat` | NEOMOD3's distribution in diameter, with albedos | `neomod3_simulator.f`, `diamod_simulator.f` |

- **NEOMOD** (Nesvorný et al. 2023) was fitted to the Catalina Sky Survey detections of 2005-2012. Its
  fit is no longer distributed: the page's own Readme calls `neomod_simulator.f` "the original NEOMOD2
  simulator".
- **NEOMOD2** (Nesvorný et al. 2024a) fitted the model again to the CSS detections of 2013-2022, over
  15 < H < 28. It is the orbital model of all three files.
- **NEOMOD3** (Nesvorný et al. 2024b) added the albedos from NEOWISE, to turn H into sizes. The
  orbits and the shares of the sources are still those of NEOMOD2.

The module downloads `input_neomod2.dat`, and reads any of the three with `--model_file`. The page
also holds `model1.tar` (July 2026), a NEOMOD3 sample of 6 million NEOs with D > 10 m (H, a, e, i, D,
albedo), not described in its Readme and not used here.

---

## The size of the body

The share of each source depends on the size of the body: in NEOMOD2 it changes linearly with H
between H = 15 and H = 28, and the disruption of NEOs near the Sun depends on H too. ν6 brings 6% of
the NEOs at H = 15 and 60% at H = 28, the inner weak resonances 18% and 1% (Table 2 of NEOMOD2). In
the Winchcombe example above, the 8:3 resonance goes from 0.3% at H = 28 to 12% at H = 17.

The model is calibrated on the Catalina Sky Survey for 15 < H < 28, and **H = 28 is a body of about
9 m** (D > 9 m for p_V = 0.14 in NEOMOD2; 8.7 m with the albedo used here, 1.3e6 kg). A trajectory
does not carry the size of the body, so:

- Without `--size`, the faintest bin of the model, 27.75 < H < 28, is used, as the NEOMOD Simulator
  does for H > 28, and the output says so. This is the right choice for nearly every fireball and
  meteorite fall, which are smaller than 9 m: 2008 TC3 (~4 m, H = 30.9) too.
- A body smaller than that given with `--size` (`--size 13kg`) also takes the faintest bin, flagged as
  fainter than the model reaches.
- A larger body, such as Chelyabinsk (~19 m, H ~26), needs its size: `--size 26.3`, `--size 1.2e7kg`
  or `--size 19m`.

Masses and diameters are converted with a density of 3500 kg/m³ and an albedo of 0.147, as
`wmpl.Utils.DebiasedMeteoroidPopulation.massToH` does: D = 1329 km 10^(-H/5)/sqrt(p_V).

For bodies smaller than the model reaches, the faintest bin is an extrapolation. Over the whole
population of that bin, ν6 is 0.64 and the 3:1 0.30. `wmpl.Utils.DebiasedMeteoroidPopulation`
(Deam et al. 2026), which `ParentBodySearch --chance_meteoroid` uses, fits the strengths of 6 of these
sources to fireballs at meteoroid sizes.

---

## Within the uncertainty of the orbit

The model is binned, 0.1 AU in a, 0.04 in e, 4 deg in i and 0.25 mag in H, and the probabilities of
an orbit are those of its bin, without interpolation, as in `give_me_prob.f`. An orbit near the edge of
a bin can change source probabilities a lot with a small change of its elements. This is common: of the
34 meteorites in `wmpl/share/Meteorites_with_Orbits.csv` with published errors of a, e and i, 22 have
fewer than 68% of clones drawn from those errors in the bin of the nominal orbit. Winchcombe moves from
the 2.5-2.6 AU bin to the 2.6-2.7 AU bin between its published orbit and the one of the pickle above
(a = 2.649 AU), and its 5:2 share goes from 0.04 to 0.13.

`--clones N` draws N clones of the trajectory's state vector from its covariance, with the draws of
`REBOUND.py --mc N --seed S` (see `wmpl.Utils.OrbitClones`), and gives the mean of each probability
over the clones in the model, with the interval 2.5-97.5% of the clones. The mean is the probability
of each source given the uncertainty of the orbit, and is the value to quote. The trajectory has to be
solved with the Monte Carlo uncertainties.

---

## Along a REBOUND integration

| Option | Description |
| :--- | :--- |
| `--neomod [all\|final]` | The source probabilities of the nominal solution and the clones at every output (`all`, the default), or at the end of the run only (`final`). |
| `--neomod_size H\|MASS\|DIAMETER` | The size of the object, as `--size` above. Without it the faintest bin is used. |

The model is loaded before the integration, so a missing network shows up at once and not at the end
of a long run. The report and the terminal give the probabilities at the first output after the object
left the Earth (before that its heliocentric osculating orbit is hyperbolic and outside the model) and
at the end, for the nominal and as the mean and the 2.5-97.5% interval of the clones:

```
Source regions in NEOMOD2 (Nesvorny et al. 2024), H 27.75-28.00 (no size given: the faintest bin of the model, for a body smaller than about 9 m):
                    t = -2.0 yr (leaving the Earth)         t = -1000.0 yr
  Source             nominal  clones mean [2.5-97.5%]        nominal  clones mean [2.5-97.5%]
  3:1                  0.701  0.692 [0.455, 0.808]             0.785  0.666 [0.376, 0.808]
  5:2                  0.132  0.144 [0.037, 0.411]             0.046  0.165 [0.037, 0.448]
  nu6                  0.145  0.142 [0.097, 0.148]             0.161  0.144 [0.080, 0.161]
  ...
```

It also writes:

| File | Contents |
| :--- | :--- |
| `rebound_neomod_sources.csv` | One row per output time and source: the nominal probability, the clones' mean, 2.5% and 97.5% percentiles, and the share of the clones in the model. |
| `rebound_neomod_sources.png` | Only with `all`: the probabilities against time as stacked areas, the nominal above and the clones' mean below, the time outside the model in grey. |
| `rebound_simulation_results.json` | The key `"neomod"`: the H bin, the sources, and every series of the CSV, with `null` outside the model and the reason the nominal is outside at each output. |

Over the 1000 years above (Winchcombe, 30 clones), the nominal jumps between bins as its elements
oscillate, while the mean of the clones changes smoothly. Beyond the predictability horizon the
nominal is one realization among the clones, and their mean is the value to read. NEOMOD gives where
the NEOs of each source are in steady state, so the probabilities of an orbit integrated back in time
are those of the population at that orbit, not a history of the body: a change along the run says how
much the answer depends on where the orbit happens to be in its secular cycle.

---

## What the probabilities assume

- The orbit is bound, with q < 1.3 AU, a ≤ 4.2 AU and i ≤ 88 deg, the region of the model. Otherwise
  no probabilities are given, with the reason (a Jupiter-family meteoroid with a > 4.2 AU is outside).
  A bin of that region without any model NEO gives none either. `give_me_prob.f` does not check q, and
  gives the values of a bin that straddles q = 1.3 AU for an orbit with q > 1.3 AU; this module does not.
- The shares are given to three decimals in the file, and sum to 1 within 0.004. They are returned as
  they are.
- Averaged over clones, each clone weighs the same: the mean is over the uncertainty of the orbit, not
  weighted by how many NEOs the model puts in each bin.

---

## The model file

The file is not part of wmpl. It is downloaded on first use from
`https://www2.boulder.swri.edu/~davidn/NEOMOD_Simulator/input_neomod2.dat`, checked against its SHA-256,
and kept as `wmpl/share/Nesvorny+_2024_NEOMOD2.dat` (ignored by git), or in `~/.cache/wmpl` if
`wmpl/share` cannot be written, as the Granvik et al. (2018) and Deam et al. (2026) models are. It
parses in 1.5 s. If the authors change the file, the SHA-256 no longer matches and the download is
refused rather than used unchecked.

The authors' server does not send the intermediate certificate of its chain (DigiCert Global G2 TLS
RSA SHA256 2020 CA1), so Python cannot verify it on its own: browsers and curl fetch the missing
certificate themselves, Python's OpenSSL does not. The module supplies that certificate, so the chain
is still verified up to a root of the system and the host name is checked. Without network, download
the file with a browser and give it with `--model_file`, or put it in `wmpl/share` under the name
above.

No licence is stated on the authors' page. Cite NEOMOD2 when using the results.

---

## How it was checked

- **Against `give_me_prob.f`.** For 220 000 orbits, 200 000 drawn uniformly over a, e, i and H and
  20 000 on the edges of the bins, the module gives the same values as a line-by-line transcription of
  `give_me_prob.f` for every orbit inside the model (94 407), and none for the rest. The only
  differences are the ones described above: 4349 orbits with q ≥ 1.3 AU and 434 with e = 1 exactly,
  to which `give_me_prob.f` gives the values of their bin.
- **Inclinations on a bin edge.** An inclination of 60 deg typed on the command line comes back from
  radians as 59.99999999999999 deg. It is rounded to 1e-9 deg, so that it falls in the 60-64 deg bin,
  as in `give_me_prob.f`.
- **Against Table 2 of NEOMOD2.** Over the whole population of the faintest bin, ν6 is 0.636 and the
  3:1 0.299, against the paper's medians at H = 28 of 0.595 and 0.313; the file is the best fit, not
  the median.
- **The meteorites with orbits** (`wmpl/share/Meteorites_with_Orbits.csv`, 51 falls): all are inside
  the model. In the faintest bin, ν6 is the most probable source of 36 of them, the 3:1 of 11 and the
  5:2 of 4. Between the bin at H = 17 and the faintest, the largest change of the probability of a
  source is 0.52 at the median.

---

## References

- Nesvorný, D., Deienno, R., Bottke, W. F., et al. (2023). NEOMOD: A New Orbital Distribution Model
  for Near-Earth Objects. *AJ* 166, 55. doi:10.3847/1538-3881/ace040
- Nesvorný, D., Vokrouhlický, D., Shelly, F., et al. (2024a). NEOMOD 2: An updated model of Near-Earth
  Objects from a decade of Catalina Sky Survey observations. *Icarus* 411, 115922.
  doi:10.1016/j.icarus.2023.115922
- Nesvorný, D., Vokrouhlický, D., Shelly, F., et al. (2024b). NEOMOD 3: The debiased size distribution
  of Near Earth Objects. *Icarus* 417, 116110. doi:10.1016/j.icarus.2024.116110
- The NEOMOD Simulator: https://www2.boulder.swri.edu/~davidn/NEOMOD_Simulator/
