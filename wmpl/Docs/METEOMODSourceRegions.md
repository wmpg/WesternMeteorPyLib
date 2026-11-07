# Source regions in METEOMOD

**`wmpl.Utils.METEOMODSourceRegions`** gives the probability that a meteoroid on a given orbit came from
each of the 52 asteroid families of METEOMOD, the orbital distribution model of metre-sized meteoroids of
Brož et al. (2024a, 2024b). It takes the orbit from a trajectory pickle, from elements typed on the
command line, or from every output of a REBOUND integration (`REBOUND.py --meteomod`).

It is the counterpart of [Source regions in NEOMOD](NEOMODSourceRegions.md). NEOMOD gives the resonances
and other regions through which NEOs leave the main belt; METEOMOD gives the asteroid families the
meteoroids were released from, with the taxonomy of each family, so that a meteorite of known type is
compared with the families of that type.

## Table of Contents

- [Quick start](#quick-start)
- [The model](#the-model)
- [Which families to compare](#which-families-to-compare)
- [Why not the authors' meteomod.py](#why-not-the-authors-meteomodpy)
- [Table 5 of the download and of the paper](#table-5-of-the-download-and-of-the-paper)
- [Along a REBOUND integration](#along-a-rebound-integration)
- [What the probabilities assume](#what-the-probabilities-assume)
- [The model file](#the-model-file)
- [How it was checked](#how-it-was-checked)
- [References](#references)

---

## Quick start

```
python -m wmpl.Utils.METEOMODSourceRegions path/to/trajectory.pickle --types L
python -m wmpl.Utils.METEOMODSourceRegions path/to/trajectory.pickle --types CM --clones 1000 --seed 1
python -m wmpl.Utils.METEOMODSourceRegions --a 2.5855 --e 0.6183 --i 0.460 --types CM
python -m wmpl.Utils.METEOMODSourceRegions --list
python -m wmpl.Rebound.REBOUND path/to/trajectory.pickle --days 36525 --mc 100 --meteomod --meteomod_types CM
```

The first run downloads the model, 19 MB, once (see [The model file](#the-model-file)). The orbit of
Winchcombe, a CM chondrite, among the CM families:

```
Source regions in METEOMOD (Broz et al. 2024), the 9 families of type CM, weighted by the meteoroid flux Phi of Table 5 (1-m meteoroids):
  Orbit: a = 2.5855 AU, e = 0.6183, i = 0.460 deg, q = 0.9869 AU
  Bin:   a 2.58-2.59 AU, i 0-1 deg

  No.   Family                          p    M_j(a,i)  Phi [1e-9/km2/yr]
  1128  Astrid (CM)                 0.435   3.834e-04               4.65
  3815  Konig (CM)                  0.216   1.669e-05              53.00
  490   Veritas (CM)                0.109   2.201e-06             202.50
  145   Adeona (CM)                 0.075   3.417e-05               9.05
  144   Vibilia (CM)                0.073   5.468e-05               5.50
  ...
```

| Option | Description |
| :--- | :--- |
| `TRAJ_PATH` | Trajectory pickle to take the orbit from. Omit it to give the orbit by hand. |
| `--q`, `--a`, `--e`, `--i` | The orbit by hand: the perihelion distance or the semi-major axis (AU), the eccentricity and the inclination (deg). |
| `--types TYPE [...]` | Compare only the families of these types (`L`, `CM CI`, ...). A family of type L/LL is kept for L and for LL. By default every family, each once, with the probabilities also summed by type (see [Without the type of the meteorite](#without-the-type-of-the-meteorite)). |
| `--sources NUMBER [...]` | Compare these families, by their number in Table 5 (`20` is Massalia, `"20'"` Massalia(2)). |
| `--weight {flux,neo}` | `flux`, the meteoroid flux on the Earth, as the model defines it (default); `neo`, the number of bodies in the NEO region, as the authors' `meteomod.py` computes it, to compare. |
| `--table {published,download}` | The values of Table 5: `published`, as in the A&A paper (default), or `download`, as in the `tab5.tex` of the archive, which the authors' page uses. See [Table 5 of the download and of the paper](#table-5-of-the-download-and-of-the-paper). |
| `--list` | List the families, their types, maps, N_neo and flux. |
| `--clones N`, `--seed S` | The mean and the 95% interval over N clones of the orbit, drawn from the trajectory's covariance, as in [NEOMOD](NEOMODSourceRegions.md#within-the-uncertainty-of-the-orbit). |
| `--model_file FILE` | An archive to read instead of the downloaded one. |
| `--save FILE` | Save the results to a JSON file. |

---

## The model

METEOMOD follows 1-m bodies released from 52 asteroid families, at least 1000 per family, with the
Yarkovsky and YORP effects, collisional reorientations and the gravity of 11 massive bodies, from the
main belt to the NEO region. For each family *j* it gives a map *M_j(a, i)* of where its bodies are once
in the NEO region (*e* > 1 − 1.3 AU/*a*), in bins of 0.01 AU in *a*, from 0 to 3.5 AU, and 1° in *i*,
from 0 to 90°, normalised to 1. Only *a* and *i* are kept, because the eccentricity the bodies had in the
family has been lost by then.

Each family is weighted by its meteoroid flux on the Earth, Φ_j = p_j N_neo,j (Eq. 9 of Brož et al.
2024a). N_neo is its number of 1-m bodies in the NEO region, from its size distribution extrapolated
with a collisional model, and p_j their mean collision probability with the Earth. Both are in Table 5
of Brož et al. (2024a). The probability that a meteoroid at (*a*, *i*) came from family *j* is

    P_j = M_j(a, i) Φ_j / Σ_k M_k(a, i) Φ_k

over the families compared. The METEOMOD page says it this way: "Since we study meteoroids colliding
with the Earth, the maps should be weighted by the flux Φ = pcoll Nneo(>1 m) [...] The probability of
coming from a source is then estimated as pj = Mj(a, i) Φ".

---

## Which families to compare

The families carry the taxonomy of their bodies, so a meteorite is compared with the families of its
type: `--types L` for an L chondrite, `--types CM` for a CM, as in the examples of the METEOMOD page
(Massalia(2) for the L chondrites, Koronis(2) for the H chondrites, Flora for the LL chondrites).
`--list` gives every family with its type.

- **The types are those of Table 5.** `maps.in`, which gives the map of each family, calls Misa CM and
  Brasilia CH. Table 5, the paper and the page call them CI and M. The module follows Table 5, so that
  `--types CM` gives the 9 CM families of the authors' example, without Misa.
- **Variants share a map.** Koronis(2), the young population of the 2.1° dust band, and Karin share the
  map of Karin. Massalia and Massalia(2), the young population of the 1.4° dust band, share that of
  Massalia; they are two populations, compared together in the authors' L example.
- **Nysa appears twice** in the authors' list, as 44 (LL) and 44′ (E), with the same map and the same
  flux. When all the families are compared it is taken once, as LL/E. Given both by number, it is counted
  twice, with a warning.

### Without the type of the meteorite

Without `--types` every family is compared, each once, and the output adds the probabilities summed by
type (in the report of REBOUND too, and under `"by_type"` in the JSON). This is the probability for a
meteoroid of any type on that orbit, and it is dominated by the families with the largest fluxes:
Massalia(2), Koronis(2) and Veritas. The maps in (*a*, *i*) overlap, so the orbit alone hardly tells the
type, and the output says so:

| Fall (its type) | Summed by type, all the families compared |
| :--- | :--- |
| Novo Mesto (L) | L 0.40, CI 0.19, H 0.18, HED 0.08 |
| Hamburg (H) | H 0.80, L 0.11, CI 0.05 |
| Chelyabinsk (LL) | L 0.46, H 0.38, CI 0.07 |
| Winchcombe (CM) | H 0.47, L 0.37, CI 0.12, CM 0.03 |

For a fall of known type, compare the families of that type.

---

## Why not the authors' meteomod.py

The archive contains the authors' script, `meteomod.py`. This module does not run it. It reads the maps
and Table 5 itself, for three reasons.

- **The weight.** `meteomod.py` weights the maps by N_neo, not by Φ. It is a copy of `neomod.py`, the
  authors' tool for kilometre-sized NEOs, for which N_neo is the right weight. It reads Φ from Table 5
  and does not use it. For a body that hit the Earth, this leaves out the collision probability, which
  differs between the families by a factor of 11: Φ/N_neo is 0.70 for Hygiea and 7.8 for Nysa, in
  10⁻⁹ km⁻² yr⁻¹ per 10⁸ bodies. The paper says so (Brož et al. 2024a, Sect. 4): "Using Eq. (8)
  [N_neo] is not sufficient though, because C-type families are found at very different locations. One
  has to compute collisional probabilities p with the Earth (Table C.4), and estimate the meteoroid flux".
  For the ordinary chondrites, the Supplementary Information of Brož et al. (2024b) notes that "at least
  for the most relevant families the fluxes are not so different from populations; with the obvious
  exception of Phocaea". The effect, measured:
  - Winchcombe among the CM families is from Veritas (0.30) with N_neo, and from Astrid (0.43, Veritas
    0.11) with Φ. Veritas has 270 times more bodies than Astrid but a collision probability 6
    times lower.
  - Of the 47 meteorites of `wmpl/share/Meteorites_with_Orbits.csv` whose type has families, 2 change
    their most probable family, both CM: Sutter's Mill (Veritas → König) and Maribo (Theobalda →
    Astrid).
  - For others the probability of the first family moves by up to 0.26: Benešov (Eunomia 0.63 → 0.37),
    Golden (Massalia(2) 0.59 → 0.81) and Grimsby (Koronis(2) 0.43 → 0.64).
  - The L and H chondrites of the examples still come from Massalia(2) and Koronis(2).

  `--weight neo` gives the values of `meteomod.py`, to compare, with a warning.
- **The maps.** `meteomod.py` reads the maps from `maps.npz` when it is there. That file is a pickled
  dictionary, and loading it would run code from a downloaded file. The maps are read here from the text
  files (`map/*/map.out`), which give the same 55 maps to the last digit.
- **The orbit is not checked.** In `meteomod.py`:
  - a hyperbolic orbit (*a* < 0) indexes the map from its far end, with no message;
  - *a* ≥ 3.5 AU or *i* ≥ 90° stops it with an IndexError;
  - an orbit outside the NEO region gets the probabilities of its (*a*, *i*) bin;
  - the eccentricity is read, but neither used nor written to `meteomod.out`.

  Here *e* is used and printed, and such orbits get no probabilities, with the reason.
- **Reading the files.** `meteomod.py` reads its files from the working directory, so it fails, or reads
  other files, when run from elsewhere. `read_maps` opens `maps.in` whatever file it is given. An orbit
  given without a designation stops `read_orbits` with an IndexError. Here the files are read from the
  archive, wherever it is.

Otherwise the values are those of `meteomod.py`: with `--weight neo` they agree within 6×10⁻¹⁶ with a
transcription of its `compute_p`.

---

## Table 5 of the download and of the paper

The `tab5.tex` of the archive, which the authors' page uses, is dated 10 October 2024, after the
publication. It differs from Table 5 as published in A&A (the same as arXiv:2406.19727v2) for three
families:

| Family | Published (default) | Download (`--table download`) |
| :--- | :--- | :--- |
| Hungaria (434) | τ_mb "300?" My, N_neo 5.9–9.4, Φ 7.6–12 | τ_mb 61 My, N_neo 29–46, Φ 37–59 |
| Baptistina (298) | Φ 11–19 | Φ 8.0–13 |
| Juno (3) | type L | type L/LL |

By default the published values are used, since the paper is what is cited. The JSON records the table
(`"table"`) and the families taken from it (`"table_changes"`), and `--list` marks them. In both versions
Φ/N_neo of Hungaria matches its collision probability in Table C.4. For Baptistina it matches only in the
published table. Witt (2732) and Brucato (4203), in the paper, are left out: Witt has no map in the
archive, and Brucato is commented out of its list of families. The middle of each range is used, as the
authors do.

--- | :--- | :--- |
| Hungaria (434) | τ_mb 61 My, N_neo 29–46, Φ 37–59 | τ_mb "300?" My, N_neo 5.9–9.4, Φ 7.6–12 |
| Baptistina (298) | Φ 8.0–13 | Φ 11–19 |

Witt (2732) and Brucato (4203), in the paper, are left out of the archive. In both versions Φ/N_neo of
Hungaria matches its collision probability in Table C.4. For Baptistina it matches only in the paper.
The middle of each range is used, as the authors do.

---

## Along a REBOUND integration

| Option | Description |
| :--- | :--- |
| `--meteomod [all\|final]` | The probabilities of the nominal solution and the clones at every output (`all`, the default), or at the end only (`final`). |
| `--meteomod_types TYPE [...]` | Compare only the families of these types, as `--types` above. |
| `--meteomod_sources NUMBER [...]` | Compare these families, as `--sources` above. |
| `--meteomod_table {published,download}` | The values of Table 5, as `--table` above. |

The outputs are those of `--neomod`, under their own names, described in
[Source regions in NEOMOD](NEOMODSourceRegions.md#along-a-rebound-integration):

- the report, with the probabilities at the departure from the Earth, at the end, at the predictability
  horizon and for the representative clone, then the papers to cite;
- `rebound_meteomod_sources.csv`;
- `rebound_meteomod_sources.png` (with `all`);
- the key `"meteomod"` of the JSON, which also holds the weight, the types, the table, and N_neo and Φ
  of each family compared, and, without `--meteomod_types`, the probabilities summed by type
  (`"by_type"`), also given in the report.

`--neomod` and `--meteomod` can be given together. The figure draws a colour per family when there are 7
or fewer. With more, it draws the 7 with the largest probabilities of the run and groups the rest as
"other".

---

## What the probabilities assume

- The maps are of 1-m bodies and Φ of 1-m meteoroids. A much larger body, of a size whose population
  differs between the families, would need other weights.
- The bin is taken as it is, without interpolation, as `meteomod.py` does.
- The orbit is in the model only if it is bound, with *q* < 1.3 AU, 0 ≤ *a* < 3.5 AU and *i* < 90°. A bin
  where none of the families compared has bodies gives no probabilities.
- Averaged over clones, each clone weighs the same.

---

## The model file

The archive is not part of wmpl. It is downloaded on first use from
`https://sirrah.troja.mff.cuni.cz/~mira/meteomod/meteomod.tar.gz` and checked against its SHA-256. It is
kept as `wmpl/share/Broz+_2024_METEOMOD.tar.gz` (ignored by git), or in `~/.cache/wmpl` if `wmpl/share`
cannot be written, as the NEOMOD, Granvik et al. (2018) and Deam et al. (2026) models are.

- Nothing is extracted: `maps.in`, `tab5.tex` and the 52 maps are read from the archive, in 4 s.
- If the authors change the archive, the SHA-256 no longer matches and the download is refused.
- No licence is stated on the page. Cite the papers when using the results.

---

## How it was checked

- **Against `meteomod.py`, transcribed.** With `--weight neo`, the probabilities agree with a
  transcription of its `read_tab5`, `read_map` and `compute_p` within 6×10⁻¹⁶. The comparison covers
  36 626 orbits and selections of families (all, CM, L, and Nysa (E) with Hungaria), 1000 of the orbits
  on the edges of the bins.
- **Against `maps.npz`.** The maps read from the text files are those of the pickled `maps.npz`, to the
  last digit, for the 55 entries. `maps.npz` was read with an unpickler limited to numpy arrays.
- **The meteorites with orbits.** The 47 of `wmpl/share/Meteorites_with_Orbits.csv` whose type has
  families were compared, with both weights (see above).

---

## References

- Brož, M., Vernazza, P., Marsset, M., Binzel, R. P., DeMeo, F., et al. (2024a). Source regions of
  carbonaceous meteorites and near-Earth objects. *A&A* 689, A183. doi:10.1051/0004-6361/202450532
- Brož, M., Vernazza, P., Marsset, M., DeMeo, F. E., Binzel, R. P., Vokrouhlický, D. & Nesvorný, D.
  (2024b). Young asteroid families as the primary source of meteorites. *Nature* 634, 566–571.
  doi:10.1038/s41586-024-08006-7
- Marsset, M., Vernazza, P., Brož, M., et al. (2024). The Massalia asteroid family as the origin of
  ordinary L chondrites. *Nature* 634, 561–565. doi:10.1038/s41586-024-08007-6
- METEOMOD: https://sirrah.troja.mff.cuni.cz/~mira/meteomod/
