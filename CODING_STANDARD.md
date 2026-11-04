# WMPL coding standard

This standard describes how code in WMPL is written. It was extracted from the code Denis Vida wrote between 2017 and
November 2022, before AI coding tools were in use: about 48 000 lines in 74 modules, measured line by line with
`git blame`. Where that code is not consistent, the most frequent form is the standard. The percentages in
parentheses give how often the rule holds in that code. The same rules hold with a June 2021 cutoff (before GitHub
Copilot), so they are long-standing habits, not drift.

New code must read as if it was written by the same person as the code around it. A change that is correct but does
not follow this standard will be asked to change before it is merged.


## 1. Layout

- Indent with 4 spaces, no tabs.
- Keep lines within **110 characters**. Long lines are wrapped at around 100 (the median wrap point is 98). Long
  strings that cannot be split, such as argparse `help` texts, may run over.
- Only ASCII characters in code, comments and docstrings (4 non-ASCII characters in 48 000 lines). Write `deg`,
  `+/-`, `sigma`, `km/s`, `->`, `>=` instead of `°`, `±`, `σ`, `→`, `≥`.
- No trailing whitespace.


## 2. Blank lines

- **3 blank lines** between top-level functions and classes (43%, the next most common are 4 and 2).
- **2 blank lines** between the imports and the first function or class (55%).
- **3 blank lines** between the methods of a class too (50%, the next most common is 2).
- The first method of a class follows the `class` line directly, or after one blank line.
- No blank line between the `def` line and the docstring (90%).
- One blank line after the docstring (98%).
- Inside a function, blank lines separate the logical blocks of code: **one blank line before each commented block**
  (75%), two before a larger section (21%).


## 3. Comments

Comments are short, plain labels for the block of code below them.

- A comment line above each logical block of a few lines, with a blank line above the comment (99%). There is about
  one comment line for every 3 lines of code, and about half of the blocks separated by blank lines start with a
  comment.
- **One line**, typically 6 words, rarely more than 12. 92% of comments are a single line.
- **One sentence, starting with a capital letter (81%), with no period at the end (99%).** Comments with two
  sentences are practically absent (19 of 5 500).
- One space after the hash: `# Compute the radiant` (93%).
- A comment that has to wrap continues with **three spaces after the hash** (`#   `), the dominant form since
  2018:

  ```python
  # Reset the mass if using the ablation modelling (prevent for getting stuck on small masses that produce
  #   too faint meteors)
  ```

- Inline comments at the end of a code line are rare (68 in 48 000 lines). Put the comment on the line above.
- No dashes, semicolons or colons used to join clauses in comments (under 1%).
- Comments describe what the code does, not the history of the change. Do not leave comments about removed code
  ("this must not be wrapped any more", "previously this used ...").
- Tags are written as `# NOTE: ...` (the only tag used).
- Sections in long functions and scripts are marked with `### TEXT ###` headers, or a full line of `#`:

  ```python
  ### Parse command line arguments ###

  ##############################################################################################################
  ```

- Do not add commented-out code. The existing code has some (11% of comment lines), but new code should not.

Example of the typical density and form (`Trajectory.py`):

```python
            # If no lags were negative (meaning all fits were bad), use the initially estimated initial
            #   velocity
            if not stddev_list:
```


## 4. Docstrings

- Always triple double quotes `"""` (100%).
- A space after the opening quotes: `""" Compute the ...` (93%).
- A one-line docstring closes on the same line, with a space before the quotes: `""" Load the plate. """` (98%).
- A wrapped summary continues on the next line indented 4 more spaces (92%).
- Sections are named exactly **`Arguments:`**, **`Keyword arguments:`** and **`Return:`** (never `Args:`,
  `Parameters:`, `Returns:`). There is a blank line before each section (99%). The section headers line up with
  the opening quotes (99%) and their items are indented 4 more spaces (99%).
- Each argument is written as `name: [type] Description.`, with the type in square brackets: `[float]`, `[int]`,
  `[str]`, `[bool]`, `[list]`, `[ndarray]`, `[object]`, `[datetime]`, etc. A description that wraps continues 4
  spaces further in (76%).
- Units go in parentheses: `(deg)`, `(m/s)`, `(km)` (71%, the rest write "in degrees").
- Defaults are given at the end of the description as `... by default.` (92%), e.g. `None by default.`
- The return value is written either as `[type] Description.` or as `name: [type] Description.` for each returned
  value.
- The closing quotes go on their own line (99%).

From `Trajectory.py`:

```python
def copyUncertainties(traj_source, traj_target, copy_mc_traj_instances=False):
    """ Copy uncertainties from one trajectory to the other.

    Arguments:
        traj_source: [Trajectory object] Trajectory object with uncertainties.
        traj_target: [Trajectory object] Trajectory object to which uncertainties will be copied.

    Keyword arguments:
        copy_mc_traj_instances: [bool] Copy all trajectory instances generated during the MC procedure.
            This will make the trajectory pickle file very large. False by default.

    Return:
        traj_target: [Trajectory object] Target trajectory object with copied uncertainties.
    """

    # Copy covariance matrices
    traj_target.orbit_cov, traj_target.state_vect_cov = copy.deepcopy(traj_source.orbit_cov), \
        copy.deepcopy(traj_source.state_vect_cov)

    # Copy uncertainties
    traj_target.uncertainties = copy.deepcopy(traj_source.uncertainties)

    # Handle individual trajectory instances
    if not copy_mc_traj_instances:
        if traj_target.uncertainties is not None:
            del traj_target.uncertainties.mc_traj_list
            traj_target.uncertainties.mc_traj_list = []


    return traj_target
```

Several returned values can also be listed as a tuple (`MetSimErosion.py`):

```python
    Return:
        (es, ev):
            - es: [float] Energy received per unit cross-section (J/m^2)
            - ev: [float] Energy received per unit mass (J/kg).
```

Every top-level function has a docstring (92%), and so do most methods (80%).

Module files start with a module docstring (77%): `""" Loading and handling scale and exact plates. """`


## 5. Naming

- Functions and methods: **camelCase** (95% / 82%): `calcOrbit`, `fitAtmPoly`, `raDec2AltAz`.
- Variables and arguments: **snake_case** (96%): `v_init`, `jdt_ref`, `ht_data`.
- Module-level constants: `UPPER_CASE`.
- Classes: `CapWords`, written explicitly with `object` as the base: `class Trajectory(object):` (86%).


## 6. Spacing in expressions

| Operator | Form | Frequency |
|---|---|---|
| `*`, `/`, `**`, `//`, `%` | no spaces: `a*b/c**2`, `x%(2*np.pi)` | 97-100% |
| `+`, `-` (binary) | spaces: `a + b - c` | 97-100% |
| `==`, `!=`, `<`, `>`, `<=`, `>=` | spaces | 98-100% |
| `=` in an assignment, `+=`, `-=`, `*=`, `/=` | spaces | 100% |
| `=` in a keyword argument or a default value | no spaces: `f(x, verbose=True)` | 98% |
| `,` | one space after | 95% |
| slice `:` | no spaces: `x[1:-1]` | 78% |
| dictionary `:` | space after only: `{'name': value}` | 80% |
| `(` | no space before it after a name, none inside | 91-100% |

So a formula reads like this: `dec_corr = np.sign(C)*np.arccos(np.sqrt(A**2 + B**2))`.


## 7. Strings

- Format with **`.format()`** (567 uses). Do not use f-strings (none) or `%` (2 uses).
- Quotes:
  - Single quotes for identifier-like strings: dictionary keys, keyword values, argparse names, plot labels and
    file names (`'ablation'`, `'deg'`, `color='k'`; 67-99% depending on use).
  - Double quotes for text meant to be read and for `.format()` templates (`"Solving the trajectory..."`,
    `"{:.2f} km/s".format(v)`; 57-83%).
  - Empty strings as `""` (96%).


## 8. Line continuation

- Continue long statements with a **backslash at the end of the line, also inside brackets** (64% of continued
  lines are inside brackets and still end with `\`).
- The continuation is indented **4 spaces** relative to the start of the statement (89%). A wrapped `def` signature
  continues 4 spaces in (93%).

```python
from wmpl.Utils.TrajConversions import geo2Cartesian, raDec2ECI, altAz2RADec_vect, \
    raDec2AltAz_vect, jd2Date
```

```python
    # Copy covariance matrices
    traj_target.orbit_cov, traj_target.state_vect_cov = copy.deepcopy(traj_source.orbit_cov), \
        copy.deepcopy(traj_source.state_vect_cov)
```


## 9. Imports

- In order: module docstring, `from __future__ import print_function, division, absolute_import`, standard
  library, third-party packages, then WMPL modules.
- `import numpy as np`, `import scipy.optimize`, `import matplotlib.pyplot as plt`.
- WMPL imports by name: `from wmpl.Utils.TrajConversions import date2JD, jd2Date`. Wrap a long import list with a
  backslash.
- Import inside a function only for optional or heavy dependencies (13%).


## 10. Idioms

- `x is None`, never `x == None` (99%).
- Convert angles with `np.radians()` and `np.degrees()`, not `np.deg2rad()` / `np.rad2deg()` (100%).
- Prefer numpy functions to `math` (70%).
- Floats are written with a digit after the point: `1.0`, not `1.` (97%).
- Return several values as a plain tuple: `return a, b`, without parentheses (99%).
- Print messages with `print()`.
- No type hints.
- Catch a named exception (`except ValueError:`, `except Exception:`), never a bare `except:`. This is the one
  rule that does not follow the most frequent form: the existing code uses a bare `except:` in 56% of cases.


## 11. Command line scripts

From `Formats/CAMS.py`:

```python
if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on the given FTPdetectinfo file. It is assumed that only one meteor per file is given.")

    arg_parser.add_argument('ftpdetectinfo_path', nargs=1, metavar='FTP_PATH', type=str, \
        help='Path to the FTPdetectinfo file. It is assumed that the CameraSites.txt and CameraTimeOffsets.txt are in the same folder.')

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser, skip_velpart=True)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
```

- The guard is written `if __name__ == "__main__":` (89%).
- The parser is `arg_parser` (96%) and the parsed arguments `cml_args` (96%).
- Argument names and options are in single quotes (99%).


## 12. Tests

The pre-2022 code has no pytest tests. It runs examples in the `if __name__ == "__main__":` block of each module.
New tests go in a `Tests/` folder next to the module, are named `test_<Module>.py`, and follow every rule above,
including docstrings and comments.


## 13. Review checklist

These are the most common ways new code departs from this standard:

- [ ] Multi-sentence comments or comment paragraphs, and comments that end with a period.
- [ ] Comments that explain the history of the change or the code that was removed.
- [ ] Non-ASCII characters (`°`, `±`, `σ`, `→`, `—`, `☉`).
- [ ] f-strings.
- [ ] Spaces around `*` and `/`, or missing spaces around `+` and `-`.
- [ ] `Args:` / `Returns:` / `Parameters:` docstring sections, or arguments without a `[type]`.
- [ ] Docstrings starting `"""Text` without the space.
- [ ] snake_case function names.
- [ ] Two blank lines between top-level functions, or no blank line before a comment block.
- [ ] Parenthesised implicit continuation instead of a trailing backslash.
- [ ] `np.deg2rad` / `np.rad2deg`, `x == None`, `1.` floats, `class Foo:` without `object`.
- [ ] Type hints.
- [ ] Inline end-of-line comments.
