""" Small model files in the format of the NEOMOD Simulator, with the binning of NEOMOD2, for the tests
of wmpl.Utils.NEOMODSourceRegions and of REBOUND.py --neomod, which need neither the network nor the
62 MB published file.
"""


# The binning of NEOMOD2: H, a (AU), e, i (deg) as (number of bins, min, max)
BINNING = {'h': (52, 15.0, 28.0), 'a': (42, 0.0, 4.2), 'e': (25, 0.0, 1.0), 'incl_deg': (22, 0.0, 88.0)}

# The sources as the header of the published file names them
SOURCES = ['nu6', '3:1', '5:2', '7:3', '8:3', '9:4', '11:5', '2:1', 'inner', 'Hun', 'Pho', 'JFC']



def writeModel(file_path, rows, model_name="TestModel"):
    """ Write a model file in the format of the NEOMOD Simulator.

    Arguments:
        file_path: [str] The file.
        rows: [list] (iH, ia, ie, ii, number of NEOs, shares of the 12 sources), the bins from 1.

    Keyword arguments:
        model_name: [str] The name in the first line. "TestModel" by default.
    """

    lines = [model_name,
        "   52 15.0 28.0 ! absolute magnitude binning: # of H bins, Hmin,Hmax",
        "   17.75 911.6 ! reference magnitude Href, # of NEOs with H<H_ref (from the MultiNest fit)",
        "   8            ! absolute magnitude distribution - # of segments",
        "   9.0 15.0 16.5 17.5 20.0 24.0 25.0 28.0 33.0 ! segment boundaries",
        "   0.38 -1.0 0.38 0.37 0.31 0.53 0.50 0.50 ! segment slopes (-1.0 indicates N(<15)=50)",
        "   42 0.0 4.2   ! semimajor axis binning: # of a bins, amin (au), amax(au)",
        "   25 0.0 1.0   ! eccentricity binning: # of e bins, emin, emax",
        "   22 0.0 88.0  ! orbital inclination binning: # of i bins, imin (deg), imax (deg)",
        "   12           ! number of model sources",
        "-"*113,
        "  iH  ia  ie  ii  binned mod.    nu6    3:1    5:2    7:3    8:3    9:4   11:5    2:1  inner    Hun    "
            "Pho    JFC",
        "-"*113]

    for ih, ia, ie, ii, n_neo, row_shares in rows:
        lines.append("{:4d}{:4d}{:4d}{:4d}  {:11.5E}".format(ih, ia, ie, ii, n_neo) \
            + "".join(" {:6.3f}".format(s) for s in row_shares))

    with open(file_path, 'w') as f:
        f.write("\n".join(lines) + "\n")



def shares(**values):
    """ Shares of the 12 sources, zero but for those given by name, a resonance as _3_1 for 3:1. """

    return [values.get(s if ':' not in s else '_' + s.replace(':', '_'), 0.0) for s in SOURCES]
