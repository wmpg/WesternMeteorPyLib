""" A small METEOMOD archive in the format of the authors' meteomod.tar.gz, for the tests of
wmpl.Utils.METEOMODSourceRegions and of REBOUND.py --meteomod, which need neither the network nor the
19 MB published archive.
"""

import io
import tarfile


# The families of the test archive: number, label in maps.in, label in Table 5, map directory, N_neo and
#   Phi as written in Table 5, and the counts of the map as {(bin in a, bin in i), from 1: count}. The
#   second family shares the map of the first, as Massalia(2) does; the third is called CM in maps.in
#   and CI in Table 5, as Misa is; the N_neo of the second is written with LaTeX and a '!', as in the
#   published table
FAMILIES = [
    ('20', "Massalia (L)", "Massalia (L)", 'massalia-4', '1.1-2.8', '8.0-20',
        {(165, 2): 30.0, (166, 2): 10.0, (230, 5): 60.0}),
    ("20'", "Massalia$_{(2)}$ (L)", "Massalia$_{(2)}$ (L)", 'massalia-4', '$\\sim$28-55!', '200-400',
        None),
    ('569', "{\\bf Misa (CM)}", "Misa (CI)", 'misa-4', '1.4-8.1', '7.3-42!',
        {(165, 2): 10.0, (260, 1): 90.0}),
    ('3', "Juno (L/LL)", "Juno (L/LL)", 'juno-4', '0.3-1.0', '0.6-1.9',
        {(165, 2): 50.0, (230, 5): 50.0}),
    ('490', "{\\bf Veritas (CM)}", "Veritas (CM)", 'veritas-4_4628', '215-249!', '188-217',
        {(260, 1): 20.0, (61, 61): 80.0}),
    ]



def familyFlux(number):
    """ The middle of the Phi of a family of the test archive. """

    low, high = [f for f in FAMILIES if f[0] == number][0][5].split('-')

    return 0.5*(float(low) + float(high.replace('!', '')))



def familyNeo(number):
    """ The middle of the N_neo of a family of the test archive. """

    text = [f for f in FAMILIES if f[0] == number][0][4].replace('$\\sim$', '').replace('!', '')
    low, high = text.split('-')

    return 0.5*(float(low) + float(high))



def familyMap(number):
    """ The counts of the map of a family of the test archive, as {(bin in a, bin in i): count}. """

    directory = [f for f in FAMILIES if f[0] == number][0][3]

    return [f for f in FAMILIES if (f[3] == directory) and (f[6] is not None)][0][6]



def _mapText(counts):
    """ A map.out file: the bins in a and i, a and i of the bin and the count, a blank line between the
        bins in a, and some bins with zero counts, as in the published files. """

    lines = []
    for ia in sorted(set([k[0] for k in counts] + [1, 350])):
        for ii in (1, 2, 90):
            lines.append("{:12d}{:12d}{:26.16f}{:26.16f}{:26.16f}".format(ia, ii, 0.01*(ia - 1), 1.0*(ii - 1),
                counts.get((ia, ii), 0.0)))
        for (ja, ji), count in sorted(counts.items()):
            if (ja == ia) and (ji not in (1, 2, 90)):
                lines.append("{:12d}{:12d}{:26.16f}{:26.16f}{:26.16f}".format(ja, ji, 0.01*(ja - 1),
                    1.0*(ji - 1), count))
        lines.append("")

    return "\n".join(lines) + "\n"



def _table5Text():
    """ A tab5.tex with the header, a commented row and a summary row of the published one. """

    lines = ["\\begin{tabular}{rllrrrrrrrll}",
        "% num. fam.                   res.      tau_g18  neo      mb         rho    N_mb            N_neo"
            "         Phi              obs.",
        "%2732 & Witt (?)               & 5:2     & 0.68 & 0.91     & 118      & 2.5  & 1.5-4         "
            "& 1.2-3.1     & 3.5-9.0        &          & \\\\"]

    for number, _, label, _, n_neo, flux, _ in FAMILIES:
        lines.append("{:>4s} & {:<22s} & 3:1     & 1.83 & 3.83     & 139      & 2.5! & 0.4-1         "
            "& {:<11s} & {:<14s} &          & \\\\".format(number, label, n_neo, flux))

    lines += ["\\\\",
        "     & H                      &         &      &          &          &      & 135-270       "
            "& 37-74       & 188-377        & \\s{34\\%}    & \\\\",
        "\\end{tabular}"]

    return "\n".join(lines) + "\n"



def _mapsInText():
    """ A maps.in, with a commented family and a separator line, as the published one. """

    lines = []
    for number, label, _, directory, _, _, _ in FAMILIES:
        lines.append("{:>4s} & {:<22s} & {:s}".format(number, label, directory))
        lines.append("\\\\")

    lines.append("%4203 & Brucato (?)            & brucato-4")

    return "\n".join(lines) + "\n"



def writeArchive(file_path, maps_in=None):
    """ Write the test archive.

    Arguments:
        file_path: [str] The .tar.gz file.

    Keyword arguments:
        maps_in: [str] A maps.in to write instead of the one of FAMILIES. None by default.
    """

    files = {
        'meteomod/maps.in': _mapsInText() if maps_in is None else maps_in,
        'meteomod/tab5.tex': _table5Text(),
        'meteomod/meteomod.py': "# The authors' script, not read\n",
        }

    for number, _, _, directory, _, _, counts in FAMILIES:
        if counts is not None:
            files['meteomod/map/{:s}/map.out'.format(directory)] = _mapText(counts)

    with tarfile.open(file_path, 'w:gz') as archive:
        for name, text in sorted(files.items()):
            data = text.encode('utf-8')
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
