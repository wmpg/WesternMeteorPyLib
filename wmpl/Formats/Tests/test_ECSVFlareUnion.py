""" Tests for the flares of the event in the report of the flares of ECSV files: the flares of different
    stations taken as the same flare, when they are consistent and each one is the only consistent flare of
    its station for the other, and the ambiguous ones left alone.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlareUnion.py -v
"""

from __future__ import print_function, division, absolute_import

from wmpl.Formats.ECSV import flareReport
from wmpl.Formats.Tests.test_ECSVFlarePressure import flarePoint



def eventRows(flares):
    """ Take the rows of the flares of the event from the report of the given flare points of 25 fps cameras.

    Arguments:
        flares: [list] (station ID, time) of every flare point.

    Return:
        (joined, alone): [tuple] Lines of the flares of the event joined from several stations, and of those
            seen by one station.
    """

    report = flareReport([flarePoint(station_id, t, 0.04) for station_id, t in flares])
    lines = report.split("Flares of the event")[1].splitlines()

    joined = [line for line in lines if line.startswith("      Flares ")]
    alone = [line for line in lines if line.startswith("      Flare ")]

    return joined, alone



def testTheFlaresOfTwoAndThreeStationsAreJoined():
    """ Check that the same flares seen by two and by three stations are each one flare of the event. """

    joined, alone = eventRows([('A', 0.50), ('A', 1.20), ('B', 0.52), ('B', 1.21), ('C', 1.22)])

    assert alone == []
    assert len(joined) == 2
    assert "1 A, 3 B:" in joined[0]
    assert "2 A, 4 B, 5 C:" in joined[1]



def testAFlareConsistentWithTwoFlaresIsAmbiguous():
    """ Check that a flare consistent with two flares of another station is not joined, and that the three of
        them are ambiguous.
    """

    joined, alone = eventRows([('A', 0.50), ('B', 0.46), ('B', 0.54)])

    assert joined == []
    assert len(alone) == 3
    assert all("ambiguous" in line for line in alone)
    assert any("1 A only, ambiguous: it could be flare(s) 2 B, 3 B" in line for line in alone)



def testFlaresAreOnlyJoinedWhenEachIsTheOnlyConsistentOneOfTheOther():
    """ Check that two flares are not joined when only one of them has the other as its only consistent flare:
        A1 is consistent only with B1, but B1 also with A2, which is also consistent with B2.
    """

    joined, alone = eventRows([('A', 0.50), ('A', 0.62), ('B', 0.56), ('B', 0.68)])

    assert joined == []
    assert len(alone) == 4



def testAGroupWithTwoFlaresOfOneStationIsLeftAlone():
    """ Check that a chain of joined flares which would hold two flares of one station is left alone as
        ambiguous: A1 with B1, B1 with C1 and C1 with A2, each the only consistent one of the other.
    """

    joined, alone = eventRows([('A', 0.50), ('B', 0.58), ('C', 0.66), ('A', 0.74)])

    assert joined == []
    assert len(alone) == 4
    assert all("ambiguous" in line for line in alone)



def testWithoutUncertaintiesTheFlaresAreNotJoined():
    """ Check that without the Monte Carlo runs the flares of different stations are not joined. """

    flares = [flarePoint('A', 0.50, 0.04), flarePoint('B', 0.52, 0.04)]
    for flare in flares:
        flare.update(t_std=None, ht_std=None, t_std_mc=None, ht_std_mc=None)

    report = flareReport(flares)

    assert "Without the Monte Carlo runs the flares of different stations are not joined." in report
    assert "Flares of the event" not in report



def rowPoint(station_id, t, row, ecsv_file=None):
    """ Make a flare point of a 25 fps camera on the given row of its file, as from flareHeights().

    Arguments:
        station_id: [str] Station ID.
        t: [float] Time from the reference time (s).
        row: [int] Position of the point among the points of the main fragment in its file.

    Keyword arguments:
        ecsv_file: [str] File of the point. None by default, for one file per station.

    Return:
        [dict] The flare point.
    """

    flare = flarePoint(station_id, t, 0.04)
    flare.update(ecsv_file=(station_id + '.ecsv') if (ecsv_file is None) else ecsv_file, row=row)

    return flare



def testTwoFlaresOnePointApartWhichOneFlareCouldBeAreNoted():
    """ Check that the report notes two flares of one station with a single point not flagged between them,
        when a flare of another station could be either, and not when two points are between them or they
        are in different files.
    """

    flares = [rowPoint('A', 0.50, 20), rowPoint('A', 0.54, 21), rowPoint('A', 0.58, 22), \
        rowPoint('B', 0.48, 10), rowPoint('B', 0.52, 11)]

    report = flareReport(flares + [rowPoint('B', 0.60, 13)])
    assert "1 A only, ambiguous: it could be flare(s) 2 B, 3 B" in report
    assert "Flares 2 and 3 of B are one point apart: if they are one flare, flag the point between them" \
        in report

    assert "one point apart" not in flareReport(flares + [rowPoint('B', 0.60, 14)])
    assert "one point apart" not in flareReport(flares + [rowPoint('B', 0.60, 13, ecsv_file='B_2.ecsv')])



if __name__ == "__main__":

    testTheFlaresOfTwoAndThreeStationsAreJoined()
    testAFlareConsistentWithTwoFlaresIsAmbiguous()
    testFlaresAreOnlyJoinedWhenEachIsTheOnlyConsistentOneOfTheOther()
    testAGroupWithTwoFlaresOfOneStationIsLeftAlone()
    testWithoutUncertaintiesTheFlaresAreNotJoined()
    testTwoFlaresOnePointApartWhichOneFlareCouldBeAreNoted()

    print("All tests passed.")
