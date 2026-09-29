#!/bin/bash
# simple script to run the correlator

# this script assumes you already activated the WMPL environment - 
# we can't do it for you as we don't know whether you're using Conda or virtualenv - 
# and so you should run it with "bash ./run_correlator.sh"

here="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
pushd ${here}/

export PYTHONPATH=$here/../../../../

rm -Rf $here/*.db
rm -Rf $here/trajectories
rm -Rf $here/phase1
rm -Rf $here/candidates


mv $here/UK00A4 $here/unused_UK00A4
python -m wmpl.Trajectory.CorrelateRMS $here --mcmode 4 --cpucores 4 --addlogsuffix
python -m wmpl.Trajectory.CorrelateRMS $here --mcmode 1 --cpucores 4 --addlogsuffix
python -m wmpl.Trajectory.CorrelateRMS $here --mcmode 2 --cpucores 4 --addlogsuffix
mv $here/unused_UK00A4 $here/UK00A4
python -m wmpl.Trajectory.CorrelateRMS $here --cpucores 4 --addlogsuffix

echo "note: solution at 20260712_225600.836_UK is borderline-solvable and will occasionally fail"
echo "list of differences"
echo "--"
find $here/trajectories/ -name '*.pickle' -exec ls -1 {} \; | while read i 
do 
    echo $(basename $i) 
done | sort -n  | diff  ${here}/expected_results.txt -
echo "--"

popd