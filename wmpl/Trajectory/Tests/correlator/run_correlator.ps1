# simple script to run the correlator

# This script expects you to have alredy activated the WMPL environment - 
# we can't do it for you as we don't know whether you're using Conda or virtualenv

$cwd=get-location
$here=$PSScriptRoot
set-location ${here}/../../../..

$env:PYTHONPATH="${here}/../../../../"

if (test-path $here/trajectories.db) {remove-item $here/*.db}
if (test-path $here/trajectories) {remove-item $here/trajectories -Recurse -Force}
if (test-path $here/phase1) {remove-item $here/phase1 -Recurse -Force}
if (test-path $here/candidates) {remove-item $here/candidates -Recurse -Force}

rename-item $here/UK00A4 unused_UK00A4
python -m wmpl.Trajectory.CorrelateRMS $here --mcmode 4 --cpucores 4 --addlogsuffix
python -m wmpl.Trajectory.CorrelateRMS $here --mcmode 1 --cpucores 4 --addlogsuffix
python -m wmpl.Trajectory.CorrelateRMS $here --mcmode 2 --cpucores 4 --addlogsuffix
rename-item $here/unused_UK00A4 UK00A4
python -m wmpl.Trajectory.CorrelateRMS $here --cpucores 4 --addlogsuffix

set-location $cwd

Write-Output "note: solution at 20260712_225600.836_UK is borderline-solvable and will occasionally fail"
write-output "list of differences"
write-output "--"
(Get-ChildItem $here\trajectories\2026\202607\2026071?\*.pickle -recurse ).name | compare-object (get-content $here\expected_results.txt)
write-output "--"