@echo off
conda activate wmpl
cd /home/jkambul2/WesternMeteorPyLib
#start python -m wmpl.MetSim.ML.GenerateSimulations /home/jkambul2/files/fixed_erosion_dataset2 6000000 --noerosion --fixed 0 0 0 0 0 1 1 1 1 1
python -m wmpl.MetSim.ML.FitErosion /home/jkambul2/files/fixed_erosion_dataset2 /home/jkambul2/files/trained_models -mn fixederosion --grouping 256 50

python -m wmpl.MetSim.ML.GenerateSimulations /home/jkambul2/files/unfixed_erh 6000000 --fixed 0 0 0 0 0 0 1 1 1 1
python -m wmpl.MetSim.ML.FitErosion /home/jkambul2/files/fixed_erosion_dataset2 /home/jkambul2/files/trained_models -mn unfixed_erh --grouping 256 50

python -m wmpl.MetSim.ML.GenerateSimulations /home/jkambul2/files/unfixed_erco 6000000 --fixed 0 0 0 0 0 0 0 1 1 1
python -m wmpl.MetSim.ML.FitErosion /home/jkambul2/files/fixed_erosion_dataset2 /home/jkambul2/files/trained_models -mn unfixed_erco --grouping 256 50

python -m wmpl.MetSim.ML.GenerateSimulations /home/jkambul2/files/fixed_erosion_dataset2 6000000 --noerosion --fixed 0 0 0 0 1 0 1 1 1 1
python -m wmpl.MetSim.ML.FitErosion /home/jkambul2/files/fixed_erosion_dataset2 /home/jkambul2/files/trained_models -mn fixederosion --grouping 256 50
pause