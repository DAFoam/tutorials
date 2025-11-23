#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# simulation preparation
cd sa-baseline/
mpirun -np 4 python runPrimalDyM.py 2>&1 | tee sa.txt
for t in $(seq 0.01 0.01 0.7); do
    reconstructPar -time $t
done
rm -r processor*
cp -r 0/. ../../train/field-inversion/0/
cp -r constant/polyMesh/ ../../train/field-inversion/constant/
cp -r 0/. ../../predict/predict-pitch-rate-0.5/fiml/0/
cp -r constant/polyMesh/ ../../predict/predict-pitch-rate-0.5/fiml/constant/

cd ../sst-reference/
mpirun -np 4 python runPrimalDyM.py 2>&1 | tee sst-ref.txt
for t in $(seq 0.01 0.01 0.7); do
    reconstructPar -time $t
done
rm -r processor*
cp sst-ref.txt ../../train/field-inversion/



