#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# simulation preparation
cd sa-pitch-pre/
mpirun -np 4 python runPrimalDyM.py
reconstructPar
rm -r processor*
cp 0.6/nuTilda.gz ../sa-baseline/0/

cd ../sst-pitch-pre/
mpirun -np 4 python runPrimalDyM.py
reconstructPar
rm -r processor*
cp 0.6/U.gz ../sst-reference/0/
cp 0.6/U.gz ../sa-baseline/0/
cp 0.6/p.gz ../sst-reference/0/
cp 0.6/p.gz ../sa-baseline/0/
cp 0.6/nut.gz ../sst-reference/0/
cp 0.6/nut.gz ../sa-baseline/0/
cp 0.6/k.gz ../sst-reference/0/
cp 0.6/omega.gz ../sst-reference/0/

cp -r constant/polyMesh/ ../sst-reference/constant/
cp -r constant/polyMesh/ ../sa-baseline/constant/
cp 0.6/polyMesh/points.gz ../sst-reference/constant/polyMesh/
cp 0.6/polyMesh/points.gz ../sa-baseline/constant/polyMesh/
cd ../

