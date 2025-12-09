#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# simulation preparation
cd sa-pre/
mpirun -np 4 python runScript.py
reconstructPar
rm -r processor*
cp 50/nuTilda.gz ../sa-pitch-pre/0/
cp -r constant/polyMesh/ ../sa-pitch-pre/constant/
cp -r constant/polyMesh/ ../../preparation-pitch-035/sa-pitch-pre/constant/

cd ../sst-pre/
mpirun -np 4 python runScript.py
reconstructPar
rm -r processor*
sed -i "/type            fixedValue;/c\        type            movingWallVelocity;" 50/U
cp 50/U.gz ../sst-pitch-pre/0/
cp 50/U.gz ../sa-pitch-pre/0/
cp 50/p.gz ../sst-pitch-pre/0/
cp 50/p.gz ../sa-pitch-pre/0/
cp 50/nut.gz ../sst-pitch-pre/0/
cp 50/nut.gz ../sa-pitch-pre/0/
cp 50/k.gz ../sst-pitch-pre/0/
cp 50/omega.gz ../sst-pitch-pre/0/
cp -r constant/polyMesh/ ../sst-pitch-pre/constant/
cp -r constant/polyMesh/ ../../preparation-pitch-035/sst-pitch-pre/constant/

cd ../

