#!/usr/bin/env bash

# the script will exit if there is any error
set -e

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit 1
fi

# for the remove command
shopt -s extglob 

# spin up the flow for 0.1 s
# spin up SST
cp constant/turbulenceProperties_SST constant/turbulenceProperties
cp system/controlDict_spinup system/controlDict
rm -rf 0
cp -r 0_orig 0
mpirun -np 2 python runPrimal.py
reconstructPar 
rm -rf 0.1/polyMesh 0.1/uniform
cp 0_orig/betaFINuTildaData 0.1/
mv 0.1 0_SST
rm -rf processor*
# spin up SA
cp constant/turbulenceProperties_SA constant/turbulenceProperties
cp system/controlDict_spinup system/controlDict
rm -rf 0
cp -r 0_orig 0
mpirun -np 2 python runPrimal.py
reconstructPar 
rm -rf 0.1/polyMesh 0.1/uniform
cp 0_orig/betaFINuTildaData 0.1/
mv 0.1 0_SA
rm -rf processor*

# run full primal using SST as reference data
cp constant/turbulenceProperties_SST constant/turbulenceProperties
cp system/controlDict_full system/controlDict
rm -rf 0
cp -r 0_SST 0
mpirun -np 2 python runPrimal.py
reconstructPar 
rm -rf processor*

# rename U to UData
getFIData -refFieldName U -refFieldType vector
getFIData -refFieldName p -refFieldType scalar

rm -rf 0.*/!(*Data.gz) 1/!(*Data.gz) 1.*/!(*Data.gz) 
rm -rf 0
cp -r 0_SA 0

# decompose all times
decomposePar -time '0:'

rm -rf 0.* 1 1.*

# get ready for the SA run
cp constant/turbulenceProperties_SA constant/turbulenceProperties
