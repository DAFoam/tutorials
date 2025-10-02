#!/usr/bin/env bash

# the script will exit if there is any error
set -e

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit 1
fi

# copy the 0 folder
cp -r 0_orig 0

# manually generate a huge boundary layer U profile
setBoundaryLayerPatch -U0 10 -blHeight 0.1 -patches "(inlet)"

# run CFD to generate ref data
mpirun -np 4 python runPrimal.py
reconstructPar 
rm -rf processor*

# extract the pressure field from the latest time
getFIData -refFieldName p -refFieldType scalar -time 9999
getFIData -refFieldName U -refFieldType vector -time 9999

# copy the data to the 0 folder and clean up
cp -rf */*Data.gz 0/
rm -rf {1..9}*
