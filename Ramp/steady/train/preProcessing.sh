#!/usr/bin/env bash

# the script will exit if there is any error
set -e

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit 1
fi

# for the remove command
shopt -s extglob 

function preProcess() 
{
  # run full primal using SST as reference data
  cp constant/turbulenceProperties_SST constant/turbulenceProperties
  cp -r 0_orig 0
  mpirun --oversubscribe -np 4 python runPrimal.py
  reconstructPar 
  rm -rf processor*
  
  # rename U to UData
  getFIData -refFieldName U -refFieldType vector -time 9999
  getFIData -refFieldName p -refFieldType scalar -time 9999
  
  mv */*Data.gz 0/
  rm -rf {1..9}*
  
  # get ready for the KW run
  cp constant/turbulenceProperties_KW constant/turbulenceProperties
}

cd c1
preProcess
cd ../c2
preProcess
cd ..

