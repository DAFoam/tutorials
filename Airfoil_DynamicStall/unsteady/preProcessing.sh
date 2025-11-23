#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# generate mesh for aoa=4
cd preparation-pitch-05/mesh/
./genMesh.sh
cp -r constant/polyMesh/ ../sa-pre/constant
cp -r constant/polyMesh/ ../sst-pre/constant
cd ../

# simulation preparation
./simPre.sh
./simPicthPre.sh
./simRun.sh

cd ../preparation-pitch-035/
# simulation preparation
./simPre.sh
./simPicthPre.sh
./simRun.sh
