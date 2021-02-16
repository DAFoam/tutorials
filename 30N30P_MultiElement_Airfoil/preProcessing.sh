#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

if [ -f "constant/polyMesh" ]; then
  echo "Mesh already exists."
else
  echo "Downloading mesh polyMesh_30N30P_MultiElement_Airfoil.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/polyMesh_30N30P_MultiElement_Airfoil.tar.gz
fi
tar -xvf polyMesh_30N30P_MultiElement_Airfoil.tar.gz
mv polyMesh constant/

echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
