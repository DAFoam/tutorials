#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

if [ -f "constant/polyMesh" ]; then
  echo "Mesh already exists."
else
  echo "Downloading mesh polyMesh_UAV_Propeller.tar"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/polyMesh_UAV_Propeller.tar
fi
tar -xvf polyMesh_UAV_Propeller.tar
mv polyMesh constant/

if [ -f "Structure.bdf" ]; then
  echo "Mesh already exists."
else
  echo "Downloading mesh StructMesh.bdf.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/StructMesh.bdf.tar.gz
fi


echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
