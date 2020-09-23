#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# get the geometry
if [ -f "NREL6_triSurface.tar.gz" ]; then
  echo "Surface geometry NREL6_triSurface.tar.gz already exists."
else
  echo "Downloading surface geometry NREL6_triSurface.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/NREL6_triSurface.tar.gz
fi
tar -xvf NREL6_triSurface.tar.gz
rm -rf constant/triSurface
mv triSurface constant

# generate mesh
echo "Generating mesh.."
blockMesh &> logMeshGeneration.txt
snappyHexMesh -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
topoSet >> logMeshGeneration.txt

# copy initial and boundary condition files
cp -r 0.orig 0


