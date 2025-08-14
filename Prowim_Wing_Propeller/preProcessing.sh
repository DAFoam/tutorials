#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing
if [ -f "triSurface.tar.gz" ]; then
  echo "Surface geometry triSurface.tar.gz already exists."
else
  echo "Downloading surface geometry triSurface.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/Prowim_Wing_triSurface.tar.gz
fi
tar -xvf triSurface.tar.gz
mv triSurface constant/

# generate mesh
echo "Generating mesh.."
blockMesh &> logMeshGeneration.txt
surfaceFeatureExtract >> logMeshGeneration.txt
snappyHexMesh -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
