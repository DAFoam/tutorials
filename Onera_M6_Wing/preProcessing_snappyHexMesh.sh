#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
echo "Generating mesh.."

if [ -f "m6_triSurface.tar.gz" ]; then
  echo "Geometry already exists."
else
  echo "Downloading geometry m6_triSurface.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/m6_triSurface.tar.gz
fi
tar -xvf m6_triSurface.tar.gz
mv m6_triSurface constant/triSurface
echo "Running snappyHexMesh mesh.."
blockMesh &> logMeshGeneration.txt
surfaceFeatureExtract >> logMeshGeneration.txt
snappyHexMesh -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
