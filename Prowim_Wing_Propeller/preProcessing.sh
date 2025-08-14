#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing
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
