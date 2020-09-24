#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
echo "Generating mesh.."

blockMesh &> logMeshGeneration.txt
cp system/mirrorMeshDict_x system/mirrorMeshDict
mirrorMesh -overwrite >> logMeshGeneration.txt
cp system/mirrorMeshDict_y system/mirrorMeshDict
mirrorMesh -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
