#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
if [ -f "DPW4_triSurface.tar.gz" ]; then
  echo "Surface geometry DPW4_triSurface.tar.gz already exists."
else
  echo "Downloading surface geometry DPW4_triSurface.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/DPW4_triSurface.tar.gz
fi
tar -xvf DPW4_triSurface.tar.gz
mv triSurface constant

echo "Generating mesh.."
blockMesh >> log.meshGeneration
surfaceFeatureExtract >> log.meshGeneration
snappyHexMesh -overwrite >> log.meshGeneration
renumberMesh -overwrite >> log.meshGeneration
createPatch -overwrite >> log.meshGeneration
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
