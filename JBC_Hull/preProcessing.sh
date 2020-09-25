#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
if [ -f "JBC_triSurface.tar.gz" ]; then
  echo "Surface geometry JBC_triSurface.tar.gz already exists."
else
  echo "Downloading surface geometry JBC_triSurface.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/JBC_triSurface.tar.gz
fi
tar -xvf JBC_triSurface.tar.gz
mv triSurface constant

if [ -f "JBC_FFD.tar.gz" ]; then
  echo "JBC_FFD.tar.gz already exists."
else
  echo "Downloading JBC_FFD.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/JBC_FFD.tar.gz
fi
tar -xvf JBC_FFD.tar.gz

echo "Generating mesh.."
blockMesh > log.meshGeneration
snappyHexMesh -overwrite >> log.meshGeneration
renumberMesh -overwrite >> log.meshGeneration
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0

