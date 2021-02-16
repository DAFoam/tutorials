#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

if [ -f "triSurface_30N30P_MultiElement_Airfoil.tar.gz" ]; then
  echo "Geometry already exists."
else
  echo "Downloading geometry triSurface_30N30P_MultiElement_Airfoil.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/triSurface_30N30P_MultiElement_Airfoil.tar.gz
fi
tar -xvf triSurface_30N30P_MultiElement_Airfoil.tar.gz
mv triSurface constant/triSurface

# generate mesh
echo "Generating mesh.."
blockMesh &> logMeshGeneration.txt
surfaceFeatureExtract >> logMeshGeneration.txt
snappyHexMesh -overwrite >> logMeshGeneration.txt
extrudeMesh >> logMeshGeneration.txt
transformPoints -scale '(1 1 0.1)' >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
