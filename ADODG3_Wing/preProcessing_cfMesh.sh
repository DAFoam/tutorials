#!/bin/bash

### Install CFMESH at https://develop.openfoam.com/Community/integration-cfmesh

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
echo "Generating mesh.."

if [ -f "adodg3_triSurface.tar.gz" ]; then
  echo "Geometry already exists."
else
  echo "Downloading geometry adodg3_triSurface.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/adodg3_triSurface.tar.gz
fi
tar -xvf adodg3_triSurface.tar.gz
mv triSurface constant/triSurface
echo "Running cfMesh mesh.."
cat constant/triSurface/wing_* > constant/triSurface/geom.stl

surfaceGenerateBoundingBox constant/triSurface/geom.stl constant/triSurface/domain.stl 20 10 10 10 0 10 &> logMeshGeneration.txt
cartesianMesh >> logMeshGeneration.txt
#createPatch -overwrite >> logMeshGeneration.txt
#renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
