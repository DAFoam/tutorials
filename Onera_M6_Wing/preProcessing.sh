#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
echo "Generating mesh.."

if [ -f "m6_surfaceMesh_fine.cgns.tar.gz" ]; then
  echo "Surface mesh m6_surfaceMesh_fine.cgns.tar.gz already exists."
else
  echo "Downloading surface mesh m6_surfaceMesh_fine.cgns.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/m6_surfaceMesh_fine.cgns.tar.gz
fi
tar -xvf m6_surfaceMesh_fine.cgns.tar.gz
# coarsen the surface mesh three times
cgns_utils coarsen m6_surfaceMesh_fine.cgns surfaceMesh.cgns
cgns_utils coarsen surfaceMesh.cgns
cgns_utils coarsen surfaceMesh.cgns
python genWingMesh.py &> logMeshGeneration.txt
plot3dToFoam -noBlank volumeMesh.xyz >> logMeshGeneration.txt
autoPatch 60 -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
