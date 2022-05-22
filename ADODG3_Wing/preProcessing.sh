#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# generate mesh
# generate mesh
echo "Generating mesh.."

if [ -f "adodg3_surfaceMesh_fine.cgns.tar.gz" ]; then
  echo "Surface mesh adodg3_surfaceMesh_fine.cgns.tar.gz already exists."
else
  echo "Downloading surface mesh adodg3_surfaceMesh_fine.cgns.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/adodg3_surfaceMesh_fine.cgns.tar.gz
fi

tar -xvf adodg3_surfaceMesh_fine.cgns.tar.gz
# coarsen the surface mesh 
cgns_utils coarsen surfaceMesh_fine.cgns surfaceMesh.cgns
python genWingMesh.py > logMeshGeneration.txt
plot3dToFoam -noBlank volumeMesh.xyz >> logMeshGeneration.txt
autoPatch 45 -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
