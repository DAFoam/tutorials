#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
echo "Generating mesh.."

if [ -f "CRM_surfMesh.cgns.tar.gz" ]; then
  echo "Surface mesh CRM_surfMesh.cgns.tar.gz already exists."
else
  echo "Downloading surface mesh CRM_surfMesh.cgns.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/CRM_surfMesh.cgns.tar.gz
fi
tar -xvf CRM_surfMesh.cgns.tar.gz
cgns_utils coarsen surfMesh.cgns
python genWingMesh.py &> logMeshGeneration.txt
plot3dToFoam -noBlank volumeMesh.xyz >> logMeshGeneration.txt
autoPatch 45 -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
