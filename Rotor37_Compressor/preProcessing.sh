#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
echo "Generating mesh.."

if [ -f "Rotor37_Mesh_Coarse.msh.tar.gz" ]; then
  echo "File Rotor37_Mesh_Coarse.msh.tar.gz already exists."
else
  echo "Downloading Rotor37_Mesh_Coarse.msh.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/Rotor37_Mesh_Coarse.msh.tar.gz
fi
tar -xvf Rotor37_Mesh_Coarse.msh.tar.gz &> logMeshGeneration.txt
fluent3DMeshToFoam Rotor37_Mesh_Coarse.msh >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
