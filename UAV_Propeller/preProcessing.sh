#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# get the geometry
if [ -f "mdolab_wing_surface_mesh.cgns.tar.gz" ]; then
  echo "Surface geometry mdolab_wing_surface_mesh.cgns.tar.gz already exists."
else
  echo "Downloading surface geometry mdolab_wing_surface_mesh.cgns.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/mdolab_wing_surface_mesh.cgns.tar.gz
fi
tar -xvf mdolab_wing_surface_mesh.cgns.tar.gz

# generate mesh
echo "Generating mesh.."

# coarsen the surface mesh three times
cgns_utils coarsen mdolab_wing_surface_mesh.cgns surfaceMesh.cgns
#cgns_utils coarsen surfaceMesh.cgns
python genWingMesh.py &> logMeshGeneration.txt
plot3dToFoam -noBlank volumeMesh.xyz >> logMeshGeneration.txt
autoPatch 60 -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
