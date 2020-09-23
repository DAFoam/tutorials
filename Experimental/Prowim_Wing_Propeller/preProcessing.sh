#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

if [ -f "prowim_wing_surface_mesh.cgns.tar.gz" ]; then
  echo "Surface mesh prowim_wing_surface_mesh.cgns.tar.gz already exists."
else
  echo "Downloading surface mesh prowim_wing_surface_mesh.cgns.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/prowim_wing_surface_mesh.cgns.tar.gz
fi
rm prowim_wing_surface_mesh.cgns
tar -xvf prowim_wing_surface_mesh.cgns.tar.gz
# coarsen the surface mesh three times
cgns_utils coarsen prowim_wing_surface_mesh.cgns surfaceMesh.cgns
cgns_utils coarsen surfaceMesh.cgns
# generate mesh
echo "Generating mesh.."
python genWingMesh.py &> logMeshGeneration.txt
plot3dToFoam -noBlank volumeMesh.xyz >> logMeshGeneration.txt
autoPatch 60 -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0
