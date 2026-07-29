#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# generate mesh
echo "Generating mesh.."
python genAirFoilMesh.py &> logMeshGeneration.txt
plot3dToFoam -noBlank volumeMesh.xyz >> logMeshGeneration.txt
autoPatch 30 -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
topoSet >> logMeshGeneration.txt
transformPoints -translate '(-0.25 0 0)' >> logMeshGeneration.txt
transformPoints -rotate-angle '((0 0 1) -4.0)' >> logMeshGeneration.txt
transformPoints -translate '(0.25 0 0)' >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
#cp -r 0.orig 0
