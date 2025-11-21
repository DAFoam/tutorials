#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# generate mesh
cd mesh/
# generate mesh for aoa=10
./genMesh.sh
cp -r constant/polyMesh/ ../train/field-inversion/aoa-10/constant/

# generate mesh for aoa=12
sed -i "/transformPoints -rotate-angle '((0 0 1) -10.0)' >> logMeshGeneration.txt/c\transformPoints -rotate-angle '((0 0 1) -12.0)' >> logMeshGeneration.txt" genMesh.sh
./genMesh.sh
cp -r constant/polyMesh/ ../train/field-inversion/aoa-12/constant/

# generate mesh for aoa=14
sed -i "/transformPoints -rotate-angle '((0 0 1) -12.0)' >> logMeshGeneration.txt/c\transformPoints -rotate-angle '((0 0 1) -14.0)' >> logMeshGeneration.txt" genMesh.sh
./genMesh.sh
cp -r constant/polyMesh/ ../train/field-inversion/aoa-14/constant/

# generate mesh for aoa=16
sed -i "/transformPoints -rotate-angle '((0 0 1) -14.0)' >> logMeshGeneration.txt/c\transformPoints -rotate-angle '((0 0 1) -16.0)' >> logMeshGeneration.txt" genMesh.sh
./genMesh.sh
cp -r constant/polyMesh/ ../train/field-inversion/aoa-16/constant/

# generate mesh for aoa=18
sed -i "/transformPoints -rotate-angle '((0 0 1) -16.0)' >> logMeshGeneration.txt/c\transformPoints -rotate-angle '((0 0 1) -18.0)' >> logMeshGeneration.txt" genMesh.sh
./genMesh.sh
cp -r constant/polyMesh/ ../train/field-inversion/aoa-18/constant/

# set aoa=10 for rerun
sed -i "/transformPoints -rotate-angle '((0 0 1) -18.0)' >> logMeshGeneration.txt/c\transformPoints -rotate-angle '((0 0 1) -10.0)' >> logMeshGeneration.txt" genMesh.sh

cp -r 0.orig/. ../train/field-inversion/aoa-10/0/
cp -r 0.orig/. ../train/field-inversion/aoa-12/0/
cp -r 0.orig/. ../train/field-inversion/aoa-14/0/
cp -r 0.orig/. ../train/field-inversion/aoa-16/0/
cp -r 0.orig/. ../train/field-inversion/aoa-18/0/

