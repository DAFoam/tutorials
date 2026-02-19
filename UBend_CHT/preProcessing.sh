#!/bin/bash

# check if openfoam environment is loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# generate mesh
echo "Generating mesh.."

if [ -f "CHT_ubend_mesh.zip" ]; then
  echo "UBend mesh already exists."
else
  echo "Downloading UBend Mesh"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/CHT_ubend_mesh.zip
fi

gunzip CHT_ubend_mesh.zip

cp -r fluidMesh aero/constant/polyMesh
cp -r solidMesh thermal/constant/polyMesh

echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cd aero
cp -r 0.orig 0

cd ../thermal
cp -r 0.orig 0
