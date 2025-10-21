#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# generate mesh
echo "Generating mesh.."
blockMesh
renumberMesh -overwrite
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0_orig 0

# create boundary layer velocity
setBoundaryLayerPatch -U0 0.01 -blHeight 0.002 -patches "(inlet)"
