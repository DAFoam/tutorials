#!/bin/bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# Generate Input Geometry
echo "Generating Input Geometry..."
(cd deformIGS && python generate_wing.py)


# Copy Necessary Files
echo "Copying Files..."
cp ./deformIGS/wing.igs .
cp ./deformIGS/OptRef_Example.json .

# Run Script to Deform the Geometry
echo "Deforming Geometry..."
python runScript.py --task=deformIGS

echo "Geometry Deformation... Done!"