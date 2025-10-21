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
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0

# run CFD to generate ref data
mpirun -np 4 python runPrimal.py
reconstructPar 
rm -rf processor*

# extract the pressure field from the latest time
getFIData -refFieldName p -refFieldType scalar -time 9999
getFIData -refFieldName U -refFieldType vector -time 9999

# copy the data to the 0 folder and clean up
cp -rf */*Data.gz 0/
rm -rf {1..9}*
