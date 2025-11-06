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

# run simpleFoam to get init field
cp -r 0_orig 0
cp -r system/controlDict_steady system/controlDict
cp -r system/fvSchemes_steady system/fvSchemes
cp -r system/fvSolution_steady system/fvSolution

# run CFD to generate ref data
mpirun -np 2 python runScript.py -task=run_model -patchV=10.0,15.0 -shape=0.01,0.01,0.03,0.03,0.01,0.01,-0.01,0.0
reconstructPar 
rm -rf processor*
rm -rf 0/*
cp -r */*.gz 0/
rm -rf {1..9}*


# run unsteady CFD to generate ref data
cp -r 0_orig 0
cp -r system/controlDict_unsteady system/controlDict
cp -r system/fvSchemes_unsteady system/fvSchemes
cp -r system/fvSolution_unsteady system/fvSolution

# run CFD to generate ref data
mpirun -np 2 python runScript_unsteady.py -task=run_model -patchV=10.0,15.0 -shape=0.01,0.01,0.03,0.03,0.01,0.01,-0.01,0.0
reconstructPar 
rm -rf processor*

# extract the pressure field from the latest time
getFIData -refFieldName p -refFieldType scalar
getFIData -refFieldName U -refFieldType vector
