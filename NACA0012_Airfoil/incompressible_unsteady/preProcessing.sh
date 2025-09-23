#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# ************* cruise ****************
# generate mesh
cd cruise
echo "Generating mesh.."
python genAirFoilMesh.py &> logMeshGeneration.txt
plot3dToFoam -noBlank volumeMesh.xyz >> logMeshGeneration.txt
autoPatch 30 -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0.orig 0


# ************* maxLift ****************
cd ../maxLift
# generate mesh
echo "Generating mesh.."
python genAirFoilMesh.py &> logMeshGeneration.txt
plot3dToFoam -noBlank volumeMesh.xyz >> logMeshGeneration.txt
autoPatch 30 -overwrite >> logMeshGeneration.txt
createPatch -overwrite >> logMeshGeneration.txt
renumberMesh -overwrite >> logMeshGeneration.txt
echo "Generating mesh.. Done!"

# copy initial and boundary condition files
cp -r 0_orig 0

# run simpleFoam
cp -r system/controlDict_simple system/controlDict
cp -r system/fvSchemes_simple system/fvSchemes
cp -r system/fvSolution_simple system/fvSolution
mpirun -np 4 python runPrimalSimple.py

# reconstruct the simpleFoam fields
reconstructPar
rm -rf processor*
rm -rf 0
mv 500 0
rm -rf 0/uniform 0/polyMesh

# run the pimpleFoam primal to get equilibrium initial fields
cp -r system/controlDict_pimple_long system/controlDict
cp -r system/fvSchemes_pimple system/fvSchemes
cp -r system/fvSolution_pimple system/fvSolution
decomposePar
mpirun -np 4 pimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
rm -rf 0
mv 1 0
rm -rf 0/uniform 0/polyMesh
cp -r system/controlDict_pimple system/controlDict

cd ../
