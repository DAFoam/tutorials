#!/bin/bash

if [ -z "$1" ]; then
  echo "No argument supplied!"
  echo "Example: ./preProcessing_parallel.sh 12"
  echo "This will generate mesh using 12 CPU cores."
  exit
fi

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# pre-processing

# get the geometry
if [ -f "NREL6_triSurface.tar.gz" ]; then
  echo "Surface geometry NREL6_triSurface.tar.gz already exists."
else
  echo "Downloading surface geometry NREL6_triSurface.tar.gz"
  wget https://github.com/dafoam/files/releases/download/v1.0.0/NREL6_triSurface.tar.gz
fi
tar -xvf NREL6_triSurface.tar.gz
rm -rf constant/triSurface
mv triSurface constant

# generate mesh
echo "Generating mesh in parallel.."
sed -i "/numberOfSubdomains/c\numberOfSubdomains $1;" system/decomposeParDict
rm -rf 1 2 3
rm -rf constant/polyMesh/*
blockMesh &> logMeshGeneration.txt
decomposePar >> logMeshGeneration.txt
mpirun -np $1 snappyHexMesh -parallel >> logMeshGeneration.txt
mpirun -np $1 checkMesh -parallel >> logMeshGeneration.txt
reconstructParMesh -latestTime >> logMeshGeneration.txt
rm -rf constant/polyMesh/*
mv 3/polyMesh/* constant/polyMesh/
rm -rf 1 2 3
rm -rf processor*
renumberMesh -overwrite >> logMeshGeneration.txt
topoSet >> logMeshGeneration.txt

# copy initial and boundary condition files
cp -r 0.orig 0
