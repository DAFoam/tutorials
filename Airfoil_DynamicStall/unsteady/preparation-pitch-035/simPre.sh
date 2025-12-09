#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# simulation preparation
cp ../preparation-pitch-05/sa-pre/50/nuTilda.gz sa-pitch-pre/0/

cp ../preparation-pitch-05/sst-pre/50/U.gz sst-pitch-pre/0/
cp ../preparation-pitch-05/sst-pre/50/U.gz sa-pitch-pre/0/
cp ../preparation-pitch-05/sst-pre/50/p.gz sst-pitch-pre/0/
cp ../preparation-pitch-05/sst-pre/50/p.gz sa-pitch-pre/0/
cp ../preparation-pitch-05/sst-pre/50/nut.gz sst-pitch-pre/0/
cp ../preparation-pitch-05/sst-pre/50/nut.gz sa-pitch-pre/0/
cp ../preparation-pitch-05/sst-pre/50/k.gz sst-pitch-pre/0/
cp ../preparation-pitch-05/sst-pre/50/omega.gz sst-pitch-pre/0/

cd ../

