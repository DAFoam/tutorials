#!/bin/bash

# Check if the OpenFOAM enviroments are loaded
if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit
fi

# =============================================================================
# Run Codes
# =============================================================================
#---------- Preprocessing ----------
cd aero
cp -r 0.orig 0

cd ../thermal
cp -r 0.orig 0