#!/bin/bash
mpirun -np 12 python runScript_field-inversion.py
reconstructPar
cp -r 0.* ../machine-learning/pitch-05/



