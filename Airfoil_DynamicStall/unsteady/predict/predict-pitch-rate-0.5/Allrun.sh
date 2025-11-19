#!/bin/bash
cd baseline && mpirun -np 4 python runScript_baseline.py && cd ..
cd reference && mpirun -np 4 python runScript_reference.py && cd ..
cd fiml && mpirun -np 4 python runScript_fiml.py && cd ..


