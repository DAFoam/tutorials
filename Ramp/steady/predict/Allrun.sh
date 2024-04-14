#!/bin/bash
cd baseline && mpirun -np 4 python runScript.py && cd ..
cd reference && mpirun -np 4 python runScript.py && cd ..
cd trained && mpirun -np 4 python runScript.py && cd ..


