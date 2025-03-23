#!/bin/bash
cd baseline && mpirun -np 2 python runScript.py && cd ..
cd reference && mpirun -np 2 python runScript.py && cd ..
cd trained && mpirun -np 2 python runScript.py && cd ..


