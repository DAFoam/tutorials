#!/bin/bash
cd baseline && mpirun --oversubscribe -np 4 python runScript.py && cd ..
cd reference && mpirun --oversubscribe -np 4 python runScript.py && cd ..
cd trained && mpirun --oversubscribe -np 4 python runScript.py && cd ..


