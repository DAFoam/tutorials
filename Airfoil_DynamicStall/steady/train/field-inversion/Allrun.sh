#!/bin/bash
cd aoa-18/
mpirun -np 4 python runScript.py
reconstructPar
mv 10000 0
cd ../
mpirun -np 4 python runScript_FI-aoa-10-16.py -index=0 2>&1 | tee logOpt1.txt
mpirun -np 4 python runScript_FI-aoa-10-16.py -index=1 2>&1 | tee logOpt2.txt
mpirun -np 4 python runScript_FI-aoa-10-16.py -index=2 2>&1 | tee logOpt3.txt
mpirun -np 4 python runScript_FI-aoa-10-16.py -index=3 2>&1 | tee logOpt4.txt
mpirun -np 4 python runScript_FI-aoa-18.py -index=4 2>&1 | tee logOpt5.txt


