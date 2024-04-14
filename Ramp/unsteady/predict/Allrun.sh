#!/bin/bash

cd baseline && ibrun -np 4 python runScript.py && cd ..
cd reference && ibrun -np 4 python runScript.py && cd ..
cd trained && ibrun -np 4 python runScript.py && cd ..


