#!/bin/bash

. ./preProcessing.sh
mpirun -np 4 python runScript.py

