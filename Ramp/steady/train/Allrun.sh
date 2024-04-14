#!/bin/bash
. ./preProcessing.sh
ibrun -np 4 python runScript_FIML.py -optimizer=SNOPT

