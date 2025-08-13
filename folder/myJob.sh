#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename

#SBATCH --time=1:00:00		# walltime limit (HH:MM:SS)
#SBATCH --nodes=4		# number of nodes
#SBATCH --ntasks-per-node=36	# 36 processor core(s) per node 
#SBATCH --job-name="test"
#SBATCH --output="log-%j.txt"	# job standard output file (%j replaced by job id)
#SBATCH --mem=180G

# LOAD MODULES
source /work/phe/DAFoam_Nova_Gcc/latest/loadDAFoam.sh


# RUN CODES
mpirun -np 144 python runScriptAeroStruct.py -task=runPrimal -case=1
