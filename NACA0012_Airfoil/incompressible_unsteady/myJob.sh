#!/bin/bash
#SBATCH --time=1:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 15 processor core(s)(CPU cores) per node 
#SBATCH --job-name="build_df"
#SBATCH --output="log-%j.txt" # job standard output file (%j replaced by job id)
#SBATCH --mail-user=your_email@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --constraint=intel

. /work/phe/DAFoam_Nova_Gcc/latest/loadDAFoam.sh
#. /work/phe/DAFoam_Nova_Gcc/3.1.2/loadDAFoam.sh
mpirun -np 18 python runScript.py

#module load singularity

#singularity exec --no-mount /scratch /work/phe/PingHe/dafoam_v312.sif /bin/bash -l -c '. /home/dafoamuser/dafoam/loadDAFoam.sh && mpirun -np 18 python runScript_v2.py'
