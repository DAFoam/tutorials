# Propeller_Benchmark

This repo can be run in two modes: hover or forward

To run the hover mode, delete the originnal "0" folder and copy "0_hover" to "0"
To run the hover mode, delete the originnal "0" folder and copy "0_forward" to "0"

To change the incoming velocity for the forward mode, delete the originnal "0" folder, go to the 0_forward/U file and change the default U value (5 m/s in the x direction) from "internalField uniform (5 0 0);" to your desired value, and the copy "0_forward" to "0".

To submit a job, run this command: `sbatch myJob.sh`. The default runtime is 1 hour and the default run mode is runPrimal. You may need to change these.


