The field inversion workflow proceeds as follows:

After all five field inversion cases (each corresponding to a distinct angle of attack) have converged, the solution fields for 
the final optimization iteration of each case must be reconstrcuted. 
For instance, if the final converged iteration for the aoa-10 case occurs at iteration 0.0011, and the aoa-12 case converges 
at iteration 0.0007, then the corresponding directories are aoa-10/0.0011 and aoa-12/0.0007. 
They contain the inversion results to be used for subsequent machine-learning training. These final-iteration solutions 
should be copied into the machine-learning data directory, each case should be placed in its designated subfolder,
Copy aoa-10/0.0011/. to ../machine-learning/aoa-10/ and copy aoa-12/0.0007/. to ../machine-learning/aoa-12/, and repeat this procedure for the remaining angles of attack.
