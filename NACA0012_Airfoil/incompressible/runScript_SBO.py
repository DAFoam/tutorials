import numpy as np
from runScript import prob as NACA0012
from dafoam.pyDAFoam import surrogateOptimization

xlimits = np.array([[-0.01, 0.01]] * 8)
xlimits = np.array([[-0.05, 0.05]] * 10)        #~ DV bounds
xlimits[-1] = [0 , 3]                           #~ last DV is AOA which has different bounds, so here we adjust the bounds
xlimits[-2] = [10 , 10 + 1e-9]                  #~ restrict far field velocity to 10m/s
xlimits[-3] = [-0.01 , 0.01]                    #~ restrict FFD points near TE of airfoil
xlimits[-4] = [-0.01 , 0.01]

surrogateOptions = {
    "optType"    : "constrained",               #~ type of optimization problem (constrained or unconstrained)     
    "criterion"  : "EI",                        #~ criterion for next evaluation point determination -> EGO algorithm
    "iters"      : 20,                          #~ num iterations to optimize function
    "numDOE"     : 10,                          #~ number of sampling points
    "seed"       : 42,                          #~ seed value to reproduce results
    "dvNames"    : ["shape" , "patchV"],        #~ names of design variables
    "dvSizes"    : [8 , 2],                     #~ number of points for each design variable
    "dvBounds"   : xlimits,                     #~ design variable bounds            
    "objFunc"    : 'scenario1.aero_post.CD',    #~ objective function
    "cons"       : ['scenario1.aero_post.CL'],  #~ quantity to constrain
    "conWeights" : [10],                        #~ constraint weight
    "consEqs"    : ["x - 0.5"],                 #~ constraint equation(s)
}

surrogateOptimization(surrogateOptions , NACA0012)
