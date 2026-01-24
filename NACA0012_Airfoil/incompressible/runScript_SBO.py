
import numpy as np
from runScript import prob as NACA0012
from dafoam.pyDAFoam import surrogateOptimization


# define design variables (name & size)
dvNames = ["shape", "patchV"]
dvSizes = [8, 2]

# prescribe bounds on design variables
xlimits = np.array([[0.0, 0.05]] * 10)

# adjust bounds for AOA, fix velocity to 10m/s
xlimits[-1] = [1, 1.5]                          
xlimits[-2] = [10, 10 + 1e-9]                  

objFunc = 'scenario1.aero_post.CD'
cons = ['scenario1.aero_post.CL', 'geometry.thickcon', 'geometry.volcon', 'geometry.rcon']
conWeights = [1e10, 10, 10, 1e2]
consEqs = ["x - 0.5", "0.5 <= x <= 3", "1 <= x", "0.8 <= x"]

surrogateOptions = {
    "criterion"  : "EI",           # criterion for next evaluation point
    "iters"      : 10,             # num iterations to optimize function
    "numDOE"     : 15,             # number of sampling points
    "seed"       : 43,             # seed value to reproduce results
    "dvNames"    : dvNames,        # names of design variables
    "dvSizes"    : dvSizes,        # number of points for each design variable
    "dvBounds"   : xlimits,        # design variable bounds
    "objFunc"    : objFunc,        # objective function
    "cons"       : cons,           # quantity to constrain
    "conWeights" : conWeights,     # constraint weight
    "consEqs"    : consEqs,        # constraint equation(s)
    "surrogate"  : "MGP",          # which surrogate model to use (Kriging-based)
}

surrogateOptimization(surrogateOptions, NACA0012)  # pass surrogateOptions and OpenMDAO model to surrogateOptimization class

