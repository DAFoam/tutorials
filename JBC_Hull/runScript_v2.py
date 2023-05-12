#!/usr/bin/env python
"""
DAFoam run script for the JBC case
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
from pygeo import *
from pyspline import *
from idwarp import *
from pyoptsparse import Optimization, OPT

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
# which optimizer to use. Options are: slsqp (default), snopt, or ipopt
parser.add_argument("--opt", help="optimizer to use", type=str, default="ipopt")
# which task to run. Options are: opt (default), run, testSensShape, or solveCL
parser.add_argument("--task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

U0 = 1.179
A0 = 12.2206
p0 = 0.0
nuTilda0 = 1.0e-4

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaces": ["hull"],
    "primalMinResTol": 1e-8,
    "useAD": {"mode": "reverse"},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["hull"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0 / 0.5 / U0 / U0 / A0,
                "addToAdjoint": True,
            },
        },
    },
    "normalizeStates": {"U": 1.0, "p": 1.0, "nuTilda": 1e-4, "phi": 1.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "adjPCLag": 1,
    # Design variable setup
    "designVar": {"shapey": {"designVarType": "FFD"}},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
}

# options for optimizers
if args.opt == "snopt":
    optOptions = {
        "Major feasibility tolerance": 1.0e-7,
        "Major optimality tolerance": 1.0e-7,
        "Minor feasibility tolerance": 1.0e-7,
        "Verify level": -1,
        "Function precision": 1.0e-7,
        "Major iterations limit": 50,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.opt == "ipopt":
    optOptions = {
        "tol": 1.0e-7,
        "constr_viol_tol": 1.0e-7,
        "max_iter": 40,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.opt == "slsqp":
    optOptions = {
        "ACC": 1.0e-7,
        "MAXIT": 50,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("opt arg not valid!")
    exit(0)



# =============================================================================
# Design variable setup
# =============================================================================
FFDFile = "./FFD/JBCFFD_32.xyz"
DVGeo = DVGeometry(FFDFile)
# Select points
iVol = 0
pts = DVGeo.getLocalIndex(iVol)
# shapez
indexList = []
indexList.extend(pts[8:12, 0, 0:4].flatten())
indexList.extend(pts[8:12, -1, 0:4].flatten())
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addLocalDV("shapey", lower=-0.5, upper=0.5, axis="y", scale=1.0, pointSelect=PS)


# =============================================================================
# DAFoam initialization
# =============================================================================
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=gcomm)
DASolver.printFamilyList()
DASolver.setMesh(mesh)
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# =============================================================================
# Constraint setup
# =============================================================================
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
DVCon.setSurface(DASolver.getTriangulatedMeshSurface(groupName=DASolver.designSurfacesGroup))

# Create reflection constraint
pts = DVGeo.getLocalIndex(0)
indSetA = []
indSetB = []
for i in range(8, 12, 1):
    for k in range(0, 4, 1):
        indSetA.append(pts[i, 0, k])
        indSetB.append(pts[i, -1, k])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=1.0, lower=0.0, upper=0.0)


# Create a volume constraint
# Volume constraints
leList = [
    [4.90000000, 0.00000000, -0.41149880],
    [4.90000000, 0.00000000, -0.40347270],
    [4.90000000, 0.00000000, -0.38803330],
    [4.90000000, 0.00000000, -0.36534750],
    [4.90000000, 0.00000000, -0.33601030],
    [4.90000000, 0.00000000, -0.31016020],
    [4.90000000, 0.00000000, -0.28327050],
    [4.90000000, 0.00000000, -0.26248810],
    [4.90000000, 0.00000000, -0.24076410],
    [4.90000000, 0.00000000, -0.20933480],
    [4.90000000, 0.00000000, -0.17458840],
    [4.90000000, 0.00000000, -0.14233480],
    [4.90000000, 0.00000000, -0.11692880],
    [4.90000000, 0.00000000, -0.09984235],
    [4.90000000, 0.00000000, -0.08874606],
    [4.90000000, 0.00000000, -0.07969946],
    [4.90000000, 0.00000000, -0.06954966],
    [4.90000000, 0.00000000, -0.05864429],
    [4.90000000, 0.00000000, -0.04829308],
    [4.90000000, 0.00000000, -0.03831457],
    [4.90000000, 0.00000000, -0.02430242],
    [4.90000000, 0.00000000, -0.00100000],
]
teList = [
    [6.70332700, 0.00000000, -0.41149880],
    [6.73692400, 0.00000000, -0.40347270],
    [6.76842800, 0.00000000, -0.38803330],
    [6.79426000, 0.00000000, -0.36534750],
    [6.81342600, 0.00000000, -0.33601030],
    [6.83648300, 0.00000000, -0.31016020],
    [6.85897100, 0.00000000, -0.28327050],
    [6.83593600, 0.00000000, -0.26248810],
    [6.80929800, 0.00000000, -0.24076410],
    [6.79395800, 0.00000000, -0.20933480],
    [6.79438900, 0.00000000, -0.17458840],
    [6.80874100, 0.00000000, -0.14233480],
    [6.83265000, 0.00000000, -0.11692880],
    [6.86250800, 0.00000000, -0.09984235],
    [6.89566400, 0.00000000, -0.08874606],
    [6.92987100, 0.00000000, -0.07969946],
    [6.96333200, 0.00000000, -0.06954966],
    [6.99621200, 0.00000000, -0.05864429],
    [7.02921500, 0.00000000, -0.04829308],
    [7.06253200, 0.00000000, -0.03831457],
    [7.09456600, 0.00000000, -0.02430242],
    [7.12000000, 0.00000000, -0.00100000],
]
DVCon.addVolumeConstraint(leList, teList, nSpan=25, nChord=50, lower=1.0, upper=1.0)


# Thickness constraint for lateral thickness
leList = [[5.01, 0.0000, -0.001], [5.01, 0.0000, -0.410]]
teList = [[6.2, 0.0000, -0.001], [6.2, 0.0000, -0.410]]
DVCon.addThicknessConstraints2D(leList, teList, nSpan=8, nChord=5, lower=1e-3, upper=1.1251, scaled=False)


# Thickness constraint for propeller shaft
leList = [[6.8, 0.0000, -0.302], [6.8, 0.0000, -0.265]]
teList = [[6.865, 0.0000, -0.302], [6.865, 0.0000, -0.265]]
DVCon.addThicknessConstraints2D(leList, teList, nSpan=5, nChord=5, lower=1.0, upper=10.0)

# Curvature constraints
DVCon.addCurvatureConstraint(
    "./FFD/hullCurv.xyz", curvatureType="KSmean", lower=0.0, upper=1.21, addToPyOpt=True, scaled=True
)

DVCon.writeTecplot("DVConstraints.dat")

# =============================================================================
# Initialize optFuncs for optimization
# =============================================================================
optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# =============================================================================
# Task
# =============================================================================
if args.task == "opt":

    optProb = Optimization("opt", objFun=optFuncs.calcObjFuncValues, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj("CD", scale=1)
    # Add physical constraints
    # optProb.addCon("CL", lower=CL_target, upper=CL_target, scale=1)

    if gcomm.rank == 0:
        print(optProb)

    DASolver.runColoring()

    opt = OPT(args.opt, options=optOptions)
    histFile = "./%s_hist.hst" % args.opt
    sol = opt(optProb, sens=optFuncs.calcObjFuncSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

elif args.task == "runPrimal":

    optFuncs.runPrimal()

elif args.task == "runAdjoint":

    optFuncs.runAdjoint()

elif args.task == "runForwardAD":

    optFuncs.runForwardAD("shapey", 0)

else:
    print("task arg not found!")
    exit(0)
