#!/usr/bin/env python
"""
DAFoam run script for the Rotor 37 case
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
import numpy as np

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


TPR_target = 1.6
MFR_target = 0.7

# Set the parameters for optimization
daOptions = {
    "solverName": "DATurboFoam",
    "useAD": {"mode": "reverse"},
    "designSurfaces": ["blade"],
    "primalMinResTol": 1e-8,
    "primalMinResTolDiff": 1e4,
    "hasIterativeBC": True,
    "objFunc": {
        "TPR": {
            "part1": {
                "type": "totalPressureRatio",
                "source": "patchToFace",
                "patches": ["inlet", "outlet"],
                "inletPatches": ["inlet"],
                "outletPatches": ["outlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
        "TTR": {
            "part1": {
                "type": "totalTemperatureRatio",
                "source": "patchToFace",
                "patches": ["inlet", "outlet"],
                "inletPatches": ["inlet"],
                "outletPatches": ["outlet"],
                "scale": 1.0,
                "addToAdjoint": False,
            }
        },
        "MFR": {
            "part1": {
                "type": "massFlowRate",
                "source": "patchToFace",
                "patches": ["inlet"],
                "scale": -1.0,
                "addToAdjoint": True,
            }
        },
        "CMZ": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["blade"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.0, 0.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "normalizeStates": {"U": 100.0, "p": 100000.0, "nuTilda": 1e-3, "phi": 1.0, "T": 300.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-5, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "gmresMaxIters": 2000, "gmresRestart": 2000},
    "checkMeshThreshold": {"maxAspectRatio": 2000.0, "maxNonOrth": 78.0, "maxSkewness": 5.0},
    "transonicPCOption": 1,
    "adjPCLag": 1,
    # Design variable setup
    "designVar": {"shapey": {"designVarType": "FFD"}, "shapez": {"designVarType": "FFD"}},
    "decomposeParDict": {"preservePatches": ["per1", "per2"]},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
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
        "max_iter": 50,
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
FFDFile = "./FFD/localFFD.xyz"
DVGeo = DVGeometry(FFDFile)
# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[1:4, :, :].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addLocalDV("shapey", lower=-0.002, upper=0.002, axis="y", scale=1000, pointSelect=PS)
DVGeo.addLocalDV("shapez", lower=-0.002, upper=0.002, axis="z", scale=1000, pointSelect=PS)


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
    optProb.addObj("CMZ", scale=1)
    # Add physical constraints
    optProb.addCon("TPR", lower=TPR_target, upper=TPR_target, scale=1)
    optProb.addCon("MFR", lower=MFR_target, upper=MFR_target, scale=1)

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
