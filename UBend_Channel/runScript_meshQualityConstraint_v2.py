#!/usr/bin/env python
"""
DAFoam run script for the U bend channel case
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

HFL0 = 241.3
HFL_weight = -0.5
CPL0 = 40.26
CPL_weight = 0.5

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleTFoam",
    "useAD": {"mode": "reverse"},
    "designSurfaces": ["ubend", "ubendup"],
    "primalMinResTol": 1e-8,
    "primalMinResTolDiff": 1e5,
    "adjJacobianOption": "JacobianFree",
    "objFunc": {
        "obj": {
            "part1": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["inlet"],
                "scale": 1.0 / CPL0 * CPL_weight,
                "addToAdjoint": True,
            },
            "part2": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["outlet"],
                "scale": -1.0 / CPL0 * CPL_weight,
                "addToAdjoint": True,
            },
            "part3": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["ubendup"],
                "scale": 1.0 / HFL0 * HFL_weight,
                "addToAdjoint": True,
            }
        },
        "skewness": {
            "part1": {
                "type": "meshQualityKS",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "coeffKS": 20.0,
                "metric": "faceSkewness",
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
        "nonOrtho": {
            "part1": {
                "type": "meshQualityKS",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "coeffKS": 1.0,
                "metric": "nonOrthoAngle",
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
    },
    "adjStateOrdering": "cell",
    "normalizeStates": {"U": 10.0, "p": 30.0, "nuTilda": 1e-3, "phi": 1.0, "T": 300.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    "adjPCLag": 100,
    # Design variable setup
    "designVar": {
        "shape": {"designVarType": "FFD"},
    },
    "writeDeformedFFDs": True,
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]],
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
DVGeo = DVGeometry("./FFD/UBendDuctFFDSym.xyz")
# Select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[7:16, :, :].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVSectionLocal("shape", secIndex="i", lower=-0.1, upper=0.1, axis=2, pointSelect=PS)

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

"""
if args.task == "opt":
    # Create a linear constraint so that the curvature at the symmetry plane is zero
    indSetA = []
    indSetB = []
    for i in range(7, 16, 1):
        indSetA.append(pts[i, 0, 0])
        indSetB.append(pts[i, 0, 1])
    DVCon.addLinearConstraintsShape(
        indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0, config="configyinner"
    )
"""
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
    optProb.addObj("obj", scale=1)
    # Add physical constraints
    optProb.addCon("skewness", lower=0., upper=4.0, scale=1)
    optProb.addCon("nonOrtho", lower=0, upper=70.0, scale=1)
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

    optFuncs.runForwardAD("shape", 0)

else:
    print("task arg not found!")
    exit(0)
