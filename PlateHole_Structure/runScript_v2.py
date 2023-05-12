#!/usr/bin/env python
"""
DAFoam run script for the Onera M4 wing at subsonic speed
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

vms0 = 2.9e4

# Set the parameters for optimization
daOptions = {
    "maxTractionBCIters": 20,
    "solverName": "DASolidDisplacementFoam",
    "designSurfaces": ["hole", "wallx", "wally"],
    "primalMinResTol": 1e-10,
    "primalMinResTolDiff": 1e10,
    "objFunc": {
        "VMS": {
            "part1": {
                "type": "vonMisesStressKS",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "scale": 1.0,
                "coeffKS": 2.0e-3,
                "addToAdjoint": True,
            }
        },
        "M": {
            "part1": {
                "type": "mass",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
    "normalizeStates": {"D": 1.0e-7},
    "adjPartDerivFDStep": {"State": 1e-5, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "adjPCLag": 20,
    # Design variable setup
    "designVar": {"shapey": {"designVarType": "FFD"}, "shapex": {"designVarType": "FFD"}},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
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
# DVGeo
FFDFile = "./FFD/plateFFD.xyz"
DVGeo = DVGeometry(FFDFile)
# select points
pts = DVGeo.getLocalIndex(0)
indexList1 = pts[2:5, 2, :].flatten()
indexList2 = pts[2:5, 4, :].flatten()
indexList = np.concatenate([indexList1, indexList2])
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addLocalDV("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS, config="configy")
indexList1 = pts[2, 2:5, :].flatten()
indexList2 = pts[4, 2:5, :].flatten()
indexList = np.concatenate([indexList1, indexList2])
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addLocalDV("shapex", lower=-1.0, upper=1.0, axis="x", scale=1.0, pointSelect=PS, config="configx")

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

leList = [[-0.5 + 1e-3, 0.0, 0.0 + 1e-3], [-0.5 + 1e-3, 0.0, 0.1 - 1e-3]]
teList = [[0.5 - 1e-3, 0.0, 0.0 + 1e-3], [0.5 - 1e-3, 0.0, 0.1 - 1e-3]]

# volume constraint
# DVCon.addVolumeConstraint(leList, teList, nSpan=2, nChord=10, lower=1.0, upper=3, scaled=True)

# thickness constraint
# DVCon.addThicknessConstraints2D(leList, teList, nSpan=2, nChord=10, lower=0.8, upper=3.0, scaled=True)

# Create linear constraints to link the shape change between k=0 and k=1
indSetA = []
indSetB = []
for i in [2, 4]:
    for j in range(2, 5):
        indSetA.append(pts[i, j, 0])
        indSetB.append(pts[i, j, 1])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0, config="configx")

indSetA = []
indSetB = []
for i in range(2, 5):
    for j in [2, 4]:
        indSetA.append(pts[i, j, 0])
        indSetB.append(pts[i, j, 1])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0, config="configy")

# Create linear constraints to link the shape change between j=2 and j=4
indSetA = []
indSetB = []
for i in range(2, 5):
    indSetA.append(pts[i, 2, 0])
    indSetB.append(pts[i, 4, 0])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=1.0, lower=0.0, upper=0.0, config="configy")

# Create linear constraints to link the shape change between i=2 and i=4
indSetA = []
indSetB = []
for j in range(2, 5):
    indSetA.append(pts[2, j, 0])
    indSetB.append(pts[4, j, 0])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=1.0, lower=0.0, upper=0.0, config="configx")

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

    optProb.addObj("M", scale=1.0)
    # Add physical constraints
    optProb.addCon("VMS", upper=vms0, scale=1.0)

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

elif args.task == "verifySens":

    optFuncs.verifySens()

else:
    print("task arg not found!")
    exit(0)
