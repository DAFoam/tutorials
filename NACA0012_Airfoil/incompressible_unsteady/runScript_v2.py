#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed (unsteady)
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
from idwarp import USMesh
from pyoptsparse import Optimization, OPT
import numpy as np


# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--opt", help="optimizer to use", type=str, default="ipopt")
parser.add_argument("--task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

# Define the global parameters here
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
aoa0 = 15.0
CD_max = 0.24
A0 = 0.1

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DAPimpleFoam",
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": True,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 100,
        "PCMatUpdateInterval": 1,
        "objFuncTimeOperator": "average",
        "reduceIO": True,
    },
    "printIntervalUnsteady": 1,
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-5,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": True,
    },
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "designVar": {
        "shape": {"designVarType": "FFD"},
        "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"}
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane and their normals
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
DVGeo = DVGeometry("./FFD/wingFFD.xyz")
DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")


def aoa(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


DASolver = PYDAFOAM(options=daOptions, comm=gcomm)

# select points
iVol = 0
pts = DVGeo.getLocalIndex(iVol)
indexList = pts[:, :, :].flatten()
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addLocalDV("shape", lower=-1.0, upper=1.0, axis="y", scale=100.0, pointSelect=PS)
DASolver.addInternalDV("aoa", [aoa0], aoa, lower=-50, upper=50, scale=1.0)

# =============================================================================
# DAFoam initialization
# =============================================================================

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

leList = [[1e-4, 0.0, 1e-4], [1e-4, 0.0, 0.1 - 1e-4]]
teList = [[0.998 - 1e-4, 0.0, 1e-4], [0.998 - 1e-4, 0.0, 0.1 - 1e-4]]

# volume constraint
DVCon.addVolumeConstraint(leList, teList, nSpan=2, nChord=10, lower=1.0, upper=3, scaled=True)

# thickness constraint
DVCon.addThicknessConstraints2D(leList, teList, nSpan=2, nChord=10, lower=0.8, upper=3.0, scaled=True)

# Create linear constraints to link the shape change between k=0 and k=1
nFFDs_x = pts.shape[0]
indSetA = []
indSetB = []
for i in range(nFFDs_x):
    for j in [0, 1]:
        indSetA.append(pts[i, j, 1])
        indSetB.append(pts[i, j, 0])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0)

# Create a linear constraint to fix the leading and trailing point.
indSetA = []
indSetB = []
for i in [0, nFFDs_x - 1]:
    for k in [0]:  # do not constrain k=1 because it is linked in the above symmetry constraint
        indSetA.append(pts[i, 0, k])
        indSetB.append(pts[i, 1, k])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=1.0, lower=0.0, upper=0.0)

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
    DASolver.addVariablesPyOpt(optProb)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    optProb.addObj("CL", scale=-1)
    optProb.addCon("CD", upper=CD_max, scale=10)

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
