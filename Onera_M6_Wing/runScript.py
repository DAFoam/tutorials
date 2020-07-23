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
parser.add_argument("--opt", help="optimizer to use", type=str, default="slsqp")
# which task to run. Options are: opt (default), run, testSensShape, or solveCL
parser.add_argument("--task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

# global parameters
U0 = 285.0
p0 = 101325.0
nuTilda0 = 4.5e-5
T0 = 300.0
CL_target = 0.270
alpha0 = 3.0
A0 = 0.7575
rho0 = 1.0  # density for normalizing CD and CL

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "turbulenceModel": "SpalartAllmaras",
    "flowCondition": "Compressible",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "UIn": {"variable": "U", "patch": "inout", "value": [U0, 0.0, 0.0]},
        "pIn": {"variable": "p", "patch": "inout", "value": [p0]},
        "TIn": {"variable": "T", "patch": "inout", "value": [T0]},
        "nuTildaIn": {"variable": "nuTilda", "patch": "inout", "value": [nuTilda0], "useWallFunction": True},
    },
    # variable bounds for compressible flow conditions
    "primalVarBounds": {
        "UUpperBound": 1000.0,
        "ULowerBound": -1000.0,
        "pUpperBound": 500000.0,
        "pLowerBound": 20000.0,
        "eUpperBound": 500000.0,
        "eLowerBound": 100000.0,
        "rhoUpperBound": 5.0,
        "rhoLowerBound": 0.2,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "alpha",
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "alphaName": "alpha",
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {"pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # transonic preconditioner to speed up the adjoint convergence
    "transonicPCOption": 1,
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0, "T": T0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "designVar": {},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "openfoam",
    "userotations": False,
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
        "Print file": "opt_SNOPT_print.out",
        "Summary file": "opt_SNOPT_summary.out",
    }
elif args.opt == "slsqp":
    optOptions = {
        "ACC": 1.0e-7,
        "MAXIT": 50,
        "IFILE": "opt_SLSQP.out",
    }
elif args.opt == "ipopt":
    optOptions = {
        "tol": 1.0e-7,
        "max_iter": 50,
        "output_file": "opt_IPOPT.out",
    }
else:
    print("opt arg not valid!")
    exit(0)


# =============================================================================
# Design variable setup
# =============================================================================
DVGeo = DVGeometry("./FFD/wingFFD.xyz")

# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")

# twist function, we keep the root twist constant so the first
# element in the twist design variable is the twist at the 2nd
# spanwise location
def twist(val, geo):
    for i in range(1, nTwists):
        geo.rot_z["bodyAxis"].coef[i] = val[i - 1]


# angle of attack
def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"UIn": {"variable": "U", "patch": "inout", "value": inletU}})
    DASolver.updateDAOption()


# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[:, :, :].flatten()
PS = geo_utils.PointSelect("list", indexList)
# shape
DVGeo.addGeoDVLocal("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
daOptions["designVar"]["shapey"] = {"designVarType": "FFD"}
# twist
DVGeo.addGeoDVGlobal("twist", np.zeros(nTwists - 1), twist, lower=-10.0, upper=10.0, scale=1.0)
daOptions["designVar"]["twist"] = {"designVarType": "FFD"}
# alpha
DVGeo.addGeoDVGlobal("alpha", [alpha0], alpha, lower=0.0, upper=10.0, scale=1.0)
daOptions["designVar"]["alpha"] = {"designVarType": "AOA", "patch": "inout", "flowAxis": "x", "normalAxis": "y"}

# =============================================================================
# DAFoam initialization
# =============================================================================
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=gcomm)
DASolver.addFamilyGroup(DASolver.getOption("designSurfaceFamily"), DASolver.getOption("designSurfaces"))
DASolver.printFamilyList()
DASolver.setMesh(mesh)
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# =============================================================================
# Constraint setup
# =============================================================================
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
DVCon.setSurface(DASolver.getTriangulatedMeshSurface(groupName=DASolver.getOption("designSurfaceFamily")))

# NOTE: the LE and TE lists are not parallel lines anymore, these two lists define lines that
# are close to the leading and trailing edges while being completely within the wing surface
leList = [[0.01, 0.0, 1e-3], [0.7, 0.0, 1.19]]
teList = [[0.79, 0.0, 1e-3], [1.135, 0.0, 1.19]]

# volume constraint
DVCon.addVolumeConstraint(leList, teList, nSpan=10, nChord=10, lower=1.0, upper=3, scaled=True)

# thickness constraint
DVCon.addThicknessConstraints2D(leList, teList, nSpan=10, nChord=10, lower=0.8, upper=3.0, scaled=True)

# Le/Te constraints
DVCon.addLeTeConstraints(0, "iLow")
DVCon.addLeTeConstraints(0, "iHigh")

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
    optProb.addCon("CL", lower=CL_target, upper=CL_target, scale=1)

    if gcomm.rank == 0:
        print(optProb)

    DASolver.runColoring()

    opt = OPT(args.opt, options=optOptions)
    histFile = "./%s_hist.hst" % args.opt
    sol = opt(optProb, sens=optFuncs.calcObjFuncSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

elif args.task == "run":

    optFuncs.run()

elif args.task == "solveCL":

    optFuncs.solveCL(CL_target, "alpha", "CL")

elif args.task == "testSensShape":

    optFuncs.testSensShape()

else:
    print("task arg not found!")
    exit(0)
