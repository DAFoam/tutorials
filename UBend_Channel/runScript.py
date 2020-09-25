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
parser.add_argument("--opt", help="optimizer to use", type=str, default="slsqp")
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
    "designSurfaces": ["ubend", "ubendup"],
    "primalMinResTol": 1e-8,
    "primalMinResTolDiff": 1e5,
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
    },
    "normalizeStates": {"U": 10.0, "p": 30.0, "nuTilda": 1e-3, "phi": 1.0, "T": 300.0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # Design variable setup
    "designVar": {
        "shapez": {"designVarType": "FFD"},
        "shapeyouter": {"designVarType": "FFD"},
        "shapeyinner": {"designVarType": "FFD"},
        "shapexinner": {"designVarType": "FFD"},
        "shapexouter1": {"designVarType": "FFD"},
        "shapexouter2": {"designVarType": "FFD"},
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "openfoam",
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
        "Print file": "opt_SNOPT_print.out",
        "Summary file": "opt_SNOPT_summary.out",
    }
elif args.opt == "slsqp":
    optOptions = {
        "ACC": 1.0e-7,
        "MAXIT": 50,
        "IFILE": "opt_SLSQP.out",
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
# shapez
indexList = []
indexList.extend(pts[7:16, :, -1].flatten())
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapez", lower=-0.04, upper=0.0077, axis="z", scale=1.0, pointSelect=PS, config="configz")
# shapeyouter
indexList = []
indexList.extend(pts[7:16, -1, :].flatten())
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapeyouter", lower=-0.02, upper=0.02, axis="y", scale=1.0, pointSelect=PS, config="configyouter")
# shapeyinner
indexList = []
indexList.extend(pts[7:16, 0, :].flatten())
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapeyinner", lower=-0.04, upper=0.04, axis="y", scale=1.0, pointSelect=PS, config="configyinner")
# shapexinner
indexList = []
indexList.extend(pts[7:16, 0, :].flatten())
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapexinner", lower=-0.04, upper=0.04, axis="x", scale=1.0, pointSelect=PS, config="configxinner")

# shapexouter1
indexList = []
indexList.extend(pts[9, -1, :].flatten())
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal(
    "shapexouter1", lower=-0.05, upper=0.05, axis="x", scale=1.0, pointSelect=PS, config="configxouter1"
)

# shapexouter2
indexList = []
indexList.extend(pts[10, -1, :].flatten())
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addGeoDVLocal("shapexouter2", lower=-0.05, upper=0.0, axis="x", scale=1.0, pointSelect=PS, config="configxouter2")


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

    indSetA = []
    indSetB = []
    for i in range(7, 16, 1):
        indSetA.append(pts[i, -1, 0])
        indSetB.append(pts[i, -1, 1])
    DVCon.addLinearConstraintsShape(
        indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0, config="configyouter"
    )

    indSetA = []
    indSetB = []
    for i in range(7, 16, 1):
        indSetA.append(pts[i, 0, 0])
        indSetB.append(pts[i, 0, 1])
    DVCon.addLinearConstraintsShape(
        indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0, config="configxinner"
    )

    indSetA = []
    indSetB = []
    for i in [9]:
        indSetA.append(pts[i, -1, 0])
        indSetB.append(pts[i, -1, 1])
    DVCon.addLinearConstraintsShape(
        indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0, config="configxouter1"
    )

    indSetA = []
    indSetB = []
    for i in [10]:
        indSetA.append(pts[i, -1, 0])
        indSetB.append(pts[i, -1, 1])
    DVCon.addLinearConstraintsShape(
        indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0, config="configxouter2"
    )
    # linear constraint to make sure the inner bend does not intersect
    minInnerBend = 0.005
    indSetA = []
    indSetB = []
    for k in range(3):
        indSetA.append(pts[8, 0, k])
        indSetB.append(pts[14, 0, k])
    DVCon.addLinearConstraintsShape(
        indSetA,
        indSetB,
        factorA=1.0,
        factorB=-1.0,
        lower=-0.03300 + minInnerBend,
        upper=0.03300 - minInnerBend,
        config="configyinner",
    )
    indSetA = []
    indSetB = []
    for k in range(3):
        indSetA.append(pts[9, 0, k])
        indSetB.append(pts[13, 0, k])
    DVCon.addLinearConstraintsShape(
        indSetA,
        indSetB,
        factorA=1.0,
        factorB=-1.0,
        lower=-0.02853 + minInnerBend,
        upper=0.02853 - minInnerBend,
        config="configyinner",
    )
    indSetA = []
    indSetB = []
    for k in range(3):
        indSetA.append(pts[10, 0, k])
        indSetB.append(pts[12, 0, k])
    DVCon.addLinearConstraintsShape(
        indSetA,
        indSetB,
        factorA=1.0,
        factorB=-1.0,
        lower=-0.01635 + minInnerBend,
        upper=0.01635 - minInnerBend,
        config="configyinner",
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
    optProb.addObj("obj", scale=1)
    # Add physical constraints
    #optProb.addCon("HFL", lower=HFL_target, upper=HFL_target, scale=1)

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

elif args.task == "testAPI":

    DASolver.setOption("primalMinResTol", 1e-2)
    DASolver.updateDAOption()
    optFuncs.runPrimal()

else:
    print("task arg not found!")
    exit(0)
