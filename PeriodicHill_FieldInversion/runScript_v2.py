#!/usr/bin/env python
"""
DAFoam run script for the periodic hills case
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
U0 = 0.028
p0 = 0.0
nuTilda0 = 1e-4
J0 = 0.0479729567  # this is the baseline L2 norm

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["bottomWall", "topWall", "inlet", "outlet"],
    "solverName": "DASimpleFoam",
    "useAD": {"mode": "reverse"},
    "primalMinResTol": 1.0e-10,
    "objFunc": {
        "FI": {
            "Ux": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "data": "UData",
                "scale": 1,
                "addToAdjoint": True,
                "weightedSum": True,
                "weight": 1 / J0,
            },
            "beta": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "data": "beta",
                "scale": 1e-10,
                "addToAdjoint": True,
                "weightedSum": False,
            },
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 2, "jacMatReOrdering": "natural", "gmresMaxIters": 3000},
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-7, "FFD": 1e-3},
    "adjPCLag": 100,
    "designVar": {},
    "fvSource": {
        "gradP": {
            "type": "uniformPressureGradient",
            "value": 6.634074021107811e-06,
            "direction": [1.0, 0.0, 0.0],
        },
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
if args.opt == "ipopt":
    optOptions = {
        "tol": 1.0e-7,
        "max_iter": 50,
        "output_file": "opt_IPOPT.out",
        "constr_viol_tol": 1.0e-7,
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 26,
        "nlp_scaling_method": "gradient-based",
        "alpha_for_y": "full",
        "recalc_y": "yes",
        "print_level": 5,
        "acceptable_tol": 1.0e-7,
    }
else:
    print("opt arg not valid!")
    exit(0)


# =============================================================================
# Design variable setup
# =============================================================================
def betaFieldInversion(val, geo):
    for idxI, v in enumerate(val):
        DASolver.setFieldValue4GlobalCellI(b"betaFieldInversion", v, idxI)


DVGeo = DVGeometry("./FFD/periodicHillFFD.xyz")
DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")

nCells = 3500
beta0 = np.ones(nCells, dtype="d")
DVGeo.addGlobalDV("beta", value=beta0, func=betaFieldInversion, lower=-5.0, upper=10.0, scale=1)
daOptions["designVar"]["beta"] = {"designVarType": "Field", "fieldName": "betaFieldInversion", "fieldType": "scalar"}

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
#:Q =============================================================================
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

    optProb.addObj("FI", scale=1)

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

else:
    print("task arg not found!")
    exit(0)
