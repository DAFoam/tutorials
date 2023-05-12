#!/usr/bin/env python
"""
DAFoam run script for the CRM wing at transonic speed
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
parser.add_argument("--opt", help="optimizer to use", type=str, default="snopt")
# which task to run. Options are: opt (default), run, testSensShape, or solveCL
parser.add_argument("--task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

# global parameters
U0 = 295.0
p0 = 101325.0
nuTilda0 = 4.5e-5
T0 = 300.0
CL_target = 0.5
alpha0 = 2.178962
A0 = 3.407014
rho0 = 1.1768  # density for normalizing CD and CL

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "useAD": {"mode": "reverse"},
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    # variable bounds for compressible flow conditions
    "primalVarBounds": {
        "UMax": 1000.0,
        "UMin": -1000.0,
        "pMax": 500000.0,
        "pMin": 20000.0,
        "eMax": 500000.0,
        "eMin": 100000.0,
        "rhoMax": 5.0,
        "rhoMin": 0.2,
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
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "natural", "gmresMaxIters": 2000, "gmresRestart": 2000},
    # transonic preconditioner to speed up the adjoint convergence
    "transonicPCOption": 2,
    "checkMeshThreshold": {"maxAspectRatio": 2000.0, "maxNonOrth": 75.0, "maxSkewness": 5.0},
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0, "T": T0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjPCLag": 10,
    "designVar": {},
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    "useRotations": False,
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
}

# options for optimizers
if args.opt == "snopt":
    optOptions = {
        "Major feasibility tolerance": 1.0e-7,
        "Major optimality tolerance": 1.0e-7,
        "Minor feasibility tolerance": 1.0e-7,
        "Verify level": -1,
        "Function precision": 1.0e-7,
        "Major iterations limit": 200,
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

# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="j")

# twist function, we keep the root twist constant so the first
# element in the twist design variable is the twist at the 2nd
# spanwise location
def twist(val, geo):
    for i in range(1, nTwists):
        geo.rot_y["bodyAxis"].coef[i] = - val[i - 1]


# angle of attack
def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), 0.0, float(U0 * np.sin(aoa))]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[:, :, :].flatten()
PS = geo_utils.PointSelect("list", indexList)
# shape
DVGeo.addLocalDV("shapez", lower=-1.0, upper=1.0, axis="z", scale=1.0, pointSelect=PS)
daOptions["designVar"]["shapez"] = {"designVarType": "FFD"}
# twist
DVGeo.addGlobalDV("twist", np.zeros(nTwists - 1), twist, lower=-10.0, upper=10.0, scale=1.0)
daOptions["designVar"]["twist"] = {"designVarType": "FFD"}
# alpha
DVGeo.addGlobalDV("alpha", [alpha0], alpha, lower=0.0, upper=10.0, scale=1.0)
daOptions["designVar"]["alpha"] = {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "z"}

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

# NOTE: the LE and TE lists are not parallel lines anymore, these two lists define lines that
# are close to the leading and trailing edges while being completely within the wing surface
LE_pt = np.array([0.01, 0.01, 0.0])
break_pt = np.array([0.848, 1.119, 0.0])
tip_pt = np.array([2.855, 3.755, 0.0])
root_chord = 1.689
break_chord = 1.036
tip_chord = 0.390

leList = [
    [LE_pt[0] + 0.01 * root_chord, LE_pt[1], LE_pt[2]],
    [break_pt[0] + 0.01 * break_chord, break_pt[1], break_pt[2]],
    [tip_pt[0] + 0.01 * tip_chord, tip_pt[1], tip_pt[2]],
]

teList = [
    [LE_pt[0] + 0.99 * root_chord, LE_pt[1], LE_pt[2]],
    [break_pt[0] + 0.99 * break_chord, break_pt[1], break_pt[2]],
    [tip_pt[0] + 0.99 * tip_chord, tip_pt[1], tip_pt[2]],
]


# volume constraint
DVCon.addVolumeConstraint(leList, teList, nSpan=25, nChord=30, lower=1.0, upper=3, scaled=True)

# thickness constraint
DVCon.addThicknessConstraints2D(leList, teList, nSpan=25, nChord=30, lower=0.5, upper=3.0, scaled=True)

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

    alpha4CLTarget = optFuncs.solveCL(CL_target, "alpha", "CL")
    alpha([alpha4CLTarget], None)

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

elif args.task == "runPrimal":

    optFuncs.runPrimal()

elif args.task == "runAdjoint":

    optFuncs.runAdjoint()

elif args.task == "runForwardAD":

    optFuncs.runForwardAD("shapez", 0)

else:
    print("task arg not found!")
    exit(0)
