#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed (multi point)
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
# which optimizer to use. Options are: slsqp (default), snopt, or ipopt
parser.add_argument("--opt", help="optimizer to use", type=str, default="ipopt")
# which task to run. Options are: opt (default), run, testSensShape, or solveCL
parser.add_argument("--task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

# global parameters
nMultiPoints = 3
MPWeights = [0.25, 0.25, 0.5]
U0 = [10.0, 10.0, 10.0]
URef = U0[0]  # we use the first U0 as reference velocity to normalize CD and CL
CL_target = [0.3, 0.7, 0.5]
alpha0 = [3.008097, 7.622412, 5.139186]
p0 = 0.0
nuTilda0 = 4.5e-5
k0 = 0.015
epsilon0 = 0.14
omega0 = 100.0
A0 = 0.1

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["wing"],
    "multiPoint": True,
    "nMultiPoints": nMultiPoints,
    "solverName": "DASimpleFoam",
    "useAD": {"mode": "reverse"},
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [URef, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "k0": {"variable": "k", "patches": ["inout"], "value": [k0]},
        "omega0": {"variable": "omega", "patches": ["inout"], "value": [omega0]},
        "epsilon0": {"variable": "epsilon", "patches": ["inout"], "value": [epsilon0]},
        "useWallFunction": True,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "mp0_alpha",
                "scale": 1.0 / (0.5 * URef * URef * A0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "alphaName": "mp0_alpha",
                "scale": 1.0 / (0.5 * URef * URef * A0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": URef,
        "p": URef * URef / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "k": k0 * 10.0,
        "epsilon": epsilon0,
        "omega": omega0,
        "phi": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-7, "FFD": 1e-3},
    "adjPCLag": 20,
    "designVar": {},
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
def dummyFunc(val, geo):
    pass


DVGeo = DVGeometry("./FFD/wingFFD.xyz")
DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")
# select points
iVol = 0
pts = DVGeo.getLocalIndex(iVol)
indexList = pts[:, :, :].flatten()
PS = geo_utils.PointSelect("list", indexList)
# shape variable
DVGeo.addLocalDV("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
daOptions["designVar"]["shapey"] = {"designVarType": "FFD"}
# angle of attack for each configuration
for i in range(nMultiPoints):
    # NOTE: here we don't need to implement the alpha function because the alpha values will be changed
    # in setMultiPointCondition. So we provide a dummyFunc
    DVGeo.addGlobalDV("mp%d_alpha" % i, alpha0[i], dummyFunc, lower=0.0, upper=10.0, scale=1.0)
    # add alpha for designVar
    daOptions["designVar"]["mp%d_alpha" % i] = {
        "designVarType": "AOA",
        "patches": ["inout"],
        "flowAxis": "x",
        "normalAxis": "y",
    }

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

leList = [[1e-4, 0.0, 1e-4], [1e-4, 0.0, 0.1 - 1e-4]]
teList = [[0.998 - 1e-4, 0.0, 1e-4], [0.998 - 1e-4, 0.0, 0.1 - 1e-4]]

# volume constraint
DVCon.addVolumeConstraint(leList, teList, nSpan=2, nChord=10, lower=1.0, upper=3, scaled=True)

# thickness constraint
DVCon.addThicknessConstraints2D(leList, teList, nSpan=2, nChord=10, lower=0.8, upper=3.0, scaled=True)

# symmetry constraint
nFFDs_x = pts.shape[0]
indSetA = []
indSetB = []
for i in range(nFFDs_x):
    for j in [0, 1]:
        indSetA.append(pts[i, j, 1])
        indSetB.append(pts[i, j, 0])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0)

# LE and TE constraint
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
# provide a function to set primal conditions
def setMultiPointCondition(xDV, index):
    aoa = xDV["mp%d_alpha" % index].real * np.pi / 180.0
    inletU = [float(U0[index] * np.cos(aoa)), float(U0[index] * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()
    return


# provide a function to assemble the funcs from MP
def setMultiPointObjFuncs(funcs, funcsMP, index):
    for key in funcs:
        if "fail" in key:
            pass
        elif "DVCon" in key:
            funcsMP[key] = funcs[key]
        elif "CD" in key:
            try:
                funcsMP["obj"] += funcs[key] * MPWeights[index]
            except Exception:
                funcsMP["obj"] = 0.0
                funcsMP["obj"] += funcs[key] * MPWeights[index]
        elif "CL" in key:
            funcsMP["mp%d_CL" % index] = funcs[key]
    return


# provide a function to assemble the funcs from MP
def setMultiPointObjFuncsSens(xDVs, funcsMP, funcsSens, funcsSensMP, index):
    for key in funcsMP:
        try:
            keySize = len(funcsMP[key])
        except Exception:
            keySize = 1
        try:
            funcsSensMP[key]
        except Exception:
            funcsSensMP[key] = {}

        if "fail" in key:
            pass
        elif "DVCon" in key:
            funcsSensMP[key]["mp%d_alpha" % index] = np.zeros((keySize, 1), "d")
            funcsSensMP[key]["shapey"] = funcsSens[key]["shapey"]
        elif "obj" in key:
            funcsSensMP[key]["mp%d_alpha" % index] = funcsSens["CD"]["mp%d_alpha" % index] * MPWeights[index]
            try:
                funcsSensMP[key]["shapey"] += funcsSens["CD"]["shapey"] * MPWeights[index]
            except Exception:
                funcsSensMP[key]["shapey"] = np.zeros(len(xDVs["shapey"]), "d")
                funcsSensMP[key]["shapey"] += funcsSens["CD"]["shapey"] * MPWeights[index]
        elif "mp%d_CL" % index in key:
            for alphaI in range(nMultiPoints):
                if alphaI == index:
                    funcsSensMP[key]["mp%d_alpha" % alphaI] = funcsSens["CL"]["mp%d_alpha" % index]
                else:
                    funcsSensMP[key]["mp%d_alpha" % alphaI] = np.zeros((keySize, 1), "d")
            funcsSensMP[key]["shapey"] = funcsSens["CL"]["shapey"]

    return


# in addition to provide DASolver etc. we need to set setMultiPointCondition,
# setMultiPointObjFuncs, and setMultiPointObjFuncsSens for optFuncs for multipoint
optFuncs.DASolver = DASolver
optFuncs.DVGeo = DVGeo
optFuncs.DVCon = DVCon
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm
optFuncs.setMultiPointCondition = setMultiPointCondition
optFuncs.setMultiPointObjFuncs = setMultiPointObjFuncs
optFuncs.setMultiPointObjFuncsSens = setMultiPointObjFuncsSens

# =============================================================================
# Task
# =============================================================================
if args.task == "opt":

    optProb = Optimization("opt", objFun=optFuncs.calcObjFuncValuesMP, comm=gcomm)
    DVGeo.addVariablesPyOpt(optProb)
    DVCon.addConstraintsPyOpt(optProb)

    # Add objective
    optProb.addObj("obj", scale=1)
    # Add physical constraints
    for i in range(nMultiPoints):
        optProb.addCon("mp%d_CL" % i, lower=CL_target[i], upper=CL_target[i], scale=1)

    if gcomm.rank == 0:
        print(optProb)

    DASolver.runColoring()

    opt = OPT(args.opt, options=optOptions)
    histFile = "./%s_hist.hst" % args.opt
    sol = opt(optProb, sens=optFuncs.calcObjFuncSensMP, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

elif args.task == "runPrimal":

    optFuncs.runPrimal(objFun=optFuncs.calcObjFuncValuesMP)

elif args.task == "runAdjoint":

    optFuncs.runAdjoint(objFun=optFuncs.calcObjFuncValuesMP, sens=optFuncs.calcObjFuncSensMP)

else:
    print("task arg not found!")
    exit(0)
