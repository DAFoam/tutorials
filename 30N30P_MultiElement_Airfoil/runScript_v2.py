#!/usr/bin/env python
"""
DAFoam run script for the 30P30N multi-element high-lift airfoil.
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
from pygeo import *
from pyspline import Curve
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
U0 = 68.0
p0 = 101325.0
nuTilda0 = 4.5e-5
T0 = 300.0
A0 = 0.1
rho0 = 1.0  # density for normalizing CD and CL
CD_target = 0.1887
CL_target = 3.416
alpha0 = 8.0

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["main", "slat", "flap"],
    "solverName": "DARhoSimpleFoam",
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
                "patches": ["main", "slat", "flap"],
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
                "patches": ["main", "slat", "flap"],
                "directionMode": "normalToFlow",
                "alphaName": "alpha",
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjPCLag": 10,
    "checkMeshThreshold": {"maxAspectRatio": 2000.0, "maxNonOrth": 75.0, "maxSkewness": 6.0},
    "designVar": {},
}

# mesh warping parameters, users need to manually specify the symmetry plane and their normals
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    "useRotations": False,
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
        "Major iterations limit": 100,
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


def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


def twistslat(val, geo):
    for i in range(2):
        geo.rot_z["slatAxis"].coef[i] = -val[0]


def translateslat(val, geo):
    C = geo.extractCoef("slatAxis")
    dx = val[0]
    dy = val[1]
    for i in range(len(C)):
        C[i, 0] = C[i, 0] + dx
    for i in range(len(C)):
        C[i, 1] = C[i, 1] + dy
    geo.restoreCoef(C, "slatAxis")


def twistflap(val, geo):
    for i in range(2):
        geo.rot_z["flapAxis"].coef[i] = -val[0]


def translateflap(val, geo):
    C = geo.extractCoef("flapAxis")
    dx = val[0]
    dy = val[1]
    for i in range(len(C)):
        C[i, 0] = C[i, 0] + dx
    for i in range(len(C)):
        C[i, 1] = C[i, 1] + dy
    geo.restoreCoef(C, "flapAxis")


DVGeo = DVGeometry("./FFD/airfoilFFD.xyz")
# Add reference axis for twist
# Slat refAxis
xSlat = [0.0169, 0.0169]
ySlat = [0.0034, 0.0034]
zSlat = [0.0, 0.1]
cSlat = Curve(x=xSlat, y=ySlat, z=zSlat, k=2)
DVGeo.addRefAxis("slatAxis", curve=cSlat, axis="z", volumes=[0])
# Flap refAxis
xFlap = [0.875, 0.875]
yFlap = [0.014, 0.014]
zFlap = [0.0, 0.1]
cFlap = Curve(x=xFlap, y=yFlap, z=zFlap, k=2)
DVGeo.addRefAxis("flapAxis", curve=cFlap, axis="z", volumes=[2])

# twist slat
twistslat0 = 0.0
DVGeo.addGlobalDV("twistslat", [twistslat0], twistslat, lower=-10.0, upper=10.0, scale=1.0)
daOptions["designVar"]["twistslat"] = {"designVarType": "FFD"}
# translate slat
translateslat0 = np.zeros(2)
DVGeo.addGlobalDV("translateslat", translateslat0, translateslat, lower=[-0.1, 0.0], upper=[0.0, 0.1], scale=1.0)
daOptions["designVar"]["translateslat"] = {"designVarType": "FFD"}
# shape main
iVol = 1
ptsMain = DVGeo.getLocalIndex(iVol)
indexListMain = ptsMain[:, :, :].flatten()
PSMain = geo_utils.PointSelect("list", indexListMain)
DVGeo.addLocalDV("shapemain", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PSMain)
daOptions["designVar"]["shapemain"] = {"designVarType": "FFD"}
# twist flap
twistflap0 = 0.0
DVGeo.addGlobalDV("twistflap", [twistflap0], twistflap, lower=-10.0, upper=10.0, scale=1.0)
daOptions["designVar"]["twistflap"] = {"designVarType": "FFD"}
# translate flap
translateflap0 = np.zeros(2)
DVGeo.addGlobalDV("translateflap", translateflap0, translateflap, lower=[0.0, -0.1], upper=[0.1, 0.0], scale=1.0)
daOptions["designVar"]["translateflap"] = {"designVarType": "FFD"}
# alpha
DVGeo.addGlobalDV("alpha", [alpha0], alpha, lower=-10.0, upper=10.0, scale=1.0)
daOptions["designVar"]["alpha"] = {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"}

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

# ******* Main ***********
leListMain = [[0.048, -0.014, 1e-6], [0.048, -0.014, 0.1 - 1e-6]]
teListMain = [[0.698, -0.014, 1e-6], [0.698, -0.014, 0.1 - 1e-6]]
# volume constraint
DVCon.addVolumeConstraint(leListMain, teListMain, nSpan=2, nChord=10, lower=1.0, upper=1.0, scaled=True)
# thickness constraint
DVCon.addThicknessConstraints2D(leListMain, teListMain, nSpan=2, nChord=10, lower=0.8, upper=3.0, scaled=True)
# NOTE: we need to add thickness and vol constraints for the tailing of the main airfoil
leListMainTrailing = [[0.702, 0.0328, 1e-6], [0.702, 0.0328, 0.1 - 1e-6]]
teListMainTrailing = [[0.854, 0.0328, 1e-6], [0.854, 0.0328, 0.1 - 1e-6]]
# volume constraint
DVCon.addVolumeConstraint(leListMainTrailing, teListMainTrailing, nSpan=2, nChord=10, lower=1.0, upper=1.0, scaled=True)
# thickness constraint
DVCon.addThicknessConstraints2D(
    leListMainTrailing, teListMainTrailing, nSpan=2, nChord=10, lower=0.8, upper=3.0, scaled=True
)
# symmetry constraint
nFFDs_x = ptsMain.shape[0]
indSetA = []
indSetB = []
for i in range(nFFDs_x):
    for j in [0, 1]:
        indSetA.append(ptsMain[i, j, 1])
        indSetB.append(ptsMain[i, j, 0])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0.0, upper=0.0)
# LE and TE constraint
indSetA = []
indSetB = []
for i in [0, nFFDs_x - 1]:
    for k in [0]:  # do not constrain k=1 because it is linked in the above symmetry constraint
        indSetA.append(ptsMain[i, 0, k])
        indSetB.append(ptsMain[i, 1, k])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=1.0, lower=0.0, upper=0.0)

#DVCon.writeTecplot("DVConstraints.dat")

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
    optProb.addObj("CL", scale=-1)
    # Add physical constraints
    optProb.addCon("CD", lower=CD_target, upper=CD_target, scale=1)

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

    optFuncs.runForwardAD("shapemain", 0)

else:
    print("task arg not found!")
    exit(0)
