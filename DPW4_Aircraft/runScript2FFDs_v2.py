#!/usr/bin/env python
"""
DAFoam run script for the DPW4 wing-body-tail case (trimmed)
This setup uses a FFD file that contains two blocks (0: wing, 1: tail)
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

LScale = 1.0  # scale such that the L0=1

T0 = 300.0
p0 = 101325.0
rho0 = p0 / T0 / 287.0
U0 = 295.0
L0 = 275.80 * 0.0254 * LScale
A0 = 594720.0 * 0.0254 * 0.0254 / 2.0 * LScale * LScale
CofR = [1325.90 * 0.0254 * LScale, 468.75 * 0.0254 * LScale, 177.95 * 0.0254 * LScale]
nuTilda0 = 4.5e-5

CL_target = 0.5
CMY_target = 0.0
alpha0 = 2.5027

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["wing", "tail"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-9,
    "useAD": {"mode": "reverse"},
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
                "patches": ["wing", "body", "tail"],
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
                "patches": ["wing", "body", "tail"],
                "directionMode": "normalToFlow",
                "alphaName": "alpha",
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
                "addToAdjoint": True,
            }
        },
        "CMY": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["wing", "body", "tail"],
                "axis": [0.0, 1.0, 0.0],
                "center": CofR,
                "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0 * L0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # transonic preconditioner to speed up the adjoint convergence
    "transonicPCOption": 1,
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0, "T": T0},
    "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
    "adjPCLag": 1,
    "checkMeshThreshold": {
        "maxAspectRatio": 1000.0,
        "maxNonOrth": 75.0,
        "maxSkewness": 8.0},
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


# =================================================================================================
# DVGeo
# =================================================================================================
# setup FFDs
# Global FFD should contain the entire aicraft
DVGeo = DVGeometry("./FFD/wingTailFFD.xyz")


def setQuaterChordRefAxis(DVGeoObj, refAxisName, ffdVolI=0):
    """
    Set the reference axis for a DVGeo object. We will use its quater chord as the reference axis
    The quater chord is located at 25% of the FFD box in the x direction.
    NOTE: this function assumes x (i index) is streamwise, y (j index) is spanwise, and z (k index) is vertical

    Input:
        DVGeoObj: an DVGeo object created by calling DVGeometry(...)

        refAxisName: a name for the reference axis, this will be used later in defining twist variables

        ffdVolI: the index of the FFD block. For a single block plot3D file, ffdVolI = 0

    Output:
        nTwistObj: the number of twist variables. This is essentially the number of grid in the y direction
    """
    coefObj = DVGeoObj.FFD.vols[ffdVolI].coef.copy()
    # shape in the y direction (1 <-> spanwise) is the number of twists
    nTwistObj = coefObj.shape[1]
    objRefAxis = np.zeros((nTwistObj, 3))
    for j in range(nTwistObj):
        # 0 means the x coordinate
        max_x = np.max(coefObj[:, j, :, 0])
        min_x = np.min(coefObj[:, j, :, 0])
        objRefAxis[j, 0] = min_x + 0.25 * (max_x - min_x)
        objRefAxis[j, 1] = np.average(coefObj[:, j, :, 1])
        objRefAxis[j, 2] = np.average(coefObj[:, j, :, 2])
    # Create the actual reference axis
    c1 = Curve(X=objRefAxis, k=2)
    DVGeoObj.addRefAxis(refAxisName, c1, volumes=[ffdVolI])

    return nTwistObj


# call the setQuaterChordRefAxis function to set the reference axis
nTwistWing = setQuaterChordRefAxis(DVGeo, "wing", ffdVolI=0)
nTwistTail = setQuaterChordRefAxis(DVGeo, "tail", ffdVolI=1)


def twist(val, geo):
    # Set all the twist values
    for i in range(len(val)):
        # i+2 because we don't change the root twist and the 2nd to the root
        # Otherwise, we will get negative mesh
        geo.rot_y["wing"].coef[i + 2] = val[i]
    geo.rot_y["wing"].coef[:2] = 0.0


def tailTwist(val, geo):
    # Fix the root twist and the 2nd to the root. Otherwise, we will get negative mesh
    geo.rot_y["tail"].coef[2:] = val[0]
    geo.rot_y["tail"].coef[:2] = 0.0


def alpha(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), 0.0, float(U0 * np.sin(aoa))]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


# FFD shape
pts = DVGeo.getLocalIndex(0)  # index 0 means the 1st element in FFD: wing
indexList = pts[:, 2:, :].flatten()  # we don't change the 1st and 2nd layers points in y
PS = geo_utils.PointSelect("list", indexList)
DVGeo.addLocalDV("shapez", lower=-1.0, upper=1.0, axis="z", scale=10.0, pointSelect=PS)
daOptions["designVar"]["shapez"] = {"designVarType": "FFD"}
# twist
DVGeo.addGlobalDV("twist", np.zeros(nTwistWing - 2), twist, lower=-10.0, upper=10.0, scale=0.1)
daOptions["designVar"]["twist"] = {"designVarType": "FFD"}
# AOA
DVGeo.addGlobalDV("alpha", alpha0, alpha, lower=0, upper=10.0, scale=1.0)
daOptions["designVar"]["alpha"] = {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "z"}
# Tail twist
DVGeo.addGlobalDV("tail", np.zeros(1), tailTwist, lower=-10, upper=10, scale=0.1)
daOptions["designVar"]["tail"] = {"designVarType": "FFD"}

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

# Le/Te constraints
DVCon.addLeTeConstraints(0, "iHigh")
DVCon.addLeTeConstraints(0, "iLow")

# (flattened)LE Root, break and tip. These are different from above
leRoot = np.array([25.22 * LScale, 3.20 * LScale, 0])
leBreak = np.array([31.1358 * LScale, 10.8712 * LScale, 0.0])
leTip = np.array([45.2307 * LScale, 29.38 * LScale, 0.0])
rootChord = 11.83165 * LScale
breakChord = 7.25 * LScale
tipChord = 2.727 * LScale

coe1 = 0.2  # in production run where the mesh is refined, set coe1=0.01
coe2 = 1.0 - coe1
xaxis = np.array([1.0, 0, 0])
leList = [leRoot + coe1 * rootChord * xaxis, leBreak + coe1 * breakChord * xaxis, leTip + coe1 * tipChord * xaxis]

teList = [leRoot + coe2 * rootChord * xaxis, leBreak + coe2 * breakChord * xaxis, leTip + coe2 * tipChord * xaxis]

DVCon.addVolumeConstraint(leList, teList, nSpan=25, nChord=30, lower=1.0, upper=3, scaled=True)

# Add the same grid of thickness constraints with minimum bound of 0.25
DVCon.addThicknessConstraints2D(leList, teList, 25, 30, lower=0.2, upper=3.0, scaled=True)

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
    optProb.addCon("CMY", lower=CMY_target, upper=CMY_target, scale=1)

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
