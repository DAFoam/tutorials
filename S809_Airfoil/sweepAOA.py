#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM
import numpy as np
import json
import sys

gcomm = MPI.COMM_WORLD

U0 = 68.0
p0 = 101325.0
T0 = 288.0
rho0 = p0 / T0 / 287.0
nuTilda0 = 1e-4
k0 = 0.2
omega0 = 1000.0
A0 = 0.01

outputFile = sys.argv[1]

aoas = [4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

# Set the parameters for optimization
daOptions = {
    "solverName": "DARhoSimpleFoam",
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0, 0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "k0": {"variable": "k", "patches": ["inout"], "value": [k0]},
        "omega0": {"variable": "omega", "patches": ["inout"], "value": [omega0]},
        "thermo:mu": 3.4e-5,
        "useWallFunction": False,
    },
    "primalMinResTol": 1.0e0,
    # "primalMinResTolDiff": 1e8,
    "primalObjStdTol": {"active": True, "objFuncName": "CL", "steps": 200, "tol": 0.0001, "tolDiff": 1e3},
    "useMeanStates": {"active": True, "start": 0.6},
    "primalVarBounds": {"omegaMin": -1e16},
    "useConstrainHbyA": True,
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
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
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
        "CM": {
            "part1": {
                "type": "moment",
                "source": "patchToFace",
                "patches": ["wing"],
                "axis": [0.0, 0.0, 1.0],
                "center": [0.25, 0.0, 0.05],
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0 * 1.0),
                "addToAdjoint": True,
            }
        },
    },
    "designVar": {
        "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
    },
    "checkMeshThreshold": {"maxAspectRatio": 10000.0},
}


def aoa(val, DASolver):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


xDV = {}
xDV["aoa"] = [0.0]

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.addInternalDV("aoa", [0.0], aoa, lower=-50, upper=50, scale=1.0)

outputDict = {}
outputDict["CD"] = []
outputDict["CL"] = []
outputDict["CM"] = []

for angle in aoas:
    if gcomm.rank == 0:
        print("AOA: ", angle)

    xDV["aoa"] = [angle]
    DASolver.setInternalDesignVars(xDV)

    DASolver()
    funcs = {}
    DASolver.evalFunctions(funcs, evalFuncs=["CD", "CL", "CM"])
    outputDict["CD"].append(funcs["CD"])
    outputDict["CL"].append(funcs["CL"])
    outputDict["CM"].append(funcs["CM"])

outputDict["AOA"] = aoas

if gcomm.rank == 0:
    with open(outputFile, "w") as fp:
        json.dump(outputDict, fp)
