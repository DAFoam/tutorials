#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed (unsteady)
"""

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
import numpy as np

# =============================================================================
# Input Parameters
# =============================================================================
gcomm = MPI.COMM_WORLD

U0 = 10.0
A0 = 0.1
aoa0 = 15.0

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-16,
    "primalMinResTolDiff": 1.0e16,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": True,
    },
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
    "designVar": {
        "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"}
    },
    "checkMeshThreshold": {"maxAspectRatio": 5000.0},
}


def aoa(val, geo):
    aoa = val[0] * np.pi / 180.0
    inletU = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
    DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": inletU}})
    DASolver.updateDAOption()


DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.addInternalDV("aoa", [aoa0], aoa, lower=-50, upper=50, scale=1.0)

optFuncs.DASolver = DASolver
optFuncs.DVGeo = None
optFuncs.DVCon = None
optFuncs.evalFuncs = ["CD", "CL"]
optFuncs.gcomm = gcomm

optFuncs.runPrimal()
