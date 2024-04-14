#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs

gcomm = MPI.COMM_WORLD

# Set the parameters for optimization
daOptions = {
    "solverName": "DAPimpleFoam",
    "primalBC": {"useWallFunction": False},
    "printIntervalUnsteady": 1,
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "objFuncTimeOperator": "average",
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["bot"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": True,
            }
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
