#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs

gcomm = MPI.COMM_WORLD

U0 = 10.0

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleFoam",
    "primalBC": {"U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0, 0]}, "useWallFunction": True},
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
