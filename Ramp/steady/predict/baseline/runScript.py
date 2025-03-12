#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs

gcomm = MPI.COMM_WORLD

U0 = 15.0

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e10,
    "primalBC": {"U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0, 0]}, "useWallFunction": True},
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["bot"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
