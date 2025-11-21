#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM

gcomm = MPI.COMM_WORLD

# aero setup
U0 = 10.0
A0 = 0.1
LRef = 1.0

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleFoam",
    "primalBC": {"useWallFunction": False},
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0),
            "addToAdjoint": False,
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0),
            "addToAdjoint": False,
        },
        "CMZ": {
            "type": "moment",
            "source": "patchToFace",
            "patches": ["wing"],
            "axis": [0.0, 0.0, 1.0],
            "center": [0.25, 0.0, 0.05],
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * LRef),
            "addToAdjoint": False,
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
