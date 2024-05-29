#!/usr/bin/env python
"""
DAFoam run script for the U bend channel case
"""

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM

# =============================================================================
# Input Parameters
# =============================================================================
gcomm = MPI.COMM_WORLD

# Set the parameters for optimization
daOptions = {
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1e-8,
    "primalMinResTolDiff": 1e8,
    "primalBC": {
        "useWallFunction": False,
    },
    "objFunc": {
        "obj": {
            "part1": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["inlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
    },
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
