#!/usr/bin/env python
"""
DAFoam run script
"""

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM

gcomm = MPI.COMM_WORLD
# =============================================================================
# Input Parameters
# =============================================================================
# UMag = 20, AOA=4.5 degs
Ux = 19.938346674662556
Uy = 1.5691819145569

# Input parameters for DAFoam
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [Ux, Uy, 0.0]},
        "useWallFunction": True,
    },
}
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
