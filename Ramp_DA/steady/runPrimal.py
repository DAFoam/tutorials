#!/usr/bin/env python
"""
DAFoam run script
"""

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM

# =============================================================================
# Input Parameters
# =============================================================================
U0 = 10.0
nPatchFaces = 50
gcomm = MPI.COMM_WORLD

# Input parameters for DAFoam
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "useWallFunction": False,
    },
    "primalVarBounds": {"omegaMin": -1e16},
}
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
