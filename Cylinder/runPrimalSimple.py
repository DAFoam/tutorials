#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed (unsteady)
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

U0 = 10.0

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["cylinder"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-16,
    "primalMinResTolDiff": 1.0e16,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": False,
    },
    "checkMeshThreshold": {"maxAspectRatio": 5000.0},
}

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver()
