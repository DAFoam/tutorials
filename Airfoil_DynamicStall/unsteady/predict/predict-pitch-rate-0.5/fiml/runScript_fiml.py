#!/usr/bin/env python
"""
Run Python tests for optimization integration
"""

from mpi4py import MPI
import os
import numpy as np
#from testFunctions import *

import openmdao.api as om
from openmdao.api import Group
from mphys.multipoint import Multipoint
from dafoam.mphys.mphys_dafoam import DAFoamBuilderUnsteady
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP

def run_tests(om, Top, comm, daOptions):

    # adjoint-deriv
    prob = om.Problem()
    prob.model = Top()
    prob.setup()
    prob.run_model()


gcomm = MPI.COMM_WORLD

# aero setup
U0 = 10.0
A0 = 0.1
LRef = 1.0

# Set the parameters for optimization
daOptions = {
    "solverName": "DAPimpleDyMFoam",
    "useAD": {"mode": "reverse", "seedIndex": 0, "dvName": "beta"},
    "primalBC": {
        "useWallFunction": False,
    },
    "checkMeshThreshold": {
        "maxAspectRatio": 2000.0,
        "maxNonOrth": 70.0,
        "maxSkewness": 4.0,
        "maxIncorrectlyOrientedFaces": 0,
    },
    "dynamicMesh": {
        "active": True,
        "mode": "rotation",
        "center": [0.25, 0.0, 0.0],
        "axis": "z",
        "omega": -0.5,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 5,
        "PCMatUpdateInterval": 1,
        "readZeroFields": True,
        "additionalOutput": ["betaFINuTilda"],
        "reduceIO": True,
     },
    "tensorflow": {
        "active": True,
        "dummy_nn_model": {
            "predictBatchSize": 500,
            "nInputs": 4,
        },
    },
    "regressionModel": {
        "active": True,
        "dummy_nn_model": {
            "modelType": "externalTensorFlow",
            "inputNames": ["PoD", "VoS", "chiSA", "PSoSS"],
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": [100, 100],
            "inputShift": [0.0, 0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0, 1.0],
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "writeFeatures": True,
            "outputUpperBound": 1e1,
            "outputLowerBound": -1e1,
            "defaultOutputValue": 1.0,
        },
    },
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
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-4,
        "gmresMaxIters": 1000,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": False,
        "useMGSO": True,
    },
    "unsteadyCompOutput": {
        "CD": ["CD"],
    },
}


class Top(Group):
    def setup(self):

        self.add_subsystem(
            "scenario",
            DAFoamBuilderUnsteady(solver_options=daOptions, mesh_options=None),
            promotes=["*"],
        )

# run the adjoint and forward ref
run_tests(om, Top, gcomm, daOptions)

