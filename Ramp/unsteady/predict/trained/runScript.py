#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM
import numpy as np
import json

gcomm = MPI.COMM_WORLD

with open("./probePointCoords.json") as f:
    probePointCoords = json.load(f)

# Set the parameters for optimization
daOptions = {
    "solverName": "DAPimpleFoam",
    "primalBC": {"useWallFunction": False},
    "printIntervalUnsteady": 1,
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "objFuncTimeOperator": "average",
    },
    "regressionModel": {
        "active": True,
        "model1": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "chiSA", "PSoSS"],
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": [20, 20],
            "inputShift": [0.0, 0.0, 0.0, 0.0],
            "inputScale": [0.001, 1.0, 0.01, 1.0],
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "defaultOutputValue": 1.0,
            "outputUpperBound": 1e1,
            "outputLowerBound": -1e1,
        }
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


def regModel(val, DASolver):
    for idxI in range(len(val)):
        val1 = float(val[idxI])
        DASolver.setRegressionParameter("model1", idxI, val1)


DASolver = PYDAFOAM(options=daOptions, comm=gcomm)

nParameters = DASolver.getNRegressionParameters("model1")

with open("./designVariable.json") as f:
    dv = json.load(f)
parameter0 = dv["parameter"]

DASolver.addInternalDV("parameter", parameter0, regModel, lower=-10, upper=10, scale=100.0)

DASolver()
