#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM
import json

gcomm = MPI.COMM_WORLD

U0 = 15.0

with open("./designVariable.json") as f:
    parameter0 = json.load(f)

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e10,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0, 0]},
        "useWallFunction": True,
    },
    "regressionModel": {
        "active": True,
        "model1": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "PSoSS", "KoU2"],
            "outputName": "betaFIK",
            "hiddenLayerNeurons": [20, 20],
            "inputShift": [0.0, 0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0, 0.1],
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "defaultOutputValue": 1.0,
            "outputUpperBound": 1e1,
            "outputLowerBound": -1e1,
            "writeFeatures": True,
        },
        "model2": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "PSoSS", "KoU2"],
            "outputName": "betaFIOmega",
            "hiddenLayerNeurons": [20, 20],
            "inputShift": [0.0, 0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0, 0.1],
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "defaultOutputValue": 1.0,
            "outputUpperBound": 1e1,
            "outputLowerBound": -1e1,
            "writeFeatures": True,
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


def regModel1(val, DASolver):
    for idxI in range(len(val)):
        val1 = float(val[idxI])
        DASolver.setRegressionParameter("model1", idxI, val1)

def regModel2(val, DASolver):
    for idxI in range(len(val)):
        val1 = float(val[idxI])
        DASolver.setRegressionParameter("model2", idxI, val1)

DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
DASolver.addInternalDV("parameter1", parameter0["parameter1"], regModel1, lower=-100, upper=100, scale=1.0)
DASolver.addInternalDV("parameter2", parameter0["parameter2"], regModel2, lower=-100, upper=100, scale=1.0)
DASolver()
