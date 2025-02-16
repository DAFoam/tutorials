#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
from mpi4py import MPI
from dafoam import PYDAFOAM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-augmented", help="whether to use augumented model", type=bool, default=False)
args = parser.parse_args()

gcomm = MPI.COMM_WORLD

U0 = 10.0

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e8,
    "primalBC": {"U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0, 0]}, "useWallFunction": True},
    "primalVarBounds": {"omegaMin": -1e16},
    "regressionModel": {
        "active": args.augmented,
        "model": {
            "writeFeatures": True,
            "modelType": "externalTensorFlow",
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
    "tensorflow": {
        "active": args.augmented,
        "model": {
            "predictBatchSize": 10000,
            "nInputs": 4,
        },
    },
    "function": {
        "drag": {
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
