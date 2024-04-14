#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
import argparse
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
from pyoptsparse import Optimization, OPT
import numpy as np
import json
np.set_printoptions(precision=8, threshold=10000)

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--task", help="type of run to do", type=str, default="opt")
parser.add_argument("--optimizer", help="optimizer to use", type=str, default="snopt")
parser.add_argument("--readDV", help="whether to read designVariable.json", type=int, default=0)
parser.add_argument("--mode", help="AD mode: either reverse or forward", type=str, default="reverse")
parser.add_argument("--seedIndex", help="which design variable index to set seeds", type=int, default=0)
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

np.random.seed(0)

# Set the parameters for optimization
daOptions = {
    "solverName": "DAPimpleFoam",
    "useAD": {"mode": args.mode},
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 50,
        "PCMatUpdateInterval": 1,
        "objFuncTimeOperator": "average",
        "reduceIO": True,
    },
    "primalBC": {"useWallFunction": False},
    "printIntervalUnsteady": 1,
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
        "VAR": {
            "pVar": {
                "type": "variance",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "scale": 0.02,
                "mode": "surface",
                "surfaceNames": ["bot"],
                "varName": "p",
                "varType": "scalar",
                "timeDependentRefData": True,
                "addToAdjoint": True,
            },
            "betaVar": {
                "type": "variance",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "scale": 0.01,
                "mode": "field",
                "varName": "betaFINuTilda",
                "varType": "scalar",
                "timeOperator": "average",
                "timeDependentRefData": False,
                "addToAdjoint": True,
            },
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-10,
        "gmresAbsTol": 1.0e-6,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
        "useNonZeroInitGuess": False,
    },
    "normalizeStates": {
        "U": 10,
        "p": 50,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "designVar": {
        "parameter": {"designVarType": "RegPar", "modelName": "model1"},
    },
}

# options for optimizers
if args.optimizer == "snopt":
    optOptions = {
        "Major feasibility tolerance": 1.0e-6,
        "Major optimality tolerance": 1.0e-6,
        "Minor feasibility tolerance": 1.0e-6,
        "Verify level": -1,
        "Function precision": 1.0e-6,
        "Major iterations limit": 50,
        "Linesearch tolerance": 0.999,
        "Hessian updates": 50,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "ipopt":
    optOptions = {
        "tol": 1.0e-6,
        "constr_viol_tol": 1.0e-6,
        "max_iter": 200,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 100,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.optimizer == "slsqp":
    optOptions = {
        "ACC": 1.0e-7,
        "MAXIT": 200,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("opt arg not valid!")
    exit(0)

# =============================================================================
# Design variable setup
# =============================================================================
def regModel(val, DASolver):
    for idxI in range(len(val)):
        val1 = float(val[idxI])
        DASolver.setRegressionParameter("model1", idxI, val1)


# =============================================================================
# DAFoam initialization
# =============================================================================
DASolver = PYDAFOAM(options=daOptions, comm=gcomm)
nParameters = DASolver.getNRegressionParameters("model1")
if args.readDV == 1:
    with open("./designVariable.json") as f:
        dv = json.load(f)
    parameter0 = dv["parameter"]
else:
    # parameter0 = np.ones(nParameters) * 0.01
    parameter0 = (np.random.rand(nParameters) - 0.5) * 0.05
DASolver.addInternalDV("parameter", parameter0, regModel, lower=-10, upper=10, scale=1.0)
evalFuncs = []
DASolver.setEvalFuncs(evalFuncs)

# =============================================================================
# Initialize optFuncs for optimization
# =============================================================================
optFuncs.DASolver = DASolver
optFuncs.DVGeo = None
optFuncs.DVCon = None
optFuncs.evalFuncs = evalFuncs
optFuncs.gcomm = gcomm

# =============================================================================
# Task
# =============================================================================
if args.task == "opt":

    optProb = Optimization("opt", objFun=optFuncs.calcObjFuncValues, comm=gcomm)
    DASolver.addVariablesPyOpt(optProb)

    optProb.addObj("VAR", scale=1)

    if gcomm.rank == 0:
        print(optProb)

    DASolver.runColoring()

    opt = OPT(args.optimizer, options=optOptions)
    histFile = "./%s_hist.hst" % args.optimizer
    sol = opt(optProb, sens=optFuncs.calcObjFuncSens, storeHistory=histFile)
    if gcomm.rank == 0:
        print(sol)

elif args.task == "runPrimal":

    optFuncs.runPrimal()

elif args.task == "runAdjoint":

    optFuncs.runAdjoint()

elif args.task == "runForwardAD":

    optFuncs.runForwardAD("parameter", args.seedIndex)

else:
    print("task arg not found!")
    exit(0)
