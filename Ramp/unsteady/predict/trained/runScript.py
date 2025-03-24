#!/usr/bin/env python

#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
import argparse
import numpy as np
from mpi4py import MPI
import json
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys.mphys_dafoam import DAFoamBuilderUnsteady

np.set_printoptions(precision=8, threshold=10000)

# =============================================================================
# Input Parameters
# =============================================================================

with open("./designVariable.json") as f:
    dv = json.load(f)
parameter0 = dv["parameter"]

# Set the parameters for optimization
daOptions = {
    "solverName": "DAPimpleFoam",
    "primalBC": {"useWallFunction": False},
    "printIntervalUnsteady": 1,
    "regressionModel": {
        "active": True,
        "reg_model1": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "chiSA", "PSoSS"],
            "outputName": "betaFINuTilda",
            "hiddenLayerNeurons": [20, 20],
            "inputShift": [0.0, 0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0, 1.0],
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "defaultOutputValue": 1.0,
            "outputUpperBound": 1e1,
            "outputLowerBound": -1e1,
        },
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["bot"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
            "timeOp": "average",
        },
    },
    "inputInfo": {
        "reg_model1": {"type": "regressionPar", "components": ["solver", "function"]},
    },
}


class Top(Multipoint):
    def setup(self):

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.add_subsystem(
            "scenario1",
            DAFoamBuilderUnsteady(solver_options=daOptions, mesh_options=None),
            promotes=["*"],
        )

    def configure(self):

        self.dvs.add_output("reg_model1", val=parameter0)

        # define the design variables to the top level
        self.add_design_var("reg_model1", lower=-100.0, upper=100.0, scaler=1.0)


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys.html")

prob.run_model()
