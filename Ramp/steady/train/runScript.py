#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
import argparse
import numpy as np
import json, copy
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic

np.set_printoptions(precision=8, threshold=10000)

parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: run_driver (default), run_model, compute_totals, check_totals
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
# which case to run
parser.add_argument("-index", help="which case index to run", type=int, default=0)
args = parser.parse_args()


# =============================================================================
# Input Parameters
# =============================================================================

np.random.seed(1)

cases = ["c1", "c2"]
U0 = [10, 20.0]
CDData = np.array([0.1683, 0.7101])


with open("./probePointCoords.json") as f:
    probePointCoords = json.load(f)

# Input parameters for DAFoam
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e3,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0[0], 0, 0]},
        "useWallFunction": True,
    },
    "primalVarBounds": {"omegaMin": -1e16},
    "regressionModel": {
        "active": True,
        "model1": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "PSoSS", "KoU2"],
            "outputName": "betaFIK",
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
            "writeFeatures": True,
        },
        "model2": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "PSoSS", "KoU2"],
            "outputName": "betaFIOmega",
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
            "writeFeatures": True,
        },
    },
    "function": {
        "pVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["bot"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "p",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
        "UFieldVar": {
            "type": "variance",
            "source": "boxToCell",
            "min": [-10.0, -10.0, -10.0],
            "max": [10.0, 10.0, 10.0],
            "scale": 0.1,
            "mode": "field",
            "varName": "U",
            "varType": "vector",
            "components": [0, 1],
            "timeDependentRefData": False,
        },
        "UProbeVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 1.0,
            "mode": "probePoint",
            "probePointCoords": probePointCoords["probePointCoords"],
            "varName": "U",
            "varType": "vector",
            "components": [0, 1],
            "timeDependentRefData": False,
        },
        "CDError": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["bot"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
            "calcRefVar": True,
            "ref": [0.0],  # we will assign this laterbecause each case has a different ref
        },
        "betaKVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 0.01,
            "mode": "field",
            "varName": "betaFIK",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
        "betaOmegaVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 0.01,
            "mode": "field",
            "varName": "betaFIOmega",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["bot"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-6,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
    },
    "normalizeStates": {
        "U": U0[0],
        "p": U0[0] * U0[0] / 2.0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "inputInfo": {
        "model1": {
            "type": "regressionPar",
            "components": ["solver", "function"],
        },
        "model2": {
            "type": "regressionPar",
            "components": ["solver", "function"],
        },
    },
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builders to initialize the DASolvers
        builders = {}
        for idxI, case in enumerate(cases):
            options = copy.deepcopy(daOptions)
            options["primalBC"]["U0"]["value"] = [U0[idxI], 0.0, 0.0]
            options["function"]["CDError"]["ref"] = [float(CDData[idxI])]
            builders[case] = DAFoamBuilder(
                options=options, mesh_options=None, scenario="aerodynamic", run_directory=case
            )
            builders[case].initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # initialize scenarios
        self.scenarios = {}
        for case in cases:
            self.scenarios[case] = self.mphys_add_scenario(case, ScenarioAerodynamic(aero_builder=builders[case]))

        self.add_subsystem("obj", om.ExecComp("value=c1+c2"))

    def configure(self):
        # configure and setup perform a similar function, i.e., initialize the optimization.
        # But configure will be run after setup

        # setup dv for the idv component
        nParameters1 = self.c1.coupling.solver.DASolver.getNRegressionParameters("model1")
        parameter01 = (np.random.rand(nParameters1) - 0.5) * 0.02
        self.dvs.add_output("parameter1", val=parameter01)

        nParameters2 = self.c1.coupling.solver.DASolver.getNRegressionParameters("model2")
        parameter02 = (np.random.rand(nParameters2) - 0.5) * 0.02
        self.dvs.add_output("parameter2", val=parameter02)

        # manually connect the dvs output to the geometry and cruise
        for case in cases:
            self.connect("parameter1", "%s.model1" % case)
            self.connect("parameter2", "%s.model2" % case)
            self.connect("%s.aero_post.pVar" % case, "obj.%s" % case)

        # define the design variables to the top level
        self.add_design_var("parameter1", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("parameter2", lower=-10.0, upper=10.0, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("obj.value", scaler=1.0)


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys.html")

# use pyoptsparse to setup optimization
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer
# options for optimizers
if args.optimizer == "SNOPT":
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-5,
        "Major optimality tolerance": 1.0e-5,
        "Minor feasibility tolerance": 1.0e-5,
        "Verify level": -1,
        "Function precision": 1.0e-5,
        "Major iterations limit": 100,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol": 1.0e-5,
        "constr_viol_tol": 1.0e-5,
        "max_iter": 100,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.optimizer == "SLSQP":
    prob.driver.opt_settings = {
        "ACC": 1.0e-5,
        "MAXIT": 100,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("optimizer arg not valid!")
    exit(1)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"

if args.task == "run_driver":
    # run the optimization
    prob.run_driver()
    opt_dv = {
        "parameter1": prob.get_val("parameter1").tolist(),
        "parameter2": prob.get_val("parameter2").tolist(),
    }
    with open("designVariable.json", "w") as f:
        json.dump(opt_dv, f)
elif args.task == "run_model":
    # just run the primal once
    prob.run_model()
elif args.task == "compute_totals":
    # just run the primal and adjoint once
    prob.run_model()
    totals = prob.compute_totals()
    if MPI.COMM_WORLD.rank == 0:
        print(totals)
elif args.task == "check_totals":
    # verify the total derivatives against the finite-difference
    prob.run_model()
    prob.check_totals(compact_print=False, step=1e-3, form="central", step_calc="abs")
else:
    print("task arg not found!")
    exit(1)
