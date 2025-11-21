#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic

np.set_printoptions(precision=8, threshold=10000)

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: run_driver (default), run_model, compute_totals, check_totals
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
# which case to run
parser.add_argument("-index", help="which case index to run", type=int, default=0)
args = parser.parse_args()

np.random.seed(0)

U0 = 10.0
A0 = 0.1
LRef = 1.0
nCells = 12236
idxI = args.index
cases = ["aoa-10", "aoa-12", "aoa-14", "aoa-16", "aoa-18"]
liftRefs = [1.058305255637404, 1.1987804783652192, 1.206293104609119, 0.964962142060612, 0.80093469978045]
liftRef = liftRefs[idxI]
case = cases[idxI]

#with open("./probePointCoords.json") as f:
    #probePointCoords = json.load(f)

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleFoam",
    #"primalMinResTol": 1.0e-5,
    "primalMinResTolDiff": 1.0e7,
    "primalBC": {
        "useWallFunction": True,
    }, 
    "regressionModel": {
        "active": True,
        "model": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "chiSA", "PSoSS"],
            "outputName": "dummy",
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
        } 
    },
    "function": {
        "liftVar": {
           "type": "force",
           "source": "patchToFace",
           "patches": ["wing"],
           "directionMode": "fixedDirection",
           "direction": [0.0, 1.0, 0.0],
           "scale": 1.0 / (0.5 * U0 * U0 * A0),
           "calcRefVar": True,
           "ref": [liftRef],
        },
        "betaVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 1,
            "mode": "field",
            "varName": "betaFINuTilda",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-3,
        "gmresMaxIters": 1000,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": False,
        "useMGSO": True,
    },
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "beta": {
            "type": "field",
            "fieldName": "betaFINuTilda",
            "fieldType": "scalar",
            "distributed": False,
            "components": ["solver", "function"],
        },
    },
}


class Top(Multipoint):
    def setup(self):

        dafoam_builder = DAFoamBuilder(options=daOptions, mesh_options=None, scenario="aerodynamic", run_directory=case)
        dafoam_builder.initialize(self.comm)

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.add_subsystem("obj", om.ExecComp("val=error+regulation"))


    def configure(self):

        # add the design variables to the dvs component's output
        self.dvs.add_output("beta", val=np.ones(nCells), distributed=False)
        self.connect("beta", "scenario1.beta")

        # define the design variables to the top level
        self.add_design_var("beta", lower=-3.0, upper=3.0, scaler=1.0)

        # add the objective
        self.connect("scenario1.aero_post.liftVar", "obj.error")
        self.connect("scenario1.aero_post.betaVar", "obj.regulation")
        self.add_objective("obj.val", scaler=1.0)


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
elif args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol": 1.0e-5,
        "constr_viol_tol": 1.0e-5,
        "max_iter": 50,
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
