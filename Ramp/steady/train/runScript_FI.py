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


parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="SNOPT")
# which task to run. Options are: opt (default), runPrimal, runAdjoint, checkTotals
parser.add_argument("-task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================

U0 = 10.0
nCells = 5000
case_dir = "c1"

np.random.seed(0)

with open("./probePointCoords.json") as f:
    probePointCoords = json.load(f)

# Input parameters for DAFoam
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e3,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0, 0]},
        "useWallFunction": True,
    },
    "primalVarBounds": {"omegaMin": -1e16},
    "objFunc": {
        "VAR": {
            "pVar": {
                "type": "variance",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "scale": 1.0,
                "mode": "surface",
                "surfaceNames": ["bot"],
                "varName": "p",
                "varType": "scalar",
                "timeDependentRefData": False,
                "addToAdjoint": True,
            },
            # "UFieldVar": {
            #    "type": "variance",
            #    "source": "boxToCell",
            #    "min": [-10.0, -10.0, -10.0],
            #    "max": [10.0, 10.0, 10.0],
            #    "scale": 0.1,
            #    "mode": "field",
            #    "varName": "U",
            #    "varType": "vector",
            #    "components": [0, 1],
            #    "timeDependentRefData": False,
            #    "addToAdjoint": True,
            # },
            # "UProbeVar": {
            #    "type": "variance",
            #    "source": "boxToCell",
            #    "min": [-10.0, -10.0, -10.0],
            #    "max": [10.0, 10.0, 10.0],
            #    "scale": 1.0,
            #    "mode": "probePoint",
            #    "probePointCoords": probePointCoords["probePointCoords"],
            #    "varName": "U",
            #    "varType": "vector",
            #    "components": [0, 1],
            #    "timeDependentRefData": False,
            #    "addToAdjoint": True,
            # "CDError": {
            #    "type": "force",
            #    "source": "patchToFace",
            #    "patches": ["bot"],
            #    "directionMode": "fixedDirection",
            #    "direction": [1.0, 0.0, 0.0],
            #    "scale": 1.0,
            #    "addToAdjoint": True,
            #    "calcRefVar": True,
            #    "ref": [0.0],  # we will assign this later because each case has a different ref
            # },
            "betaVar": {
                "type": "variance",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "scale": 1.0,
                "mode": "field",
                "varName": "betaFIOmega",
                "varType": "scalar",
                "timeOperator": "average",
                "timeDependentRefData": False,
                "addToAdjoint": True,
            },
        },
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["bot"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": 1.0,
                "addToAdjoint": False,
            }
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-6,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
    },
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "designVar": {
        "beta": {"designVarType": "Field", "fieldName": "betaFIOmega", "fieldType": "scalar", "distributed": False},
    },
}


def betaFunction(val, DASolver):
    for idxI, v in enumerate(val):
        DASolver.setFieldValue4GlobalCellI(b"betaFIOmega", v, idxI)
        DASolver.updateBoundaryConditions(b"betaFIOmega", b"scalar")


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(options=daOptions, mesh_options=None, scenario="aerodynamic", run_directory=case_dir)
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

    def configure(self):
        # configure and setup perform a similar function, i.e., initialize the optimization.
        # But configure will be run after setup

        # add the objective function to the cruise scenario
        self.cruise.aero_post.mphys_add_funcs()

        # pass this aoa function to the cruise group
        self.cruise.coupling.solver.add_dv_func("beta", betaFunction)
        self.cruise.aero_post.add_dv_func("beta", betaFunction)

        # add the design variables to the dvs component's output
        self.dvs.add_output("beta", val=np.ones(nCells), distributed=False)
        self.connect("beta", "cruise.beta")

        # define the design variables to the top level
        self.add_design_var("beta", lower=-5.0, upper=10.0, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("cruise.aero_post.VAR", scaler=1.0)


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
        "Major iterations limit": 200,
        "Linesearch tolerance": 0.999,
        "Hessian updates": 200,
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

if args.task == "opt":
    # run the optimization
    prob.run_driver()
elif args.task == "runPrimal":
    # just run the primal once
    prob.run_model()
elif args.task == "runAdjoint":
    # just run the primal and adjoint once
    prob.run_model()
    totals = prob.compute_totals()
    if MPI.COMM_WORLD.rank == 0:
        print(totals)
elif args.task == "checkTotals":
    # verify the total derivatives against the finite-difference
    prob.run_model()
    prob.check_totals(
        # of=["cruise.aero_post.CD", "cruise.aero_post.CL"],
        # wrt=["shape", "aoa"],
        compact_print=True,
        step=1e-2,
        form="central",
        step_calc="abs",
        show_progress=True,
    )
else:
    print("task arg not found!")
    exit(1)
