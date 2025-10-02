#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
import numpy as np
import json
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP


parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: run_driver (default), run_model, compute_totals, check_totals
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================
U0 = 0.01
p0 = 0.0
nCells = 6400
PLRef = 0.0001
TRef = 300.0
dTRef = 3.0
w = 0.2

# Input parameters for DAFoam
daOptions = {
    "solverName": "DATopoChtFoam",
    "primalMinResTol": 1.0e-8,
    "function": {
        "TP1": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["inlet"],
            "scale": 1.0,
        },
        "TP2": {
            "type": "totalPressure",
            "source": "patchToFace",
            "patches": ["outlet"],
            "scale": 1.0,
        },
        "TOut": {
            "type": "patchMean",
            "source": "patchToFace",
            "patches": ["outlet"],
            "varName": "T",
            "varType": "scalar",
            "index": 0,
            "scale": 1.0,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-5,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "gmresMaxIters": 2000,
        "gmresRestart": 2000,
        "gmresTolDiff": 1.0e3,
    },
    "normalizeStates": {
        "U": U0,
        "p": 0.1,
        "phi": 1.0,
        "T": 300.0,
    },
    "adjPCLag": 1,
    "inputInfo": {
        "eta": {
            "type": "field",
            "fieldName": "eta",
            "fieldType": "scalar",
            "components": ["solver", "function"],
            "distributed": 0,
        },
    },
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(daOptions, None, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # add an exec comp to compute obj
        self.add_subsystem(
            "obj",
            om.ExecComp(
                "val=w*(TP1-TP2)/PLRef-(1-w)*(TOut-TRef)/dTRef",
                w={"val": w, "constant": True},
                PLRef={"val": PLRef, "constant": True},
                TRef={"val": TRef, "constant": True},
            ),
        )

    def configure(self):

        # add the design variables to the dvs component's output
        self.dvs.add_output("eta", val=np.ones(nCells))
        # manually connect the dvs output to the geometry and scenario1
        self.connect("eta", "scenario1.eta")

        # define the design variables to the top level
        self.add_design_var("eta", lower=0, upper=1, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("obj.val", scaler=1.0)
        self.connect("scenario1.aero_post.TP1", "obj.TP1")
        self.connect("scenario1.aero_post.TP2", "obj.TP2")
        self.connect("scenario1.aero_post.TOut", "obj.TOut")


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys.html")

# initialize the optimization function
optFuncs = OptFuncs(daOptions, prob)

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
        "Linesearch tolerance": 0.9999,
        "Hessian updates": 10000,
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
        "bound_frac": 1e-5, # allow IPOPT to use init dv closer to upper bound
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
