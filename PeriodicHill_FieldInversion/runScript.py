#!/usr/bin/env python
"""
DAFoam run script for the field inversion of PH case
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic


parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: opt (default), runPrimal, runAdjoint, checkTotals
parser.add_argument("-task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================
U0 = 0.028
nuTilda0 = 1e-4
J0 = 0.0479729567  # this is the baseline L2 norm
nCells = 3500
dp0 = 6.634074021107811e-06

# Input parameters for DAFoam
daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e5,
    "primalBC": {
        "fvSource": {"value": dp0, "comp": 0},
    },
    "objFunc": {
        "FI": {
            "Ux": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "data": "UData",
                "scale": 1,
                "addToAdjoint": True,
                "weightedSum": True,
                "weight": 1 / J0,
            },
            "beta": {
                "type": "fieldInversion",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "data": "beta",
                "scale": 1e-10,
                "addToAdjoint": True,
                "weightedSum": False,
            },
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-8, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "designVar": {
        "beta": {"designVarType": "Field", "fieldName": "betaFieldInversion", "fieldType": "scalar"},
    },
}

# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(options=daOptions, mesh_options=None, scenario="aerodynamic")
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

        # define an angle of attack function to change the U direction at the far field
        def betaFieldInversion(val, DASolver):
            for idxI, v in enumerate(val):
                DASolver.setFieldValue4GlobalCellI(b"betaFieldInversion", v, idxI)
                DASolver.updateBoundaryConditions(b"betaFieldInversion", b"scalar")

        # pass this aoa function to the cruise group
        self.cruise.coupling.solver.add_dv_func("beta", betaFieldInversion)
        self.cruise.aero_post.add_dv_func("beta", betaFieldInversion)

        # add the design variables to the dvs component's output
        self.dvs.add_output("beta", val=np.array([1] * nCells))
        self.connect("beta", "cruise.beta")

        # define the design variables to the top level
        self.add_design_var("beta", lower=-5.0, upper=10.0, scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("cruise.aero_post.FI", scaler=1.0)


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
        "Nonderivative linesearch": None,
        "Linesearch tolerance": 0.999,
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
else:
    print("optimizer arg not valid!")
    exit(1)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.options["print_opt_prob"] = False
prob.driver.hist_file = "OptView.hst"

if args.task == "opt":
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
        of=["cruise.aero_post.functionals.FI"],
        wrt=["beta"],
        compact_print=False,
        show_progress=True,
        step=1e-3,
        form="central",
        step_calc="abs",
    )
else:
    print("task arg not found!")
    exit(1)
