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

nCells = 31080

U0 = 68.0
p0 = 101325.0
T0 = 288.0
rho0 = p0 / T0 / 287.0
nuTilda0 = 1e-4
k0 = 0.2
omega0 = 1000.0
A0 = 0.01

aoa = 14.0  # [4.0, 14.0, 20.0]
CLData = 1.0478  # [0.6035, 1.0478, 0.9219]

aoaRad = aoa * np.pi / 180.0
inletU = [float(U0 * np.cos(aoaRad)), float(U0 * np.sin(aoaRad)), 0]
flowDir = [float(np.cos(aoaRad)), float(np.sin(aoaRad)), 0.0]
normalDir = [-float(np.sin(aoaRad)), float(np.cos(aoaRad)), 0.0]

# Input parameters for DAFoam
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e6,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": inletU},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "k0": {"variable": "k", "patches": ["inout"], "value": [k0]},
        "omega0": {"variable": "omega", "patches": ["inout"], "value": [omega0]},
        "thermo:mu": 3.4e-5,
        "useWallFunction": False,
    },
    "primalVarBounds": {"omegaMin": -1e16},
    "function": {
        "CLError": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "fixedDirection",
            "direction": normalDir,
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
            "calcRefVar": True,
            "ref": [CLData],
        },
        "betaVar": {
            "type": "variance",
            "source": "boxToCell",
            "min": [-100.0, -100.0, -100.0],
            "max": [100.0, 100.0, 100.0],
            "scale": 1.0,
            "mode": "field",
            "varName": "betaFINuTilda",
            "varType": "scalar",
                "timeDependentRefData": False,
                "timeDependentRefData": False,
                "addToAdjoint": True,
            },
            "timeDependentRefData": False,
                "addToAdjoint": True,
            },
        },
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "fixedDirection",
            "direction": flowDir,
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "fixedDirection",
            "direction": normalDir,
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-6,
        "pcFillLevel": 2,
        "jacMatReOrdering": "natural",
        "gmresMaxIters": 3000,
        "gmresRestart": 3000,
    },
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "checkMeshThreshold": {"maxAspectRatio": 10000.0},
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
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # setup a composite objective
        self.add_subsystem("obj", om.ExecComp("val=error+regulation"))

    def configure(self):
        # add the design variables to the dvs component's output
        self.dvs.add_output("beta", val=np.ones(nCells), distributed=False)
        self.connect("beta", "scenario1.beta")

        # define the design variables to the top level
        self.add_design_var("beta", lower=-5.0, upper=10.0, scaler=1.0)

        # add objective and constraints to the top level
        # we can connect any function in daOption to obj's terms
        self.connect("scenario1.aero_post.CLError", "obj.error")
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
