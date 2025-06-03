#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed (unsteady)
"""

# =============================================================================
# Imports
# =============================================================================
import argparse, os
import numpy as np
from mpi4py import MPI
import json
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys.mphys_dafoam import DAFoamBuilderUnsteady, DAFoamBuilder
from pygeo.mphys import OM_DVGEOCOMP
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
args = parser.parse_args()

# Define the global parameters here
U0Cruise = 20.0
U0MaxLift = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
aoa0MaxLift = 20.0
aoa0Cruise = 3.0
CLCruise = 0.5
CLMaxLift = 1.0
A0 = 0.1
rho0 = 1.0


# Set the parameters for optimization
daOptionsCruise = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0Cruise, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0Cruise * U0Cruise * A0 * rho0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0Cruise * U0Cruise * A0 * rho0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0Cruise,
        "p": U0Cruise * U0Cruise / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inout"],
            "flowAxis": "x",
            "normalAxis": "y",
            "components": ["solver", "function"],
        },
    },
}

daOptionsMaxLift = {
    "designSurfaces": ["wing"],
    "solverName": "DAPimpleFoam",
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0MaxLift, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 50,
        "PCMatUpdateInterval": 100000,
        "reduceIO": True,
        "readZeroFields": False,
    },
    "printIntervalUnsteady": 1,
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0MaxLift * U0MaxLift * A0 * rho0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0MaxLift * U0MaxLift * A0 * rho0),
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1e-5,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": True,
        "useMGSO": True,
        "dynAdjustTol": True,
    },
    "normalizeStates": {
        "U": U0MaxLift,
        "p": U0MaxLift * U0MaxLift / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inout"],
            "flowAxis": "x",
            "normalAxis": "y",
            "components": ["solver", "function"],
        },
    },
    "checkMeshThreshold": {"maxAspectRatio": 5000.0},
    "unsteadyCompOutput": {
        "CD": ["CD"],
        "CL": ["CL"],
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane and their normals
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}


class Top(Multipoint):
    def setup(self):

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the max_lift scenario (unsteady)
        self.add_subsystem("geometry_max_lift", OM_DVGEOCOMP(file="cruise/FFD/wingFFD.xyz", type="ffd"))

        self.add_subsystem(
            "scenario_max_lift",
            DAFoamBuilderUnsteady(solver_options=daOptionsMaxLift, mesh_options=meshOptions, run_directory="maxLift"),
        )
        self.connect("geometry_max_lift.x_aero0", "scenario_max_lift.x_aero")

        # add the cruise scenario (steady)
        cruise_builder = DAFoamBuilder(daOptionsCruise, meshOptions, scenario="aerodynamic", run_directory="cruise")
        cruise_builder.initialize(self.comm)

        self.add_subsystem("mesh_cruise", cruise_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem("geometry_cruise", OM_DVGEOCOMP(file="cruise/FFD/wingFFD.xyz", type="ffd"))
        self.mphys_add_scenario("scenario_cruise", ScenarioAerodynamic(aero_builder=cruise_builder))
        self.connect("mesh_cruise.x_aero0", "geometry_cruise.x_aero_in")
        self.connect("geometry_cruise.x_aero0", "scenario_cruise.x_aero")

    def configure(self):

        # create geometric DV setup
        points_max_lift = self.scenario_max_lift.get_surface_mesh()
        points_cruise = self.mesh_cruise.mphys_get_surface_mesh()

        # add pointset
        self.geometry_max_lift.nom_add_discipline_coords("aero", points_max_lift)
        self.geometry_cruise.nom_add_discipline_coords("aero", points_cruise)

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.scenario_max_lift.DASolver.getTriangulatedMeshSurface()
        self.geometry_max_lift.nom_setConstraintSurface(tri_points)
        # no need to set the cruise constraint because the max_lift and cruise have the same airfoil geo

        # use the shape function to define shape variables for 2D airfoil
        pts = self.geometry_max_lift.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        for i in range(1, pts.shape[0] - 1):
            for j in range(pts.shape[1]):
                # k=0 and k=1 move together to ensure symmetry
                shapes.append({pts[i, j, 0]: dir_y, pts[i, j, 1]: dir_y})
        # LE/TE shape, the j=0 and j=1 move in opposite directions so that
        # the LE/TE are fixed
        for i in [0, pts.shape[0] - 1]:
            shapes.append({pts[i, 0, 0]: dir_y, pts[i, 0, 1]: dir_y, pts[i, 1, 0]: -dir_y, pts[i, 1, 1]: -dir_y})
        self.geometry_max_lift.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)
        self.geometry_cruise.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # setup the volume and thickness constraints
        leList = [[1e-4, 0.0, 1e-4], [1e-4, 0.0, 0.1 - 1e-4]]
        teList = [[0.998 - 1e-4, 0.0, 1e-4], [0.998 - 1e-4, 0.0, 0.1 - 1e-4]]
        self.geometry_max_lift.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=2, nChord=10)
        self.geometry_max_lift.nom_addVolumeConstraint("volcon", leList, teList, nSpan=2, nChord=10)
        self.geometry_max_lift.nom_addLERadiusConstraints("rcon", leList, 2, [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])
        # NOTE: we no longer need to define the sym and LE/TE constraints
        # because these constraints are defined in the above shape function

        self.dvs.add_output("shape", val=np.zeros(len(shapes)))
        self.dvs.add_output("x_aero_in", val=points_max_lift, distributed=True)
        self.dvs.add_output("patchV_cruise", val=np.array([U0Cruise, aoa0Cruise]))
        self.dvs.add_output("patchV_maxLift", val=np.array([U0MaxLift, aoa0MaxLift]))
        self.connect("x_aero_in", "geometry_max_lift.x_aero_in")
        self.connect("shape", "geometry_max_lift.shape")
        self.connect("shape", "geometry_cruise.shape")
        self.connect("patchV_maxLift", "scenario_max_lift.patchV")
        self.connect("patchV_cruise", "scenario_cruise.patchV")
        # define the design variables to the top level
        self.add_design_var("shape", lower=-0.1, upper=0.1, scaler=10.0)
        self.add_design_var("patchV_cruise", lower=[U0Cruise, 0.0], upper=[U0Cruise, 10.0], scaler=0.1)
        self.add_design_var("patchV_maxLift", lower=[U0MaxLift, 0.0], upper=[U0MaxLift, 50.0], scaler=0.1)
        self.add_objective("scenario_cruise.aero_post.CD", scaler=-1.0)
        self.add_constraint("scenario_cruise.aero_post.CL", lower=CLCruise)
        self.add_constraint("scenario_max_lift.CL", lower=CLMaxLift)
        self.add_constraint("geometry_max_lift.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry_max_lift.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry_max_lift.rcon", lower=0.8, scaler=1.0)


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
