#!/usr/bin/env python
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


# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: run_driver (default), run_model, compute_totals, check_totals
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()

vms0 = 2.9e4

# Set the parameters for optimization
daOptions = {
    "maxTractionBCIters": 20,
    "solverName": "DASolidDisplacementFoam",
    "designSurfaces": ["hole", "wallx", "wally"],
    "primalMinResTol": 1e-10,
    "primalMinResTolDiff": 1e10,
    "function": {
        "VMS": {
            "type": "vonMisesStressKS",
            "source": "allCells",
            "scale": 1.0,
            "coeffKS": 2.0e-3,
        },
        "M": {
            "type": "variableVolSum",
            "source": "allCells",
            "varName": "solid:rho",
            "varType": "scalar",
            "index": 0,
            "isSquare": 0,
            "multiplyVol": 1,
            "divByTotalVol": 0,
            "scale": 1.0,
        },
    },
    "normalizeStates": {"D": 1.0e-7},
    "adjEqnOption": {"gmresRelTol": 1.0e-10, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "adjPCLag": 20,
    # Design variable setup
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
    },
}

# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/plateFFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        # need to manually connect the x_aero0 between the geometry component and the scenario1
        # scenario group
        self.connect("geometry.x_aero0", "scenario1.x_aero")

    def configure(self):

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # use the shape function to define shape variables for 2D airfoil
        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_x = np.array([1.0, 0.0, 0.0])
        dir_y = np.array([0.0, 1.0, 0.0])
        shapesX = []
        shapesY = []
        for i in range(2, 5):
            for j in [2, 4]:
                # k=0 and k=1 move together to ensure symmetry
                shapesY.append({pts[i, j, 0]: dir_y, pts[i, j, 1]: dir_y})
        self.geometry.nom_addShapeFunctionDV(dvName="shapeY", shapes=shapesY)
        for i in [2, 4]:
            for j in range(2, 5):
                # k=0 and k=1 move together to ensure symmetry
                shapesX.append({pts[i, j, 0]: dir_x, pts[i, j, 1]: dir_x})
        self.geometry.nom_addShapeFunctionDV(dvName="shapeX", shapes=shapesX)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shapeX", val=np.array([0] * len(shapesX)))
        self.dvs.add_output("shapeY", val=np.array([0] * len(shapesY)))
        # manually connect the dvs output to the geometry and scenario1
        self.connect("shapeX", "geometry.shapeX")
        self.connect("shapeY", "geometry.shapeY")

        # define the design variables to the top level
        self.add_design_var("shapeX", lower=-1.0, upper=1.0, scaler=10.0)
        self.add_design_var("shapeY", lower=-1.0, upper=1.0, scaler=10.0)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.M", scaler=1.0)
        self.add_constraint("scenario1.aero_post.VMS", upper=vms0, scaler=1.0)


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
