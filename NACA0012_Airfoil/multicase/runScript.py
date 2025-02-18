#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed (multicase)
Here we optimize the case using both SA and SST models and the objective func
is the averaged drag between SA and SST
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
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
k0 = 0.015
omega0 = 100.0
CL_target = 0.5
aoa0 = 5.0
A0 = 0.1
rho0 = 1.0

# Input parameters for DAFoam
daOptionsSA = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
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
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
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

daOptionsSST = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "k0": {"variable": "k", "patches": ["inout"], "value": [k0]},
        "omega0": {"variable": "omega", "patches": ["inout"], "value": [omega0]},
        "useWallFunction": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "k": k0,
        "omega": omega0,
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

# Mesh deformation setup
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers for both cases (they share the same mesh option)
        dafoam_builder_sa = DAFoamBuilder(daOptionsSA, meshOptions, scenario="aerodynamic", run_directory="SA")
        dafoam_builder_sa.initialize(self.comm)

        dafoam_builder_sst = DAFoamBuilder(daOptionsSST, meshOptions, scenario="aerodynamic", run_directory="SST")
        dafoam_builder_sst.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh_sa", dafoam_builder_sa.get_mesh_coordinate_subsystem())
        self.add_subsystem("mesh_sst", dafoam_builder_sst.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD)
        self.add_subsystem("geometry_sa", OM_DVGEOCOMP(file="SA/FFD/wingFFD.xyz", type="ffd"))
        self.add_subsystem("geometry_sst", OM_DVGEOCOMP(file="SST/FFD/wingFFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("scenario_sa", ScenarioAerodynamic(aero_builder=dafoam_builder_sa))
        self.mphys_add_scenario("scenario_sst", ScenarioAerodynamic(aero_builder=dafoam_builder_sst))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh_sa.x_aero0", "geometry_sa.x_aero_in")
        self.connect("geometry_sa.x_aero0", "scenario_sa.x_aero")

        self.connect("mesh_sst.x_aero0", "geometry_sst.x_aero_in")
        self.connect("geometry_sst.x_aero0", "scenario_sst.x_aero")

        self.add_subsystem("obj", om.ExecComp("value=(cd_sa+cd_sst)/2"))

    def configure(self):

        # get the surface coordinates from the mesh component
        points_sa = self.mesh_sa.mphys_get_surface_mesh()
        points_sst = self.mesh_sst.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry_sa.nom_add_discipline_coords("aero", points_sa)
        self.geometry_sst.nom_add_discipline_coords("aero", points_sst)

        # set the triangular points to the geometry component for geometric constraints
        tri_points_sa = self.mesh_sa.mphys_get_triangulated_surface()
        self.geometry_sa.nom_setConstraintSurface(tri_points_sa)

        tri_points_sst = self.mesh_sst.mphys_get_triangulated_surface()
        self.geometry_sst.nom_setConstraintSurface(tri_points_sst)

        # use the shape function to define shape variables for 2D airfoil
        pts = self.geometry_sa.DVGeo.getLocalIndex(0)
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
        self.geometry_sa.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)
        self.geometry_sst.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # setup the volume and thickness constraints
        leList = [[1e-4, 0.0, 1e-4], [1e-4, 0.0, 0.1 - 1e-4]]
        teList = [[0.998 - 1e-4, 0.0, 1e-4], [0.998 - 1e-4, 0.0, 0.1 - 1e-4]]
        self.geometry_sa.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=2, nChord=10)
        self.geometry_sa.nom_addVolumeConstraint("volcon", leList, teList, nSpan=2, nChord=10)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * len(shapes)))
        self.dvs.add_output("patchV_sa", val=np.array([U0, aoa0]))
        self.dvs.add_output("patchV_sst", val=np.array([U0, aoa0]))
        # manually connect the dvs output to the geometry and cruise
        # sa and sst cases share the same shape
        self.connect("patchV_sa", "scenario_sa.patchV")
        self.connect("shape", "geometry_sa.shape")
        self.connect("patchV_sst", "scenario_sst.patchV")
        self.connect("shape", "geometry_sst.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=10.0)
        self.add_design_var("patchV_sa", lower=[U0, 0.0], upper=[U0, 10.0], scaler=0.1)
        self.add_design_var("patchV_sst", lower=[U0, 0.0], upper=[U0, 10.0], scaler=0.1)

        # add objective and constraints to the top level
        self.connect("scenario_sa.aero_post.CD", "obj.cd_sa")
        self.connect("scenario_sst.aero_post.CD", "obj.cd_sst")
        self.add_objective("obj.value", scaler=1.0)

        self.add_constraint("scenario_sa.aero_post.CL", equals=CL_target, scaler=1.0)
        self.add_constraint("scenario_sst.aero_post.CL", equals=CL_target, scaler=1.0)
        self.add_constraint("geometry_sa.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry_sa.volcon", lower=1.0, scaler=1.0)


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys.html")

# initialize the optimization function
optFuncs = OptFuncs([daOptionsSA, daOptionsSST], prob)

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
    # solve CL
    optFuncs.findFeasibleDesign(
        ["scenario_sa.aero_post.CL", "scenario_sst.aero_post.CL"],
        ["patchV_sa", "patchV_sst"],
        designVarsComp=[1, 1],
        targets=[CL_target, CL_target],
    )
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
