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
# which task to run. Options are: opt (default), runPrimal, runAdjoint, checkTotals
parser.add_argument("-task", help="type of run to do", type=str, default="opt")
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
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "designVar": {
        "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
        "shape": {"designVarType": "FFD"},
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
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
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
    "designVar": {
        "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
        "shape": {"designVarType": "FFD"},
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
        self.mphys_add_scenario("cruise_sa", ScenarioAerodynamic(aero_builder=dafoam_builder_sa))
        self.mphys_add_scenario("cruise_sst", ScenarioAerodynamic(aero_builder=dafoam_builder_sst))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh_sa.x_aero0", "geometry_sa.x_aero_in")
        self.connect("geometry_sa.x_aero0", "cruise_sa.x_aero")

        self.connect("mesh_sst.x_aero0", "geometry_sst.x_aero_in")
        self.connect("geometry_sst.x_aero0", "cruise_sst.x_aero")

        self.add_subsystem("obj", om.ExecComp("value=(cd_sa+cd_sst)/2"))

    def configure(self):
        # configure and setup perform a similar function, i.e., initialize the optimization.
        # But configure will be run after setup

        # add the objective function to the cruise scenario
        self.cruise_sa.aero_post.mphys_add_funcs()
        self.cruise_sst.aero_post.mphys_add_funcs()

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

        # define an angle of attack function to change the U direction at the far field
        # the sa and sst can share the same function
        def aoa(val, DASolver):
            aoa = val[0] * np.pi / 180.0
            U = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
            # we need to update the U value only
            DASolver.setOption("primalBC", {"U0": {"value": U}})
            DASolver.updateDAOption()

        # pass this aoa function to the cruise group
        self.cruise_sa.coupling.solver.add_dv_func("aoa", aoa)
        self.cruise_sa.aero_post.add_dv_func("aoa", aoa)

        # pass this aoa function to the cruise group
        self.cruise_sst.coupling.solver.add_dv_func("aoa", aoa)
        self.cruise_sst.aero_post.add_dv_func("aoa", aoa)

        # select the FFD points to move
        # pts is same for both cases (same FFD), so we can reuse
        pts = self.geometry_sa.DVGeo.getLocalIndex(0)

        nShapes = self.geometry_sa.nom_addLocalDV(dvName="shape")
        nShapes = self.geometry_sst.nom_addLocalDV(dvName="shape")

        # setup the symmetry constraint to link the y displacement between k=0 and k=1
        nFFDs_x = pts.shape[0]
        nFFDs_y = pts.shape[1]
        indSetA = []
        indSetB = []
        for i in range(nFFDs_x):
            for j in range(nFFDs_y):
                indSetA.append(pts[i, j, 0])
                indSetB.append(pts[i, j, 1])
        # indSet are same between sa and sst, we don't need to repeat
        self.geometry_sa.nom_addLinearConstraintsShape("linearcon", indSetA, indSetB, factorA=1.0, factorB=-1.0)

        # setup the volume and thickness constraints
        leList = [[1e-4, 0.0, 1e-4], [1e-4, 0.0, 0.1 - 1e-4]]
        teList = [[0.998 - 1e-4, 0.0, 1e-4], [0.998 - 1e-4, 0.0, 0.1 - 1e-4]]
        self.geometry_sa.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=2, nChord=10)
        self.geometry_sa.nom_addVolumeConstraint("volcon", leList, teList, nSpan=2, nChord=10)
        # add the LE/TE constraints
        self.geometry_sa.nom_add_LETEConstraint("lecon", volID=0, faceID="iLow", topID="k")
        self.geometry_sa.nom_add_LETEConstraint("tecon", volID=0, faceID="iHigh", topID="k")

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("aoa_sa", val=np.array([aoa0]))
        self.dvs.add_output("aoa_sst", val=np.array([aoa0]))
        # manually connect the dvs output to the geometry and cruise
        # sa and sst cases share the same shape
        self.connect("aoa_sa", "cruise_sa.aoa")
        self.connect("shape", "geometry_sa.shape")
        self.connect("aoa_sst", "cruise_sst.aoa")
        self.connect("shape", "geometry_sst.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        self.add_design_var("aoa_sa", lower=0.0, upper=10.0, scaler=1.0)
        self.add_design_var("aoa_sst", lower=0.0, upper=10.0, scaler=1.0)

        # add objective and constraints to the top level
        self.connect("cruise_sa.aero_post.CD", "obj.cd_sa")
        self.connect("cruise_sst.aero_post.CD", "obj.cd_sst")

        self.add_objective("obj.value", scaler=1.0)

        self.add_constraint("cruise_sa.aero_post.CL", equals=CL_target, scaler=1.0)
        self.add_constraint("cruise_sst.aero_post.CL", equals=CL_target, scaler=1.0)

        self.add_constraint("geometry_sa.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry_sa.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry_sa.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry_sa.lecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry_sa.linearcon", equals=0.0, scaler=1.0, linear=True)


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

if args.task == "opt":
    # solve CL
    optFuncs.findFeasibleDesign(
        ["cruise_sa.aero_post.CL", "cruise_sst.aero_post.CL"], ["aoa_sa", "aoa_sst"], targets=[CL_target, CL_target]
    )
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
        of=["cruise.aero_post.CD", "cruise.aero_post.CL"],
        wrt=["shape", "aoa"],
        compact_print=True,
        step=1e-3,
        form="central",
        step_calc="abs",
    )
else:
    print("task arg not found!")
    exit(1)
