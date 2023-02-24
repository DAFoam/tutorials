#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed (multipoint)
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
from pygeo.mphys  import OM_DVGEOCOMP
from pygeo import geo_utils


parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: opt (default), runPrimal, runAdjoint, checkTotals
parser.add_argument("-task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================
# we have two flight conditions
U0 = [10.0, 8.0]
p0 = 0.0
nuTilda0 = 4.5e-5
CL_target = [0.5, 0.6]
aoa0 = [5.0, 6.0]
A0 = 0.1
# rho is used for normalizing CD and CL
rho0 = 1.0

# define the BC and objFunc dicts for each flight condition
fc0 = {"primalBC": {"U0": {"value": [U0[0], 0.0, 0.0]}},
       "objFunc": {"CD": {"part1": {"scale": 1.0 / (0.5 * U0[0] * U0[0] * A0 * rho0)}},
                   "CL": {"part1": {"scale": 1.0 / (0.5 * U0[0] * U0[0] * A0 * rho0)}}}}
fc1 = {"primalBC": {"U0": {"value": [U0[1], 0.0, 0.0]}},
       "objFunc": {"CD": {"part1": {"scale": 1.0 / (0.5 * U0[1] * U0[1] * A0 * rho0)}},
                   "CL": {"part1": {"scale": 1.0 / (0.5 * U0[1] * U0[1] * A0 * rho0)}}}}

# Input parameters for DAFoam
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0[0], 0.0, 0.0]},
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
                # the scale here is not important because we will replace it with
                # the values defined in fc0 and fc1 later
                "scale": 1.0 / (0.5 * U0[0] * U0[0] * A0 * rho0),
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
                "scale": 1.0 / (0.5 * U0[0] * U0[0] * A0 * rho0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0[0],
        "p": U0[0] * U0[0] / 2.0,
        "nuTilda": nuTilda0 * 10.0,
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

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/wingFFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("cruise0", ScenarioAerodynamic(aero_builder=dafoam_builder))
        self.mphys_add_scenario("cruise1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        # need to manually connect the x_aero0 between the geometry component and the cruise
        # scenario group
        self.connect("geometry.x_aero0", "cruise0.x_aero")
        self.connect("geometry.x_aero0", "cruise1.x_aero")

        # add an exec comp to average two drags, the weights are 0.5 and 0.5
        self.add_subsystem("obj", om.ExecComp("CD_AVG=0.5*CD0+0.5*CD1"))

    def configure(self):
        # configure and setup perform a similar function, i.e., initialize the optimization.
        # But configure will be run after setup

        # we set the fc (flight conditions) to each cruise conditions
        self.cruise0.coupling.mphys_set_options(fc0)
        self.cruise0.aero_post.mphys_set_options(fc0)

        self.cruise1.coupling.mphys_set_options(fc1)
        self.cruise1.aero_post.mphys_set_options(fc1)

        # add the objective function to the cruise scenario
        self.cruise0.aero_post.mphys_add_funcs()
        self.cruise1.aero_post.mphys_add_funcs()

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # define an angle of attack function to change the U direction at the far field
        # here the function is different from the single point, we only change the flow
        # direction, not its magnitude
        def aoa(val, DASolver):
            aoa = float(val[0] * np.pi / 180.0)
            U = DASolver.getOption("primalBC")["U0"]["value"]
            UAll = np.sqrt(U[0] ** 2 + U[1] ** 2 + U[2] ** 2)
            U = [float(UAll * np.cos(aoa)), float(UAll * np.sin(aoa)), 0]
            DASolver.setOption("primalBC", {"U0": {"value": U}})
            DASolver.updateDAOption()

        # pass this aoa function to the cruise group. we need to do it for each condition
        self.cruise0.coupling.solver.add_dv_func("aoa", aoa)
        self.cruise0.aero_post.add_dv_func("aoa", aoa)
        self.cruise1.coupling.solver.add_dv_func("aoa", aoa)
        self.cruise1.aero_post.add_dv_func("aoa", aoa)

        # select the FFD points to move
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[:, :, :].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS)

        # setup the symmetry constraint to link the y displacement between k=0 and k=1
        nFFDs_x = pts.shape[0]
        nFFDs_y = pts.shape[1]
        indSetA = []
        indSetB = []
        for i in range(nFFDs_x):
            for j in range(nFFDs_y):
                indSetA.append(pts[i, j, 0])
                indSetB.append(pts[i, j, 1])
        self.geometry.nom_addLinearConstraintsShape("linearcon", indSetA, indSetB, factorA=1.0, factorB=-1.0)

        # setup the volume and thickness constraints
        leList = [[1e-4, 0.0, 1e-4], [1e-4, 0.0, 0.1 - 1e-4]]
        teList = [[0.998 - 1e-4, 0.0, 1e-4], [0.998 - 1e-4, 0.0, 0.1 - 1e-4]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=2, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=2, nChord=10)
        # LE/TE constrants
        self.geometry.nom_add_LETEConstraint("lecon", volID=0, faceID="iLow", topID="k")
        self.geometry.nom_add_LETEConstraint("tecon", volID=0, faceID="iHigh", topID="k")

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        # NOTE: we have two separated aoa variables for the two flight conditions
        self.dvs.add_output("aoa0", val=np.array([aoa0[0]]))
        self.dvs.add_output("aoa1", val=np.array([aoa0[1]]))
        # manually connect the dvs output to the geometry and cruise
        self.connect("aoa0", "cruise0.aoa")
        self.connect("aoa1", "cruise1.aoa")
        self.connect("shape", "geometry.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        self.add_design_var("aoa0", lower=0.0, upper=10.0, scaler=1.0)
        self.add_design_var("aoa1", lower=0.0, upper=10.0, scaler=1.0)

        # add objective and constraints to the top level
        # we have two separated lift constraints for for the two flight conditions
        self.add_constraint("cruise0.aero_post.CL", equals=CL_target[0], scaler=1.0)
        self.add_constraint("cruise1.aero_post.CL", equals=CL_target[1], scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.linearcon", equals=0.0, scaler=1.0, linear=True)

        # here we use the CD_AVG defined above as the obj func.
        self.add_objective("obj.CD_AVG", scaler=1.0)
        self.connect("cruise0.aero_post.CD", "obj.CD0")
        self.connect("cruise1.aero_post.CD", "obj.CD1")


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


if args.task == "opt":
    # solve CL
    optFuncs.findFeasibleDesign(["cruise0.aero_post.CL", "cruise1.aero_post.CL"], ["aoa0", "aoa1"], targets=CL_target)
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
        of=["cruise.aero_post.CD", "cruise.aero_post.CL"], wrt=["shape", "aoa"], compact_print=True, step=1e-3, form="central", step_calc="abs"
    )
else:
    print("task arg not found!")
    exit(1)
