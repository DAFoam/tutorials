#!/usr/bin/env python
"""
DAFoam run script for the multi-element airfoil at subsonic conditions
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from pyspline import Curve
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
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
U0 = 68.0
p0 = 101325.0
T0 = 300.0
nuTilda0 = 4.5e-5
CL_target = 3.416
aoa0 = 12.92958
A0 = 0.1
# rho is used for normalizing CD and CL
rho0 = p0 / T0 / 287

# Input parameters for DAFoam
daOptions = {
    "designSurfaces": ["main", "slat", "flap"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["main", "slat", "flap"],
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
                "patches": ["main", "slat", "flap"],
                "directionMode": "normalToFlow",
                "alphaName": "aoa",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
        "skewness": {
            "part1": {
                "type": "meshQualityKS",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "coeffKS": 20.0,
                "metric": "faceSkewness",
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
        "nonOrtho": {
            "part1": {
                "type": "meshQualityKS",
                "source": "boxToCell",
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "coeffKS": 1.0,
                "metric": "nonOrthoAngle",
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "checkMeshThreshold": {"maxAspectRatio": 2000.0, "maxNonOrth": 75.0, "maxSkewness": 8.0},
    "designVar": {
        "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
        "shape": {"designVarType": "FFD"},
        "twistslat": {"designVarType": "FFD"},
        "translateslat": {"designVarType": "FFD"},
        "twistflap": {"designVarType": "FFD"},
        "translateflap": {"designVarType": "FFD"},
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/airfoilFFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        # need to manually connect the x_aero0 between the geometry component and the cruise
        # scenario group
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):
        # configure and setup perform a similar function, i.e., initialize the optimization.
        # But configure will be run after setup

        # add the objective function to the cruise scenario
        self.cruise.aero_post.mphys_add_funcs()

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # Create reference axis for the twist variables
        xSlat = [0.0169, 0.0169]
        ySlat = [0.0034, 0.0034]
        zSlat = [0.0, 0.1]
        cSlat = Curve(x=xSlat, y=ySlat, z=zSlat, k=2)
        # Note here we set raySize=5 to avoid the warning when having highly skewed FFDs
        # "ray might not have been longenough to intersect the nearest curve."
        self.geometry.nom_addRefAxis(name="slatAxis", curve=cSlat, axis="z", volumes=[0], raySize=5)

        xFlap = [0.875, 0.875]
        yFlap = [0.014, 0.014]
        zFlap = [0.0, 0.1]
        cFlap = Curve(x=xFlap, y=yFlap, z=zFlap, k=2)
        self.geometry.nom_addRefAxis(name="flapAxis", curve=cFlap, axis="z", volumes=[2], raySize=5)

        # define an angle of attack function to change the U direction at the far field
        def aoa(val, DASolver):
            aoa = val[0] * np.pi / 180.0
            U = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
            # we need to update the U value only
            DASolver.setOption("primalBC", {"U0": {"value": U}})
            DASolver.updateDAOption()

        def twistslat(val, geo):
            for i in range(2):
                geo.rot_z["slatAxis"].coef[i] = -val[0]

        def translateslat(val, geo):
            C = geo.extractCoef("slatAxis")
            dx = val[0]
            dy = val[1]
            for i in range(len(C)):
                C[i, 0] = C[i, 0] + dx
            for i in range(len(C)):
                C[i, 1] = C[i, 1] + dy
            geo.restoreCoef(C, "slatAxis")

        def twistflap(val, geo):
            for i in range(2):
                geo.rot_z["flapAxis"].coef[i] = -val[0]

        def translateflap(val, geo):
            C = geo.extractCoef("flapAxis")
            dx = val[0]
            dy = val[1]
            for i in range(len(C)):
                C[i, 0] = C[i, 0] + dx
            for i in range(len(C)):
                C[i, 1] = C[i, 1] + dy
            geo.restoreCoef(C, "flapAxis")

        # pass this aoa function to the cruise group
        self.cruise.coupling.solver.add_dv_func("aoa", aoa)
        self.cruise.aero_post.add_dv_func("aoa", aoa)

        # add the global shape variable
        self.geometry.nom_addGlobalDV(dvName="twistslat", value=[0.0], func=twistslat)
        self.geometry.nom_addGlobalDV(dvName="translateslat", value=np.zeros(2), func=translateslat)
        self.geometry.nom_addGlobalDV(dvName="twistflap", value=[0.0], func=twistflap)
        self.geometry.nom_addGlobalDV(dvName="translateflap", value=np.zeros(2), func=translateflap)

        # select the FFD points to move
        pts = self.geometry.DVGeo.getLocalIndex(1)
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
        leListMain = [[0.048, -0.014, 1e-6], [0.048, -0.014, 0.1 - 1e-6]]
        teListMain = [[0.698, -0.014, 1e-6], [0.698, -0.014, 0.1 - 1e-6]]
        self.geometry.nom_addThicknessConstraints2D("thickcon_main", leListMain, teListMain, nSpan=2, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon_main", leListMain, teListMain, nSpan=2, nChord=10)
        # NOTE: we need to add thickness and vol constraints for the tailing of the main airfoil
        leListMainTrailing = [[0.702, 0.0328, 1e-6], [0.702, 0.0328, 0.1 - 1e-6]]
        teListMainTrailing = [[0.854, 0.0328, 1e-6], [0.854, 0.0328, 0.1 - 1e-6]]
        self.geometry.nom_addThicknessConstraints2D(
            "thickcon_main_te", leListMainTrailing, teListMainTrailing, nSpan=2, nChord=10
        )
        self.geometry.nom_addVolumeConstraint(
            "volcon_main_te", leListMainTrailing, teListMainTrailing, nSpan=2, nChord=10
        )

        # add the LE/TE constraints
        self.geometry.nom_add_LETEConstraint("lecon", volID=1, faceID="iLow", topID="k")
        self.geometry.nom_add_LETEConstraint("tecon", volID=1, faceID="iHigh", topID="k")

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("aoa", val=np.array([aoa0]))
        self.dvs.add_output("twistslat", val=np.array([0.0]))
        self.dvs.add_output("translateslat", val=np.zeros(2))
        self.dvs.add_output("twistflap", val=np.array([0.0]))
        self.dvs.add_output("translateflap", val=np.zeros(2))
        # manually connect the dvs output to the geometry and cruise
        self.connect("aoa", "cruise.aoa")
        self.connect("shape", "geometry.shape")
        self.connect("twistslat", "geometry.twistslat")
        self.connect("translateslat", "geometry.translateslat")
        self.connect("twistflap", "geometry.twistflap")
        self.connect("translateflap", "geometry.translateflap")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        self.add_design_var("aoa", lower=0.0, upper=20.0, scaler=1.0)
        self.add_design_var("twistslat", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("translateslat", lower=[-0.1, 0.0], upper=[0.0, 0.1], scaler=1.0)
        self.add_design_var("twistflap", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("translateflap", lower=[0.0, -0.1], upper=[0.1, 0.0], scaler=1.0)

        # add objective and constraints to the top level
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=CL_target, scaler=1.0)
        self.add_constraint("cruise.aero_post.skewness", upper=6.0, scaler=1.0)
        self.add_constraint("cruise.aero_post.nonOrtho", upper=70.0, scaler=1.0)
        self.add_constraint("geometry.thickcon_main", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon_main", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.thickcon_main_te", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon_main_te", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.linearcon", equals=0.0, scaler=1.0, linear=True)


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
    optFuncs.findFeasibleDesign(["cruise.aero_post.CL"], ["aoa"], targets=[CL_target])
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
