#!/usr/bin/env python
"""
DAFoam run script for the NACA2412 airfoil at low-speed
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
from dafoam.mphys import DAFoamBuilder, OptFuncs, DAFoamLinearConstraint, DAFoamVSPVolume
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
CL_target = 0.5
aoa0 = 2.0
A0 = 0.1
# rho is used for normalizing CD and CL
rho0 = 1.0
n_cst_coeffs = 7

# Input parameters for DAFoam
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-7,
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
    "checkMeshThreshold": {"maxNonOrth": 70.0, "maxSkewness": 20.0, "maxAspectRatio": 5000.0},
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="airfoil.vsp3", type="vsp"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        # need to manually connect the x_aero0 between the geometry component and the scenario1
        # scenario group
        self.connect("geometry.x_aero0", "scenario1.x_aero")

        # thickness constraint
        varA = []
        varB = []
        for j in range(n_cst_coeffs):
            varA.append(f"UpperCoeff_{j}")
            varB.append(f"LowerCoeff_{j}")
        self.add_subsystem(
            "thickness",
            DAFoamLinearConstraint(varA=varA, coeffA=1.0, varB=varB, coeffB=-1.0, size=1, output_name="thickness_val"),
            promotes=["*"],
        )

        # LE C1 continuity constraint
        self.add_subsystem(
            "le_c1",
            DAFoamLinearConstraint(
                varA=["UpperCoeff_0"], coeffA=1.0, varB=["LowerCoeff_0"], coeffB=1.0, size=1, output_name="le_c1_val"
            ),
            promotes=["*"],
        )

        # add volume constraint
        vsp_vars = []
        for i in range(2):
            for j in range(n_cst_coeffs):
                vsp_vars.append(f"NACA:UpperCoeff_{i}:Au_{j}")
                vsp_vars.append(f"NACA:LowerCoeff_{i}:Al_{j}")
        self.add_subsystem(
            "volume",
            DAFoamVSPVolume(
                vsp_file="airfoil.vsp3",
                vsp_vars=vsp_vars,
                slice_dir="x",
                n_slices=10,
                output_name="volume_val",
                step=1e-3,
                relativeStep=False,
            ),
        )

    def configure(self):

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        for i in range(2):
            for j in range(n_cst_coeffs):
                self.geometry.nom_addVSPVariable("NACA", f"UpperCoeff_{i}", f"Au_{j}", scaledStep=False, dh=1e-3)
                self.geometry.nom_addVSPVariable("NACA", f"LowerCoeff_{i}", f"Al_{j}", scaledStep=False, dh=1e-3)

        # add the design variables to the dvs component's output
        # these are for NACA0012
        CST = np.array([0.17299, 0.15121, 0.16626, 0.13844, 0.14289, 0.13999, 0.14070])
        for j in range(n_cst_coeffs):
            self.dvs.add_output(f"UpperCoeff_{j}", val=CST[j])
            self.connect(f"UpperCoeff_{j}", f"geometry.NACA:UpperCoeff_0:Au_{j}")
            self.connect(f"UpperCoeff_{j}", f"geometry.NACA:UpperCoeff_1:Au_{j}")
            self.connect(f"UpperCoeff_{j}", f"volume.NACA:UpperCoeff_0:Au_{j}")
            self.connect(f"UpperCoeff_{j}", f"volume.NACA:UpperCoeff_1:Au_{j}")
            self.dvs.add_output(f"LowerCoeff_{j}", val=-CST[j])
            self.connect(f"LowerCoeff_{j}", f"geometry.NACA:LowerCoeff_0:Al_{j}")
            self.connect(f"LowerCoeff_{j}", f"geometry.NACA:LowerCoeff_1:Al_{j}")
            self.connect(f"LowerCoeff_{j}", f"volume.NACA:LowerCoeff_0:Al_{j}")
            self.connect(f"LowerCoeff_{j}", f"volume.NACA:LowerCoeff_1:Al_{j}")

        # define the design variables to the top level
        for j in range(n_cst_coeffs):
            self.add_design_var(f"UpperCoeff_{j}", lower=-1.0, upper=1.0, scaler=1.0)
            self.add_design_var(f"LowerCoeff_{j}", lower=-1.0, upper=1.0, scaler=1.0)

        self.dvs.add_output("patchV", val=np.array([U0, aoa0]))
        self.connect("patchV", "scenario1.patchV")
        self.add_design_var("patchV", lower=[U0, 0.0], upper=[U0, 10.0], scaler=0.1)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.CD", scaler=1.0)
        self.add_constraint("scenario1.aero_post.CL", equals=CL_target, scaler=1.0)

        # volume constraint
        self.add_constraint("volume.volume_val", lower=1.0, scaler=1.0)

        # 50% thickness
        for j in range(1, n_cst_coeffs):
            self.add_constraint(f"thickness_val_{j}", lower=0.5 * 2.0 * CST[j], scaler=1.0, linear=True)
        # LE radius does not change
        self.add_constraint("thickness_val_0", lower=2.0 * CST[0], scaler=1.0, linear=True)
        # LE C1 continuous
        self.add_constraint("le_c1_val_0", equals=0.0, scaler=1.0, linear=True)


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
    # solve CL
    optFuncs.findFeasibleDesign(["scenario1.aero_post.CL"], ["patchV"], targets=[CL_target], designVarsComp=[1])
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
    prob.check_totals(compact_print=False, step=1e-2, form="central", step_calc="abs")
else:
    print("task arg not found!")
    exit(1)
