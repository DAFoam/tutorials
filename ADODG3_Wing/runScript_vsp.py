#!/usr/bin/env python
"""
DAFoam run script for the wing with OpenVSP parameterization
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
U0 = 100.0
p0 = 101325.0
T0 = 300.0
rho0 = p0 / T0 / 287.0
nuTilda0 = 4.5e-5
# Tu 0.5%, nu_r = 5
k0 = 0.375
epsilon0 = 168.75
omega0 = 5000.0
CL_target = 0.375
aoa0 = 4.9202844
A0 = 3.0
n_cst_coeffs = 7

daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "k0": {"variable": "k", "patches": ["inout"], "value": [k0]},
        "epsilon0": {"variable": "epsilon", "patches": ["inout"], "value": [epsilon0]},
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
    "adjEqnOption": {
        "gmresRelTol": 1.0e-6,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
        "gmresMaxIters": 1000,
        "gmresRestart": 1000,
    },
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": 1e-3,
        "k": 1.0,
        "omega": 100.0,
        "epsilon": 10.0,
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
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="wing.vsp3", type="vsp"), promotes=["*"])

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh.x_aero0", "x_aero_in")
        # need to manually connect the x_aero0 between the geometry component and the scenario1
        # scenario group
        self.connect("x_aero0", "scenario1.x_aero")

        # thickness constraint
        varA = []
        varB = []
        for i in range(2):
            for j in range(n_cst_coeffs):
                varA.append(f"Wing:UpperCoeff_{i}:Au_{j}")
                varB.append(f"Wing:LowerCoeff_{i}:Al_{j}")
        self.add_subsystem(
            "thickness",
            DAFoamLinearConstraint(varA=varA, coeffA=1.0, varB=varB, coeffB=-1.0, size=1, output_name="thickness_val"),
            promotes=["*"],
        )

        # LE C1 continuity constraint
        varA = []
        varB = []
        for i in range(2):
            varA.append(f"Wing:UpperCoeff_{i}:Au_0")
            varB.append(f"Wing:LowerCoeff_{i}:Al_0")
        self.add_subsystem(
            "le_c1",
            DAFoamLinearConstraint(varA=varA, coeffA=1.0, varB=varB, coeffB=1.0, size=1, output_name="le_c1_val"),
            promotes=["*"],
        )

        # add volume constraint
        vsp_vars = []
        for i in range(2):
            for j in range(n_cst_coeffs):
                vsp_vars.append(f"Wing:UpperCoeff_{i}:Au_{j}")
                vsp_vars.append(f"Wing:LowerCoeff_{i}:Al_{j}")
        vsp_vars.append("Wing:XSec_1:Twist")
        self.add_subsystem(
            "volume",
            DAFoamVSPVolume(
                vsp_file="airfoil.vsp3",
                vsp_vars=vsp_vars,
                slice_dir="z",
                n_slices=10,
                output_name="volume_val",
                step=1e-3,
                relativeStep=False,
            ),
            promotes=["*"],
        )

    def configure(self):

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # add shape var
        # NACA0012 upper profile CST coeff, the lower profile is just -CST
        CST = np.array([0.17299, 0.15121, 0.16626, 0.13844, 0.14289, 0.13999, 0.14070])
        for i in range(2):
            for j in range(n_cst_coeffs):
                self.geometry.nom_addVSPVariable("Wing", f"UpperCoeff_{i}", f"Au_{j}", scaledStep=False, dh=1e-3)
                self.geometry.nom_addVSPVariable("Wing", f"LowerCoeff_{i}", f"Al_{j}", scaledStep=False, dh=1e-3)
                self.dvs.add_output(f"Wing:UpperCoeff_{i}:Au_{j}", val=CST[j])
                self.dvs.add_output(f"Wing:LowerCoeff_{i}:Al_{j}", val=-CST[j])
                self.add_design_var(f"Wing:UpperCoeff_{i}:Au_{j}", lower=-1.0, upper=1.0, scaler=1.0)
                self.add_design_var(f"Wing:LowerCoeff_{i}:Al_{j}", lower=-1.0, upper=1.0, scaler=1.0)

        # add twist var
        self.geometry.nom_addVSPVariable("Wing", "XSec_1", "Twist", scaledStep=False, dh=1e-3)
        self.dvs.add_output("Wing:XSec_1:Twist", val=0.0)
        self.add_design_var("Wing:XSec_1:Twist", lower=-1.0, upper=1.0, scaler=0.1)

        self.dvs.add_output("patchV", val=np.array([U0, aoa0]))
        self.connect("patchV", "scenario1.patchV")
        self.add_design_var("patchV", lower=[U0, 0.0], upper=[U0, 10.0], scaler=0.1)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.CD", scaler=1.0)
        self.add_constraint("scenario1.aero_post.CL", equals=CL_target, scaler=1.0)

        # volume constraint
        self.add_constraint("volume_val", lower=1.0, scaler=1.0)

        # 50% thickness
        for i in range(2):
            for j in range(1, n_cst_coeffs):
                indexI = i * n_cst_coeffs + j
                self.add_constraint(f"thickness_val_{indexI}", lower=0.5 * 2.0 * CST[j], scaler=1.0, linear=True)
        # LE radius does not change
        for i in [0, n_cst_coeffs]:
            self.add_constraint(f"thickness_val_{i}", lower=2.0 * CST[0], scaler=1.0, linear=True)
        # LE C1 continuous
        for i in range(2):
            self.add_constraint(f"le_c1_val_{i}", equals=0.0, scaler=1.0, linear=True)


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
