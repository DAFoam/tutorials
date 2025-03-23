#!/usr/bin/env python
import os
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils


parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="SLSQP")
# which task to run. Options are: run_driver (default), run_model, compute_totals, check_totals
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================

TPR0 = 1.5927
MFR0 = 0.59763

# Input parameters for DAFoam
daOptions = {
    "solverName": "DATurboFoam",
    "designSurfaces": ["blade"],
    "primalMinResTol": 1e-8,
    "primalMinResTolDiff": 1e4,
    "maxCorrectBCCalls": 10,
    "function": {
        "TPR": {
            "type": "totalPressureRatio",
            "source": "patchToFace",
            "patches": ["inlet", "outlet"],
            "inletPatches": ["inlet"],
            "outletPatches": ["outlet"],
            "scale": 1.0,
        },
        "TTR": {
            "type": "totalTemperatureRatio",
            "source": "patchToFace",
            "patches": ["inlet", "outlet"],
            "inletPatches": ["inlet"],
            "outletPatches": ["outlet"],
            "scale": 1.0,
        },
        "MFR": {
            "type": "massFlowRate",
            "source": "patchToFace",
            "patches": ["inlet"],
            "scale": -1.0,
        },
        "CMZ": {
            "type": "moment",
            "source": "patchToFace",
            "patches": ["blade"],
            "axis": [0.0, 0.0, 1.0],
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
    },
    "normalizeStates": {"U": 100.0, "p": 100000.0, "nuTilda": 1e-3, "phi": 1.0, "T": 300.0},
    "adjEqnOption": {
        "gmresRelTol": 1.0e-5,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
        "gmresMaxIters": 2000,
        "gmresRestart": 2000,
    },
    "checkMeshThreshold": {"maxAspectRatio": 2000.0, "maxNonOrth": 78.0, "maxSkewness": 5.0},
    "transonicPCOption": 1,
    "adjPCLag": 1,
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
    },
    "decomposeParDict": {"preservePatches": ["per1", "per2"]},
}

# Mesh deformation setup
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/localFFD.xyz", type="ffd"))

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

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # select the FFD points to move
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[1:4, :, :].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shapey", axis="y", pointSelect=PS)
        self.geometry.nom_addLocalDV(dvName="shapez", axis="z", pointSelect=PS)
        # add the design variables to the dvs component's output
        self.dvs.add_output("shapey", val=np.array([0] * nShapes))
        self.dvs.add_output("shapez", val=np.array([0] * nShapes))
        # manually connect the dvs output to the geometry and cruise
        self.connect("shapey", "geometry.shapey")
        self.connect("shapez", "geometry.shapez")
        # define the design variables to the top level
        self.add_design_var("shapey", lower=-0.002, upper=0.002, scaler=1000.0)
        self.add_design_var("shapez", lower=-0.002, upper=0.002, scaler=1000.0)

        # add objective and constraints to the top level
        self.add_objective("cruise.aero_post.CMZ", scaler=1.0)
        self.add_constraint("cruise.aero_post.MFR", equals=MFR0, scaler=1.0)
        self.add_constraint("cruise.aero_post.TPR", lower=TPR0, scaler=1.0)


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
