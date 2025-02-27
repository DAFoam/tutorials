#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed (unsteady)
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import os
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys.mphys_dafoam import DAFoamBuilderUnsteady
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

# Define the global parameters here
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
A0 = 0.1

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["cylinder"],
    "solverName": "DAPimpleFoam",
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "useWallFunction": True,
    },
    "unsteadyAdjoint": {
        "mode": "timeAccurate",
        "PCMatPrecomputeInterval": 100,
        "PCMatUpdateInterval": 1,
        "reduceIO": True,
        "zeroInitFields": False,
    },
    "printIntervalUnsteady": 1,
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["cylinder"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0),
            "timeOp": "average",
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["cylinder"],
            "directionMode": "fixedDirection",
            "direction": [0.0, 1.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0),
            "timeOp": "average",
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-100,
        "gmresAbsTol": 1.0e-6,
        "gmresMaxIters": 100,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "useNonZeroInitGuess": False,
        "useMGSO": True,
    },
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
    },
    "checkMeshThreshold": {"maxAspectRatio": 5000.0},
}

# mesh warping parameters, users need to manually specify the symmetry plane and their normals
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]],
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/FFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.add_subsystem(
            "scenario1",
            DAFoamBuilderUnsteady(solver_options=daOptions, mesh_options=meshOptions),
            promotes=["*"],
        )

        self.connect("x_aero0", "x_aero")

    def configure(self):

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # use the shape function to define shape variables for 2D
        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        for i in range(1, pts.shape[0] - 1):
            # k=0 and k=1 move together to ensure symmetry in z direction
            # j=0 and j=1 move in opposite direction to ensure symmetry wrt y=0
            shapes.append({pts[i, 0, 0]: dir_y, pts[i, 0, 1]: dir_y, pts[i, 1, 0]: -dir_y, pts[i, 1, 1]: -dir_y})
        self.geometry.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # setup the volume and thickness constraints
        leList = [[-0.5 + 1e-4, 0.0, 1e-4], [-0.5 + 1e-4, 0.0, 0.1 - 1e-4]]
        teList = [[0.5 - 1e-4 - 1e-4, 0.0, 1e-4], [0.5 - 1e-4, 0.0, 0.1 - 1e-4]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=2, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=2, nChord=10)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * len(shapes)))
        # manually connect the dvs output to the geometry and scenario1
        self.connect("shape", "geometry.shape")

        # define the design variables
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=10.0)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.CD", scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.3, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)


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
