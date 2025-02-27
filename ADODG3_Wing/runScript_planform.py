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
CL_target = 0.375
CMX_upper = 1.0
aoa0 = 2.0
A0 = 3.0
span0 = 3.0
chord0 = 1.0

daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1.0e4,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "checkMeshThreshold": {
        "maxAspectRatio": 3000.0,
        "maxNonOrth": 75.0,
        "maxSkewness": 6.0,
        "maxIncorrectlyOrientedFaces": 3,
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/wingFFD.xyz", type="ffd"))

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

        # Create reference axis for the twist variable
        nRefAxPts = self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k")

        # Set up the span variable, here val[0] is the span change in %
        def span(val, geo):
            # coordinates for the reference axis
            refAxisCoef = geo.extractCoef("wingAxis")
            # the relative location of a point in the ref axis
            # refAxisS[0] = 0, and refAxisS[-1] = 1
            refAxisS = geo.refAxis.curves[0].s
            deltaSpan = span0 * val[0] / 100.0
            # linearly change the refAxis coef along the span
            for i in range(nRefAxPts):
                refAxisCoef[i, 2] += refAxisS[i] * deltaSpan
            geo.restoreCoef(refAxisCoef, "wingAxis")

        # add span variable
        self.geometry.nom_addGlobalDV(dvName="span", value=np.array([0]), func=span)

        # Set up the taper variable, val[0] is the chord change in % at the root and
        # val[1] is the chord change at the tip, the chords at other spanwise locations
        # will be linearly determined by the root and tip chords
        def taper(val, geo):
            refAxisS = geo.refAxis.curves[0].s
            cRoot = chord0 * val[0] / 100.0
            cTip = chord0 * val[1] / 100.0
            for i in range(nRefAxPts):
                geo.scale_x["wingAxis"].coef[i] = 1.0 + refAxisS[i] * (cTip - cRoot) + cRoot

        # add taper variable
        self.geometry.nom_addGlobalDV(dvName="taper", value=np.array([0, 0]), func=taper)

        # setup the volume and thickness constraints
        leList = [[0.02, 0.0, 1e-3], [0.02, 0.0, 2.9]]
        teList = [[0.95, 0.0, 1e-3], [0.95, 0.0, 2.9]]
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=25, nChord=30)

        # add the design variables to the dvs component's output
        self.dvs.add_output("span", val=np.array([0]))
        self.dvs.add_output("taper", val=np.array([0, 0]))
        self.dvs.add_output("patchV", val=np.array([U0, aoa0]))
        # manually connect the dvs output to the geometry and scenario1
        self.connect("span", "geometry.span")
        self.connect("taper", "geometry.taper")
        self.connect("patchV", "scenario1.patchV")

        # define the design variables
        self.add_design_var("span", lower=-30.0, upper=30.0, scaler=0.1)
        self.add_design_var("taper", lower=-30.0, upper=30.0, scaler=0.1)
        self.add_design_var("patchV", lower=[U0, 0.0], upper=[U0, 10.0], scaler=0.1)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.CD", scaler=1.0)
        self.add_constraint("scenario1.aero_post.CL", equals=CL_target, scaler=1.0)


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
    prob.check_totals(compact_print=False, step=1e-3, form="central", step_calc="abs")
else:
    print("task arg not found!")
    exit(1)
