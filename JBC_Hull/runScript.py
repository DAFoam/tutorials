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
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: run_driver (default), run_model, compute_totals, check_totals
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================

U0 = 1.179
A0 = 12.2206
p0 = 0.0
nuTilda0 = 1.0e-4

# Set the parameters for optimization
daOptions = {
    "solverName": "DASimpleFoam",
    "designSurfaces": ["hull"],
    "primalMinResTol": 1e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["hull"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0 / 0.5 / U0 / U0 / A0,
        },
    },
    "normalizeStates": {"U": 1.0, "p": 1.0, "nuTilda": 1e-4, "phi": 1.0},
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "adjPCLag": 1,
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/JBCFFD_32.xyz", type="ffd"))

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

        # select the FFD points to move
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = []
        indexList.extend(pts[8:12, 0, 0:4].flatten())
        indexList.extend(pts[8:12, -1, 0:4].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", axis="y", pointSelect=PS)

        # Create reflection constraint
        indSetA = []
        indSetB = []
        for i in range(8, 12, 1):
            for k in range(0, 4, 1):
                indSetA.append(pts[i, 0, k])
                indSetB.append(pts[i, -1, k])
        self.geometry.nom_addLinearConstraintsShape("reflect", indSetA, indSetB, factorA=1.0, factorB=1.0)

        # setup the volume and thickness constraints
        leList = [
            [4.90000000, 0.00000000, -0.41149880],
            [4.90000000, 0.00000000, -0.40347270],
            [4.90000000, 0.00000000, -0.38803330],
            [4.90000000, 0.00000000, -0.36534750],
            [4.90000000, 0.00000000, -0.33601030],
            [4.90000000, 0.00000000, -0.31016020],
            [4.90000000, 0.00000000, -0.28327050],
            [4.90000000, 0.00000000, -0.26248810],
            [4.90000000, 0.00000000, -0.24076410],
            [4.90000000, 0.00000000, -0.20933480],
            [4.90000000, 0.00000000, -0.17458840],
            [4.90000000, 0.00000000, -0.14233480],
            [4.90000000, 0.00000000, -0.11692880],
            [4.90000000, 0.00000000, -0.09984235],
            [4.90000000, 0.00000000, -0.08874606],
            [4.90000000, 0.00000000, -0.07969946],
            [4.90000000, 0.00000000, -0.06954966],
            [4.90000000, 0.00000000, -0.05864429],
            [4.90000000, 0.00000000, -0.04829308],
            [4.90000000, 0.00000000, -0.03831457],
            [4.90000000, 0.00000000, -0.02430242],
            [4.90000000, 0.00000000, -0.00100000],
        ]
        teList = [
            [6.70332700, 0.00000000, -0.41149880],
            [6.73692400, 0.00000000, -0.40347270],
            [6.76842800, 0.00000000, -0.38803330],
            [6.79426000, 0.00000000, -0.36534750],
            [6.81342600, 0.00000000, -0.33601030],
            [6.83648300, 0.00000000, -0.31016020],
            [6.85897100, 0.00000000, -0.28327050],
            [6.83593600, 0.00000000, -0.26248810],
            [6.80929800, 0.00000000, -0.24076410],
            [6.79395800, 0.00000000, -0.20933480],
            [6.79438900, 0.00000000, -0.17458840],
            [6.80874100, 0.00000000, -0.14233480],
            [6.83265000, 0.00000000, -0.11692880],
            [6.86250800, 0.00000000, -0.09984235],
            [6.89566400, 0.00000000, -0.08874606],
            [6.92987100, 0.00000000, -0.07969946],
            [6.96333200, 0.00000000, -0.06954966],
            [6.99621200, 0.00000000, -0.05864429],
            [7.02921500, 0.00000000, -0.04829308],
            [7.06253200, 0.00000000, -0.03831457],
            [7.09456600, 0.00000000, -0.02430242],
            [7.12000000, 0.00000000, -0.00100000],
        ]
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=25, nChord=50)

        # Thickness constraint for lateral thickness
        leList = [[5.01, 0.0000, -0.001], [5.01, 0.0000, -0.410]]
        teList = [[6.2, 0.0000, -0.001], [6.2, 0.0000, -0.410]]
        self.geometry.nom_addThicknessConstraints2D("thickcon1", leList, teList, nSpan=8, nChord=5)

        # Thickness constraint for propeller shaft
        leList = [[6.8, 0.0000, -0.302], [6.8, 0.0000, -0.265]]
        teList = [[6.865, 0.0000, -0.302], [6.865, 0.0000, -0.265]]
        self.geometry.nom_addThicknessConstraints2D("thickcon2", leList, teList, nSpan=5, nChord=5)

        # TODO: this 2D curvature constraint is not in mphys_pygeo yet, need to use 1D curv constraint instead
        # self.geometry.nom_addCurvatureConstraint(
        #    "./FFD/hullCurv.xyz", curvatureType="KSmean", lower=0.0, upper=1.21, addToPyOpt=True, scaled=True
        # )

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        # manually connect the dvs output to the geometry and scenario1
        self.connect("shape", "geometry.shape")

        # define the design variables
        self.add_design_var("shape", lower=-0.5, upper=0.5, scaler=10.0)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.CD", scaler=1.0)
        self.add_constraint("geometry.thickcon1", lower=1e-3, upper=1.125, scaler=1.0)
        self.add_constraint("geometry.thickcon2", lower=1.0, upper=10.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.reflect", equals=0.0, scaler=1.0, linear=True)


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
