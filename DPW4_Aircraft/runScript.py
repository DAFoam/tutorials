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
from pyspline import *

parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: run_driver (default), run_model, compute_totals, check_totals
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================

LScale = 1.0  # scale such that the L0=1

T0 = 300.0
p0 = 101325.0
rho0 = p0 / T0 / 287.0
U0 = 295.0
L0 = 275.80 * 0.0254 * LScale
A0 = 594720.0 * 0.0254 * 0.0254 / 2.0 * LScale * LScale
CofR = [1325.90 * 0.0254 * LScale, 468.75 * 0.0254 * LScale, 177.95 * 0.0254 * LScale]
nuTilda0 = 4.5e-5

CL_target = 0.5
CMY_target = 0.0
aoa0 = 2.5027

# Set the parameters for optimization
daOptions = {
    "designSurfaces": ["wing", "tail", "body"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing", "body", "tail"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing", "body", "tail"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0),
        },
        "CMY": {
            "type": "moment",
            "source": "patchToFace",
            "patches": ["wing", "body", "tail"],
            "axis": [0.0, 1.0, 0.0],
            "center": CofR,
            "scale": 1.0 / (0.5 * rho0 * U0 * U0 * A0 * L0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    # transonic preconditioner to speed up the adjoint convergence
    "transonicPCOption": 1,
    "normalizeStates": {"U": U0, "p": p0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0, "T": T0},
    "adjPCLag": 1,
    "checkMeshThreshold": {"maxAspectRatio": 2000.0, "maxNonOrth": 75.0, "maxSkewness": 8.0},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inout"],
            "flowAxis": "x",
            "normalAxis": "z",
            "components": ["solver", "function"],
        },
    },
}

# Mesh deformation setup
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    "useRotations": False,
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/dpw4FFD.xyz", type="ffd"))

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
        coef = self.geometry.DVGeo.FFD.vols[0].coef.copy()
        # First determine the reference chord lengths:
        nTwist = coef.shape[1]
        sweep_ref = np.zeros((nTwist + 1, 3))
        for j in range(nTwist):
            max_x = np.max(coef[:, j, :, 0])
            min_x = np.min(coef[:, j, :, 0])
            sweep_ref[j + 1, 0] = min_x + 0.25 * (max_x - min_x)
            sweep_ref[j + 1, 1] = np.average(coef[:, j, :, 1])
            sweep_ref[j + 1, 2] = np.average(coef[:, j, :, 2])
        
        # Now add on the first point which is just the second one, projected
        # onto the sym plane
        sweep_ref[0, :] = sweep_ref[1, :].copy()
        sweep_ref[0, 1] = 0.0
        
        # Create the actual reference axis
        c1 = Curve(X=sweep_ref, k=2)
        self.geometry.nom_addRefAxis(name="wing", c1, volumes=[0, 5])

        # Now the tail reference axis
        x = np.array([2365.0, 2365.0]) * 0.0254
        y = np.array([0, 840 / 2.0]) * 0.0254
        z = np.array([255.0, 255.0]) * 0.0254
        c2 = Curve(x=x, y=y, z=z, k=2)
        self.geometry.nom_addRefAxis(name="tail", c2, volumes=[25])

        # Set up global design variables. We dont change the root twist
        def wingTwist(val, geo):
            for i in range(nTwist):
                geo.rot_y["wing"].coef[i + 1] = -val[i]
            
            # Also set the twist of the root to the SOB twist
            geo.rot_y["wing"].coef[0] = -val[0]
        
        def tailTwist(val, geo):
            # Set one twist angle for the tail
            geo.rot_y["tail"].coef[:] = val[0]

        # add twist variable
        self.geometry.nom_addGlobalDV(dvName="wingTwist", value=np.array([0] * nTwist), func=wingTwist)
        # add twist variable
        self.geometry.nom_addGlobalDV(dvName="tailTwist", value=np.array([0]), func=tailTwist)

        # select the FFD points to move
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[:, :, :].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", pointSelect=PS, axis="z")

        # setup the volume and thickness constraints
        # (flattened)LE Root, break and tip. These are adifferent from above
        leRoot = np.array([25.22 * LScale, 3.20 * LScale, 0])
        leBreak = np.array([31.1358 * LScale, 10.8712 * LScale, 0.0])
        leTip = np.array([45.2307 * LScale, 29.38 * LScale, 0.0])
        rootChord = 11.83165 * LScale
        breakChord = 7.25 * LScale
        tipChord = 2.727 * LScale
        
        coe1 = 0.2  # in production run where the mesh is refined, set coe1=0.01
        coe2 = 1.0 - coe1
        xaxis = np.array([1.0, 0, 0])
        leList = [leRoot + coe1 * rootChord * xaxis, leBreak + coe1 * breakChord * xaxis, leTip + coe1 * tipChord * xaxis]
        teList = [leRoot + coe2 * rootChord * xaxis, leBreak + coe2 * breakChord * xaxis, leTip + coe2 * tipChord * xaxis]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=25, nChord=30)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=25, nChord=30)
        # add the LE/TE constraints
        self.geometry.nom_add_LETEConstraint("lecon", volID=0, faceID="iLow")
        self.geometry.nom_add_LETEConstraint("tecon", volID=0, faceID="iHigh")

        # add the design variables to the dvs component's output
        self.dvs.add_output("wingTwist", val=np.array([0] * nTwist))
        self.dvs.add_output("tailTwist", val=np.array([0]))
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("patchV", val=np.array([U0, aoa0]))
        # manually connect the dvs output to the geometry and scenario1
        self.connect("wingTwist", "geometry.wingTwist")
        self.connect("tailTwist", "geometry.tailTwist")
        self.connect("shape", "geometry.shape")
        self.connect("patchV", "scenario1.patchV")

        # define the design variables
        self.add_design_var("wingTwist", lower=-10.0, upper=10.0, scaler=0.1)
        self.add_design_var("tailTwist", lower=-10.0, upper=10.0, scaler=0.1)
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=10.0)
        self.add_design_var("patchV", lower=[U0, 0.0], upper=[U0, 10.0], scaler=0.1)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.CD", scaler=1.0)
        self.add_constraint("scenario1.aero_post.CL", equals=CL_target, scaler=1.0)
        self.add_constraint("scenario1.aero_post.CMY", equals=CMY_target, scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)


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
    optFuncs.findFeasibleDesign(["scenario1.aero_post.CL", "scenario1.aero_post.CMY"], ["patchV", "tailTwist"], targets=[CL_target, CMY_target], designVarsComp=[1, 0])
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
