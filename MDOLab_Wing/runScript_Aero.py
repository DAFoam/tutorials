#!/usr/bin/env python
import os
import argparse
import numpy as np

import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys_dafoam import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from mphys.solver_builders.mphys_dvgeo import OM_DVGEOCOMP

parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: opt (default), runPrimal
parser.add_argument("-task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()

U0 = 100.0
p0 = 101325.0
nuTilda0 = 4.5e-5
T0 = 300.0
CL_target = 0.5
aoa0 = 2.0
rho0 = p0 / T0 / 287.0
A0 = 45.5


class Top(Multipoint):
    def setup(self):
        daOptions = {
            "designSurfaces": ["wing"],
            "solverName": "DARhoSimpleFoam",
            "primalMinResTol": 1.0e-8,
            "primalBC": {
                "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
                "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
                "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
                "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
                "useWallFunction": True,
            },
            # variable bounds for compressible flow conditions
            "primalVarBounds": {
                "UMax": 1000.0,
                "UMin": -1000.0,
                "pMax": 500000.0,
                "pMin": 20000.0,
                "eMax": 500000.0,
                "eMin": 100000.0,
                "rhoMax": 5.0,
                "rhoMin": 0.2,
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
                "T": T0,
                "nuTilda": 1e-3,
                "phi": 1.0,
            },
            "adjPartDerivFDStep": {"State": 1e-6},
            "checkMeshThreshold": {
                "maxAspectRatio": 1000.0,
                "maxNonOrth": 70.0,
                "maxSkewness": 8.0,
            },
            "designVar": {
                "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
                "twist": {"designVarType": "FFD"},
                "shape": {"designVarType": "FFD"},
            },
        }

        meshOptions = {
            "gridFile": os.getcwd(),
            "fileType": "OpenFOAM",
            # point and normal for the symmetry plane
            "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        }

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(ffd_file="FFD/wingFFD.xyz"))

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
        self.cruise.aero_post.mphys_add_funcs(["CD", "CL"])

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # Create reference axis
        nRefAxPts = self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k")

        # Set up global design variables
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wingAxis"].coef[i] = -val[i - 1]

        # define an angle of attack function to change the U direction at the far field
        def aoa(val, DASolver):
            aoa = val[0] * np.pi / 180.0
            U = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
            # we need to update the U value only
            DASolver.setOption("primalBC", {"U0": {"value": U}})
            DASolver.updateDAOption()

        # pass this aoa function to the cruise group
        self.cruise.coupling.solver.add_dv_func("aoa", aoa)
        self.cruise.aero_post.add_dv_func("aoa", aoa)

        # add twist variable
        self.geometry.nom_addGeoDVGlobal(dvName="twist", value=np.array([0] * (nRefAxPts - 1)), func=twist)

        # add shape variable
        nShapes = self.geometry.nom_addGeoDVLocal(dvName="shape")

        leList = [[0.1, 0, 0.01], [7.5, 0, 13.9]]
        teList = [[4.9, 0, 0.01], [8.9, 0, 13.9]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_add_LETEConstraint("lecon", 0, "iLow")
        self.geometry.nom_add_LETEConstraint("tecon", 0, "iHigh")

        # add dvs to ivc and connect
        self.dvs.add_output("twist", val=np.array([0] * (nRefAxPts - 1)))
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("aoa", val=np.array([aoa0]))
        self.connect("twist", "geometry.twist")
        self.connect("shape", "geometry.shape")
        self.connect("aoa", "cruise.aoa")

        # define the design variables
        self.add_design_var("twist", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        self.add_design_var("aoa", lower=0.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=CL_target, scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

# use pyoptsparse to setup optimization
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer
# options for optimizers
if args.optimizer == "SNOPT":
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-7,
        "Major optimality tolerance": 1.0e-7,
        "Minor feasibility tolerance": 1.0e-7,
        "Verify level": -1,
        "Function precision": 1.0e-7,
        "Major iterations limit": 100,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol": 1.0e-7,
        "constr_viol_tol": 1.0e-7,
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
        "ACC": 1.0e-7,
        "MAXIT": 100,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("optimizer arg not valid!")
    exit(1)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.hist_file = "opt.hst"

if args.task == "opt":
    prob.run_driver()
elif args.task == "runPrimal":
    prob.run_model()
else:
    print("task arg not found!")
    exit(1)
