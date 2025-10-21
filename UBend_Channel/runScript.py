#!/usr/bin/env python
"""
DAFoam run script for the U bend channel case v4
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
from mpi4py import MPI
import os
import numpy as np
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="SLSQP")
# which task to run. Options are: run_driver (default), run_model, compute_totals, check_totals
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

# =============================================================================
# daOptions Setup
# =============================================================================
CPL_weight = 0.50                 # Weight of pressure loss (PL) in Obj. Function
HFX_weight = CPL_weight - 1.0     # Weight of heat flux (HFX) in obj. Function
HFX0 = 305                        # HFX value for baseline design
CPL0 = 85.23 - 35.62              # PL value for baseline design
U0 = 8.4                          # Fluid flow (m/s) in x-direction

daOptionsAero = {
    "solverName": "DASimpleFoam",
    "designSurfaces": ["ubend"],
    "useAD": {"mode": "reverse"},
    "primalMinResTol": 1e-8,
    "primalMinResTolDiff": 1e7,
    "writeMinorIterations": True,
    "wallDistanceMethod": "daCustom",
    "primalBC" : {
        "useWallFunction": True,
    },

    "function": {

            "TP1": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["inlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            },

            "TP2": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["outlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            },

            "HFX": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["ubend"],
                "scale": 1.0,
                "addToAdjoint": True,
            },

    },

    "adjStateOrdering": "cell",

    "adjEqnOption": {"gmresRelTol"      : 1e-5,
                     "gmresTolDiff"     : 1e4,
                     "pcFillLevel"      : 2,
                     "jacMatReOrdering" : "natural",
                     "gmresMaxIters"    : 3000,
                     "gmresRestart"     : 3000},

    "normalizeStates": {"U"       : U0,
                        "p"       : (U0 * U0) / 2.,
                        "nuTilda" : 1e-3,
                        "phi"     : 1.0,
                        "T"       : 300},

    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver" , "function"]},
    },

    "outputInfo": {
        "q_convect": {
            "type"       : "thermalCouplingOutput",
            "patches"    : ["ubend"],
            "components" : ["thermalCoupling"],
        },
    },

}

# =============================================================================
# Mesh Setup
# =============================================================================
meshOptions = {
    "gridFile"       : os.getcwd(),
    "fileType"       : "OpenFOAM",
    "symmetryPlanes" : [],
}

# =============================================================================
# Top class To Setup The Optimization Problem
# =============================================================================
class Top(Multipoint):
    def setup(self):

        # initialize builders
        dafoam_builder = DAFoamBuilder(daOptionsAero , meshOptions , scenario = "aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add design variable component and promote to top level
        self.add_subsystem("dvs" , om.IndepVarComp() , promotes = ["*"])

        # add mesh component
        self.add_subsystem("mesh_aero" , dafoam_builder.get_mesh_coordinate_subsystem())

        # add geometry component
        self.add_subsystem("geometry_aero" , OM_DVGEOCOMP(file = "FFD/testFFD.xyz" , type = "ffd"))

        # add a scenario (flow condition) for optimization. For no themal (solid) use ScenarioAerodynamic, for thermal (solid) use ScenarioAerothermal
        self.mphys_add_scenario("scenario" , ScenarioAerodynamic(aero_builder = dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        self.connect("mesh_aero.x_aero0" , "geometry_aero.x_aero_in")
        self.connect("geometry_aero.x_aero0" , "scenario.x_aero")

        # add obj val for PL
        self.add_subsystem("OBJ" , om.ExecComp("val = scalePL * (TP1 - TP2) + (scaleHFX * HFX)" , scalePL = {'val' : CPL_weight / CPL0 , 'constant' : True} , scaleHFX = {'val' : HFX_weight / HFX0 , 'constant' : True}))

    def configure(self):

        # initialize the optimization
        super().configure()

        # get surface coordinates from mesh component
        points_aero = self.mesh_aero.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry_aero.nom_add_discipline_coords("aero" , points_aero)

        # get FFD points
        pts = self.geometry_aero.nom_getDVGeo().getLocalIndex(0)

        #---------- setup DVs ----------
        # shapex
        indexList = []
        indexList.extend(pts[7:16 , 1 , :].flatten())
        PS = geo_utils.PointSelect("list" , indexList)
        shapexUpper = self.geometry_aero.nom_addLocalDV(dvName = "shapexUpper" , pointSelect = PS , axis = "x")

        # shapey
        indexList = []
        indexList.extend(pts[7:16 , 1 , :].flatten())
        PS = geo_utils.PointSelect("list" , indexList)
        shapeyUpper = self.geometry_aero.nom_addLocalDV(dvName = "shapeyUpper" , pointSelect = PS , axis = "y")

        # shapez
        indexList = []
        indexList.extend(pts[7:16 , 1 , :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        shapezUpper = self.geometry_aero.nom_addLocalDV(dvName = "shapezUpper" , pointSelect = PS , axis = "z")

        # shapex
        indexList = []
        indexList.extend(pts[7:16 , 0 , :].flatten())
        PS = geo_utils.PointSelect("list" , indexList)
        shapexLower = self.geometry_aero.nom_addLocalDV(dvName = "shapexLower" , pointSelect = PS , axis = "x")

        # shapey
        indexList = []
        indexList.extend(pts[7:16 , 0 , :].flatten())
        PS = geo_utils.PointSelect("list" , indexList)
        shapeyLower = self.geometry_aero.nom_addLocalDV(dvName = "shapeyLower" , pointSelect = PS , axis = "y")

        # shapez
        indexList = []
        indexList.extend(pts[7:16 , 0 , :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        shapezLower = self.geometry_aero.nom_addLocalDV(dvName = "shapezLower" , pointSelect = PS , axis = "z")

        #---------- Add Outputs For The DVs ----------
        self.dvs.add_output("shapexUpper" , val = np.array([0]*shapexUpper))
        self.dvs.add_output("shapeyUpper" , val = np.array([0]*shapeyUpper))
        self.dvs.add_output("shapezUpper" , val = np.array([0]*shapezUpper))

        self.dvs.add_output("shapexLower" , val = np.array([0]*shapexLower))
        self.dvs.add_output("shapeyLower" , val = np.array([0]*shapeyLower))
        self.dvs.add_output("shapezLower" , val = np.array([0]*shapezLower))

        #---------- Connect The Design Variables To The Geometry ----------
        self.connect("shapexUpper" , "geometry_aero.shapexUpper")
        self.connect("shapeyUpper" , "geometry_aero.shapeyUpper")
        self.connect("shapezUpper" , "geometry_aero.shapezUpper")

        self.connect("shapexLower" , "geometry_aero.shapexLower")
        self.connect("shapeyLower" , "geometry_aero.shapeyLower")
        self.connect("shapezLower" , "geometry_aero.shapezLower")

        #---------- Define The Design Variables To The Top Level ----------
        self.add_design_var("shapexUpper" , lower = -0.04 , upper = 0.04 , scaler = 25.0)
        self.add_design_var("shapeyUpper" , lower = -0.04 , upper = 0.04 , scaler = 25.0)
        self.add_design_var("shapezUpper" , lower = -0.04 , upper = 0.04 , scaler = 25.0)
        self.add_design_var("shapexLower" , lower = -0.04 , upper = 0.04 , scaler = 25.0)
        self.add_design_var("shapeyLower" , lower = -0.04 , upper = 0.04 , scaler = 25.0)
        self.add_design_var("shapezLower" , lower = -0.04 , upper = 0.04 , scaler = 25.0)

        # add objective and constraints
        self.connect("scenario.aero_post.TP1" , "OBJ.TP1")
        self.connect("scenario.aero_post.TP2" , "OBJ.TP2")
        self.connect("scenario.aero_post.HFX" , "OBJ.HFX")
        self.add_objective("OBJ.val" , scaler = 1.0)

# =============================================================================
# Problem Setup
# =============================================================================
prob = om.Problem(reports = None)
prob.model = Top()
prob.setup(mode = "rev")
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer

if args.optimizer == "SNOPT":
    prob.driver.opt_settings = {
        "Major feasibility tolerance" : 1.0e-5,
        "Major optimality tolerance"  : 1.0e-5,
        "Minor feasibility tolerance" : 1.0e-5,
        "Verify level"                : -1,
        "Function precision"          : 1.0e-5,
        "Major iterations limit"      : 100,
        "Nonderivative linesearch"    : None,
        "Print file"                  : "opt_SNOPT_print.txt",
        "Summary file"                : "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol"                        : 1.0e-5,
        "constr_viol_tol"            : 1.0e-5,
        "max_iter"                   : 100,
        "print_level"                : 5,
        "output_file"                : "opt_IPOPT.txt",
        "mu_strategy"                : "adaptive",
        "limited_memory_max_history" : 10,
        "nlp_scaling_method"         : "none",
        "alpha_for_y"                : "full",
        "recalc_y"                   : "yes",
    }
elif args.optimizer == "SLSQP":
    prob.driver.opt_settings = {
        "ACC"   : 1.0e-5,
        "MAXIT" : 100,
        "IFILE" : "opt_SLSQP.txt",
    }
else:
    print("optimizer arg not valid!")
    exit(1)

prob.driver.options["debug_print"] = ["nl_cons" , "objs" , "desvars"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"

if args.task == "run_driver":
    prob.run_driver()

elif args.task == "run_model":
    prob.run_model()

elif args.task == "compute_totals":
    prob.run_model()
    totals = prob.compute_totals()

    if MPI.COMM_WORLD.rank == 0:
        print(totals)

elif args.task == "check_totals":
    prob.run_model()

    prob.check_totals(
        compact_print = False,
        step          = 1e-4,
        form          = "central",
        step_calc     = "abs",
    )

else:
    print("task arg not found!")
    exit(1)