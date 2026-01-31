#!/usr/bin/env python

# Spring 2025
# Ubend Channel runScript With Solid Domain -> Full Pipe Version -> DAFoam v4.0.0
# Chris Psenica - Iowa State University - i-Design Lab

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
from funtofem.mphys import MeldThermalBuilder
from pygeo import geo_utils
from mphys.scenario_aerothermal import ScenarioAeroThermal
from pygeo.mphys import OM_DVGEOCOMP

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-optimizer" , help = "optimizer to use" , type = str , default = "IPOPT")
parser.add_argument("-task" , help = "type of run to do" , type = str , default = "run_driver")
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

# =============================================================================
# Set The Parameters For The Optimization
# =============================================================================
#---------- Flow Conditions ----------
U = 5.0              # Fluid flow (m/s) inside the pipe
Uf = 5.0             # Fluid flow (m/s) of the far field
nuTilda0 = 1.5e-4
p0 = 0.0

#---------- Obj. Function Weights ----------
TMweight = 0.9
TM_baseline = 305.5 ; PL_baseline = 35.07 - 12.76
scaleTM = TMweight / TM_baseline ; scalePL = (1 - TMweight) / PL_baseline
HFX_baseline = -122.85

# =============================================================================
# DaOptions Aero Setup
# =============================================================================
daOptionsAero = {
    #---------- DaOptions Parameters ----------
    "solverName"          : "DASimpleFoam",
    "designSurfaces"      : ["ubend_inner" , "ubend_outer"],
    "useAD"               : {"mode": "reverse"},
    "primalMinResTol"     : 1e-12,
    "primalMinResTolDiff" : 1e11,
    "wallDistanceMethod"  : "daCustom",

    "primalBC" : {
        "U0": {"variable": "U" , "patches": ["inlet"] , "value": [0.0 , U , 0.0]},
        "U1": {"variable": "U" , "patches": ["farfield_inlet"] , "value": [0.0 , 0.0 , -Uf]},
        "p0": {"variable": "p" , "patches": ["outlet"], "value": [p0]},
        "p1": {"variable": "p" , "patches": ["farfield_outlet"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [nuTilda0]},
        "nuTilda1": {"variable": "nuTilda", "patches": ["farfield_inlet"], "value": [nuTilda0]},
        "useWallFunction": True,
    },

    #---------- Objective Function ----------
    "function": {

        "TP1": {
            "type"         : "totalPressure",
            "source"       : "patchToFace",
            "patches"      : ["inlet"],
            "scale"        : 1.0,
            "addToAdjoint" : True,
        },

        "TP2": {
            "type"         : "totalPressure",
            "source"       : "patchToFace",
            "patches"      : ["outlet"],
            "scale"        : 1.0,
            "addToAdjoint" : True,
        },

        "Tmean": {
            "type"         : "patchMean",
            "source"       : "patchToFace",
            "patches"      : ["outlet"],
            "varName"      : "T",
            "varType"      : "scalar",
            "component"    : 0,
            "scale"        : 1.0,
            "addToAdjoint" : True,
        },

        "HFX": {
            "type"         : "wallHeatFlux",
            "source"       : "patchToFace",
            "byUnitArea"   : False,
            "patches"      : ["ubend_inner"],
            "scale"        : 1.0,
            "addToAdjoint" : False,
        },

    },

    #---------- Optimization Parameters ----------
    "adjStateOrdering" : "cell",

    "normalizeStates": {"U"       : U ,
                        "p"       : (U * U) / 2. ,
                        "nuTilda" : 1.5e-3 ,
                        "phi"     : 1.0 ,
                        "T"       : 300},

    "adjEqnOption": {"gmresRelTol"         : 1.0e-2,
                     "gmresTolDiff"        : 1.0e2,
                     "pcFillLevel"         : 2,
                     "jacMatReOrdering"    : "natural",
                     "gmresMaxIters"       : 2000,
                     "gmresRestart"        : 2000,
                     "dynAdjustTol"        : True,
                     "useNonZeroInitGuess" : True},

    #---------- Coupling Info ----------
    "inputInfo": {
        "aero_vol_coords" : {"type": "volCoord", "components": ["solver", "function"]},
        "T_convect": {
            "type"        : "thermalCouplingInput",
            "patches"     : ["ubend_inner" , "ubend_outer"],
            "components"  : ["solver" , "function"],
        },
    },

    "outputInfo": {
        "q_convect": {
            "type"       : "thermalCouplingOutput",
            "patches"    : ["ubend_inner" , "ubend_outer"],
            "components" : ["thermalCoupling"],
        },
    },
}

# =============================================================================
# DaOptions Thermal Setup
# =============================================================================
daOptionsThermal = {
    #---------- DaOptions Parameters ----------
    "designSurfaces"      : ["ubend_inner_solid" , "ubend_outer_solid"],
    "solverName"          : "DAHeatTransferFoam",
    "primalMinResTol"     : 1.0e-12,
    "primalMinResTolDiff" : 1.0e11,
    "wallDistanceMethod"  : "daCustom",
    "discipline"          : "thermal",

    #---------- Objective Function ----------
    "function": {

        "HFXsolid": {
            "type"         : "wallHeatFlux",
            "source"       : "patchToFace",
            "byUnitArea"   : False,
            "patches"      : ["ubend_inner_solid"],
            "scale"        : 1.0,
            "addToAdjoint" : False,
        },

    },

    #---------- Optimization Parameters ----------
    "adjStateOrdering" : "cell",

    "adjEqnOption": {"gmresRelTol"         : 1.0e-3,
                     "gmresTolDiff"        : 1e2,
                     "pcFillLevel"         : 1,
                     "jacMatReOrdering"    : "natural",
                     "gmresMaxIters"       : 1000,
                     "gmresRestart"        : 1000,
                     "dynAdjustTol"        : True,
                     "useNonZeroInitGuess" : True},

    "normalizeStates": {"T" : 300},

    #---------- Coupling Info ----------
    "inputInfo": {
        "thermal_vol_coords" : {"type": "volCoord", "components": ["solver", "function"]},
        "q_conduct": {
            "type"           : "thermalCouplingInput",
            "patches"        : ["ubend_inner_solid" , "ubend_outer_solid"],
            "components"     : ["solver"],
        },
    },

    "outputInfo": {
        "T_conduct": {
            "type"       : "thermalCouplingOutput",
            "patches"    : ["ubend_inner_solid" , "ubend_outer_solid"],
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

        #---------- Initialize Builders ----------
        dafoam_builder_aero = DAFoamBuilder(daOptionsAero , meshOptions , scenario = "aerothermal" , run_directory = "aero")
        dafoam_builder_aero.initialize(self.comm)

        dafoam_builder_thermal = DAFoamBuilder(daOptionsThermal , meshOptions , scenario = "aerothermal" , run_directory = "thermal")
        dafoam_builder_thermal.initialize(self.comm)

        thermalxfer_builder = MeldThermalBuilder(dafoam_builder_aero , dafoam_builder_thermal , n = 1 , beta = 0.5)
        thermalxfer_builder.initialize(self.comm)

        #---------- Add Design Variable Component And Promote To Top Level ----------
        self.add_subsystem("dvs" , om.IndepVarComp() , promotes = ["*"])

        #---------- Add Mesh Component ----------
        self.add_subsystem("mesh_aero" , dafoam_builder_aero.get_mesh_coordinate_subsystem())
        self.add_subsystem("mesh_thermal" , dafoam_builder_thermal.get_mesh_coordinate_subsystem())

        #---------- Add Geometry Component ----------
        self.add_subsystem("geometry_aero" , OM_DVGEOCOMP(file = "aero/FFD/UBendFFD.xyz" , type = "ffd"))
        self.add_subsystem("geometry_thermal" , OM_DVGEOCOMP(file = "aero/FFD/UBendFFD.xyz" , type = "ffd"))

        #---------- Add Scenario (Flow Condition) For Optimization ----------
        '''
        For no thermal (solid) use ScenarioAerodynamic, for thermal (solid) use ScenarioAerothermal
        we pass the builder to the scenario to actually run the flow and adjoint
        '''
        self.mphys_add_scenario(
            "scenario" ,
            ScenarioAeroThermal(aero_builder = dafoam_builder_aero , thermal_builder = dafoam_builder_thermal , thermalxfer_builder = thermalxfer_builder),
            om.NonlinearBlockGS(maxiter = 30 , iprint = 2 , use_aitken = True , rtol = 1e-7 , atol = 2e-2),
            om.LinearBlockGS(maxiter = 30 , iprint = 2 , use_aitken = True , rtol = 1e-6 , atol = 1e-2),
        )

        #---------- Manually Connect Aero & Thermal Between The Mesh & Geometry Components ----------
        self.connect("mesh_aero.x_aero0" , "geometry_aero.x_aero_in")
        self.connect("geometry_aero.x_aero0" , "scenario.x_aero")

        self.connect("mesh_thermal.x_thermal0" , "geometry_thermal.x_thermal_in")
        self.connect("geometry_thermal.x_thermal0" , "scenario.x_thermal")

        #---------- Add Objective Function Component ----------
        self.add_subsystem("OBJ" , om.ExecComp("val = scalePL * (TP1 - TP2) + (scaleTM * Tmean)" , scalePL = {'val' : scalePL , 'constant' : True} , scaleTM = {'val' : scaleTM , 'constant' : True}))

    def configure(self):

        #---------- Initialize The Optimization ----------
        super().configure()

        #---------- Get Surface Coordinates From Mesh Component ----------
        points_aero = self.mesh_aero.mphys_get_surface_mesh()
        points_thermal = self.mesh_thermal.mphys_get_surface_mesh()

        #---------- Add Pointset To The Geometry Component ----------
        self.geometry_aero.nom_add_discipline_coords("aero" , points_aero)
        self.geometry_thermal.nom_add_discipline_coords("thermal" , points_thermal)

        #---------- Create Design Variables And Assign Them To FFD Points ----------
        # get FFD points
        pts = self.geometry_aero.nom_getDVGeo().getLocalIndex(0)

        # shapex
        indexList = []
        indexList.extend(pts[7:16 , 1 , :].flatten())
        PS = geo_utils.PointSelect("list" , indexList)
        shapexUpper = self.geometry_aero.nom_addLocalDV(dvName = "shapexUpper" , pointSelect = PS , axis = "x")
        shapexUpper = self.geometry_thermal.nom_addLocalDV(dvName = "shapexUpper" , pointSelect = PS , axis = "x")

        # shapey
        indexList = []
        indexList.extend(pts[7:16 , 1 , :].flatten())
        PS = geo_utils.PointSelect("list" , indexList)
        shapeyUpper = self.geometry_aero.nom_addLocalDV(dvName = "shapeyUpper" , pointSelect = PS , axis = "y")
        shapeyUpper = self.geometry_thermal.nom_addLocalDV(dvName = "shapeyUpper" , pointSelect = PS , axis = "y")

        # shapez
        indexList = []
        indexList.extend(pts[7:16 , 1 , :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        shapezUpper = self.geometry_aero.nom_addLocalDV(dvName = "shapezUpper" , pointSelect = PS , axis = "z")
        shapezUpper = self.geometry_thermal.nom_addLocalDV(dvName = "shapezUpper" , pointSelect = PS , axis = "z")

        # shapex
        indexList = []
        indexList.extend(pts[7:16 , 0 , :].flatten())
        PS = geo_utils.PointSelect("list" , indexList)
        shapexLower = self.geometry_aero.nom_addLocalDV(dvName = "shapexLower" , pointSelect = PS , axis = "x")
        shapexLower = self.geometry_thermal.nom_addLocalDV(dvName = "shapexLower" , pointSelect = PS , axis = "x")

        # shapey
        indexList = []
        indexList.extend(pts[7:16 , 0 , :].flatten())
        PS = geo_utils.PointSelect("list" , indexList)
        shapeyLower = self.geometry_aero.nom_addLocalDV(dvName = "shapeyLower" , pointSelect = PS , axis = "y")
        shapeyLower = self.geometry_thermal.nom_addLocalDV(dvName = "shapeyLower" , pointSelect = PS , axis = "y")

        # shapez
        indexList = []
        indexList.extend(pts[7:16 , 0 , :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        shapezLower = self.geometry_aero.nom_addLocalDV(dvName = "shapezLower" , pointSelect = PS , axis = "z")
        shapezLower = self.geometry_thermal.nom_addLocalDV(dvName = "shapezLower" , pointSelect = PS , axis = "z")

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

        self.connect("shapexUpper" , "geometry_thermal.shapexUpper")
        self.connect("shapeyUpper" , "geometry_thermal.shapeyUpper")
        self.connect("shapezUpper" , "geometry_thermal.shapezUpper")
        self.connect("shapexLower" , "geometry_thermal.shapexLower")
        self.connect("shapeyLower" , "geometry_thermal.shapeyLower")
        self.connect("shapezLower" , "geometry_thermal.shapezLower")

        #---------- Define The Design Variables To The Top Level ----------
        self.add_design_var("shapexUpper" , lower = -0.04 , upper = 0.04 , scaler = 10.0)
        self.add_design_var("shapeyUpper" , lower = -0.04 , upper = 0.04 , scaler = 10.0)
        self.add_design_var("shapezUpper" , lower = -0.04 , upper = 0.04 , scaler = 10.0)
        self.add_design_var("shapexLower" , lower = -0.01 , upper = 0.01 , scaler = 40.0)
        self.add_design_var("shapeyLower" , lower = -0.01 , upper = 0.01 , scaler = 40.0)
        self.add_design_var("shapezLower" , lower = -0.01 , upper = 0.01 , scaler = 40.0)

        #---------- Add Objective And Constraints ----------
        self.connect("scenario.aero_post.TP1"   , "OBJ.TP1")
        self.connect("scenario.aero_post.TP2"   , "OBJ.TP2")
        self.connect("scenario.aero_post.Tmean" , "OBJ.Tmean")
        self.add_objective("OBJ.val" , scaler = 1.0)

# =============================================================================
# OpenMDAO Setup
# =============================================================================
#---------- OpenMDAO Parameters ----------
prob = om.Problem(reports = None)
prob.model = Top()
prob.setup(mode = "rev")
om.n2(prob , show_browser = False , outfile = "n2.html")

#---------- Use Pyoptsparse To Setup The Optimization ----------
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer

#---------- Optimizer Parameters ----------
if args.optimizer == "SNOPT":
    prob.driver.opt_settings = {
        "Major feasibility tolerance" : 1.0e-5,
        "Major optimality tolerance"  : 1.0e-5,
        "Minor feasibility tolerance" : 1.0e-5,
        "Verify level"                : -1,
        "Function precision"          : 1.0e-5,
        "Major iterations limit"      : 1000,
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
    print("opt arg not valid!")
    exit(0)

prob.driver.options["debug_print"] = ["nl_cons" , "objs" , "desvars" , "totals"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"

#---------- Run Task Parameters ----------
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
    results       = prob.check_totals(
    of            = ["OBJ.val"],
    wrt           = ["shapexUpper" , "shapeyUpper" , "shapezUpper" , "shapexLower" , "shapeyLower" , "shapezLower"],
    compact_print = False,
    step          = 1e-3,
    form          = "central",
    step_calc     = "abs",
)

else:
    print("task arg not found!")
    exit(1)
