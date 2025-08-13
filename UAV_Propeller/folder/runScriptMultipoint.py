#!/usr/bin/env python
import os
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from tacs.mphys import TacsBuilder
from funtofem.mphys import MeldBuilder
from mphys.scenario_aerostructural import ScenarioAeroStructural
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils

import tacsSetup

parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="SNOPT")
# which task to run. Options are: opt (default), runPrimal, runAdjoint, checkTotals
parser.add_argument("-task", help="type of run to do", type=str, default="opt")
# which case to opt. Options are: 1 - twist
parser.add_argument("-case", help="which case to optimize", type=int, default=1)
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================
omega0 = [463.2, 576.5, 523.8]
thrust_target = [5.0, 8.0, 6.5]
m0 = 0.02824

fc0 = {"primalBC": {"MRF": omega0[0]}}
fc1 = {"primalBC": {"MRF": omega0[1]}}
fc2 = {"primalBC": {"MRF": omega0[2]}}

if args.case == 1:
    # twist only
    DVDict = {
        "twist": {"designVarType": "FFD"},
    }
elif args.case == 2:
    # twist and shape
    DVDict = {
        "twist": {"designVarType": "FFD"},
        "shape": {"designVarType": "FFD"},
    }
elif args.case == 3:
    # twist, shape, and chord
    DVDict = {
        "twist": {"designVarType": "FFD"},
        "shape": {"designVarType": "FFD"},
        "chord": {"designVarType": "FFD"},
    }
elif args.case == 4:
    # twist, shape, chord, and span
    DVDict = {
        "twist": {"designVarType": "FFD"},
        "shape": {"designVarType": "FFD"},
        "chord": {"designVarType": "FFD"},
        "span": {"designVarType": "FFD"},
    }
else:
    print("case not valid!")
    exit(1)

# always use MRF as dv
DVDict["MRF"] = {"designVarType": "BC"}

daOptions = {
    "designSurfaces": ["blade"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1.0e4,
    # set it to True for hover case
    "hasIterativeBC": True,
    "useConstrainHbyA": True,
    "couplingInfo": {
        "aerostructural": {
            "active": True,
            "pRef": 101325.0,
            "propMovement": False,
            "couplingSurfaceGroups": {"bladeGroup": ["blade"]},
        }
    },  # set the ref pressure for computing force for FSI
    "primalBC": {
        "MRF": omega0[0],
        "useWallFunction": False,
    },
    "primalVarBounds": {
        "omegaMin": -1e16,
    },
    "objFunc": {
        "power": {
            "part1": {
                "type": "power",
                "source": "patchToFace",
                "patches": ["blade"],
                "axis": [1.0, 0.0, 0.0],
                "center": [0.0, 0.0, 0.0],
                "scale": -2.0,
                "addToAdjoint": True,
            }
        },
        "thrust": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["blade"],
                "directionMode": "fixedDirection",
                "direction": [1.0, 0.0, 0.0],
                "scale": -2.0,
                "addToAdjoint": True,
            }
        },
        "skewness": {
            "part1": {
                "type": "meshQualityKS",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "coeffKS": 20.0,
                "metric": "faceSkewness",
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
        "nonOrtho": {
            "part1": {
                "type": "meshQualityKS",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "coeffKS": 1.0,
                "metric": "nonOrthoAngle",
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-3,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
        "gmresMaxIters": 1000,
        "gmresRestart": 1000,
        "useNonZeroInitGuess": True,
    },
    "adjPCLag": 1,
    "normalizeStates": {
        "U": 50.0,
        "p": 101325.0,
        "T": 300.0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "checkMeshThreshold": {
        "maxNonOrth": 89.0,
        "maxSkewness": 5.0,
    },
    "designVar": DVDict,
    "decomposeParDict": {"preservePatches": ["cyc1", "cyc2"]},
}

# Mesh deformation setup
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    "useRotations": False,
    # point and normal for the symmetry plane
    "symmetryPlanes": [],
}

# TACS Setup
tacsOptions = {
    "element_callback": tacsSetup.element_callback,
    "problem_setup": tacsSetup.problem_setup,
    "mesh_file": "./StructMesh.bdf",
}

# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        aero_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerostructural")
        aero_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the aerodynamic mesh component
        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        # create the builder to initialize TACS
        struct_builder = TacsBuilder(tacsOptions)
        struct_builder.initialize(self.comm)

        # add the structure mesh component
        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        # load and displacement transfer builder (meld), isym sets the symmetry plan axis (k)
        xfer_builder = MeldBuilder(aero_builder, struct_builder, isym=-1, check_partials=False)
        xfer_builder.initialize(self.comm)

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/FFD.xyz", type="ffd"))

        # primal and adjoint solution options, i.e., nonlinear block Gauss-Seidel for aerostructural analysis
        # and linear block Gauss-Seidel for the coupled adjoint
        nonlinear_solver0 = om.NonlinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-6, atol=100.0)
        linear_solver0 = om.LinearBlockGS(maxiter=5, iprint=2, use_aitken=True, rtol=1e-5, atol=1e0)
        # add the coupling aerostructural scenario
        self.mphys_add_scenario(
            "hover0",
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=xfer_builder
            ),
            nonlinear_solver0,
            linear_solver0,
        )

        nonlinear_solver1 = om.NonlinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-6, atol=100.0)
        linear_solver1 = om.LinearBlockGS(maxiter=5, iprint=2, use_aitken=True, rtol=1e-5, atol=1e0)
        self.mphys_add_scenario(
            "hover1",
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=xfer_builder
            ),
            nonlinear_solver1,
            linear_solver1,
        )

        nonlinear_solver2 = om.NonlinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-6, atol=100.0)
        linear_solver2 = om.LinearBlockGS(maxiter=5, iprint=2, use_aitken=True, rtol=1e-5, atol=1e0)
        self.mphys_add_scenario(
            "hover2",
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=xfer_builder
            ),
            nonlinear_solver2,
            linear_solver2,
        )

        # need to manually connect the vars in the geo component to hover
        for discipline in ["aero"]:
            self.connect("geometry.x_%s0" % discipline, "hover0.x_%s0_masked" % discipline)
            self.connect("geometry.x_%s0" % discipline, "hover1.x_%s0_masked" % discipline)
            self.connect("geometry.x_%s0" % discipline, "hover2.x_%s0_masked" % discipline)
        for discipline in ["struct"]:
            self.connect("geometry.x_%s0" % discipline, "hover0.x_%s0" % discipline)
            self.connect("geometry.x_%s0" % discipline, "hover1.x_%s0" % discipline)
            self.connect("geometry.x_%s0" % discipline, "hover2.x_%s0" % discipline)

        # more manual connection
        self.connect("mesh_aero.x_aero0", "geometry.x_aero_in")
        self.connect("mesh_struct.x_struct0", "geometry.x_struct_in")

        # add an exec comp to average obj
        self.add_subsystem("obj", om.ExecComp("value=0.5*power0+0.25*power1+0.25*power2"))

    def configure(self):

        # configure and setup perform a similar function, i.e., initialize the optimization.
        # But configure will be run after setup

        # call this to configure the coupling solver
        super().configure()

        self.hover0.coupling.aero.mphys_set_options(fc0)
        self.hover1.coupling.aero.mphys_set_options(fc1)
        self.hover2.coupling.aero.mphys_set_options(fc2)

        self.hover0.aero_post.mphys_set_options(fc0)
        self.hover1.aero_post.mphys_set_options(fc1)
        self.hover2.aero_post.mphys_set_options(fc2)

        # add the objective function to the hover scenario
        self.hover0.aero_post.mphys_add_funcs()
        self.hover1.aero_post.mphys_add_funcs()
        self.hover2.aero_post.mphys_add_funcs()

        # get the surface coordinates from the mesh component
        points = self.mesh_aero.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)
        self.geometry.nom_add_discipline_coords("struct")

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh_aero.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # we don't change the first 3 FFD layers from the root
        shapeStartIdx = 3

        # select the FFD points to move
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[:, :, shapeStartIdx:].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", axis="x", pointSelect=PS)

        # Create reference axis for the twist variable
        nRefAxPts = self.geometry.nom_addRefAxis(name="bladeAxis", yFraction=0.5, alignIndex="k", volumes=[0])

        # Set up global design variables. We dont change the root twist
        def twist(val, geo):
            for i in range(shapeStartIdx, nRefAxPts):
                geo.rot_z["bladeAxis"].coef[i] = val[i - shapeStartIdx]

        self.geometry.nom_addGlobalDV(dvName="twist", value=np.array([0] * (nRefAxPts - shapeStartIdx)), func=twist)

        # Set up the span variable, here val[0] is the span change in m
        def span(val, geo):
            # coordinates for the reference axis
            refAxisCoef = geo.extractCoef("bladeAxis")
            # the relative location of a point in the ref axis
            # refAxisS[0] = 0, and refAxisS[-1] = 1
            refAxisS = geo.refAxis.curves[0].s
            deltaSpan = val[0]
            # linearly change the refAxis coef along the span
            for i in range(shapeStartIdx, nRefAxPts):
                refAxisCoef[i, 2] += refAxisS[i - shapeStartIdx] * deltaSpan
            geo.restoreCoef(refAxisCoef, "bladeAxis")

        self.geometry.nom_addGlobalDV(dvName="span", value=np.array([0]), func=span)

        # chord var
        def chord(val, geo):
            for i in range(shapeStartIdx, nRefAxPts):
                geo.scale_y["bladeAxis"].coef[i] = val[i - shapeStartIdx]

        self.geometry.nom_addGlobalDV(dvName="chord", value=np.array([1] * (nRefAxPts - shapeStartIdx)), func=chord)

        def MRF(val, DASolver):
            omega = float(val[0])
            # we need to update the U value only
            DASolver.setOption("primalBC", {"MRF": omega})
            DASolver.updateDAOption()

        self.hover0.coupling.aero.solver.add_dv_func("MRF", MRF)
        self.hover0.aero_post.add_dv_func("MRF", MRF)

        self.hover1.coupling.aero.solver.add_dv_func("MRF", MRF)
        self.hover1.aero_post.add_dv_func("MRF", MRF)

        self.hover2.coupling.aero.solver.add_dv_func("MRF", MRF)
        self.hover2.aero_post.add_dv_func("MRF", MRF)

        # setup the volume and thickness constraints
        leList = [[-0.0034, -0.013, 0.03], [-0.0034, -0.013, 0.148]]
        teList = [[0.00355, 0.0133, 0.03], [0.00355, 0.0133, 0.148]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=20, nChord=10)
        # self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=20, nChord=10)

        self.geometry.nom_addCurvatureConstraint1D(
            "curvature1",
            start=[0, 0, 0.02],
            end=[0, 0, 0.149],
            nPts=50,
            axis=[1, 0, 0],
            curvatureType="mean",
            scaled=True,
        )

        self.geometry.nom_addCurvatureConstraint1D(
            "curvature2",
            start=[0, 0, 0.02],
            end=[0, 0, 0.149],
            nPts=50,
            axis=[-1, 0, 0],
            curvatureType="mean",
            scaled=True,
        )

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("MRF0", val=np.array([omega0[0]]))
        self.dvs.add_output("MRF1", val=np.array([omega0[1]]))
        self.dvs.add_output("MRF2", val=np.array([omega0[2]]))
        self.dvs.add_output("twist", val=np.array([0] * (nRefAxPts - shapeStartIdx)))
        self.dvs.add_output("chord", val=np.array([1] * (nRefAxPts - shapeStartIdx)))
        self.dvs.add_output("span", val=np.array([0]))
        # manually connect the dvs output to the geometry and hover
        self.connect("shape", "geometry.shape")
        self.connect("MRF0", "hover0.MRF")
        self.connect("MRF1", "hover1.MRF")
        self.connect("MRF2", "hover2.MRF")
        self.connect("twist", "geometry.twist")
        self.connect("span", "geometry.span")
        self.connect("chord", "geometry.chord")

        # define the design variables
        self.add_design_var("MRF0", lower=0.8 * omega0[0], upper=1.2 * omega0[0], scaler=0.1)
        self.add_design_var("MRF1", lower=0.8 * omega0[1], upper=1.2 * omega0[1], scaler=0.1)
        self.add_design_var("MRF2", lower=0.8 * omega0[2], upper=1.2 * omega0[2], scaler=0.1)
        if args.case >= 1:
            self.add_design_var("twist", lower=-50.0, upper=50.0, scaler=0.1)
        if args.case >= 2:
            self.add_design_var("shape", lower=-0.05, upper=0.05, scaler=100.0)
        if args.case >= 3:
            self.add_design_var("chord", lower=0.5, upper=2.0, scaler=1)
        if args.case >= 4:
            self.add_design_var("span", lower=-0.1, upper=0.1, scaler=100)

        # add objective and constraints to the top level
        self.add_objective("obj.value", scaler=1.0)
        self.connect("hover0.aero_post.power", "obj.power0")
        self.connect("hover1.aero_post.power", "obj.power1")
        self.connect("hover2.aero_post.power", "obj.power2")

        self.add_constraint("hover0.aero_post.thrust", equals=thrust_target[0], scaler=1.0)
        self.add_constraint("hover1.aero_post.thrust", equals=thrust_target[1], scaler=1.0)
        self.add_constraint("hover2.aero_post.thrust", equals=thrust_target[2], scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.8, upper=3.0, scaler=1.0)
        # self.add_constraint("geometry.volcon", lower=-100, upper=1.1, scaler=1.0)
        self.add_constraint("hover0.aero_post.skewness", upper=4.0, scaler=1.0)
        self.add_constraint("hover0.aero_post.nonOrtho", upper=80.0, scaler=0.1)
        self.add_constraint("hover2.ks_vmfailure", lower=-100.0, upper=0.5, scaler=1.0)
        self.add_constraint("hover0.mass", lower=-100.0, upper=m0 * 1.1, scaler=10.0)
        self.add_constraint("geometry.curvature1", upper=1.5, scaler=1.0)
        self.add_constraint("geometry.curvature2", upper=1.5, scaler=1.0)


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero_struct.html")

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
        "Linesearch tolerance": 0.99,
        "Hessian updates": 10000,
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
# prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"


if args.task == "opt":
    # solve CL
    # optFuncs.findFeasibleDesign(
    #    ["hover.aero_post.functionals.thrust"], ["MRF"], targets=[thrust_target], epsFD=[1e-2], tol=1e-3
    # )
    # run the optimization
    prob.run_driver()
elif args.task == "runPrimal":
    # just run the primal once
    prob.run_model()
elif args.task == "runAdjoint":
    # just run the primal and adjoint once
    prob.run_model()
    totals = prob.compute_totals()
    if MPI.COMM_WORLD.rank == 0:
        print(totals)
elif args.task == "checkTotals":
    # verify the total derivatives against the finite-difference
    prob.run_model()
    prob.check_totals(
        # of=["hover.aero_post.functionals.power", "hover.aero_post.functionals.thrust"],
        wrt=["span"],
        compact_print=False,
        step=1e-3,
        form="central",
        step_calc="abs",
    )
else:
    print("task arg not found!")
    exit(1)
