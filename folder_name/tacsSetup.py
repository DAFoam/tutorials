from __future__ import print_function
import os

# ==============================================================================
# External Python modules
# ==============================================================================
from pprint import pprint

import numpy as np
from tacs import functions, constitutive, elements

# Material properties
rho = 1170  # density, kg/m^3
E = 2.5e9  # elastic modulus, Pa
nu = 0.33  # poisson's ratio
ys = 70e6  # yield stress, Pa

# Rotational velocity vector (rad/s)
omega = np.array([523.6, 0.0, 0.0])
# Rotation center (center of disk
rotCenter = np.zeros(3)

def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.SolidConstitutive(prop, t=1.0, tNum=dvNum)

    model = elements.LinearElasticity3D(con)
    basis = elements.LinearTetrahedralBasis()
    elem = elements.Element3D(model, basis)

    return elem


def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """
    # Add TACS Functions
    # Only include mass from elements that belong to pytacs components (i.e. skip concentrated masses)
    problem.addFunction("mass", functions.StructuralMass)
    problem.addFunction("ks_vmfailure", functions.KSFailure, safetyFactor=0.5, ksWeight=1000.0)

    # Add centrifugal load
    # problem.addCentrifugalLoad(omega, rotCenter)
