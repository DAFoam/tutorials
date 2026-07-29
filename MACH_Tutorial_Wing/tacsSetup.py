import numpy as np
from tacs import elements, constitutive, functions

# Material properties
rho = 2780.0  # density, kg/m^3
E = 73.1e9  # elastic modulus, Pa
nu = 0.33  # poisson's ratio
kcorr = 5.0 / 6.0  # shear correction factor
ys = 324.0e6  # yield stress, Pa

# Shell thickness
t = 0.003  # m
tMin = 0.002  # m
tMax = 0.05  # m


def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum, tlb=tMin, tub=tMax)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = None
    elem = elements.Quad4Shell(transform, con)

    return elem


def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """
    # Add TACS Functions
    # Only include mass from elements that belong to pytacs components (i.e. skip concentrated masses)
    problem.addFunction("mass", functions.StructuralMass)
    problem.addFunction("ks_vmfailure", functions.KSFailure, safetyFactor=1.0, ksWeight=50.0)

    # Add gravity load
    g = np.array([0.0, 0.0, -9.81])  # m/s^2
    problem.addInertialLoad(g)
