"""
This is a modified version of the script used in Chauhan et al. 2021, 
"RANS-Based Aerodynamic Shape Optimization of a Wing Considering 
Propeller-Wing Interaction". If this script is used for published work,
please include a reference to Chauhan et al.
"""

import numpy as np
from pygeo import pyGeo

naf = 2

airfoil_list = ['NACA642A015.dat'] * naf

# Airfoil leading edge positions
x = np.linspace(0.0, 0.0, naf)
y = np.linspace(0.0, 0.0, naf)
z = np.linspace(0.0, 0.63, naf)

offset = np.zeros((naf,2)) # x-y offset applied to airfoil position before scaling

# Airfoil rotations
rot_x = [0.0] * naf
rot_y = [0.0] * naf
rot_z = [0.0] * naf

# Airfoil scaling
chord = [0.24] * naf # chord lengths

wing = pyGeo('liftingSurface', xsections=airfoil_list, scale=chord, offset=offset,
    x=x, y=y, z=z, rotX=rot_x, rotY=rot_y, rotZ=rot_z, tip='rounded', bluntTe=True,
    squareTeTip=True, teHeight=0.001)

wing.writeTecplot('wing.dat')
wing.writeIGES('wing.igs')