#!/usr/bin/env python
"""
This script reads a coarse airfoil profile, refine the profile using spline function,
and outputs it as surfaceMesh.xyz. Then it generate a 3D volume mesh with nSpan layers
in the z direction using pyHyp, available at https://github.com/mdolab/pyhyp
Note: the airfoil data should be seperated into PS and SS surfaces, they should start from the
LE and ends at TE. We use blunt TE so truncate the PS and SS data at about 99.8% of the chord.
"""

from pyhyp import pyHyp
import numpy as np
from pyspline import *

R = 0.5
zSpan = 0.1
nPoints = 50
NpExtrude = 50
yWall = 1e-3
marchDist = 30.0

x = []
y = []
for i in range(nPoints):
    theta = 2 * np.pi / nPoints * i
    x.append(R * np.cos(theta))
    y.append(R * np.sin(theta))

x.append(R * np.cos(0))
y.append(R * np.sin(0))

# Write the plot3d input file:
f = open("surfaceMesh.xyz", "w")
f.write("1\n")
f.write("%d %d %d\n" % (nPoints + 1, 2, 1))
for iDim in range(3):
    for z in [0, zSpan]:
        for i in range(nPoints + 1):
            if iDim == 0:
                f.write("%20.16f\n" % x[i])
            elif iDim == 1:
                f.write("%20.16f\n" % y[i])
            else:
                f.write("%20.16f\n" % z)
f.close()

options = {
    # ---------------------------
    #        Input Parameters
    # ---------------------------
    "inputFile": "surfaceMesh.xyz",
    "unattachedEdgesAreSymmetry": False,
    "outerFaceBC": "farfield",
    "autoConnect": True,
    "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
    "families": "wall",
    # ---------------------------
    #        Grid Parameters
    # ---------------------------
    "N": NpExtrude,
    "s0": yWall,
    "marchDist": marchDist,
    # ---------------------------
    #   Pseudo Grid Parameters
    # ---------------------------
    "ps0": -1.0,
    "pGridRatio": -1.0,
    "cMax": 1.0,
    # ---------------------------
    #   Smoothing parameters
    # ---------------------------
    "epsE": 2.0,
    "epsI": 4.0,
    "theta": 2.0,
    "volCoef": 0.20,
    "volBlend": 0.0005,
    "volSmoothIter": 20,
}


hyp = pyHyp(options=options)
hyp.run()
hyp.writePlot3D("volumeMesh.xyz")
