#!/usr/bin/env python
"""
This script reads a coarse airfoil profile, refine the profile using spline function,
and outputs it as surfaceMesh.xyz. Then it generate a 3D volume mesh with nSpan layers
in the z direction using pyHyp, available at https://github.com/mdolab/pyhyp
Note: the airfoil data should be seperated into PS and SS surfaces, they should start from the
LE and ends at TE. We use blunt TE so truncate the PS and SS data at about 99.8% of the chord.
"""

from pyhyp import pyHyp
import numpy
import openvsp as vsp

vsp.ClearVSPModel()
vsp.ReadVSPFile("airfoil.vsp3")
vsp.Update()

vsp.ExportFile("surfaceMesh.xyz", vsp.SET_ALL, vsp.EXPORT_PLOT3D)

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
    "N": 40,
    "s0": 5e-3,
    "marchDist": 20.0,
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
