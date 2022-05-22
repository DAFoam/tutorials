from pyhyp import pyHyp

fileName = "surfaceMesh.cgns"

options = {
    # ---------------------------
    #        Input Parameters
    # ---------------------------
    "inputFile": fileName,
    "fileType": "CGNS",
    "unattachedEdgesAreSymmetry": True,
    "outerFaceBC": "farfield",
    "autoConnect": True,
    "BC": {},
    "families": "wall",
    # ---------------------------
    #        Grid Parameters
    # ---------------------------
    "N": 65,
    "s0": 3.0e-4,
    "marchDist": 30.0,
    #'nConstantStart':1,
    # ---------------------------
    #   Pseudo Grid Parameters
    # ---------------------------
    "ps0": -1.0,
    "pGridRatio": -1.0,
    "cMax": 0.1,
    # ---------------------------
    #   Smoothing parameters
    # ---------------------------
    #"epsE": 0.5,
    #"epsI": 1.0,
    #"theta": 2.0,
    #"volCoef": 0.20,
    #"volBlend": 0.0005,
    #"volSmoothIter": 20,
    #'kspreltol':1e-4,
}

hyp = pyHyp(options=options)
hyp.run()
hyp.writePlot3D("volumeMesh.xyz")

