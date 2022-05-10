from pyhyp import pyHyp

fileName = "surfMesh.cgns"
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
    "N": 53,
    "s0": 1.0e-4,
    "marchDist": 25 * 3.758151,
    #'nConstantStart':0,
    # ---------------------------
    #   Pseudo Grid Parameters
    # ---------------------------
    "ps0": -1.0,
    "pGridRatio": 1.1,
    "cMax": 5.0,
    # ---------------------------
    #   Smoothing parameters
    # ---------------------------
    "epsE": 1.0,
    "epsI": 2.0,
    "theta": 3.0,
    "volCoef": 0.16,
    "volBlend": 0.0005,
    "volSmoothIter": 30,
    "kspRelTol": 1e-4,
    "kspMaxIts": 50,
    "kspSubspaceSize": 50,
}

hyp = pyHyp(options=options)
hyp.run()
# hyp.writeCGNS('volumeMesh.cgns')
hyp.writePlot3D("volumeMesh.xyz")
