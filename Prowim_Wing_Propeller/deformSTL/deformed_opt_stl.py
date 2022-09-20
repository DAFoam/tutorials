#!/usr/bin/env python
"""
Deform the STL based on the optimized design vars
"""

import sys
from stl import mesh
import numpy as np
from pygeo import DVGeometry, geo_utils
import json

# read in the FFD
DVGeo = DVGeometry("wingFFD.xyz")

# read in the stl file
myMesh = mesh.Mesh.from_file("wing.stl")

# parse the point cloud in the stl
nFaces = len(myMesh.x)
surfaceCoordinates = []
for i in range(nFaces):
    for j in range(3):
        coords = [myMesh.x[i][j], myMesh.y[i][j], myMesh.z[i][j]]
        surfaceCoordinates.append(coords)

# add the STL point cloud to pyGeo
DVGeo.addPointSet(surfaceCoordinates, "surfacePoints")

# Shape design variables setup. NOTE: this needs to be EXACTLY same as the runScript.py, including the scaling
# ---------
# nTwists is the number of FFD points in the spanwise direction
nTwists = DVGeo.addRefAxis("bodyAxis", xFraction=0.25, alignIndex="k")
# twist function, we keep the root twist constant so the first
# element in the twist design variable is the twist at the 2nd
# spanwise location
def twist(val, geo):
    for i in range(1, nTwists):
        geo.rot_z["bodyAxis"].coef[i] = val[i - 1]

# select points
pts = DVGeo.getLocalIndex(0)
indexList = pts[:, :, :].flatten()
PS = geo_utils.PointSelect("list", indexList)
# shape
DVGeo.addLocalDV("shapey", lower=-1.0, upper=1.0, axis="y", scale=1.0, pointSelect=PS)
# twist
DVGeo.addGlobalDV("twist", np.zeros(nTwists - 1), twist, lower=-10.0, upper=10.0, scale=1.0)

# NO need to add the alpha variable because it does not impact the shape

# read in the design variable json file 
# (NOTE: the design variable values can be from a DAFoam optimization log)
with open("./designVars.json") as f:
    dvs = json.load(f)

print("Deforming the stl with new pyGeo DVs:", dvs)

# set the dv values in the json file to the pyGeo obj and update the point cloud
DVGeo.setDesignVars(dvs)
newSurfaceCoordinates = DVGeo.update("surfacePoints")

# get the new surface point cloud and assign it to the deformed stl file
counterI = 0
for i in range(nFaces):
    for j in range(3):
        myMesh.x[i][j] = newSurfaceCoordinates[counterI][0]
        myMesh.y[i][j] = newSurfaceCoordinates[counterI][1]
        myMesh.z[i][j] = newSurfaceCoordinates[counterI][2]
        counterI += 1

myMesh.save("deformed.stl")

print("Done. File saved to deformed.stl")
