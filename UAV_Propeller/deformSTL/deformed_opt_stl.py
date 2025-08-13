#!/usr/bin/env python
"""
Deform the STL based on the optimized design vars
"""

import sys
from stl import mesh
import numpy as np
from pygeo import DVGeometry, geo_utils
import json

# which case to deform
# case1: twist only
# case2: twist+shape
# case3: twist+shape+chord
# case4: twist+shape+chord+span
case = 4

# read in the FFD
DVGeo = DVGeometry("FFD.xyz")

# now we can read in the design variables.
# NOTE: the design variable should define only one blade simulated in CFD. we will manually
# map them to the other blade
with open("./designVars.json") as f:
    dvs = json.load(f)

# read in the baseline stl file in meters
myMesh = mesh.Mesh.from_file("baseline_meters.stl")

print("Deforming baseline_meters.stl")

# read the point cloud from the stl file and add it to the DVGeo pointSet
nFaces = len(myMesh.x)
surfaceCoordinates = []
for i in range(nFaces):
    for j in range(3):
        coords = [myMesh.x[i][j], myMesh.y[i][j], myMesh.z[i][j]]
        surfaceCoordinates.append(coords)
DVGeo.addPointSet(surfaceCoordinates, "surfacePoints")

# DV setup. NOTE: this needs to be exact same as the runScript.py, including the scaling
# NOTE: because we have two blades in the STL while we simulated one blade in CFD
# Here we need to duplicate the DVGeo setting from runScript.py, one for each blade
# NOTE: we have rotate the blade FFD by 180, we have three FFD blocks as follows
# block 1: blade 1
# block 2: blade 2
# block 3: spinner

shapeStartIdx = 3

nRefAxPts0 = DVGeo.addRefAxis("bladeAxis0", yFraction=0.5, alignIndex="k", volumes=[0])
nRefAxPts1 = DVGeo.addRefAxis("bladeAxis1", yFraction=0.5, alignIndex="k", volumes=[1])


def twist0(val, geo):
    for i in range(shapeStartIdx, nRefAxPts0):
        geo.rot_z["bladeAxis0"].coef[i] = val[i - shapeStartIdx]


def twist1(val, geo):
    for i in range(shapeStartIdx, nRefAxPts1):
        geo.rot_z["bladeAxis1"].coef[i] = val[i - shapeStartIdx]


def chord0(val, geo):
    for i in range(shapeStartIdx, nRefAxPts0):
        geo.scale_y["bladeAxis0"].coef[i] = val[i - shapeStartIdx]


def chord1(val, geo):
    for i in range(shapeStartIdx, nRefAxPts1):
        geo.scale_y["bladeAxis1"].coef[i] = val[i - shapeStartIdx]


def span0(val, geo):
    # coordinates for the reference axis
    refAxisCoef = geo.extractCoef("bladeAxis0")
    # the relative location of a point in the ref axis
    # refAxisS[0] = 0, and refAxisS[-1] = 1
    refAxisS = geo.refAxis.curves[0].s
    deltaSpan = val[0]
    # linearly change the refAxis coef along the span
    for i in range(shapeStartIdx, nRefAxPts0):
        refAxisCoef[i, 2] += refAxisS[i - shapeStartIdx] * deltaSpan
    geo.restoreCoef(refAxisCoef, "bladeAxis0")


def span1(val, geo):
    # coordinates for the reference axis
    refAxisCoef = geo.extractCoef("bladeAxis1")
    # the relative location of a point in the ref axis
    # refAxisS[0] = 0, and refAxisS[-1] = 1
    refAxisS = geo.refAxis.curves[0].s
    deltaSpan = val[0]
    # linearly change the refAxis coef along the span
    for i in range(shapeStartIdx, nRefAxPts1):
        refAxisCoef[i, 2] += refAxisS[i - shapeStartIdx] * deltaSpan
    geo.restoreCoef(refAxisCoef, "bladeAxis1")


# twist variables
if case >= 1:

    DVGeo.addGlobalDV("twist0", np.zeros(nRefAxPts0 - shapeStartIdx), twist0, lower=-50.0, upper=50.0, scale=0.1)
    DVGeo.addGlobalDV("twist1", np.zeros(nRefAxPts1 - shapeStartIdx), twist1, lower=-50.0, upper=50.0, scale=0.1)

    # enforce the twist symmetry
    dvs["twist1"] = []
    for i in range(len(dvs["twist0"])):
        dvs["twist1"].append(-dvs["twist0"][i])

    # shape variables
if case >= 2:

    # blade 0
    pts = DVGeo.getLocalIndex(0)
    indexList = pts[:, :, shapeStartIdx:].flatten()
    PS = geo_utils.PointSelect("list", indexList)
    DVGeo.addLocalDV("shape0", lower=-1e8, upper=1e8, axis="x", scale=100.0, pointSelect=PS)
    # blade 1
    pts = DVGeo.getLocalIndex(1)
    indexList = pts[:, :, shapeStartIdx:].flatten()
    PS = geo_utils.PointSelect("list", indexList)
    DVGeo.addLocalDV("shape1", lower=-1e8, upper=1e8, axis="x", scale=100.0, pointSelect=PS)

    # enforce the shape symmetry
    dvs["shape1"] = []
    for i in range(len(dvs["shape0"])):
        dvs["shape1"].append(dvs["shape0"][i])


# chord variables
if case >= 3:
    DVGeo.addGlobalDV("chord0", np.ones(nRefAxPts0 - shapeStartIdx), chord0, lower=0.5, upper=2.0, scale=1)
    DVGeo.addGlobalDV("chord1", np.ones(nRefAxPts1 - shapeStartIdx), chord1, lower=0.5, upper=2.0, scale=1)

    # enforce the chord symmetry
    dvs["chord1"] = []
    for i in range(len(dvs["chord0"])):
        dvs["chord1"].append(dvs["chord0"][i])

# span variables
if case >= 4:
    DVGeo.addGlobalDV("span0", np.zeros(1), span0, lower=-0.1, upper=0.1, scale=100)
    DVGeo.addGlobalDV("span1", np.zeros(1), span1, lower=-0.1, upper=0.1, scale=100)

    # enforce the span symmetry
    dvs["span1"] = []
    dvs["span1"].append(-dvs["span0"][0])

# update the design vars in DVGeo and update the design surface
DVGeo.setDesignVars(dvs)
newSurfaceCoordinates = DVGeo.update("surfacePoints")

# Now we can get the updated point cloud and assign them back to the stl file
counterI = 0
for i in range(nFaces):
    for j in range(3):
        myMesh.x[i][j] = newSurfaceCoordinates[counterI][0]
        myMesh.y[i][j] = newSurfaceCoordinates[counterI][1]
        myMesh.z[i][j] = newSurfaceCoordinates[counterI][2]
        counterI += 1

print("Saving the deformed geometry to deformed_meters.stl")
myMesh.save("deformed_meters.stl")
