import numpy as np
import sys


def writeFFDFile(fileName, nBlocks, nx, ny, nz, points):
    """
    Take in a set of points and write the plot 3dFile
    """

    f = open(fileName, "w")

    f.write("%d\n" % nBlocks)
    for i in range(nBlocks):
        f.write("%d %d %d " % (nx[i], ny[i], nz[i]))
    # end
    f.write("\n")
    for block in range(nBlocks):
        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write("%f " % points[block][i, j, k, 0])
                # end
            # end
        # end
        f.write("\n")

        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write("%f " % points[block][i, j, k, 1])
                # end
            # end
        # end
        f.write("\n")

        for k in range(nz[block]):
            for j in range(ny[block]):
                for i in range(nx[block]):
                    f.write("%f " % points[block][i, j, k, 2])
                # end
            # end
        # end
    # end
    f.close()
    return


def returnBlockPoints(corners, nx, ny, nz):
    """
    corners needs to be 8 x 3
    """
    points = np.zeros([nx, ny, nz, 3])

    # points 1 - 4 are the iMin face
    # points 5 - 8 are the iMax face

    for idim in range(3):
        edge1 = np.linspace(corners[0][idim], corners[4][idim], nx)
        edge2 = np.linspace(corners[1][idim], corners[5][idim], nx)
        edge3 = np.linspace(corners[2][idim], corners[6][idim], nx)
        edge4 = np.linspace(corners[3][idim], corners[7][idim], nx)

        for i in range(nx):
            edge5 = np.linspace(edge1[i], edge3[i], ny)
            edge6 = np.linspace(edge2[i], edge4[i], ny)
            for j in range(ny):
                edge7 = np.linspace(edge5[j], edge6[j], nz)
                points[i, j, :, idim] = edge7
            # end
        # end
    # end

    return points


# We have two blocks for this case
nBlocks = 2
corners = np.zeros([nBlocks, 8, 3])

nx = [10, 2]
ny = [10, 10]
nz = [2, 2]

################ wing FFD ##############
iVol = 0
rootChordRef = 11.8
tipChordRef = 2.7
rootThicknessMax = 3.8
tipThicknessMax = 2.2
Span = 29.0

leRoot = [25.2, 3.2, 4.2]
leTip = [45.1, 29.4, 6.7]

teRoot = [36.5, 3.2, 3.4]
teTip = [48.5, 29.4, 6.9]

corners[iVol, 0, :] = [leRoot[0] - rootChordRef * 0.1, leRoot[1] - Span * 0.03, leRoot[2] - rootThicknessMax * 0.3]
corners[iVol, 1, :] = [leRoot[0] - rootChordRef * 0.1, leRoot[1] - Span * 0.03, leRoot[2] + rootThicknessMax * 0.3]
corners[iVol, 2, :] = [leTip[0] - tipChordRef * 0.1, leTip[1] + Span * 0.03, leTip[2] - tipThicknessMax * 0.3]
corners[iVol, 3, :] = [leTip[0] - tipChordRef * 0.1, leTip[1] + Span * 0.03, leTip[2] + tipThicknessMax * 0.3]
corners[iVol, 4, :] = [teRoot[0] + rootChordRef * 0.1, teRoot[1] - Span * 0.03, teRoot[2] - rootThicknessMax * 0.3]
corners[iVol, 5, :] = [teRoot[0] + rootChordRef * 0.1, teRoot[1] - Span * 0.03, teRoot[2] + rootThicknessMax * 0.3]
corners[iVol, 6, :] = [teTip[0] + tipChordRef * 0.1, teTip[1] + Span * 0.03, teTip[2] - tipThicknessMax * 0.3]
corners[iVol, 7, :] = [teTip[0] + tipChordRef * 0.1, teTip[1] + Span * 0.03, teTip[2] + tipThicknessMax * 0.3]


################ tail FFD ##############
iVol = 1
rootChordRef = 5.7
tipChordRef = 2.5
rootThicknessMax = 2.5
tipThicknessMax = 1.5
Span = 10

leRoot = [56.7, 1.8, 6.6]
leTip = [64.3, 10.8, 7.7]

teRoot = [62.5, 0.4, 6.6]
teTip = [67.0, 10.8, 7.7]

corners[iVol, 0, :] = [leRoot[0] - rootChordRef * 0.1, leRoot[1] - Span * 0.03, leRoot[2] - rootThicknessMax * 0.3]
corners[iVol, 1, :] = [leRoot[0] - rootChordRef * 0.1, leRoot[1] - Span * 0.03, leRoot[2] + rootThicknessMax * 0.3]
corners[iVol, 2, :] = [leTip[0] - tipChordRef * 0.1, leTip[1] + Span * 0.03, leTip[2] - tipThicknessMax * 0.3]
corners[iVol, 3, :] = [leTip[0] - tipChordRef * 0.1, leTip[1] + Span * 0.03, leTip[2] + tipThicknessMax * 0.3]
corners[iVol, 4, :] = [teRoot[0] + rootChordRef * 0.1, teRoot[1] - Span * 0.03, teRoot[2] - rootThicknessMax * 0.3]
corners[iVol, 5, :] = [teRoot[0] + rootChordRef * 0.1, teRoot[1] - Span * 0.03, teRoot[2] + rootThicknessMax * 0.3]
corners[iVol, 6, :] = [teTip[0] + tipChordRef * 0.1, teTip[1] + Span * 0.03, teTip[2] - tipThicknessMax * 0.3]
corners[iVol, 7, :] = [teTip[0] + tipChordRef * 0.1, teTip[1] + Span * 0.03, teTip[2] + tipThicknessMax * 0.3]


points = []
for block in range(nBlocks):
    points.append(returnBlockPoints(corners[block], nx[block], ny[block], nz[block]))

# print points
fileName = "wingTailFFD.xyz"
writeFFDFile(fileName, nBlocks, nx, ny, nz, points)