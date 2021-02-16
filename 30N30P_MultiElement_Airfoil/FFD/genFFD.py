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


# We have three blocks for this case
nBlocks = 3
corners = np.zeros([nBlocks, 8, 3])

nx = [5, 5, 5]
ny = [2, 2, 2]
nz = [2, 2, 2]

################ slat FFD ##############
corners[0,0,:] = [-0.025,-0.131,0.0]
corners[0,1,:] = [-0.025,-0.131,0.1]
corners[0,2,:] = [-0.109,-0.110,0.0]
corners[0,3,:] = [-0.109,-0.110,0.1]
corners[0,4,:] = [ 0.033,0.008,0.0]
corners[0,5,:] = [ 0.033,0.008,0.10]
corners[0,6,:] = [ 0.003,0.026,0.0]
corners[0,7,:] = [ 0.003,0.026,0.1]

################ main FFD ##############
corners[1,0,:] = [0.036,-.0700,0.0]
corners[1,1,:] = [0.036,-0.0700,0.1]
corners[1,2,:] = [0.036,0.0700,0.0]
corners[1,3,:] = [0.036,0.0700,0.1]
corners[1,4,:] = [0.80,-.07,0.0]
corners[1,5,:] = [0.80,-.07,0.10]
corners[1,6,:] = [0.91,0.07,0.0]
corners[1,7,:] = [0.91,0.07,0.1]

################ flap FFD ##############
corners[2,0,:] = [0.843,-0.017,0.0]
corners[2,1,:] = [0.843,-0.017,0.1]
corners[2,2,:] = [0.897,0.050,0.0]
corners[2,3,:] = [0.897,0.050,0.1]
corners[2,4,:] = [1.125,-0.176,0.0]
corners[2,5,:] = [1.125,-0.176,0.10]
corners[2,6,:] = [1.157,-0.125,0.0]
corners[2,7,:] = [1.157,-0.125,0.1]


points = []
for block in range(nBlocks):
    points.append(returnBlockPoints(corners[block], nx[block], ny[block], nz[block]))

# print points
fileName = "airfoilFFD.xyz"
writeFFDFile(fileName, nBlocks, nx, ny, nz, points)
