#!/usr/bin/env python

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys

fileName=sys.argv[1]

f=open(fileName, "r")
lines=f.readlines()
f.close()

CD=[]
CL=[]
for line in lines:
    if "CD-part1-force:" in line:
        cols=line.split()
        CD.append(float(cols[1]))

    if "CL-part1-force:" in line:
        cols=line.split()
        CL.append(float(cols[1]))

fig, ax = plt.subplots(figsize=(20, 4), nrows=1)
plt.plot(CL[:])
#plt.xlim([0, 3])
#plt.ylim([-0.5, 0.5])
plt.xlabel("Iters")
plt.ylabel("$C_L$")
plt.savefig("TimeSeriesCL.pdf", bbox_inches="tight")  # save the figure to file
plt.close()  # close the figure

fig, ax = plt.subplots(figsize=(20, 4), nrows=1)
plt.plot(CD[:])
#plt.xlim([0, 3])
#plt.ylim([0.15, 0.3])
plt.xlabel("Iters")
plt.ylabel("$C_D$")
plt.savefig("TimeSeriesCD.pdf", bbox_inches="tight")  # save the figure to file
plt.close()  # close the figure
