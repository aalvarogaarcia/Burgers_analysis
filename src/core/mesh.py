# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 06:15:23 2024

@author: Jesús Pueblas
"""

import numpy as np

# Get the mesh point coordinates
# Inputs:
#   N   : Number of mesh points
#   Nref: Number of mesh refinement
# Output:
#   x   : coordinates of the mesh
def getMesh(N,Nref):
    x = np.zeros(N)
    dx = 1./(N-1)
    for i in range(0,N):
        x[i] = i*dx

    for i in range(0,Nref):
        xold = np.zeros(N)
        for j in range(0,N):
            xold[j] = x[j]
        Nold = N
        N = 2*Nold - 1
        x = np.zeros(N)
        for j in range(0,Nold-1):
            x[2*j]   = xold[j]
            x[2*j+1] = 0.5*(xold[j]+xold[j+1])
        x[N-1] = xold[Nold-1]
    return x

# Get the mesh for high-order scheme
# Inputs:
#   x     : coordinates of the mesh
#   coords: local element coordinates
# Output:
#   xho   : coordinates of the mesh with the element inner points

def getMeshHO(x,coords):
    nElements = len(x)-1
    p         = len(coords)-1
    xho = np.zeros(nElements*(p+1))
    for i in range(0,nElements):
        xA = x[i]
        xB = x[i+1]
        for j in range(0,p+1):
            xho[j+i*(p+1)] = xA + 0.5*(xB-xA)*(1.+coords[j])
    return xho

