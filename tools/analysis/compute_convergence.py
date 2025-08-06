# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:24:55 2024

@author: Jesús Pueblas
"""

from sys import argv
from randw import *
import matplotlib.pyplot as plt
from mesh import *

# Computes the L2Norm betwee two solutions
# contained in documenta and documentb
# documentb is one refinement level from documenta
def getL2Norm(documenta,documentb):
    dof    = 0.
    l2norm = 0.
    Na      = int(getValueFromLabel(documenta,"N"))
    Nb      = int(getValueFromLabel(documentb,"N"))
    pa      = int(getValueFromLabel(documenta,"P"))
    pb      = int(getValueFromLabel(documentb,"P"))
    Nrefa   = int(getValueFromLabel(documenta,"NREF"))
    Nrefb   = int(getValueFromLabel(documentb,"NREF"))
    if (Na != Nb):
        return dof,l2norm
    if (pa != pb):
        return dof,l2norm
    if (Nrefb-Nrefa != 1):
        return dof,l2norm
    
    # Create the cells mesh from documenta
    xa = getMesh(Na,Nrefa)
    dof = len(xa)
    # Read the mesh and solution from docuemnta
    xsa,usa = GetMeshAndSolution(documenta)
    # Read the mesh and solution from docuemntb
    xsb,usb = GetMeshAndSolution(documentb)
    #Take the solution of the boundary cells for mesh a
    nnodea = len(usa)
    ncellsa = int(nnodea / (pa+1))
    ua = np.zeros(dof)
    ua[0] = usa[0]
    ua[dof-1] = usa[nnodea-1]
    for icell in range(0,ncellsa-1):
        inodeL = icell*(pa+1)+pa
        inodeR = (icell+1)*(pa+1)
        ua[icell+1] = 0.5*(usa[inodeL]+usa[inodeR])

    #Take the solution of the boundary cells for mesh b
    nnodeb = len(usb)
    ncellsb = int(nnodeb / (pb+1))
    ub = np.zeros(dof)
    ub[0] = usb[0]
    ub[dof-1] = usb[nnodeb-1]
    for icell in range(0,ncellsa-1):
        inodeL = 2*icell*(pa+1)+2*(pa+1)-1
        inodeR = inodeL+1
        ub[icell+1] = 0.5*(usb[inodeL]+usb[inodeR])
        
    for inode in range(0,dof):
        delta = ub[inode] - ua[inode]
        l2norm += delta * delta
    
    l2norm = np.sqrt(l2norm/dof)
    dof = len(xsa)

    return dof,l2norm    
    

documents = []
for i in range(1,len(argv)):
  inputfile=open(argv[i],'r')
  documents.append(inputfile.readlines()) 
  inputfile.close()

dof    = np.zeros(len(documents)-1)
l2norm = np.zeros(len(documents)-1)

for i in range(0,len(documents)-1):
  dofi,l2normi = getL2Norm(documents[i],documents[i+1])
  print(dofi,l2normi)
  dof[i]    = 1./dofi
  l2norm[i] = l2normi

plt.plot(dof,l2norm,'o-',label="p=1")

doftrend = np.zeros(2)
l2etrend = np.zeros(2)

# Parameters for p=3
#c1 = 600
#p = 3

# Parameters for p=4
c1 = 2000
p = 2

doftrend[0] = 1./4
doftrend[1] = 1./1000
l2etrend[0] = c1*doftrend[0]**(p+1)
l2etrend[1] = c1*doftrend[1]**(p+1)
plt.plot(doftrend,l2etrend,'--',label="(1/dof)^2")

plt.xlabel('1/dof')
plt.ylabel('l2norm(error)')
plt.legend(loc='upper left')
plt.xscale("log")
plt.yscale("log")
plt.show()
