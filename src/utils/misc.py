# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 07:01:23 2024

@author: Jesús Pueblas
"""

import numpy as np
from scipy import *
from scipy.integrate import quad
from scipy.integrate import simpson
from math import erf
from sys import exit
from randw import *

# Returns the initial spectrum for a given length wave
def E0(k):
    # Constant to adjust the turbulent intensity to 0.7%
    A = 0.000060605
    slope = -5/3
    Ek = 0
    if (k>=1 and k<=5):
        Ek = A * 5**slope
    elif (k>5):
        Ek = A * k**slope
    return Ek

# Fill initial solution
def FillInitialSolution(U,x,IniS,Np,p,Nref):
    N=len(U)
    if (IniS=="SINE"):
        for i in range(0,N):
            U[i] = np.sin(2.*np.pi*x[i])
    elif (IniS=="GAUSSIAN"):
        for i in range(0,N):
            U[i] = np.exp(-0.5*((x[i]-0.5)/0.1)**2)
    elif (IniS=="SQUARE"):
        for i in range(0,N):
            xi = x[i]
            U[i] = 0.
            if (xi>=0.25 and xi<0.75):
                U[i] = 1.
    elif (IniS=="TURBULENT"):
        Nharms = 1280
        for i in range(0,N):
            U[i] = 1.
        for k in range(1,Nharms+1):
            Ek = E0(k)
            beta = np.random.uniform(0.,2*np.pi)
            for i in range(0,N):
                U[i] += np.sqrt(2.*Ek)*np.sin(2.*k*np.pi*x[i]+beta)
        
        umean = 0.
        for i in range(0,N):
            umean += U[i]
        umean /= N
        
        up = 0
        
        for i in range(0,N):
            up += (U[i]-umean)**2
        up /= N
        up = np.sqrt(up)
        print("umean:",umean,"up:",up)
    else:
        # asume is a file name
        inputfile=open(IniS,'r')
        document = inputfile.readlines()
        inputfile.close()

        # Read the contents of the input file
        Nd     = int(getValueFromLabel(document,"N"))
        pd     = int(getValueFromLabel(document,"P"))
        Nrefd  = int(getValueFromLabel(document,"NREF"))
        if (Nd != Np or pd != p or Nrefd != Nref):
            print("Initial solution not compatible with current parameters")
            exit()
        xd,Ud = GetMeshAndSolution(document)
        for i in range(0,N):
            U[i] = Ud[i]
        
        
            
