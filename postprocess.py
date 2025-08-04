# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:24:55 2024

@author: Jesús Pueblas
"""

from sys import argv
from randw import *
from lagpol import *
import matplotlib.pyplot as plt

def collpaseCommonNodes(x,U,p):
    Ne = int(len(x) / (p+1))
    newSize = Ne*p + 1
    xc = np.zeros(newSize)
    Uc = np.zeros(newSize)
    xc[0] = x[0]
    Uc[0] = (U[0]+U[-1])*0.5
    
    ip = 1
    for icell in range(0,Ne):
        iL = icell*(p+1)
        for jp in range(1,p):
            xc[ip] = x[iL+jp]
            Uc[ip] = U[iL+jp]
            ip += 1
        iR = icell*(p+1) + p
        iN = iR + 1
        if (iN > len(x)-1):
            iN = 0
        um = (U[iR]+U[iN])*0.5
        xc[ip] = x[iR]
        Uc[ip] = um
        ip += 1
    return xc,Uc
            

def getSolutionInUniformMesh(x,U,p):
    monCoef = getLagrangeMonomialCoefficients(p)
    Ne = int(len(x) / (p+1))
    xeq = np.zeros(len(x))
    Ueq = np.zeros(len(x))
    for icell in range(0,Ne):
        iL = icell*(p+1)
        iR = iL + p
        xL = x[iL]
        xR = x[iR]
        dx = (xR - xL) / p
        for jp in range(0,p+1):
            xj = xL + jp*dx
            xeq[iL+jp] = xj
            chi = 2*(xj-xL)/(xR-xL)-1
            Uj = 0.
            for kp in range(0,p+1):
                Lj = getLagrangeValue(monCoef,chi,kp)
                Uj += U[iL+kp] * Lj
            Ueq[iL+jp] = Uj
    return xeq,Ueq
    
def getTKEFFT(x,U):
    N = len(U)
    X = np.abs(np.fft.fft(U)) / N
    k = np.linspace(0, N-1,N)
    kplot = k[0:int(N/2+1)]
    Xplot = 2 * X[0:int(N/2+1)]
    Xplot[0] /= 2
    
    for k in range(1,len(Xplot)):
        Xplot[k] = 0.25*Xplot[k]**2

    # Remove the DC since it is the mean value
    Xplot = np.delete(Xplot,0)
    kplot = np.delete(kplot,0)
    
    return kplot,Xplot

if __name__ == "__main__":
    fig, ax = plt.subplots(1,2)
    ax[0].set_title("Solution") 
    ax[1].set_title("Energy spectrum")

    for i in range(1,len(argv)):
      inputfile=open(argv[i],'r')
      documenti = inputfile.readlines()
      inputfile.close()
      
      xi,ui = GetMeshAndSolution(documenti)
      p     = int(getValueFromLabel(documenti,"P"))
      
      xe,ue = getSolutionInUniformMesh(xi,ui,p)
      xc,uc = collpaseCommonNodes(xe,ue,p)

      # Remove last element for proper FFT
      xt = xc[:-1]
      ut = uc[:-1]

      fi,ffti = getTKEFFT(xt,ut)
      
      plt.subplot(1,2,1)
      plt.plot(xi, ui,label=argv[i])
      plt.subplot(1,2,2)
      plt.plot(fi, ffti)

    plt.subplot(1,2,1)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(loc="upper right")
    plt.subplot(1,2,2)
    plt.xlabel("k")
    plt.ylabel("TKE")
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()
    
