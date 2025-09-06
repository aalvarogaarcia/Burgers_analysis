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
from src.utils.randw import *

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
def FillInitialSolution_1D(U,x,IniS,Np,p,Nref):
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

def FillInitialSolution_2D(U, x, y, IniS, Nx, Ny, p, Nref):
    """
    Rellena el vector de estado U con una condición inicial 2D.
    U contiene 'u' y 'v' concatenados.
    """
    num_nodes_per_var = len(U) // 2
    u_view = U[:num_nodes_per_var]
    v_view = U[num_nodes_per_var:]

    x_grid, y_grid = x, y
    
    if len(x) != num_nodes_per_var or len(y) != num_nodes_per_var:
        print("ADVERTENCIA (FillInitialSolution_2D): Las coordenadas de entrada no coinciden con el tamaño de U. Se generará una rejilla interna.")
        # Crea el mapa completo usando meshgrid
        xx, yy = np.meshgrid(x, y)
        x_grid = xx.flatten()
        y_grid = yy.flatten()
        
        # Comprobación de seguridad final
        if len(x_grid) != num_nodes_per_var:
            raise ValueError(
                f"Error fatal en FillInitialSolution_2D: No se pudo crear una rejilla consistente. "
                f"Se esperaban {num_nodes_per_var} nodos, pero se generaron {len(x_grid)}."
            )
            
    
    if IniS == "TAYLOR_GREEN":
        # Vórtice de Taylor-Green: una solución analítica para Navier-Stokes
        # que es excelente para verificar la convergencia del código.
        # u(x,y) = sin(pi*x) * cos(pi*y)
        # v(x,y) = -cos(pi*x) * sin(pi*y)
        u_view[:] = np.sin(np.pi * x) * np.cos(np.pi * y)
        v_view[:] = -np.cos(np.pi * x) * np.sin(np.pi * y)
    
    elif IniS == "GAUSSIAN_2D":
        # Un pulso Gaussiano en el centro del dominio para la componente u
        u_view[:] = np.exp(-100 * ((x - 0.5)**2 + (y - 0.5)**2))
        v_view[:] = 0.0 # Sin velocidad inicial en v
        
    else:
        print(f"ADVERTENCIA: Condición inicial 2D '{IniS}' no reconocida. Usando cero.")
        u_view[:] = 0.0
        v_view[:] = 0.0
        
            
