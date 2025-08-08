# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 06:44:53 2024

@author: Jesús Pueblas
"""

#!/usr/bin/python

import numpy as np
from src.core.mesh import *
from src.utils.misc import *
from src.utils.randw import *
from src.core.lagpol import *
from sys import argv
from sys import exit
from src.core.ode import *
import os
import matplotlib.pyplot as plt

# Print the usage of the program
def Usage():
    print("Usage: fr-burgers-turbulent.py inputfilename                          ")
    print(" Mandatory labels for the input file name:                            ")
    print("N      value  # Number of mesh points                                 ")
    print("Nref   value  # Number of refinement levels of the mesh               ")
    print("P      value  # polynomial order to reconstruct the solution          ")
    print("VISC   value  # Viscosity                                             ")
    print("INISOL value  # Initial solution type: SINE,GAUSSIAN,SQUARE,TURBULENT ")
    print("DT     value  # Time step                                             ")
    print("TSIM   value  # Simulation time                                       ")
    print("NDUMP  value  # Interval of iterations to dump a solution             ")
    print("--- Optional LES Parameters ---")
    print("USE_LES         TRUE/FALSE   # Activate LES model")
    print("SGS_MODEL_TYPE  model_name   # e.g., smagorinsky_dynamic")
    print("SGS_FILTER_RATIO value       # Delta_hat / Delta")
    print("SGS_AVG_TYPE    global/local # Averaging for dynamic constant")
    print("SGS_CS_MIN      value        # Minimum Cs for clipping")


def Run(document,lab):
    # Read the contents of the input file
    N      = int(getValueFromLabel(document,"N"))
    p      = int(getValueFromLabel(document,"P"))
    v      = float(getValueFromLabel(document,"VISC"))
    Nref   = int(getValueFromLabel(document,"NREF"))
    IniS   = getValueFromLabel(document,"INISOL")
    dt     = float(getValueFromLabel(document,"DT"))
    tsim   = float(getValueFromLabel(document,"TSIM"))
    Ndump  = int(getValueFromLabel(document,"NDUMP"))
    
    # --- Leer parámetros LES ---
    use_les_str = getValueFromLabel(document, "USE_LES")
    use_les_simulation = False # Default a no usar LES
    if use_les_str != "ERROR": # "ERROR" es lo que devuelve getValueFromLabel si no encuentra la etiqueta
        if use_les_str.upper() == "TRUE":
            use_les_simulation = True
    
    sgs_model_parameters = None
    if use_les_simulation:
        sgs_model_type_str = getValueFromLabel(document, "SGS_MODEL_TYPE")
        sgs_filter_ratio_str = getValueFromLabel(document, "SGS_FILTER_RATIO")
        sgs_avg_type_str = getValueFromLabel(document, "SGS_AVG_TYPE")
        sgs_cs_min_str = getValueFromLabel(document, "SGS_CS_MIN")

        # Es importante manejar el caso donde las etiquetas SGS no estén si USE_LES es TRUE
        if "ERROR" in [sgs_model_type_str, sgs_filter_ratio_str, sgs_avg_type_str, sgs_cs_min_str]:
            print("ADVERTENCIA: USE_LES es TRUE pero faltan algunos parámetros SGS. Desactivando LES.")
            use_les_simulation = False
        else:
            sgs_model_parameters = {
                'model_type': sgs_model_type_str,
                'filter_width_ratio': float(sgs_filter_ratio_str),
                'avg_type': sgs_avg_type_str,
                'Cs_min': float(sgs_cs_min_str)
                # Aquí puedes añadir más parámetros si tu modelo los necesita
            }
            print(f"Simulación LES activada con el modelo: {sgs_model_parameters['model_type']}")
    else:
        print("Simulación DNS/ILES (sin modelo SGS explícito).")
#################################################################

    # Set the number of iterations and adjust dt to run the exact simulation time
    Nmax = int(tsim/dt)
    if (Nmax*dt<tsim):
        Nmax += 1
    dt = tsim / Nmax

    # Create the mesh
    x = getMesh(N,Nref)

    # Get the local element coordinates and derivatives of polynomials
    lobattoPoints,Lp,gLp = getStandardElementData(p)

    # Get the high-order mesh
    x = getMeshHO(x,lobattoPoints)

    nnode = len(x)

    # Fill the initial solution
    U = np.zeros(nnode)

    FillInitialSolution(U,x,IniS,N,p,Nref)

    graph, = ax.plot(x, U, label=lab)
    ax.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('u')
    figure.canvas.draw() # Dibujar estado inicial
    figure.canvas.flush_events()

    for it in range(0,Nmax):
        U = RK4(dt,U,p,x,v,Lp,gLp,use_les_simulation, sgs_model_parameters)
        print("it:",it,"t:",(it+1)*dt)
        
        if use_les_simulation and sgs_model_parameters['model_type'] == 'smagorinsky_dynamic':
            Cd_current = sgs_model.get_last_calculated_Cd()
            print(f"it: {it}, t: {(it+1)*dt:.4f}, Cd_dynamic: {Cd_current:.4e}")

        
        if (it % Ndump == 0):
            graph.set_xdata(x)
            graph.set_ydata(U)
            figure.canvas.draw()
            figure.canvas.flush_events()
            WriteFile(lab,x,U,N,p,v,Nref,IniS,dt,tsim,Ndump, use_les_simulation, sgs_model_parameters)
    
    WriteFile(lab,x,U,N,p,v,Nref,IniS,dt,tsim,Ndump, use_les_simulation, sgs_model_parameters)
    graph.set_xdata(x)
    graph.set_ydata(U)
    figure.canvas.draw()
    figure.canvas.flush_events()

    
    
#################################################################
# Input parameters:                                             #
# -----------------                                             #
# N     : number of points                                      #
# Nref  : Number of refinement levels of the mesh               #
# p     : Polynomial order to reconstruct the solution          #
# v     : Viscosity                                             #
# IniS  : Type of initial solution                              #
# dt    : Time step                                             #
# tsim  : Simulation time                                       #
# Ndump : Interval of iterations to dump a solution             #
#                                                               #
# --- LES Parameters (Optional) ---                             #
# USE_LES         TRUE   # TRUE to use LES, FALSE for ILES      #
# If USE_LES is TRUE, the following parameters are expected:    #
#                                                               #
#   SGS_MODEL_TYPE:  type of SGS used                           # 
#   SGS_FILTER_RATIO: delta for dynamic model                   #
#   SGS_AVG_TYPE: type for dynamic constant                     #
#   SGS_CS_MIN: minimum constant (smagorinsky)                  #
#                                                               #
#                                                               #
#################################################################
# Equations to solve:                                           #
# -------------------                                           #
# du/dt + d(u^2/2)/dx - v·d2(u)/dx2 = 0                         #
# x = [0,1]                                                     #
# Boundary conditions                                           #
# u(0) = u(1):                                                  #
#################################################################

# Read the input file:
Nargs=len(argv)
if (Nargs<2):
    Usage()
    exit()

# Figure set-up
figure, ax = plt.subplots(figsize=(10, 8))

for ifile in range(1,len(argv)):
    inputfile=open(argv[ifile],'r')
    document = inputfile.readlines()
    Run(document,argv[ifile])
    inputfile.close()

plt.show()
