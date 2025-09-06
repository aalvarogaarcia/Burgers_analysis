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
from src.models.sgs_model import *
from src.core.residual import *
from sys import argv
from sys import exit
from src.core.ode import *
import os
import matplotlib.pyplot as plt
import glob

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
    scheme = str(getValueFromLabel(document,"SCHEME")).lower()
    IniS   = getValueFromLabel(document,"INISOL")
    dt     = float(getValueFromLabel(document,"DT"))
    tsim   = float(getValueFromLabel(document,"TSIM"))
    Ndump  = int(getValueFromLabel(document,"NDUMP"))
    
    residual_func = None
    x = get_mesh_1d(N, Nref)

    if scheme == 'fr':
        print(f"Configurando para el esquema: FR (P={p})")
        residual_func = getResidualBrurgersFR
        # Nota: los parámetros LES solo son compatibles con FR por ahora
    elif scheme in ['dc', 'upwind']:
        print(f"Configurando para el esquema de bajo orden: {scheme.upper()}")
        U = np.zeros(N)
        dx = x[1] - x[0] # Asumimos malla uniforme
        if scheme == 'dc':
            residual_func = get_residual_dc_1d
            
        else: # upwind
            residual_func = get_residual_upwind_1d
            
    else:
        raise ValueError(f"Esquema '{scheme}' no reconocido. Use 'FR', 'DC' o 'UPWIND'.")
    
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

    
    # Get the high-order mesh
    if scheme == 'fr':
        # Get the local element coordinates and derivatives of polynomials
        lobattoPoints,Lp,gLp = getStandardElementData(p)
        x = get_mesh_ho_1d(x,lobattoPoints)

    nnode = len(x)

    # Fill the initial solution
    U = np.zeros(nnode)

    FillInitialSolution_1D(U,x,IniS,N,p,Nref)

    graph, = ax.plot(x, U, label=lab)
    ax.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('u')
    figure.canvas.draw() # Dibujar estado inicial
    figure.canvas.flush_events()

    for it in range(0,Nmax):
        if scheme == 'fr':
            residual_args = (p,x,v,Lp,gLp,use_les_simulation, sgs_model_parameters)
        else:
            residual_args = (x, dx, v, use_les_simulation, sgs_model_parameters)
            
            
        U = RK4(dt,U, residual_func, *residual_args)
        print("it:",it,"t:",(it+1)*dt)
        
        if use_les_simulation and sgs_model_parameters['model_type'] == 'smagorinsky_dynamic':
            Cd_current = get_last_calculated_Cd()
            print(f"it: {it}, t: {(it+1)*dt:.4f}, Cd_dynamic: {Cd_current:.4e}")

        
        if (it % Ndump == 0):
            graph.set_xdata(x)
            graph.set_ydata(U)
            figure.canvas.draw()
            figure.canvas.flush_events()
            WriteFile_1D(lab,x,U,N,p,v,Nref,IniS,dt,tsim,Ndump, scheme, use_les_simulation, sgs_model_parameters)
    
    WriteFile_1D(lab,x,U,N,p,v,Nref,IniS,dt,tsim,Ndump, scheme, use_les_simulation, sgs_model_parameters)
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

if __name__ == "__main__":
    Nargs=len(argv)
    if (Nargs<2):
        Usage()
        exit()
    
    files_to_process = []
    for arg in argv:
        expand_files = glob.glob(arg)
        
        if not expand_files:
            print(f"Aviso: No se encontraron archivos que coincidan con el patrón '{arg}'.")
        
        files_to_process.extend(expand_files)
        
    if not files_to_process:
        print("Error: No se proporcionaron archivos de entrada válidos.")
        Usage()
        exit()
        
        
    

# Figure set-up
    figure, ax = plt.subplots(figsize=(10, 8))

    for ifile in range(1,len(files_to_process)):
        inputfile=open(files_to_process[ifile],'r')
        document = inputfile.readlines()
        Run(document,files_to_process[ifile])
        inputfile.close()

    plt.show()
