#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:47:26 2025

@author: aalvarogaarcia

Reconstrucción de flujos Burgers en 2D
"""

# fr-burgers-2d.py

import numpy as np
from sys import argv, exit

from src.core.mesh import *
from src.utils.misc import *
from src.utils.io import *
from src.core.basis import*
from src.core.time_integration import *
from src.core.discretization import *
import traceback

def Usage():
    print("Usage: fr-burgers-2d.py inputfilename")
    print(" Mandatory labels:")
    print("NX, NY, P, VISC, INISOL, DT, TSIM, NDUMP")
    print(" Optional LES labels:")
    print("USE_LES, SGS_MODEL_TYPE, SGS_C_VREMAN, SGS_CS_CONSTANT")

def Run(document, lab):
    # Leer parámetros
    Nx = int(getValueFromLabel(document, "NX"))
    Ny = int(getValueFromLabel(document, "NY"))
    p = int(getValueFromLabel(document, "P"))
    v = float(getValueFromLabel(document, "VISC")) # 'v' se define aquí como la viscosidad
    IniS = getValueFromLabel(document, "INISOL")
    dt = float(getValueFromLabel(document, "DT"))
    tsim = float(getValueFromLabel(document, "TSIM"))
    Ndump = int(getValueFromLabel(document, "NDUMP"))
    Nref = 0

    # Leer parámetros LES
    use_les = getValueFromLabel(document, "USE_LES").upper() == "TRUE"
    sgs_params = None
    if use_les:
        sgs_params = {'model_type': getValueFromLabel(document, "SGS_MODEL_TYPE")}
        if sgs_params['model_type'] == 'vreman':
            sgs_params['c_vreman'] = float(getValueFromLabel(document, "SGS_C_VREMAN"))
        elif sgs_params['model_type'] == 'smagorinsky':
            sgs_params['Cs'] = float(getValueFromLabel(document, "SGS_CS_CONSTANT"))

    # Configuración de la simulación
    Nmax = int(tsim / dt)
    if (Nmax * dt < tsim): Nmax += 1
    dt = tsim / Nmax

    # Crear malla y estado inicial
    lobatto_points, Lp_matrix, gp_array = getStandardElementData(p)
    x_grid, y_grid = get_2d_cartesian_mesh(Nx, Ny)
    x_ho, y_ho = get_mesh_ho_2d(x_grid, y_grid, p, lobatto_points)
    num_nodes = len(x_ho)
    U = np.zeros(2 * num_nodes)
    FillInitialSolution_2D(U, x_ho, y_ho, IniS, Nx, Ny, p, Nref)

    # Bucle temporal principal
    for it in range(Nmax):
        # 'v' (viscosidad) se pasa aquí en el paquete de argumentos
        args_for_residual = (p, (x_ho, y_ho), v, (Lp_matrix, gp_array), Nx, Ny, use_les, sgs_params)
        U = rk4(dt, U, get_residual_2d, args_for_residual)
        
        current_time = (it + 1) * dt
        print(f"it: {it+1}/{Nmax}, t: {current_time:.4f}")

        if (it + 1) % Ndump == 0:
            WriteFile_2D(lab, x_ho, y_ho, U, Nx, Ny, p, v, Nref, IniS, dt, tsim, Ndump)

    # Guardado final
    WriteFile_2D(lab, x_ho, y_ho, U, Nx, Ny, p, v, Nref, IniS, dt, tsim, Ndump)

# --- Bloque de ejecución principal ---
if len(argv) < 2:
    Usage()
    exit()

inputfile_path = argv[1]
try:
    with open(inputfile_path, 'r') as f:
        document = f.readlines()
    Run(document, inputfile_path)
except FileNotFoundError:
    print(f"Error: Archivo de entrada no encontrado en '{inputfile_path}'")
except Exception as e:
    print(f"Ocurrió un error: {e}")
    traceback.print_exc() # Imprime el traceback completo para más detalles
