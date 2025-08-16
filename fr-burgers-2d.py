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

from src.core.mesh import get_2d_cartesian_mesh, get_mesh_ho_2d
from src.utils.misc import FillInitialSolution_2D
from src.utils.randw import getValueFromLabel, WriteFile_2D, WriteInputData
from src.core.lagpol import getStandardElementData
from src.core.ode import RK4
from src.core.residual import *
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
    scheme = str(getValueFromLabel(document, "SCHEME"))
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


    if scheme.lower() == "fr": 
    # --- Rama para Flux Reconstruction (FR) de alto orden ---
        print(f"INFO: Configurando simulación con esquema FR, Orden P={p}.")
    
    # Crear malla y operadores para FR
        lobatto_points, Lp_matrix, gp_array = getStandardElementData(p)
        x_grid, y_grid = get_2d_cartesian_mesh(Nx, Ny)
        x_ho, y_ho = get_mesh_ho_2d(x_grid, y_grid, p, lobatto_points)
    
        num_nodes = len(x_ho)
        U = np.zeros(2 * num_nodes)
        FillInitialSolution_2D(U, x_ho, y_ho, IniS, Nx, Ny, p, Nref)

    # Bucle temporal principal para FR
        for it in range(Nmax):
        # Los argumentos son específicos para el residuo de FR
            U_last_stable = np.copy(U)
            args_for_residual = (p, (x_ho, y_ho), v, (Lp_matrix, gp_array), Nx, Ny, use_les, sgs_params)
            U = RK4(dt, U, get_residual_2d, *args_for_residual)
            
            if np.isnan(U).any():
                print(f"¡ERROR! Inestabilidad numérica detectada en la iteración {it+1}.")
                print(f"Guardando el último estado estable en t = {it*dt:.4f}.")
                lab_failed = lab.replace('.out', '_FAILED.out')
                WriteFile_2D(lab_failed, x_ho, y_ho, U_last_stable, Nx, Ny, p, v, Nref, IniS, dt, tsim, Ndump, scheme, use_les, sgs_params)
                break
            
            current_time = (it + 1) * dt
            print(f"it: {it+1}/{Nmax}, t: {current_time:.4f} (FR)")

            if (it + 1) % Ndump == 0:
                WriteFile_2D(lab, x_ho, y_ho, U, Nx, Ny, p, v, Nref, IniS, dt, tsim, Ndump, scheme)
                
            
        else: 
            simulation_completed = True
                


    elif scheme.lower() == 'dc':
    # --- Rama para Diferencias Centradas (DC) de 2º orden ---
        print(f"INFO: Configurando simulación con esquema DC. El orden P={p} será ignorado.")
        p_dc = 0 # En DC, cada nodo es un elemento, p=0.
        
   
    # La malla es la malla cartesiana simple
        x_coords, y_coords = get_2d_cartesian_mesh(Nx, Ny)
        
        xx, yy = np.meshgrid(x_coords, y_coords)
        x_coords_full = xx.flatten()
        y_coords_full = yy.flatten()
        
        num_nodes = Nx * Ny
        U = np.zeros(2 * num_nodes)
        FillInitialSolution_2D(U, x_coords_full, y_coords_full, IniS, Nx, Ny, p_dc, Nref)
        
    # Bucle temporal principal para DC
        for it in range(Nmax):
            U_last_stable = np.copy(U)
        # Los argumentos son más simples para el residuo de DC
            args_for_residual = ((x_coords, y_coords), v, Nx, Ny, use_les, sgs_params)
        # ¡¡IMPORTANTE: Llamamos a una nueva función de residuo!!
            U = RK4(dt, U, get_residual_2d_dc, *args_for_residual)
            
            if np.isnan(U).any():
                print(f"¡ERROR! Inestabilidad numérica detectada en la iteración {it+1}.")
                print(f"Guardando el último estado estable en t = {it*dt:.4f}.")
                lab_failed = lab.replace('.out', '_FAILED.out')
                WriteFile_2D(lab_failed, x_coords, y_coords, U_last_stable, Nx, Ny, p_dc, v, Nref, IniS, dt, tsim, Ndump, scheme, use_les, sgs_params)
                break # Salir del bucle y continuar con el siguiente archivo

            
            current_time = (it + 1) * dt
            print(f"it: {it+1}/{Nmax}, t: {current_time:.4f} (DC)")

            if (it + 1) % Ndump == 0:
                WriteFile_2D(lab, x_coords, y_coords, U, Nx, Ny, p_dc, v, Nref, IniS, dt, tsim, Ndump)

        else: 
            simulation_completed = True
    
    else:
        raise ValueError(f"Esquema '{scheme}' no reconocido. Use 'fr' o 'dc'.")

    if simulation_completed:
        print("Simulación completada con éxito. Guardando estado final.")
        if scheme.lower() == 'fr':
            WriteFile_2D(lab, x_ho, y_ho, U, Nx, Ny, p, v, Nref, IniS, dt, tsim, Ndump, scheme, use_les, sgs_params)
        elif scheme.lower() == 'dc':
            WriteFile_2D(lab, x_coords, y_coords, U, Nx, Ny, p_dc, v, Nref, IniS, dt, tsim, Ndump, scheme, use_les, sgs_params)

    
# --- Bloque de ejecución principal ---
if len(argv) < 2:
    Usage()
    exit()

inputfiles = argv[1:]
try:
    for path in inputfiles:
        with open(path, 'r') as f:
            document = f.readlines()
        Run(document, path)
except FileNotFoundError:
    print(f"Error: Archivo de entrada no encontrado en '{inputfile_path}'")
except Exception as e:
    print(f"Ocurrió un error: {e}")
    traceback.print_exc() # Imprime el traceback completo para más detalles
