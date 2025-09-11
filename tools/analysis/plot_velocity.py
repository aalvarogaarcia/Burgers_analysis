#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para visualizar campos de velocidad 2D a partir de datos de simulación.

Este script lee ficheros de datos con formato de cabecera y columnas (x, y, u, v),
interpola los datos a una rejilla estructurada y genera gráficos de contorno
para las componentes 'u', 'v' y la magnitud de la velocidad.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import os
import argparse

# --- Funciones de Lectura de Datos ---

def get_value_from_label(filepath, label):
    """Extrae un valor numérico de la cabecera de un fichero."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if label in line:
                    return int(line.split()[-1])
    except FileNotFoundError:
        print(f"Error: Fichero no encontrado {filepath}")
    except Exception as e:
        print(f"Error leyendo la etiqueta '{label}' de {filepath}: {e}")
    return None

def read_solution_data(filepath):
    """
    Lee el bloque de datos de un fichero de simulación de forma robusta.
    Se detiene cuando las filas ya no tienen 4 columnas.
    """
    data_rows = []
    try:
        with open(filepath, 'r') as f:
            in_solution_block = False
            for line in f:
                if 'BEGIN_SOLUTION' in line:
                    in_solution_block = True
                    continue
                
                if in_solution_block:
                    parts = line.strip().split()
                    # --- INICIO DE LA CORRECCIÓN ---
                    # Comprobar si la línea tiene 4 componentes numéricos
                    if len(parts) == 4:
                        try:
                            # Convertir cada parte a flotante
                            row = [float(p) for p in parts]
                            data_rows.append(row)
                        except ValueError:
                            # Si la conversión falla, no es una línea de datos válida. Parar.
                            break
                    else:
                        # Si la línea no tiene 4 columnas, asumimos que los datos terminaron.
                        break

# --- Función Principal de Visualización ---

def plot_velocity_field(filepath):
    """
    Lee un fichero de datos, procesa y visualiza los campos de velocidad.
    """
    print(f"--- Procesando fichero: {os.path.basename(filepath)} ---")
    
    # 1. Leer metadatos y datos de la solución
    nx = get_value_from_label(filepath, 'NX')
    ny = get_value_from_label(filepath, 'NY')
    data = read_solution_data(filepath)

    if nx is None or ny is None or data is None:
        print(f"No se pudo procesar el fichero {filepath}. Saltando.")
        return

    x, y, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # 2. Crear una rejilla estructurada para la interpolación
    grid_x, grid_y = np.mgrid[min(x):max(x):nx*1j, min(y):max(y):ny*1j]

    # 3. Interpolar los datos no estructurados a la rejilla estructurada
    print("Interpolando datos a la rejilla estructurada...")
    u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0.0)
    v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0.0)
    
    # Calcular la magnitud de la velocidad
    magnitude_grid = np.sqrt(u_grid**2 + v_grid**2)

    # 4. Crear los gráficos
    print("Generando gráficos...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Título general de la figura
    fig.suptitle(f'Campo de Velocidad - {os.path.basename(filepath)}', fontsize=16)

    # Gráfico para la componente U
    contour_u = axes[0].contourf(grid_x, grid_y, u_grid, levels=50, cmap='viridis')
    fig.colorbar(contour_u, ax=axes[0], label='Velocidad (m/s)')
    axes[0].set_title('Componente U de la Velocidad')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_aspect('equal', adjustable='box')

    # Gráfico para la componente V
    contour_v = axes[1].contourf(grid_x, grid_y, v_grid, levels=50, cmap='viridis')
    fig.colorbar(contour_v, ax=axes[1], label='Velocidad (m/s)')
    axes[1].set_title('Componente V de la Velocidad')
    axes[1].set_xlabel('x (m)')
    axes[1].set_aspect('equal', adjustable='box')

    # Gráfico para la Magnitud
    contour_mag = axes[2].contourf(grid_x, grid_y, magnitude_grid, levels=50, cmap='inferno')
    fig.colorbar(contour_mag, ax=axes[2], label='Velocidad (m/s)')
    axes[2].set_title('Magnitud de la Velocidad')
    axes[2].set_xlabel('x (m)')
    axes[2].set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para el título general
    
    # 5. Guardar la figura
    output_filename = f"campo_velocidad_{os.path.splitext(os.path.basename(filepath))[0]}.png"
    plt.savefig(output_filename, dpi=150)
    print(f"¡Éxito! Gráfico guardado en: {output_filename}\n")
    plt.close(fig) # Cerrar la figura para liberar memoria

# --- Bloque de Ejecución ---

if __name__ == "__main__":
    # Configurar el parser para argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Genera visualizaciones de campos de velocidad a partir de ficheros de datos de simulación."
    )
    parser.add_argument(
        'files', 
        metavar='FICHERO', 
        type=str, 
        nargs='+',
        help='Uno o más ficheros de datos .txt para procesar.'
    )
    
    args = parser.parse_args()

    # Procesar cada fichero proporcionado
    for file in args.files:
        plot_velocity_field(file)