#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para el cálculo y visualización del espectro de energía cinética.
Versión final: maneja errores de formato, múltiples resoluciones y asigna
colores distintos a cada caso de estudio.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import os
import glob
from collections import defaultdict
import argparse

# --- Funciones para leer datos ---

def getValueFromLabel(file_path, label):
    """Extrae un valor numérico de la cabecera de un fichero."""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if label in line:
                    return int(line.split()[-1])
    except Exception as e:
        print(f"Advertencia: No se pudo leer la etiqueta '{label}' de {file_path}: {e}")
    return None

def ReadBlockData(file_path):
    """
    Lee el bloque de datos de un fichero de simulación de forma robusta.
    Se detiene cuando las filas ya no tienen 4 columnas.
    """
    data_rows = []
    try:
        with open(file_path, 'r') as f:
            in_solution_block = False
            for line in f:
                if 'BEGIN_SOLUTION' in line:
                    in_solution_block = True
                    continue
                
                if in_solution_block:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        try:
                            row = [float(p) for p in parts]
                            data_rows.append(row)
                        except ValueError:
                            break
                    else:
                        break
                        
    except FileNotFoundError:
        print(f"Error: Fichero no encontrado {file_path}")
        return None
    except Exception as e:
        print(f"Error al leer el bloque de datos de {file_path}: {e}")
        return None
        
    if not data_rows:
        print(f"Advertencia: No se encontraron datos válidos en {file_path}")
        return None
        
    return np.array(data_rows)

# --- Lógica del espectro ---

def compute_energy_spectrum_2d(u_grid, v_grid):
    """Calcula el espectro de energía 1D (promedio azimutal)."""
    ny, nx = u_grid.shape
    if nx < 2 or ny < 2: return np.array([]), np.array([])

    u_hat = np.fft.fft2(u_grid)
    v_hat = np.fft.fft2(v_grid)
    
    ke_hat_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)**2
    
    kx = np.fft.fftfreq(nx, d=1.0/nx)
    ky = np.fft.fftfreq(ny, d=1.0/ny)
    
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='xy')
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    k_bins = np.arange(0.5, min(nx, ny) // 2, 1.)
    if len(k_bins) < 2: return np.array([]), np.array([])
        
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    energy_spectrum, _ = np.histogram(k_magnitude.flatten(), bins=k_bins, weights=ke_hat_2d.flatten())
    count, _ = np.histogram(k_magnitude.flatten(), bins=k_bins)
    
    energy_1d = np.zeros_like(k_vals)
    valid_bins = count > 0
    energy_1d[valid_bins] = energy_spectrum[valid_bins] / count[valid_bins]
            
    return k_vals, energy_1d

def extract_label(filepath, nx, ny):
    """Crea una etiqueta descriptiva para la leyenda del gráfico."""
    base_name = os.path.basename(filepath).replace('.txt','').replace('*','')
    label = base_name.replace('_', ' ')
    return f"{label} ({nx}x{ny})"

def process_pattern(file_pattern):
    """
    Procesa ficheros que coinciden con un patrón, agrupándolos por resolución.
    """
    files = glob.glob(file_pattern)
    if not files:
        print(f"Alerta: No se encontraron ficheros para el patrón '{file_pattern}'")
        return []

    files_by_resolution = defaultdict(list)
    for f in files:
        nx = getValueFromLabel(f, 'NX')
        ny = getValueFromLabel(f, 'NY')
        if nx is not None and ny is not None:
            files_by_resolution[(nx, ny)].append(f)

    results = []
    for (nx, ny), file_list in files_by_resolution.items():
        print(f"Procesando {len(file_list)} snapshots para la resolución {nx}x{ny}...")
        
        grid_x, grid_y = np.mgrid[0:1:nx*1j, 0:1:ny*1j]
        total_spec, count, k_out = None, 0, None

        for filepath in file_list:
            data = ReadBlockData(filepath)
            if data is None: continue

            x, y, u, v = data[:,0], data[:,1], data[:,2], data[:,3]
            u_grid = griddata((x, y), u, (grid_x, grid_y), method='linear', fill_value=0.0)
            v_grid = griddata((x, y), v, (grid_x, grid_y), method='linear', fill_value=0.0)
            
            k, spec = compute_energy_spectrum_2d(u_grid, v_grid)

            if k.size == 0: continue
            if total_spec is None:
                total_spec = np.zeros_like(spec)
                k_out = k

            if spec.shape == total_spec.shape:
                total_spec += spec
                count += 1
        
        if count > 0 and k_out is not None:
            label = extract_label(file_list[0], nx, ny)
            results.append({"k": k_out, "spec": total_spec / count, "label": label})
            
    return results

# --- Main ---
def main(case_patterns, output_filename):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    
    # --- NOVEDAD: Generar una paleta de colores automática ---
    # Se crea un color diferente por cada patrón de entrada (ej. "*.txt", "DC_*.txt")
    colors = plt.cm.viridis(np.linspace(0, 1, len(case_patterns)))
    
    all_results = []
    for i, pattern in enumerate(case_patterns):
        results = process_pattern(pattern)
        for res in results:
            # Asignar el color correspondiente al patrón
            res['color'] = colors[i]
            all_results.append(res)
    
    if not all_results:
        print("No se encontraron datos válidos para graficar.")
        return

    # Graficar todos los resultados
    for res in all_results:
        plt.loglog(res['k'], res['spec'], lw=2, marker='o', markersize=4, color=res['color'], label=res['label'])

    # Añadir línea de referencia de Kolmogorov k^(-5/3)
    all_k = np.concatenate([res['k'] for res in all_results])
    if all_k.size > 0:
        k_ref = np.logspace(np.log10(max(1, all_k.min())), np.log10(all_k.max()), 20)
        plt.loglog(k_ref, 5e-3 * k_ref**(-5/3), 'k--', alpha=0.7, label=r'Pendiente $k^{-5/3}$')

    plt.title('Espectro de Energía Cinética', fontsize=16)
    plt.xlabel('Número de Onda (k)', fontsize=12)
    plt.ylabel('Energía E(k)', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.ylim(bottom=1e-8)
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n¡Éxito! Gráfico de espectro guardado en: {os.path.abspath(output_filename)}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera un gráfico del espectro de energía cinética.")
    parser.add_argument('files', metavar='FICHERO', type=str, nargs='+', help='Patrones de ficheros a procesar (ej. "DC_*.txt" "FR_*.txt").')
    parser.add_argument('-o', '--output', type=str, default="espectro_energia_color.png", help='Nombre del fichero de salida para la imagen.')
    args = parser.parse_args()
    
    main(args.files, args.output)