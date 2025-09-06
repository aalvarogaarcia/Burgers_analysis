#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 17:41:41 2025

@author: aalvarogaarcia

Módulo para el cálculo y visualización del espectro de energía cinética,
con capacidad para promediar en el tiempo y comparar múltiples casos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import os
import glob
from collections import defaultdict

# Añade la ruta al directorio raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def compute_energy_spectrum_2d(u_grid, v_grid):
    """
    Calcula el espectro de energía 1D a partir de campos 2D ESTRUCTURADOS
    mediante un promedio acimutal.
    """
    ny, nx = u_grid.shape # Nota: numpy usa (filas, columnas) -> (y, x)
    if nx < 2 or ny < 2:
        return np.array([]), np.array([])

    u_hat = np.fft.fft2(u_grid)
    v_hat = np.fft.fft2(v_grid)
    
    # Normalización correcta para la densidad espectral de energía
    ke_hat_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)**2
    
    kx = np.fft.fftfreq(nx, d=1.0/nx)
    ky = np.fft.fftfreq(ny, d=1.0/ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='xy') # Usar 'xy' para consistencia (columnas, filas)
    
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Crear bins para el promediado acimutal
    k_bins = np.arange(0.5, min(nx, ny) // 2, 1.)
    if len(k_bins) < 2:
        return np.array([]), np.array([])
        
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    # Usar np.histogram para sumar la energía en cada "anillo" k
    energy_spectrum, _ = np.histogram(k_magnitude.flatten(), bins=k_bins, weights=ke_hat_2d.flatten())
    count, _ = np.histogram(k_magnitude.flatten(), bins=k_bins)
    
    # Evitar división por cero
    valid_bins = count > 0
    energy_1d = np.zeros_like(k_vals)
    energy_1d[valid_bins] = energy_spectrum[valid_bins] / count[valid_bins]
            
    return k_vals, energy_1d

def get_simulation_base_name(filepath):
    """
    Extrae el nombre base de una simulación a partir de un snapshot.
    Ej: 'DC_ILES_forced_snapshot_0001.txt' -> 'DC_ILES_forced'
    """
    base = os.path.basename(filepath)
    return base.split('_snapshot_')[0]

def main(filepaths_patterns, output_filename):
    """
    Función principal para calcular, promediar y graficar espectros de múltiples casos.
    """
    # Agrupar ficheros por simulación base
    simulations = defaultdict(list)
    for pattern in filepaths_patterns:
        for filepath in sorted(glob.glob(pattern)):
            base_name = get_simulation_base_name(filepath)
            simulations[base_name].append(filepath)

    if not simulations:
        print("Error: No se encontraron archivos que coincidan con los patrones dados.")
        return

    # --- Iniciar la figura para la comparación ---
    plt.figure(figsize=(12, 8))
    
    for base_name, filepaths in simulations.items():
        print(f"Procesando simulación '{base_name}' con {len(filepaths)} snapshots...")
        
        all_spectra = []
        k_values = None

        # Procesar cada snapshot de la simulación
        for filepath in filepaths:
            with open(filepath, 'r') as f:
                document = f.readlines()
            
            data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
            if not data_lines: continue
                
            data = np.loadtxt(data_lines)
            nx = int(getValueFromLabel(document, "NX")); ny = int(getValueFromLabel(document, "NY"))
            x, y, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

            if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
                print(f"  -> Advertencia: Datos inestables en {os.path.basename(filepath)}. Snapshot omitido.")
                continue

            # Interpolar a una rejilla uniforme para la FFT
            x_vec = np.linspace(min(x), max(x), nx); y_vec = np.linspace(min(y), max(y), ny)
            grid_x, grid_y = np.meshgrid(x_vec, y_vec, indexing='ij')
            u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0)
            v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0)
            
            k, E_k = compute_energy_spectrum_2d(u_grid, v_grid)
            
            if k.size > 0:
                all_spectra.append(E_k)
                if k_values is None:
                    k_values = k
        
        # --- Promediar los espectros de la simulación ---
        if not all_spectra:
            print(f"  -> No se pudieron calcular espectros para '{base_name}'.")
            continue
            
        avg_spectrum = np.mean(np.array(all_spectra), axis=0)
        
        # --- Graficar el espectro promediado ---
        plt.loglog(k_values, avg_spectrum, 'o-', label=base_name, markerfacecolor='white', markersize=5)
    
    # --- Añadir línea de referencia de Kolmogorov ---
    if k_values is not None and k_values.size > 2:
        k_ref = k_values[len(k_values) // 4 : len(k_values) // 2]
        if k_ref.size > 0:
            # Encontrar el primer espectro válido para escalar la línea
            first_valid_spectrum = next((np.mean(np.array(s), axis=0) for s in all_spectra if s), None)
            if first_valid_spectrum is not None:
                C = np.mean(first_valid_spectrum[len(k_values) // 4 : len(k_values) // 2] * k_ref**(5/3))
                plt.loglog(k_ref, C * k_ref**(-5/3), 'k--', label=r'$k^{-5/3}$ (Kolmogorov)', zorder=0)

    plt.title('Espectro de Energía Cinética Promediado en el Tiempo', fontsize=16)
    plt.xlabel('Número de Onda (k)', fontsize=12); plt.ylabel('Energía E(k)', fontsize=12)
    plt.legend(); plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.ylim(bottom=1e-12) # Límite inferior para evitar problemas con valores muy pequeños
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=150)
    print(f"\n¡Éxito! Gráfico de espectros guardado en: {os.path.abspath(output_filename)}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python compute_spectrum.py \"ruta/caso1/*.txt\" \"ruta/caso2/*.txt\" ... [nombre_salida.png]")
        sys.exit(1)
    
    if sys.argv[-1].lower().endswith('.png'):
        output_name = sys.argv[-1]
        patterns = sys.argv[1:-1]
    else:
        output_name = "spectrum_comparison.png"
        patterns = sys.argv[1:]

    main(patterns, output_filename=output_name)