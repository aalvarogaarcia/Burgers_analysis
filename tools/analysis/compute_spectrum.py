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
import re

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
    
    ke_hat_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)**2
    
    kx = np.fft.fftfreq(nx, d=1.0/nx)
    ky = np.fft.fftfreq(ny, d=1.0/ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='xy')
    
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    k_bins = np.arange(0.5, min(nx, ny) // 2, 1.)
    if len(k_bins) < 2:
        return np.array([]), np.array([])
        
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    energy_spectrum, _ = np.histogram(k_magnitude.flatten(), bins=k_bins, weights=ke_hat_2d.flatten())
    count, _ = np.histogram(k_magnitude.flatten(), bins=k_bins)
    
    valid_bins = count > 0
    energy_1d = np.zeros_like(k_vals)
    energy_1d[valid_bins] = energy_spectrum[valid_bins] / count[valid_bins]
            
    return k_vals, energy_1d

def get_simulation_base_name(filepath):
    """
    Extrae el nombre base de una simulación a partir de un snapshot.
    """
    base = os.path.basename(filepath)
    # Busca 'DC_Vreman_convergence_NXX' y devuelve 'NXX' para usar en el ordenamiento
    match_n = re.search(r'_N(\d+)\.txt$', base)
    if match_n:
        return f"N{match_n.group(1)}"
    return base.split('_snapshot_')[0]

def extract_label_from_pattern(filepath):
    """
    Extrae una etiqueta descriptiva del patrón del fichero, incluyendo el coeficiente y la resolución.
    """
    base_name = os.path.basename(filepath).replace('.txt','').replace('*','')
    
    # Extraer el patrón general y luego la resolución si existe
    match_general = re.match(r'(DC_Vreman_convergence_N\d+)', base_name)
    if match_general:
        return match_general.group(1).replace('DC_Vreman_convergence_', 'Vreman_') # Ejemplo: Vreman_N33
        
    return base_name if base_name else "Serie"


def main(filepaths_patterns, output_filename):
    """
    Función principal para calcular, promediar y graficar espectros de múltiples casos.
    """
    simulations = defaultdict(list)
    for pattern in filepaths_patterns:
        for filepath in sorted(glob.glob(pattern)):
            base_name = extract_label_from_pattern(filepath) # Usamos la nueva función para agrupar
            simulations[base_name].append(filepath)

    if not simulations:
        print("Error: No se encontraron archivos que coincidan con los patrones dados.")
        return

    # Ordenar las simulaciones por resolución (N33, N65, N129)
    # Esto asume que el nombre base contiene "N<numero>"
    sorted_simulations = sorted(simulations.items(), key=lambda item: int(re.search(r'N(\d+)', item[0]).group(1)))

    plt.figure(figsize=(12, 8))
    
    # --- Parámetros para la línea de referencia de Kolmogorov ---
    k_kolmogorov_start = 2.0  # Número de onda donde empieza la línea de Kolmogorov
    k_kolmogorov_end = 8.0    # Número de onda donde termina la línea de Kolmogorov
    # Rango para ajustar la constante C a los datos de la malla más fina
    kolmogorov_fit_k_start = 3.0
    kolmogorov_fit_k_end = 5.0
    kolmogorov_reference_spectrum = None
    kolmogorov_reference_k = None

    for base_name, filepaths in sorted_simulations: # Iteramos sobre las simulaciones ordenadas
        print(f"Procesando simulación '{base_name}' con {len(filepaths)} snapshots...")
        
        all_spectra = []
        k_values = None

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

            x_vec = np.linspace(min(x), max(x), nx); y_vec = np.linspace(min(y), max(y), ny)
            grid_x, grid_y = np.meshgrid(x_vec, y_vec, indexing='ij')
            u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0)
            v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0)
            
            k, E_k = compute_energy_spectrum_2d(u_grid, v_grid)
            
            if k.size > 0:
                all_spectra.append(E_k)
                if k_values is None:
                    k_values = k
        
        if not all_spectra:
            print(f"  -> No se pudieron calcular espectros para '{base_name}'.")
            continue
            
        avg_spectrum = np.mean(np.array(all_spectra), axis=0)
        
        # Filtrar valores no positivos o no finitos para la gráfica log-log
        valid_indices = (k_values > 0) & (avg_spectrum > 0) & np.isfinite(k_values) & np.isfinite(avg_spectrum)
        if np.any(valid_indices):
            plt.loglog(k_values[valid_indices], avg_spectrum[valid_indices], 'o-', label=base_name, markerfacecolor='white', markersize=5)
            
            # Si esta es la malla de mayor resolución, la guardamos para la referencia de Kolmogorov
            if base_name == sorted_simulations[-1][0]: # Comprueba si es la última (mayor resolución)
                kolmogorov_reference_spectrum = avg_spectrum
                kolmogorov_reference_k = k_values


    # --- Dibuja la línea de referencia de Kolmogorov ---
    if kolmogorov_reference_spectrum is not None and kolmogorov_reference_k is not None:
        # Define un rango de K para la línea de referencia
        k_kolmogorov = np.logspace(np.log10(k_kolmogorov_start), np.log10(k_kolmogorov_end), 50)
        
        # Ajusta la constante C al espectro de la malla más fina en un rango específico
        fit_indices = (kolmogorov_reference_k >= kolmogorov_fit_k_start) & \
                      (kolmogorov_reference_k <= kolmogorov_fit_k_end) & \
                      (kolmogorov_reference_spectrum > 0) & \
                      np.isfinite(kolmogorov_reference_spectrum)
        
        if np.any(fit_indices):
            # Calcula C promediando en el rango de ajuste
            C = np.mean(kolmogorov_reference_spectrum[fit_indices] * kolmogorov_reference_k[fit_indices]**(5/3))
            
            if C > 1e-12: # Asegurarse de que C no sea despreciable
                plt.loglog(k_kolmogorov, C * k_kolmogorov**(-5/3), 'k--', label=r'$k^{-5/3}$ (Kolmogorov)', zorder=0)

    plt.title('Espectro de Energía Cinética Promediado en el Tiempo', fontsize=16)
    plt.xlabel('Número de Onda (k)', fontsize=12); plt.ylabel('Energía E(k)', fontsize=12)
    plt.legend(); plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.ylim(bottom=1e-12)
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=150)
    print(f"\n¡Éxito! Gráfico de espectros guardado en: {os.path.abspath(output_filename)}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python compute_spectrum.py \"ruta/caso1/*.txt\" \"ruta/caso2/*.txt\" ... [nombre_salida.png]")
        sys.exit(1)
    
    # Determina si el último argumento es el nombre del archivo de salida
    if sys.argv[-1].lower().endswith('.png'):
        output_name = sys.argv[-1]
        patterns = sys.argv[1:-1]
    else:
        output_name = "spectrum_comparison.png" # Nombre por defecto
        patterns = sys.argv[1:]
        
    main(patterns, output_filename=output_name)