#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 17:41:41 2025

@author: aalvarogaarcia

Módulo para el cálculo y visualización del espectro de energía cinética.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import os
import glob

# Añade la ruta al directorio raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def compute_energy_spectrum_2d(u_grid, v_grid):
    """
    Calcula el espectro de energía 1D a partir de campos 2D ESTRUCTURADOS
    mediante un promedio acimutal.
    """
    nx, ny = u_grid.shape
    if nx < 2 or ny < 2:
        return np.array([]), np.array([])

    u_hat = np.fft.fft2(u_grid)
    v_hat = np.fft.fft2(v_grid)
    
    ke_hat_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)**2
    
    kx = np.fft.fftfreq(nx, d=1.0/nx)
    ky = np.fft.fftfreq(ny, d=1.0/ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    k_bins = np.arange(0.5, min(nx, ny) // 2, 1.)
    if len(k_bins) < 2:
        return np.array([]), np.array([])
        
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    energy_spectrum, _ = np.histogram(k_magnitude.flatten(), bins=k_bins, weights=ke_hat_2d.flatten())
    count, _ = np.histogram(k_magnitude.flatten(), bins=k_bins)
    
    valid_bins = count > 0
    # Inicializar con ceros para evitar problemas si no hay datos en algunos bins
    energy_1d = np.zeros_like(k_vals)
    energy_1d[valid_bins] = energy_spectrum[valid_bins] / count[valid_bins]
            
    return k_vals, energy_1d

def main(filepaths_patterns):
    """Función principal para calcular y graficar el espectro de uno o más archivos."""
    filepaths = []
    for pattern in filepaths_patterns:
        filepaths.extend(glob.glob(pattern))

    if not filepaths:
        print("Error: No se encontraron archivos que coincidan con los patrones dados.")
        return

    for filepath in filepaths:
        print(f"Calculando espectro para: {filepath}")
        
        with open(filepath, 'r') as f:
            document = f.readlines()
        
        data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
        if not data_lines:
            print(f"  -> Advertencia: No se encontró bloque de solución. Saltando archivo.")
            continue
            
        data = np.loadtxt(data_lines)
        nx = int(getValueFromLabel(document, "NX")); ny = int(getValueFromLabel(document, "NY"))
        x, y, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            print("  -> ADVERTENCIA: Datos inestables (inf/NaN) detectados. El espectro no será representativo.")
            continue

        x_vec = np.linspace(min(x), max(x), nx); y_vec = np.linspace(min(y), max(y), ny)
        grid_x, grid_y = np.meshgrid(x_vec, y_vec, indexing='ij')

        u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0)
        v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0)
        
        k, E_k = compute_energy_spectrum_2d(u_grid, v_grid)
        
        if len(k) < 2:
            print("  -> Advertencia: No hay suficientes datos para generar un espectro significativo.")
            continue

        plt.figure(figsize=(8, 6))
        plt.loglog(k, E_k, 'o-', label='Espectro de Energía', markerfacecolor='white')
        
        k_ref = k[len(k) // 4 : len(k) // 2]
        if len(k_ref) > 0:
            C = np.mean(E_k[len(k) // 4 : len(k) // 2] * k_ref**(5/3))
            if C > 1e-12: # Solo dibujar si la energía no es despreciable
                plt.loglog(k_ref, C * k_ref**(-5/3), 'k--', label=r'$k^{-5/3}$ (Kolmogorov)')
        
        plt.title(f'Espectro de Energía Cinética\n{os.path.basename(filepath)}', fontsize=14)
        plt.xlabel('Número de Onda (k)', fontsize=12); plt.ylabel('Energía E(k)', fontsize=12)
        plt.legend(); plt.grid(True, which="both", ls="--", linewidth=0.5)
        
        # --- CORRECCIÓN DE ROBUSTEZ AQUÍ ---
        E_k_positive = E_k[E_k > 0]
        if E_k_positive.size > 0:
            bottom_limit = max(1e-12, np.min(E_k_positive) / 10)
        else:
            bottom_limit = 1e-12 # Usar un valor por defecto si no hay energía positiva
        plt.ylim(bottom=bottom_limit)
        
        plt.tight_layout()
        
        output_dir = os.path.dirname(filepath).replace("inputs", "outputs")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        base_filename = os.path.basename(filepath).replace('.txt', '_spectrum.png')
        output_filename = os.path.join(output_dir, base_filename)
        plt.savefig(output_filename, dpi=150)
        print(f"Espectro guardado en: {output_filename}")
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python compute_spectrum.py \"ruta/a/resultados/*.txt\"")
        sys.exit(1)
    main(sys.argv[1:])
