#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 17:41:41 2025

@author: aalvarogaarcia
"""

# tools/analysis/compute_spectrum.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import os

# Añade la ruta al directorio raíz del proyecto para poder importar desde 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def compute_energy_spectrum_2d(u, v):
    """Calcula el espectro de energía cinética 1D a partir de campos 2D."""
    nx, ny = u.shape
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    
    # Espectro de energía 2D, normalizado por el número de puntos
    ke_hat_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)**2
    
    # Crear la rejilla de números de onda
    kx = np.fft.fftfreq(nx, d=1.0/nx)
    ky = np.fft.fftfreq(ny, d=1.0/ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Agrupar la energía en "anillos" de número de onda (promedio acimutal)
    k_bins = np.arange(0.5, nx // 2, 1.)
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    energy_spectrum = np.zeros_like(k_vals)
    
    for i in range(len(k_vals)):
        mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i+1])
        if np.any(mask):
            energy_spectrum[i] = ke_hat_2d[mask].sum()
            
    return k_vals, energy_spectrum

def main(filepaths):
    """Función principal para calcular y graficar el espectro de uno o más archivos."""
    for filepath in filepaths:
        print(f"Calculando espectro para: {filepath}")
        
        with open(filepath, 'r') as f:
            document = f.readlines()
        
        data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
        if not data_lines:
            print(f"  -> Advertencia: No se encontró bloque de solución. Saltando archivo.")
            continue
        data = np.loadtxt(data_lines)
        
        nx = int(getValueFromLabel(document, "NX"))
        ny = int(getValueFromLabel(document, "NY"))
        
        x, y, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            print("  -> ADVERTENCIA: Datos inestables (inf/NaN) detectados. El espectro no será representativo.")
        
        # Interpolar a una rejilla uniforme para poder usar FFT
        grid_x, grid_y = np.mgrid[min(x):max(x):complex(nx), min(y):max(y):complex(ny)]
        u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0)
        v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0)
        
        k, E_k = compute_energy_spectrum_2d(u_grid, v_grid)
        
        # Graficar
        plt.figure(figsize=(10, 6))
        plt.loglog(k, E_k, 'o-', label='Espectro de Energía')
        
        # Dibujar pendientes de referencia (inspirado en Fig. 8 del paper)
        idx_ref = len(k) // 4
        if idx_ref > 0 and E_k[idx_ref] > 1e-15:
            C1 = E_k[idx_ref] * k[idx_ref]**(5/3)
            plt.loglog(k[1:], C1 * k[1:]**(-5/3), 'k--', label=r'$k^{-5/3}$ (Kolmogorov)')
        
        plt.title(f'Espectro de Energía Cinética - {os.path.basename(filepath)}')
        plt.xlabel('Número de Onda (k)')
        plt.ylabel('Energía E(k)')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.ylim(bottom=1e-12) # Evitar problemas con valores cero
        
        # Guardar la figura en la carpeta 'outputs' correspondiente
        output_dir = os.path.dirname(filepath).replace("inputs", "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_filename = os.path.basename(filepath).replace('.txt', '_spectrum.png')
        output_filename = os.path.join(output_dir, base_filename)

        plt.savefig(output_filename, dpi=150)
        print(f"Espectro guardado en: {output_filename}")
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python compute_spectrum.py <archivo1.txt> [archivo2.txt]...")
        sys.exit(1)
    main(sys.argv[1:])