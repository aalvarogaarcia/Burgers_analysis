# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:24:55 2024

@author: Jesús Pueblas
"""

# postprocess.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from scipy.interpolate import griddata
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import *

# --- FUNCIONES DE AYUDA ---

def get_mesh_and_solution_1d(document):
    """Lee los datos de un archivo de solución 1D (x, u)."""
    data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
    num_nodes = len(data_lines)
    x = np.zeros(num_nodes)
    u = np.zeros(num_nodes)
    for i, line in enumerate(data_lines):
        fields = line.split()
        if len(fields) == 2:
            x[i] = float(fields[0])
            u[i] = float(fields[1])
    return x, u

def get_mesh_and_solution_2d(document):
    """Lee los datos de un archivo de solución 2D (x, y, u, v)."""
    data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
    num_nodes = len(data_lines)
    x = np.zeros(num_nodes)
    y = np.zeros(num_nodes)
    u = np.zeros(num_nodes)
    v = np.zeros(num_nodes)
    for i, line in enumerate(data_lines):
        fields = line.split()
        if len(fields) == 4:
            x[i] = float(fields[0])
            y[i] = float(fields[1])
            u[i] = float(fields[2])
            v[i] = float(fields[3])
    return x, y, u, v

def get_tke_spectrum_2d(u, v, Nx, Ny, p_order):
    """
    Calcula el espectro de energía 1D a partir de campos de velocidad 2D
    mediante un promedio acimutal del espectro 2D.
    """
    # 1. Remodelar los datos a una rejilla 2D.
    # El número de nodos por dimensión en la malla de alto orden es (Ne*p + 1)
    # Para una malla discontinua, es más complejo. Asumiremos que los datos
    # se pueden remodelar a una rejilla cuadrada para la FFT.
    nodes_per_dim = int(np.sqrt(len(u)))
    if nodes_per_dim**2 != len(u):
        print("ADVERTENCIA: El espectro 2D asume una malla de nodos cuadrada.")
        return np.array([]), np.array([])
        
    u_grid = u.reshape((nodes_per_dim, nodes_per_dim))
    v_grid = v.reshape((nodes_per_dim, nodes_per_dim))

    # 2. Restar la media para obtener las fluctuaciones
    u_prime = u_grid - np.mean(u_grid)
    v_prime = v_grid - np.mean(v_grid)

    # 3. Calcular la FFT 2D de cada componente de velocidad
    u_hat = np.fft.fft2(u_prime)
    v_hat = np.fft.fft2(v_prime)

    # 4. Calcular el espectro de energía 2D (proporcional a |û|^2 + |v̂|^2)
    # Se usa fftshift para mover la frecuencia cero al centro de la matriz
    tke_spectrum_2d = np.real(u_hat * np.conj(u_hat) + v_hat * np.conj(v_hat))
    tke_spectrum_2d_shifted = np.fft.fftshift(tke_spectrum_2d)

    # 5. Crear la rejilla de números de onda (k_x, k_y)
    kx = np.fft.fftshift(np.fft.fftfreq(nodes_per_dim, d=1.0/nodes_per_dim))
    ky = np.fft.fftshift(np.fft.fftfreq(nodes_per_dim, d=1.0/nodes_per_dim))
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # 6. Calcular el radio de los números de onda (k) y aplanarlo
    k_radius = np.sqrt(kx_grid**2 + ky_grid**2).flatten()
    tke_spectrum_flat = tke_spectrum_2d_shifted.flatten()

    # 7. Realizar el promedio acimutal
    # Se agrupa la energía en "bins" o "cajas" según su radio k
    k_bins = np.arange(0.5, nodes_per_dim//2, 1.)
    
    # Se usa np.histogram para sumar la energía en cada bin
    energy_in_bins, _ = np.histogram(k_radius, bins=k_bins, weights=tke_spectrum_flat)
    
    # Se cuenta cuántos puntos caen en cada bin para normalizar
    count_in_bins, _ = np.histogram(k_radius, bins=k_bins)
    
    # Evitar división por cero
    valid_bins = count_in_bins > 0
    
    # El espectro 1D es la energía promediada en cada anillo
    k_1d = k_bins[:-1][valid_bins]
    tke_1d = energy_in_bins[valid_bins] / count_in_bins[valid_bins]
    
    return k_1d, tke_1d

def getTKEFFT(x, U):
    """
    Calcula el Espectro de Energía Cinética para datos 1D usando FFT.
    (Esta es tu función original).
    """
    N = len(U)
    if N == 0:
        return np.array([]), np.array([])
        
    X = np.abs(np.fft.fft(U)) / N
    k = np.linspace(0, N-1, N)
    kplot = k[0:int(N/2+1)]
    Xplot = 2 * X[0:int(N/2+1)]
    
    if len(Xplot) > 0:
        Xplot[0] /= 2
    
    for k_idx in range(1, len(Xplot)):
        Xplot[k_idx] = 0.25 * Xplot[k_idx]**2

    if len(Xplot) > 0:
        Xplot = np.delete(Xplot, 0)
        kplot = np.delete(kplot, 0)
    
    return kplot, Xplot

# --- FUNCIONES DE GRÁFICA ---

def plot_1d_results(ax1, ax2, x, u, label):
    """Crea las gráficas para resultados 1D (solución y espectro)."""
    # Gráfica de la solución
    ax1.plot(x, u, label=label)
    ax1.set_title("Solución u(x)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.grid(True)
    if label: ax1.legend()
    
    # Gráfica del Espectro de Energía llamando a getTKEFFT
    # Nos aseguramos de tener al menos 2 puntos para el FFT
    if len(x) > 1:
        xt = x[:-1]
        ut = u[:-1]
        k, tke = getTKEFFT(xt, ut)
        if len(k) > 0:
            ax2.loglog(k, tke, label=label)

    ax2.set_title("Espectro de Energía (1D)")
    ax2.set_xlabel("Número de Onda (k)")
    ax2.set_ylabel("Energía")
    ax2.grid(True, which="both", ls="--")
    if label: ax2.legend()


def plot_2d_results(ax1, ax2, x, y, u, v, resolution=200):
    """Crea las gráficas de contorno para resultados 2D."""
    # 1. Crear una rejilla uniforme para la visualización
    grid_x = np.linspace(np.min(x), np.max(x), resolution)
    grid_y = np.linspace(np.min(y), np.max(y), resolution)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    
    # 2. Interpolar los datos a la rejilla uniforme
    points = np.vstack((x, y)).T
    grid_u = griddata(points, u, (grid_xx, grid_yy), method='cubic')
    grid_v = griddata(points, v, (grid_xx, grid_yy), method='cubic')
    
    # 3. Graficar los campos
    im1 = ax1.pcolormesh(grid_xx, grid_yy, grid_u, shading='auto', cmap='viridis')
    ax1.set_title("Campo de Velocidad 'u'")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect('equal', 'box')
    
    im2 = ax2.pcolormesh(grid_xx, grid_yy, grid_v, shading='auto', cmap='viridis')
    ax2.set_title("Campo de Velocidad 'v'")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect('equal', 'box')
    
    return im1, im2


# --- Bloque de Ejecución Principal (Universal) ---
if __name__ == "__main__":
    if len(argv) < 2:
        print("Uso: python postprocess.py 'patron/de/ficheros/*.txt'")
        print("Ejemplo: python postprocess.py 'data/outputs/1d_dissipation_study/*.txt'")
        sys.exit(1)

    # --- 1. Usar glob para expandir los comodines y obtener la lista de archivos ---
    files_to_process = []
    for arg in argv[1:]:
        # glob.glob encuentra todos los archivos que coinciden con el patrón
        files_to_process.extend(glob.glob(arg))

    if not files_to_process:
        print(f"Error: No se encontraron archivos para el patrón '{' '.join(argv[1:])}'")
        sys.exit(1)

    # --- 2. Crear UNA SOLA figura y ejes ANTES de empezar el bucle ---
    print(f"Superponiendo los resultados de {len(files_to_process)} archivo(s)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Comparativa de Resultados 1D", fontsize=16)

    # --- 3. Recorrer cada archivo y dibujar en los MISMOS ejes ---
    for filepath in files_to_process:
        try:
            with open(filepath, 'r') as f:
                document = f.readlines()
            
            # Detección de formato para asegurarse de que solo se procesan archivos 1D
            first_data_line = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")[0]
            if len(first_data_line.split()) != 2:
                print(f"Aviso: Se omite el archivo '{os.path.basename(filepath)}' por no ser de formato 1D.")
                continue

            print(f"Procesando y superponiendo: {os.path.basename(filepath)}")
            
            # Extraer datos del archivo
            x, u = get_mesh_and_solution_1d(document)
            
            # Crear una etiqueta limpia para la leyenda del gráfico
            label = os.path.basename(filepath).replace('.txt', '').replace('_', ' ')
            
            # Dibujar en los ejes ya existentes
            # (asumiendo que plot_1d_results dibuja en ax1 y ax2)
            plot_1d_results(ax1, ax2, x, u, label=label)

        except Exception as e:
            print(f"Ocurrió un error procesando {filepath}: {e}")

    # --- 4. Finalizar y mostrar el gráfico DESPUÉS de haber procesado todos los archivos ---
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()