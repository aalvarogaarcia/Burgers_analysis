# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 06:15:23 2024

@author: Jesús Pueblas
@co-author: Álvaro García

Funciones 1D y 2D
"""
# mesh.py

import numpy as np

# --- FUNCIONES 1D ORIGINALES (RENOMBRADAS) ---

def get_mesh_1d(N, Nref):
    """
    Genera las coordenadas de una malla 1D, refinándola Nref veces.
    """
    x = np.zeros(N)
    dx = 1.0 / (N - 1)
    for i in range(0, N):
        x[i] = i * dx

    for i in range(0, Nref):
        xold = np.copy(x)
        Nold = len(xold)
        N = 2 * Nold - 1
        x = np.zeros(N)
        for j in range(0, Nold - 1):
            x[2*j] = xold[j]
            x[2*j+1] = 0.5 * (xold[j] + xold[j+1])
        x[N-1] = xold[Nold-1]
    return x

def get_mesh_ho_1d(x, coords):
    """
    Genera los nodos de alto orden para una malla 1D.
    """
    nElements = len(x) - 1
    p = len(coords) - 1
    xho = np.zeros(nElements * (p + 1))
    for i in range(0, nElements):
        xA = x[i]
        xB = x[i+1]
        for j in range(0, p + 1):
            xho[j + i * (p + 1)] = xA + 0.5 * (xB - xA) * (1.0 + coords[j])
    return xho


# --- NUEVAS FUNCIONES PARA 2D ---

def get_2d_cartesian_mesh(Nx, Ny, Lx=1.0, Ly=1.0):
    """
    Genera los ejes de coordenadas para una malla cartesiana 2D estructurada.
    
    Args:
        Nx (int): Número de puntos en la dirección x.
        Ny (int): Número de puntos en la dirección y.
        Lx (float): Longitud del dominio en x.
        Ly (float): Longitud del dominio en y.
        
    Returns:
        tuple: Una tupla con dos arrays 1D (x_coords, y_coords).
    """
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)
    return x_coords, y_coords

def get_mesh_ho_2d(x_coords, y_coords, p_order, lobatto_points):
    """
    Genera los nodos de alto orden para una malla cartesiana 2D.
    
    Crea una rejilla de (p_order+1)x(p_order+1) nodos dentro de cada
    elemento cuadrilátero. Los nodos se devuelven en dos arrays 1D
    (x_ho, y_ho) con un ordenamiento consistente.
    
    Args:
        x_coords (np.array): Array 1D con las coordenadas x de la malla base.
        y_coords (np.array): Array 1D con las coordenadas y de la malla base.
        p_order (int): Orden del polinomio (P).
        lobatto_points (np.array): Coordenadas 1D de los puntos de Lobatto en [-1, 1].
        
    Returns:
        tuple: Una tupla con dos arrays 1D (x_ho, y_ho) que contienen todas
               las coordenadas de los nodos de alto orden.
    """
    num_elements_x = len(x_coords) - 1
    num_elements_y = len(y_coords) - 1
    nodes_per_element_1d = p_order + 1
    
    total_nodes = num_elements_x * num_elements_y * (nodes_per_element_1d**2)
    
    x_ho = np.zeros(total_nodes)
    y_ho = np.zeros(total_nodes)
    
    node_idx = 0  # Índice global para los nodos de alto orden
    
    # Iteramos sobre cada elemento de la malla
    for j in range(num_elements_y):
        for i in range(num_elements_x):
            # Coordenadas de las esquinas del elemento (i, j)
            x_sw = x_coords[i]      # Sur-Oeste
            y_sw = y_coords[j]
            x_se = x_coords[i+1]    # Sur-Este
            y_ne = y_coords[j+1]    # Nor-Este
            
            # Tamaño del elemento
            dx = x_se - x_sw
            dy = y_ne - y_sw
            
            # Iteramos sobre la rejilla de puntos de Lobatto dentro del elemento
            # El orden es importante: se recorren primero las "filas" en xi, luego las "columnas" en eta
            for l in range(nodes_per_element_1d):  # Dirección eta (local y)
                for m in range(nodes_per_element_1d):  # Dirección xi (local x)
                    
                    # Coordenadas locales en el elemento de referencia [-1, 1] x [-1, 1]
                    xi_m = lobatto_points[m]
                    eta_l = lobatto_points[l]
                    
                    # Mapeo bilineal del punto (xi_m, eta_l) a coordenadas físicas (x, y)
                    # Para una malla cartesiana, la fórmula se simplifica enormemente:
                    x_phys = x_sw + 0.5 * dx * (1.0 + xi_m)
                    y_phys = y_sw + 0.5 * dy * (1.0 + eta_l)
                    
                    x_ho[node_idx] = x_phys
                    y_ho[node_idx] = y_phys
                    node_idx += 1
                    
    return x_ho, y_ho