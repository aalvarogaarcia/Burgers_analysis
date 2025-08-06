# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:24:57 2024

@author: Jesús Pueblas
"""

from src.core.discretization import *

def rk4(dt, u, residual_func, residual_args):
    """
    Implementación genérica del método Runge-Kutta de 4º orden.

    Args:
        dt (float): Paso de tiempo.
        u (np.ndarray): El campo de la solución actual.
        residual_func (function): La función que calcula el residuo 
                                  (ej. getResidual_2D o getResidual).
        residual_args (tuple): Una tupla que contiene todos los argumentos 
                               adicionales que necesita la 'residual_func'.
    
    Returns:
        np.ndarray: El campo de la solución en el siguiente paso de tiempo.
    """
    
    # k1
    r = residual_func(u, *residual_args)
    k1 = dt * r

    # k2
    r = residual_func(u + 0.5 * k1, *residual_args)
    k2 = dt * r

    # k3
    r = residual_func(u + 0.5 * k2, *residual_args)
    k3 = dt * r

    # k4
    r = residual_func(u + k3, *residual_args)
    k4 = dt * r

    u_new = u + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    
    return u_new
