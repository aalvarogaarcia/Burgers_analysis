# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:24:57 2024

@author: Jesús Pueblas
"""

# ode.py

import numpy as np

def RK4(dt, U, residual_func, *args):
    """
    Integrador Runge-Kutta de 4º orden genérico.

    Args:
        dt (float): Paso de tiempo.
        U (np.array): Vector de estado actual.
        residual_func (function): La función que calcula el residuo (ej. get_residual_2d).
        *args: Argumentos adicionales que necesite la función de residuo.
    """
    unew = np.copy(U)
    us = np.copy(U)

    # k1
    ks = residual_func(us, *args)
    unew += ks * dt / 6.0
    
    # k2
    us = U + ks * dt / 2.0
    ks = residual_func(us, *args)
    unew += ks * dt / 3.0
    
    # k3
    us = U + ks * dt / 2.0
    ks = residual_func(us, *args)
    unew += ks * dt / 3.0
    
    # k4
    us = U + ks * dt
    ks = residual_func(us, *args)
    unew += ks * dt / 6.0
    
    return unew