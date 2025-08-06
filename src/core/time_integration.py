# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:24:57 2024

@author: Jesús Pueblas
"""

from residual import *

# Estimate the time step
def RK4(dt,u,p,x,v,Lp,gp, use_les=False, sgs_params=None):
    
    N = len(u)
    us = np.zeros(N)
    unew =  np.zeros(N)
    us = u
    unew = u

    ks = getResidualBrurgersFR(p,us,x,v,Lp,gp,use_les,sgs_params)
    unew += ks*dt/6.
    us = u + ks * dt / 2.
    ks = getResidualBrurgersFR(p,us,x,v,Lp,gp,use_les,sgs_params)
    unew += ks*dt/3.
    us = u + ks * dt / 2.
    ks = getResidualBrurgersFR(p,us,x,v,Lp,gp,use_les,sgs_params)
    unew += ks*dt/3.
    us = u + ks * dt
    ks = getResidualBrurgersFR(p,us,x,v,Lp,gp,use_les,sgs_params)
    unew += ks*dt/6.
    
    return unew