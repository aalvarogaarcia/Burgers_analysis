# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:45:13 2024

@author: Jesús Pueblas
"""
import numpy as np
import sgs_model

# Get the residual for the flux reconstruction scheme and the Burgers equation
def getResidualBrurgersFR(p, U, x, v_molecular, Lp, gp, use_les=False, sgs_params=None): # <--- Nuevos argumentos
    nnode = len(U)
    R = np.zeros(nnode)
    ncells = int(nnode / (p + 1))
    
    # --- Paso 1: Calcular gradientes dU/dx (como antes) ---
    dudx = np.zeros(nnode) # Este es d(U_bar)/dx
    for i in range(0, ncells):
        inodeLneigh = (i - 1) * (p + 1) + p
        inodeRneigh = (i + 1) * (p + 1)
        if (i == 0):
            inodeLneigh = nnode - 1
        if (i == ncells - 1):
            inodeRneigh = 0
        
        uLneigh = U[inodeLneigh]
        uRneigh = U[inodeRneigh]
        inodeL = i * (p + 1)
        inodeR = i * (p + 1) + p
        uL = U[inodeL]
        uR = U[inodeR]
        
        xL_cell = x[inodeL] # Coordenada x del nodo izquierdo del elemento i
        xR_cell = x[inodeR] # Coordenada x del nodo derecho del elemento i
        dchidx = 2.0 / (xR_cell - xL_cell) if (xR_cell - xL_cell) != 0 else 0 # Jacobiano

        for j in range(0, p + 1): # j es el índice local del nodo dentro del elemento
            inodej = i * (p + 1) + j
            sumaU = 0.0
            for l_local in range(0, p + 1): # l_local es el índice local del polinomio de Lagrange
                inodel_global = i * (p + 1) + l_local
                sumaU += Lp[j][l_local] * U[inodel_global] # Lp[j_nodo_eval][l_polinomio]
            
            gpL_val = gp[j]
            gpR_val = -gp[p - j] # Como en el código original
            sumaU += 0.5 * (uLneigh - uL) * gpL_val + 0.5 * (uRneigh - uR) * gpR_val
            dudx[inodej] = sumaU * dchidx

    # --- Paso 2: Calcular flujos discontinuos ---
    F_convective_molecular = 0.5 * U * U  # Flujo convectivo molecular (o resuelto)
    F_diffusive_molecular = -v_molecular * dudx # Flujo difusivo molecular

    # --- Paso 2a: Calcular flujo SGS si LES está activado ---
    F_sgs = np.zeros(nnode) # Inicializar flujo SGS
    if use_les and sgs_params is not None:
        sgs_model_type = sgs_params.get('model_type', 'smagorinsky_dynamic')
        if sgs_model_type == 'smagorinsky_dynamic':
            # Calcular Cd dinámicamente
            # Lp y gp se pasan porque la función en sgs_model.py los necesita para calcular derivadas internas
            # si no se pasa dudx directamente.
            Cd_val_dynamic = sgs_model.calculate_dynamic_smagorinsky_constant(
                U_bar=U, 
                p_order=p, 
                x_coords=x, 
                Lp_matrix=Lp, 
                gp_array=gp, # Se necesita para las derivadas dentro de la función de Cd
                filter_width_ratio=sgs_params.get('filter_width_ratio', 2.0),
                avg_type=sgs_params.get('avg_type', 'global'),
                cs_min_val=sgs_params.get('Cs_min',0.01)
            )
            # Calcular el flujo SGS usando el Cd dinámico
            F_sgs = sgs_model.get_sgs_flux_smagorinsky_dynamic(
                U_bar=U, 
                p_order=p, 
                x_coords=x, 
                Lp_matrix=Lp, 
                gp_array=gp, # También necesita gp para el cálculo de dUb_dx interno
                Cd_dynamic=Cd_val_dynamic 
                # Los demás parámetros como filter_width_ratio no son necesarios aquí si Cd ya está calculado.
            )
        # Aquí se podrían añadir otros modelos SGS:
        # elif sgs_model_type == 'smagorinsky_standard':
        #     F_sgs = sgs_model.get_sgs_flux_smagorinsky_standard(U, dudx, p, x, sgs_params)
        # ...
    
    # Flujo convectivo total discontinuo (molecular + SGS)
    Fdc_total = F_convective_molecular + F_sgs
    Fdd_total = F_diffusive_molecular # El flujo difusivo es solo el molecular

    # --- Paso 3: Corregir flujos en las interfaces (como antes) ---
    R = np.zeros(nnode) # Residuo final
    for i in range(0, ncells):
        inodeLneigh = (i - 1) * (p + 1) + p
        inodeRneigh = (i + 1) * (p + 1)
        if (i == 0): inodeLneigh = nnode - 1
        if (i == ncells - 1): inodeRneigh = 0
            
        ULneigh = U[inodeLneigh] # U en el nodo p del elemento vecino izquierdo
        URneigh = U[inodeRneigh] # U en el nodo 0 del elemento vecino derecho
        
        # Flujos discontinuos en los bordes del elemento vecino
        # Para el flujo convectivo total
        Fdc_total_Lneigh = Fdc_total[inodeLneigh]
        Fdc_total_Rneigh = Fdc_total[inodeRneigh]
        # Para el flujo difusivo total
        Fdd_total_Lneigh = Fdd_total[inodeLneigh]
        Fdd_total_Rneigh = Fdd_total[inodeRneigh]

        inodeL_cell = i * (p + 1) # Nodo 0 del elemento actual i
        inodeR_cell = i * (p + 1) + p # Nodo p del elemento actual i
        
        UL_cell = U[inodeL_cell] # U en el nodo 0 del elemento actual
        UR_cell = U[inodeR_cell] # U en el nodo p del elemento actual

        # Flujos discontinuos en los bordes del elemento actual
        Fdc_total_L_cell = Fdc_total[inodeL_cell]
        Fdc_total_R_cell = Fdc_total[inodeR_cell]
        Fdd_total_L_cell = Fdd_total[inodeL_cell]
        Fdd_total_R_cell = Fdd_total[inodeR_cell]
        
        # Interfaz Izquierda: Flujo numérico de Rusanov para el flujo convectivo total
        # UIL es el promedio de U en la interfaz izquierda
        UIL_interface = 0.5 * (UL_cell + ULneigh)
        # abs(UIL_interface) es la velocidad de onda para Rusanov
        Fc_interface_L = 0.5 * (Fdc_total_Lneigh + Fdc_total_L_cell - abs(UIL_interface) * (UL_cell - ULneigh))
        
        # Interfaz Derecha: Flujo numérico de Rusanov para el flujo convectivo total
        # UIR es el promedio de U en la interfaz derecha
        UIR_interface = 0.5 * (UR_cell + URneigh)
        Fc_interface_R = 0.5 * (Fdc_total_R_cell + Fdc_total_Rneigh - abs(UIR_interface) * (URneigh - UR_cell)) # Ojo con el orden (URneigh - UR_cell) si se sigue la lógica de U_R - U_L

        # Para el flujo difusivo, se suele usar un promedio simple en la interfaz
        Fd_interface_L = 0.5 * (Fdd_total_Lneigh + Fdd_total_L_cell)
        Fd_interface_R = 0.5 * (Fdd_total_R_cell + Fdd_total_Rneigh)

        xL_coord_cell = x[inodeL_cell]
        xR_coord_cell = x[inodeR_cell]
        dchidx_cell = 2.0 / (xR_coord_cell - xL_coord_cell) if (xR_coord_cell - xL_coord_cell) != 0 else 0

        for j in range(0, p + 1): # j es el índice local del nodo
            inodej_global = inodeL_cell + j
            
            # Derivada de los flujos discontinuos internos
            sum_Fdc_deriv = 0.0
            sum_Fdd_deriv = 0.0
            for l_local in range(0, p + 1):
                inodel_global = inodeL_cell + l_local
                sum_Fdc_deriv += Lp[j][l_local] * Fdc_total[inodel_global]
                sum_Fdd_deriv += Lp[j][l_local] * Fdd_total[inodel_global]
            
            # Contribución de las correcciones de flujo en las interfaces
            gpL_val = gp[j]
            gpR_val = -gp[p - j]
            
            # Corrección para el flujo convectivo total
            correction_Fc = (Fc_interface_L - Fdc_total_L_cell) * gpL_val + \
                            (Fc_interface_R - Fdc_total_R_cell) * gpR_val
            
            # Corrección para el flujo difusivo total
            # La formulación original era: 0.5*(FddLneigh-FddL)*gpL+0.5*(FddRneigh-FddR)*gpR
            # Que es (Fd_interface_L - Fdd_total_L_cell)*gpL + (Fd_interface_R - Fdd_total_R_cell)*gpR si Fd_interface es promedio
            # Manteniendo la forma original:
            correction_Fd = 0.5 * (Fdd_total_Lneigh - Fdd_total_L_cell) * gpL_val + \
                            0.5 * (Fdd_total_Rneigh - Fdd_total_R_cell) * gpR_val
                            
            R[inodej_global] = -(sum_Fdc_deriv + correction_Fc + sum_Fdd_deriv + correction_Fd) * dchidx_cell
            
    return R