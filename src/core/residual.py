# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:45:13 2024

@author: Jesús Pueblas
"""
import numpy as np
from ..models.sgs_model import *
from ..models import sgs_model
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


def calculate_gradients_2d(U, p, x_coords, y_coords, lagrange_data, Nx, Ny):
    num_nodes_per_var = len(U) // 2
    u, v = U[:num_nodes_per_var], U[num_nodes_per_var:]
    Lp_matrix, gp_array = lagrange_data
    num_elements_x, num_elements_y = Nx - 1, Ny - 1
    nodes_per_element_1d = p + 1
    dudx, dudy = np.zeros(num_nodes_per_var), np.zeros(num_nodes_per_var)
    dvdx, dvdy = np.zeros(num_nodes_per_var), np.zeros(num_nodes_per_var)

    for j_elem in range(num_elements_y):
        for i_elem in range(num_elements_x):
            base_idx = (j_elem * num_elements_x + i_elem) * (nodes_per_element_1d**2)
            element_nodes_indices = [base_idx + l * nodes_per_element_1d + m for l in range(nodes_per_element_1d) for m in range(nodes_per_element_1d)]
            u_element = u[element_nodes_indices].reshape((nodes_per_element_1d, nodes_per_element_1d))
            v_element = v[element_nodes_indices].reshape((nodes_per_element_1d, nodes_per_element_1d))
            deriv_u_ksi, deriv_v_ksi = u_element @ Lp_matrix.T, v_element @ Lp_matrix.T
            deriv_u_eta, deriv_v_eta = Lp_matrix @ u_element, Lp_matrix @ v_element
            idx_sw, idx_se = element_nodes_indices[0], element_nodes_indices[p]
            idx_nw = element_nodes_indices[p * nodes_per_element_1d]
            dx, dy = x_coords[idx_se] - x_coords[idx_sw], y_coords[idx_nw] - y_coords[idx_sw]
            dksi_dx, deta_dy = (2.0 / dx if dx != 0 else 0), (2.0 / dy if dy != 0 else 0)
            dudx[element_nodes_indices] = (deriv_u_ksi * dksi_dx).flatten()
            dudy[element_nodes_indices] = (deriv_u_eta * deta_dy).flatten()
            dvdx[element_nodes_indices] = (deriv_v_ksi * dksi_dx).flatten()
            dvdy[element_nodes_indices] = (deriv_v_eta * deta_dy).flatten()
    return dudx, dudy, dvdx, dvdy

def get_residual_2d(U, p, coords, v_molecular, lagrange_data, Nx, Ny, use_les=False, sgs_params=None):
    x_ho, y_ho = coords
    Lp_matrix, gp_array = lagrange_data
    num_nodes_per_var = len(U) // 2
    u, v_vel = U[:num_nodes_per_var], U[num_nodes_per_var:]

    dudx, dudy, dvdx, dvdy = calculate_gradients_2d(U, p, x_ho, y_ho, lagrange_data, Nx, Ny)
    
    nu_e = np.zeros(num_nodes_per_var)
    if use_les and sgs_params:
        model_type = sgs_params.get('model_type')
        if model_type == 'vreman':
            c_vreman = sgs_params.get('c_vreman', 0.07)
            nu_e = sgs_model.calculate_vreman_eddy_viscosity(dudx, dudy, dvdx, dvdy, p, Nx, Ny, x_ho, y_ho, c_vreman)
        elif model_type == 'smagorinsky':
            Cs = sgs_params.get('Cs', 0.1)
            nu_e = sgs_model.calculate_smagorinsky_eddy_viscosity(dudx, dudy, dvdx, dvdy, p, Nx, Ny, Cs)
            
    total_viscosity = v_molecular + nu_e

    Fu, Gu = (0.5 * u**2 - total_viscosity * dudx), (u * v_vel - total_viscosity * dudy)
    Fv, Gv = (u * v_vel - total_viscosity * dvdx), (0.5 * v_vel**2 - total_viscosity * dvdy)
    
    div_F_u, div_G_u = np.zeros(num_nodes_per_var), np.zeros(num_nodes_per_var)
    div_F_v, div_G_v = np.zeros(num_nodes_per_var), np.zeros(num_nodes_per_var)
    
    num_elements_x, num_elements_y = Nx - 1, Ny - 1
    nodes_per_element_1d = p + 1

    for j_elem in range(num_elements_y):
        for i_elem in range(num_elements_x):
            base_idx = (j_elem * num_elements_x + i_elem) * (nodes_per_element_1d**2)
            i_L, i_R = (i_elem - 1 + num_elements_x) % num_elements_x, (i_elem + 1) % num_elements_x
            j_B, j_T = (j_elem - 1 + num_elements_y) % num_elements_y, (j_elem + 1) % num_elements_y
            
            for l in range(nodes_per_element_1d):
                idx_L, idx_R = base_idx + l*nodes_per_element_1d, base_idx + l*nodes_per_element_1d + p
                idx_L_neigh = (j_elem*num_elements_x + i_L)*(nodes_per_element_1d**2) + l*nodes_per_element_1d + p
                idx_R_neigh = (j_elem*num_elements_x + i_R)*(nodes_per_element_1d**2) + l*nodes_per_element_1d
                
                u_normal_L = 0.5 * (u[idx_L] + u[idx_L_neigh])
                F_common_u_L = 0.5 * (Fu[idx_L] + Fu[idx_L_neigh] - abs(u_normal_L) * (u[idx_L] - u[idx_L_neigh]))
                F_common_v_L = 0.5 * (Fv[idx_L] + Fv[idx_L_neigh] - abs(u_normal_L) * (v_vel[idx_L] - v_vel[idx_L_neigh]))
                
                u_normal_R = 0.5 * (u[idx_R] + u[idx_R_neigh])
                F_common_u_R = 0.5 * (Fu[idx_R] + Fu[idx_R_neigh] - abs(u_normal_R) * (u[idx_R] - u[idx_R_neigh]))
                F_common_v_R = 0.5 * (Fv[idx_R] + Fv[idx_R_neigh] - abs(u_normal_R) * (v_vel[idx_R] - v_vel[idx_R_neigh]))
                
                for m in range(nodes_per_element_1d):
                    curr = base_idx + l*nodes_per_element_1d + m
                    gp_L, gp_R = gp_array[m], -gp_array[p-m]
                    div_F_u[curr] += (F_common_u_L - Fu[idx_L])*gp_L + (F_common_u_R - Fu[idx_R])*gp_R
                    div_F_v[curr] += (F_common_v_L - Fv[idx_L])*gp_L + (F_common_v_R - Fv[idx_R])*gp_R

            for m in range(nodes_per_element_1d):
                idx_B, idx_T = base_idx + m, base_idx + p*nodes_per_element_1d + m
                idx_B_neigh = (j_B*num_elements_x + i_elem)*(nodes_per_element_1d**2) + p*nodes_per_element_1d + m
                idx_T_neigh = (j_T*num_elements_x + i_elem)*(nodes_per_element_1d**2) + m
                
                v_normal_B = 0.5 * (v_vel[idx_B] + v_vel[idx_B_neigh])
                G_common_u_B = 0.5 * (Gu[idx_B] + Gu[idx_B_neigh] - abs(v_normal_B) * (u[idx_B] - u[idx_B_neigh]))
                G_common_v_B = 0.5 * (Gv[idx_B] + Gv[idx_B_neigh] - abs(v_normal_B) * (v_vel[idx_B] - v_vel[idx_B_neigh]))

                v_normal_T = 0.5 * (v_vel[idx_T] + v_vel[idx_T_neigh])
                G_common_u_T = 0.5 * (Gu[idx_T] + Gu[idx_T_neigh] - abs(v_normal_T) * (u[idx_T] - u[idx_T_neigh]))
                G_common_v_T = 0.5 * (Gv[idx_T] + Gv[idx_T_neigh] - abs(v_normal_T) * (v_vel[idx_T] - v_vel[idx_T_neigh]))

                for l in range(nodes_per_element_1d):
                    curr = base_idx + l*nodes_per_element_1d + m
                    gp_B, gp_T = gp_array[l], -gp_array[p-l]
                    div_G_u[curr] += (G_common_u_B - Gu[idx_B])*gp_B + (G_common_u_T - Gu[idx_T])*gp_T
                    div_G_v[curr] += (G_common_v_B - Gv[idx_B])*gp_B + (G_common_v_T - Gv[idx_T])*gp_T

    num_elem = num_elements_x * num_elements_y
    for i in range(num_elem):
        nodes_slice = slice(i*nodes_per_element_1d**2, (i+1)*nodes_per_element_1d**2)
        div_F_u[nodes_slice] += (Fu[nodes_slice].reshape((nodes_per_element_1d, nodes_per_element_1d)) @ Lp_matrix.T).flatten()
        div_G_u[nodes_slice] += (Lp_matrix @ Gu[nodes_slice].reshape((nodes_per_element_1d, nodes_per_element_1d))).flatten()
        div_F_v[nodes_slice] += (Fv[nodes_slice].reshape((nodes_per_element_1d, nodes_per_element_1d)) @ Lp_matrix.T).flatten()
        div_G_v[nodes_slice] += (Lp_matrix @ Gv[nodes_slice].reshape((nodes_per_element_1d, nodes_per_element_1d))).flatten()

    R_u, R_v = np.zeros(num_nodes_per_var), np.zeros(num_nodes_per_var)
    for j_elem in range(num_elements_y):
        for i_elem in range(num_elements_x):
            base_idx = (j_elem * num_elements_x + i_elem) * (nodes_per_element_1d**2)
            nodes_slice = slice(base_idx, base_idx + nodes_per_element_1d**2)
            dx = x_ho[base_idx + p] - x_ho[base_idx]
            dy = y_ho[base_idx + p*nodes_per_element_1d] - y_ho[base_idx]
            dksi_dx, deta_dy = (2.0/dx if dx!=0 else 0), (2.0/dy if dy!=0 else 0)
            
            R_u[nodes_slice] = - (div_F_u[nodes_slice] * dksi_dx + div_G_u[nodes_slice] * deta_dy)
            R_v[nodes_slice] = - (div_F_v[nodes_slice] * dksi_dx + div_G_v[nodes_slice] * deta_dy)

    return np.concatenate((R_u, R_v))





# ==============================================================================
# --- MODELOS SGS PARA DIFERENCIAS CENTRADAS ---
# ==============================================================================

def calculate_smagorinsky_cd_2d(dudx, dudy, dvdx, dvdy, dx, dy, Cs):
    """Calcula la viscosidad turbulenta para el modelo de Smagorinsky en una malla CD."""
    # El ancho del filtro Delta se basa en el volumen de la celda.
    delta_sq = dx * dy
    
    # Tensor de velocidad de deformación S_ij
    S11 = dudx
    S12 = 0.5 * (dudy + dvdx)
    S22 = dvdy
    
    # Magnitud del tensor de deformación |S| = sqrt(2 * S_ij * S_ij)
    S_mag_sq = 2 * (S11**2 + 2 * S12**2 + S22**2)
    S_mag = np.sqrt(S_mag_sq)
    
    # Viscosidad turbulenta nu_t = (Cs * Delta)^2 * |S|
    nu_t = (Cs**2) * delta_sq * S_mag
    return nu_t

def calculate_vreman_cd_2d(dudx, dudy, dvdx, dvdy, dx, dy, Cv):
    """Calcula la viscosidad turbulenta para el modelo de Vreman en una malla CD."""
    delta_sq = dx * dy
    nu_t = np.zeros_like(dudx)

    for i in range(len(dudx)):
        # Tensor gradiente de velocidad G
        G = np.array([[dudx[i], dudy[i]], [dvdx[i], dvdy[i]]])
        
        # Invariante beta (simplificado para 2D)
        beta = G[0,0]**2 + G[0,1]**2 + G[1,0]**2 + G[1,1]**2
        
        if beta < 1e-12:
            continue
            
        # Viscosidad turbulenta nu_t = Cv * Delta * sqrt(det(G*G^T) / beta)
        # Para 2D, esto se simplifica. Una formulación común es:
        # nu_t = Cv * sqrt( (B_beta) / (alpha_ij*alpha_ij) ) * Delta
        # Donde B_beta es un término más complejo. Usamos una aproximación robusta.
        nu_t[i] = (Cv**2) * delta_sq * np.sqrt(np.abs(np.linalg.det(G)))

    return nu_t

# ==============================================================================
# --- CÁLCULO DE DERIVADAS Y RESIDUOS ---
# ==============================================================================

def calculate_cd_derivative_1d(field, dx):
    """Calcula la derivada 1D con condiciones periódicas."""
    dfdx = np.zeros_like(field)
    dfdx[1:-1] = (field[2:] - field[:-2]) / (2 * dx)
    dfdx[0] = (field[1] - field[-1]) / (2 * dx)
    dfdx[-1] = (field[0] - field[-2]) / (2 * dx)
    return dfdx

def getResidual_CD_1D(U, p, x, v, Lp, gp, N, use_les, sgs_params):
    """Calcula el residuo 1D para CD, ahora con LES (Smagorinsky estático)."""
    dx = x[1] - x[0]
    
    dUdx = calculate_cd_derivative_1d(U, dx)
    
    # Viscosidad turbulenta
    nu_sgs = 0.0
    if use_les:
        sgs_model_type = sgs_params.get('SGS_MODEL_TYPE', 'NONE')
        if sgs_model_type == 'SMAGORINSKY':
            Cs = sgs_params.get('CS', 0.1)
            delta_sq = dx**2
            nu_sgs = (Cs**2) * delta_sq * np.abs(dUdx)
        else:
            print(f"ADVERTENCIA: Modelo LES '{sgs_model_type}' no implementado para CD 1D.")
            
    nu_total = v + nu_sgs
    
    # Flujo convectivo F = U^2 / 2
    F = 0.5 * U**2
    dFdx = calculate_cd_derivative_1d(F, dx)
    
    # Flujo disipativo total G = nu_total * dU/dx
    G = nu_total * dUdx
    dGdx = calculate_cd_derivative_1d(G, dx)
    
    R = dGdx - dFdx
    return R

def calculate_cd_derivative_2d(field, dx, dy, Nx, Ny, direction):
    """Calcula la derivada 2D con condiciones periódicas."""
    df_dir = np.zeros_like(field)
    field_2d = field.reshape((Ny, Nx))
    
    if direction == 'x':
        denom = 2 * dx
        for j in range(Ny):
            for i in range(Nx):
                i_plus_1, i_minus_1 = (i + 1) % Nx, (i - 1 + Nx) % Nx
                val = (field_2d[j, i_plus_1] - field_2d[j, i_minus_1]) / denom
                df_dir[j * Nx + i] = val
    elif direction == 'y':
        denom = 2 * dy
        for j in range(Ny):
            for i in range(Nx):
                j_plus_1, j_minus_1 = (j + 1) % Ny, (j - 1 + Ny) % Ny
                val = (field_2d[j_plus_1, i] - field_2d[j_minus_1, i]) / denom
                df_dir[j * Nx + i] = val
    return df_dir

def getResidual_CD_2D(U, p, coords, v, basis, Nx, Ny, use_les, sgs_params):
    """Calcula el residuo 2D para CD, ahora con LES."""
    u, v_field = U[:Nx*Ny], U[Nx*Ny:]
    dx = 1.0 / (Nx - 1) if Nx > 1 else 1.0
    dy = 1.0 / (Ny - 1) if Ny > 1 else 1.0

    # 1. Calcular derivadas de velocidad
    dudx = calculate_cd_derivative_2d(u, dx, dy, Nx, Ny, 'x')
    dudy = calculate_cd_derivative_2d(u, dx, dy, Nx, Ny, 'y')
    dvdx = calculate_cd_derivative_2d(v_field, dx, dy, Nx, Ny, 'x')
    dvdy = calculate_cd_derivative_2d(v_field, dx, dy, Nx, Ny, 'y')

    # 2. Calcular viscosidad turbulenta (SGS)
    nu_sgs = 0.0
    if use_les:
        sgs_model_type = sgs_params.get('SGS_MODEL_TYPE', 'NONE')
        if sgs_model_type == 'SMAGORINSKY':
            Cs = sgs_params.get('CS', 0.1)
            nu_sgs = calculate_smagorinsky_cd_2d(dudx, dudy, dvdx, dvdy, dx, dy, Cs)
        elif sgs_model_type == 'VREMAN':
            Cv = sgs_params.get('CV', 0.1)
            nu_sgs = calculate_vreman_cd_2d(dudx, dudy, dvdx, dvdy, dx, dy, Cv)
        else:
            print(f"ADVERTENCIA: Modelo LES '{sgs_model_type}' no implementado para CD 2D.")
            
    nu_total = v + nu_sgs

    # 3. Calcular flujos convectivos y sus derivadas
    F_u, G_u = u*u, u*v_field
    F_v, G_v = u*v_field, v*v_field
    dFudx = calculate_cd_derivative_2d(F_u, dx, dy, Nx, Ny, 'x')
    dGudy = calculate_cd_derivative_2d(G_u, dx, dy, Nx, Ny, 'y')
    dFvdx = calculate_cd_derivative_2d(F_v, dx, dy, Nx, Ny, 'x')
    dGvdy = calculate_cd_derivative_2d(G_v, dx, dy, Nx, Ny, 'y')
    
    # 4. Calcular flujos disipativos totales y sus derivadas
    G_diss_u_x, G_diss_u_y = nu_total * dudx, nu_total * dudy
    G_diss_v_x, G_diss_v_y = nu_total * dvdx, nu_total * dvdy
    dG_diss_udx = calculate_cd_derivative_2d(G_diss_u_x, dx, dy, Nx, Ny, 'x')
    dG_diss_udy = calculate_cd_derivative_2d(G_diss_u_y, dx, dy, Nx, Ny, 'y')
    dG_diss_vdx = calculate_cd_derivative_2d(G_diss_v_x, dx, dy, Nx, Ny, 'x')
    dG_diss_vdy = calculate_cd_derivative_2d(G_diss_v_y, dx, dy, Nx, Ny, 'y')

    # 5. Ensamblar el residuo final
    R_u = (dG_diss_udx + dG_diss_udy) - (dFudx + dGudy)
    R_v = (dG_diss_vdx + dG_diss_vdy) - (dFvdx + dGvdy)

    return np.concatenate((R_u, R_v))


































