# sgs_model.py
import numpy as np

## --- Funciones Auxiliares (Uso Interno) --- ##

# --- AÑADIDO: Implementación del modelo dinámico para Diferencias Finitas (FD) ---
def _calculate_dynamic_viscosity_fd(U, dx, sgs_params):
    """
    Calcula la viscosidad turbulenta local para el modelo dinámico en un esquema de 
    Diferencias Finitas (FD) como DC o Upwind.
    """
    # Extraer parámetros del diccionario de configuración
    filter_ratio = sgs_params.get('filter_width_ratio', 2.0)
    avg_type = sgs_params.get('avg_type', 'global')
    cs_min = sgs_params.get('Cs_min', 0.0)
    
    # 1. Filtro de prueba para FD (convolución con filtro de caja)
    kernel_size = int(np.ceil(filter_ratio)) * 2 - 1
    kernel = np.ones(kernel_size) / kernel_size
    def apply_test_filter_fd(field):
        return np.convolve(field, kernel, mode='same')

    # 2. Aplicar filtros y calcular derivadas
    U_hat = apply_test_filter_fd(U)
    S_bar = np.gradient(U, dx)
    S_hat = np.gradient(U_hat, dx)
    abs_S_bar = np.abs(S_bar)
    abs_S_hat = np.abs(S_hat)
    
    # 3. Anchos de filtro
    Delta_sq = dx**2
    Delta_hat_sq = (filter_ratio * dx)**2

    # 4. Tensores L y M de la identidad de Germano (formulación de Lilly)
    U_bar_sq_filtered = apply_test_filter_fd(U**2)
    L_f = U_bar_sq_filtered - U_hat**2
    
    alpha = 2 * Delta_sq * abs_S_bar * S_bar
    alpha_filtered = apply_test_filter_fd(alpha)
    beta = 2 * Delta_hat_sq * abs_S_hat * S_hat
    M_f = alpha_filtered - beta

    # 5. Calcular la constante dinámica Cd = Cs^2
    if avg_type.lower() == 'global':
        numerator = np.sum(L_f * M_f)
        denominator = np.sum(M_f * M_f)
    else: # Promedio local (menos estable)
        numerator = L_f * M_f
        denominator = M_f * M_f
    
    Cd = numerator / (denominator + 1e-16) # Evitar división por cero
    Cd = np.maximum(Cd, cs_min**2) # Aplicar clipping
    
    # 6. Devolver la viscosidad SGS local: nu_sgs = Cd * Delta^2 * |S_bar|
    nu_sgs = Cd * Delta_sq * abs_S_bar
    
    # Guardar el Cd promedio para monitoreo
    global _last_calculated_Cd_dynamic
    _last_calculated_Cd_dynamic = np.mean(Cd) if isinstance(Cd, np.ndarray) else Cd
    
    return nu_sgs

# (Las funciones auxiliares existentes para FR permanecen aquí)
def _calculate_fr_derivative_1d(U_field, p_order, x_coords, Lp_matrix, gp_array):
    # ... (código existente sin cambios)
    n_nodes = len(U_field)
    n_cells = n_nodes // (p_order + 1)
    dU_dx = np.zeros_like(U_field)
    for i in range(n_cells):
        inodeLneigh = (i - 1) * (p_order + 1) + p_order
        inodeRneigh = (i + 1) * (p_order + 1)
        if i == 0: inodeLneigh = n_nodes - 1
        if i == n_cells - 1: inodeRneigh = 0
        uLneigh = U_field[inodeLneigh]
        uRneigh = U_field[inodeRneigh]
        inodeL_cell = i * (p_order + 1)
        inodeR_cell = i * (p_order + 1) + p_order
        uL_cell = U_field[inodeL_cell]
        uR_cell = U_field[inodeR_cell]
        xL_cell = x_coords[inodeL_cell]
        xR_cell = x_coords[inodeR_cell]
        dchidx = 2.0 / (xR_cell - xL_cell) if (xR_cell - xL_cell) != 0 else 0
        for j_local in range(p_order + 1):
            inodej_global = inodeL_cell + j_local
            sumaU_deriv = 0.0
            for l_local in range(p_order + 1):
                inodel_global = inodeL_cell + l_local
                sumaU_deriv += Lp_matrix[j_local][l_local] * U_field[inodel_global]
            gpL_val = gp_array[j_local]
            gpR_val = -gp_array[p_order - j_local]
            sumaU_deriv += 0.5 * (uLneigh - uL_cell) * gpL_val + 0.5 * (uRneigh - uR_cell) * gpR_val
            dU_dx[inodej_global] = sumaU_deriv * dchidx
    return dU_dx

def apply_test_filter(field_data, p_order, x_coords, n_cells, filter_width_ratio=2.0):
    # ... (código existente sin cambios)
    field_filtered = np.zeros_like(field_data)
    num_neighbor_cells = int(np.floor(filter_width_ratio / 2.0))
    for i in range(n_cells):
        cell_indices = [(i + j) % n_cells for j in range(-num_neighbor_cells, num_neighbor_cells + 1)]
        nodes_in_window = []
        for cell_idx in cell_indices:
            start_node = cell_idx * (p_order + 1)
            end_node = start_node + (p_order + 1)
            nodes_in_window.extend(field_data[start_node:end_node])
        avg_val = np.mean(nodes_in_window) if nodes_in_window else 0
        target_nodes = slice(i * (p_order + 1), (i + 1) * (p_order + 1))
        field_filtered[target_nodes] = avg_val
    return field_filtered

## --- Interfaz para Esquemas de Bajo Orden (DC/Upwind) --- ##

# --- AÑADIDO: La función "dispatcher" que soluciona el error ---
def calculate_sgs_viscosity_1d(U, dudx, dx, sgs_params):
    """
    Interfaz principal para calcular la viscosidad SGS en esquemas de bajo orden.
    Llama al modelo correspondiente especificado en sgs_params.
    """
    # El segundo argumento 'dudx' se mantiene por compatibilidad con la llamada,
    # aunque no es usado por el modelo dinámico, que calcula sus propias derivadas.
    model_type = sgs_params.get('model_type', '').lower()

    if model_type == 'smagorinsky_dynamic':
        return _calculate_dynamic_viscosity_fd(U, dx, sgs_params)
    
    # Aquí se podrían añadir otros modelos para DC/Upwind en el futuro
    # elif model_type == 'smagorinsky_constant':
    #     Cs = sgs_params.get('Cs', 0.1)
    #     nu_sgs = (Cs * dx)**2 * np.abs(dudx)
    #     return nu_sgs
    
    else:
        # Si el modelo no es reconocido, no se añade viscosidad SGS.
        if model_type: # Solo mostrar advertencia si se especificó un modelo
            print(f"ADVERTENCIA: Modelo SGS '{model_type}' no reconocido para 1D FD. No se añade viscosidad.")
        return 0.0

## --- Modelo Dinámico Smagorinsky para 1D FR --- ##
# ... (Todo el resto del archivo, desde aquí hasta el final, permanece sin cambios)

_last_calculated_Cd_dynamic = 0.0

def calculate_dynamic_smagorinsky_constant(U_bar, p_order, x_coords, Lp_matrix, gp_array,
                                           filter_width_ratio=2.0, avg_type='global',
                                           cs_min=0.01):
    global _last_calculated_Cd_dynamic
    n_nodes = len(U_bar)
    n_cells = n_nodes // (p_order + 1)
    fr_args = (p_order, x_coords, Lp_matrix, gp_array)
    dUb_dx = _calculate_fr_derivative_1d(U_bar, *fr_args)
    abs_S_bar = np.abs(dUb_dx)
    U_hat = apply_test_filter(U_bar, p_order, x_coords, n_cells, filter_width_ratio)
    dUh_dx = _calculate_fr_derivative_1d(U_hat, *fr_args)
    abs_S_hat = np.abs(dUh_dx)
    Delta_sq_local = np.zeros(n_nodes)
    for i in range(n_cells):
        inodeL_cell = i * (p_order + 1)
        inodeR_cell = i * (p_order + 1) + p_order
        h_e = x_coords[inodeR_cell] - x_coords[inodeL_cell]
        delta_val = h_e / (p_order + 1.0) if p_order > 0 else h_e
        Delta_sq_local[inodeL_cell:inodeR_cell + 1] = delta_val**2
    Delta_hat_sq_local = (filter_width_ratio**2) * Delta_sq_local
    U_bar_sq_filtered = apply_test_filter(U_bar**2, p_order, x_coords, n_cells, filter_width_ratio)
    L_f = U_bar_sq_filtered - U_hat**2
    alpha = 2 * Delta_sq_local * abs_S_bar * dUb_dx
    alpha_filtered = apply_test_filter(alpha, p_order, x_coords, n_cells, filter_width_ratio)
    beta = 2 * Delta_hat_sq_local * abs_S_hat * dUh_dx
    M_f = alpha_filtered - beta
    if avg_type.lower() == 'global':
        numerator = np.sum(L_f * M_f)
        denominator = np.sum(M_f * M_f)
    else:
        print("ADVERTENCIA: Promediado local no implementado, usando global.")
        numerator = np.sum(L_f * M_f)
        denominator = np.sum(M_f * M_f)
    Cd_val = numerator / denominator if np.abs(denominator) > 1e-12 else 0.0
    Cd_min = cs_min**2 
    Cd_val = max(Cd_val, Cd_min)
    _last_calculated_Cd_dynamic = Cd_val
    return Cd_val

def get_sgs_flux_smagorinsky_dynamic(U_bar, p_order, x_coords, Lp_matrix, gp_array, Cd_dynamic):
    dUb_dx = _calculate_fr_derivative_1d(U_bar, p_order, x_coords, Lp_matrix, gp_array)
    abs_S_bar = np.abs(dUb_dx)
    n_nodes = len(U_bar)
    n_cells = n_nodes // (p_order + 1)
    Delta_sq_local = np.zeros(n_nodes)
    for i in range(n_cells):
        inodeL_cell = i * (p_order + 1)
        inodeR_cell = i * (p_order + 1) + p_order
        h_e = x_coords[inodeR_cell] - x_coords[inodeL_cell]
        delta_val = h_e / (p_order + 1.0) if p_order > 0 else h_e
        Delta_sq_local[inodeL_cell:inodeR_cell + 1] = delta_val**2
    nu_SGS_local = Cd_dynamic * Delta_sq_local * abs_S_bar
    tau_SGS = -nu_SGS_local * dUb_dx
    return tau_SGS

## --- Modelos de Viscosidad Turbulenta 2D (Smagorinsky & Vreman) --- ##

def calculate_smagorinsky_eddy_viscosity(dudx, dudy, dvdx, dvdy, Nx, Ny, Cs):
    num_nodes = len(dudx)
    nu_e = np.zeros(num_nodes)
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    delta_sq = dx * dy 
    for i in range(num_nodes):
        S11 = dudx[i]
        S22 = dvdy[i]
        S12 = 0.5 * (dudy[i] + dvdx[i])
        S_mag = np.sqrt(2 * (S11**2 + S22**2 + 2*S12**2))
        nu_e[i] = (Cs**2) * delta_sq * S_mag
    return nu_e

def calculate_vreman_eddy_viscosity(dudx, dudy, dvdx, dvdy, Nx, Ny, c_vreman=0.07):
    num_nodes = len(dudx)
    nu_e = np.zeros(num_nodes)
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    delta_sq = dx * dy
    for i in range(num_nodes):
        alpha_11, alpha_12 = dudx[i], dvdx[i]
        alpha_21, alpha_22 = dudy[i], dvdy[i]
        norm_alpha_sq = alpha_11**2 + alpha_12**2 + alpha_21**2 + alpha_22**2
        if norm_alpha_sq < 1e-12:
            nu_e[i] = 0.0
            continue
        beta_11 = delta_sq * (alpha_11**2 + alpha_21**2)
        beta_12 = delta_sq * (alpha_11*alpha_12 + alpha_21*alpha_22)
        beta_22 = delta_sq * (alpha_12**2 + alpha_22**2)
        B_beta = beta_11 * beta_22 - beta_12**2
        B_beta = max(0, B_beta)
        nu_e[i] = c_vreman * np.sqrt(B_beta / norm_alpha_sq)
    return nu_e

def get_sgs_stress_vreman(nu_e, dudx, dudy, dvdx, dvdy):
    S11 = dudx
    S22 = dvdy
    S12 = 0.5 * (dudy + dvdx)
    tau_xx = -2 * nu_e * S11
    tau_yy = -2 * nu_e * S22
    tau_xy = -2 * nu_e * S12
    return tau_xx, tau_yy, tau_xy

## --- Funciones de Utilidad --- ##

def get_last_calculated_Cd():
    global _last_calculated_Cd_dynamic
    return _last_calculated_Cd_dynamic