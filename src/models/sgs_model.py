# sgs_model.py
import numpy as np

# Necesitaremos funciones de lagpol.py y posiblemente mesh.py si no se pasan todos los datos
# from lagpol import getLobattoPoints, Lp, gp # O mejor, pasar estos como argumentos

def apply_test_filter(field_data, p_order, x_coords, elements_connectivity, filter_width_ratio=2.0):
    """
    Aplica un filtro de prueba espacial al campo 'field_data'.
    Este es un punto crítico y su implementación depende de la naturaleza de la malla FR.
    Una aproximación para FR podría ser:
    1. Reconstruir la solución polinomial en cada elemento.
    2. Aplicar el filtro de convolución sobre la reconstrucción.
    3. Evaluar el resultado en los puntos de solución.

    Como primera aproximación más simple (y menos precisa para FR general):
    Promediar los valores de los nodos en un vecindario más amplio.
    O, si los elementos son de tamaño similar, promediar los valores de elementos vecinos.

    Args:
        field_data (np.array): El campo 1D a filtrar (ej. U o U^2/2).
        p_order (int): Orden del polinomio.
        x_coords (np.array): Coordenadas de los nodos.
        elements_connectivity (list or np.array): Información sobre qué nodos pertenecen a qué elemento.
                                                Podría ser simplemente ncells = len(field_data) // (p_order + 1).
        filter_width_ratio (float): Relación entre el ancho del filtro de prueba y el de la malla.
                                    Un valor de 2 significa que el filtro de prueba es el doble de ancho.
    Returns:
        np.array: El campo filtrado.
    """
    n_nodes = len(field_data)
    n_cells = n_nodes // (p_order + 1)
    field_filtered = np.zeros_like(field_data)

    # Placeholder para una implementación de filtro de prueba.
    # Esta es una parte no trivial y debe diseñarse con cuidado.
    # Ejemplo MUY simplificado: promedio sobre un número de celdas vecinas.
    # Para FR, esto debería ser más sofisticado (ej. reconstrucción y luego integración).
    
    # print(f"ADVERTENCIA: 'apply_test_filter' usa un promedio móvil simple y necesita revisión para FR.")
    # field_filtered = np.convolve(field_data, np.ones(window_size_nodes)/window_size_nodes, mode='same')
    
    # Un enfoque más orientado a celdas (aún simplificado):
    # Para cada celda, promediar los valores de las celdas dentro de un vecindario
    # definido por filter_width_ratio.
    num_neighbor_cells = int(np.floor(filter_width_ratio / 2.0)) # Celdas a cada lado

    for i in range(n_cells):
        # Determinar el rango de celdas para el promedio
        start_cell_idx = max(0, i - num_neighbor_cells)
        end_cell_idx = min(n_cells, i + num_neighbor_cells + 1)
        
        # Extraer los nodos de estas celdas
        nodes_in_filter_window = []
        for cell_k_idx in range(start_cell_idx, end_cell_idx):
            start_node = cell_k_idx * (p_order + 1)
            end_node = (cell_k_idx + 1) * (p_order + 1)
            nodes_in_filter_window.extend(field_data[start_node:end_node])
            
        if nodes_in_filter_window:
            avg_val = np.mean(nodes_in_filter_window)
        else:
            # Fallback si la ventana está vacía (no debería ocurrir con buena lógica)
            current_cell_nodes = slice(i * (p_order + 1), (i + 1) * (p_order + 1))
            avg_val = np.mean(field_data[current_cell_nodes])

        # Asignar el valor promediado a todos los nodos de la celda central 'i'
        target_nodes = slice(i * (p_order + 1), (i + 1) * (p_order + 1))
        field_filtered[target_nodes] = avg_val
        
    # Implementar condiciones de contorno periódicas para el filtro si es necesario
    # Este filtro es bastante crudo y puede necesitar ser reemplazado por uno más adecuado para FR,
    # como una proyección L2 a un espacio de menor orden o una convolución más formal.
    #print(f"ADVERTENCIA: 'apply_test_filter' usa un promedio simple basado en celdas y necesita revisión para FR.")
    return field_filtered

_last_calculated_Cd_dynamic = 0.0

def calculate_dynamic_smagorinsky_constant(U_bar, p_order, x_coords, Lp_matrix, gp_array,
                                           filter_width_ratio=2.0, avg_type='global',
                                           cs_min_val=0.01):
    """
    Calcula la constante dinámica Cd = Cs^2.

    Args:
        U_bar (np.array): Campo de velocidad resuelto.
        p_order (int): Orden del polinomio.
        x_coords (np.array): Coordenadas de los nodos.
        Lp_matrix (np.array): Matriz de derivadas de Lagrange (p+1, p+1).
        gp_array (np.array): Array de funciones de corrección de FR en los bordes (p+1).
        filter_width_ratio (float): Relación Delta_hat / Delta.
        avg_type (str): 'global' para promediar sobre todo el dominio, 
                        'local' para un promedio local (más complejo de implementar aquí).
    Returns:
        float: Constante Cd. Puede ser un array si avg_type es 'local' y se calcula por elemento.
               Por ahora, devuelve un float global.
    """
    global _last_calculated_Cd_dynamic # Indicar que vamos a modificar la variable global del módulo
    
    n_nodes = len(U_bar)
    n_cells = n_nodes // (p_order + 1)

    # --- 0. Calcular derivadas del campo resuelto d(U_bar)/dx ---
    # Esta es la derivada completa al estilo FR, como en residual.py
    dUb_dx = np.zeros_like(U_bar)
    # (Reutilizar la lógica de residual.py para calcular d(U_bar)/dx)
    # (Este código es una réplica de la lógica en residual.py para dudx)
    for i in range(n_cells):
        inodeLneigh = (i - 1) * (p_order + 1) + p_order
        inodeRneigh = (i + 1) * (p_order + 1)
        if i == 0: inodeLneigh = n_nodes - 1 # Asumiendo periodicidad
        if i == n_cells - 1: inodeRneigh = 0 # Asumiendo periodicidad
        
        uLneigh = U_bar[inodeLneigh]
        uRneigh = U_bar[inodeRneigh]
        inodeL_cell = i * (p_order + 1)
        inodeR_cell = i * (p_order + 1) + p_order
        uL_cell = U_bar[inodeL_cell]
        uR_cell = U_bar[inodeR_cell]
        
        xL_cell = x_coords[inodeL_cell]
        xR_cell = x_coords[inodeR_cell]
        dchidx = 2.0 / (xR_cell - xL_cell) if (xR_cell - xL_cell) != 0 else 0

        for j_local in range(p_order + 1):
            inodej_global = inodeL_cell + j_local
            sumaU_deriv = 0.0
            for l_local in range(p_order + 1):
                inodel_global = inodeL_cell + l_local
                sumaU_deriv += Lp_matrix[j_local][l_local] * U_bar[inodel_global]
            
            gpL_val = gp_array[j_local]
            gpR_val = -gp_array[p_order - j_local] # Como en residual.py
            sumaU_deriv += 0.5 * (uLneigh - uL_cell) * gpL_val + \
                           0.5 * (uRneigh - uR_cell) * gpR_val
            dUb_dx[inodej_global] = sumaU_deriv * dchidx
    
    abs_S_bar = np.abs(dUb_dx)

    # --- 1. Aplicar filtro de prueba a U_bar para obtener U_hat ---
    U_hat = apply_test_filter(U_bar, p_order, x_coords, n_cells, filter_width_ratio)

    # --- 2. Calcular derivadas del campo doblemente filtrado d(U_hat)/dx ---
    # (Reutilizar la lógica anterior para d(U_hat)/dx)
    dUh_dx = np.zeros_like(U_hat)
    for i in range(n_cells):
        inodeLneigh = (i - 1) * (p_order + 1) + p_order
        inodeRneigh = (i + 1) * (p_order + 1)
        if i == 0: inodeLneigh = n_nodes - 1
        if i == n_cells - 1: inodeRneigh = 0
        
        uLneigh = U_hat[inodeLneigh]
        uRneigh = U_hat[inodeRneigh]
        inodeL_cell = i * (p_order + 1)
        inodeR_cell = i * (p_order + 1) + p_order
        uL_cell = U_hat[inodeL_cell]
        uR_cell = U_hat[inodeR_cell]
        
        xL_cell = x_coords[inodeL_cell]
        xR_cell = x_coords[inodeR_cell]
        dchidx = 2.0 / (xR_cell - xL_cell) if (xR_cell - xL_cell) != 0 else 0

        for j_local in range(p_order + 1):
            inodej_global = inodeL_cell + j_local
            sumaU_deriv = 0.0
            for l_local in range(p_order + 1):
                inodel_global = inodeL_cell + l_local
                sumaU_deriv += Lp_matrix[j_local][l_local] * U_hat[inodel_global]
            
            gpL_val = gp_array[j_local]
            gpR_val = -gp_array[p_order - j_local]
            sumaU_deriv += 0.5 * (uLneigh - uL_cell) * gpL_val + \
                           0.5 * (uRneigh - uR_cell) * gpR_val
            dUh_dx[inodej_global] = sumaU_deriv * dchidx
            
    abs_S_hat = np.abs(dUh_dx)

    # --- 3. Calcular anchos de filtro Delta y Delta_hat ---
    Delta_sq_local = np.zeros(n_nodes) # (Delta_i)^2 para cada nodo i
    for i in range(n_cells):
        inodeL_cell = i * (p_order + 1)
        inodeR_cell = i * (p_order + 1) + p_order
        h_e = x_coords[inodeR_cell] - x_coords[inodeL_cell] # Tamaño del elemento
        # Definición común de Delta para elementos de alto orden
        delta_val = h_e / (p_order + 1.0) if p_order > 0 else h_e
        nodes_in_cell = slice(inodeL_cell, inodeR_cell + 1)
        Delta_sq_local[nodes_in_cell] = delta_val**2
        
    Delta_hat_sq_local = (filter_width_ratio**2) * Delta_sq_local

    # --- 4. Calcular L_f = \widehat{(\bar{u}^2/2)} - (\hat{\bar{u}})^2/2 ---
    U_bar_sq_div2 = 0.5 * U_bar**2
    U_bar_sq_div2_filtered = apply_test_filter(U_bar_sq_div2, p_order, x_coords, n_cells, filter_width_ratio)
    U_hat_sq_div2 = 0.5 * U_hat**2
    L_f = U_bar_sq_div2_filtered - U_hat_sq_div2

    # --- 5. Calcular M_f = X_f - \hat{X}_f ---
    # X_f = Delta^2 * |d(U_bar)/dx| * d(U_bar)/dx
    # \hat{X}_f = (Delta_hat)^2 * |d(U_hat)/dx| * d(U_hat)/dx
    X_f = Delta_sq_local * abs_S_bar * dUb_dx # Usa dUb_dx
    X_f_hat = Delta_hat_sq_local * abs_S_hat * dUh_dx # Usa dUh_dx
    M_f = X_f - X_f_hat
    
    # --- 6. Calcular Cd ---
    # Promediado <.>
    if avg_type == 'global':
        numerator = np.sum(L_f * M_f) # Suma en lugar de media para evitar problemas con N
        denominator = np.sum(M_f * M_f)
    else: # Placeholder para promediado local (más complejo)
        print("ADVERTENCIA: Promediado local no implementado, usando global.")
        numerator = np.sum(L_f * M_f)
        denominator = np.sum(M_f * M_f)

    Cd_val = 0.0
    if np.abs(denominator) > 1e-12: # Evitar división por cero
        Cd_val = numerator / denominator
    
    # Clipping de Cd (debe ser positivo)
    # Cs_min típicamente 0.0 - 0.15. Cd_min = Cs_min^2
    Cd_min = cs_min_val**2 
    Cd_val = max(Cd_val, Cd_min)
    
    _last_calculated_Cd_dynamic= Cd_val
    
    return Cd_val # Devuelve un único valor global por ahora

def get_sgs_flux_smagorinsky_dynamic(U_bar, p_order, x_coords, Lp_matrix, gp_array,
                                     Cd_dynamic, # Calculado externamente o aquí mismo
                                     # Si se calcula aquí, los args de arriba son necesarios
                                     filter_width_ratio=2.0, avg_type='global'):
    """
    Calcula el flujo SGS tau_SGS = -nu_SGS * dU_bar/dx con Cd dinámico.
    """
    n_nodes = len(U_bar)
    n_cells = n_nodes // (p_order + 1)

    # --- Calcular d(U_bar)/dx (como antes) ---
    dUb_dx = np.zeros_like(U_bar)
    for i in range(n_cells): # Copiado de arriba, idealmente en una función separada
        inodeLneigh = (i - 1) * (p_order + 1) + p_order
        inodeRneigh = (i + 1) * (p_order + 1)
        if i == 0: inodeLneigh = n_nodes - 1
        if i == n_cells - 1: inodeRneigh = 0
        uLneigh = U_bar[inodeLneigh]
        uRneigh = U_bar[inodeRneigh]
        inodeL_cell = i * (p_order + 1)
        inodeR_cell = i * (p_order + 1) + p_order
        uL_cell = U_bar[inodeL_cell]
        uR_cell = U_bar[inodeR_cell]
        xL_cell = x_coords[inodeL_cell]
        xR_cell = x_coords[inodeR_cell]
        dchidx = 2.0 / (xR_cell - xL_cell) if (xR_cell - xL_cell) != 0 else 0
        for j_local in range(p_order + 1):
            inodej_global = inodeL_cell + j_local
            sumaU_deriv = 0.0
            for l_local in range(p_order + 1):
                inodel_global = inodeL_cell + l_local
                sumaU_deriv += Lp_matrix[j_local][l_local] * U_bar[inodel_global]
            gpL_val = gp_array[j_local]
            gpR_val = -gp_array[p_order - j_local]
            sumaU_deriv += 0.5 * (uLneigh - uL_cell) * gpL_val + \
                           0.5 * (uRneigh - uR_cell) * gpR_val
            dUb_dx[inodej_global] = sumaU_deriv * dchidx
            
    abs_S_bar = np.abs(dUb_dx)

    # --- Calcular Delta^2_local ---
    Delta_sq_local = np.zeros(n_nodes)
    for i in range(n_cells):
        inodeL_cell = i * (p_order + 1)
        inodeR_cell = i * (p_order + 1) + p_order
        h_e = x_coords[inodeR_cell] - x_coords[inodeL_cell]
        delta_val = h_e / (p_order + 1.0) if p_order > 0 else h_e
        nodes_in_cell = slice(inodeL_cell, inodeR_cell + 1)
        Delta_sq_local[nodes_in_cell] = delta_val**2

    # --- Calcular nu_SGS y tau_SGS ---
    nu_SGS_local = Cd_dynamic * Delta_sq_local * abs_S_bar # nu_SGS es un campo ahora
    tau_SGS = -nu_SGS_local * dUb_dx
    
    return tau_SGS


def get_last_calculated_Cd():
    """
    Retorna el último valor de Cd dinámico calculado.
    """
    global _last_calculated_Cd_dynamic # Necesario para leer la variable global del módulo
    return _last_calculated_Cd_dynamic





def calculate_smagorinsky_eddy_viscosity(dudx, dudy, dvdx, dvdy, p, Nx, Ny, Cs):
    """
    Calcula el campo de viscosidad turbulenta (nu_e) usando el modelo estándar de Smagorinsky.
    """
    num_nodes = len(dudx)
    nu_e = np.zeros(num_nodes)

    # 1. Definir el ancho del filtro Delta
    # Usaremos una definición estándar: Delta = (dx*dy)^(1/2)
    num_elements_x = Nx - 1
    num_elements_y = Ny - 1
    dx = 1.0 / num_elements_x
    dy = 1.0 / num_elements_y
    delta_sq = dx * dy # Delta al cuadrado

    # 2. Bucle sobre cada nodo para calcular la magnitud del tensor de deformación |S|
    for i in range(num_nodes):
        # Componentes del tensor de tasa de deformación S_ij
        S11 = dudx[i]
        S22 = dvdy[i]
        S12 = 0.5 * (dudy[i] + dvdx[i])
        S21 = S12
        # El resto de componentes (S33, S13, etc.) son cero en 2D

        # Calcular la magnitud |S| = sqrt(2 * S_ij * S_ij)
        S_mag_sq = 2 * (S11**2 + S22**2 + S12**2 + S21**2)
        S_mag = np.sqrt(S_mag_sq)

        # 3. Calcular la viscosidad turbulenta
        nu_e[i] = (Cs**2) * delta_sq * S_mag
        
    return nu_e


def calculate_vreman_eddy_viscosity(dudx, dudy, dvdx, dvdy, Nx, Ny, c_vreman=0.07):
    """
    Calcula el campo de viscosidad turbulenta (nu_e) usando el modelo de Vreman.
    Este modelo se basa en el paper: "An eddy-viscosity subgrid-scale model for
    turbulent shear flow: Algebraic theory and applications" por A. W. Vreman (2004).

    Args:
        dudx, dudy, dvdx, dvdy (np.array): Derivadas del campo de velocidad.
        Nx, Ny (int): Número de puntos en cada dirección.
        c_vreman (float): Constante del modelo de Vreman. El valor por defecto es 0.07,
                          recomendado en el paper.

    Returns:
        np.array: Campo de viscosidad turbulenta nu_e.
    """
    num_nodes = len(dudx)
    nu_e = np.zeros(num_nodes)

    # 1. Definir el ancho del filtro Delta al cuadrado (Delta^2)
    # Se usa una definición estándar consistente con el modelo de Smagorinsky ya implementado.
    num_elements_x = Nx - 1
    num_elements_y = Ny - 1
    dx = 1.0 / num_elements_x
    dy = 1.0 / num_elements_y
    delta_sq = dx * dy

    # Bucle sobre cada nodo para calcular la viscosidad de Vreman
    for i in range(num_nodes):
        # 2. Tensor de gradiente de velocidad alpha_ij para 2D (el resto de componentes son 0)
        alpha_11 = dudx[i]
        alpha_12 = dvdx[i] # d(v)/dx
        alpha_21 = dudy[i] # d(u)/dy
        alpha_22 = dvdy[i]

        # 3. Denominador del modelo: alpha_ij * alpha_ij (norma de Frobenius al cuadrado)
        # alpha_ij*alpha_ij = sum_{i,j} alpha_ij^2
        norm_alpha_sq = alpha_11**2 + alpha_12**2 + alpha_21**2 + alpha_22**2

        # Si el gradiente es cero, la viscosidad también lo es para evitar división por cero.
        if norm_alpha_sq < 1e-12:
            nu_e[i] = 0.0
            continue

        # 4. Calcular el tensor beta_ij = Delta^2 * (alpha^T * alpha)
        # beta = Delta^2 * | alpha_11^2 + alpha_21^2     alpha_11*alpha_12 + alpha_21*alpha_22 |
        #                | alpha_12*alpha_11 + alpha_22*alpha_21     alpha_12^2 + alpha_22^2 |
        beta_11 = delta_sq * (alpha_11**2 + alpha_21**2)
        beta_12 = delta_sq * (alpha_11*alpha_12 + alpha_21*alpha_22)
        beta_22 = delta_sq * (alpha_12**2 + alpha_22**2)
        # beta_13, beta_23, beta_33, etc., son cero en 2D.

        # 5. Calcular el invariante B_beta
        # Para 2D, B_beta = beta_11*beta_22 - beta_12^2
        B_beta = beta_11 * beta_22 - beta_12**2
        
        # El paper asegura B_beta >= 0. Se añade un clipping por si acaso hay errores numéricos.
        if B_beta < 0:
            B_beta = 0

        # 6. Calcular la viscosidad turbulenta de Vreman
        nu_e[i] = c_vreman * np.sqrt(B_beta / norm_alpha_sq)
        
    return nu_e





def get_sgs_stress_vreman(nu_e, dudx, dudy, dvdx, dvdy):
    """
    Calcula los componentes del tensor de estrés sub-escala (SGS) tau_ij.
    tau_ij = -2 * nu_e * S_ij, donde S_ij es el tensor de tasa de deformación.

    Args:
        nu_e (np.array): Campo de viscosidad turbulenta (del modelo de Vreman).
        dudx, dudy, dvdx, dvdy (np.array): Derivadas del campo de velocidad.

    Returns:
        tuple[np.array, np.array, np.array]: Componentes del tensor de estrés SGS:
                                             (tau_xx, tau_yy, tau_xy).
    """
    # Componentes del tensor de tasa de deformación S_ij
    S11 = dudx
    S22 = dvdy
    S12 = 0.5 * (dudy + dvdx)

    # Componentes del tensor de estrés SGS (tau_ij)
    tau_xx = -2 * nu_e * S11
    tau_yy = -2 * nu_e * S22
    tau_xy = -2 * nu_e * S12 # tau_yx = tau_xy

    return tau_xx, tau_yy, tau_xy














