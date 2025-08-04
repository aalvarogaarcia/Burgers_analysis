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


def calculate_vreman_eddy_viscosity(dudx, dudy, dvdx, dvdy, p, Nx, Ny, x_ho, y_ho, c_vreman):
    """
    Calcula el campo de viscosidad turbulenta (nu_e) usando el modelo de Vreman.
    """
    num_nodes = len(dudx)
    nu_e = np.zeros(num_nodes)
    
    # 1. Definir el ancho del filtro Delta
    # Usaremos una definición común: Delta = sqrt(dx*dy) / (p+1)
    num_elements_x = Nx - 1
    num_elements_y = Ny - 1
    dx = 1.0 / num_elements_x
    dy = 1.0 / num_elements_y
    delta_sq = (dx * dy) / ((p + 1)**2) # Delta al cuadrado

    # 2. Bucle sobre cada nodo para calcular los tensores
    for i in range(num_nodes):
        # Ensamblar el tensor de gradiente de velocidad alpha (en 2D)
        alpha = np.array([
            [dudx[i], dudy[i], 0],
            [dvdx[i], dvdy[i], 0],
            [0,       0,       0]
        ])
        
        # Calcular el denominador: alpha_ij * alpha_ij (norma de Frobenius al cuadrado)
        alpha_ij_sq = np.sum(alpha**2)
        
        if alpha_ij_sq < 1e-12: # Evitar división por cero
            continue

        # Calcular el tensor beta = delta^2 * alpha^T * alpha
        beta = delta_sq * (alpha.T @ alpha)
        
        # Calcular B_beta (un invariante de la matriz beta)
        b11, b12, b13 = beta[0,0], beta[0,1], beta[0,2]
        b22, b23 = beta[1,1], beta[1,2]
        b33 = beta[2,2]
        
        B_beta = b11*b22 - b12**2 + b11*b33 - b13**2 + b22*b33 - b23**2
        
        # Salvaguarda numérica como se sugiere en el paper 
        if B_beta < 1e-8:
             nu_e[i] = 0
             continue

        # 3. Calcular la viscosidad turbulenta
        nu_e[i] = c_vreman * np.sqrt(B_beta / alpha_ij_sq)
        
        return nu_e