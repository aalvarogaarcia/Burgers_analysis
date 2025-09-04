# src/core/forcing.py
import numpy as np

def generate_deterministic_forcing(coords, nx, ny, amplitude, k_max=4):
    """
    Genera un campo de forzante 2D determinista y estacionario.
    El forzante inyecta energía en un rango de números de onda bajos.

    Args:
        coords (tuple): Tupla (x, y) con las coordenadas de los nodos.
        nx (int): Número de puntos de la malla base en la dirección x.
        ny (int): Número de puntos de la malla base en la dirección y.
        amplitude (float): Amplitud (fuerza) del forzante.
        k_max (int): Número de onda máximo a forzar. La energía se inyectará
                     en los modos con |kx| <= k_max y |ky| <= k_max.

    Returns:
        np.array: Un vector que contiene los componentes (fx, fy) del forzante
                  concatenados, con el mismo formato que el vector de solución U.
    """
    x, y = coords
    num_nodes = len(x)
    
    fx = np.zeros(num_nodes)
    fy = np.zeros(num_nodes)
    
    # El dominio es [0, 1] x [0, 1], por lo que L = 1.
    # Los números de onda son k = 2 * pi * n / L
    wavenumbers = np.arange(1, k_max + 1)
    
    num_modes = 0
    for kx_mode in wavenumbers:
        for ky_mode in wavenumbers:
            kx = 2 * np.pi * kx_mode
            ky = 2 * np.pi * ky_mode
            
            # Añadir modos con fases aleatorias para romper la simetría
            # Se usan fases diferentes para fx y fy para crear un campo no trivial
            phase_x = np.random.uniform(0, 2 * np.pi)
            phase_y = np.random.uniform(0, 2 * np.pi)

            fx += np.cos(kx * x + ky * y + phase_x)
            fy += np.sin(kx * x + ky * y + phase_y) # Usar sin para fy
            num_modes += 1
            
    if num_modes > 0:
        # Normalizar para que la amplitud no dependa del número de modos
        fx /= np.sqrt(np.mean(fx**2 + fy**2))
        fy /= np.sqrt(np.mean(fx**2 + fy**2))
        
        fx *= amplitude
        fy *= amplitude

    # Devolver el forzante concatenado
    return np.concatenate((fx, fy))
