import numpy as np
import os

# Asegúrate de que los imports funcionen ajustando la ruta si es necesario
from src.core.mesh import   get_2d_cartesian_mesh
from src.utils.misc import FillInitialSolution_2D
from src.utils.randw import WriteFile_2D

# --- Parámetros para la solución inicial ---
# Estos valores son necesarios para la cabecera del archivo de salida.
params = {
    'NX': 65,         # Número de celdas
    'NY': 65,
    'P': 3,           # Orden del polinomio (no afecta a la solución inicial en sí)
    'VISC': 0.0,      # Viscosidad
    'Nref': 0,        # Niveles de refinamiento
    'INISOL': 'TAYLOR_GREEN', # Condición inicial
    'DT': 0.0005,
    'TSIM': 0.0,      # Tiempo de simulación 0 para no procesar nada
    'NDUMP': 1,
    'SCHEME': 'DC'
}

# --- Generación de la solución ---

# 1. Crear la malla 2D
Nx, Ny = params['NX'], params['NY']
if params['INISOL'] == 'GAUSSIAN_2D':
    x_coords, y_coords = get_2d_cartesian_mesh(Nx, Ny)

else:
    x_coords, y_coords = get_2d_cartesian_mesh(Nx, Ny, Lx=(2*np.pi), Ly=(2*np.pi))
        
xx, yy = np.meshgrid(x_coords, y_coords)
x_base = xx.flatten()
y_base = yy.flatten()
        
num_nodes = Nx * Ny
U_initial = np.zeros(2 * num_nodes)

# 2. Inicializar el vector de solución


# 3. Rellenar con la condición inicial 'SINE'
FillInitialSolution_2D(U_initial, x_base, y_base, params['INISOL'], params['NX'], params['NY'], params['P'], params['Nref'])

# 4. Escribir la solución a un archivo .txt
output_filename = "SolucionInicial.txt"
WriteFile_2D(
    output_filename,
    x_base,
    y_base,
    U_initial,
    params['NX'],
    params['NY'],
    params['P'],
    params['VISC'],
    params['Nref'],
    params['INISOL'],
    params['DT'],
    params['TSIM'],
    params['NDUMP'],
    params['SCHEME']
)

print(f"Archivo '{output_filename}' generado con éxito.")
print("Contiene la solución inicial 'SINE' sin ningún paso de simulación.")