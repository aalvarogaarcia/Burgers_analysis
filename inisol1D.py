import numpy as np
import os

# Asegúrate de que los imports funcionen ajustando la ruta si es necesario
from src.core.mesh import   get_mesh_1d, get_mesh_ho_1d
from src.utils.misc import FillInitialSolution_1D
from src.utils.randw import WriteFile_1D
from src.core.lagpol import getStandardElementData

# --- Parámetros para la solución inicial ---
# Estos valores son necesarios para la cabecera del archivo de salida.
params = {
    'N': 80,         # Número de celdas
    'P': 3,           # Orden del polinomio (no afecta a la solución inicial en sí)
    'VISC': 0.0,      # Viscosidad
    'Nref': 0,        # Niveles de refinamiento
    'INISOL': 'SINE', # Condición inicial
    'DT': 0.0005,
    'TSIM': 0.0,      # Tiempo de simulación 0 para no procesar nada
    'NDUMP': 1,
    'SCHEME': 'FR'
}

# --- Generación de la solución ---

# 1. Crear la malla 1D
x_base = get_mesh_1d(params['N'], params['Nref'])

# 2. Obtener los datos del elemento estándar para FR (puntos de Lobatto)
lobatto_points, _, _ = getStandardElementData(params['P'])

# 3. Crear la malla final de alto orden (con los puntos de Lobatto)
# ESTE ES EL PASO CLAVE QUE FALTABA
x_high_order_mesh = get_mesh_ho_1d(x_base, lobatto_points)

# 2. Inicializar el vector de solución
U_initial = np.zeros(len(x_high_order_mesh))

# 3. Rellenar con la condición inicial 'SINE'
FillInitialSolution_1D(U_initial, x_high_order_mesh, params['INISOL'], params['N'], params['P'], params['Nref'])

# 4. Escribir la solución a un archivo .txt
output_filename = "data/inputs/convergence_study_1d/SolucionInicial.txt"
WriteFile_1D(
    output_filename,
    x_high_order_mesh,
    U_initial,
    params['N'],
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