# tools/analysis/plot_convergence.py
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

# --- Añadir la ruta al directorio raíz para importar utilidades ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def get_solution_from_file(filepath):
    """
    Lee los datos de un archivo de solución y los devuelve limpios,
    promediando duplicados en 'x' y comprobando si hay NaNs.
    """
    try:
        with open(filepath, 'r') as f:
            document = f.readlines()
        p_order = int(getValueFromLabel(document, "P"))
        n_elements = int(getValueFromLabel(document, "N"))
        data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
        if not data_lines: return None, None, None
        data = np.loadtxt(data_lines)
        if not np.all(np.isfinite(data[:, 1])):
            print(f"  -> Advertencia: Se encontraron valores NaN/Inf en {os.path.basename(filepath)}. Se omitirá.")
            return None, None, None
        df = pd.DataFrame({'x': data[:, 0], 'u': data[:, 1]})
        solution_averaged = df.groupby('x')['u'].mean().reset_index()
        x_unique = solution_averaged['x'].values
        u_averaged = solution_averaged['u'].values
        return p_order, n_elements, (x_unique, u_averaged)
    except Exception as e:
        print(f"  -> Error procesando {os.path.basename(filepath)}: {e}")
        return None, None, None

def calculate_l2_error(coarse_solution, fine_solution):
    """
    Calcula el error L2 interpolando la solución 'coarse' a los puntos de la 'fine'.
    """
    x_coarse, u_coarse = coarse_solution
    x_fine, u_fine = fine_solution
    interp_func = interp1d(x_coarse, u_coarse, kind='cubic', fill_value="extrapolate")
    u_coarse_interp = interp_func(x_fine)
    error_l2 = np.sqrt(np.mean((u_coarse_interp - u_fine)**2))
    return error_l2

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nUso: python tools/analysis/plot_convergence.py \"ruta/a/resultados/*.txt\"")
        sys.exit(1)

    filepaths = glob.glob(sys.argv[1])
    if not filepaths:
        print(f"Error: No se encontraron archivos para el patrón '{sys.argv[1]}'")
        sys.exit(1)
        
    print(f"Procesando {len(filepaths)} archivos del estudio de convergencia...")
    
    results_by_p = {}
    for f in sorted(filepaths):
        p, n, solution_data = get_solution_from_file(f)
        if p is not None:
            if p not in results_by_p: results_by_p[p] = []
            results_by_p[p].append({'n': n, 'data': solution_data})

    convergence_data = {}
    for p, results in results_by_p.items():
        sorted_results = sorted(results, key=lambda r: r['n'])
        if len(sorted_results) < 2: continue
        
        convergence_data[p] = {'dof_inv': [], 'error': []}
        fine_solution = sorted_results[-1]['data']

        for coarse_res in sorted_results[:-1]:
            dof = len(coarse_res['data'][0])
            error = calculate_l2_error(coarse_res['data'], fine_solution)
            convergence_data[p]['dof_inv'].append(1.0 / dof)
            convergence_data[p]['error'].append(error)

    # --- Generar la gráfica ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for p, data in sorted(convergence_data.items()):
        if not data['dof_inv']: continue
        
        dof_inv = np.array(data['dof_inv'])
        error = np.array(data['error'])
        
        plot = ax.loglog(dof_inv, error, 'o-', label=f'P={p}', markersize=8, linewidth=2)
        
        # Graficar la línea de tendencia teórica O(h^(p+1))
        color = plot[0].get_color()
        # La pendiente en un plot log-log de error vs 1/dof es (p+1)
        C = error[0] / (dof_inv[0]**(p + 1)) 
        trend_dof_inv = np.array([dof_inv[0], dof_inv[-1]])
        trend_error = C * (trend_dof_inv**(p + 1))
        ax.loglog(trend_dof_inv, trend_error, '--', color=color, label=f'~dof$^{{-({p+1})}}$')

    ax.set_title('Análisis de Convergencia del Solver FR', fontsize=16)
    ax.set_xlabel('1 / Grados de Libertad (dof)', fontsize=12)
    ax.set_ylabel('Error en Norma L2', fontsize=12)
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()