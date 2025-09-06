# tools/analysis/analizar_convergencia.py
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd # Importamos pandas para un manejo de datos más sencillo

# --- Añadir la ruta al directorio raíz para importar utilidades ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def get_solution_from_file(filepath):
    """
    Lee los datos de un archivo de solución, promedia los valores en las interfaces
    de los elementos para eliminar duplicados en 'x', y devuelve los datos limpios.
    Devuelve None si el archivo contiene NaN o no se puede procesar.
    """
    try:
        with open(filepath, 'r') as f:
            document = f.readlines()

        p_order = int(getValueFromLabel(document, "P"))
        n_elements = int(getValueFromLabel(document, "N"))
        
        data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
        if not data_lines:
            print(f"  -> Advertencia: No se encontró bloque de solución en {os.path.basename(filepath)}.")
            return None, None, None

        data = np.loadtxt(data_lines)
        
        # Comprobar si hay valores inválidos (NaN o Inf) ANTES de procesar
        if not np.all(np.isfinite(data[:, 1])):
            print(f"  -> Advertencia: Se encontraron valores NaN/Inf en {os.path.basename(filepath)}. Se omitirá.")
            return None, None, None

        # --- CORRECCIÓN CLAVE: Usar pandas para promediar duplicados ---
        # Creamos un DataFrame para manejar los datos fácilmente
        df = pd.DataFrame({'x': data[:, 0], 'u': data[:, 1]})
        
        # Agrupamos por la coordenada 'x' y calculamos la media de 'u' para cada 'x' único
        solution_averaged = df.groupby('x')['u'].mean().reset_index()
        
        # Extraemos los datos limpios y ordenados
        x_unique = solution_averaged['x'].values
        u_averaged = solution_averaged['u'].values
            
        return p_order, n_elements, (x_unique, u_averaged)
    except Exception as e:
        print(f"  -> Error procesando {os.path.basename(filepath)}: {e}")
        return None, None, None

def calculate_l2_error(coarse_solution, fine_solution):
    """
    Calcula el error en norma L2 entre una solución 'coarse' y una 'fine'.
    La solución 'fine' se interpola a los puntos de la malla 'coarse' para la comparación.
    """
    x_coarse, u_coarse = coarse_solution
    x_fine, u_fine = fine_solution
    
    # Crear una función de interpolación a partir de la solución fina (que ya no tiene duplicados)
    interp_func = interp1d(x_fine, u_fine, kind='cubic', fill_value="extrapolate")
    
    # Evaluar la solución fina interpolada en los puntos de la malla gruesa
    u_fine_interp = interp_func(x_coarse)
    
    # Calcular el error en norma L2
    error_l2 = np.sqrt(np.sum((u_coarse - u_fine_interp)**2) / len(u_coarse))
    
    return error_l2

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        print("\nUso: python analizar_convergencia.py \"ruta/a/los/resultados_de_convergencia/*.txt\"")
        print("\nEjemplo desde la raíz del proyecto:")
        print("python tools/analysis/analizar_convergencia.py \"data/outputs/convergence_study_1d/*.txt\"")
        print("\nIMPORTANTE: ¡Usa comillas dobles alrededor de la ruta!")
        sys.exit(1)

    # --- 1. Recopilar y agrupar todos los datos válidos ---
    filepaths = glob.glob(argv[1])
    if not filepaths:
        print(f"Error: No se encontraron archivos para el patrón '{argv[1]}'")
        sys.exit(1)
        
    print(f"Procesando {len(filepaths)} archivos del estudio de convergencia...")
    
    results_by_p = {}
    for f in sorted(filepaths): # Ordenar para un procesamiento predecible
        p, n, solution_data = get_solution_from_file(f)
        if p is not None:
            if p not in results_by_p:
                results_by_p[p] = []
            results_by_p[p].append({'n': n, 'data': solution_data})

    # --- 2. Calcular errores para cada grado polinomial ---
    convergence_data = {}
    for p, results in results_by_p.items():
        sorted_results = sorted(results, key=lambda r: r['n'])
        
        if len(sorted_results) < 2:
            print(f"  -> Advertencia: Se necesita al menos 2 mallas para calcular el error para p={p}. Se omite.")
            continue
        
        convergence_data[p] = {'dof': [], 'error': []}
        
        fine_solution = sorted_results[-1]['data']

        for i in range(len(sorted_results) - 1):
            coarse_res = sorted_results[i]
            # El número de grados de libertad (dof) es el número de puntos únicos
            dof = len(coarse_res['data'][0])
            error = calculate_l2_error(coarse_res['data'], fine_solution)
            
            convergence_data[p]['dof'].append(dof)
            convergence_data[p]['error'].append(error)

    # --- 3. Generar la gráfica ---
    fig, ax = plt.subplots(figsize=(14, 9))
    
    slope_diff_percentages = []

    for p, data in sorted(convergence_data.items()):
        if not data['dof']: continue
        
        dof = np.array(data['dof'], dtype = float)
        error = np.array(data['error'])
        
        # --- Cálculo de la pendiente observada ---
        # Usamos un ajuste lineal en el espacio logarítmico (polyfit de grado 1)
        log_dof = np.log(dof)
        log_error = np.log(error)
        observed_slope, _ = np.polyfit(log_dof, log_error, 1)
        
        theoretical_slope = -(p + 1)
        
        # Calcular y almacenar la diferencia porcentual
        diff = 100 * abs((observed_slope - theoretical_slope) / theoretical_slope)
        slope_diff_percentages.append(diff)
        print(f"  -> Para p={p}: Pendiente teórica={theoretical_slope:.2f}, Observada={observed_slope:.2f} (Diferencia: {diff:.2f}%)")

        # Graficar los puntos de error calculados
        plot = ax.loglog(dof, error, 'o-', label=f'Error calculado (p={p})', markersize=8, linewidth=2)
        
        # Graficar la línea de tendencia teórica
        color = plot[0].get_color()
        C = error[0] / (dof[0]**(-(p + 1)))
        trend_dof = np.array([dof[0], dof[-1]], dtype=float)
        trend_error = C * (trend_dof**(-(p + 1)))
        ax.loglog(trend_dof, trend_error, '--', color=color, label=f'Pendiente teórica O(h$^{{{p+1}}}$)')

    # --- Cálculo de la estadística final y adición del subtítulo ---
    if slope_diff_percentages:
        avg_diff = np.mean(slope_diff_percentages)
        fig.suptitle(f'Desviación media de la pendiente teórica: {avg_diff:.2f}%', fontsize=14, y=0.92)
    
    ax.set_title('Análisis de Convergencia del Solver FR', fontsize=16, pad=30)
    ax.set_xlabel('Grados de Libertad (dof)', fontsize=12)
    ax.set_ylabel('Error en Norma L2', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, which="both", ls="--")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.show()