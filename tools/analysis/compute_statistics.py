# tools/analysis/compute_statistics.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
from collections import defaultdict

# Añade la ruta al directorio raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def analyze_snapshot(filepath):
    """Analiza un único archivo de snapshot y devuelve la energía cinética total."""
    try:
        with open(filepath, 'r') as f:
            document = f.readlines()
            
        data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
        if not data_lines:
            return None, None
        
        data = np.loadtxt(data_lines)
        u, v = data[:, 2], data[:, 3]
        
        valid_mask = np.isfinite(u) & np.isfinite(v)
        if np.sum(valid_mask) == 0:
            return None, None

        total_ke = 0.5 * np.mean(u[valid_mask]**2 + v[valid_mask]**2)
        
        # Extraer parámetros para el etiquetado
        scheme = getValueFromLabel(document, "SCHEME")
        p_order = getValueFromLabel(document, "P")
        visc = float(getValueFromLabel(document, "VISC"))
        
        label = f"{scheme.upper()}-P{p_order} Visc={visc}"
        
        return total_ke, label
        
    except Exception:
        return None, None

def main(case_patterns):
    """
    Procesa múltiples directorios de casos, los agrupa por modelo y los grafica
    en subplots separados para una comparación de estilo académico.
    """
    all_results = defaultdict(list)
    
    # --- 1. Recopilar y agrupar todos los datos ---
    for pattern in case_patterns:
        filepaths = sorted([f for f in glob.glob(pattern) if '_FAILED' not in os.path.basename(f)])
        if not filepaths:
            print(f"ADVERTENCIA: No se encontraron archivos para el patrón '{pattern}'.")
            continue
            
        print(f"Procesando {len(filepaths)} archivos para el caso: '{pattern}'")
        
        # Extraer el nombre base del modelo del patrón (ej. 'vreman', 'smagorinsky')
        model_name = os.path.basename(os.path.dirname(pattern)).capitalize()
        if not model_name: # Fallback si el patrón es local
            model_name = "Resultados"

        with open(filepaths[0], 'r') as f:
            document = f.readlines()
        dt = float(getValueFromLabel(document, "DT"))
        ndump = int(getValueFromLabel(document, "NDUMP"))
        
        case_series = defaultdict(lambda: {'times': [], 'kes': []})

        for i, filepath in enumerate(filepaths):
            time = (i + 1) * ndump * dt
            ke, label = analyze_snapshot(filepath)
            
            if ke is not None and label is not None:
                case_series[label]['times'].append(time)
                case_series[label]['kes'].append(ke)
        
        for label, data in case_series.items():
            all_results[model_name].append({'label': label, 'data': data})

    if not all_results:
        print("\nNo se procesó ningún archivo. No se ha generado ningún gráfico.")
        return

    # --- 2. Crear los subplots ---
    num_models = len(all_results)
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 6 * num_models), sharex=True, squeeze=False)
    axes = axes.flatten() # Asegurarse de que axes es un array 1D

    fig.suptitle('Evolución Temporal de la Energía Cinética por Modelo', fontsize=16)

    model_list = sorted(all_results.keys())

    for i, model_name in enumerate(model_list):
        ax = axes[i]
        for result in all_results[model_name]:
            ax.plot(result['data']['times'], result['data']['kes'], 'o-', 
                    label=result['label'], markerfacecolor='white', markersize=6)
        
        ax.set_title(f"Modelo: {model_name}", fontsize=12)
        ax.set_ylabel('Energía Cinética Total')
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend()
        ax.set_yscale('log')

    axes[-1].set_xlabel('Tiempo (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = 'statistics_comparison.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n¡Éxito! Gráfico comparativo guardado en: {os.path.abspath(output_filename)}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python compute_statistics.py \"ruta/caso1/*.txt\" \"ruta/caso2/*.txt\" ...")
        print("\nEjemplo desde la raíz del proyecto:")
        print("python tools/analysis/compute_statistics.py \"data/outputs/vreman/*.txt\" \"data/outputs/smagorinsky/*.txt\"")
        print("\nIMPORTANTE: ¡Usa comillas dobles alrededor de cada ruta!")
        sys.exit(1)
    main(sys.argv[1:])
