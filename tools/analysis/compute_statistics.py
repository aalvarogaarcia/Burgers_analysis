# tools/analysis/compute_statistics.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
from collections import defaultdict
import re

# Añade la ruta al directorio raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def analyze_snapshot(filepath):
    """
    Analiza un único archivo de snapshot y devuelve la energía cinética total.
    Devuelve None si los datos son inválidos o la energía es infinita.
    """
    try:
        with open(filepath, 'r') as f:
            document = f.readlines()
            
        data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
        if not data_lines: return None
        
        data = np.loadtxt(data_lines)
        u, v = data[:, 2], data[:, 3]
        
        valid_mask = np.isfinite(u) & np.isfinite(v)
        if np.sum(valid_mask) == 0: return None

        # Calcular KE usando un tipo de dato que maneje números grandes
        ke_per_node = 0.5 * (u[valid_mask].astype(np.float64)**2 + v[valid_mask].astype(np.float64)**2)
        total_ke = np.mean(ke_per_node)
        
        # Si el resultado final es infinito o NaN, la simulación ha explotado. Devolver None.
        if not np.isfinite(total_ke):
            print(f"  -> Advertencia: Inestabilidad detectada (KE infinita) en {os.path.basename(filepath)}. Snapshot omitido.")
            return None
        
        return total_ke
        
    except Exception as e:
        print(f"  -> Error procesando {filepath}: {e}")
        return None

def extract_label_from_pattern(pattern):
    """
    Extrae una etiqueta descriptiva del patrón del fichero, incluyendo el coeficiente.
    """
    base_name = os.path.basename(pattern).replace('.txt','').replace('*','')
    
    smag_match = re.search(r'Smagorinsky_Cs([\d.]+)', base_name, re.IGNORECASE)
    vrem_match = re.search(r'Vreman_Cv([\d.]+)', base_name, re.IGNORECASE)
    
    if smag_match:
        return f"Smagorinsky (Cs={smag_match.group(1)})"
    if vrem_match:
        return f"Vreman (Cv={vrem_match.group(1)})"
    if 'ILES' in base_name.upper():
        return 'ILES'
        
    return base_name if base_name else "Serie"

def main(case_patterns):
    """
    Procesa múltiples series de snapshots, las agrupa por patrón y las grafica
    en una única figura comparativa.
    """
    if not case_patterns:
        print("No se proporcionaron patrones de archivos.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    
    for pattern in case_patterns:
        filepaths = sorted([f for f in glob.glob(pattern) if '_FAILED' not in os.path.basename(f)])
        
        if not filepaths:
            print(f"ADVERTENCIA: No se encontraron archivos para el patrón '{pattern}'.")
            continue
            
        model_label = extract_label_from_pattern(pattern)
        print(f"Procesando {len(filepaths)} archivos para el modelo: '{model_label}'")
        
        # Extraer parámetros de tiempo del primer snapshot
        with open(filepaths[0], 'r') as f:
            document = f.readlines()
        dt = float(getValueFromLabel(document, "DT"))
        ndump = int(getValueFromLabel(document, "NDUMP"))
        
        times, kes = [], []
        for i, filepath in enumerate(filepaths):
            time = (i + 1) * ndump * dt
            ke = analyze_snapshot(filepath)
            
            # Solo añadir si el valor es válido
            if ke is not None:
                times.append(time)
                kes.append(ke)
        
        if times:
            # --- CORRECCIÓN DE ROBUSTEZ FINAL ---
            # Filtrar los datos para asegurarse de que son válidos para un plot logarítmico
            valid_indices = [i for i, ke_val in enumerate(kes) if np.isfinite(ke_val) and ke_val > 0]
            if not valid_indices:
                print(f"  -> No hay datos válidos para graficar para el modelo '{model_label}'.")
                continue

            valid_times = [times[i] for i in valid_indices]
            valid_kes = [kes[i] for i in valid_indices]

            ax.plot(valid_times, valid_kes, 'o-', label=model_label, markerfacecolor='white', markersize=6)

    ax.set_title('Evolución Temporal de la Energía Cinética Total', fontsize=16)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Energía Cinética Total')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_filename = 'statistics_comparison.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n¡Éxito! Gráfico comparativo guardado en: {os.path.abspath(output_filename)}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python compute_statistics.py \"ruta/caso1/*.txt\" \"ruta/caso2/*.txt\" ...")
        sys.exit(1)
    
    main(sys.argv[1:])