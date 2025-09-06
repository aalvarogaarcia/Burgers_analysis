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
    """Analiza un único archivo de snapshot y devuelve la energía cinética total."""
    try:
        with open(filepath, 'r') as f:
            document = f.readlines()
            
        data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
        if not data_lines:
            return None
        
        data = np.loadtxt(data_lines)
        u, v = data[:, 2], data[:, 3]
        
        valid_mask = np.isfinite(u) & np.isfinite(v)
        if np.sum(valid_mask) == 0:
            return None

        total_ke = 0.5 * np.mean(u[valid_mask]**2 + v[valid_mask]**2)
        return total_ke
        
    except Exception as e:
        print(f"Error procesando {filepath}: {e}")
        return None

def extract_label_from_pattern(pattern):
    """Extrae una etiqueta descriptiva del patrón del fichero."""
    # Busca palabras clave conocidas en el patrón
    if 'ILES' in pattern.upper(): return 'ILES'
    if 'SMAGORINSKY' in pattern.upper(): return 'Smagorinsky'
    if 'VREMAN' in pattern.upper(): return 'Vreman'
    
    # Si no encuentra, usa un nombre genérico basado en el fichero
    try:
        base_name = os.path.basename(pattern).replace('.txt','').replace('*','')
        return base_name if base_name else "Serie"
    except:
        return "Serie"

def main(case_patterns):
    """
    Procesa múltiples series de snapshots, las agrupa por patrón y las grafica
    en una única figura comparativa.
    """
    if not case_patterns:
        print("No se proporcionaron patrones de archivos.")
        return

    # --- 1. Iniciar la figura de Matplotlib ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- 2. Recopilar y procesar datos para cada patrón ---
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
            
            if ke is not None:
                times.append(time)
                kes.append(ke)
        
        # --- 3. Graficar la serie de datos si se encontraron resultados ---
        if times:
            ax.plot(times, kes, 'o-', label=model_label, markerfacecolor='white', markersize=6)

    # --- 4. Finalizar y mostrar el gráfico ---
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
        print("\nEjemplo desde la raíz del proyecto:")
        print("python tools/analysis/compute_statistics.py \"data/outputs/dc_decay_comparison/*ILES*.txt\" \"data/outputs/dc_decay_comparison/*Smagorinsky*.txt\"")
        print("\nIMPORTANTE: ¡Usa comillas dobles alrededor de cada ruta!")
        sys.exit(1)
    
    main(sys.argv[1:])