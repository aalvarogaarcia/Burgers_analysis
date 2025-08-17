# tools/analysis/compute_statistics.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

# Añade la ruta al directorio raíz del proyecto para poder importar desde 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def analyze_snapshot(filepath):
    """Analiza un único archivo y devuelve la energía cinética total."""
    try:
        with open(filepath, 'r') as f:
            document = f.readlines()
            
        data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
        if not data_lines:
            print(f"  -> Advertencia: No se encontró bloque de solución en {os.path.basename(filepath)}")
            return None
        
        data = np.loadtxt(data_lines)
        u, v = data[:, 2], data[:, 3]
        
        # Ignorar valores no finitos que provienen de inestabilidades
        valid_mask = np.isfinite(u) & np.isfinite(v)
        if np.sum(valid_mask) == 0:
            print(f"  -> Advertencia: Todos los datos en {os.path.basename(filepath)} son inválidos (NaN/inf).")
            return None

        # La energía cinética total es el promedio de 0.5 * (u^2 + v^2)
        total_ke = 0.5 * np.mean(u[valid_mask]**2 + v[valid_mask]**2)
        return total_ke
    except Exception as e:
        print(f"  -> Error procesando {os.path.basename(filepath)}: {e}")
        return None

def main(case_patterns):
    """
    Procesa múltiples patrones de casos (directorios) y los grafica juntos
    para comparar la evolución de la energía cinética.
    """
    plt.figure(figsize=(12, 8))
    found_any_files = False
    
    for pattern in case_patterns:
        # Busca archivos que coincidan con el patrón (ej. 'data/outputs/vreman/*.txt')
        # Excluye los archivos _FAILED.txt
        filepaths = sorted([f for f in glob.glob(pattern) if '_FAILED' not in os.path.basename(f)])
        
        # --- MEJORA: AÑADIR MENSAJES DE ESTADO ---
        if not filepaths:
            print(f"ADVERTENCIA: No se encontraron archivos para el patrón '{pattern}'.")
            print("Verifica la ruta y asegúrate de que los archivos de resultados existen.")
            continue
            
        found_any_files = True
        print(f"Procesando {len(filepaths)} archivos para el caso: '{pattern}'")
        
        times = []
        total_kes = []
        
        # Extraer parámetros del primer archivo para calcular el tiempo
        with open(filepaths[0], 'r') as f:
            document = f.readlines()
        dt = float(getValueFromLabel(document, "DT"))
        ndump = int(getValueFromLabel(document, "NDUMP"))
        
        # Iterar sobre todos los archivos de salida de una simulación
        for i, filepath in enumerate(filepaths):
            # El tiempo de cada snapshot se calcula a partir del intervalo de guardado
            # Asumimos que el primer guardado ocurre en la iteración NDUMP
            time = (i + 1) * ndump * dt
            ke = analyze_snapshot(filepath)
            
            if ke is not None:
                times.append(time)
                total_kes.append(ke)

        # Usar el nombre del directorio como etiqueta para el gráfico
        case_label = filepath.split('/')[-1]
        if not case_label: # Si el patrón es local (ej. *.txt)
            case_label = os.path.basename(pattern).replace('*.txt', '')
        plt.plot(times, total_kes, 'o-', label=case_label, markerfacecolor='None')

    # Solo mostrar y guardar el gráfico si se procesó al menos un archivo
    if found_any_files:
        plt.title('Evolución Temporal de la Energía Cinética Total')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Energía Cinética Total (Promedio)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.yscale('log') # La escala logarítmica es ideal para ver el decaimiento
        
        output_filename = 'statistics_comparison.png'
        plt.savefig(output_filename, dpi=150)
        print(f"\n¡Éxito! Gráfico comparativo guardado en: {os.path.abspath(output_filename)}")
        plt.show()
    else:
        print("\nNo se procesó ningún archivo. No se ha generado ningún gráfico.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python compute_statistics.py \"ruta/caso1/*.txt\" \"ruta/caso2/*.txt\" ...")
        print("\nEjemplo desde la carpeta raíz del proyecto:")
        print("python tools/analysis/compute_statistics.py \"data/outputs/vreman/*.txt\" \"data/outputs/smagorinsky/*.txt\" \"data/outputs/no_les/*.txt\"")
        print("\nIMPORTANTE: ¡Usa comillas dobles alrededor de cada ruta!")
        sys.exit(1)
    main(sys.argv[1:])