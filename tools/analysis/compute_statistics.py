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
        # Busca archivos que coincidan con el patrón
        filepaths = sorted([f for f in glob.glob(pattern) if '_FAILED' not in os.path.basename(f)])
        
        if not filepaths:
            print(f"ADVERTENCIA: No se encontraron archivos para el patrón '{pattern}'.")
            continue
            
        found_any_files = True
        print(f"Procesando {len(filepaths)} archivos para el caso: '{pattern}'")
        
        times = []
        total_kes = []
        
        # Extraer parámetros del primer archivo para calcular el tiempo
        try:
            with open(filepaths[0], 'r') as f:
                document = f.readlines()
            dt = float(getValueFromLabel(document, "DT"))
            ndump = int(getValueFromLabel(document, "NDUMP"))
        except (IOError, ValueError, IndexError) as e:
            print(f"  -> Error leyendo parámetros de {filepaths[0]}: {e}. Saltando este patrón.")
            continue
        
        # Iterar sobre todos los archivos de salida
        for i, filepath in enumerate(filepaths):
            time = (i + 1) * ndump * dt
            ke = analyze_snapshot(filepath)
            
            if ke is not None:
                times.append(time)
                total_kes.append(ke)

        # --- MÉTODO DE ETIQUETADO MEJORADO ---
        # Si solo hay un archivo, usa su nombre base.
        if len(filepaths) == 1:
            case_label = os.path.basename(filepaths[0]).replace('.txt', '')
        # Si hay varios, encuentra el prefijo común entre ellos para la etiqueta.
        else:
            # Encuentra el prefijo común de los nombres de archivo
            common_prefix = os.path.basename(os.path.commonprefix([os.path.basename(f) for f in filepaths]))
            # Limpia la etiqueta eliminando caracteres extra al final
            case_label = common_prefix.rstrip('._-')
        
        if not case_label: # Fallback por si no encuentra un prefijo
            case_label = pattern

        plt.plot(times, total_kes, 'o-', label=case_label, markerfacecolor='None', markersize=6)

    # Configuración final del gráfico
    if found_any_files:
        plt.title('Evolución Temporal de la Energía Cinética Total')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Energía Cinética Total (Promedio)')
        plt.grid(True, which="both", ls="--")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left') # Mueve la leyenda fuera del gráfico
        plt.yscale('log')
        
        output_filename = 'statistics_comparison.png'
        # Ajusta el layout para que la leyenda no se corte
        plt.tight_layout(rect=[0, 0, 0.85, 1])
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