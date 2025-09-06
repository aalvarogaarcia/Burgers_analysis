# tools/generate_cases/generate_fr_smagorinsky_comparison.py
import os
import itertools

def write_config_file(config, filename, subdirectory):
    """
    Escribe un diccionario de configuración a un archivo .txt en una subcarpeta específica.
    Esta función es genérica y maneja tanto casos ILES como LES.
    """
    base_directory = "data/inputs"
    target_directory = os.path.join(base_directory, subdirectory)

    # Crear el directorio de salida si no existe
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Directorio creado: {target_directory}")

    filepath = os.path.join(target_directory, filename)
    
    with open(filepath, 'w') as f:
        # Escribir la cabecera del fichero
        f.write(f"# Input file for 1D FR vs Smagorinsky comparison\n")
        
        # Escribir los parámetros de forma alineada
        f.write(f"{'N':<20}{config['N']}\n")
        f.write(f"{'P':<20}{config['P']}\n")
        f.write(f"{'SCHEME':<20}{config['SCHEME']}\n")
        f.write(f"{'VISC':<20}{config['VISC']:.8f}\n")
        f.write(f"{'INISOL':<20}{config['INISOL']}\n")
        f.write(f"{'NREF':<20}{config.get('NREF', 0)}\n")
        f.write(f"{'DT':<20}{config['DT']:.8f}\n")
        f.write(f"{'TSIM':<20}{config['TSIM']:.8f}\n")
        f.write(f"{'NDUMP':<20}{config['NDUMP']}\n")
        
        # --- Escribir sección de LES ---
        f.write("# --- LES Parameters ---\n")
        use_les = config.get('USE_LES', False)
        f.write(f"{'USE_LES':<20}{str(use_les).upper()}\n")
        
        # Si LES está activado, escribir los parámetros del modelo Smagorinsky
        if use_les:
            f.write(f"{'SGS_MODEL_TYPE':<20}{config.get('SGS_MODEL_TYPE', 'smagorinsky_dynamic')}\n")
            f.write(f"{'SGS_FILTER_RATIO':<20}{config.get('SGS_FILTER_RATIO', 2.0)}\n")
            f.write(f"{'SGS_AVG_TYPE':<20}{config.get('SGS_AVG_TYPE', 'global')}\n")
            f.write(f"{'SGS_CS_MIN':<20}{config.get('SGS_CS_MIN', 0.01):.4f}\n")

def generate_comparison_cases():
    """
    Genera un conjunto de casos de prueba para comparar FR (ILES) vs FR con 
    el modelo dinámico de Smagorinsky (LES).
    """
    print("--- Iniciando la generación de casos para la comparación FR vs Smagorinsky (1D) ---")
    
    # --- Parámetros Base ---
    base_config = {
        'SCHEME': 'FR',
        'VISC': 0.0,
        'INISOL': 'SQUARE',
        'NREF': 0,
        'DT': 0.00005,
        'TSIM': 0.5,
        'NDUMP': 100
    }

    # --- Variaciones de Parámetros ---
    p_values = [3, 4]               # Dos grados polinomiales para la comparación
    n_values = [30, 50, 70, 90, 110] # Cinco resoluciones de malla para cada 'p'
    
    # Esto generará 2 * 5 = 10 pares de casos (20 ficheros en total)
    variations = list(itertools.product(p_values, n_values))
    
    # Directorio donde se guardarán los nuevos ficheros
    subdirectory = "fr_smagorinsky_comparison"
    
    # --- Bucle de Generación ---
    for p, n in variations:
        
        # --- 1. Generar el caso ILES (sin modelo SGS) ---
        iles_config = base_config.copy()
        iles_config['P'] = p
        iles_config['N'] = n
        iles_config['USE_LES'] = False
        
        iles_filename = f"FR_p{p}_n{n}_ILES.txt"
        write_config_file(iles_config, iles_filename, subdirectory)
        
        # --- 2. Generar el caso LES (con Smagorinsky dinámico) ---
        les_config = base_config.copy()
        les_config['P'] = p
        les_config['N'] = n
        les_config['USE_LES'] = True
        # Parámetros específicos del modelo (puedes ajustarlos si lo necesitas)
        les_config['SGS_MODEL_TYPE'] = 'smagorinsky_dynamic'
        les_config['SGS_FILTER_RATIO'] = 2.0
        les_config['SGS_AVG_TYPE'] = 'global'
        les_config['SGS_CS_MIN'] = 0.01
        
        les_filename = f"FR_p{p}_n{n}_LES_Smago.txt"
        write_config_file(les_config, les_filename, subdirectory)

    print(f"\n¡Éxito! Se generaron {len(variations) * 2} ficheros de casos en 'data/inputs/{subdirectory}/'")

# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    generate_comparison_cases()