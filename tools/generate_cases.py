# tools/generate_cases.py
#
# Script avanzado para generar conjuntos de casos de prueba para el TFM.
# Permite crear sistemáticamente los ficheros de entrada para estudios
# de convergencia y comparativas de modelos SGS.

import os
import copy

# --- CONFIGURACIÓN BASE ---
# Mantenemos una configuración por defecto como punto de partida.

def get_default_config_2d():
    """Devuelve un diccionario con una configuración 2D por defecto."""
    return {
        'PROB_TYPE': '2D',
        'SCHEME': 'FR',
        'P': 3,
        'NX': 32,
        'NY': 32,
        'NVAR': 2,
        'VISC': 1.0e-3,
        'TSIM': 5.0,
        'DT': 1.0e-4,
        'NDUMP': 500,
        'INISOL': 'TURB',
        'USE_LES': False,
        'SGS_MODEL_TYPE': 'NONE',
        'CS': 0.1,
        'CV': 0.1,
        'FILTER_TYPE': 'BOX',
        'TEST_FILTER_WIDTH': 2.0,
    }

# --- FUNCIÓN DE ESCRITURA ---
# La función para escribir el fichero no cambia, pero nos aseguramos
# de que apunte al directorio correcto.

def write_config_file(config, filename, directory="data/inputs"):
    """
    Escribe un diccionario de configuración a un archivo .txt.

    Args:
        config (dict): Parámetros de la simulación.
        filename (str): Nombre del archivo de salida.
        directory (str): Directorio donde se guardará el archivo (relativo a la raíz).
    """
    # Nos aseguramos de que la ruta sea relativa a la raíz del proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, directory)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write(f"## --- Simulation Case: {filename} --- ##\n\n")
        for key, value in config.items():
            if isinstance(value, bool):
                value_str = '1' if value else '0'
            else:
                value_str = str(value)
            f.write(f"{key.ljust(20)} {value_str}\n")

    print(f"-> Caso generado: '{os.path.join(directory, filename)}'")


# --- GENERADORES DE ESTUDIOS ---

def generate_convergence_study_cases(p_order=3, scheme='FR'):
    """
    Genera los ficheros de entrada para un estudio de convergencia de malla.
    Se crean casos con resoluciones crecientes (16, 32, 64, 128).
    
    Objetivo del TFM relacionado: Verificación del solver (Paso 2 del plan).
    """
    print(f"\n--- Generando Casos para Estudio de Convergencia (P={p_order}, Esquema={scheme}) ---")
    base_resolutions = [16, 32, 64, 128]
    
    for n in base_resolutions:
        config = get_default_config_2d()
        config['SCHEME'] = scheme
        config['P'] = p_order
        config['NX'] = n
        config['NY'] = n
        config['USE_LES'] = False # El estudio de convergencia se hace sobre el solver base
        config['SGS_MODEL_TYPE'] = 'NONE'
        config['TSIM'] = 0.5 # Tiempo corto, solo para verificar convergencia
        
        filename = f"Conv_{scheme}_P{p_order}_N{n}.txt"
        write_config_file(config, filename)

def generate_model_comparison_cases(resolution=64, p_order=3):
    """
    Genera un conjunto de casos para comparar los modelos ILES, Smagorinsky y Vreman
    con un esquema numérico y una resolución fijos.

    Objetivo del TFM relacionado: Comparativa de modelos SGS (Pasos 1, 3, 4 del plan).
    """
    print(f"\n--- Generando Casos para Comparativa de Modelos (N={resolution}, P={p_order}) ---")
    
    # 1. Caso ILES (sin modelo explícito) con esquema FR
    config_iles_fr = get_default_config_2d()
    config_iles_fr['SCHEME'] = 'FR'
    config_iles_fr['P'] = p_order
    config_iles_fr['NX'] = resolution
    config_iles_fr['NY'] = resolution
    config_iles_fr['USE_LES'] = False
    config_iles_fr['SGS_MODEL_TYPE'] = 'NONE'
    write_config_file(config_iles_fr, f"Compare_FR_ILES_N{resolution}_P{p_order}.txt")

    # 2. Caso ILES (sin modelo explícito) con esquema DC
    config_iles_dc = copy.deepcopy(config_iles_fr)
    config_iles_dc['SCHEME'] = 'DC'
    write_config_file(config_iles_dc, f"Compare_DC_ILES_N{resolution}_P{p_order}.txt")

    # 3. Caso LES con Smagorinsky (usando FR)
    config_smag = copy.deepcopy(config_iles_fr)
    config_smag['USE_LES'] = True
    config_smag['SGS_MODEL_TYPE'] = 'SMAGORINSKY'
    config_smag['CS'] = 0.15
    write_config_file(config_smag, f"Compare_FR_Smagorinsky_N{resolution}_P{p_order}.txt")

    # 4. Caso LES con Vreman (usando FR)
    config_vreman = copy.deepcopy(config_iles_fr)
    config_vreman['USE_LES'] = True
    config_vreman['SGS_MODEL_TYPE'] = 'VREMAN'
    config_vreman['CV'] = 0.1
    write_config_file(config_vreman, f"Compare_FR_Vreman_N{resolution}_P{p_order}.txt")


# --- PUNTO DE ENTRADA PRINCIPAL ---

if __name__ == "__main__":
    
    # Generar los casos para el estudio de convergencia con el esquema FR
    # Esto es crucial para el Paso 2 de nuestro plan: verificar la precisión del solver.
    generate_convergence_study_cases(p_order=3, scheme='FR')
    
    # Generar los casos para el estudio de convergencia con el esquema DC
    generate_convergence_study_cases(p_order=2, scheme='DC') # DC es de orden 2

    # Generar los casos para la comparativa principal de modelos.
    # Esto aborda los Pasos 1, 3 y 4 del plan: comparar ILES vs LES (Smagorinsky, Vreman).
    # Usamos una resolución media (64x64) que es típica para este tipo de análisis.
    generate_model_comparison_cases(resolution=64, p_order=3)

    print("\n¡Generación de casos de prueba completada!")

