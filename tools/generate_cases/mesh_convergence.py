# tools/generate_cases/generate_mesh_convergence.py
import os

def write_config_file(config, filename, subdirectory):
    """
    Escribe un diccionario de configuración a un archivo .txt en una subcarpeta específica.
    """
    base_directory = "data/inputs"
    target_directory = os.path.join(base_directory, subdirectory)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Directorio creado: {target_directory}")

    filepath = os.path.join(target_directory, filename)
    
    with open(filepath, 'w') as f:
        print(f"Generando archivo: {filepath}...")
        
        # Escribir parámetros estándar
        f.write(f"{'NX':<20}{config['NX']}\n")
        f.write(f"{'NY':<20}{config['NY']}\n")
        f.write(f"{'P':<20}{config['P']}\n")
        f.write(f"{'SCHEME':<20}{config['SCHEME']}\n")
        f.write(f"{'VISC':<20}{config['VISC']:.8f}\n")
        f.write(f"{'INISOL':<20}{config['INISOL']}\n")
        f.write(f"{'DT':<20}{config['DT']:.8f}\n")
        f.write(f"{'TSIM':<20}{config['TSIM']:.8f}\n")
        f.write(f"{'NDUMP':<20}{config['NDUMP']}\n")
        
        # Escribir parámetros LES
        f.write(f"# --- LES Parameters ---\n")
        f.write(f"{'USE_LES':<20}{str(config.get('USE_LES', False)).upper()}\n")
        
        if config.get('USE_LES', False):
            model_type = config.get('SGS_MODEL_TYPE', 'NONE')
            f.write(f"{'SGS_MODEL_TYPE':<20}{model_type}\n")
            if model_type == 'VREMAN':
                f.write(f"{'SGS_C_VREMAN':<20}{config.get('SGS_C_VREMAN', 0.07):.4f}\n")
        
        # Omitir parámetros de forzamiento ya que no se usan en este estudio
        f.write(f"# --- Forcing Parameters ---\n")
        f.write(f"{'USE_FORCING':<20}FALSE\n")

def generate_mesh_convergence_study():
    """
    Genera los casos de estudio para analizar la convergencia de malla
    del modelo de Vreman en un entorno de turbulencia en decaimiento.
    """
    print("\n--- GENERANDO CASOS PARA: ANÁLISIS DE CONVERGENCIA DE MALLA (VREMAN) ---")
    
    # Se usa el caso de decaimiento (Taylor-Green) como base.
    base_config = {
        'P': 3,
        'SCHEME': 'DC',
        'TSIM': 1.0, 
        'NDUMP': 500,
        'VISC': 0.005,
        'INISOL': 'TAYLOR_GREEN',
        'USE_LES': True,
        'SGS_MODEL_TYPE': 'VREMAN',
        'SGS_C_VREMAN': 0.07 # Usamos el coeficiente estándar
    }
    
    subdirectory = "dc_vreman_convergence"
    
    # --- Definir las resoluciones de malla y los pasos de tiempo ---
    # Es crucial reducir DT al refinar la malla para mantener la estabilidad (condición CFL)
    mesh_resolutions = [
        {'N': 33,  'DT': 0.0002},   # Malla gruesa
        {'N': 65,  'DT': 0.0001},   # Malla media
        {'N': 129, 'DT': 0.00005}   # Malla fina
    ]
    
    print(f"\nGenerando {len(mesh_resolutions)} casos para el estudio de convergencia de Vreman...")
    for res in mesh_resolutions:
        config = base_config.copy()
        N = res['N']
        config['NX'] = N
        config['NY'] = N
        config['DT'] = res['DT']
        
        # Crear un nombre de fichero descriptivo
        filename = f"DC_Vreman_convergence_N{N}.txt"
        write_config_file(config, filename, subdirectory)

# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("==========================================================")
    print("INICIANDO GENERACIÓN DE CASOS PARA EL ESTUDIO DE CONVERGENCIA DE MALLA")
    print("==========================================================")
    
    generate_mesh_convergence_study()

    print("\n=======================================================")
    print("GENERACIÓN DE CASOS DE CONVERGENCIA DE MALLA COMPLETADA.")
    print("=======================================================")