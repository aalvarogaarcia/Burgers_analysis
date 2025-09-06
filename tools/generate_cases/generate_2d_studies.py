# tools/generate_cases/generate_2d_studies.py
import os

def write_config_file(config, filename, subdirectory):
    """
    Escribe un diccionario de configuración a un archivo .txt en una subcarpeta específica.
    Esta función es genérica y se reutiliza para todos los casos.
    """
    base_directory = "data/inputs"
    target_directory = os.path.join(base_directory, subdirectory)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Directorio creado: {target_directory}")

    filepath = os.path.join(target_directory, filename)
    
    with open(filepath, 'w') as f:
        print(f"Generando archivo: {filepath}...")
        
        # Escribir parámetros estándar de forma alineada
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
                f.write(f"{'SGS_C_VREMAN':<20}{config.get('SGS_C_VREMAN', 0.07)}\n")
            elif model_type == 'SMAGORINSKY':
                f.write(f"{'SGS_CS_CONSTANT':<20}{config.get('SGS_CS_CONSTANT', 0.15)}\n")

        # Escribir parámetros de forzamiento
        f.write(f"# --- Forcing Parameters ---\n")
        use_forcing = config.get('USE_FORCING', False)
        f.write(f"{'USE_FORCING':<20}{str(use_forcing).upper()}\n")
        if use_forcing:
            f.write(f"{'FORCING_K_MIN':<20}{config.get('FORCING_K_MIN', 0.0):.1f}\n")
            f.write(f"{'FORCING_K_MAX':<20}{config.get('FORCING_K_MAX', 0.0):.1f}\n")
            f.write(f"{'FORCING_AMPLITUDE':<20}{config.get('FORCING_AMPLITUDE', 0.0):.4f}\n")

def generate_decay_cases():
    """
    Genera los 3 casos de estudio para la turbulencia en decaimiento con esquema DC.
    """
    print("\n--- GENERANDO CASOS PARA: TURBULENCIA EN DECAIMIENTO (DC) ---")
    
    base_config = {
        'NX': 65, 'NY': 65, 'P': 3, # P no se usa en DC, pero se mantiene por consistencia
        'SCHEME': 'DC',
        'DT': 0.0001, 'TSIM': 1.0, 'NDUMP': 500,
        'VISC': 0.005,
        'INISOL': 'TAYLOR_GREEN',
        'USE_FORCING': False
    }
    
    subdirectory = "dc_decay_comparison"

    # Caso 1: ILES (sin modelo)
    iles_config = base_config.copy()
    iles_config['USE_LES'] = False
    write_config_file(iles_config, "DC_ILES_decay.txt", subdirectory)
    
    # Caso 2: Smagorinsky
    smagorinsky_config = base_config.copy()
    smagorinsky_config.update({
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15
    })
    write_config_file(smagorinsky_config, "DC_Smagorinsky_decay.txt", subdirectory)
    
    # Caso 3: Vreman
    vreman_config = base_config.copy()
    vreman_config.update({
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07
    })
    write_config_file(vreman_config, "DC_Vreman_decay.txt", subdirectory)

def generate_forced_cases():
    """
    Genera los 3 casos de estudio para la turbulencia forzada con esquema DC.
    """
    print("\n--- GENERANDO CASOS PARA: TURBULENCIA FORZADA (DC) ---")
    
    base_config = {
        'NX': 65, 'NY': 65, 'P': 3,
        'SCHEME': 'DC',
        'DT': 0.0001, 'TSIM': 2.0, 'NDUMP': 1000,
        'VISC': 0.001,
        'INISOL': 'TAYLOR_GREEN', # La condición inicial es menos crítica con forzante
        'USE_FORCING': True,
        'FORCING_K_MIN': 2.0,
        'FORCING_K_MAX': 5.0,
        'FORCING_AMPLITUDE': 0.05
    }
    
    subdirectory = "dc_forced_comparison"
    
    # Caso 1: ILES (con forzante)
    iles_config = base_config.copy()
    iles_config['USE_LES'] = False
    write_config_file(iles_config, "DC_ILES_forced.txt", subdirectory)
    
    # Caso 2: Smagorinsky (con forzante)
    smagorinsky_config = base_config.copy()
    smagorinsky_config.update({
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15
    })
    write_config_file(smagorinsky_config, "DC_Smagorinsky_forced.txt", subdirectory)
    
    # Caso 3: Vreman (con forzante)
    vreman_config = base_config.copy()
    vreman_config.update({
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07
    })
    write_config_file(vreman_config, "DC_Vreman_forced.txt", subdirectory)

# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("==========================================================")
    print("INICIANDO GENERACIÓN DE CASOS DE ESTUDIO 2D")
    print("==========================================================")
    
    # Generar los casos de turbulencia en decaimiento
    generate_decay_cases()
    
    # Generar los casos de turbulencia forzada
    generate_forced_cases()

    print("\n=======================================================")
    print("GENERACIÓN DE TODOS LOS CASOS 2D COMPLETADA.")
    print("=======================================================")