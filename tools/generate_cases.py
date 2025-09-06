# tools/generate_cases.py
import os

def write_config_file(config, filename, subdirectory):
    """
    Escribe un diccionario de configuración a un archivo .txt en una subcarpeta específica,
    usando el formato requerido (mayúsculas y alineado).
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
                f.write(f"{'SGS_C_VREMAN':<20}{config.get('SGS_C_VREMAN', 0.07)}\n")
            elif model_type == 'SMAGORINSKY':
                f.write(f"{'SGS_CS_CONSTANT':<20}{config.get('SGS_CS_CONSTANT', 0.15)}\n")

        # Escribir parámetros de forzamiento (si existen en el config)
        if config.get('USE_FORCING', False):
            f.write(f"# --- Forcing Parameters ---\n")
            f.write(f"{'USE_FORCING':<20}{str(config.get('USE_FORCING', False)).upper()}\n")
            f.write(f"{'FORCING_K_MIN':<20}{config.get('FORCING_K_MIN', 0.0):.1f}\n")
            f.write(f"{'FORCING_K_MAX':<20}{config.get('FORCING_K_MAX', 0.0):.1f}\n")
            f.write(f"{'FORCING_AMPLITUDE':<20}{config.get('FORCING_AMPLITUDE', 0.0):.4f}\n")

def generate_case_set(base_config, subdirectory, variations):
    """
    Función modular que genera un conjunto de casos de prueba (FR y DC)
    para una configuración base y una lista de variaciones.
    """
    print(f"\n--- GENERANDO CASOS PARA: {subdirectory.upper()} ---")
    
    for scheme in ['FR', 'DC']:
        for var in variations:
            config = base_config.copy()
            config.update(var)
            config['SCHEME'] = scheme
            filename = f"{scheme}_{var['id']}.txt"
            write_config_file(config, filename, subdirectory)

# --- PUNTO DE ENTRADA PRINCIPAL PARA CASOS DE DECAIMIENTO ---
if __name__ == "__main__":
    print("==========================================================")
    print("INICIANDO GENERACIÓN DE CASOS DE TURBULENCIA EN DECaimiento")
    print("==========================================================")
    
    base_config = {
        'NX': 33, 'NY': 33, 'P': 3, 'DT': 0.0001, 'TSIM': 1.0, 'NDUMP': 500,
        'VISC': 0.005
    }
    
    # Definir las variaciones para los casos de decaimiento
    decay_variations = [
        {'INISOL': 'TAYLOR_GREEN', 'id': 'decay'},
    ]
    
    # 1. Casos ILES (Implicit LES)
    iles_config = base_config.copy()
    iles_config.update({'USE_LES': False})
    generate_case_set(iles_config, "iles_decay", decay_variations)
    
    # 2. Casos Smagorinsky
    smagorinsky_config = base_config.copy()
    smagorinsky_config.update({
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15
    })
    generate_case_set(smagorinsky_config, "smagorinsky_decay", decay_variations)
    
    # 3. Casos Vreman
    vreman_config = base_config.copy()
    vreman_config.update({
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07
    })
    generate_case_set(vreman_config, "vreman_decay", decay_variations)

    print("\n=======================================================")
    print("GENERACIÓN DE CASOS DE DECAIMIENTO COMPLETADA.")
    print("Para generar casos forzados, ejecute 'generate_forced_cases.py'")
    print("=======================================================")
