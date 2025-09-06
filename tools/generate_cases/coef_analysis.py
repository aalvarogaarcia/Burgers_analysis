# tools/generate_cases/coef_analysis.py
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
                f.write(f"{'SGS_C_VREMAN':<20}{config.get('SGS_C_VREMAN', 0.07):.4f}\n")
            elif model_type == 'SMAGORINSKY':
                f.write(f"{'SGS_CS_CONSTANT':<20}{config.get('SGS_CS_CONSTANT', 0.15):.4f}\n")

        # Escribir parámetros de forzamiento
        f.write(f"# --- Forcing Parameters ---\n")
        use_forcing = config.get('USE_FORCING', False)
        f.write(f"{'USE_FORCING':<20}{str(use_forcing).upper()}\n")
        if use_forcing:
            f.write(f"{'FORCING_K_MIN':<20}{config.get('FORCING_K_MIN', 0.0):.1f}\n")
            f.write(f"{'FORCING_K_MAX':<20}{config.get('FORCING_K_MAX', 0.0):.1f}\n")
            f.write(f"{'FORCING_AMPLITUDE':<20}{config.get('FORCING_AMPLITUDE', 0.0):.4f}\n")

def generate_coefficient_study_cases():
    """
    Genera los casos de estudio para analizar la sensibilidad a los coeficientes
    de los modelos Smagorinsky y Vreman en un entorno de turbulencia forzada.
    """
    print("\n--- GENERANDO CASOS PARA: ANÁLISIS DE COEFICIENTES (DC, FORZADO) ---")
    
    # Usamos el caso de turbulencia forzada como base, ya que el estado
    # estacionario facilita la comparación del efecto de los coeficientes.
    base_config = {
        'NX': 65, 'NY': 65, 'P': 3,
        'SCHEME': 'DC',
        'DT': 0.0001, 'TSIM': 2.0, 'NDUMP': 1000,
        'VISC': 0.001,
        'INISOL': 'TAYLOR_GREEN',
        'USE_LES': True, # LES siempre está activado para este estudio
        'USE_FORCING': True,
        'FORCING_K_MIN': 2.0,
        'FORCING_K_MAX': 5.0,
        'FORCING_AMPLITUDE': 0.05
    }
    
    subdirectory = "dc_coefficient_study"
    
    # --- Variaciones para el Modelo de Smagorinsky ---
    smagorinsky_coeffs = [0.10, 0.15, 0.20] # Valores bajo, estándar y alto
    
    print(f"\nGenerando {len(smagorinsky_coeffs)} casos para Smagorinsky...")
    for cs_val in smagorinsky_coeffs:
        config = base_config.copy()
        config['SGS_MODEL_TYPE'] = 'SMAGORINSKY'
        config['SGS_CS_CONSTANT'] = cs_val
        
        # Crear un nombre de fichero descriptivo
        filename = f"DC_Smagorinsky_Cs{cs_val:.2f}.txt"
        write_config_file(config, filename, subdirectory)

    # --- Variaciones para el Modelo de Vreman ---
    vreman_coeffs = [0.05, 0.07, 0.10] # Valores bajo, estándar y alto
    
    print(f"\nGenerando {len(vreman_coeffs)} casos para Vreman...")
    for cv_val in vreman_coeffs:
        config = base_config.copy()
        config['SGS_MODEL_TYPE'] = 'VREMAN'
        config['SGS_C_VREMAN'] = cv_val
        
        filename = f"DC_Vreman_Cv{cv_val:.2f}.txt"
        write_config_file(config, filename, subdirectory)
        
# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("==========================================================")
    print("INICIANDO GENERACIÓN DE CASOS PARA EL ESTUDIO DE COEFICIENTES")
    print("==========================================================")
    
    generate_coefficient_study_cases()

    print("\n=======================================================")
    print("GENERACIÓN DE CASOS DE COEFICIENTES COMPLETADA.")
    print("=======================================================")