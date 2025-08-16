#
# SCRIPT PARA LA GENERACIÓN AUTOMÁTICA Y ORGANIZADA DE CASOS DE PRUEBA 2D
#
import os

def write_config_file(config, filename, subdirectory):
    """
    Escribe un diccionario de configuración a un archivo .txt en una subcarpeta específica,
    usando el formato requerido (mayúsculas y alineado).
    """
    # Directorio base donde se guardarán todas las carpetas de casos
    base_directory = "data/inputs"
    
    # Crear la ruta completa a la subcarpeta (ej. 'data/inputs/vreman')
    target_directory = os.path.join(base_directory, subdirectory)

    # Asegurarse de que el directorio de salida existe; si no, lo crea
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
        f.write(f"{'USE_LES':<20}{str(config['USE_LES']).upper()}\n")
        
        if config['USE_LES']:
            model_type = config.get('SGS_MODEL_TYPE', 'NONE')
            f.write(f"{'SGS_MODEL_TYPE':<20}{model_type}\n")
            if model_type == 'VREMAN':
                f.write(f"{'SGS_C_VREMAN':<20}{config['SGS_C_VREMAN']}\n")
            elif model_type == 'SMAGORINSKY':
                f.write(f"{'SGS_CS_CONSTANT':<20}{config['SGS_CS_CONSTANT']}\n")

def generate_vreman_cases():
    """Genera un conjunto de casos de prueba para el modelo de Vreman."""
    print("\n--- GENERANDO CASOS PARA VREMAN ---")
    subdirectory = "vreman"
    base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'SCHEME': 'FR', 'DT': 0.0001, 'NDUMP': 500,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07,
    }

    # Variaciones
    variations = [
        {'INISOL': 'TAYLOR_GREEN', 'VISC': 0.01,   'TSIM': 0.2, 'id': 'TG_Visc01_T02'},
        {'INISOL': 'TAYLOR_GREEN', 'VISC': 0.005,  'TSIM': 1.0, 'id': 'TG_Visc005_T1'},
        {'INISOL': 'GAUSSIAN_2D',  'VISC': 0.01,   'TSIM': 0.5, 'id': 'Gauss_Visc01_T05'},
    ]
    
    for var in variations:
        config = base_config.copy()
        config.update(var)
        filename = f"Case_{var['id']}.txt"
        write_config_file(config, filename, subdirectory)

def generate_smagorinsky_cases():
    """Genera un conjunto de casos de prueba para el modelo de Smagorinsky."""
    print("\n--- GENERANDO CASOS PARA SMAGORINSKY ---")
    subdirectory = "smagorinsky"
    base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'SCHEME': 'FR', 'DT': 0.0001, 'NDUMP': 500,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15,
    }
    
    # Usamos las mismas variaciones para poder comparar directamente
    variations = [
        {'INISOL': 'TAYLOR_GREEN', 'VISC': 0.01,   'TSIM': 0.2, 'id': 'TG_Visc01_T02'},
        {'INISOL': 'TAYLOR_GREEN', 'VISC': 0.005,  'TSIM': 1.0, 'id': 'TG_Visc005_T1'},
        {'INISOL': 'GAUSSIAN_2D',  'VISC': 0.01,   'TSIM': 0.5, 'id': 'Gauss_Visc01_T05'},
    ]
    
    for var in variations:
        config = base_config.copy()
        config.update(var)
        filename = f"Case_{var['id']}.txt"
        write_config_file(config, filename, subdirectory)

def generate_no_les_cases():
    """Genera casos de prueba sin modelo LES (DNS/ILES) como base de comparación."""
    print("\n--- GENERANDO CASOS SIN LES (BASE) ---")
    subdirectory = "no_les"
    base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'SCHEME': 'FR', 'DT': 0.0001, 'NDUMP': 500,
        'USE_LES': False,
    }

    # Usamos las mismas variaciones para tener una referencia clara
    variations = [
        {'INISOL': 'TAYLOR_GREEN', 'VISC': 0.01,   'TSIM': 0.2, 'id': 'TG_Visc01_T02'},
        {'INISOL': 'TAYLOR_GREEN', 'VISC': 0.005,  'TSIM': 1.0, 'id': 'TG_Visc005_T1'},
        {'INISOL': 'GAUSSIAN_2D',  'VISC': 0.01,   'TSIM': 0.5, 'id': 'Gauss_Visc01_T05'},
    ]
    
    for var in variations:
        config = base_config.copy()
        config.update(var)
        filename = f"Case_{var['id']}.txt"
        write_config_file(config, filename, subdirectory)

# --- PUNTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    print("=================================================")
    print("INICIANDO GENERACIÓN DE ARCHIVOS DE CONFIGURACIÓN")
    print("=================================================")
    
    generate_vreman_cases()
    generate_smagorinsky_cases()
    generate_no_les_cases()
    
    print("\n=================================================")
    print("TODOS LOS ARCHIVOS HAN SIDO GENERADOS CON ÉXITO.")
    print("Ahora puedes ejecutar tus simulaciones por directorios, por ejemplo:")
    print("python fr-burgers-2d.py data/inputs/vreman/*.txt")
    print("=================================================")