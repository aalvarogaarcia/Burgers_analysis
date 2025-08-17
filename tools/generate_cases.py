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
        f.write(f"{'USE_LES':<20}{str(config['USE_LES']).upper()}\n")
        
        if config['USE_LES']:
            model_type = config.get('SGS_MODEL_TYPE', 'NONE')
            f.write(f"{'SGS_MODEL_TYPE':<20}{model_type}\n")
            if model_type == 'VREMAN':
                f.write(f"{'SGS_C_VREMAN':<20}{config['SGS_C_VREMAN']}\n")
            elif model_type == 'SMAGORINSKY':
                f.write(f"{'SGS_CS_CONSTANT':<20}{config['SGS_CS_CONSTANT']}\n")

def generate_case_set(base_config, subdirectory):
    """
    Función modular que genera un conjunto completo de 10 casos de prueba
    (5 FR y 5 DC) para una configuración base dada.
    """
    print(f"\n--- GENERANDO CASOS PARA: {subdirectory.upper()} ---")

    # --- Definir las variaciones de viscosidad ---
    # 2 casos Taylor-Green
    tg_variations = [
        {'INISOL': 'TAYLOR_GREEN', 'VISC': 0.05, 'id': base_config['id'] + 'TG_Visc0.05'},
        {'INISOL': 'TAYLOR_GREEN', 'VISC': 0.5,  'id': base_config['id'] + 'TG_Visc0.5'},
    ]
    # 3 casos Gaussianos
    gauss_variations = [
        {'INISOL': 'GAUSSIAN_2D',  'VISC': 0.05, 'id': base_config['id'] + 'Gauss_Visc0.05'},
        {'INISOL': 'GAUSSIAN_2D',  'VISC': 0.5,  'id': base_config['id'] + 'Gauss_Visc0.5'},
        {'INISOL': 'GAUSSIAN_2D',  'VISC': 0.8,  'id': base_config['id'] + 'Gauss_Visc0.8'},
    ]
    all_variations = tg_variations + gauss_variations

    # --- Generar 5 casos para el esquema FR ---
    for var in all_variations:
        config = base_config.copy()
        config.update(var)
        config['SCHEME'] = 'FR'
        filename = f"FR_{var['id']}.txt"
        write_config_file(config, filename, subdirectory)

    # --- Generar 5 casos para el esquema DC ---
    for var in all_variations:
        config = base_config.copy()
        config.update(var)
        config['SCHEME'] = 'DC'
        filename = f"DC_{var['id']}.txt"
        write_config_file(config, filename, subdirectory)

# --- PUNTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    print("=================================================")
    print("INICIANDO GENERACIÓN DE CASOS DE PRUEBA ESTABLES")
    print("=================================================")
    
    # --- 1. Generar casos SIN LES (ILES / DNS) ---
    # Estos casos usarán viscosidades altas para garantizar la estabilidad.
    no_les_base_config = {
        'NX': 33, 'NY': 33, 'P': 5, 'DT': 0.0001, 'TSIM': 1., 'NDUMP': 500,
        'USE_LES': False, 'id': 't_1_iles_' 
    }
    generate_case_set(no_les_base_config, "no_les_stable")

    # --- 2. Generar casos con SMAGORINSKY ---
    # Usamos viscosidades más bajas, ya que el modelo SGS añade disipación.
    smagorinsky_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': 1.0, 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15,
        'id': 't_1_smg0.15_'
    }
    generate_case_set(smagorinsky_base_config, "smagorinsky")
    
    # --- 3. Generar casos con VREMAN ---
    vreman_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': 1.0, 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07,
        'id': 't_1_vrn0.07_'
    }
    generate_case_set(vreman_base_config, "vreman")
    
    
    smagorinsky_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': 1.0, 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.07,
        'id': 't_1_smg0.07_'
    }
    generate_case_set(smagorinsky_base_config, "smagorinsky")
    
    # --- 3. Generar casos con VREMAN ---
    vreman_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': 1.0, 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.15,
        'id': 't_1_vrn0.15_'
        
    }
    
    generate_case_set(vreman_base_config, "vreman")
    
    
       
    
    
   ########################################
   #-------------T = 2 -------------------#
   ######################################## 
    
    #--- 1. Generar casos SIN LES (ILES / DNS) ---
    # Estos casos usarán viscosidades altas para garantizar la estabilidad.
    no_les_base_config = {
        'NX': 33, 'NY': 33, 'P': 5, 'DT': 0.0001, 'TSIM': 2., 'NDUMP': 500,
        'USE_LES': False, 'id': 't_2_iles_' 
    }
    generate_case_set(no_les_base_config, "no_les_stable")

    # --- 2. Generar casos con SMAGORINSKY ---
    # Usamos viscosidades más bajas, ya que el modelo SGS añade disipación.
    smagorinsky_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': 2., 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15,
        'id': 't_2_smg0.15_'
    }
    generate_case_set(smagorinsky_base_config, "smagorinsky")
    
    # --- 3. Generar casos con VREMAN ---
    vreman_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': 2., 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07,
        'id': 't_2_vrn0.07_'
    }
    generate_case_set(vreman_base_config, "vreman")
    
    
    smagorinsky_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': 2., 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.07,
        'id': 't_2_smg0.07_'
    }
    generate_case_set(smagorinsky_base_config, "smagorinsky")
    
    # --- 3. Generar casos con VREMAN ---
    vreman_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': 2., 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.15,
        'id': 't_2_vrn0.15_'
        
    }
    
    generate_case_set(vreman_base_config, "vreman")
    
    
    ########################################
    #-------------T = 0.5 -------------------#
    ######################################## 
     
    #--- 1. Generar casos SIN LES (ILES / DNS) ---
    # Estos casos usarán viscosidades altas para garantizar la estabilidad.
    no_les_base_config = {
        'NX': 33, 'NY': 33, 'P': 5, 'DT': 0.0001, 'TSIM': .5, 'NDUMP': 500,
        'USE_LES': False, 'id': 't_5_iles_' 
    }
    generate_case_set(no_les_base_config, "no_les_stable")

    # --- 2. Generar casos con SMAGORINSKY ---
    # Usamos viscosidades más bajas, ya que el modelo SGS añade disipación.
    smagorinsky_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': .5, 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15,
        'id': 't_5_smg0.15_'
    }
    generate_case_set(smagorinsky_base_config, "smagorinsky")
    
    # --- 3. Generar casos con VREMAN ---
    vreman_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': .5, 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07,
        'id': 't_5_vrn0.07_'
    }
    generate_case_set(vreman_base_config, "vreman")
    
    
    smagorinsky_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': .5, 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.07,
        'id': 't_5_smg0.07_'
    }
    generate_case_set(smagorinsky_base_config, "smagorinsky")
    
    # --- 3. Generar casos con VREMAN ---
    vreman_base_config = {
        'NX': 33, 'NY': 33, 'P': 2, 'DT': 0.0001, 'TSIM': .5, 'NDUMP': 1000,
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.15,
        'id': 't_5_vrn0.15_'
        
    }
    
    generate_case_set(vreman_base_config, "vreman")

    
 
    
       
    
    
    
    
    
    
    
    
    
    
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("\n=================================================")
    print("TODOS LOS ARCHIVOS HAN SIDO GENERADOS CON ÉXITO.")
    print("=================================================")