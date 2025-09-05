import os

# ==============================================================================
# SECCIÓN PARA CASOS 1D
# ==============================================================================

def write_1d_config_file(config, filename, subdirectory):
    """
    Escribe un fichero de configuración para el solver 1D.
    """
    base_directory = "data/inputs"
    target_directory = os.path.join(base_directory, subdirectory)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    filepath = os.path.join(target_directory, filename)
    
    with open(filepath, 'w') as f:
        f.write("# Input file for fr-burgers-turbulent.py program\n")
        f.write(f"N                  {config['N']} # Number of mesh points\n")
        f.write(f"NREF                 {config['NREF']} # Number of refinement levels of the mesh\n")
        f.write(f"P                    {config['P']} # Polynomial degree for FR scheme\n")
        f.write(f"VISC          {config['VISC']:.6f} # Viscosity\n")
        f.write(f"INISOL       {config['INISOL']} # Initial solution type\n")
        f.write(f"DT            {config['DT']:.6f} # Time step\n")
        f.write(f"TSIM          {config['TSIM']:.6f} # Maximum simulation time\n")
        f.write(f"NDUMP               {config['NDUMP']} # Interval to dump a solution\n")
        f.write("# --- LES Parameters ---\n")
        f.write(f"USE_LES                   {'TRUE' if config.get('USE_LES', False) else 'FALSE'}\n")

def generate_1d_convergence_cases():
    """
    Genera el conjunto de ficheros de entrada para el estudio de convergencia 1D.
    """
    print("--- Iniciando generación de casos de convergencia 1D ---")
    base_config_1d = {
        'NREF': 0, 'VISC': 0.01, 'INISOL': 'SINE', 'TSIM': 2.0, 'NDUMP': 2000, 'USE_LES': False
    }
    p_values = [2, 3]
    n_values = [10, 20, 40, 80, 160]
    
    for p in p_values:
        for n in n_values:
            case_config = base_config_1d.copy()
            case_config['P'] = p
            case_config['N'] = n
            case_config['DT'] = 0.0001 # DT fijo para el caso 1D es más estable
            
            filename = f"conv_1d_p{p}_n{n}.txt"
            subdirectory = "convergence_study_1d"
            write_1d_config_file(case_config, filename, subdirectory)
    print(f"Se generaron {len(p_values) * len(n_values)} ficheros 1D en data/inputs/convergence_study_1d/\n")

# ==============================================================================
# SECCIÓN PARA CASOS 2D
# ==============================================================================

def write_2d_config_file(config, filename, subdirectory):
    """
    Escribe un fichero de configuración para el solver 2D.
    """
    base_directory = "data/inputs"
    target_directory = os.path.join(base_directory, subdirectory)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    filepath = os.path.join(target_directory, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"{'NX':<20}{config['NX']}\n")
        f.write(f"{'NY':<20}{config['NY']}\n")
        f.write(f"{'P':<20}{config['P']}\n")
        f.write(f"{'SCHEME':<20}{config.get('SCHEME', 'FR')}\n")
        f.write(f"{'VISC':<20}{config['VISC']:.8f}\n")
        f.write(f"{'NREF':<20}{config.get('NREF', 0)}\n")
        f.write(f"{'INISOL':<20}{config['INISOL']}\n")
        f.write(f"{'DT':<20}{config['DT']:.8f}\n")
        f.write(f"{'TSIM':<20}{config['TSIM']:.8f}\n")
        f.write(f"{'NDUMP':<20}{config['NDUMP']}\n")
        f.write("# --- LES Parameters ---\n")
        f.write(f"{'USE_LES':<20}{'TRUE' if config.get('USE_LES', False) else 'FALSE'}\n")
        # Asegurarse de que los parámetros del modelo no se escriban si no se usan
        if config.get('USE_LES', False):
            f.write(f"{'SGS_MODEL_TYPE':<20}{config.get('SGS_MODEL_TYPE', 'NONE')}\n")

def calculate_adaptive_dt(CFL_target, N, P, U_max=1.0, L=2*3.14159):
    """
    Calcula un paso de tiempo seguro basado en la condición CFL para 2D.
    La estabilidad para FR a menudo escala con N y P^2.
    """
    dx = L / N
    dt = CFL_target * (dx / U_max) / (P**2)
    return dt

def generate_2d_convergence_cases():
    """
    Genera el conjunto de ficheros de entrada para el estudio de convergencia 2D.
    """
    print("--- Iniciando generación de casos de convergencia 2D ---")
    base_config_2d = {
        'SCHEME': 'FR', 'VISC': 0.01, 'INISOL': 'TAYLOR_GREEN', 'TSIM': 0.5, 'NDUMP': 5000, 'USE_LES': False, 'NREF': 0
    }
    p_values = [2, 3]
    n_values = [9, 17, 33, 65, 129]
    CFL_const = 0.5 # Constante CFL conservadora para garantizar estabilidad

    for p in p_values:
        for n in n_values:
            case_config = base_config_2d.copy()
            case_config['P'] = p
            case_config['NX'] = n
            case_config['NY'] = n
            
            # Calculamos un DT adaptativo para garantizar la estabilidad numérica
            case_config['DT'] = calculate_adaptive_dt(CFL_const, n, p)
            
            filename = f"conv_2d_p{p}_n{n}.txt"
            subdirectory = "convergence_study_2d"
            write_2d_config_file(case_config, filename, subdirectory)
    print(f"Se generaron {len(p_values) * len(n_values)} ficheros 2D en data/inputs/convergence_study_2d/\n")

# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("=========================================================")
    print("INICIANDO GENERACIÓN DE TODOS LOS CASOS DE CONVERGENCIA")
    print("=========================================================\n")
    
    # Generar casos 1D
    generate_1d_convergence_cases()
    
    # Generar casos 2D
    generate_2d_convergence_cases()
    
    print("---------------------------------------------------------")
    print("¡Proceso completado! Todos los ficheros han sido creados.")
    print("---------------------------------------------------------")