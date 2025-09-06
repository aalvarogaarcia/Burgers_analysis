import os
import sys
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
    p_values = [2, 3, 4, 5]
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
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("=========================================================")
    print("INICIANDO GENERACIÓN DE TODOS LOS CASOS DE CONVERGENCIA")
    print("=========================================================\n")
    
    # Generar casos 1D
    generate_1d_convergence_cases()
    
    
    print("---------------------------------------------------------")
    print("¡Proceso completado! Todos los ficheros han sido creados.")
    print("---------------------------------------------------------")