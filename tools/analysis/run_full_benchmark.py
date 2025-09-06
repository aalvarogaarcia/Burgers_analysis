# tools/analysis/run_full_benchmark.py
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pstats
import io

# --- PARTE 1: GENERACIÓN DE CASOS ---

def write_benchmark_config(config, filename):
    """Escribe un fichero de configuración para el benchmark."""
    target_directory = "data/inputs/dc_benchmark"
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Directorio creado: {target_directory}")

    filepath = os.path.join(target_directory, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"{'NX':<20}{config['NX']}\n")
        f.write(f"{'NY':<20}{config['NY']}\n")
        f.write(f"{'P':<20}{config['P']}\n")
        f.write(f"{'SCHEME':<20}{config['SCHEME']}\n")
        f.write(f"{'VISC':<20}{config['VISC']:.8f}\n")
        f.write(f"{'INISOL':<20}{config['INISOL']}\n")
        f.write(f"{'DT':<20}{config['DT']:.8f}\n")
        f.write(f"{'TSIM':<20}{config['TSIM']:.8f}\n")
        f.write(f"{'NDUMP':<20}{config['NDUMP']}\n")
        f.write(f"# --- LES Parameters ---\n")
        f.write(f"{'USE_LES':<20}{str(config.get('USE_LES', False)).upper()}\n")
        if config.get('USE_LES', False):
            model_type = config.get('SGS_MODEL_TYPE', 'NONE')
            f.write(f"{'SGS_MODEL_TYPE':<20}{model_type}\n")
            if model_type == 'VREMAN':
                f.write(f"{'SGS_C_VREMAN':<20}{config.get('SGS_C_VREMAN', 0.07)}\n")
            elif model_type == 'SMAGORINSKY':
                f.write(f"{'SGS_CS_CONSTANT':<20}{config.get('SGS_CS_CONSTANT', 0.15)}\n")
        f.write(f"# --- Forcing Parameters ---\n")
        f.write(f"{'USE_FORCING':<20}FALSE\n")
    return filepath

def generate_benchmark_cases():
    """Genera los 3 ficheros de configuración necesarios para el benchmark."""
    print("--- Generando ficheros de configuración para el benchmark ---")
    base_config = {
        'NX': 65, 'NY': 65, 'P': 3, 'SCHEME': 'DC', 'DT': 0.0001, 
        'TSIM': 0.5, 'NDUMP': 5000, 'VISC': 0.005, 'INISOL': 'TAYLOR_GREEN'
    }
    
    # ILES
    iles_config = base_config.copy()
    iles_config['USE_LES'] = False
    f1 = write_benchmark_config(iles_config, "DC_ILES_benchmark.txt")
    
    # Smagorinsky
    smag_config = base_config.copy()
    smag_config.update({'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15})
    f2 = write_benchmark_config(smag_config, "DC_Smagorinsky_benchmark.txt")

    # Vreman
    vreman_config = base_config.copy()
    vreman_config.update({'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07})
    f3 = write_benchmark_config(vreman_config, "DC_Vreman_benchmark.txt")
    
    print("--- Ficheros generados con éxito ---\n")
    return {"DC-ILES": f1, "DC-Smagorinsky": f2, "DC-Vreman": f3}

# --- PARTE 2: EJECUCIÓN Y ANÁLISIS ---

def run_timing_benchmark(cases_to_run, num_runs=3):
    """Ejecuta cada caso varias veces para medir el tiempo de ejecución."""
    results = {name: [] for name in cases_to_run.keys()}

    for i in range(num_runs):
        print(f"--- INICIANDO RONDA DE BENCHMARK DE TIEMPO {i+1}/{num_runs} ---")
        for name, filepath in cases_to_run.items():
            print(f"Ejecutando: {name}...")
            command = f"python fr-burgers-2d.py \"{filepath}\""
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            
            for line in result.stdout.splitlines():
                if "Total Execution Time:" in line:
                    time_str = line.split(":")[1].strip().split(" ")[0]
                    results[name].append(float(time_str))
                    break
    return results

def run_profiling_benchmark(case_filepath):
    """Ejecuta cProfile en un caso y devuelve el análisis."""
    print(f"\n--- EJECUTANDO PROFILING DETALLADO PARA: {os.path.basename(case_filepath)} ---")
    profile_output_file = "benchmark.prof"
    command = f"python -m cProfile -o {profile_output_file} fr-burgers-2d.py \"{case_filepath}\""
    
    try:
        subprocess.run(command, shell=True, check=True)
        
        # Analizar el resultado del profiling
        s = io.StringIO()
        stats = pstats.Stats(profile_output_file, stream=s)
        stats.strip_dirs()
        
        s.write("\n\n--- TOP 15 FUNCIONES POR TIEMPO TOTAL (CUMULATIVO) ---\n")
        stats.sort_stats('cumulative').print_stats(15)
        
        s.write("\n\n--- TOP 15 FUNCIONES POR TIEMPO INTERNO (TOTAL) ---\n")
        stats.sort_stats('tottime').print_stats(15)
        
        return s.getvalue()

    except subprocess.CalledProcessError as e:
        return f"Error ejecutando el profiler: {e}"
    finally:
        if os.path.exists(profile_output_file):
            os.remove(profile_output_file)

# --- PARTE 3: PUNTO DE ENTRADA PRINCIPAL ---

if __name__ == "__main__":
    # 1. Generar los ficheros de configuración
    cases = generate_benchmark_cases()
    
    # 2. Ejecutar el benchmark de tiempo
    timing_results = run_timing_benchmark(cases, num_runs=3)

    # 3. Calcular estadísticas y mostrar tabla de tiempos
    avg_times = {name: np.mean(times) for name, times in timing_results.items() if times}
    std_devs = {name: np.std(times) for name, times in timing_results.items() if times}

    print("\n\n--- RESULTADOS FINALES DEL BENCHMARK DE TIEMPO ---")
    print(f"{'Modelo':<20} | {'Tiempo Promedio (s)':<20} | {'Desv. Estándar (s)':<20}")
    print("-" * 65)
    for name in avg_times:
        print(f"{name:<20} | {avg_times[name]:<20.4f} | {std_devs[name]:<20.4f}")

    # 4. Generar gráfico de barras
    if avg_times:
        labels, times, errors = list(avg_times.keys()), list(avg_times.values()), list(std_devs.values())
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, times, yerr=errors, capsize=5)
        ax.set_ylabel('Tiempo de Ejecución Promedio (s)')
        ax.set_title('Benchmark de Rendimiento de Modelos SGS (Esquema DC)')
        ax.bar_label(bars, fmt='{:.2f}s')
        plt.tight_layout()
        plt.savefig("benchmark_results.png")
        print("\nGráfico de barras guardado en 'benchmark_results.png'")
        plt.show()

    # 5. Ejecutar el profiling detallado sobre el caso más complejo (Vreman)
    profile_analysis = run_profiling_benchmark(cases["DC-Vreman"])
    print(profile_analysis)