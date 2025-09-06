# tools/generate_1d_dissipation_study.py
import os

def write_1d_config_file(config, filename, subdirectory):
    """
    Escribe la configuración para un caso 1D en el formato requerido.
    """
    base_dir = "data/inputs"
    target_dir = os.path.join(base_dir, subdirectory)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filepath = os.path.join(target_dir, filename)
    print(f"Generando caso 1D: {filepath}")

    with open(filepath, 'w') as f:
        f.write("# Input file for 1D Burgers dissipation study\n")
        f.write(f"{'N':<20}{config['N']}\n")
        f.write(f"{'P':<20}{config['P']}\n")
        f.write(f"{'SCHEME':<20}{config['SCHEME']}\n")
        f.write(f"{'VISC':<20}{config['VISC']:.8f}\n")
        f.write(f"{'INISOL':<20}{config['INISOL']}\n")
        f.write(f"{'NREF':<20}{config.get('NREF', 0)}\n")
        f.write(f"{'DT':<20}{config['DT']:.8f}\n")
        f.write(f"{'TSIM':<20}{config['TSIM']:.8f}\n")
        f.write(f"{'NDUMP':<20}{config['NDUMP']}\n")
        f.write("# --- LES Parameters ---\n")
        f.write(f"{'USE_LES':<20}FALSE\n")

if __name__ == "__main__":
    print("======================================================")
    print("Generando casos para el estudio de disipación 1D")
    print("======================================================")

    # Configuración base para el test de la onda cuadrada
    base_config = {
        'N': 100,         # Número de celdas
        'VISC': 0.0,      # Sin viscosidad física para aislar el efecto numérico
        'INISOL': 'SQUARE', # Condición inicial con un salto brusco
        'DT': 0.0005,
        'TSIM': 0.5,
        'NDUMP': 100,
        'NREF': 0        # No se usa refinamiento en 1D
    }

    # --- Caso 1: Reconstrucción de Flujos (FR) ---
    fr_config = base_config.copy()
    fr_config['SCHEME'] = 'FR'
    fr_config['P'] = 3 # Alto orden
    write_1d_config_file(fr_config, "FR_p3_square_wave.txt", "1d_dissipation_study")

    # --- Caso 2: Diferencias Centradas (DC) ---
    dc_config = base_config.copy()
    dc_config['SCHEME'] = 'DC'
    dc_config['P'] = 1 # El grado no es relevante, pero se mantiene por consistencia
    write_1d_config_file(dc_config, "DC_square_wave.txt", "1d_dissipation_study")

    # --- Caso 3: Upwind de 1er Orden (UPWIND) ---
    upwind_config = base_config.copy()
    upwind_config['SCHEME'] = 'UPWIND'
    upwind_config['P'] = 1 # El grado no es relevante
    write_1d_config_file(upwind_config, "UPWIND_square_wave.txt", "1d_dissipation_study")

    print("\nGeneración de casos completada.")