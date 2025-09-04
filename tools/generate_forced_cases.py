# tools/generate_forced_cases.py

import os
# Importamos las funciones reutilizables del módulo original
from generate_cases import generate_case_set

# --- PUNTO DE ENTRADA PRINCIPAL PARA CASOS FORZADOS ---
if __name__ == "__main__":
    print("=======================================================")
    print("INICIANDO GENERACIÓN DE CASOS DE TURBULENCIA FORZADA")
    print("=======================================================")

    # Configuración base común para todos los casos forzados
    forced_base_config = {
        'NX': 65, 'NY': 65, 'P': 3, 'DT': 0.0001, 'TSIM': 2.0, 'NDUMP': 1000,
        'VISC': 0.001,
        'USE_FORCING': True,
        'FORCING_K_MIN': 2.0,
        'FORCING_K_MAX': 5.0,
        'FORCING_AMPLITUDE': 0.05
    }

    # Definir las variaciones para los casos forzados
    # Usamos una única condición inicial, ya que el forzamiento dominará el flujo.
    forced_variations = [
        {'INISOL': 'TAYLOR_GREEN', 'id': 'forced'},
    ]

    # 1. Generar casos FORZADOS SIN LES (ILES)
    iles_config = forced_base_config.copy()
    iles_config.update({'USE_LES': False})
    generate_case_set(iles_config, "iles_forced", forced_variations)

    # 2. Generar casos FORZADOS con SMAGORINSKY
    smagorinsky_config = forced_base_config.copy()
    smagorinsky_config.update({
        'USE_LES': True, 'SGS_MODEL_TYPE': 'SMAGORINSKY', 'SGS_CS_CONSTANT': 0.15
    })
    generate_case_set(smagorinsky_config, "smagorinsky_forced", forced_variations)

    # 3. Generar casos FORZADOS con VREMAN
    vreman_config = forced_base_config.copy()
    vreman_config.update({
        'USE_LES': True, 'SGS_MODEL_TYPE': 'VREMAN', 'SGS_C_VREMAN': 0.07
    })
    generate_case_set(vreman_config, "vreman_forced", forced_variations)

    print("\n=================================================")
    print("TODOS LOS ARCHIVOS FORZADOS HAN SIDO GENERADOS.")
    print("=================================================")
