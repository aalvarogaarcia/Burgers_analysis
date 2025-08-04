# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:03:39 2025

@author: agarm

Modulo de acciones y reaciones de la UI
"""

from shiny import render, reactive, req

import matplotlib.pyplot as plt
import numpy as np

# --- Importa todos tus módulos de simulación ---
from mesh import getMesh, getMeshHO
from miscellaneous import FillInitialSolution
from randw import getValueFromLabel
from lagpol import getStandardElementData
from ode import RK4
import sgs_model
from postprocess import getTKEFFT, collpaseCommonNodes, getSolutionInUniformMesh

# --- Función Auxiliar para Ejecutar una Simulación ---
def run_single_simulation(params, update_log_callback):
    # (Esta función no cambia, la dejo aquí por completitud)
    try:
        update_log_callback(f"Iniciando simulación con P={params['P']}, Visc={params['VISC']}...\n")
        N, Nref, p_order, v_molecular = params.get('N', 100), params.get('NREF', 0), params['P'], params['VISC']
        IniS, dt, tsim, Ndump = params['INISOL'], params.get('DT', 0.001), params.get('TSIM', 1.0), params.get('NDUMP', 1000)
        use_les, sgs_params = params['USE_LES'], params['SGS_PARAMS']
        Nmax = int(tsim / dt)
        if Nmax * dt < tsim: Nmax += 1
        dt = tsim / Nmax
        lobattoPoints, Lp_matrix, gp_array = getStandardElementData(p_order)
        x_ho_mesh = getMeshHO(getMesh(N, Nref), lobattoPoints)
        U = np.zeros(len(x_ho_mesh))
        FillInitialSolution(U, x_ho_mesh, IniS, N, p_order, Nref)
        for it in range(Nmax):
            U = RK4(dt, U, p_order, x_ho_mesh, v_molecular, Lp_matrix, gp_array, use_les, sgs_params)
            if it % 200 == 0:
                update_log_callback(f"  ... it: {it}/{Nmax}\n")
        update_log_callback("Simulación completada. Post-procesando...\n")
        print(f"--- DEBUG POST: Entrando a post-procesamiento. Shape de la solución U: {U.shape} ---")
        xe, ue = getSolutionInUniformMesh(x_ho_mesh, U, p_order)
        print(f"--- DEBUG POST: Shape tras getSolutionInUniformMesh (ue): {ue.shape} ---")
        xc, uc = collpaseCommonNodes(xe, ue, p_order)
        print(f"--- DEBUG POST: Shape tras collpaseCommonNodes (uc): {uc.shape} ---")
        if uc.size == 0:
            raise ValueError("collpaseCommonNodes devolvió un array vacío.")
        xt, ut = xc[:-1], uc[:-1]
        print(f"--- DEBUG POST: Shape del array para FFT (ut): {ut.shape} ---")
        if ut.size == 0:
            raise ValueError("El array de entrada para getTKEFFT está vacío.")
        k_data, tke_data = getTKEFFT(xt, ut)
        print(f"--- DEBUG POST: Shape de los resultados de FFT. k_data: {k_data.shape}, tke_data: {tke_data.shape} ---")
        print("--- DEBUG POST: Post-procesamiento finalizado con éxito. ---")
        return {'x': x_ho_mesh, 'u': U, 'k': k_data, 'tke': tke_data, 'label': params.get('label', 'Simulación Manual')}

    except Exception as e:
        print(f"!!! ERROR DENTRO DE run_single_simulation: {e}")
        import traceback
        traceback.print_exc()
        update_log_callback(f"ERROR: {e}\n")
        return None


def server(input, output, session):
    
    simulation_results = reactive.Value([])
    log_text = reactive.Value("Listo para iniciar la simulación.\n")

    def append_log(text):
        log_text.set(log_text() + text)

    # --- Lógica para la Pestaña "Entrada Manual" ---
    @reactive.Effect # <--- AÑADIR ESTA LÍNEA
    @reactive.event(input.run_manual_button)
    def handle_manual_run():
        print("--- DEBUG 1: Botón 'run_manual_button' presionado. ---")
        simulation_results.set([])
        log_text.set("")
        sgs_params = None
        if input.use_les():
            sgs_params = {'model_type': input.sgs_model_type(), 'Cs_min': input.sgs_cs_min(), 'filter_width_ratio': 2.0, 'avg_type': 'global'}
        params = {'P': input.p_order(), 'VISC': input.visc(), 'INISOL': input.inisol(), 'USE_LES': input.use_les(), 'SGS_PARAMS': sgs_params, 'N': 100, 'TSIM': 1.0, 'DT': 0.001, 'label': f"P={input.p_order()}, Visc={input.visc()}"}
        print(f"--- DEBUG 2: Parámetros para la simulación: {params} ---")
        print("--- DEBUG 3: Llamando a run_single_simulation... ---")
        result = run_single_simulation(params, append_log)
        print(f"--- DEBUG 3.1: ...llamada a run_single_simulation finalizada. Resultado es None: {result is None} ---")
        if result:
            print("--- DEBUG 4: Actualizando 'simulation_results' con el nuevo resultado. ---")
            simulation_results.set([result])

    # --- Lógica para la Pestaña "Cargar Archivos" ---
    @reactive.Effect # <--- AÑADIR ESTA LÍNEA
    @reactive.event(input.run_files_button)
    def handle_files_run():
        print("--- DEBUG 1 (Archivos): Botón 'run_files_button' presionado. ---")
        req(input.upload_files())
        log_text.set("")
        results_list = []
        files = input.upload_files()
        print(f"--- DEBUG 2 (Archivos): {len(files)} archivo(s) para procesar. ---")
        for file_info in files:
            append_log(f"--- Procesando archivo: {file_info['name']} ---\n")
            try:
                with open(file_info["datapath"], "r") as f:
                    document = f.readlines()
                use_les = getValueFromLabel(document, "USE_LES").upper() == "TRUE"
                sgs_params = None
                if use_les:
                    sgs_params = {'model_type': getValueFromLabel(document, "SGS_MODEL_TYPE"), 'Cs_min': float(getValueFromLabel(document, "SGS_CS_MIN")), 'filter_width_ratio': float(getValueFromLabel(document, "SGS_FILTER_RATIO")), 'avg_type': getValueFromLabel(document, "SGS_AVG_TYPE")}
                params = {'P': int(getValueFromLabel(document, "P")), 'VISC': float(getValueFromLabel(document, "VISC")), 'INISOL': getValueFromLabel(document, "INISOL"), 'N': int(getValueFromLabel(document, "N")), 'NREF': int(getValueFromLabel(document, "NREF")), 'DT': float(getValueFromLabel(document, "DT")), 'TSIM': float(getValueFromLabel(document, "TSIM")), 'USE_LES': use_les, 'SGS_PARAMS': sgs_params, 'label': file_info['name']}
                result = run_single_simulation(params, append_log)
                if result:
                    results_list.append(result)
            except Exception as e:
                print(f"!!! ERROR DENTRO DE handle_files_run: {e}")
                append_log(f"ERROR procesando {file_info['name']}: {e}\n")
        print(f"--- DEBUG 4 (Archivos): Actualizando 'simulation_results' con {len(results_list)} resultado(s). ---")
        simulation_results.set(results_list)

    # --- Definición de los Outputs (Gráficas y Log) ---
    @output
    @render.plot
    def main_plot():
        print("--- DEBUG 5: Renderizando 'main_plot'... ---")
        results = simulation_results()
        req(results)
        print("--- DEBUG 5.1: ...'main_plot' tiene resultados para graficar. ---")
        fig, ax = plt.subplots()
        for res in results:
            ax.plot(res['x'], res['u'], label=res['label'])
        ax.set_title("Solución u(x)"), ax.set_xlabel("x"), ax.set_ylabel("u"), ax.grid(True)
        if len(results) > 1: ax.legend()
        return fig

    @output
    @render.plot
    def spectrum_plot():
        print("--- DEBUG 6: Renderizando 'spectrum_plot'... ---")
        results = simulation_results()
        req(results)
        print("--- DEBUG 6.1: ...'spectrum_plot' tiene resultados para graficar. ---")
        fig, ax = plt.subplots()
        for res in results:
            ax.loglog(res['k'], res['tke'], label=res['label'])
        ax.set_title("Espectro de Energía"), ax.set_xlabel("Número de Onda (k)"), ax.set_ylabel("Energía"), ax.grid(True, which="both", ls="--")
        if len(results) > 1: ax.legend()
        return fig

    @output
    @render.text
    def simulation_log():
        return log_text()