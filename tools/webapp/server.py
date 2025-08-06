# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:03:39 2025

@author: agarm

Modulo de acciones y reaciones de la UI
"""


# server.py (versión final sin barra de progreso y con log mejorado)

from shiny import render, reactive, req, ui, session
import matplotlib.pyplot as plt
import numpy as np
import traceback

# --- Importa todos tus módulos de simulación ---
from mesh import *
from miscellaneous import * # Re-importar la 1D si se renombra
from randw import getValueFromLabel
from lagpol import getStandardElementData
from ode import RK4
from residual import get_residual_2d
import sgs_model

from postprocess import *



# --- Función Auxiliar para Ejecutar una Simulación (Modificada) ---
async def run_single_simulation(params, update_log_callback):
    # (Esta función no necesita más cambios)
    try:
        update_log_callback(f"Iniciando simulación: {params.get('label', '')}\n")
        
        N, Nref, p_order, v_molecular = params.get('N', 100), params.get('NREF', 0), params['P'], params['VISC']
        IniS, dt, tsim = params['INISOL'], params['DT'], params['TSIM']
        use_les, sgs_params = params['USE_LES'], params['SGS_PARAMS']
        Ndump = params.get('NDUMP', 100) 
        Nx = params.get('NX', N) # Usar N si NX no está definido (para 1D)
        Ny = params.get('NY', 0) # 0 si no está definido (para 1D)

        Nmax = int(tsim / dt)
        if Nmax * dt < tsim: Nmax += 1
        dt = tsim / Nmax
        
        lobattoPoints, Lp_matrix, gp_array = getStandardElementData(p_order)
        
        # Lógica para crear malla y condición inicial 1D o 2D
        is_2d = Ny > 0
        if is_2d:
            x_ho, y_ho = get_mesh_ho_2d(get_2d_cartesian_mesh(Nx, Ny)[0], get_2d_cartesian_mesh(Nx, Ny)[1], p_order, lobattoPoints)
            num_nodes = len(x_ho)
            U = np.zeros(2 * num_nodes)
            FillInitialSolution_2D(U, x_ho, y_ho, IniS, Nx, Ny, p_order, Nref)
            residual_function = get_residual_2d
            args_for_residual = (p_order, (x_ho, y_ho), v_molecular, (Lp_matrix, gp_array), Nx, Ny)
        else: # 1D
            x_ho = get_mesh_ho_1d(get_mesh_1d(Nx, Nref), lobattoPoints)
            y_ho = None # No hay coordenadas y
            num_nodes = len(x_ho)
            U = np.zeros(num_nodes)
            FillInitialSolution_1D(U, x_ho, IniS, Nx, p_order, Nref) # Asumiendo que esta es la función 1D
            residual_function = getResidualBrurgersFR # Asumiendo que esta es la función 1D
            args_for_residual = (p_order, x_ho, v_molecular, Lp_matrix, gp_array)
            
        update_log_callback("Ejecutando bucle temporal...\n")
        
        for it in range(Nmax):
            U = RK4(dt, U, residual_function, *args_for_residual)
            if (it + 1) % Ndump == 0 or it == Nmax - 1:
                # ... (resto del log sin cambios)
                pass # El log se podría añadir aquí

        update_log_callback("Simulación completada. Post-procesando...\n")
        
        # Empaquetar resultados
        results = {'label': params.get('label')}
        if is_2d:
            results.update({'x': x_ho, 'y': y_ho, 'u': U[:num_nodes], 'v': U[num_nodes:]})
            k_1d, tke_1d = get_tke_spectrum_2d(u_final, v_final, Nx, Ny, p_order)
            results.update({'x': x_ho, 'y': y_ho, 'u': u_final, 'v': v_final, 'k_1d': k_1d, 'tke_1d': tke_1d})
        else:
            k, tke = getTKEFFT(x_ho[:-1], U[:-1])
            results.update({'x': x_ho, 'u': U, 'k': k, 'tke': tke})
            
        return results

    except Exception as e:
        update_log_callback(f"ERROR en la simulación '{params.get('label', '')}': {e}\n")
        traceback.print_exc()
        return None

# --- Función Principal del Servidor ---
def server(input, output, session):
    
    simulation_results = reactive.Value([])
    log_text = reactive.Value("Listo para iniciar la simulación.\n")

    def append_log(text):
        log_text.set(log_text() + text)

    @reactive.Effect
    @reactive.event(input.run_manual_button)
    async def handle_manual_run():
        simulation_results.set([])
        log_text.set("")
        
        sgs_params = None
        if input.use_les():
            sgs_params = {
                'model_type': input.sgs_model_type(), 'Cs_min': input.sgs_cs_min(),
                'filter_width_ratio': input.sgs_filter_ratio(), 'avg_type': 'global'
            }
        
        params = {
            'P': input.p_order(), 'VISC': input.visc(), 'INISOL': input.inisol(), 
            'USE_LES': input.use_les(), 'SGS_PARAMS': sgs_params, 'N': input.n_points(),
            'TSIM': input.tsim(), 'DT': input.dt(),
            'label': f"N={input.n_points()}, P={input.p_order()}, Visc={input.visc()}"
        }
        
        # La llamada a la simulación ya no necesita pasar nada relacionado con la barra de progreso
        result = await run_single_simulation(params, append_log)
        if result:
            simulation_results.set([result])

    @reactive.Effect
    @reactive.event(input.run_files_button)
    async def handle_files_run():
        req(input.upload_files())
        simulation_results.set([])
        log_text.set("")
        
        results_list = []
        files = input.upload_files()
        
        for i, file_info in enumerate(files):
            append_log(f"--- Procesando archivo: {file_info['name']} ---\n")
            try:
                with open(file_info["datapath"], "r") as f:
                    document = f.readlines()
                use_les = getValueFromLabel(document, "USE_LES").upper() == "TRUE"
                sgs_params = None
                if use_les:
                    sgs_params = { 'model_type': getValueFromLabel(document, "SGS_MODEL_TYPE"), 'Cs_min': float(getValueFromLabel(document, "SGS_CS_MIN")), 'filter_width_ratio': float(getValueFromLabel(document, "SGS_FILTER_RATIO")), 'avg_type': getValueFromLabel(document, "SGS_AVG_TYPE") }
                params = { 'P': int(getValueFromLabel(document, "P")), 'VISC': float(getValueFromLabel(document, "VISC")), 'INISOL': getValueFromLabel(document, "INISOL"), 'N': int(getValueFromLabel(document, "N")), 'NREF': int(getValueFromLabel(document, "NREF")), 'DT': float(getValueFromLabel(document, "DT")), 'TSIM': float(getValueFromLabel(document, "TSIM")), 'USE_LES': use_les, 'SGS_PARAMS': sgs_params, 'label': file_info['name'] }
                
                # La llamada a la simulación ya no necesita pasar nada relacionado con la barra de progreso
                result = await run_single_simulation(params, append_log)
                if result:
                    results_list.append(result)
            except Exception as e:
                append_log(f"ERROR procesando {file_info['name']}: {e}\n")
        
        simulation_results.set(results_list)

    # --- Definición de los Outputs (sin cambios) ---
    @output
    @render.ui
    def stability_advisor():
        n, p, dt, visc = input.n_points(), input.p_order(), input.dt(), input.visc()
        if n <= 1: return
        u_max, h_e = 1.0, 1.0 / (n - 1)
        dx_eff = h_e / (p + 1.0)
        if dx_eff == 0: return
        cfl_conv = (u_max * dt) / dx_eff
        cfl_diff = (visc * dt) / (dx_eff**2)
        if cfl_conv > 0.5 or cfl_diff > 0.2:
            msg, style = "¡Peligro! Configuración probablemente INESTABLE.", "color: red; font-weight: bold; background-color: #f8d7da; padding: 5px; border-radius: 5px; margin-top: 10px;"
        elif cfl_conv > 0.2 or cfl_diff > 0.1:
            msg, style = "Precaución: Parámetros exigentes.", "color: #856404; font-weight: bold; background-color: #fff3cd; padding: 5px; border-radius: 5px; margin-top: 10px;"
        else:
            msg, style = "Seguro: Configuración probablemente estable.", "color: #155724; font-weight: bold; background-color: #d4edda; padding: 5px; border-radius: 5px; margin-top: 10px;"
        return ui.div(f"Asesor de Estabilidad: {msg}", ui.div(f"(CFL_conv ≈ {cfl_conv:.2f}, CFL_diff ≈ {cfl_diff:.2f})"), style=style)

    @output
    @render.plot
    def main_plot():
        results = simulation_results()
        req(results)
        
        # Detectar si el resultado es 1D o 2D
        is_2d = 'y' in results[0]

        if is_2d:
            fig, ax = plt.subplots(figsize=(7, 6))
            res = results[0] # Mostrar solo el primer resultado 2D
            im, _ = plot_2d_results(ax, ax, res['x'], res['y'], res['u'], res['v']) # Reutilizamos la función
            ax.set_title("Campo de Velocidad 'u'") # Solo mostramos u en el plot principal
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        else: # 1D
            fig, ax = plt.subplots(figsize=(7, 6))
            for res in results:
                ax.plot(res['x'], res['u'], label=res['label'])
            ax.set_title("Solución u(x)")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(True)
            if any(res.get('label') for res in results): ax.legend()
        
        return fig

    @output
    @render.plot
    def spectrum_plot():
        results = simulation_results()
        req(results)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("Espectro de Energía")
        ax.set_xlabel("Número de Onda (k)")
        ax.set_ylabel("Energía")
        
        for res in results:
            # Detectar si el resultado es 1D o 2D
            if 'y' in res: # Es 2D
                if res.get('k_1d') is not None and res.get('tke_1d') is not None:
                    ax.loglog(res['k_1d'], res['tke_1d'], label=f"{res['label']} (2D)")
            else: # Es 1D
                if res.get('k') is not None and res.get('tke') is not None:
                    ax.loglog(res['k'], res['tke'], label=f"{res['label']} (1D)")
        
        ax.grid(True, which="both", ls="--")
        if any(res.get('label') for res in results): ax.legend()
        
        return fig

    @output
    @render.text
    def simulation_log():
        return log_text()