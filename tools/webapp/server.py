# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:03:39 2025

@author: agarm

Modulo de acciones y reaciones de la UI
"""
# server.py
from shiny import render, reactive, req, ui
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback
import sys
import os
from scipy.interpolate import griddata

# --- Importaciones de Módulos del Proyecto ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core.mesh import get_2d_cartesian_mesh, get_mesh_ho_2d
from src.utils.misc import FillInitialSolution_2D
from src.core.lagpol import getStandardElementData
from src.core.ode import RK4
from src.core.residual import get_residual_2d, get_residual_2d_dc
from src.core.forcing import generate_deterministic_forcing

# --- Funciones Auxiliares ---
def compute_vorticity(u_grid, v_grid, x_coords_vec, y_coords_vec):
    """
    Calcula la vorticidad (componente z) en una rejilla ESTRUCTURADA.
    Usa los vectores de coordenadas 1D para el cálculo del gradiente.
    """
    # Derivada de u con respecto a y (a lo largo del eje 0 de la matriz)
    dudy = np.gradient(u_grid, y_coords_vec, axis=0)
    
    # Derivada de v con respecto a x (a lo largo del eje 1 de la matriz)
    dvdx = np.gradient(v_grid, x_coords_vec, axis=1)
    
    return dvdx - dudy

def calculate_total_ke(u, v):
    return 0.5 * np.mean(u**2 + v**2)

# --- Guía de Estilos para el Gráfico de Comparación ---
STYLE_GUIDE = {
    "FR-ILES": {"color": "black", "marker": "o", "linestyle": "-"},
    "DC-ILES": {"color": "black", "marker": "s", "linestyle": "--"},
    "FR-Smagorinsky": {"color": "red", "marker": "o", "linestyle": "-"},
    "DC-Smagorinsky": {"color": "red", "marker": "s", "linestyle": "--"},
    "FR-Vreman": {"color": "blue", "marker": "o", "linestyle": "-"},
    "DC-Vreman": {"color": "blue", "marker": "s", "linestyle": "--"},
    "FR-ILES-Forced": {"color": "purple", "marker": "o", "linestyle": "-"},
    "DC-ILES-Forced": {"color": "purple", "marker": "s", "linestyle": "--"},
    "FR-Smagorinsky-Forced": {"color": "orange", "marker": "o", "linestyle": "-"},
    "DC-Smagorinsky-Forced": {"color": "orange", "marker": "s", "linestyle": "--"},
    "FR-Vreman-Forced": {"color": "green", "marker": "o", "linestyle": "-"},
    "DC-Vreman-Forced": {"color": "green", "marker": "s", "linestyle": "--"},
}
DEFAULT_STYLE = {"color": "gray", "marker": "x", "linestyle": ":"}


# --- Función de Simulación ---
async def run_2d_simulation(params, update_log_callback, progress_handler):
    try:
        update_log_callback(f"Iniciando simulación 2D con esquema '{params['SCHEME']}'...\n"); nx, ny, p, v = params['NX'], params['NY'], params['P'], params['VISC']; inisol, dt, tsim = params['INISOL'], params['DT'], params['TSIM']; scheme, use_les, sgs_params = params['SCHEME'], params['USE_LES'], params['SGS_PARAMS']; n_max = int(tsim / dt); dt = tsim / n_max if n_max > 0 else 0; update_log_callback(f"Parámetros: NX={nx}, NY={ny}, P={p}, Esquema={scheme.upper()}, Visc={v}\n");
        if use_les: update_log_callback(f"LES activado: Modelo={sgs_params.get('model_type')}\n")
        if scheme.lower() == "fr":
            lobatto_points, lp_matrix, gp_array = getStandardElementData(p); x_grid, y_grid = get_2d_cartesian_mesh(nx, ny); x_ho, y_ho = get_mesh_ho_2d(x_grid, y_grid, p, lobatto_points); num_nodes = len(x_ho); U = np.zeros(2 * num_nodes); FillInitialSolution_2D(U, x_ho, y_ho, inisol, nx, ny, p, 0); residual_func = get_residual_2d; coords = (x_ho, y_ho)
        elif scheme.lower() == 'dc':
            p_dc = 0; x_coords_dc, y_coords_dc = get_2d_cartesian_mesh(nx, ny); xx, yy = np.meshgrid(x_coords_dc, y_coords_dc); x_flat, y_flat = xx.flatten(), yy.flatten(); num_nodes = nx * ny; U = np.zeros(2 * num_nodes); FillInitialSolution_2D(U, x_flat, y_flat, inisol, nx, ny, p_dc, 0); residual_func = get_residual_2d_dc; coords = (x_flat, y_flat)
        else: raise ValueError("Esquema no reconocido.")
        forcing_term = None
        if params.get('USE_FORCING', False):
            update_log_callback("Generando campo de forzante...\n"); forcing_term = generate_deterministic_forcing(coords, nx, ny, amplitude=params['FORCING_AMPLITUDE'], k_max=params['FORCING_K_MAX'])
        if scheme.lower() == "fr": args = (p, coords, v, (lp_matrix, gp_array), nx, ny, use_les, sgs_params, forcing_term)
        else: args = (coords, v, nx, ny, use_les, sgs_params, forcing_term)
        update_log_callback("Ejecutando bucle temporal...\n")
        for it in range(n_max):
            U = RK4(dt, U, residual_func, *args)
            if np.isnan(U).any(): update_log_callback(f"¡ERROR! Inestabilidad numérica en la iteración {it+1}. Abortando.\n"); return None
            progress_value = (it + 1) / n_max; progress_handler.set(progress_value, detail=f"Iteración {it + 1}/{n_max}")
        update_log_callback("Simulación completada con éxito.\n")
        return {"type": "2D", "x": coords[0], "y": coords[1], "u": U[:num_nodes], "v": U[num_nodes:], "nx": nx, "ny": ny, "params": params}
    except Exception as e:
        update_log_callback(f"ERROR DURANTE LA SIMULACIÓN: {e}\n"); traceback.print_exc(file=sys.stdout); return None

# --- Lógica Principal del Servidor ---
def server(input, output, session):
    simulation_results = reactive.Value(None)
    log_text = reactive.Value("Listo para iniciar la simulación 2D.\n")
    comparison_data = reactive.Value([])
    selected_point_details = reactive.Value(None)

    def append_log(text):
        log_text.set(log_text() + text)

    @reactive.Effect
    @reactive.event(input.run_2d_button)
    async def handle_2d_run():
        simulation_results.set(None); log_text.set(""); selected_point_details.set(None); sgs_params = None
        if input.use_les():
            model_type = input.sgs_model_type(); sgs_params = {'model_type': model_type}
            if model_type == 'smagorinsky': sgs_params['Cs'] = input.sgs_cs_constant()
            elif model_type == 'vreman': sgs_params['c_vreman'] = input.sgs_c_vreman()
        params = {'NX': input.nx(),'NY': input.ny(),'P': input.p_order(),'SCHEME': input.scheme(),'VISC': input.visc(),'INISOL': input.inisol_2d(),'DT': input.dt(),'TSIM': input.tsim(),'USE_LES': input.use_les(),'SGS_PARAMS': sgs_params,'USE_FORCING': input.use_forcing(),'FORCING_AMPLITUDE': input.forcing_amplitude(),'FORCING_K_MAX': input.forcing_k_max()}
        with ui.Progress(min=0, max=1) as p:
            p.set(0, message="Iniciando simulación..."); result = await run_2d_simulation(params, append_log, p)
        if result: simulation_results.set(result)

    @reactive.Effect
    @reactive.event(input.add_to_comparison_button)
    def add_to_comparison():
        res = simulation_results();
        if not res: return
        total_ke = calculate_total_ke(res['u'], res['v']); params = res['params']
        model_key = f"{params['SCHEME'].upper()}"
        if params['USE_LES']: model_key += f"-{params['SGS_PARAMS']['model_type'].capitalize()}"
        else: model_key += "-ILES"
        if params['USE_FORCING']: model_key += "-Forced"
        new_entry = {"model_key": model_key, "time": params['TSIM'], "total_ke": total_ke, "params": params}
        current_data = comparison_data(); current_data.append(new_entry); comparison_data.set(current_data)
        append_log(f"Resultado añadido a la comparación (KE = {total_ke:.4f}).\n")

    @reactive.Effect
    @reactive.event(input.clear_comparison_button)
    def clear_comparison():
        comparison_data.set([]); selected_point_details.set(None); append_log("Gráfico de comparación limpiado.\n")
        
    @reactive.Effect
    @reactive.event(input.comparison_plot_click)
    def handle_plot_click():
        click_data = input.comparison_plot_click();
        if not click_data or not comparison_data(): return
        min_dist = float('inf'); closest_point = None
        domain = click_data['domain']; x_range = domain['right'] - domain['left']; log_y_range = np.log(domain['top']) - np.log(domain['bottom'])
        if x_range == 0 or log_y_range == 0: return
        for point in comparison_data():
            dist_x_norm = ((point['time'] - click_data['x']) / x_range)**2
            log_y_point = np.log(point['total_ke']); log_y_click = np.log(click_data['y'])
            dist_y_norm = ((log_y_point - log_y_click) / log_y_range)**2
            dist = dist_x_norm + dist_y_norm
            if dist < min_dist: min_dist = dist; closest_point = point
        if closest_point: selected_point_details.set(closest_point)

    @output
    @render.plot
    def results_plot_2d():
        results = simulation_results(); req(results)
        if results.get("type") == "2D":
            x, y, u, v, nx, ny = results['x'], results['y'], results['u'], results['v'], results['nx'], results['ny']
            
            # --- CORRECCIÓN EN LA CREACIÓN DE LA REJILLA Y CÁLCULO DE VORTICIDAD ---
            # 1. Crear vectores de coordenadas 1D para la rejilla de visualización
            x_vec = np.linspace(min(x), max(x), nx * 2)
            y_vec = np.linspace(min(y), max(y), ny * 2)
            
            # 2. Crear la rejilla 2D usando meshgrid, que es más explícito
            grid_x, grid_y = np.meshgrid(x_vec, y_vec, indexing='ij')

            # 3. Interpolar los datos (desestructurados) a la nueva rejilla (estructurada)
            u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0)
            v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0)
            
            # 4. Calcular cantidades derivadas SOBRE LA REJILLA ESTRUCTURADA
            kinetic_energy = 0.5 * (u_grid**2 + v_grid**2)
            vorticity = compute_vorticity(u_grid, v_grid, x_vec, y_vec) # Usar los vectores 1D
            
            # --- El resto del ploteo sigue igual ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 5)); fig.suptitle('Resultados de la Simulación 2D', fontsize=16)
            
            im1 = axes[0].imshow(kinetic_energy.T, extent=(min(x_vec), max(x_vec), min(y_vec), max(y_vec)), origin='lower', cmap='viridis')
            axes[0].set_title('Energía Cinética'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_aspect('equal'); fig.colorbar(im1, ax=axes[0])
            
            vort_vmax = np.nanpercentile(np.abs(vorticity), 99.8) + 1e-9
            im2 = axes[1].imshow(vorticity.T, extent=(min(x_vec), max(x_vec), min(y_vec), max(y_vec)), origin='lower', cmap='seismic', vmin=-vort_vmax, vmax=vort_vmax)
            axes[1].set_title('Vorticidad (z)'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y'); axes[1].set_aspect('equal'); fig.colorbar(im2, ax=axes[1])
            
            skip = max(1, len(x_vec) // 16)
            axes[2].quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip], u_grid[::skip, ::skip], v_grid[::skip, ::skip])
            axes[2].set_title('Campo de Velocidad'); axes[2].set_xlabel('x'); axes[2].set_ylabel('y'); axes[2].set_aspect('equal')
            plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig

    @output
    @render.text
    def simulation_log(): return log_text()
        
    @output
    @render.ui
    def show_add_button(): return simulation_results() is not None

    @output
    @render.plot
    def comparison_plot():
        data = comparison_data();
        fig, ax = plt.subplots(figsize=(10, 7))
        if not data:
            ax.text(0.5, 0.5, "Añade resultados desde la pestaña 'Simulación'", ha='center', va='center', fontsize=12, color='gray'); ax.set_xlabel("Tiempo de Simulación (s)"); ax.set_ylabel("Energía Cinética Total"); return fig
        grouped_data = {};
        for point in data:
            key = point['model_key']
            if key not in grouped_data: grouped_data[key] = []
            grouped_data[key].append(point)
        for key, points in grouped_data.items():
            style = STYLE_GUIDE.get(key, DEFAULT_STYLE)
            sorted_points = sorted(points, key=lambda p: p['time'])
            times = [p['time'] for p in sorted_points]; kes = [p['total_ke'] for p in sorted_points]
            ax.plot(times, kes, label=key, markerfacecolor='white', markersize=7, **style)
        ax.set_title("Comparación de la Evolución de la Energía Cinética", fontsize=14); ax.set_xlabel("Tiempo de Simulación (s)", fontsize=12); ax.set_ylabel("Energía Cinética Total", fontsize=12); ax.set_yscale('log'); ax.grid(True, which="both", ls="--"); ax.legend(fontsize='medium'); return fig
        
    @output
    @render.data_frame
    def point_details_table():
        details = selected_point_details();
        if not details: return pd.DataFrame({"Parámetro": ["-"], "Valor": ["-"]})
        params = details['params']; display_data = {"Energía Cinética Total": f"{details['total_ke']:.5f}","Modelo Base": details['model_key'],"Tiempo": params['TSIM'],"Viscosidad": params['VISC'],"DT": params['DT'],"Malla": f"{params['NX']}x{params['NY']}", "Orden (P)": params['P'],"Cond. Inicial": params['INISOL']}
        if params['USE_LES']:
            sgs = params['SGS_PARAMS']; display_data["Modelo LES"] = sgs['model_type']
            if sgs['model_type'] == 'smagorinsky': display_data["Cs"] = sgs['Cs']
            if sgs['model_type'] == 'vreman': display_data["c_vreman"] = sgs['c_vreman']
        if params['USE_FORCING']:
            display_data["Forzante"] = "Activado"; display_data["Amplitud Forzante"] = params['FORCING_AMPLITUDE']; display_data["k_max Forzante"] = params['FORCING_K_MAX']
        return pd.DataFrame(list(display_data.items()), columns=["Parámetro", "Valor"])
