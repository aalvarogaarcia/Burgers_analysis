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
def compute_vorticity(u, v, x_grid, y_grid):
    dudy = np.gradient(u, y_grid[:, 0], axis=0)
    dvdx = np.gradient(v, x_grid[0, :], axis=1)
    return dvdx - dudy

def calculate_total_ke(u, v):
    """Calcula la energía cinética total promediada en el dominio."""
    return 0.5 * np.mean(u**2 + v**2)

# --- Función de Simulación (sin cambios) ---
async def run_2d_simulation(params, update_log_callback, progress_handler):
    # ... (Esta función es la misma que la versión anterior y no necesita cambios) ...
    try:
        update_log_callback(f"Iniciando simulación 2D con esquema '{params['SCHEME']}'...\n")
        nx, ny, p, v = params['NX'], params['NY'], params['P'], params['VISC']
        inisol, dt, tsim = params['INISOL'], params['DT'], params['TSIM']
        scheme, use_les, sgs_params = params['SCHEME'], params['USE_LES'], params['SGS_PARAMS']
        n_max = int(tsim / dt); dt = tsim / n_max if n_max > 0 else 0
        update_log_callback(f"Parámetros: NX={nx}, NY={ny}, P={p}, Esquema={scheme.upper()}, Visc={v}\n")
        if use_les: update_log_callback(f"LES activado: Modelo={sgs_params.get('model_type')}\n")
        if scheme.lower() == "fr":
            lobatto_points, lp_matrix, gp_array = getStandardElementData(p)
            x_grid, y_grid = get_2d_cartesian_mesh(nx, ny)
            x_ho, y_ho = get_mesh_ho_2d(x_grid, y_grid, p, lobatto_points)
            num_nodes = len(x_ho); U = np.zeros(2 * num_nodes)
            FillInitialSolution_2D(U, x_ho, y_ho, inisol, nx, ny, p, 0)
            residual_func = get_residual_2d; coords = (x_ho, y_ho)
        elif scheme.lower() == 'dc':
            p_dc = 0; x_coords_dc, y_coords_dc = get_2d_cartesian_mesh(nx, ny)
            xx, yy = np.meshgrid(x_coords_dc, y_coords_dc); x_flat, y_flat = xx.flatten(), yy.flatten()
            num_nodes = nx * ny; U = np.zeros(2 * num_nodes)
            FillInitialSolution_2D(U, x_flat, y_flat, inisol, nx, ny, p_dc, 0)
            residual_func = get_residual_2d_dc; coords = (x_flat, y_flat)
        else: raise ValueError("Esquema no reconocido.")
        forcing_term = None
        if params.get('USE_FORCING', False):
            update_log_callback("Generando campo de forzante...\n")
            forcing_term = generate_deterministic_forcing(coords, nx, ny, amplitude=params['FORCING_AMPLITUDE'], k_max=params['FORCING_K_MAX'])
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
    # --- Almacenamiento de Estado ---
    simulation_results = reactive.Value(None)
    log_text = reactive.Value("Listo para iniciar la simulación 2D.\n")
    # NUEVO: Valor reactivo para guardar los datos de la comparación
    comparison_data = reactive.Value([])
    # NUEVO: Valor reactivo para guardar los detalles del punto clickeado
    selected_point_details = reactive.Value(None)

    def append_log(text):
        log_text.set(log_text() + text)

    # --- Manejadores de Eventos ---
    @reactive.Effect
    @reactive.event(input.run_2d_button)
    async def handle_2d_run():
        # ... (lógica para ejecutar la simulación, sin cambios) ...
        simulation_results.set(None); log_text.set(""); selected_point_details.set(None)
        sgs_params = None
        if input.use_les():
            model_type = input.sgs_model_type(); sgs_params = {'model_type': model_type}
            if model_type == 'smagorinsky': sgs_params['Cs'] = input.sgs_cs_constant()
            elif model_type == 'vreman': sgs_params['c_vreman'] = input.sgs_c_vreman()
        params = {'NX': input.nx(),'NY': input.ny(),'P': input.p_order(),'SCHEME': input.scheme(),'VISC': input.visc(),'INISOL': input.inisol_2d(),'DT': input.dt(),'TSIM': input.tsim(),'USE_LES': input.use_les(),'SGS_PARAMS': sgs_params,'USE_FORCING': input.use_forcing(),'FORCING_AMPLITUDE': input.forcing_amplitude(),'FORCING_K_MAX': input.forcing_k_max()}
        with ui.Progress(min=0, max=1) as p:
            p.set(0, message="Iniciando simulación...")
            result = await run_2d_simulation(params, append_log, p)
        if result: simulation_results.set(result)

    # --- NUEVOS MANEJADORES ---
    @reactive.Effect
    @reactive.event(input.add_to_comparison_button)
    def add_to_comparison():
        res = simulation_results()
        if not res: return
        
        # Calcular KE y preparar datos para añadir
        total_ke = calculate_total_ke(res['u'], res['v'])
        params = res['params']
        
        # Crear una etiqueta única para agrupar simulaciones
        model_key = f"{params['SCHEME'].upper()}"
        if params['USE_LES']: model_key += f"-{params['SGS_PARAMS']['model_type'].capitalize()}"
        else: model_key += "-ILES"
        if params['USE_FORCING']: model_key += "-Forced"
        
        new_entry = {
            "model_key": model_key,
            "time": params['TSIM'],
            "total_ke": total_ke,
            "params": params # Guardar todos los parámetros para la tabla
        }
        
        # Añadir a la lista existente
        current_data = comparison_data()
        current_data.append(new_entry)
        comparison_data.set(current_data)
        append_log(f"Resultado añadido a la comparación (KE = {total_ke:.4f}).\n")

    @reactive.Effect
    @reactive.event(input.clear_comparison_button)
    def clear_comparison():
        comparison_data.set([])
        selected_point_details.set(None)
        append_log("Gráfico de comparación limpiado.\n")
        
    @reactive.Effect
    @reactive.event(input.comparison_plot_click)
    def handle_plot_click():
        click_data = input.comparison_plot_click()
        if not click_data: return

        # Buscar el punto más cercano en los datos de comparación
        min_dist = float('inf')
        closest_point = None
        for point in comparison_data():
            dist = (point['time'] - click_data['x'])**2 + (point['total_ke'] - click_data['y'])**2
            if dist < min_dist:
                min_dist = dist
                closest_point = point
        
        if closest_point:
            selected_point_details.set(closest_point)

    # --- Salidas de la Interfaz ---
    @output
    @render.plot
    def results_plot_2d():
        # ... (sin cambios) ...
        results = simulation_results(); req(results)
        if results.get("type") == "2D":
            x, y, u, v, nx, ny = results['x'], results['y'], results['u'], results['v'], results['nx'], results['ny']
            grid_x, grid_y = np.mgrid[min(x):max(x):complex(nx*2), min(y):max(y):complex(ny*2)]
            u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0); v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0)
            kinetic_energy = 0.5 * (u_grid**2 + v_grid**2); vorticity = compute_vorticity(u_grid, v_grid, grid_x, grid_y)
            fig, axes = plt.subplots(1, 3, figsize=(22, 6)); fig.suptitle('Resultados de la Simulación 2D', fontsize=16)
            im1 = axes[0].imshow(kinetic_energy.T, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), origin='lower', cmap='viridis'); axes[0].set_title('Energía Cinética'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_aspect('equal'); fig.colorbar(im1, ax=axes[0])
            vort_vmax = np.nanpercentile(np.abs(vorticity), 99.5); im2 = axes[1].imshow(vorticity.T, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), origin='lower', cmap='seismic', vmin=-vort_vmax, vmax=vort_vmax); axes[1].set_title('Vorticidad (z)'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y'); axes[1].set_aspect('equal'); fig.colorbar(im2, ax=axes[1])
            skip = max(1, (nx*2) // 16); axes[2].quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip], u_grid[::skip, ::skip], v_grid[::skip, ::skip]); axes[2].set_title('Campo de Velocidad'); axes[2].set_xlabel('x'); axes[2].set_ylabel('y'); axes[2].set_aspect('equal')
            plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig

    @output
    @render.text
    def simulation_log():
        return log_text()
        
    # --- NUEVAS SALIDAS ---
    @output
    @render.ui
    def show_add_button():
        # Lógica para mostrar el botón solo cuando hay un resultado válido
        return simulation_results() is not None

    @output
    @render.plot
    def comparison_plot():
        data = comparison_data()
        if not data:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No hay datos para comparar", ha='center', va='center'); return fig

        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Agrupar datos por la clave del modelo
        grouped_data = {}
        for point in data:
            key = point['model_key']
            if key not in grouped_data: grouped_data[key] = []
            grouped_data[key].append(point)
            
        # Graficar puntos y líneas para cada grupo
        for key, points in grouped_data.items():
            # Ordenar por tiempo para dibujar la línea correctamente
            sorted_points = sorted(points, key=lambda p: p['time'])
            times = [p['time'] for p in sorted_points]
            kes = [p['total_ke'] for p in sorted_points]
            ax.plot(times, kes, '-o', label=key, markerfacecolor='white')

        ax.set_title("Comparación de la Evolución de la Energía Cinética")
        ax.set_xlabel("Tiempo de Simulación (s)")
        ax.set_ylabel("Energía Cinética Total")
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--")
        ax.legend()
        return fig
        
    @output
    @render.data_frame
    def point_details_table():
        details = selected_point_details()
        if not details:
            return pd.DataFrame({"Parámetro": ["-"], "Valor": ["Haz clic en un punto del gráfico para ver sus detalles."]})
        
        # Formatear los parámetros para una visualización clara
        params = details['params']
        display_data = {
            "Energía Cinética Total": f"{details['total_ke']:.5f}",
            "Modelo Base": details['model_key'],
            "Tiempo de Simulación": params['TSIM'],
            "Viscosidad": params['VISC'],
            "Paso de Tiempo (DT)": params['DT'],
            "Malla": f"{params['NX']}x{params['NY']}",
            "Orden Polinomial (P)": params['P'],
            "Cond. Inicial": params['INISOL'],
        }
        if params['USE_LES']:
            sgs = params['SGS_PARAMS']
            display_data["Modelo LES"] = sgs['model_type']
            if sgs['model_type'] == 'smagorinsky': display_data["Cs"] = sgs['Cs']
            if sgs['model_type'] == 'vreman': display_data["c_vreman"] = sgs['c_vreman']
        
        if params['USE_FORCING']:
            display_data["Forzante"] = "Activado"
            display_data["Amplitud Forzante"] = params['FORCING_AMPLITUDE']
            display_data["k_max Forzante"] = params['FORCING_K_MAX']

        return pd.DataFrame(list(display_data.items()), columns=["Parámetro", "Valor"])
