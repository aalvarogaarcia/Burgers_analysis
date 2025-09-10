# tools/analysis/vortex_comparaison.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
from scipy.interpolate import griddata

# Añade la ruta al directorio raíz para poder importar desde 'src' y otros módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Reutilizamos las funciones robustas que ya existen en plot_results
from tools.analysis.plot_results import load_solution_data, compute_vorticity


def main(filepaths_patterns, output_filename="analysis.png", plot_type='vorticity'):
    """
    Función principal que carga resultados 2D y grafica la vorticidad o la magnitud de velocidad.
    """
    filepaths = []
    for pattern in filepaths_patterns:
        filepaths.extend(sorted(glob.glob(pattern)))

    if not filepaths:
        print("Error: No se encontraron archivos que coincidan con los patrones dados.")
        return

    num_files = len(filepaths)
    
    rows = num_files // 3 + (num_files % 3 > 0) 
    cols = min(num_files, 3)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
    
    # --- MODIFICACIÓN: Título principal dinámico ---
    if num_files > 1:
        if plot_type == 'vorticity':
            title_text = 'Análisis Comparativo de Campos de Vorticidad'
        else:
            title_text = 'Análisis Comparativo de Magnitud de Velocidad'
        fig.suptitle(title_text, fontsize=22, y=0.96)
    
    axes_flat = axes.flatten()

    for i, filepath in enumerate(filepaths):
        ax = axes_flat[i]
        
        print(f"Procesando y dibujando: {os.path.basename(filepath)}")
        
        x, y, u, v, nx, ny = load_solution_data(filepath)
        if x is None:
            ax.text(0.5, 0.5, 'Error al cargar datos', ha='center', va='center')
            continue
            
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            ax.text(0.5, 0.5, 'Datos Inestables (NaN/Inf)', ha='center', va='center')
            u[~np.isfinite(u)] = 0; v[~np.isfinite(v)] = 0
            
        resolution_factor = 2
        x_vec = np.linspace(min(x), max(x), nx * resolution_factor)
        y_vec = np.linspace(min(y), max(y), ny * resolution_factor)
        grid_x, grid_y = np.meshgrid(x_vec, y_vec, indexing='ij')
        
        points = np.vstack((x, y)).T
        u_grid = griddata(points, u, (grid_x, grid_y), method='cubic', fill_value=0)
        v_grid = griddata(points, v, (grid_x, grid_y), method='cubic', fill_value=0)
        
        # --- MODIFICACIÓN: Selección de la variable a graficar ---
        if plot_type == 'vorticity':
            plot_variable = compute_vorticity(u_grid, v_grid, x_vec, y_vec)
            v_max = np.nanpercentile(np.abs(plot_variable), 99.8) + 1e-9
            v_min = -v_max
            cmap = 'seismic'
            cbar_label = 'Intensidad de Vorticidad ($s^{-1}$)'
        elif plot_type == 'magnitude':
            plot_variable = np.sqrt(u_grid**2 + v_grid**2)
            v_max = np.nanpercentile(plot_variable, 99.8)
            v_min = np.nanmin(plot_variable)
            cmap = 'viridis' # Mapa de color más adecuado para magnitudes
            cbar_label = 'Magnitud de Velocidad ||V|| (m/s)'
        elif plot_type == 'vector':
            magnitude = np.sqrt(u_grid**2 + v_grid**2)
            v_max = np.nanpercentile(magnitude, 99.8)
            v_min = np.nanmin(magnitude)
            cmap = 'viridis'
            cbar_label = 'Magnitud de Velocidad ||V|| (m/s)'
            im = ax.imshow(magnitude.T, extent=(min(x), max(x), min(y), max(y)), 
                           origin='lower', cmap=cmap, vmin=v_min, vmax=v_max)
            
            # Submuestreo para que el gráfico de vectores sea legible
            skip = max(1, nx // 25) # Apuntar a ~25 flechas por lado
            ax.quiver(grid_x[::skip, ::skip].T, grid_y[::skip, ::skip].T,
                      u_grid[::skip, ::skip].T, v_grid[::skip, ::skip].T,
                      color='white', scale=v_max*30)
        
        else:
            print(f"Error: --plot-type '{plot_type}' no es válido. Use 'vorticity' o 'magnitude'.")
            return

        im = ax.imshow(plot_variable.T, 
                       extent=(min(x_vec), max(x_vec), min(y_vec), max(y_vec)), 
                       origin='lower', cmap=cmap, 
                       vmin=v_min, vmax=v_max)
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=12)
        
        if num_files > 1:
            ax.text(0.05, 0.95, f'({chr(97 + i)})', transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.75))
        
        ax.set_title(os.path.basename(filepath).replace('.txt', ''), fontsize=14)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

    for i in range(num_files, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.92] if num_files > 1 else None)
    plt.savefig(output_filename, dpi=150)
    print(f"\n¡Éxito! Gráfico guardado en: {os.path.abspath(output_filename)}")
    plt.show()

if __name__ == "__main__":
    args = sys.argv[1:]
    
    # --- MODIFICACIÓN: Parseo de argumentos para --plot-type ---
    plot_type_arg = 'vorticity' # Valor por defecto
    if '--plot-type' in args:
        try:
            type_index = args.index('--plot-type')
            plot_type_arg = args[type_index + 1]
            # Eliminar el argumento y su valor de la lista para no confundir al resto del script
            args.pop(type_index)
            args.pop(type_index)
        except (ValueError, IndexError):
            print("Error: El argumento --plot-type necesita un valor ('vorticity', 'magnitude' o 'vector').")
            sys.exit(1)

    if not args:
        print("\nUso: python vortex_comparaison.py \"ruta/a/resultados.txt\" [--plot-type vorticity|magnitude|vector] [nombre_salida.png]")
        sys.exit(1)

    if args[-1].lower().endswith('.png'):
        output_name = args[-1]
        patterns = args[:-1]
    else:
        output_name = "analysis.png"
        patterns = args
        
    main(patterns, output_filename=output_name, plot_type=plot_type_arg)