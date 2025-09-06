# tools/analysis/plot_comparison_grid.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import math
from scipy.interpolate import griddata

# Añade la ruta al directorio raíz para poder importar desde 'src' y otros módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Reutilizamos las funciones robustas que ya existen en plot_results
from tools.analysis.plot_results import load_solution_data, compute_vorticity

def main(filepaths_patterns, output_filename="comparison_grid.png"):
    """
    Función principal que carga múltiples resultados 2D y los grafica
    en una única figura con una rejilla de subplots etiquetados en una sola fila.
    """
    filepaths = []
    for pattern in filepaths_patterns:
        filepaths.extend(sorted(glob.glob(pattern))) # Ordenar para consistencia

    if not filepaths:
        print("Error: No se encontraron archivos que coincidan con los patrones dados.")
        return

    num_files = len(filepaths)
    
    # --- CORRECCIÓN DE LAYOUT: Forzar una sola fila ---
    rows = 1
    cols = num_files
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
    fig.suptitle('Análisis Comparativo de Campos de Vorticidad', fontsize=22, y=0.96)
    
    axes_flat = axes.flatten()

    for i, filepath in enumerate(filepaths):
        ax = axes_flat[i]
        
        print(f"Procesando y dibujando: {os.path.basename(filepath)}")
        
        # --- Carga y procesamiento de datos ---
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
        
        vorticity = compute_vorticity(u_grid, v_grid, x_vec, y_vec)
        
        # --- Ploteo en el subplot correspondiente ---
        vort_vmax = np.nanpercentile(np.abs(vorticity), 99.8) + 1e-9
        im = ax.imshow(vorticity.T, 
                       extent=(min(x_vec), max(x_vec), min(y_vec), max(y_vec)), 
                       origin='lower', cmap='seismic', 
                       vmin=-vort_vmax, vmax=vort_vmax)
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # --- CORRECCIÓN DE ETIQUETA: Añadir fondo para legibilidad ---
        ax.text(0.05, 0.95, f'({chr(97 + i)})', transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.75))
        
        ax.set_title(os.path.basename(filepath).replace('.txt', ''), fontsize=14)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

    for i in range(num_files, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(output_filename, dpi=150)
    print(f"\n¡Éxito! Gráfico comparativo guardado en: {os.path.abspath(output_filename)}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python plot_comparison_grid.py \"ruta/a/resultados1.txt\" \"ruta/a/resultados2.txt\" ...")
        sys.exit(1)
    
    if sys.argv[-1].lower().endswith('.png'):
        output_name = sys.argv[-1]
        patterns = sys.argv[1:-1]
    else:
        output_name = "comparison_grid.png"
        patterns = sys.argv[1:]
        
    main(patterns, output_filename=output_name)