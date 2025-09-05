# tools/analysis/plot_results.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import os
import glob

# Añade la ruta al directorio raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def load_solution_data(filepath):
    """Carga los datos de un archivo de solución .txt."""
    with open(filepath, 'r') as f:
        document = f.readlines()
    data_lines = ReadBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION")
    if not data_lines:
        print(f"Error: No se encontró el bloque de solución en {filepath}")
        return None, None, None, None, None, None
    data = np.loadtxt(data_lines)
    nx = int(getValueFromLabel(document, "NX"))
    ny = int(getValueFromLabel(document, "NY"))
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], nx, ny

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

def main(filepaths_patterns):
    """Función principal para graficar los resultados de uno o más archivos."""
    
    # Expande los patrones de glob (ej. *.txt) a una lista de archivos
    filepaths = []
    for pattern in filepaths_patterns:
        filepaths.extend(glob.glob(pattern))

    if not filepaths:
        print("Error: No se encontraron archivos que coincidan con los patrones dados.")
        return

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Advertencia: El archivo '{filepath}' no existe. Saltando...")
            continue

        print(f"Procesando: {filepath}")
        x, y, u, v, nx, ny = load_solution_data(filepath)
        if x is None: continue

        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            print("  -> ADVERTENCIA: ¡Resultados inestables detectados (inf/NaN)!")
            u[~np.isfinite(u)] = 0
            v[~np.isfinite(v)] = 0

        # --- CORRECCIÓN EN LA CREACIÓN DE LA REJILLA Y CÁLCULO DE VORTICIDAD ---
        # 1. Crear vectores de coordenadas 1D para la rejilla de visualización
        resolution_factor = 2
        x_vec = np.linspace(min(x), max(x), nx * resolution_factor)
        y_vec = np.linspace(min(y), max(y), ny * resolution_factor)
        
        # 2. Crear la rejilla 2D usando meshgrid
        grid_x, grid_y = np.meshgrid(x_vec, y_vec, indexing='ij')

        # 3. Interpolar los datos a la nueva rejilla
        u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0)
        v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0)
        
        # 4. Calcular cantidades derivadas SOBRE LA REJILLA ESTRUCTURADA
        kinetic_energy = 0.5 * (u_grid**2 + v_grid**2)
        vorticity = compute_vorticity(u_grid, v_grid, x_vec, y_vec)

        # --- Ploteo ---
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle(f'Resultados para: {os.path.basename(filepath)}', fontsize=16)

        im1 = axes[0].imshow(kinetic_energy.T, extent=(min(x_vec), max(x_vec), min(y_vec), max(y_vec)), origin='lower', cmap='viridis')
        axes[0].set_title('Energía Cinética'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_aspect('equal'); fig.colorbar(im1, ax=axes[0])
        
        vort_vmax = np.nanpercentile(np.abs(vorticity), 99.8) + 1e-9
        im2 = axes[1].imshow(vorticity.T, extent=(min(x_vec), max(x_vec), min(y_vec), max(y_vec)), origin='lower', cmap='seismic', vmin=-vort_vmax, vmax=vort_vmax)
        axes[1].set_title('Vorticidad (z)'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y'); axes[1].set_aspect('equal'); fig.colorbar(im2, ax=axes[1])

        skip = max(1, len(x_vec) // 16)
        axes[2].quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip], u_grid[::skip, ::skip], v_grid[::skip, ::skip])
        axes[2].set_title('Campo de Velocidad'); axes[2].set_xlabel('x'); axes[2].set_ylabel('y'); axes[2].set_aspect('equal')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_dir = os.path.dirname(filepath).replace("inputs", "outputs")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        base_filename = os.path.basename(filepath).replace('.txt', '_fields.png')
        output_filename = os.path.join(output_dir, base_filename)
        
        plt.savefig(output_filename, dpi=150)
        print(f"Gráfico guardado en: {output_filename}")
        plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python plot_results.py \"ruta/a/los/resultados/*.txt\"")
        sys.exit(1)
    main(sys.argv[1:])
