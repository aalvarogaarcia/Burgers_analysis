# tools/analysis/plot_results.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import os

# Añade la ruta al directorio raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.randw import getValueFromLabel, ReadBlockData

def load_solution_data(filepath):
    """Carga los datos de un archivo de solución .txt."""
    # ... (esta función no necesita cambios) ...
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

def compute_vorticity(u, v, x_grid, y_grid):
    """Calcula la vorticidad (componente z) en una rejilla estructurada."""
    dudy = np.gradient(u, y_grid[:, 0], axis=0)
    dvdx = np.gradient(v, x_grid[0, :], axis=1)
    return dvdx - dudy

def main(filepaths):
    """Función principal para graficar los resultados de uno o más archivos."""
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Advertencia: El archivo '{filepath}' no existe. Saltando...")
            continue

        print(f"Procesando: {filepath}")
        x, y, u, v, nx, ny = load_solution_data(filepath)
        if x is None: continue

        # --- MEJORA 1: DETECCIÓN DE INESTABILIDAD ---
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            print("  -> ADVERTENCIA: ¡Resultados inestables detectados (inf/NaN)!")
            # Reemplazar valores no finitos para poder graficar algo
            u[~np.isfinite(u)] = 0
            v[~np.isfinite(v)] = 0

        # Interpolar a una rejilla más fina para una visualización suave
        grid_x, grid_y = np.mgrid[min(x):max(x):complex(nx*2), min(y):max(y):complex(ny*2)]
        u_grid = griddata((x, y), u, (grid_x, grid_y), method='cubic', fill_value=0)
        v_grid = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=0)
        
        # Calcular cantidades derivadas
        kinetic_energy = 0.5 * (u_grid**2 + v_grid**2)
        vorticity = compute_vorticity(u_grid, v_grid, grid_x, grid_y)

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle(f'Resultados para: {os.path.basename(filepath)}', fontsize=16)

        # Gráfico de Energía Cinética con rango de color robusto
        ke_vmax = np.nanpercentile(kinetic_energy, 99.5) # Ignorar el 0.5% de valores más altos
        im1 = axes[0].imshow(kinetic_energy.T, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), origin='lower', cmap='viridis', vmax=ke_vmax)
        axes[0].set_title('Energía Cinética')
        axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
        fig.colorbar(im1, ax=axes[0], label='KE')
        
        # Gráfico de Vorticidad con rango de color robusto
        vort_vmax = np.nanpercentile(np.abs(vorticity), 99.5)
        im2 = axes[1].imshow(vorticity.T, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), origin='lower', cmap='seismic', vmin=-vort_vmax, vmax=vort_vmax)
        axes[1].set_title('Vorticidad (z)')
        axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
        fig.colorbar(im2, ax=axes[1], label='Vorticidad')

        # Gráfico de Campo de Velocidad
        skip = max(1, nx // 10)
        axes[2].quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip], u_grid[::skip, ::skip], v_grid[::skip, ::skip], color='k')
        axes[2].set_title('Campo de Velocidad')
        axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
        axes[2].set_aspect('equal')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # --- MEJORA 2: RUTA DE SALIDA CORRECTA ---
        # Construir la ruta de salida reemplazando 'inputs' por 'outputs'
        output_dir = os.path.dirname(filepath).replace("inputs", "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_filename = os.path.basename(filepath).replace('.txt', '_fields.png')
        output_filename = os.path.join(output_dir, base_filename)
        
        plt.savefig(output_filename, dpi=150)
        print(f"Gráfico guardado en: {output_filename}")
        plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python plot_results.py <archivo1.txt> <archivo2.txt> ...")
        sys.exit(1)
    main(sys.argv[1:])