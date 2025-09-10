# tools/analysis/plot_convergence.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import re

# --- Importar Funciones del Repositorio ---
# Se añade la ruta raíz del proyecto para que el script pueda encontrar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.randw import getValueFromLabel, GetMeshAndSolution
from src.core.mesh import get_mesh_1d

def getL2Norm_from_consecutive_meshes(document_coarse, document_fine):
    """
    Calcula la norma L2 entre dos soluciones de mallas consecutivas,
    basado en la lógica original de tu 'compute_convergence.py'.
    """
    # Extraer parámetros de los documentos
    pa = int(getValueFromLabel(document_coarse, "P"))
    pb = int(getValueFromLabel(document_fine, "P"))
    Na = int(getValueFromLabel(document_coarse, "N"))
    Nb = int(getValueFromLabel(document_fine, "N"))

    # Condición de validez: mismo polinomio y la malla 'b' debe ser aproximadamente el doble de 'a'
    if pa != pb or not (Nb == 2 * Na or Nb == 2 * Na -1):
        # print(f"  -> Advertencia: Par de mallas no consecutivas (N={Na}, N={Nb}). Saltando.")
        return 0., 0.
        
    xsa, usa = GetMeshAndSolution(document_coarse)
    xsb, usb = GetMeshAndSolution(document_fine)
    
    if xsa.size == 0 or xsb.size == 0:
        return 0., 0.

    # Interpolar la solución más fina (b) a los puntos de la malla gruesa (a) para comparar
    usb_interpolated = np.interp(xsa, xsb, usb)
    
    # Calcular la norma L2 de la diferencia
    delta = usa - usb_interpolated
    l2norm = np.sqrt(np.mean(delta**2))
    
    # Los grados de libertad son los de la malla gruesa
    dof = len(xsa)

    return dof, l2norm

def extract_params_from_filename(filepath):
    """Extrae 'p' y 'n' del nombre del archivo."""
    basename = os.path.basename(filepath)
    p_match = re.search(r'_p(\d+)_', basename)
    n_match = re.search(r'_n(\d+)\.txt', basename)
    p = int(p_match.group(1)) if p_match else 0
    n = int(n_match.group(1)) if n_match else 0
    return p, n

def main(input_dir):
    """
    Función principal que procesa un directorio, calcula errores de convergencia
    y genera la gráfica final.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: La ruta proporcionada no es un directorio válido: {input_dir}")
        return

    all_files = glob.glob(os.path.join(input_dir, "conv_1d_p*_n*.txt"))
    families = {}
    for f in all_files:
        p, _ = extract_params_from_filename(f)
        if p not in families: families[p] = []
        families[p].append(f)

    if not families:
        print(f"No se encontraron archivos de resultados en '{input_dir}'")
        return

    # --- Inicio de la Gráfica ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for p, files in sorted(families.items()):
        print(f"\nProcesando familia P={p}...")
        sorted_files = sorted(files, key=lambda f: extract_params_from_filename(f)[1])
        
        if len(sorted_files) < 2:
            print(f"  -> Se necesitan al menos 2 mallas para el análisis de P={p}. Saltando.")
            continue

        dofs_inv, errors = [], []

        # Comparar mallas consecutivas (n, 2n), (2n, 4n), etc.
        for i in range(len(sorted_files) - 1):
            coarse_file = sorted_files[i]
            fine_file = sorted_files[i+1]
            
            with open(coarse_file, 'r') as f: doc_coarse = f.readlines()
            with open(fine_file, 'r') as f: doc_fine = f.readlines()

            dof, error = getL2Norm_from_consecutive_meshes(doc_coarse, doc_fine)
            
            if dof > 0 and not np.isnan(error):
                dofs_inv.append(1.0 / dof)
                errors.append(error)
                n1, n2 = extract_params_from_filename(coarse_file)[1], extract_params_from_filename(fine_file)[1]
                print(f"  -> Error L2 (n={n1} vs n={n2}): {error:.4e}")

        if dofs_inv:
            line, = ax.loglog(dofs_inv, errors, 'o-', label=f'P={p} (medido)')
            
            # Graficar pendiente teórica O(h^(p+1))
            C = errors[0] / (dofs_inv[0]**(p + 1))
            dof_trend = np.array([dofs_inv[0], dofs_inv[-1]])
            error_trend = C * dof_trend**(p + 1)
            ax.loglog(dof_trend, error_trend, '--', color=line.get_color(), label=f'~dof$^{{-({p+1})}}$ (teórico)')

    # --- Configuración final de la gráfica ---
    ax.set_xlabel('1 / Grados de Libertad (dof)')
    ax.set_ylabel('Error en Norma L2')
    ax.set_title('Análisis de Convergencia del Solver FR')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    output_filename = "analisis_convergencia_solver_fr.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📈 Gráfica de convergencia guardada en: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nUso: python tools/analysis/plot_convergence.py <ruta_al_directorio_de_resultados>")
        print("Ejemplo: python tools/analysis/plot_convergence.py data/outputs/convergence_study_1d/")
        sys.exit(1)
    
    results_directory = sys.argv[1]
    main(results_directory)