"""
Created on Thu Jun 12 11:03:39 2025

@author: agarm

Modulo visual y de estilo de la UI
"""

# ui.py
from shiny import ui

# Definimos la Interfaz de Usuario como una variable que exportaremos
app_ui = ui.page_fluid(
    # Título general de la aplicación
    ui.h1("Interfaz para Simulación de la Ecuación de Burgers"),
    
    # Corregido: La estructura es ui.layout_sidebar(ui.sidebar(...), main_content...)
    ui.layout_sidebar(
        # Corregido: Se usa ui.sidebar() para definir el panel lateral.
        # Todos los controles de entrada van dentro de este componente.
        ui.sidebar(
            ui.navset_tab(
                # --- Pestaña 1: Entrada Manual ---
                ui.nav_panel("Entrada Manual",
                    ui.h4("Parámetros de Simulación"),
                    ui.input_slider("p_order", "Orden Polinomial (P)", min=1, max=8, value=2),
                    ui.input_slider("visc", "Viscosidad (VISC)", min=0.001, max=0.1, value=0.01, step=0.001),
                    ui.input_slider("n_points", "Puntos de Malla (N)", min=50, max=500, value=100, step=10),
                    ui.input_slider("tsim", "Tiempo de Simulación", min=0.1, max=5.0, value=1.0, step=0.1),
                    ui.input_slider("dt", "Paso de Tiempo (DT)", min=1e-5, max=1e-3, value=1e-4, step=1e-5),
                    
                    ui.output_ui("stability_advisor"),
                    
                    ui.input_select("inisol", "Condición Inicial", {
                        "SINE": "Seno",
                        "GAUSSIAN": "Gaussiana",
                        "SQUARE": "Onda Cuadrada",
                        "TURBULENT": "Turbulenta"
                    }),
                    
                    ui.hr(),
                    ui.input_checkbox("use_les", "Activar LES", value=False),
                    
                    # Panel condicional que solo aparece si la casilla LES está marcada
                    ui.panel_conditional(
                        "input.use_les === true", # Condición JavaScript
                        ui.h5("Parámetros SGS"),
                        ui.input_select("sgs_model_type", "Modelo SGS", 
                                        {"smagorinsky_dynamic": "Smagorinsky Dinámico"}),
                        ui.input_slider("sgs_cs_min", "Valor Mínimo de Cs", min=0.0, max=0.25, value=0.01, step=0.01),
                        ui.input_slider("sgs_filter_ratio", "Razón del Filtro (Δ̂/Δ)", min=1.5, max=3.0, value=2.0, step=0.1)
                    ),
                    
                    ui.hr(),
                    # Botón para ejecutar la simulación con los parámetros manuales
                    ui.input_action_button("run_manual_button", "Ejecutar Simulación Manual", class_="btn-primary w-100"),
                ),
                
                # --- Pestaña 2: Cargar Archivos ---
                ui.nav_panel("Cargar Archivos",
                    ui.h4("Análisis desde Archivos"),
                    # Widget para subir archivos de texto
                    ui.input_file("upload_files", "Selecciona uno o más archivos .txt",
                                  accept=[".txt"],  # Aceptar solo archivos .txt
                                  multiple=True),   # Permitir múltiples archivos
                    
                    ui.hr(),
                    # Botón para ejecutar la simulación desde los archivos cargados
                    ui.input_action_button("run_files_button", "Ejecutar desde Archivos", class_="btn-success w-100"),
                ),
            )
        ), # Fin de ui.sidebar
        
        # Corregido: El contenido del panel principal va directamente como los siguientes argumentos
        # de ui.layout_sidebar, sin el contenedor ui.panel_main.
        ui.h2("Resultados"),
        ui.output_plot("main_plot"),
        ui.output_plot("spectrum_plot"),
        ui.h4("Log de la Simulación"),
        ui.output_text_verbatim("simulation_log", placeholder=True),
    )
)