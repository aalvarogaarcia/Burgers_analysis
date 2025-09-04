"""
Created on Thu Jun 12 11:03:39 2025

@author: agarm

Modulo visual y de estilo de la UI
"""
# ui.py
from shiny import ui

# Se define la Interfaz de Usuario completa
app_ui = ui.page_fluid(
    ui.h1("Interfaz para Simulación de la Ecuación de Burgers"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.navset_tab(
                # --- Pestaña 1: Simulación 2D ---
                ui.nav_panel("Simulación 2D",
                    ui.h4("Parámetros de Malla y Esquema"),
                    ui.input_numeric("nx", "Nodos en X (NX)", value=33),
                    ui.input_numeric("ny", "Nodos en Y (NY)", value=33),
                    ui.input_slider("p_order", "Orden Polinomial (P)", min=1, max=8, value=4),
                    ui.input_radio_buttons("scheme", "Esquema Numérico",
                                           choices={"fr": "Flux Reconstruction (FR)", "dc": "Diferencias Centradas (DC)"},
                                           selected="fr"),

                    ui.hr(),
                    ui.h4("Parámetros Físicos y Temporales"),
                    ui.input_select("inisol_2d", "Condición Inicial 2D", {
                        "TAYLOR_GREEN": "Vórtice de Taylor-Green",
                        "GAUSSIAN_2D": "Pulso Gaussiano 2D"
                    }),
                    ui.input_numeric("visc", "Viscosidad (VISC)", value=0.005),
                    ui.input_numeric("tsim", "Tiempo de Simulación", value=1.0),
                    ui.input_numeric("dt", "Paso de Tiempo (DT)", value=0.0001),

                    ui.hr(),
                    ui.h4("Forzante de Turbulencia"),
                    ui.input_checkbox("use_forcing", "Activar Forzante", value=False),
                    ui.panel_conditional(
                        "input.use_forcing === true",
                        ui.input_slider("forcing_amplitude", "Amplitud del Forzante", min=0.1, max=10.0, value=1.0, step=0.1),
                        ui.input_slider("forcing_k_max", "Número de Onda Máx. (k_max)", min=1, max=8, value=4, step=1)
                    ),

                    ui.hr(),
                    ui.h4("Modelo de Sub-escala (LES)"),
                    ui.input_checkbox("use_les", "Activar LES", value=False),
                    ui.panel_conditional(
                        "input.use_les === true",
                        ui.input_select("sgs_model_type", "Tipo de Modelo SGS",
                                        choices={"smagorinsky": "Smagorinsky", "vreman": "Vreman"}),
                        ui.panel_conditional(
                            "input.sgs_model_type === 'smagorinsky'",
                            ui.input_slider("sgs_cs_constant", "Constante de Smagorinsky (Cs)", min=0.01, max=0.3, value=0.15, step=0.01)
                        ),
                        ui.panel_conditional(
                            "input.sgs_model_type === 'vreman'",
                            ui.input_slider("sgs_c_vreman", "Constante de Vreman (c_vreman)", min=0.01, max=0.2, value=0.07, step=0.01)
                        )
                    ),

                    ui.hr(),
                    ui.input_action_button("run_2d_button", "Ejecutar Simulación 2D", class_="btn-primary w-100"),
                    
                    # --- NUEVO: Botón para añadir resultado a la comparación ---
                    ui.div(
                        ui.input_action_button("add_to_comparison_button", "Añadir a la Comparación", class_="btn-info w-100 mt-2"),
                        # Este botón solo será visible si hay un resultado válido
                        ui.panel_conditional("output.show_add_button === true")
                    )
                ),

                # --- NUEVA PESTAÑA: Comparación Estadística ---
                ui.nav_panel("Comparación Estadística",
                    ui.h3("Evolución de la Energía Cinética Total"),
                    ui.p("Añade resultados desde la pestaña 'Simulación 2D' para compararlos aquí."),
                    ui.output_plot("comparison_plot", click=True), # Habilitar clics en el gráfico
                    ui.hr(),
                    ui.h4("Detalles del Punto Seleccionado"),
                    ui.output_data_frame("point_details_table"), # Tabla para mostrar detalles
                    ui.hr(),
                    ui.input_action_button("clear_comparison_button", "Limpiar Gráfico", class_="btn-danger mt-2")
                )
            )
        ),
        
        # --- Panel Principal de Resultados ---
        ui.h2("Resultados de la Simulación"),
        ui.output_plot("results_plot_2d"),
        ui.hr(),
        ui.h4("Log de la Simulación"),
        ui.output_text_verbatim("simulation_log", placeholder=True),
    )
)
