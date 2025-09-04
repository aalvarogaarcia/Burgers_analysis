"""
Created on Thu Jun 12 11:03:39 2025

@author: agarm

Modulo visual y de estilo de la UI
"""
# ui.py
from shiny import ui

# Se define la Interfaz de Usuario completa con una estructura de pestañas principal
app_ui = ui.page_fluid(
    ui.h2("Análisis Numérico Interactivo de la Ecuación de Burgers"),
    
    # --- PESTAÑAS PRINCIPALES DE NAVEGACIÓN ---
    ui.navset_card_tab(
        # --- PESTAÑA 1: SIMULACIÓN ---
        ui.nav_panel("🧪 Simulación",
            ui.layout_sidebar(
                ui.sidebar(
                    # Los controles de la simulación 2D van aquí
                    ui.h4("Parámetros de Simulación 2D"),
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
                        ui.input_slider("forcing_amplitude", "Amplitud", min=0.1, max=10.0, value=1.0, step=0.1),
                        ui.input_slider("forcing_k_max", "k_max", min=1, max=8, value=4, step=1)
                    ),
                    ui.hr(),
                    ui.h4("Modelo de Sub-escala (LES)"),
                    ui.input_checkbox("use_les", "Activar LES", value=False),
                    ui.panel_conditional(
                        "input.use_les === true",
                        ui.input_select("sgs_model_type", "Tipo de Modelo SGS",
                                        choices={"smagorinsky": "Smagorinsky", "vreman": "Vreman"}),
                        ui.panel_conditional("input.sgs_model_type === 'smagorinsky'",
                            ui.input_slider("sgs_cs_constant", "Constante Cs", min=0.01, max=0.3, value=0.15, step=0.01)),
                        ui.panel_conditional("input.sgs_model_type === 'vreman'",
                            ui.input_slider("sgs_c_vreman", "Constante c_vreman", min=0.01, max=0.2, value=0.07, step=0.01))
                    ),
                    ui.hr(),
                    ui.input_action_button("run_2d_button", "Ejecutar Simulación", class_="btn-primary w-100"),
                    ui.div(
                        ui.input_action_button("add_to_comparison_button", "Añadir a la Comparación", class_="btn-info w-100 mt-2"),
                        ui.panel_conditional("output.show_add_button === true")
                    )
                ),
                # Panel principal para los resultados de la simulación
                ui.h3("Resultados de la Simulación"),
                ui.output_plot("results_plot_2d"),
                ui.hr(),
                ui.h4("Log de la Simulación"),
                ui.output_text_verbatim("simulation_log", placeholder=True),
            )
        ),
        
        # --- PESTAÑA 2: COMPARACIÓN ---
        ui.nav_panel("📊 Comparación",
            ui.h3("Análisis Comparativo de Energía Cinética"),
            ui.p("Añade resultados desde la pestaña 'Simulación' para compararlos. Haz clic en un punto del gráfico para ver sus detalles."),
            ui.row(
                ui.column(8, ui.output_plot("comparison_plot", click=True)),
                ui.column(4, 
                    ui.h5("Detalles del Punto Seleccionado"),
                    ui.output_data_frame("point_details_table"),
                    ui.input_action_button("clear_comparison_button", "Limpiar Gráfico", class_="btn-danger mt-3 w-100")
                )
            )
        )
    )
)
