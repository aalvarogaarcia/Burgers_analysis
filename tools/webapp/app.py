# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:03:40 2025

@author: agarm

Lanzamiento App
"""


# app.py
from shiny import App

# Importar la UI definida en ui.py
from ui import app_ui

# Importar la lógica del servidor definida en server.py
from server import server

# Crear la aplicación Shiny
app = App(app_ui, server)
