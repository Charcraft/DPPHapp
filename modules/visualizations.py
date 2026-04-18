"""
Módulo de Visualizaciones - VERSIÓN MOCK (Simulación)
"""
import plotly.graph_objects as go

def plot_histogram(*args, **kwargs):
    fig = go.Figure()
    fig.update_layout(title="Histograma (Disponible en v0.2.0)")
    return fig

def plot_kde(*args, **kwargs):
    fig = go.Figure()
    fig.update_layout(title="KDE (Disponible en v0.2.0)")
    return fig

def plot_boxplot(*args, **kwargs):
    fig = go.Figure()
    fig.update_layout(title="Boxplot (Disponible en v0.2.0)")
    return fig

def plot_z_test(*args, **kwargs):
    fig = go.Figure()
    fig.update_layout(title="Campana de Gauss y Zona Crítica (Disponible en v0.3.0)")
    return fig