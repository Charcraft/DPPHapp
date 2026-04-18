"""
modules/visualizations.py
──────────────────────────
Responsabilidad única: generar todas las figuras de Plotly.

Paleta restringida a blanco / negro / escalas de grises para
mantener el estilo profesional y minimalista de la aplicación.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# ─────────────────────────────────────────────
# Constantes de la paleta cromática
# ─────────────────────────────────────────────
C_BG = "#FFFFFF"          # Fondo del gráfico
C_PAPER = "#F8F8F8"       # Fondo del área de papel
C_GRID = "#E0E0E0"        # Líneas de rejilla
C_TEXT = "#1A1A1A"        # Texto principal
C_AXIS = "#555555"        # Ejes secundarios
C_BAR = "#2B2B2B"         # Barras del histograma
C_LINE = "#000000"        # Líneas principales
C_FILL = "rgba(60,60,60,0.15)"   # Relleno KDE suave
C_BOX = "#3A3A3A"         # Cuerpo del boxplot
C_REJECT = "rgba(40,40,40,0.20)" # Zona de rechazo
C_ACCEPT = "rgba(200,200,200,0.15)" # Zona de no rechazo

FONT_FAMILY = "Roboto Mono, monospace"


def _base_layout(title: str = "") -> dict:
    """Devuelve el layout base compartido por todas las figuras."""
    return dict(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=14, color=C_TEXT)),
        font=dict(family=FONT_FAMILY, size=11, color=C_TEXT),
        paper_bgcolor=C_PAPER,
        plot_bgcolor=C_BG,
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(
            gridcolor=C_GRID,
            linecolor=C_AXIS,
            tickcolor=C_AXIS,
            tickfont=dict(color=C_TEXT),
            title_font=dict(color=C_TEXT),
        ),
        yaxis=dict(
            gridcolor=C_GRID,
            linecolor=C_AXIS,
            tickcolor=C_AXIS,
            tickfont=dict(color=C_TEXT),
            title_font=dict(color=C_TEXT),
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=C_GRID,
            borderwidth=1,
            font=dict(size=10),
        ),
    )


def plot_histogram(series: pd.Series, column_name: str, bins: int = 30) -> go.Figure:
    """
    Histograma con curva de densidad normal superpuesta.

    Parámetros
    ----------
    series : pd.Series  — Variable a graficar.
    column_name : str   — Nombre para etiquetas.
    bins : int          — Número de bins.
    """
    clean = series.dropna()
    mu, sigma = float(clean.mean()), float(clean.std(ddof=1))

    fig = go.Figure()

    # — Barras del histograma (frecuencia relativa) —
    fig.add_trace(
        go.Histogram(
            x=clean,
            nbinsx=bins,
            histnorm="probability density",
            name="Frecuencia observada",
            marker=dict(color=C_BAR, line=dict(color=C_BG, width=0.8)),
            opacity=0.85,
        )
    )

    # — Curva normal teórica —
    x_range = np.linspace(clean.min() - 3, clean.max() + 3, 400)
    y_normal = stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_normal,
            mode="lines",
            name=f"Normal teórica (μ={mu:.2f}, σ={sigma:.2f})",
            line=dict(color=C_LINE, width=2, dash="dash"),
        )
    )

    layout = _base_layout(f"Histograma — {column_name}")
    layout["xaxis"]["title"] = column_name
    layout["yaxis"]["title"] = "Densidad"
    fig.update_layout(**layout)
    return fig


def plot_kde(series: pd.Series, column_name: str) -> go.Figure:
    """
    Estimación de densidad por kernel (KDE) con área sombreada.
    """
    clean = series.dropna().values
    kde = stats.gaussian_kde(clean, bw_method="scott")
    x_range = np.linspace(clean.min() - 1, clean.max() + 1, 500)
    y_kde = kde(x_range)

    fig = go.Figure()

    # — Área sombreada —
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([[x_range[0]], x_range, [x_range[-1]]]),
            y=np.concatenate([[0], y_kde, [0]]),
            fill="toself",
            fillcolor=C_FILL,
            line=dict(color="rgba(0,0,0,0)"),
            name="Área KDE",
            showlegend=False,
        )
    )

    # — Línea KDE —
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_kde,
            mode="lines",
            name="Densidad KDE",
            line=dict(color=C_LINE, width=2.5),
        )
    )

    # — Línea de media —
    mu = float(clean.mean())
    fig.add_vline(
        x=mu,
        line=dict(color=C_AXIS, width=1.5, dash="dot"),
        annotation_text=f"μ={mu:.2f}",
        annotation_font=dict(size=10, color=C_AXIS),
    )

    layout = _base_layout(f"Densidad KDE — {column_name}")
    layout["xaxis"]["title"] = column_name
    layout["yaxis"]["title"] = "Densidad"
    fig.update_layout(**layout)
    return fig


def plot_boxplot(series: pd.Series, column_name: str) -> go.Figure:
    """
    Boxplot horizontal con puntos individuales superpuestos (jitter).
    """
    clean = series.dropna()
    fig = go.Figure()

    # — Boxplot —
    fig.add_trace(
        go.Box(
            x=clean,
            name=column_name,
            boxmean=True,          # Muestra la media además de la mediana
            marker=dict(
                color=C_BOX,
                outliercolor="#888888",
                line=dict(outliercolor="#888888", outlierwidth=1),
            ),
            line=dict(color=C_LINE, width=1.5),
            fillcolor="rgba(200,200,200,0.3)",
            whiskerwidth=0.5,
            orientation="h",
        )
    )

    layout = _base_layout(f"Boxplot — {column_name}")
    layout["yaxis"]["showticklabels"] = False
    layout["xaxis"]["title"] = column_name
    layout["height"] = 260
    fig.update_layout(**layout)
    return fig


def plot_z_test(
    z_stat: float,
    alpha: float,
    test_type: str,
) -> go.Figure:
    """
    Curva normal estándar con zona(s) de rechazo sombreadas y
    posición del estadístico Z marcada.

    Parámetros
    ----------
    z_stat   : float  — Valor del estadístico Z calculado.
    alpha    : float  — Nivel de significancia (ej. 0.05).
    test_type: str    — "bilateral" | "cola_izquierda" | "cola_derecha"
    """
    x = np.linspace(-4.5, 4.5, 1000)
    y = stats.norm.pdf(x)

    fig = go.Figure()

    # — Curva normal —
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            name="N(0,1)",
            line=dict(color=C_LINE, width=2),
        )
    )

    # — Zonas de rechazo según tipo de prueba —
    def _shade_region(x_mask, label, showlegend=True):
        """Agrega un área sombreada para x donde x_mask es True."""
        x_seg = x[x_mask]
        y_seg = y[x_mask]
        if len(x_seg) == 0:
            return
        x_fill = np.concatenate([[x_seg[0]], x_seg, [x_seg[-1]]])
        y_fill = np.concatenate([[0], y_seg, [0]])
        fig.add_trace(
            go.Scatter(
                x=x_fill, y=y_fill,
                fill="toself",
                fillcolor=C_REJECT,
                line=dict(color="rgba(0,0,0,0)"),
                name=label,
                showlegend=showlegend,
            )
        )

    if test_type == "bilateral":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        _shade_region(x <= -z_crit, f"Rechazo (α/2={alpha/2:.3f})")
        _shade_region(x >= z_crit, f"Rechazo (α/2={alpha/2:.3f})", showlegend=False)
        _add_critical_lines(fig, [-z_crit, z_crit], alpha, test_type)

    elif test_type == "cola_izquierda":
        z_crit = stats.norm.ppf(alpha)
        _shade_region(x <= z_crit, f"Rechazo (α={alpha})")
        _add_critical_lines(fig, [z_crit], alpha, test_type)

    else:  # cola_derecha
        z_crit = stats.norm.ppf(1 - alpha)
        _shade_region(x >= z_crit, f"Rechazo (α={alpha})")
        _add_critical_lines(fig, [z_crit], alpha, test_type)

    # — Estadístico Z calculado —
    y_at_z = float(stats.norm.pdf(z_stat))
    fig.add_trace(
        go.Scatter(
            x=[z_stat],
            y=[y_at_z],
            mode="markers+text",
            name=f"Z calculado = {z_stat:.4f}",
            marker=dict(color=C_LINE, size=10, symbol="diamond"),
            text=[f"  Z={z_stat:.3f}"],
            textposition="top right",
            textfont=dict(size=10, color=C_LINE),
        )
    )

    # — Línea vertical del estadístico —
    fig.add_vline(
        x=z_stat,
        line=dict(color=C_LINE, width=1.5, dash="dash"),
        annotation_text="",
    )

    layout = _base_layout("Distribución Normal Estándar — Prueba Z")
    layout["xaxis"]["title"] = "Z"
    layout["yaxis"]["title"] = "Densidad"
    layout["xaxis"]["range"] = [-4.5, 4.5]
    layout["height"] = 380
    fig.update_layout(**layout)
    return fig


def _add_critical_lines(
    fig: go.Figure, z_crits: list[float], alpha: float, test_type: str
):
    """Agrega líneas verticales punteadas en los valores críticos."""
    for zc in z_crits:
        fig.add_vline(
            x=zc,
            line=dict(color="#888888", width=1.2, dash="dot"),
            annotation_text=f"Zc={zc:.3f}",
            annotation_font=dict(size=9, color="#888888"),
            annotation_position="top",
        )