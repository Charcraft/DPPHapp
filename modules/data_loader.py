"""
modules/data_loader.py
──────────────────────
Responsabilidad única: proporcionar datos al resto de la aplicación.

Dos fuentes posibles:
  1. Archivo CSV subido por el usuario.
  2. Datos sintéticos generados con NumPy (útil para demos).
"""

import io
import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────
# Constantes de configuración para datos sintéticos
# ─────────────────────────────────────────────
SYNTHETIC_PRESETS = {
    "Normal (μ=70, σ=10)": {
        "dist": "normal",
        "params": {"loc": 70, "scale": 10},
        "description": "Distribución perfectamente normal. Ideal para validar la prueba Z.",
    },
    "Sesgada a la derecha (Log-Normal)": {
        "dist": "lognormal",
        "params": {"mean": 4.0, "sigma": 0.5},
        "description": "Distribución asimétrica positiva. Frecuente en salarios o precios.",
    },
    "Con Outliers": {
        "dist": "normal_outliers",
        "params": {"loc": 50, "scale": 8, "n_outliers": 5},
        "description": "Normal contaminada con valores atípicos extremos.",
    },
    "Bimodal": {
        "dist": "bimodal",
        "params": {"loc1": 40, "scale1": 5, "loc2": 70, "scale2": 5},
        "description": "Mezcla de dos normales. Viola el supuesto de unimodalidad.",
    },
}


def generate_synthetic_data(preset_name: str, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame con datos sintéticos según un preset predefinido.

    Parámetros
    ----------
    preset_name : str
        Nombre del preset (clave de SYNTHETIC_PRESETS).
    n : int
        Número de observaciones (se recomienda n >= 30 para el TLC).
    seed : int
        Semilla para reproducibilidad.

    Retorna
    -------
    pd.DataFrame con columna 'valor' y columna auxiliar 'id'.
    """
    rng = np.random.default_rng(seed)
    preset = SYNTHETIC_PRESETS[preset_name]
    dist = preset["dist"]
    p = preset["params"]

    if dist == "normal":
        data = rng.normal(loc=p["loc"], scale=p["scale"], size=n)

    elif dist == "lognormal":
        data = rng.lognormal(mean=p["mean"], sigma=p["sigma"], size=n)

    elif dist == "normal_outliers":
        data = rng.normal(loc=p["loc"], scale=p["scale"], size=n)
        # Inyectar outliers extremos
        outlier_indices = rng.choice(n, size=p["n_outliers"], replace=False)
        data[outlier_indices] = rng.normal(
            loc=p["loc"] + 4 * p["scale"], scale=2, size=p["n_outliers"]
        )

    elif dist == "bimodal":
        half = n // 2
        group1 = rng.normal(loc=p["loc1"], scale=p["scale1"], size=half)
        group2 = rng.normal(loc=p["loc2"], scale=p["scale2"], size=n - half)
        data = np.concatenate([group1, group2])
        rng.shuffle(data)

    else:
        raise ValueError(f"Distribución desconocida: {dist}")

    df = pd.DataFrame({"id": range(1, n + 1), "valor": data})
    return df


def load_csv(uploaded_file) -> pd.DataFrame | None:
    """
    Lee un archivo CSV subido a través de st.file_uploader.

    Retorna
    -------
    pd.DataFrame si la lectura es exitosa, None en caso de error.
    """
    try:
        # Soporta tanto texto como bytes
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content))
        return df
    except Exception as exc:
        st.error(f"❌ Error al leer el CSV: {exc}")
        return None


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Retorna los nombres de columnas numéricas del DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def compute_descriptive_stats(series: pd.Series) -> dict:
    """
    Calcula estadísticas descriptivas esenciales para una Serie numérica.

    Retorna
    -------
    dict con: n, mean, std, variance, median, min, max, q1, q3, iqr, skewness, kurtosis.
    """
    clean = series.dropna()
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))

    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),          # Desviación estándar muestral
        "variance": float(clean.var(ddof=1)),      # Varianza muestral
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "skewness": float(clean.skew()),
        "kurtosis": float(clean.kurtosis()),       # Exceso de curtosis
    }