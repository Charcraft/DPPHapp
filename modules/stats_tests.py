"""
modules/stats_tests.py
───────────────────────
Responsabilidad única: implementar la lógica matemática de la Prueba Z
de una muestra con varianza poblacional conocida o estimada.

Fórmula del estadístico Z:
    Z = (x̄ - μ₀) / (σ / √n)

Donde:
    x̄  = media muestral
    μ₀ = media hipotética bajo H0
    σ  = desviación estándar (poblacional si se conoce, muestral si no)
    n  = tamaño de la muestra
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────
# Resultado estructurado de la prueba
# ─────────────────────────────────────────────
@dataclass
class ZTestResult:
    """Contenedor inmutable con todos los resultados de la prueba Z."""

    # Parámetros de entrada
    sample_mean: float
    mu_0: float
    sigma: float
    n: int
    alpha: float
    test_type: str          # "bilateral" | "cola_izquierda" | "cola_derecha"
    h0_text: str
    h1_text: str

    # Resultados calculados
    z_stat: float
    p_value: float
    z_critical: float | list[float]  # Puede ser uno o dos valores críticos
    reject_h0: bool
    decision_text: str
    error_margin: float     # Error estándar de la media


def run_z_test(
    sample: "pd.Series",
    mu_0: float,
    sigma: float | None,
    alpha: float,
    test_type: str,
    h0_text: str,
    h1_text: str,
) -> ZTestResult:
    """
    Ejecuta la Prueba Z de una muestra.

    Parámetros
    ----------
    sample    : pd.Series — Muestra de datos numéricos.
    mu_0      : float     — Media hipotética bajo H0.
    sigma     : float|None — Desviación estándar poblacional conocida.
                             Si es None, se usa la muestral (apropiado para n>=30).
    alpha     : float     — Nivel de significancia (0 < alpha < 1).
    test_type : str       — "bilateral" | "cola_izquierda" | "cola_derecha"
    h0_text   : str       — Texto descriptivo de H0.
    h1_text   : str       — Texto descriptivo de H1.

    Retorna
    -------
    ZTestResult con todos los valores calculados.
    """
    clean = sample.dropna().values

    n = len(clean)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 observaciones para la prueba Z.")

    x_bar = float(np.mean(clean))

    # Si no se provee σ poblacional, se usa la desviación muestral
    sigma_used = float(sigma) if sigma is not None else float(np.std(clean, ddof=1))

    # Error estándar de la media
    se = sigma_used / np.sqrt(n)
    if se == 0:
        raise ValueError("La desviación estándar es 0; la prueba Z no es aplicable.")

    # ── Estadístico Z ──────────────────────────────────────────
    z_stat = (x_bar - mu_0) / se

    # ── p-value según tipo de prueba ───────────────────────────
    if test_type == "bilateral":
        p_value = 2.0 * (1 - stats.norm.cdf(abs(z_stat)))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        z_critical = [-z_crit, z_crit]
        reject_h0 = abs(z_stat) > z_crit

    elif test_type == "cola_izquierda":
        p_value = stats.norm.cdf(z_stat)
        z_crit = stats.norm.ppf(alpha)          # Valor negativo
        z_critical = z_crit
        reject_h0 = z_stat < z_crit

    else:  # cola_derecha
        p_value = 1.0 - stats.norm.cdf(z_stat)
        z_crit = stats.norm.ppf(1 - alpha)      # Valor positivo
        z_critical = z_crit
        reject_h0 = z_stat > z_crit

    # ── Decisión textual ───────────────────────────────────────
    if reject_h0:
        decision_text = (
            f"Se RECHAZA H₀ al nivel de significancia α = {alpha}. "
            f"Existe evidencia estadística suficiente para afirmar que {h1_text}."
        )
    else:
        decision_text = (
            f"NO se rechaza H₀ al nivel de significancia α = {alpha}. "
            f"No hay evidencia estadística suficiente para rechazar que {h0_text}."
        )

    return ZTestResult(
        sample_mean=x_bar,
        mu_0=mu_0,
        sigma=sigma_used,
        n=n,
        alpha=alpha,
        test_type=test_type,
        h0_text=h0_text,
        h1_text=h1_text,
        z_stat=z_stat,
        p_value=p_value,
        z_critical=z_critical,
        reject_h0=reject_h0,
        decision_text=decision_text,
        error_margin=se,
    )


def format_hypothesis_display(h0: str, h1: str, mu_0: float) -> tuple[str, str]:
    """
    Genera texto formateado LaTeX-compatible para mostrar las hipótesis.
    Retorna (h0_display, h1_display).
    """
    h0_display = f"H₀: {h0}  (μ = {mu_0})"
    h1_display = f"H₁: {h1}"
    return h0_display, h1_display


def interpret_p_value(p_value: float, alpha: float) -> str:
    """
    Retorna una interpretación pedagógica del p-value para mostrar al estudiante.
    """
    if p_value < 0.001:
        strength = "evidencia muy fuerte"
    elif p_value < 0.01:
        strength = "evidencia fuerte"
    elif p_value < 0.05:
        strength = "evidencia moderada"
    elif p_value < 0.10:
        strength = "evidencia débil"
    else:
        strength = "evidencia insuficiente"

    comparison = "menor que" if p_value <= alpha else "mayor que"

    return (
        f"El p-value ({p_value:.4f}) es {comparison} α ({alpha}), "
        f"lo que indica {strength} contra H₀."
    )