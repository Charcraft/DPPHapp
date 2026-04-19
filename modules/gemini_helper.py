"""
modules/gemini_helper.py  —  StatLab v1.0.0
════════════════════════════════════════════════════════════════════════════════
Responsabilidad del módulo:
  - Gestionar la API Key de Gemini (Hardcodeada a petición del usuario).
  - Construir el prompt pedagógico para la Prueba Z.
  - Llamar a Gemini y devolver la respuesta o un mensaje de error amigable.
  - NUNCA propagar excepciones al nivel de la UI.
"""

import os
import streamlit as st

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

_MODEL_NAME = "gemini-2.0-flash"


# ─────────────────────────────────────────────────────────────────────────────
# Obtención de API Key (Hardcodeada)
# ─────────────────────────────────────────────────────────────────────────────
def _get_api_key() -> str | None:
    """
    Retorna la API Key directamente.
    Se ha integrado aquí directamente según tu solicitud.
    """
    # Aquí está tu API Key directa. Cámbiala si generas una nueva.
    return "AIzaSyDQRne1t8dLdrL1TGfE62xWZgrwGJ-AwrY"


# ─────────────────────────────────────────────────────────────────────────────
# Instrucciones en la UI cuando falta la clave
# ─────────────────────────────────────────────────────────────────────────────
def _show_setup_instructions():
    """
    Muestra en la interfaz de Streamlit un error si la API Key está vacía.
    """
    st.error(
        "**Módulo de IA deshabilitado — API Key no válida**\n\n"
        "Asegúrate de haber pegado correctamente tu clave de Google Gemini "
        "en la función `_get_api_key()` dentro de `gemini_helper.py`."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constructor del prompt (¡Error de h0_text solucionado!)
# ─────────────────────────────────────────────────────────────────────────────
def build_z_test_prompt(
    *,
    sample_mean: float,
    mu_0: float,
    sigma: float,
    n: int,
    z_stat: float,
    p_value: float,
    alpha: float,
    test_type: str,
    reject_h0: bool,
    h0_text: str,  # <-- Estos parámetros ahora están correctamente declarados
    h1_text: str,  # <-- para coincidir con lo que envía app.py
    skewness: float,
    kurtosis: float,
) -> str:
    """Construye el prompt pedagógico estructurado para la Prueba Z."""
    tipo_map = {
        "bilateral":      "bilateral (dos colas)",
        "cola_izquierda": "unilateral izquierda (cola izquierda)",
        "cola_derecha":   "unilateral derecha (cola derecha)",
    }
    decision_str = "SE RECHAZA H0" if reject_h0 else "NO SE RECHAZA H0"

    return f"""
Eres un profesor experto en estadística inferencial. Explica de forma estructurada
y pedagógica los resultados de la siguiente Prueba Z. Tu audiencia son estudiantes
universitarios de ingeniería en su primer curso de estadística.

## Datos de la prueba

- Tipo: Prueba Z de una muestra — {tipo_map.get(test_type, test_type)}
- H0: {h0_text}  (mu = {mu_0})
- H1: {h1_text}

## Resultados numéricos

| Parámetro              | Valor           |
|------------------------|-----------------|
| Tamaño de muestra n    | {n}             |
| Media muestral x̄      | {sample_mean:.4f} |
| Desv. estándar usada σ | {sigma:.4f}     |
| Sesgo (skewness)       | {skewness:.4f}  |
| Curtosis (exceso)      | {kurtosis:.4f}  |
| Estadístico Z          | {z_stat:.4f}    |
| p-value                | {p_value:.6f}   |
| Nivel de signif. α     | {alpha}         |
| **Decisión** | **{decision_str}** |

## Tarea

Responde en español con exactamente estas secciones:

**1. Interpretación del estadístico Z**
Qué significa Z = {z_stat:.4f} en términos de desviaciones estándar respecto a mu0.

**2. Interpretación del p-value**
Qué representa p = {p_value:.6f} en el contexto de esta prueba.

**3. Justificación de la decisión**
Por qué {decision_str} (matemática y conceptualmente).

**4. Supuestos de la Prueba Z**
Si los datos cumplen supuestos: n={n}, sesgo={skewness:.3f}, curtosis={kurtosis:.3f}.

**5. Conclusión práctica**
Qué implica esta decisión en términos reales.

Sé conciso pero completo. Usa notación matemática donde aporte claridad.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Llamada a la API
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_gemini_explanation(prompt: str) -> tuple[str, str | None]:
    """
    Envía el prompt a Gemini y retorna (respuesta, error).
    """
    if not GEMINI_AVAILABLE:
        return (
            "",
            (
                "La librería `google-generativeai` no está instalada.\n\n"
                "Ejecuta:  `pip install google-generativeai`  y reinicia la app."
            ),
        )

    api_key = _get_api_key()
    if not api_key:
        return ("", "API_KEY_MISSING")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_MODEL_NAME)
        gen_config = genai.types.GenerationConfig(
            temperature=0.25,
            max_output_tokens=1400,
        )
        response = model.generate_content(prompt, generation_config=gen_config)
        return (response.text, None)

    except Exception as exc:
        err = str(exc)

        if "API_KEY_INVALID" in err or "400" in err:
            msg = (
                "La API Key no es válida o fue revocada.\n\n"
                "**Cómo solucionarlo:**\n"
                "1. Copia la clave de nuevo desde aistudio.google.com.\n"
                "2. Asegúrate de que no tenga espacios extra.\n"
                "3. Verifica que el proyecto de Google Cloud tiene la API de Gemini habilitada."
            )
        elif "429" in err or "RESOURCE_EXHAUSTED" in err:
            msg = (
                "Se alcanzó el límite de solicitudes de Gemini.\n\n"
                "Espera 30-60 segundos e intenta nuevamente. "
                "Si el problema persiste, revisa tu cuota en console.cloud.google.com."
            )
        elif "503" in err or "UNAVAILABLE" in err:
            msg = (
                "El servicio de Gemini no está disponible (mantenimiento).\n\n"
                "Intenta nuevamente en 2-3 minutos."
            )
        elif "SAFETY" in err:
            msg = (
                "La respuesta fue bloqueada por los filtros de seguridad de Gemini. "
                "Intenta con un dataset diferente."
            )
        elif "SSL" in err or "ConnectionError" in err or "timeout" in err.lower():
            msg = (
                "No fue posible conectar con Gemini.\n\n"
                "Verifica tu conexión a internet e intenta nuevamente."
            )
        else:
            msg = f"Error inesperado:\n```\n{err}\n```"

        return ("", msg)


# ─────────────────────────────────────────────────────────────────────────────
# Renderizador centralizado de errores para la UI
# ─────────────────────────────────────────────────────────────────────────────
def render_gemini_error(error_msg: str):
    """
    Muestra el error de Gemini de forma elegante.
    """
    if error_msg == "API_KEY_MISSING":
        _show_setup_instructions()
    else:
        st.warning("No fue posible obtener la explicación de Gemini.\n\n" + error_msg)
        st.caption(
            "Si el error persiste, verifica tu API Key o consulta "
            "el estado del servicio en status.cloud.google.com."
        )