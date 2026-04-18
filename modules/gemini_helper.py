"""
modules/gemini_helper.py
─────────────────────────
Responsabilidad única: integrar la API de Google Gemini para generar
explicaciones estadísticas en lenguaje natural.

Requiere: pip install google-generativeai python-dotenv
"""

import os

import streamlit as st

# Importación condicional para no romper la app si la librería no está instalada
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────
MODEL_NAME = "gemini-1.5-flash"   # Modelo rápido y eficiente para texto


def _get_api_key() -> str | None:
    """
    Obtiene la API Key de Gemini en este orden de prioridad:
      1. Secrets de Streamlit (st.secrets) — recomendado en producción.
      2. Variable de entorno GEMINI_API_KEY — para desarrollo local.
    """
    # Primero intentar Streamlit Secrets (deployment en Streamlit Cloud)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    # Luego variable de entorno (desarrollo local con .env)
    return os.getenv("GEMINI_API_KEY")


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
    h0_text: str,
    h1_text: str,
    skewness: float,
    kurtosis: float,
) -> str:
    """
    Construye el prompt estructurado para enviar a Gemini.

    El prompt está diseñado para obtener una explicación pedagógica
    paso a paso, apropiada para un estudiante de ingeniería.
    """
    tipo_map = {
        "bilateral": "bilateral (dos colas)",
        "cola_izquierda": "unilateral izquierda (cola izquierda)",
        "cola_derecha": "unilateral derecha (cola derecha)",
    }

    decision = "SE RECHAZA H₀" if reject_h0 else "NO SE RECHAZA H₀"

    return f"""
Eres un profesor experto en estadística inferencial. Explica de forma clara, 
estructurada y pedagógica los resultados de la siguiente prueba de hipótesis.
Dirígete a un estudiante universitario de ingeniería en su primer curso de estadística.

## Contexto de la prueba realizada

**Tipo de prueba:** Prueba Z de una muestra ({tipo_map.get(test_type, test_type)})

**Hipótesis:**
- H₀: {h0_text}  (μ = {mu_0})
- H₁: {h1_text}

**Parámetros de la muestra:**
- Tamaño de muestra (n): {n}
- Media muestral (x̄): {sample_mean:.4f}
- Desviación estándar usada (σ): {sigma:.4f}
- Sesgo (skewness): {skewness:.4f}
- Curtosis (exceso): {kurtosis:.4f}

**Resultados del contraste:**
- Estadístico Z calculado: {z_stat:.4f}
- p-value: {p_value:.6f}
- Nivel de significancia (α): {alpha}
- **Decisión: {decision}**

## Tarea

Por favor responde en español y estructura tu respuesta así:

1. **Interpretación del estadístico Z**: ¿Qué significa este valor en términos de 
   desviaciones estándar desde la media hipotética?

2. **Interpretación del p-value**: Explica qué representa el p-value en el contexto 
   específico de esta prueba.

3. **Justificación de la decisión**: Explica matemáticamente y conceptualmente por 
   qué se rechaza o no se rechaza H₀.

4. **Supuestos de la prueba Z**: Comenta brevemente si los datos parecen cumplir 
   los supuestos (considera n={n}, sesgo={skewness:.3f}, curtosis={kurtosis:.3f}).

5. **Conclusión práctica**: ¿Qué implica esta decisión en términos reales/aplicados?

Usa notación matemática donde sea apropiado. Sé conciso pero completo.
""".strip()


@st.cache_data(show_spinner=False)
def get_gemini_explanation(prompt: str) -> tuple[str, str | None]:
    """
    Envía el prompt a Gemini y retorna la respuesta.

    Usa caché de Streamlit para evitar llamadas repetidas con el mismo prompt.

    Retorna
    -------
    (respuesta_texto, mensaje_error)
    Si todo fue bien, mensaje_error es None.
    """
    if not GEMINI_AVAILABLE:
        return (
            "",
            "La librería `google-generativeai` no está instalada. "
            "Ejecuta: pip install google-generativeai",
        )

    api_key = _get_api_key()
    if not api_key:
        return (
            "",
            "No se encontró la API Key de Gemini. "
            "Agrega GEMINI_API_KEY a tu archivo `.env` o a Streamlit Secrets.",
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        # Configuración de generación: respuestas determinísticas y concisas
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,       # Baja temperatura = respuestas más consistentes
            max_output_tokens=1200,
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        return (response.text, None)

    except Exception as exc:
        error_msg = str(exc)
        # Mensajes de error amigables para errores comunes
        if "API_KEY" in error_msg.upper() or "400" in error_msg:
            return ("", f"API Key inválida o sin permisos. Detalle: {error_msg}")
        elif "429" in error_msg:
            return ("", "Se alcanzó el límite de solicitudes de Gemini. Espera un momento.")
        else:
            return ("", f"Error al conectar con Gemini: {error_msg}")