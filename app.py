"""
app.py
───────
Archivo principal de la aplicación.
Orquesta la navegación y conecta todos los módulos.

Ejecutar con:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

# ── Cargar variables de entorno (.env) ─────────────────────────────────────
load_dotenv()

# ── Importar módulos propios ────────────────────────────────────────────────
from modules.data_loader import (
    SYNTHETIC_PRESETS,
    compute_descriptive_stats,
    generate_synthetic_data,
    get_numeric_columns,
    load_csv,
    load_local_csv,
)
from modules.gemini_helper import build_z_test_prompt, get_gemini_explanation
from modules.stats_tests import (
    format_hypothesis_display,
    interpret_p_value,
    run_z_test,
)
from modules.visualizations import plot_boxplot, plot_histogram, plot_kde, plot_z_test


# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN GLOBAL DE STREAMLIT
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StatLab · Análisis Estadístico",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado: paleta blanco / negro / grises ──────────────────────
st.markdown(
    """
    <style>
    /* ── Fuentes ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500&family=Syne:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto Mono', monospace;
    }

    /* ── Fondo y colores base ────────────────────────── */
    .stApp { background-color: #F5F5F5; }

    /* ── Sidebar ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #2A2A2A;
    }
    [data-testid="stSidebar"] * { color: #E8E8E8 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #AAAAAA !important; }

    /* ── Título principal ────────────────────────────── */
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0D0D0D;
        letter-spacing: -0.03em;
        border-bottom: 3px solid #0D0D0D;
        padding-bottom: 0.4rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.78rem;
        color: #888888;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    /* ── Tarjetas de métricas ────────────────────────── */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left: 4px solid #0D0D0D;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .metric-label {
        font-size: 0.68rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 500;
        color: #0D0D0D;
        font-family: 'Syne', sans-serif;
    }

    /* ── Tarjeta de resultado de prueba ──────────────── */
    .result-reject {
        background: #0D0D0D;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 1rem 1.4rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .result-no-reject {
        background: #FFFFFF;
        color: #0D0D0D;
        border: 2px solid #0D0D0D;
        border-radius: 4px;
        padding: 1rem 1.4rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Hipótesis display ───────────────────────────── */
    .hypothesis-box {
        background: #FFFFFF;
        border: 1px solid #DDDDDD;
        border-radius: 4px;
        padding: 0.8rem 1.2rem;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.85rem;
        color: #2A2A2A;
        margin-bottom: 0.5rem;
    }

    /* ── Tabs ────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #0D0D0D;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Syne', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888888;
        padding: 0.7rem 1.4rem;
        border-radius: 0;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #0D0D0D !important;
        border-bottom: 2px solid #0D0D0D;
        background: transparent;
    }

    /* ── Botones ─────────────────────────────────────── */
    .stButton > button {
        background-color: #0D0D0D;
        color: #FFFFFF;
        border: none;
        border-radius: 3px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.55rem 1.4rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #333333; }

    /* ── Text areas ──────────────────────────────────── */
    textarea {
        font-family: 'Roboto Mono', monospace !important;
        font-size: 0.82rem !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 3px !important;
    }

    /* ── Expander ────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #555555;
    }

    /* ── Sección de IA ───────────────────────────────── */
    .ai-response {
        background: #FAFAFA;
        border-left: 3px solid #555555;
        padding: 1rem 1.4rem;
        font-size: 0.85rem;
        line-height: 1.8;
        border-radius: 0 4px 4px 0;
    }

    /* ── Divider ─────────────────────────────────────── */
    hr { border-color: #E0E0E0; margin: 1.2rem 0; }

    /* ── Info / Warning boxes ────────────────────────── */
    .stAlert { border-radius: 3px; font-size: 0.82rem; }

    /* ── Ocultar marca Streamlit ─────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# 2. FUNCIONES AUXILIARES DE UI
# ════════════════════════════════════════════════════════════════════════════

def render_metric(label: str, value: str):
    """Renderiza una tarjeta de métrica con estilo personalizado."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_banner(reject: bool, decision_text: str):
    """Renderiza el banner de decisión estadística."""
    css_class = "result-reject" if reject else "result-no-reject"
    icon = "✗ " if reject else "○ "
    st.markdown(
        f'<div class="{css_class}">{icon}{decision_text}</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR: CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple:
    """
    Renderiza el panel lateral de carga de datos.
    Retorna (df, selected_column, stats) o (None, None, None) si no hay datos.

    Gestión de Session State
    ────────────────────────
    El DataFrame siempre se persiste en st.session_state["df"] para que
    no desaparezca al cambiar entre las pestañas de la app.
    La clave "data_source_loaded" registra qué fuente está activa y
    permite detectar cambios de fuente para limpiar el estado anterior.
    """
    with st.sidebar:
        st.markdown(
            "<p style='font-family:Syne,sans-serif;font-size:1.1rem;"
            "font-weight:700;margin-bottom:0.1rem;'>◼ StatLab</p>"
            "<p style='font-size:0.65rem;color:#666;letter-spacing:0.15em;"
            "text-transform:uppercase;margin-bottom:1.5rem;'>Statistical Analysis Tool</p>",
            unsafe_allow_html=True,
        )

        st.markdown("**Fuente de datos**")
        data_source = st.radio(
            "Selecciona origen",
            ["Datos Sintéticos", "Subir CSV", "Dataset de Ejemplo"],
            label_visibility="collapsed",
        )

        # Detectar cambio de fuente y limpiar estado previo para evitar
        # que datos de una fuente anterior contaminen la nueva selección.
        if st.session_state.get("data_source_loaded") != data_source:
            st.session_state.pop("df", None)
            st.session_state.pop("z_result", None)
            st.session_state.pop("z_stats", None)
            st.session_state["data_source_loaded"] = data_source

        df = None

        # ── Opción 1: Datos Sintéticos ──────────────────────────────────
        if data_source == "Datos Sintéticos":
            preset = st.selectbox(
                "Distribución",
                list(SYNTHETIC_PRESETS.keys()),
            )
            n_samples = st.slider("Tamaño de muestra (n)", 30, 500, 100, step=10)
            seed = st.number_input("Semilla aleatoria", value=42, step=1)

            st.caption(SYNTHETIC_PRESETS[preset]["description"])

            if st.button("Generar datos", use_container_width=True):
                df = generate_synthetic_data(preset, n_samples, int(seed))
                st.session_state["df"] = df
                st.success(f"✓ {n_samples} observaciones generadas")
                st.rerun()

            # Carga inicial silenciosa si no hay datos en sesión todavía
            if "df" not in st.session_state:
                df = generate_synthetic_data(preset, n_samples, int(seed))
                st.session_state["df"] = df

        # ── Opción 2: CSV subido por el usuario ─────────────────────────
        elif data_source == "Subir CSV":
            uploaded = st.file_uploader("Archivo CSV", type=["csv"])
            if uploaded is not None:
                # Siempre re-leer cuando hay un archivo presente.
                # getvalue() (implementado en load_csv) es idempotente
                # y seguro ante múltiples re-runs de Streamlit.
                df = load_csv(uploaded)
                if df is not None:
                    st.session_state["df"] = df
                    st.success(f"✓ {len(df)} filas cargadas")

            elif "df" not in st.session_state:
                st.info("Sube un archivo CSV para continuar.")

        # ── Opción 3: Dataset de Ejemplo ────────────────────────────────
        else:
            EXAMPLE_PATH = "data/ejemplo.csv"
            st.caption(
                "Carga automáticamente el dataset de ejemplo incluido "
                f"en `{EXAMPLE_PATH}`."
            )

            if st.button("Cargar Ejemplo", use_container_width=True):
                df = load_local_csv(EXAMPLE_PATH)
                if df is not None:
                    st.session_state["df"] = df
                    st.success(f"✓ Ejemplo cargado — {len(df)} filas")
                    st.rerun()

            # Si ya se cargó antes en esta sesión, no hace falta volver a leer
            if "df" not in st.session_state:
                st.info(f"Pulsa **Cargar Ejemplo** para usar `{EXAMPLE_PATH}`.")

        # ── Recuperar DataFrame de sesión (persiste entre tabs) ─────────
        if df is None:
            df = st.session_state.get("df")

        if df is None:
            return None, None, None

        # ── Selección de columna ────────────────────────────────────────
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            st.error("El dataset no tiene columnas numéricas.")
            return None, None, None

        st.divider()
        selected_col = st.selectbox("Variable a analizar", numeric_cols)

        # Estadísticas descriptivas en sidebar
        st.divider()
        stats = compute_descriptive_stats(df[selected_col])
        st.markdown("**Resumen rápido**")
        st.markdown(
            f"n = **{stats['n']}** &nbsp;|&nbsp; "
            f"μ = **{stats['mean']:.3f}** &nbsp;|&nbsp; "
            f"σ = **{stats['std']:.3f}**",
            unsafe_allow_html=True,
        )

        return df, selected_col, stats


# ════════════════════════════════════════════════════════════════════════════
# 4. TAB 1: EXPLORACIÓN DE DATOS
# ════════════════════════════════════════════════════════════════════════════

def render_exploration_tab(df, selected_col: str, stats: dict):
    """Renderiza la pestaña de exploración descriptiva y visual."""

    st.markdown("### Estadísticas Descriptivas")

    # ── Métricas en columnas ────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_metric("n", str(stats["n"]))
    with c2:
        render_metric("Media (x̄)", f"{stats['mean']:.4f}")
    with c3:
        render_metric("Desv. Est. (s)", f"{stats['std']:.4f}")
    with c4:
        render_metric("Mediana", f"{stats['median']:.4f}")
    with c5:
        render_metric("Sesgo", f"{stats['skewness']:.4f}")
    with c6:
        render_metric("Curtosis", f"{stats['kurtosis']:.4f}")

    st.markdown("---")

    # ── Gráficas ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**Histograma**")
        bins = st.slider("Número de bins", 10, 80, 30, key="hist_bins")
        fig_hist = plot_histogram(df[selected_col], selected_col, bins)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.markdown("**Densidad KDE**")
        st.markdown("<br>", unsafe_allow_html=True)  # espaciado visual
        fig_kde = plot_kde(df[selected_col], selected_col)
        st.plotly_chart(fig_kde, use_container_width=True)

    # ── Boxplot ─────────────────────────────────────────────────────────────
    st.markdown("**Boxplot**")
    fig_box = plot_boxplot(df[selected_col], selected_col)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Sección de análisis interpretativo (estudiante) ─────────────────────
    st.markdown("---")
    st.markdown("### ✏ Tu análisis exploratorio")
    st.caption(
        "Observa las gráficas anteriores y responde las preguntas. "
        "Este análisis previo te ayudará a interpretar mejor los resultados de la prueba."
    )

    with st.expander("📋 Preguntas guía — Análisis EDA", expanded=True):
        col_q1, col_q2 = st.columns(2)

        with col_q1:
            ans_normal = st.text_area(
                "¿La distribución parece normal? ¿Por qué?",
                placeholder="Ej: Sí parece normal porque la curva KDE es simétrica "
                            "y el histograma muestra una forma de campana...",
                height=120,
                key="ans_normal",
            )
            ans_sesgo = st.text_area(
                "¿Observas sesgo? ¿A qué lado?",
                placeholder="Ej: Hay un ligero sesgo positivo (a la derecha) "
                            "porque la cola derecha es más larga...",
                height=120,
                key="ans_sesgo",
            )

        with col_q2:
            ans_outliers = st.text_area(
                "¿Hay outliers visibles en el boxplot?",
                placeholder="Ej: Se observan 3 puntos atípicos por encima del bigote "
                            "superior, lo cual podría afectar la media...",
                height=120,
                key="ans_outliers",
            )
            ans_conclusion = st.text_area(
                "¿Es apropiado aplicar una prueba Z? ¿Por qué?",
                placeholder="Ej: Sí, porque n=100 >= 30 y el Teorema Central del "
                            "Límite garantiza normalidad asintótica de x̄...",
                height=120,
                key="ans_conclusion",
            )

    # ── Vista previa del dataset ─────────────────────────────────────────────
    with st.expander("🗂 Vista previa del dataset"):
        st.dataframe(
            df.head(50).style.format(precision=4),
            use_container_width=True,
            height=260,
        )
        st.caption(f"Mostrando primeras 50 de {len(df)} filas.")


# ════════════════════════════════════════════════════════════════════════════
# 5. TAB 2: PRUEBA DE HIPÓTESIS
# ════════════════════════════════════════════════════════════════════════════

def render_hypothesis_tab(df, selected_col: str, stats: dict):
    """Renderiza la pestaña de configuración y resultados de la Prueba Z."""

    st.markdown("### Configuración de la Prueba Z")

    # ── Panel de configuración ──────────────────────────────────────────────
    with st.container():
        col_conf1, col_conf2 = st.columns([2, 1])

        with col_conf1:
            st.markdown("**Definición de hipótesis**")

            h_col1, h_col2 = st.columns(2)
            with h_col1:
                h0_text = st.text_input(
                    "Hipótesis nula (H₀)",
                    value=f"la media poblacional es igual a {stats['mean']:.2f}",
                    help="Escribe el enunciado completo de H₀",
                )
            with h_col2:
                h1_text = st.text_input(
                    "Hipótesis alternativa (H₁)",
                    value=f"la media poblacional es diferente de {stats['mean']:.2f}",
                    help="Escribe el enunciado completo de H₁",
                )

            mu_0 = st.number_input(
                "Valor hipotético de μ bajo H₀",
                value=round(stats["mean"], 2),
                step=0.01,
                format="%.4f",
                help="¿Con qué valor de μ quieres contrastar tu muestra?",
            )

        with col_conf2:
            st.markdown("**Parámetros del test**")

            test_type = st.selectbox(
                "Tipo de prueba",
                ["bilateral", "cola_izquierda", "cola_derecha"],
                format_func=lambda x: {
                    "bilateral": "Bilateral (H₁: μ ≠ μ₀)",
                    "cola_izquierda": "Cola izquierda (H₁: μ < μ₀)",
                    "cola_derecha": "Cola derecha (H₁: μ > μ₀)",
                }[x],
            )

            alpha = st.select_slider(
                "Nivel de significancia (α)",
                options=[0.01, 0.05, 0.10],
                value=0.05,
                format_func=lambda x: f"α = {x}",
            )

            use_known_sigma = st.toggle(
                "Usar σ conocida",
                value=False,
                help="Activa si conoces la desviación estándar poblacional exacta.",
            )
            if use_known_sigma:
                sigma_input = st.number_input(
                    "Desviación estándar poblacional (σ)",
                    value=round(stats["std"], 4),
                    step=0.001,
                    format="%.4f",
                    min_value=0.0001,
                )
            else:
                sigma_input = None
                st.caption(f"Se usará σ muestral = {stats['std']:.4f}")

    st.markdown("---")

    # ── Ejecutar prueba ─────────────────────────────────────────────────────
    if st.button("▶  Ejecutar Prueba Z", use_container_width=False):
        try:
            result = run_z_test(
                sample=df[selected_col],
                mu_0=mu_0,
                sigma=sigma_input,
                alpha=alpha,
                test_type=test_type,
                h0_text=h0_text,
                h1_text=h1_text,
            )
            st.session_state["z_result"] = result
            st.session_state["z_stats"] = stats
        except ValueError as e:
            st.error(f"Error en la prueba: {e}")
            return

    # ── Mostrar resultados si ya se ejecutó ─────────────────────────────────
    result = st.session_state.get("z_result")
    if result is None:
        st.info("Configura los parámetros y haz clic en **Ejecutar Prueba Z**.")
        return

    st.markdown("### Resultados de la Prueba")

    # Hipótesis formateadas
    st.markdown(
        f'<div class="hypothesis-box">H₀: {result.h0_text}  (μ₀ = {result.mu_0})</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hypothesis-box">H₁: {result.h1_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Métricas clave
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_metric("Estadístico Z", f"{result.z_stat:.4f}")
    with r2:
        render_metric("p-value", f"{result.p_value:.6f}")
    with r3:
        render_metric("α (alpha)", str(result.alpha))
    with r4:
        render_metric("n", str(result.n))

    # Interpretación del p-value
    st.caption(interpret_p_value(result.p_value, result.alpha))

    # Banner de decisión
    st.markdown("<br>", unsafe_allow_html=True)
    render_decision_banner(result.reject_h0, result.decision_text)

    # Gráfica de la prueba Z
    st.markdown("<br>", unsafe_allow_html=True)
    fig_z = plot_z_test(result.z_stat, result.alpha, result.test_type)
    st.plotly_chart(fig_z, use_container_width=True)

    # Tabla de valores críticos
    with st.expander("📐 Valores críticos y región de rechazo"):
        zc = result.z_critical
        if isinstance(zc, list):
            st.markdown(
                f"Prueba **bilateral** con α={result.alpha}:  "
                f"Región de rechazo → Z < {zc[0]:.4f}  ó  Z > {zc[1]:.4f}"
            )
        elif result.test_type == "cola_izquierda":
            st.markdown(
                f"Prueba **cola izquierda** con α={result.alpha}:  "
                f"Región de rechazo → Z < {zc:.4f}"
            )
        else:
            st.markdown(
                f"Prueba **cola derecha** con α={result.alpha}:  "
                f"Región de rechazo → Z > {zc:.4f}"
            )

    st.markdown("---")

    # ── Módulo IA: Explicación de Gemini ────────────────────────────────────
    render_gemini_section(result, stats)


# ════════════════════════════════════════════════════════════════════════════
# 6. SECCIÓN IA (GEMINI)
# ════════════════════════════════════════════════════════════════════════════

def render_gemini_section(result, stats: dict):
    """Renderiza el bloque de explicación con IA (Google Gemini)."""

    st.markdown("### ◈ Explicación con Inteligencia Artificial")
    st.caption(
        "Gemini analizará los resultados estadísticos y te ofrecerá "
        "una explicación pedagógica paso a paso."
    )

    col_btn, col_info = st.columns([1, 3])

    with col_btn:
        run_gemini = st.button("Consultar a Gemini ✦", use_container_width=True)

    with col_info:
        api_key_present = bool(
            os.getenv("GEMINI_API_KEY")
            or (
                hasattr(st, "secrets")
                and "GEMINI_API_KEY" in st.secrets
            )
        )
        if not api_key_present:
            st.warning(
                "⚠ No se detectó GEMINI_API_KEY. "
                "Agrega tu clave al archivo `.env` para activar este módulo."
            )

    if run_gemini:
        prompt = build_z_test_prompt(
            sample_mean=result.sample_mean,
            mu_0=result.mu_0,
            sigma=result.sigma,
            n=result.n,
            z_stat=result.z_stat,
            p_value=result.p_value,
            alpha=result.alpha,
            test_type=result.test_type,
            reject_h0=result.reject_h0,
            h0_text=result.h0_text,
            h1_text=result.h1_text,
            skewness=stats["skewness"],
            kurtosis=stats["kurtosis"],
        )

        with st.spinner("Consultando a Gemini..."):
            explanation, error = get_gemini_explanation(prompt)

        if error:
            st.error(f"❌ {error}")
        else:
            st.markdown(
                f'<div class="ai-response">{explanation}</div>',
                unsafe_allow_html=True,
            )

            # Expander con el prompt enviado (útil para aprendizaje)
            with st.expander("🔍 Ver prompt enviado a Gemini"):
                st.code(prompt, language="markdown")


# ════════════════════════════════════════════════════════════════════════════
# 7. PUNTO DE ENTRADA PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Función principal que orquesta el renderizado de la aplicación."""

    # ── Encabezado ──────────────────────────────────────────────────────────
    st.markdown(
        '<p class="main-title">StatLab</p>'
        '<p class="subtitle">Herramienta de Análisis Estadístico Inferencial</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar (carga de datos) ─────────────────────────────────────────────
    df, selected_col, stats = render_sidebar()

    if df is None:
        # Estado inicial sin datos cargados
        st.markdown(
            """
            <div style="text-align:center; padding: 4rem 2rem; color:#AAAAAA;">
                <p style="font-size:2.5rem;">◼</p>
                <p style="font-family:'Syne',sans-serif; font-size:1rem; 
                   letter-spacing:0.1em; text-transform:uppercase;">
                   Carga datos para comenzar
                </p>
                <p style="font-size:0.8rem;">
                   Usa el panel lateral para generar datos sintéticos o subir un CSV.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Tabs de navegación ──────────────────────────────────────────────────
    tab_explore, tab_test = st.tabs(
        ["01 — Exploración de Datos", "02 — Prueba de Hipótesis"]
    )

    with tab_explore:
        render_exploration_tab(df, selected_col, stats)

    with tab_test:
        render_hypothesis_tab(df, selected_col, stats)


# ── Ejecutar ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()