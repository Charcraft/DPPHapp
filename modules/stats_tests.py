"""
Módulo de Pruebas Estadísticas - VERSIÓN MOCK (Simulación)
"""

class DummyZResult:
    # Simulamos el resultado de la prueba para que app.py no falle al buscar las variables
    z_stat = 0.0
    p_value = 0.5
    reject_null = False
    alpha = 0.05
    test_type = "bilateral"
    critical_value = 1.96

def format_hypothesis_display(*args, **kwargs):
    return "H0 en construcción vs H1"

def interpret_p_value(*args, **kwargs):
    return "Interpretación estadística en construcción (v0.3.0)..."

def run_z_test(*args, **kwargs):
    return DummyZResult()