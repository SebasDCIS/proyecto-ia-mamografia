"""
Helpers de visualización del dashboard.

Funciones para formatear, colorear y presentar los resultados del
sistema de auditoría mamográfica de forma legible para el usuario.
"""

from typing import Any, Dict


# =============================================================================
# COLORES POR SEVERIDAD (paleta tipo semáforo)
# =============================================================================

COLORES_SEVERIDAD = {
    "critica": {
        "bg": "#FCEBEB",      # rojo muy claro
        "border": "#A32D2D",  # rojo oscuro
        "text": "#501313",    # rojo profundo
        "icon": "🚨",
    },
    "alta": {
        "bg": "#FAEEDA",      # naranja muy claro
        "border": "#BA7517",  # naranja oscuro
        "text": "#633806",    # naranja profundo
        "icon": "⚠️",
    },
    "media": {
        "bg": "#FAEEDA",      # naranja claro
        "border": "#EF9F27",  # naranja medio
        "text": "#854F0B",    # naranja oscuro
        "icon": "⚠️",
    },
    "baja": {
        "bg": "#FBEAF0",      # rosa muy claro
        "border": "#D4537E",  # rosa medio
        "text": "#72243E",    # rosa oscuro
        "icon": "ℹ️",
    },
}

COLOR_COHERENTE = {
    "bg": "#EAF3DE",          # verde muy claro
    "border": "#639922",      # verde
    "text": "#27500A",        # verde oscuro
    "icon": "✅",
}

COLOR_PRECAUCION = {
    "bg": "#FAEEDA",          # amarillo muy claro
    "border": "#EF9F27",      # amarillo medio
    "text": "#854F0B",        # amarillo oscuro
    "icon": "⚠️",
}

COLOR_ERROR = {
    "bg": "#F1EFE8",          # gris claro
    "border": "#888780",      # gris medio
    "text": "#2C2C2A",        # gris oscuro
    "icon": "❓",
}


# =============================================================================
# ETIQUETAS LEGIBLES
# =============================================================================

ETIQUETAS_ESTADO = {
    "coherente": "Coherente",
    "coherente_equivalente": "Coherente (equivalente clínico)",
    "coherente_con_precaucion": "Coherente con precaución",
    "notificacion": "Notificación suave",
    "incoherente": "Inconsistencia detectada",
    "no_procesable": "No procesable",
    "error": "Error de procesamiento",
}

ETIQUETAS_CATEGORIAS = {
    "biopsia_histologia": "Biopsia histológica",
    "derivacion_oncologica": "Derivación oncológica",
    "estudio_complementario_imagen": "Estudio complementario por imagen",
    "correlacion_ecografica": "Correlación ecográfica",
    "comparacion_estudios_previos": "Comparación con estudios previos",
    "control_corto_plazo": "Control a corto plazo (6 meses)",
    "control_anual": "Control anual",
    "criterio_medico": "Criterio médico (no específica)",
}

ETIQUETAS_VERIFICACION_ML = {
    "confirmado": "Confirmado",
    "confirmado_doble": "Validación cruzada",
    "ml_no_confirma": "ML disiente (regex prioritaria)",
    "discrepante_real": "Discrepancia detectada",
    "ml_inseguro": "ML sin confianza",
    "no_verificable": "No verificable",
    "no_ejecutado": "ML desactivado",
}


# =============================================================================
# FUNCIONES DE FORMATO
# =============================================================================

def obtener_estilo_resultado(resultado: Dict[str, Any]) -> Dict[str, str]:
    """Selecciona la paleta de colores según el estado del cotejo."""
    estado = resultado.get("cotejo_acr", {}).get("estado", "")
    severidad = resultado.get("cotejo_acr", {}).get("severidad")
    alerta = resultado.get("cotejo_acr", {}).get("alerta", False)

    if estado == "error":
        return COLOR_ERROR

    if alerta and severidad in COLORES_SEVERIDAD:
        return COLORES_SEVERIDAD[severidad]

    if estado in ("coherente_con_precaucion", "notificacion"):
        return COLOR_PRECAUCION

    if estado.startswith("coherente"):
        return COLOR_COHERENTE

    return COLOR_ERROR


def etiqueta_estado(estado: str) -> str:
    """Devuelve una etiqueta legible para el estado del cotejo."""
    return ETIQUETAS_ESTADO.get(estado, estado.replace("_", " ").capitalize())


def etiqueta_categoria(categoria: str) -> str:
    """Devuelve una etiqueta legible para una categoría de recomendación."""
    if categoria is None:
        return "No detectada"
    return ETIQUETAS_CATEGORIAS.get(categoria, categoria.replace("_", " "))


def etiqueta_verificacion_ml(estado: str) -> str:
    """Devuelve una etiqueta legible para el estado de verificación ML."""
    if estado is None:
        return "—"
    return ETIQUETAS_VERIFICACION_ML.get(estado, estado.replace("_", " "))


def banner_resultado_html(resultado: Dict[str, Any]) -> str:
    """Genera el HTML del banner de resultado con colores tipo semáforo."""
    estilo = obtener_estilo_resultado(resultado)
    estado = resultado.get("cotejo_acr", {}).get("estado", "")
    severidad = resultado.get("cotejo_acr", {}).get("severidad")
    alerta = resultado.get("cotejo_acr", {}).get("alerta", False)
    informe_id = resultado.get("informe_id") or "sin_id"

    if alerta:
        titulo = f"Alerta detectada · Severidad {severidad.upper() if severidad else '?'}"
    else:
        titulo = etiqueta_estado(estado)

    return f"""
    <div style="background: {estilo['bg']}; border-left: 4px solid {estilo['border']};
                padding: 14px 16px; border-radius: 8px; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 24px;">{estilo['icon']}</span>
            <div>
                <div style="font-size: 16px; font-weight: 500; color: {estilo['text']};">
                    {titulo}
                </div>
                <div style="font-size: 12px; color: {estilo['border']};">
                    {informe_id}
                </div>
            </div>
        </div>
    </div>
    """


def formato_confianza_ml(confianza: float) -> str:
    """Formatea la confianza del ML como porcentaje legible."""
    if confianza is None:
        return "—"
    return f"{confianza:.2%}"
