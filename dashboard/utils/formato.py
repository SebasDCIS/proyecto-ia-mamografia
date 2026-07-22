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

COLOR_REVISION = {
    "bg": "#FBF0E4",          # naranja muy claro
    "border": "#D97B29",      # naranja
    "text": "#7A3F0B",        # naranja oscuro
    "icon": "🔍",
}


# =============================================================================
# ETIQUETAS LEGIBLES
# =============================================================================

ETIQUETAS_ESTADO = {
    "coherente": "Coherente",
    "coherente_equivalente": "Coherente (equivalente clínico)",
    "coherente_con_precaucion": "Coherente con precaución",
    "incoherente": "Incoherente",
    "revision_extraccion": "Revisión por extracción",
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
    "confirmado": "Lectura ML concuerda",
    "confirmado_doble": "Lectura ML recupera confianza",
    "ml_no_confirma": "Lectura ML difiere (prima la literal)",
    "discrepante_real": "Lectura incierta (revisar)",
    "ml_inseguro": "Lectura ML sin confianza",
    "no_verificable": "Sin lectura de apoyo",
    "no_ejecutado": "Apoyo ML desactivado",
    "alerta_omision_buscador": "Omisión de BI-RADS",
}


# Etiquetas legibles para los términos gatillo del detector de omisión.
# Las claves son los stems internos del vocabulario (buscador_birads.py);
# los valores son su forma presentable para un clínico.
ETIQUETAS_EVIDENCIA = {
    # hallazgos
    "nodulo": "nódulo",
    "masa": "masa",
    "microcalcificaci": "microcalcificaciones",
    "calcificaci": "calcificaciones",
    "lesion": "lesión",
    "realce": "realce",
    "distorsion": "distorsión de la arquitectura",
    "asimetria": "asimetría",
    "ectasia": "ectasia ductal",
    "adenopatia": "adenopatía",
    "espiculad": "márgenes espiculados",
    "sospechos": "hallazgo sospechoso",
    "papilar": "lesión papilar",
    "quistic": "imagen quística",
    "dilatacion ductal": "dilatación ductal",
    "foco de realce": "foco de realce",
    # acciones / recomendaciones
    "se sugiere": "«se sugiere…»",
    "se recomienda": "«se recomienda…»",
    "se aconseja": "«se aconseja…»",
    "sugerimos": "«sugerimos…»",
    "biopsia": "biopsia",
    "histolog": "estudio histológico",
    "puncion": "punción",
    "correlacion": "correlación",
    "recategoriz": "recategorización",
    "control": "control / seguimiento",
    "seguimiento": "seguimiento",
    "derivar": "derivación",
    "derivacion": "derivación",
    "estudio complementario": "estudio complementario",
}


# =============================================================================
# FUNCIONES DE FORMATO
# =============================================================================

def etiqueta_evidencia(termino: str) -> str:
    """Convierte un término gatillo interno en una etiqueta legible."""
    if termino is None:
        return "—"
    return ETIQUETAS_EVIDENCIA.get(termino, termino.replace("_", " "))

def obtener_estilo_resultado(resultado: Dict[str, Any]) -> Dict[str, str]:
    """Selecciona la paleta de colores según el estado del cotejo."""
    estado = resultado.get("cotejo_acr", {}).get("estado", "")
    severidad = resultado.get("cotejo_acr", {}).get("severidad")
    alerta = resultado.get("cotejo_acr", {}).get("alerta", False)

    if estado == "error":
        return COLOR_ERROR

    if estado == "revision_extraccion":
        return COLOR_REVISION

    if alerta and severidad in COLORES_SEVERIDAD:
        return COLORES_SEVERIDAD[severidad]

    if estado == "coherente_con_precaucion":
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


def _color_solido_y_grupo(resultado: Dict[str, Any]):
    """Devuelve (color_solido, texto_blanco, grupo, subtitulo, titulo) para el
    badge estilo mockup: badge de color pleno con texto blanco."""
    estado = resultado.get("cotejo_acr", {}).get("estado", "")
    severidad = resultado.get("cotejo_acr", {}).get("severidad")
    alerta = resultado.get("cotejo_acr", {}).get("alerta", False)

    if estado == "revision_extraccion":
        return ("#D97B29", "REQUIERE REVISIÓN",
                "El sistema no pudo leer el informe con confianza.",
                "REVISIÓN POR EXTRACCIÓN")
    if alerta:
        # color pleno por severidad
        col = {"critica": "#B03A2E", "alta": "#C0392B",
               "media": "#D97B29", "baja": "#B8860B"}.get(severidad, "#C0392B")
        sev = severidad.upper() if severidad else "?"
        return (col, "REQUIERE REVISIÓN",
                "La recomendación no es coherente con el BI-RADS.",
                f"INCOHERENTE · {sev}")
    return ("#2E8B70", "SIN ALERTA",
            "La recomendación es coherente con el BI-RADS.",
            "COHERENTE")


def banner_resultado_html(resultado: Dict[str, Any]) -> str:
    """Banner de resultado estilo mockup: rótulo de grupo, badge de color pleno
    con el desenlace, y subtítulo explicativo. HTML en una línea."""
    color, grupo, subtitulo, titulo = _color_solido_y_grupo(resultado)
    return (
        '<div style="margin-bottom:0.5rem;">'
        f'<div style="font-size:12px; letter-spacing:.5px; color:#6B7B84; '
        f'text-transform:uppercase; margin-bottom:6px;">{grupo}</div>'
        f'<div style="background:{color}; padding:16px 20px; border-radius:10px;">'
        f'<span style="font-size:22px; font-weight:700; color:white; '
        f'letter-spacing:.5px;">{titulo}</span></div>'
        f'<div style="font-size:13px; color:#6B7B84; font-style:italic; '
        f'margin-top:6px;">{subtitulo}</div>'
        '</div>'
    )


def tarjetas_campos_html(birads, detectada, esperada) -> str:
    """Tres tarjetas: BI-RADS extraído, recomendación detectada, esperada ACR.
    HTML en una sola línea (sin indentación) para que Streamlit lo renderice
    en vez de tratarlo como bloque de código."""
    def celda(etq, val):
        val = val if (val is not None and str(val).strip()) else "—"
        return (
            '<div style="flex:1; background:white; border:1px solid #E3E8EB; '
            'border-radius:8px; padding:10px 14px;">'
            f'<div style="font-size:12px; color:#6B7B84; margin-bottom:2px;">{etq}</div>'
            f'<div style="font-size:16px; font-weight:600; color:#1F2A30;">{val}</div>'
            '</div>'
        )
    return (
        '<div style="display:flex; gap:10px; margin:14px 0;">'
        + celda("BI-RADS extraído", birads)
        + celda("Recomendación detectada", detectada)
        + celda("Esperada por ACR", esperada)
        + '</div>'
    )


def panel_detalle_html(mensaje: str) -> str:
    """Panel de detalle con fondo azul claro. HTML en una línea para render."""
    return (
        '<div style="background:#EAF2F5; border:1px solid #B8D2DC; '
        'border-radius:8px; padding:14px 16px; margin-bottom:12px;">'
        f'<div style="font-size:14px; color:#274B57;">{mensaje}</div>'
        '</div>'
    )


def formato_confianza_ml(confianza: float) -> str:
    """Formatea la confianza del ML como porcentaje legible."""
    if confianza is None:
        return "—"
    return f"{confianza:.2%}"


def panel_omision_html(resultado: Dict[str, Any]) -> str:
    """Genera el HTML del panel destacado para una alerta de omisión de BI-RADS.

    Se usa cuando estado_procesamiento == 'alerta_omision': el informe tiene
    hallazgos y/o recomendaciones pero NO declara una categoría BI-RADS.
    """
    cotejo = resultado.get("cotejo_acr", {})
    severidad = cotejo.get("severidad") or "alta"
    estilo = COLORES_SEVERIDAD.get(severidad, COLORES_SEVERIDAD["alta"])
    informe_id = resultado.get("informe_id") or "sin_id"
    mensaje = cotejo.get("mensaje", "")

    return f"""
    <div style="background: {estilo['bg']}; border-left: 6px solid {estilo['border']};
                padding: 16px 18px; border-radius: 8px; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 26px;">{estilo['icon']}</span>
            <div>
                <div style="font-size: 17px; font-weight: 600; color: {estilo['text']};">
                    Omisión de categoría BI-RADS · Severidad {severidad.upper()}
                </div>
                <div style="font-size: 12px; color: {estilo['border']};">
                    {informe_id}
                </div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 14px; color: {estilo['text']};">
            {mensaje}
        </div>
    </div>
    """
