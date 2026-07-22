"""
Página 1 — Informe Individual

Permite procesar UN informe mamográfico de tres formas:
1. Pegar texto en un textarea
2. Subir un archivo TXT
3. Subir un archivo PDF

Muestra el resultado con colores tipo semáforo según severidad.
"""

import sys
import os
import json
from pathlib import Path

# Asegurar que el directorio raíz del proyecto está en el path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.predict import procesar_informe, _leer_pdf
from dashboard.utils.formato import (
    obtener_estilo_resultado,
    banner_resultado_html,
    tarjetas_campos_html,
    panel_detalle_html,
    panel_omision_html,
    etiqueta_estado,
    etiqueta_categoria,
    etiqueta_evidencia,
    etiqueta_verificacion_ml,
    formato_confianza_ml,
)


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Informe Individual - Auditoría Mamográfica",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Procesamiento de informe individual")
st.caption(
    "Procesa UN informe mamográfico. Acepta texto pegado, archivo TXT o PDF."
)

# =============================================================================
# INICIALIZAR SESSION STATE
# =============================================================================

if "resultado_actual" not in st.session_state:
    st.session_state["resultado_actual"] = None

if "texto_informe" not in st.session_state:
    st.session_state["texto_informe"] = ""

# =============================================================================
# SECCIÓN 1 - MODO DE ENTRADA
# =============================================================================

st.markdown("### 1. Modo de entrada")

modo = st.radio(
    "Selecciona cómo quieres ingresar el informe",
    options=["📋 Pegar texto", "📄 Archivo TXT", "📕 Archivo PDF"],
    horizontal=True,
    label_visibility="collapsed",
)

# Variable para guardar el texto que finalmente se procesará
texto_informe = ""

# ---------- Modo A: Pegar texto ----------
if modo == "📋 Pegar texto":
    texto_pegado = st.text_area(
        "Pega aquí el texto completo del informe mamográfico",
        height=250,
        placeholder=(
            "INFORME DE MAMOGRAFIA\n\n"
            "HALLAZGOS: ...\n\n"
            "CONCLUSION: BI-RADS X - ...\n\n"
            "RECOMENDACIONES:\n"
            "- ..."
        ),
        key="textarea_informe",
    )
    if texto_pegado.strip():
        texto_informe = texto_pegado

# ---------- Modo B: Archivo TXT ----------
elif modo == "📄 Archivo TXT":
    archivo_txt = st.file_uploader(
        "Sube un archivo .txt con el informe",
        type=["txt"],
        accept_multiple_files=False,
        key="uploader_txt",
    )
    if archivo_txt is not None:
        try:
            contenido = archivo_txt.read().decode("utf-8")
            texto_informe = contenido
            with st.expander("Ver contenido del archivo cargado"):
                st.text(contenido[:2000] + ("..." if len(contenido) > 2000 else ""))
        except UnicodeDecodeError:
            st.error(
                "El archivo no se pudo decodificar como UTF-8. "
                "Asegúrate de que sea un archivo de texto válido."
            )

# ---------- Modo C: Archivo PDF ----------
elif modo == "📕 Archivo PDF":
    archivo_pdf = st.file_uploader(
        "Sube un archivo .pdf con el informe",
        type=["pdf"],
        accept_multiple_files=False,
        key="uploader_pdf",
    )
    if archivo_pdf is not None:
        # Guardar temporalmente y leer con pdfplumber
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(archivo_pdf.getvalue())
                tmp_path = tmp.name

            with st.spinner("Extrayendo texto del PDF..."):
                contenido = _leer_pdf(tmp_path)

            os.unlink(tmp_path)
            texto_informe = contenido

            with st.expander("Ver texto extraído del PDF"):
                st.text(contenido[:2000] + ("..." if len(contenido) > 2000 else ""))
        except ImportError:
            st.error(
                "pdfplumber no está instalado. Ejecuta: pip install pdfplumber"
            )
        except ValueError as e:
            st.error(
                f"No se pudo extraer texto del PDF: {e}"
            )
        except Exception as e:
            st.error(f"Error inesperado al leer el PDF: {e}")

st.divider()

# =============================================================================
# SECCIÓN 2 - OPCIONES Y BOTÓN PROCESAR
# =============================================================================

st.markdown("### 2. Opciones de procesamiento")

col_opc1, col_opc2 = st.columns([1, 1])

with col_opc1:
    informe_id = st.text_input(
        "ID del informe (opcional)",
        placeholder="paciente_12345",
        help="Identificador para auditoría. Si se omite, se genera automáticamente.",
    )

with col_opc2:
    usar_ml = st.checkbox(
        "Usar verificador ML (DistilBETO)",
        value=True,
        help="Si está activo, ejecuta la verificación dual de extracción. "
             "Añade ~500ms de procesamiento.",
    )
    usar_ner = st.checkbox(
        "Usar extractor NER (respaldo IA)",
        value=True,
        help="Si la extracción por reglas no encuentra la recomendación, "
             "intenta con el modelo NER entrenado (DistilBETO). Requiere el "
             "modelo en models/ner_recomendacion_final.",
    )

# Botón principal
st.write("")
boton_procesar = st.button(
    "🔍 Procesar informe",
    type="primary",
    use_container_width=False,
    disabled=(not texto_informe.strip()),
)

if not texto_informe.strip() and modo == "📋 Pegar texto":
    st.caption("Pega un informe arriba para habilitar el botón")
elif not texto_informe.strip():
    st.caption("Sube un archivo para habilitar el botón")

# =============================================================================
# SECCIÓN 3 - PROCESAR Y MOSTRAR RESULTADO
# =============================================================================

if boton_procesar and texto_informe.strip():
    with st.spinner("Procesando informe..."):
        try:
            resultado = procesar_informe(
                full_report=texto_informe,
                informe_id=informe_id if informe_id.strip() else None,
                usar_verificador_ml=usar_ml,
                usar_ner_recomendacion=usar_ner,
            )
            st.session_state["resultado_actual"] = resultado
        except Exception as e:
            st.error(f"Error inesperado al procesar el informe: {e}")
            st.session_state["resultado_actual"] = None

# Mostrar el resultado si existe
if st.session_state["resultado_actual"] is not None:
    resultado = st.session_state["resultado_actual"]

    st.divider()
    st.markdown("### 3. Resultado")

    # ---------- Vista según el tipo de resultado ----------
    estado_proc = resultado.get("estado_procesamiento")

    if estado_proc == "alerta_omision":
        # === OMISIÓN: hay hallazgos/recomendación pero NO hay BI-RADS asignado ===
        st.markdown(panel_omision_html(resultado), unsafe_allow_html=True)

        evidencia = (
            resultado.get("verificacion_ml", {})
            .get("alerta_omision", {})
            .get("evidencia", {})
        )
        hallazgos = evidencia.get("hallazgos_detectados", [])
        acciones = evidencia.get("acciones_detectadas", [])

        st.markdown("#### ¿Por qué se marcó como omisión?")
        st.caption(
            "El informe describe contenido diagnóstico pero no declara una "
            "categoría BI-RADS. Señales detectadas en el texto:"
        )
        col_ev1, col_ev2 = st.columns(2)
        with col_ev1:
            st.markdown("**Hallazgos radiológicos**")
            if hallazgos:
                for h in hallazgos:
                    st.markdown(f"- {etiqueta_evidencia(h)}")
            else:
                st.caption("—")
        with col_ev2:
            st.markdown("**Acciones / recomendaciones**")
            if acciones:
                for a in acciones:
                    st.markdown(f"- {etiqueta_evidencia(a)}")
            else:
                st.caption("—")

        accion_sugerida = resultado.get("cotejo_acr", {}).get("accion_sugerida", "")
        if accion_sugerida:
            st.markdown("#### Acción sugerida")
            st.warning(accion_sugerida)

    elif estado_proc == "error":
        # === ERROR de procesamiento no recuperable ===
        st.markdown(banner_resultado_html(resultado), unsafe_allow_html=True)
        st.error(
            f"El procesamiento falló en el paso "
            f"**{resultado.get('paso_fallido', '?')}** "
            f"({resultado.get('error_tipo', 'Error')})."
        )
        if resultado.get("error_mensaje"):
            st.caption(resultado["error_mensaje"])

    else:
        # === RESULTADO NORMAL (coherente o alerta de incoherencia) ===
        # Diseño estilo tarjeta: badge + tres campos + panel de detalle
        st.markdown(banner_resultado_html(resultado), unsafe_allow_html=True)

        st.markdown(
            tarjetas_campos_html(
                birads=resultado["birads"]["valor"],
                detectada=etiqueta_categoria(resultado["recomendacion"]["categoria_principal"]),
                esperada=etiqueta_categoria(resultado["cotejo_acr"]["recomendacion_esperada"]),
            ),
            unsafe_allow_html=True,
        )

        st.markdown("**Detalle:**")
        st.markdown(
            panel_detalle_html(resultado["cotejo_acr"]["mensaje"]),
            unsafe_allow_html=True,
        )

        # Indicador de la fuente de extracción de la recomendación
        _fuente_rec = resultado["recomendacion"].get("fuente_extraccion")
        if _fuente_rec == "ner_distilbeto":
            st.caption("🤖 Recomendación localizada por el extractor NER (IA), "
                       "tras no hallarla las reglas.")
        elif _fuente_rec in ("regex_full_report", "regex_frases_gatillo",
                             "columna_recommendations"):
            st.caption("📐 Recomendación localizada por reglas (regex).")

        # ---------- Apoyo de extracción de la recomendación (NER), si aplica ----------
        _dual = resultado["recomendacion"].get("extraccion_dual") or {}
        if _dual.get("ner_solicitado"):
            st.markdown("#### Apoyo de extracción de la recomendación (NER)")
            st.caption(
                "Las reglas (regex) son la vía primaria. El NER (IA) es un respaldo "
                "que localiza la recomendación cuando las reglas no la encuentran, y "
                "permite contrastar ambas lecturas."
            )
            _CONCORD = {
                "concuerdan": "✅ Reglas y NER concuerdan",
                "difieren":   "⚠️ Reglas y NER difieren",
                "solo_regex": "📐 Solo las reglas la hallaron",
                "solo_ner":   "🤖 Solo el NER la halló",
                "ninguno":    "🔍 Ninguno la halló",
            }
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                span_r = _dual.get("regex_span") or "—"
                st.markdown("**Extracción por reglas**")
                st.markdown(f"<span style='color:#274B57'>{span_r[:80]}</span>",
                            unsafe_allow_html=True)
            with col_e2:
                if not _dual.get("ner_disponible"):
                    span_n = "modelo no disponible"
                else:
                    span_n = _dual.get("ner_span") or "no localizó"
                st.markdown("**Extracción por NER (IA)**")
                st.markdown(f"<span style='color:#274B57'>{span_n[:80]}</span>",
                            unsafe_allow_html=True)
            with col_e3:
                st.markdown("**Concordancia**")
                st.markdown(_CONCORD.get(_dual.get("concordancia"), "—"))
            fuente_usada = _dual.get("fuente_usada")
            if fuente_usada == "ner":
                st.caption("La recomendación usada fue la del **NER** (las reglas no la hallaron).")
            elif fuente_usada == "regex":
                st.caption("La recomendación usada fue la de las **reglas** (vía primaria).")

        # ---------- Apoyo de lectura del BI-RADS (ML), si aplica ----------
        if resultado["verificacion_ml"]["estado"] != "no_ejecutado":
            st.markdown("#### Apoyo de lectura del BI-RADS (ML)")
            st.caption(
                "El ML es un apoyo de lectura: refuerza o cuestiona la extracción "
                "literal, que es la autoridad. No emite juicio clínico."
            )

            col_v1, col_v2, col_v3 = st.columns(3)

            with col_v1:
                st.metric(
                    "Lectura literal (regex)",
                    f"BI-RADS {resultado['birads']['valor']}",
                    delta=resultado["birads"]["confianza"],
                    delta_color="off",
                )

            with col_v2:
                ml_birads = resultado["verificacion_ml"]["birads_ml"]
                ml_conf = resultado["verificacion_ml"].get("confianza_ml")
                st.metric(
                    "Lectura de apoyo (ML)",
                    f"BI-RADS {ml_birads}" if ml_birads is not None else "—",
                    delta=formato_confianza_ml(ml_conf),
                    delta_color="off",
                )

            with col_v3:
                # Concordancia como texto (st.metric trunca; usamos markdown)
                estado_ml = resultado["verificacion_ml"]["estado"]
                st.markdown("**Concordancia**")
                st.markdown(etiqueta_verificacion_ml(estado_ml))

            # Mensaje del apoyo de lectura
            if resultado["verificacion_ml"].get("mensaje"):
                st.caption(resultado["verificacion_ml"]["mensaje"])

    # ---------- Acciones ----------
    st.markdown("#### Acciones")

    col_act1, col_act2, col_act3 = st.columns([1, 1, 2])

    with col_act1:
        json_str = json.dumps(
            resultado, indent=2, ensure_ascii=False, default=str
        )
        st.download_button(
            label="📥 Descargar JSON",
            data=json_str,
            file_name=f"reporte_{resultado.get('informe_id', 'sin_id')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_act2:
        if st.button("🔄 Procesar otro informe", use_container_width=True):
            st.session_state["resultado_actual"] = None
            st.rerun()

    # ---------- JSON completo (expander) ----------
    with st.expander("Ver JSON completo del resultado"):
        st.json(resultado)

# =============================================================================
# AYUDA EN EL SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### Ayuda")
    st.markdown(
        """
        **Modos de entrada:**

        - **Pegar texto:** copia y pega el informe completo en el textarea
        - **Archivo TXT:** sube un archivo de texto plano
        - **Archivo PDF:** sube un PDF con texto digital (no escaneado)

        **Resultado del cotejo (auditoría binaria):**

        - **Coherente:** la recomendación coincide con lo esperado,
          es un equivalente aceptable o es más cauta. Sin alerta.
        - **Incoherente:** la recomendación se queda corta frente a
          la norma. Genera alerta, graduada por severidad.
        - **Revisión por extracción:** no se pudo extraer o clasificar
          la recomendación; se deriva a revisión humana.

        **Severidad de la incoherencia:**

        - 🚨 **Crítica:** BI-RADS 4 o 5 sin biopsia ni derivación
        - ⚠️ **Alta:** desvío relevante (BI-RADS 0 o 6)
        - ⚠️ **Media:** a revisar sin urgencia (p. ej. acción
          invasiva sobre un benigno)
        - ℹ️ **Baja:** incoherencia leve, sin riesgo para la paciente
        """
    )

    st.divider()

    st.markdown("### Sobre el sistema")
    st.markdown(
        """
        Pipeline de 4 módulos validado sobre el corpus de Vázquez Noguera et al. (2025).

        **Exactitud extracción BI-RADS:** 99.9%
        **Lectura BI-RADS · apoyo ML (CV):** ≈0.89
        **Tasa de incoherencias:** 1.1%

        La extracción por reglas es el lector primario. El apoyo ML es un segundo
        canal de lectura independiente (mide lectura, no juicio clínico).
        """
    )
