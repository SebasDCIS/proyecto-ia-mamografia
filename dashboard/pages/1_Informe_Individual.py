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
    etiqueta_estado,
    etiqueta_categoria,
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

    # ---------- Banner principal ----------
    banner = banner_resultado_html(resultado)
    st.markdown(banner, unsafe_allow_html=True)

    # ---------- Resumen del caso ----------
    st.markdown("#### Resumen del caso")

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.markdown(
            f"""
            - **BI-RADS declarado:** {resultado['birads']['valor']}
            - **Confianza extracción:** {resultado['birads']['confianza']}
            - **Fuente:** `{resultado['birads']['fuente']}`
            """
        )

    with col_res2:
        st.markdown(
            f"""
            - **Recomendación esperada (ACR):** {etiqueta_categoria(resultado['cotejo_acr']['recomendacion_esperada'])}
            - **Recomendación detectada:** {etiqueta_categoria(resultado['recomendacion']['categoria_principal'])}
            - **Confiabilidad técnica:** {resultado['confiabilidad_tecnica_global']}
            """
        )

    # ---------- Mensaje clínico ----------
    st.markdown("#### Mensaje clínico")
    st.info(resultado["cotejo_acr"]["mensaje"])

    # ---------- Verificación dual (si aplica) ----------
    if resultado["verificacion_ml"]["estado"] != "no_ejecutado":
        st.markdown("#### Verificación dual de extracción")

        col_v1, col_v2, col_v3 = st.columns(3)

        with col_v1:
            st.metric(
                "Regex",
                f"BI-RADS {resultado['birads']['valor']}",
                delta=resultado["birads"]["confianza"],
                delta_color="off",
            )

        with col_v2:
            ml_birads = resultado["verificacion_ml"]["birads_ml"]
            ml_conf = resultado["verificacion_ml"]["confianza_ml"]
            st.metric(
                "DistilBETO (ML)",
                f"BI-RADS {ml_birads}" if ml_birads is not None else "—",
                delta=formato_confianza_ml(ml_conf),
                delta_color="off",
            )

        with col_v3:
            estado_ml = resultado["verificacion_ml"]["estado"]
            st.metric(
                "Estado",
                etiqueta_verificacion_ml(estado_ml),
            )

        # Mensaje de la verificación ML
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

        **Estados posibles del cotejo:**

        - **Coherente:** la recomendación coincide con lo esperado por ACR
        - **Coherente equivalente:** alternativa clínicamente aceptable
        - **Coherente con precaución:** conducta más conservadora
        - **Notificación suave:** caso atípico no crítico
        - **Inconsistencia:** alerta clínica que requiere revisión

        **Severidad de las alertas:**

        - 🚨 **Crítica:** BI-RADS 5/6 sin biopsia
        - ⚠️ **Alta:** BI-RADS 4 sin biopsia
        - ⚠️ **Media:** BI-RADS 0 con conducta inadecuada
        - ℹ️ **Baja:** notificaciones suaves
        """
    )

    st.divider()

    st.markdown("### Sobre el sistema")
    st.markdown(
        """
        Pipeline de 4 módulos validado sobre 4 357 informes.

        **Macro F1 extractor:** 0.9995
        **Macro F1 DistilBETO:** 0.9386
        **Tasa de alertas:** 1.33%
        """
    )
