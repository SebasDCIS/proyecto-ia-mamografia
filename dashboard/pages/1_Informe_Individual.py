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
from src.recursos.limpieza_informe import limpiar_informe
from dashboard.utils.formato import (
    obtener_estilo_resultado,
    banner_resultado_html,
    tarjetas_campos_html,
    panel_detalle_html,
    panel_omision_html,
    etiqueta_estado,
    etiqueta_categoria,
    etiqueta_evidencia,
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
            with st.expander("Ver contenido del archivo cargado (anonimizado)"):
                st.caption(
                    "Se muestra el texto tras la capa de limpieza. Los datos "
                    "identificatorios aparecen como [MEDICO], [RUT] o "
                    "[DATO_PACIENTE]: la pantalla no debe exponerlos."
                )
                _prev, _, _ = limpiar_informe(contenido)
                st.text(_prev[:2000] + ("..." if len(_prev) > 2000 else ""))
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

            with st.expander("Ver texto extraído del PDF (anonimizado)"):
                st.caption(
                    "Se muestra el texto tras la capa de limpieza. Los datos "
                    "identificatorios aparecen como [MEDICO], [RUT] o "
                    "[DATO_PACIENTE]: la pantalla no debe exponerlos."
                )
                _prev, _, _ = limpiar_informe(contenido)
                st.text(_prev[:2000] + ("..." if len(_prev) > 2000 else ""))
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
    # El verificador ML del Módulo 4 no se ofrece en la interfaz clínica.
    # Se midió por cuatro vías que no aporta sobre la extracción reglada: la
    # ablación lo derrumba de 0,939 a 0,544 y un regex de una línea lo empata
    # sobre su propia ventana. Mostrar su lectura sugeriría una confirmación
    # independiente que no existe, porque lee la misma cadena que el regex ya
    # extrajo. Para reproducir su evaluación: python -m src.predict --con-ml
    # (ver docs/BITACORA.md).
    usar_ml = False

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
                "contenido_parcial": "⚠️ Una vía extrajo de más (revisar spans)",
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

            # Panel de depuración: permite verificar qué texto EXACTO se procesó
            with st.expander("🔧 Depuración de extracción (verificar texto)"):
                st.caption(
                    "El NER solo puede devolver palabras presentes en el texto de "
                    "entrada. Usa esto para confirmar qué se procesó realmente."
                )
                # IMPORTANTE: se muestra el texto YA LIMPIO, no el crudo. El
                # panel mostraba la entrada original y por tanto exponía el
                # nombre del radiólogo y el RUT que la capa de limpieza sí
                # redacta. Lo que interesa depurar es lo que el pipeline
                # realmente procesó.
                _txt_proc, _hubo_limpieza, _elim = limpiar_informe(texto_informe)
                st.markdown(f"**Largo del texto:** {len(texto_informe)} caracteres de entrada, "
                            f"{len(_txt_proc)} tras la limpieza")
                st.markdown("**Últimos 300 caracteres del informe YA PROCESADO** "
                            "(anonimizado; aquí se ve si quedó pie de página o descargo):")
                st.code(_txt_proc[-300:] if len(_txt_proc) > 300 else _txt_proc)
                if _elim:
                    with st.expander(f"Ver los {len(_elim)} fragmentos retirados por la capa de privacidad"):
                        st.caption(
                            "Se listan solo las ETIQUETAS de lo retirado, no su contenido, "
                            "para no reintroducir el dato personal en pantalla."
                        )
                        _tipos = {}
                        for _e in _elim:
                            _k = ("RUT" if any(c.isdigit() for c in _e) and "-" in _e
                                  else "nombre o firma" if len(_e.split()) <= 6
                                  else "línea de ruido")
                            _tipos[_k] = _tipos.get(_k, 0) + 1
                        for _k, _v in _tipos.items():
                            st.markdown(f"- {_k}: {_v}")
                st.markdown(f"**Span extraído por reglas:** `{_dual.get('regex_span') or '—'}`")
                st.markdown(f"**Span extraído por NER:** `{_dual.get('ner_span') or '—'}`")
                _en_texto = (_dual.get("ner_span","").split()[0].lower() in _txt_proc.lower()
                             if _dual.get("ner_span") else False)
                st.markdown(f"**¿El span del NER está en el texto de entrada?:** "
                            f"{'✅ Sí' if _en_texto else '❌ No — posible resultado en caché, reprocesa'}")

        # El panel del verificador ML se retiró de la interfaz clínica junto con
        # el módulo. Ver el comentario en la sección de opciones.

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

        **Extracción BI-RADS (Macro F1):** 0.9995
        **Localización de la recomendación · NER (F1 de span):** 0.9991
        **Tasa de incoherencias:** 1.15% (50 de 4 357, 19 críticas)

        La extracción por reglas es la única autoridad del sistema. Se entrenó un
        verificador DistilBETO para contrastarla y se retiró tras medir por cuatro
        vías que no aporta: la ablación lo derrumba de 0.939 a 0.544, y un regex de
        una línea lo empata sobre su propia ventana.
        """
    )
