"""
Dashboard de Auditoría de Informes Mamográficos
Sistema BME513 - Universidad de Valparaíso

Landing page principal con descripción del proyecto y navegación.
Ejecutar con: streamlit run dashboard/app.py
"""

import streamlit as st

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Auditoría Mamográfica - BME513",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Sistema de auditoría técnica de informes mamográficos en español "
            "desarrollado para el curso BME513 - Inteligencia Artificial en Salud "
            "(Doctorado en Ciencias e Ingeniería para la Salud, Universidad de "
            "Valparaíso)."
        )
    }
)

# =============================================================================
# CABECERA
# =============================================================================

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <span style="font-size: 36px;">🩺</span>
        <div>
            <h1 style="margin: 0; padding: 0; font-size: 28px;">
                Auditoría de Informes Mamográficos
            </h1>
            <p style="margin: 0; color: #666; font-size: 14px;">
                Sistema BME513 · Universidad de Valparaíso · Versión 1.0
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =============================================================================
# DESCRIPCIÓN DEL PROYECTO
# =============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### ¿Qué hace este sistema?

        Procesa informes mamográficos en español y **detecta inconsistencias técnicas**
        entre la categoría BI-RADS asignada por el radiólogo y la recomendación
        clínica emitida, según el estándar BI-RADS/ACR.

        Por ejemplo: si un informe declara **BI-RADS 5** (lesión altamente sospechosa)
        pero recomienda solo *control anual*, el sistema genera una **alerta crítica**
        para revisión humana.

        ### Pipeline de 4 módulos

        1. **Extractor BI-RADS** (regex sobre el bloque de conclusión)
        2. **Verificador ML** (DistilBETO como segunda opinión técnica)
        3. **Extractor de recomendaciones** (reglas clínicas + TF-IDF)
        4. **Motor de cotejo ACR** (tabla normativa adaptada a práctica chilena)

        ### Filosofía Human-on-the-Loop

        El sistema **detecta y reporta inconsistencias**, pero **no decide la conducta
        clínica**. Cada alerta incluye trazabilidad completa para que el revisor
        humano (radiólogo, médico tratante) tome la decisión final.
        """
    )

with col2:
    st.markdown("### Cómo usarlo")

    st.info(
        "**Informe individual** \n\n"
        "Pega un informe, sube un archivo TXT o un PDF. Resultado inmediato.",
        icon="📝"
    )

    st.success(
        "**Validado sobre 4 357 informes** \n\n"
        "Corpus público en español (Vázquez Noguera et al., 2025).",
        icon="✅"
    )

    st.warning(
        "**Procesamiento local** \n\n"
        "Sin envío de datos a servicios externos. Cumple Ley 19.628.",
        icon="🔒"
    )

st.divider()

# =============================================================================
# MÉTRICAS DEL SISTEMA
# =============================================================================

st.markdown("### Métricas del sistema")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Macro F1 extractor BI-RADS", "0.9995")

with m2:
    st.metric("Macro F1 DistilBETO", "0.9386")

with m3:
    st.metric("Tasa de alertas clínicas", "1.33%")

with m4:
    st.metric("Tiempo por informe", "~12 ms")

st.caption(
    "Métricas evaluadas sobre el corpus completo de 4 357 informes "
    "(Vázquez Noguera et al., 2025)."
)

st.divider()

# =============================================================================
# NAVEGACIÓN A PÁGINAS
# =============================================================================

st.markdown("### Páginas disponibles")

st.markdown(
    """
    Usa el menú lateral (izquierda) para navegar entre las páginas:

    - **📝 Informe Individual** — Procesar un informe (texto pegado, TXT o PDF)
    - **📁 Procesamiento Batch** *(próximamente)* — Procesar múltiples informes
    - **📊 Estadísticas del Corpus** *(próximamente)* — Visualización de resultados
    """
)

# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Desarrollado por Sebastián Inostroza Hurtado · "
    "Doctorado en Ciencias e Ingeniería para la Salud · "
    "Universidad de Valparaíso · Junio 2026 · "
    "[Repositorio en GitHub](https://github.com/SebasDCIS/proyecto-ia-mamografia)"
)
