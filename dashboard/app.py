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

        ### Pipeline de módulos

        1. **Extractor BI-RADS** (regex + buscador híbrido sobre la conclusión)
        2. **Apoyo de lectura ML** (DistilBETO relee y refuerza la extracción del BI-RADS)
        3. **Extractor de recomendaciones** (reglas clínicas + sinónimos, con **extractor NER** —DistilBETO— como respaldo para redacciones no vistas)
        4. **Motor de cotejo ACR** (tabla normativa según estándar BI-RADS/ACR)

        La arquitectura es **híbrida**: las reglas, transparentes y auditables, son
        la vía primaria; la IA (NER) aporta robustez ante redacciones nuevas cuando
        las reglas no localizan la recomendación. Cuando ninguno resuelve con
        confianza, el caso se deriva a **revisión humana**.

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
        "Corpus público en español (Vázquez Noguera et al., 2025, origen "
        "paraguayo). Despliegue orientado al contexto chileno.",
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
    # Medido vs. la etiqueta BI-RADS del corpus.
    st.metric("Exactitud extracción BI-RADS", "99.9%")

with m2:
    # NER de recomendación (test deduplicado, notebook 11). Extracción de span.
    st.metric("Extractor NER recomendación (F1)", "0.999")

with m3:
    # Cifra medida sobre el corpus completo con el pipeline actual: 50/4357.
    st.metric("Tasa de incoherencias", "1.15%")

with m4:
    st.metric("Procesamiento", "local")

st.caption(
    "Exactitud de extracción de BI-RADS medida contra la etiqueta del corpus "
    "(Vázquez Noguera et al., 2025). El extractor NER de recomendaciones alcanza "
    "F1≈0.999 en test deduplicado; nota: el corpus es homogéneo, por lo que la "
    "generalización real se evalúa con informes externos. La arquitectura híbrida "
    "combina reglas transparentes con IA de respaldo, manteniendo al humano en el bucle."
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
