"""
Tabla normativa ACR para cotejo BI-RADS vs Recomendación.

Este módulo contiene únicamente CONSTANTES (tabla de recomendaciones esperadas
por BI-RADS, equivalentes aceptables, niveles de severidad). Está separado
del código de lógica para facilitar mantenimiento y revisión clínica.

Tabla construida con base en:
- BI-RADS Atlas ACR (American College of Radiology) 5ta edición
- Validación clínica del proyecto BME513 (Universidad de Valparaíso)
- Análisis empírico sobre corpus de Vázquez Noguera et al. (2025)

Validado sobre 4 347 informes:
- 44 alertas reales (1.0% del corpus)
- 2 alertas críticas (BI-RADS 5 sin biopsia ni derivación)
- 34 alertas altas (BI-RADS 0 y BI-RADS 4)
- 8 alertas medias (BI-RADS 3)

Autor: Sebastián Inostroza Hurtado
Fecha: Junio 2026
"""

from typing import Dict, List


# =============================================================================
# 1. TABLA NORMATIVA ACR
# =============================================================================
#
# Estructura de cada entrada:
#   - esperada:                  categoría que ACR define como conducta estándar
#   - equivalentes_aceptables:   categorías que NO generan alerta (alternativas válidas)
#   - equivalentes_con_notificacion: categorías que generan notificación suave
#                                    (no inconsistencia, pero vale la pena mencionar)
#   - severidad:                 nivel de alerta si NO se cumple la conducta esperada
#   - descripcion:               significado clínico del BI-RADS
#
# Severidades:
#   - 'critica': riesgo grave de retraso diagnóstico (BI-RADS 5)
#   - 'alta':    inconsistencia clínica relevante (BI-RADS 0, 4)
#   - 'media':   conducta no óptima pero no urgente (BI-RADS 3, 6)
#   - 'baja':    desviación menor del protocolo (BI-RADS 1, 2)

TABLA_ACR: Dict[int, Dict] = {
    0: {
        "esperada": "estudio_complementario_imagen",
        "equivalentes_aceptables": [
            "correlacion_ecografica",         # implica ecografía a corto plazo
            "comparacion_estudios_previos",   # buscar info existente para completar
        ],
        "equivalentes_con_notificacion": [],
        "severidad": "alta",
        "descripcion": (
            "Estudio incompleto: el radiólogo no puede emitir conclusión "
            "definitiva. Requiere completar con estudios adicionales pronto "
            "(días a pocas semanas). Plazos largos (control a 6 meses) NO "
            "son compatibles con esta categoría."
        ),
    },
    1: {
        "esperada": "control_anual",
        "equivalentes_aceptables": [
            "criterio_medico",                 # delegación válida en hallazgo normal
            "correlacion_ecografica",          # ecografía por mamas densas: práctica estándar
            "estudio_complementario_imagen",   # imagen complementaria por densidad: aceptable
        ],
        "equivalentes_con_notificacion": [],
        # Conductas que, pese a ser "más agresivas" que el control anual, se marcan
        # para revisión: un estudio SIN hallazgos no debería motivar vigilancia
        # estrecha ni acción invasiva (posible inconsistencia con la categoría).
        "marcar_revision": {
            "control_corto_plazo": "baja",
            "biopsia_histologia": "media",
            "derivacion_oncologica": "media",
        },
        "severidad": "baja",
        "descripcion": (
            "Sin hallazgos. Seguimiento rutinario en 12 meses."
        ),
    },
    2: {
        "esperada": "control_anual",
        "equivalentes_aceptables": [
            "correlacion_ecografica",          # precaución adicional, no inadecuada
            "estudio_complementario_imagen",   # imagen complementaria por densidad: aceptable
            "criterio_medico",
        ],
        "equivalentes_con_notificacion": [
            "control_corto_plazo",  # excesivo para benigno definitivo, pero no peligroso
        ],
        # Simetría con BI-RADS 1: acciones invasivas sobre un hallazgo benigno
        # definitivo se marcan para revisión (posible inconsistencia). Se corrobora
        # con 2 casos reales de BI-RADS 2 + biopsia en el corpus.
        "marcar_revision": {
            "biopsia_histologia": "media",
            "derivacion_oncologica": "media",
        },
        "severidad": "baja",
        "descripcion": (
            "Hallazgos benignos definitivos. Seguimiento rutinario en 12 meses."
        ),
    },
    3: {
        "esperada": "control_corto_plazo",
        "equivalentes_aceptables": [
            "biopsia_histologia",  # conducta más agresiva, no es error clínico
        ],
        "equivalentes_con_notificacion": [],
        "severidad": "media",
        "descripcion": (
            "Probablemente benigno (VPP <2%). Requiere vigilancia activa "
            "con control a corto plazo (típicamente 6 meses)."
        ),
    },
    4: {
        "esperada": "biopsia_histologia",
        "equivalentes_aceptables": [
            "derivacion_oncologica",  # especialista ordenará biopsia
        ],
        "equivalentes_con_notificacion": [],
        "severidad": "alta",
        "descripcion": (
            "Sospecha activa de malignidad (VPP 2-95%, según subcategoría). "
            "Requiere confirmación histológica mediante biopsia."
        ),
    },
    5: {
        "esperada": "biopsia_histologia",
        "equivalentes_aceptables": [
            "derivacion_oncologica",  # derivar al especialista que ordenará biopsia y manejo
        ],
        "equivalentes_con_notificacion": [],
        "severidad": "critica",
        "descripcion": (
            "Altamente sospechoso de malignidad (VPP >95%). Requiere biopsia "
            "urgente o derivación a especialista para su manejo. Recomendaciones "
            "que no impliquen acción diagnóstica/derivación generan alerta crítica "
            "por riesgo de retraso diagnóstico grave."
        ),
    },
    6: {
        "esperada": "derivacion_oncologica",
        "equivalentes_aceptables": [
            "criterio_medico",       # paciente ya en manejo, decisión delegada
            "biopsia_histologia",    # podría tratarse de re-biopsia
        ],
        "equivalentes_con_notificacion": [],
        "severidad": "media",
        "descripcion": (
            "Malignidad confirmada histológicamente. El manejo oncológico "
            "típicamente ya está en curso al momento del informe."
        ),
    },
}


# =============================================================================
# 2. POSICIÓN DE CADA CATEGORÍA EN LA JERARQUÍA DE URGENCIA CLÍNICA
# =============================================================================
#
# Importada desde vocabulario_clinico.py. Replicada aquí como referencia para
# que tabla_acr.py sea autocontenido y la lógica de cotejo no tenga que
# importar de dos archivos.
#
# Más bajo el número = más urgente.

JERARQUIA_URGENCIA: List[str] = [
    "biopsia_histologia",            # 1. acción diagnóstica más urgente
    "derivacion_oncologica",         # 2. manejo especializado
    "estudio_complementario_imagen", # 3. duda diagnóstica - resolver pronto
    "correlacion_ecografica",        # 4. duda diagnóstica - resolver pronto
    "comparacion_estudios_previos",  # 5. duda diagnóstica - resolver pronto
    "control_corto_plazo",           # 6. plan definido a 6 meses
    "control_anual",                 # 7. plan definido a 12 meses
    "criterio_medico",               # 8. delegación de decisión
]


# =============================================================================
# 3. MENSAJES CLÍNICOS POR TIPO DE ALERTA
# =============================================================================
#
# Plantillas de mensaje que explican la inconsistencia detectada para cada
# combinación BI-RADS x severidad. Se usan en el reporte de alerta.

MENSAJES_INCONSISTENCIA: Dict[int, str] = {
    0: (
        "BI-RADS 0 indica que el estudio está incompleto y requiere "
        "completarse con imágenes adicionales o búsqueda de exámenes "
        "previos. La recomendación detectada no apunta a resolver el "
        "estudio incompleto a corto plazo."
    ),
    1: (
        "BI-RADS 1 (sin hallazgos) corresponde a control anual rutinario. "
        "La recomendación detectada se desvía del protocolo estándar."
    ),
    2: (
        "BI-RADS 2 (benigno definitivo) corresponde a control anual "
        "rutinario. La recomendación detectada se desvía del protocolo "
        "estándar."
    ),
    3: (
        "BI-RADS 3 (probablemente benigno) requiere vigilancia activa con "
        "control a corto plazo. La recomendación detectada no implica el "
        "seguimiento estrecho que esta categoría exige."
    ),
    4: (
        "BI-RADS 4 (sospecha de malignidad) requiere confirmación histológica "
        "mediante biopsia. La recomendación detectada no contempla la "
        "biopsia ni derivación a especialista."
    ),
    5: (
        "BI-RADS 5 indica alta sospecha de malignidad (VPP >95%) y requiere "
        "biopsia urgente. La recomendación detectada no implica biopsia, "
        "lo cual puede ocasionar retraso diagnóstico grave. Esta "
        "contradicción interna del informe sugiere que la categorización "
        "BI-RADS 5 podría no reflejar adecuadamente el grado de certeza "
        "del radiólogo."
    ),
    6: (
        "BI-RADS 6 (malignidad confirmada) típicamente implica manejo "
        "oncológico ya iniciado. La recomendación detectada no menciona "
        "derivación ni manejo especializado."
    ),
}


# =============================================================================
# 4. SUGERENCIAS GENÉRICAS PARA EL REVISOR
# =============================================================================
#
# Recordatorios al médico revisor cuando se genera una alerta. NO son
# prescripciones médicas, solo recordatorios de proceso.

SUGERENCIAS_GENERICAS: List[str] = [
    "Considerar revisión del caso por el radiólogo emisor para verificar "
    "coherencia entre la conclusión BI-RADS y la recomendación emitida.",
    
    "Verificar si la categorización BI-RADS refleja con precisión los "
    "hallazgos descritos en el cuerpo del informe.",
    
    "Esta alerta no constituye prescripción médica. Toda conducta "
    "diagnóstica o terapéutica es responsabilidad del médico tratante.",
]
