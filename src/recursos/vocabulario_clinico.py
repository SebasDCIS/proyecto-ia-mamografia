"""
Vocabulario clínico para el extractor de recomendaciones mamográficas.

Este módulo contiene únicamente CONSTANTES (typos, patrones regex, jerarquía
clínica, frases de referencia para TF-IDF). Está separado del código de
lógica para facilitar el mantenimiento:

- Si se descubre un nuevo typo en otro corpus → agregar en TYPOS_CLINICOS
- Si se ve una nueva forma de escribir una categoría → agregar en PATRONES_POR_CATEGORIA
- Si cambia la jerarquía clínica → ajustar JERARQUIA_CLINICA
- Si se quiere agregar más frases de referencia → ampliar FRASES_REFERENCIA_TFIDF

Validado sobre el corpus de Vázquez Noguera et al. (2025):
    - 100% de cobertura regex sobre 4 347 recomendaciones
    - 8 categorías clínicas
    - 9 typos identificados

Autor: Sebastián Inostroza Hurtado
Fecha: Mayo 2026
"""

from typing import Dict, List


# =============================================================================
# 1. DICCIONARIO DE TYPOS CLÍNICOS DETECTADOS EN EL CORPUS
# =============================================================================

TYPOS_CLINICOS: Dict[str, str] = {
    r"\bcografia\b":           "ecografia",        # COGRAFIA: falta E inicial
    r"\bsuerimos\b":           "sugerimos",        # SUERIMOS: falta G
    r"\bsuegerimos\b":         "sugerimos",        # SUEGERIMOS: G extra
    r"\bmamografica\b":        "mamografico",      # género equivocado en frase
    r"\brecategorizaicon\b":   "recategorizacion", # transposición de letras
    r"\bmanual\b":             "anual",            # autocorrector ANUAL → MANUAL
    r"\banula\b":              "anual",            # orden de letras
    r"\btratatne\b":           "tratante",         # transposición
    r"\bcontro\b":             "control",          # falta L final
}


# =============================================================================
# 2. CATEGORÍAS CLÍNICAS Y SU SIGNIFICADO
# =============================================================================

CATEGORIAS_CLINICAS: Dict[str, str] = {
    "estudio_complementario_imagen": (
        "Hallazgo requiere otra técnica de imagen complementaria "
        "(ecografía, resonancia, magnificación, compresión focalizada)."
    ),
    "correlacion_ecografica": (
        "Duda diagnóstica que requiere ecografía complementaria a corto plazo."
    ),
    "comparacion_estudios_previos": (
        "Buscar exámenes anteriores del expediente para apreciar evolución."
    ),
    "control_anual": (
        "Seguimiento rutinario en 12 meses (mamografía y/o ecografía)."
    ),
    "control_corto_plazo": (
        "Vigilancia activa con próximo estudio en 3 a 6 meses."
    ),
    "biopsia_histologia": (
        "Confirmación tisular requerida (caracterización histológica, "
        "biopsia, estudio histológico)."
    ),
    "criterio_medico": (
        "Recomendación delegada al médico tratante o caso ambiguo que "
        "requiere supervisión clínica."
    ),
    "derivacion_oncologica": (
        "Manejo por especialista en oncología."
    ),
}


# =============================================================================
# 3. JERARQUÍA CLÍNICA PARA SELECCIÓN DE CATEGORÍA PRINCIPAL
# =============================================================================
#
# Principio: "anteponerse a lo peor". Las acciones diagnósticas urgentes
# priman sobre planes con plazo definido. Una solicitud de información
# adicional (correlación, comparación) señala duda diagnóstica activa,
# más urgente que un control ya programado.

JERARQUIA_CLINICA: List[str] = [
    "biopsia_histologia",            # 1. acción más urgente (diagnóstico definitivo)
    "derivacion_oncologica",         # 2. manejo especializado
    "estudio_complementario_imagen", # 3. duda diagnóstica - resolver pronto
    "correlacion_ecografica",        # 4. duda diagnóstica - resolver pronto
    "comparacion_estudios_previos",  # 5. duda diagnóstica - resolver pronto
    "control_corto_plazo",           # 6. plan definido a 6 meses
    "control_anual",                 # 7. plan definido a 12 meses
    "criterio_medico",               # 8. delegación de decisión
]


# =============================================================================
# 4. PATRONES REGEX POR CATEGORÍA (sobre texto YA normalizado)
# =============================================================================
#
# Estos patrones operan sobre texto que ya pasó por:
#   1. NFKD (sin tildes)
#   2. Minúsculas
#   3. Corrección de typos del diccionario
#
# Validado: cobertura 100% sobre el corpus de 4 347 recomendaciones.

PATRONES_POR_CATEGORIA: Dict[str, List[str]] = {
    "estudio_complementario_imagen": [
        # Patrones específicos (no precedidos por "correlacion con")
        r"(?<!correlacion\scon\s)ecografia\s+mamaria\s+(\w+\s+)?(actualizada\s+)?para\s+(posterior\s+)?recategorizacion",
        r"ecografia\s+complementaria",
        r"estudio\s+ecografico\s+complementario",
        r"estudio\s+complementario\s+de\s+ecografia",
        r"(?<!correlacion\scon\s)ecografia\s+mamaria\s+(actualizada\s+)?(debido\s+al|por\s+el)\s+patron",
        # FIX T4: "actualizada" solo se considera estudio complementario si NO está precedido por "correlacion con"
        r"(?<!correlacion\scon\s)ecografia\s+mamaria\s+actualizada(?!\s+y\s+control)",
        r"complementar\s+(el\s+estudio\s+)?con\s+(una\s+)?ecografia",
        r"(?<!correlacion\scon\s)ecografia\s+mamaria\s+y\s+(de\s+la\s+region\s+)?axilar",
        r"sugerimos\s+ecografia\s+mamaria",
        r"(?<!correlacion\scon\s)ecografia\s+mamaria\s+bilateral",
        # FIX T3: patrón general para "ecografía mamaria" sin sufijo específico (cuando NO es correlación)
        r"sugiere\s+ecografia\s+mamaria\s+y\s+",
        r"sugiere\s+ecografia\s+mamaria\.",
        r"sugiere\s+ecografia\s+mamaria\s*$",
        r"compresion\s+focalizada",
        r"incidencias\s+con\s+(magnificacion|compresion)",
        r"complementar\s+con\s+(ecografia|rm)",
        r"\brm\b|resonancia\s+magnetica",
        r"magnificacion(es)?",
    ],
    "correlacion_ecografica": [
        r"correlacion\s+(con\s+)?ecograf",
        r"correlacionar\s+(este\s+estudio\s+)?con\s+ecograf",
    ],
    "comparacion_estudios_previos": [
        r"comparacion\s+con\s+estudios?\s+(anteriores?|previos?)",
        r"correlacion\s+con\s+estudios?\s+(anteriores?|previos?)",
        r"comparacion\s+con\s+(mamografia|estudios?)\s+(anteriores?|previos?)",
        r"correlacion\s+con\s+los\s+mismos",
        r"para\s+apreciar\s+evolucion",
    ],
    "control_anual": [
        r"control\s+mamografico\s+anual",
        r"control\s+anual",
        r"mamografia\s+de\s+control\s+anual",
        r"controles?\s+anuales?",
        r"control\s+mamografico\s+y\s+ecografico\s+anual",
        r"control\s+ecografico\s+y\s+mamografico\s+anual",
        r"controles?\s+mamografico[s]?\s+y\s+ecografico[s]?\s+anuales?",
        r"controles?\s+ecografico[s]?\s+y\s+mamografico[s]?\s+anuales?",
    ],
    "control_corto_plazo": [
        r"control\s+semestral",
        r"control\s+en\s+6\s+meses",
        r"control\s+en\s+seis\s+meses",
        r"control\s+a\s+los?\s+6\s+meses",
        r"seguimiento\s+a\s+6\s+meses",
        r"control\s+en\s+3\s+meses",
        r"control\s+ecografico\s+semestral",
        r"control\s+ecografico\s+en\s+(6|seis)\s+meses",
        r"control\s+ecografico\s+y\s+mamografia.{0,30}en\s+(6|seis)\s+meses",
    ],
    "biopsia_histologia": [
        r"caracterizacion\s+histologica",
        r"estudio\s+histologico",
        r"biopsia",
        r"muestra\s+histopatologica",
    ],
    "criterio_medico": [
        r"criterio\s+del\s+medico\s+tratante",
        r"segun\s+criterio\s+medico",
        r"decidir\s+conducta",
        r"controles\s+habituales",
        r"control\s+mamografico\s+bianual",
        r"control\s+bianual",
        r"\bbianual(es)?\b",
        r"controles?\s+a\s+corto\s+plazo",
        r"antecedentes\s+clinicos.{0,50}patologia\s+extramamaria",
        r"descartar\s+patologia\s+extramamaria",
    ],
    "derivacion_oncologica": [
        r"derivacion\s+a\s+(oncolog|especialista)",
        r"manejo\s+oncologico",
        r"evaluacion\s+oncologica",
    ],
}


# =============================================================================
# 5. FRASES DE REFERENCIA PARA SIMILITUD TF-IDF (capa de fallback)
# =============================================================================
#
# Cuando un texto NO cae en ninguna regex, se compara por similitud coseno
# TF-IDF contra estas frases. Si supera el umbral (0.55 por defecto), se
# asigna la categoría correspondiente con confianza media.
#
# Frases construidas a partir de las plantillas dominantes del corpus,
# en su forma ya normalizada (NFKD, minúsculas, sin typos).

FRASES_REFERENCIA_TFIDF: Dict[str, List[str]] = {
    "estudio_complementario_imagen": [
        "se sugiere ecografia mamaria para posterior recategorizacion",
        "se sugiere ecografia mamaria debido al patron mamografico",
        "se sugiere ecografia complementaria",
        "sugerimos complementar el estudio con ecografia mamaria",
        "se sugiere resonancia magnetica mamaria",
        "se sugiere incidencias con magnificacion",
        "se sugiere ecografia mamaria y de la region axilar",
    ],
    "correlacion_ecografica": [
        "se sugiere correlacion con ecografia mamaria",
        "se sugiere correlacion con ecografia mamaria debido al patron",
        "se sugiere correlacionar este estudio con ecografia",
    ],
    "comparacion_estudios_previos": [
        "se sugiere comparacion con estudios anteriores",
        "se sugiere correlacion con estudios previos",
        "no contamos con estudios previos por lo cual sugerimos correlacion",
        "para apreciar evolucion",
    ],
    "control_anual": [
        "se sugiere control mamografico anual",
        "se sugiere control anual",
        "se sugiere mamografia de control anual",
        "se sugiere control mamografico y ecografico anual",
    ],
    "control_corto_plazo": [
        "se sugiere control semestral",
        "se sugiere control en seis meses",
        "se sugiere control ecografico semestral",
        "se sugiere seguimiento a seis meses",
    ],
    "biopsia_histologia": [
        "se sugiere caracterizacion histologica",
        "se sugiere estudio histologico",
        "se sugiere biopsia",
        "se sugiere muestra histopatologica de la lesion",
    ],
    "criterio_medico": [
        "controles segun criterio del medico tratante",
        "se sugiere controles habituales",
        "se sugiere control mamografico bianual",
        "decidir conducta segun criterio del medico tratante",
    ],
    "derivacion_oncologica": [
        "se sugiere derivacion a oncologia",
        "se sugiere manejo oncologico",
        "se sugiere evaluacion oncologica",
    ],
}


# =============================================================================
# 6. ENCABEZADOS DEL BLOQUE DE RECOMENDACIONES EN FULL_REPORT
# =============================================================================
#
# Para el fallback cuando la columna Recommendations viene vacía y hay que
# extraer el bloque de recomendaciones desde el Full_Report.

ENCABEZADOS_RECOMENDACIONES: List[str] = [
    r"recomendaciones?\s*:",
    r"recomendacion\s*:",
    r"indicaciones?\s*:",
    r"sugerencias?\s*:",
    r"conducta\s+a?\s*seguir\s*:",
    r"plan\s*:",
]


# =============================================================================
# 7. UMBRAL DE SIMILITUD TF-IDF
# =============================================================================

UMBRAL_SIMILITUD_TFIDF: float = 0.55  # validado experimentalmente
