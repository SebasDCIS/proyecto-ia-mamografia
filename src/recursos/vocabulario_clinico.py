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
        r"complementar\s+(el\s+|un\s+)?(estudio|examen)?\s*con\s+(una\s+)?ecograf",
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
        # Pistas de INTENCIÓN "incompleto" (estudio aún no concluyente) — señales
        # textuales robustas, ciegas al BI-RADS. Refuerzan la categoría ante
        # informes externos que expresen la misma intención con otras palabras.
        r"(?<!correlacion\scon\s)ecograf\w*\s+.*?para\s+(poder\s+)?concluir",
        r"(?<!correlacion\scon\s)ecograf\w*\s+.*?para\s+caracterizar",
        r"(?<!correlacion\scon\s)ecograf\w*\s+.*?(hallazgo|nodulo|imagen)\s+indetermina",
        r"(?<!correlacion\scon\s)ecograf\w*\s+.*?para\s+definir",
        r"para\s+(poder\s+)?concluir\s+.*?ecograf",
        r"caracterizar\s+.*?con\s+ecograf",
    ],
    "correlacion_ecografica": [
        r"correlacion\s+(con\s+)?ecograf",
        r"correlacionar\s+(este\s+estudio\s+)?con\s+ecograf",
        # Pistas de INTENCIÓN "confirmado" (diagnóstico ya hecho, la ecografía
        # solo confirma) — señales textuales, ciegas al BI-RADS.
        r"confirmar\s+(con\s+)?ecograf",
        r"confirmar\s+.*?hallazgo\s+.*?con\s+ecograf",
        r"ecograf\w*\s+.*?para\s+confirmar",
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
        # "mamografia anual" / "ecografia anual" (estudio anual = control anual)
        r"(mamografia|ecografia|mamografico|ecografico)\s+anual",
    ],
    "control_corto_plazo": [
        # NOTA: se acepta la forma verbal ("controlar", "controlarse") además del
        # sustantivo. Un informe chileno real decía "se sugiere CONTROLAR en seis
        # meses" y ningún patrón lo reconocía, porque todos exigían "control" +
        # espacio. El sufijo opcional (?:ar|arse|arla|arlo) cubre esas formas.
        r"control(?:ar|arse|arla|arlo)?\s+semestral",
        r"control(?:ar|arse|arla|arlo)?\s+(?:en|a)\s+(?:los?\s+)?(?:6|seis)\s+meses",
        r"control(?:ar|arse|arla|arlo)?\s+(?:en|a)\s+(?:los?\s+)?(?:3|tres)\s+meses",
        r"control(?:ar|arse|arla|arlo)?\s+(?:en|a)\s+(?:los?\s+)?(?:4|cuatro)\s+meses",
        r"seguimiento\s+(?:a|en)\s+(?:6|seis)\s+meses",
        r"seguimiento\s+semestral",
        r"control(?:ar|arse)?\s+ecografico\s+semestral",
        r"control\s+ecografico\s+(?:en|a)\s+(?:6|seis)\s+meses",
        r"control\s+ecografico\s+y\s+mamografia.{0,30}(?:en|a)\s+(?:6|seis)\s+meses",
        # Formas donde el intervalo precede al verbo
        r"(?:6|seis)\s+meses.{0,15}control",
        r"a\s+corto\s+plazo",
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


# =============================================================================
# 8. SINÓNIMOS CLÍNICOS (normalización semántica ligera)
# =============================================================================
#
# Mapea términos equivalentes a una FORMA CANÓNICA antes de aplicar las reglas.
# Objetivo: dar flexibilidad léxica sin recurrir a embeddings (probados y
# descartados por colapsar distinciones clínicas). Solo se normalizan sinónimos
# de significado clínico EQUIVALENTE y no ambiguos.
#
# El patrón es una regex (con \b para límites de palabra) y el valor es el
# término canónico que las reglas ya reconocen. Se aplica sobre texto ya en
# minúsculas y sin tildes.

SINONIMOS_CLINICOS: Dict[str, str] = {
    # --- Modalidad ecográfica: ultrasonido/US/sonografía -> ecografia ---
    r"\bultrasonido\b":            "ecografia",
    r"\bultrasonografia\b":        "ecografia",
    r"\bsonografia\b":             "ecografia",
    r"\becotomografia\b":          "ecografia",
    r"\becotomografico\b":         "ecografico",
    r"\bus\s+mamari":              "ecografia mamari",   # "US mamario/a"
    r"\bus\s+de\s+mama":           "ecografia de mama",
    r"\beco\s+mamari":             "ecografia mamari",   # "eco mamaria"
    r"\beco\s+de\s+mama":          "ecografia de mama",
    r"\becodoppler\b":             "ecografia doppler",

    # --- Resonancia magnética: RM/RMN/MRI -> resonancia magnetica ---
    r"\brmn\b":                    "resonancia magnetica",
    r"\bmri\b":                    "resonancia magnetica",
    r"\brm\s+mamari":              "resonancia magnetica mamari",
    r"\brm\s+de\s+mama":           "resonancia magnetica de mama",

    # --- Biopsia / confirmación tisular: siglas y variantes -> biopsia ---
    r"\bbacaf\b":                  "biopsia",   # biopsia por aspiración con aguja fina
    r"\bpaaf\b":                   "biopsia",   # punción aspiración con aguja fina
    r"\bcore\s*biops":             "biopsia",   # core biopsy
    r"\btru[\s\-]?cut\b":          "biopsia",   # tru-cut
    r"\bpuncion\s+con\s+aguja":    "biopsia",
    r"\bpuncion\s+aspiraci":       "biopsia",
    r"\bbiopsia\s+percutanea\b":   "biopsia",
    r"\bmicrobiopsia\b":           "biopsia",
    r"\bestudio\s+citolog":        "biopsia",   # estudio citológico (confirmación)
    r"\bestudio\s+anatomo":        "biopsia",   # estudio anatomopatológico

    # --- Proyecciones/técnicas mamográficas adicionales -> estudio compl. ---
    r"\bincidencias?\s+adicional": "magnificacion",
    r"\bproyecciones?\s+adicional":"magnificacion",
    r"\bcompresion\s+localizada\b":"compresion focalizada",
    r"\bcompresion\s+puntual\b":   "compresion focalizada",
    r"\btomosintesis\b":           "magnificacion",   # técnica complementaria

    # --- Verbos/perífrasis de recomendación -> "se sugiere" ---
    r"\bse aconseja\b":            "se sugiere",
    r"\bse indica\b":              "se sugiere",
    r"\bse solicita\b":            "se sugiere",
    r"\bse propone\b":             "se sugiere",
    r"\bconviene\b":               "se sugiere",
    r"\bameritaria\b":             "amerita",

    # --- Seguimiento / vigilancia -> "control" ---
    r"\bseguimiento\b":            "control",
    r"\bvigilancia\b":             "control",
    r"\bmonitoreo\b":              "control",
    r"\bmonitorizacion\b":         "control",

    # --- Periodicidad anual -> "anual" ---
    r"\ben un ano\b":              "anual",
    r"\bcada ano\b":               "anual",
    r"\banualmente\b":             "anual",
    r"\bcada 12 meses\b":          "anual",
    r"\bcada doce meses\b":        "anual",

    # --- Corto plazo / semestral -> "semestral" (control_corto_plazo) ---
    r"\ben seis meses\b":          "semestral",
    r"\ben 6 meses\b":             "semestral",
    r"\bcada seis meses\b":        "semestral",
    r"\bcada 6 meses\b":           "semestral",
    r"\bsemestralmente\b":         "semestral",

    # --- Biopsia / punción -> "biopsia" ---
    r"\bpuncionar\b":              "biopsia",
    r"\bpuncion\b":                "biopsia",
    r"\bnucleobiopsia\b":          "biopsia",
    r"\bcore\s+biopsy\b":          "biopsia",
    r"\bmuestra\s+de\s+tejido\b":  "biopsia",

    # --- Derivación / especialista -> "derivacion a oncologia" ---
    r"\bderivar\s+a\s+oncolog":    "derivacion a oncolog",
    r"\breferir\s+a\s+oncolog":    "derivacion a oncolog",
    r"\breferencia\s+a\s+oncolog": "derivacion a oncolog",

    # --- Comparación con previos ---
    r"\bcomparar\s+con\s+(estudios\s+)?(previos|anteriores)": "comparacion con estudios previos",
    r"\bcotejar\s+con\s+(estudios\s+)?(previos|anteriores)":  "comparacion con estudios previos",
}
