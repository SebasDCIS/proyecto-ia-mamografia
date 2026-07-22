"""
Extractor de recomendaciones clínicas en informes mamográficos en español.

Módulo del MVP del proyecto BME513 (Universidad de Valparaíso).

Implementa tres funciones públicas con trazabilidad completa para auditoría:

  1. extraer_texto_recomendacion(...): obtiene el texto de la recomendación
     desde la columna Recommendations o, como fallback, del Full_Report.

  2. clasificar_recomendacion(texto): clasifica la recomendación en una o
     más categorías clínicas mediante regex (capa 1) y similitud TF-IDF
     (capa 2 de fallback). Siempre devuelve trazabilidad detallada.

  3. generar_reporte_auditoria(resultado): convierte el diccionario de
     trazabilidad en un reporte de texto formateado para presentación
     al usuario (radiólogo, auditor clínico).

Función helper:

  - guardar_auditoria(resultado, informe_id): persiste el resultado en
    JSON para revisión posterior.

Arquitectura en capas (ver vocabulario_clinico.py para los recursos):

    [texto crudo]
         ↓
    Capa 1: Normalizador (NFKD + minúsculas + diccionario de typos)
         ↓
    Capa 2: Regex sobre patrones por categoría (100% cobertura en corpus)
         ↓
    ¿categoría detectada?
      ├─ SÍ → devolver, confianza=alta
      └─ NO → ir a Capa 3
         ↓
    Capa 3: Similitud TF-IDF contra frases de referencia (umbral 0.55)
         ↓
    ¿similitud ≥ umbral?
      ├─ SÍ → devolver, confianza=media + nota de método
      └─ NO → categoría='ambigua' + alerta para revisión humana

Validado sobre Vázquez Noguera et al. (2025):
    - 100% cobertura regex sobre 4 347 recomendaciones del corpus
    - 13 tests inline ejecutables con: python src/extractor_recomendacion.py

Autor: Sebastián Inostroza Hurtado
Fecha: Mayo 2026
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recursos.vocabulario_clinico import (
    CATEGORIAS_CLINICAS,
    SINONIMOS_CLINICOS,
    ENCABEZADOS_RECOMENDACIONES,
    FRASES_REFERENCIA_TFIDF,
    JERARQUIA_CLINICA,
    PATRONES_POR_CATEGORIA,
    TYPOS_CLINICOS,
    UMBRAL_SIMILITUD_TFIDF,
)


# =============================================================================
# CONFIGURACIÓN E INICIALIZACIÓN INTERNA
# =============================================================================

# Construir el vectorizador TF-IDF UNA SOLA VEZ al cargar el módulo.
# Esto evita re-entrenar el TF-IDF en cada llamada.
def _construir_vectorizador_tfidf() -> Tuple[TfidfVectorizer, Dict[str, Any]]:
    """Construye y entrena el TF-IDF sobre las frases de referencia.

    Returns:
        Tupla (vectorizador entrenado, dict con embeddings y metadatos).
    """
    todas_frases: List[str] = []
    categoria_por_frase: List[str] = []

    for categoria, frases in FRASES_REFERENCIA_TFIDF.items():
        for frase in frases:
            todas_frases.append(frase)
            categoria_por_frase.append(categoria)

    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matriz_referencias = vectorizador.fit_transform(todas_frases)

    return vectorizador, {
        "frases": todas_frases,
        "categorias": categoria_por_frase,
        "matriz": matriz_referencias,
    }


# Inicialización al importar el módulo (una sola vez)
_VECTORIZADOR_TFIDF, _DATOS_REFERENCIA = _construir_vectorizador_tfidf()


# Patrón compilado de encabezados de recomendaciones (para fallback)
_PATRON_INICIO_RECOMENDACIONES = re.compile(
    r"\b(" + "|".join(ENCABEZADOS_RECOMENDACIONES) + r")",
    re.IGNORECASE,
)

# Patrones siguientes que marcan el fin del bloque de recomendaciones
_PATRON_FIN_BLOQUE = re.compile(
    r"\b(observaciones?|firma|atte|atentamente|cordialmente|dr\.|dra\.)\s*:?",
    re.IGNORECASE,
)


# =============================================================================
# FUNCIONES INTERNAS (NORMALIZACIÓN)
# =============================================================================

def _normalizar_texto(texto: str) -> Tuple[str, List[Dict[str, str]]]:
    """Normaliza texto y devuelve el texto procesado + log de cambios.

    Aplica en orden:
    1. NFKD (descomposición Unicode) para quitar tildes
    2. Conversión a minúsculas
    3. Corrección de typos clínicos conocidos

    Args:
        texto: string a normalizar.

    Returns:
        Tupla (texto_normalizado, lista de typos corregidos para auditoría).
    """
    if not isinstance(texto, str):
        return "", []

    # Nivel 1: NFKD + minúsculas
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()

    # Nivel 2: corregir typos (registrando cada corrección)
    typos_corregidos: List[Dict[str, str]] = []
    for typo_pattern, correcto in TYPOS_CLINICOS.items():
        match = re.search(typo_pattern, texto)
        if match:
            original = match.group(0)
            typos_corregidos.append({
                "original": original,
                "corregido": correcto,
                "patron": typo_pattern,
            })
            texto = re.sub(typo_pattern, correcto, texto)

    # Nivel 3: normalizar sinónimos clínicos a su forma canónica.
    # Da flexibilidad léxica (seguimiento->control, "en un año"->anual, etc.)
    # sin recurrir a embeddings. Se registra cada cambio para auditoría.
    for sin_pattern, canonico in SINONIMOS_CLINICOS.items():
        match = re.search(sin_pattern, texto)
        if match:
            typos_corregidos.append({
                "original": match.group(0),
                "corregido": canonico,
                "patron": sin_pattern,
                "tipo": "sinonimo",
            })
            texto = re.sub(sin_pattern, canonico, texto)

    return texto, typos_corregidos


# =============================================================================
# FUNCIONES INTERNAS (LOCALIZACIÓN EN FULL_REPORT)
# =============================================================================

def _extraer_bloque_recomendaciones_de_full_report(
    full_report: str,
) -> Optional[Dict[str, Any]]:
    """Busca y extrae el bloque RECOMENDACIONES del Full_Report.

    Args:
        full_report: texto completo del informe.

    Returns:
        Dict con texto del bloque, posiciones e información para auditoría;
        o None si no se encuentra el bloque.
    """
    if not isinstance(full_report, str) or not full_report.strip():
        return None

    match_inicio = _PATRON_INICIO_RECOMENDACIONES.search(full_report)
    if not match_inicio:
        return None

    inicio_bloque = match_inicio.end()
    encabezado = match_inicio.group(0).strip().rstrip(":").strip()

    # Buscar fin del bloque
    resto = full_report[inicio_bloque:]
    match_fin = _PATRON_FIN_BLOQUE.search(resto)

    if match_fin:
        fin_bloque = inicio_bloque + match_fin.start()
    else:
        fin_bloque = len(full_report)

    texto_bloque = full_report[inicio_bloque:fin_bloque].strip()

    return {
        "texto": texto_bloque,
        "inicio": inicio_bloque,
        "fin": fin_bloque,
        "encabezado": encabezado,
    }


# Frases que introducen una recomendación clínica en prosa (sin encabezado).
# Se usan como fallback (Vía 2.5) cuando no existe un bloque RECOMENDACIONES.
# NOTA: se matchean sobre texto SIN acentos y en minúsculas (ver _quita_acentos),
# por eso basta con las formas sin tilde ("correlacion", "seria", "debera").
_FRASES_GATILLO_RECOMENDACION = re.compile(
    r"\b("
    # Impersonales "se + verbo"
    r"se\s+sugiere[n]?|se\s+recomienda[n]?|se\s+aconseja[n]?|se\s+indica[n]?|"
    r"se\s+solicita[n]?|se\s+deriva[n]?|se\s+requiere[n]?|se\s+recomiendan|"
    # Primera persona plural
    r"sugerimos|recomendamos|solicitamos|indicamos|aconsejamos|derivamos|requerimos|"
    # Primera persona singular
    r"sugiero|recomiendo|solicito|indico|aconsejo|derivo|"
    # Verbos sueltos de necesidad / conducta
    r"amerita[n]?|ameritar|requiere[n]?|necesita[n]?|"
    # Perífrasis impersonales de recomendación
    r"es\s+necesario|es\s+recomendable|es\s+conveniente|es\s+aconsejable|es\s+ideal|"
    r"seria\s+ideal|seria\s+conveniente|seria\s+recomendable|seria\s+aconsejable|"
    r"convendria|convendra|conviene|amerita\s+realizar|"
    r"idealmente|"
    # Correlación / comparación como directiva
    r"correlacion\s+con|correlacionar\s+con|comparar\s+con|"
    # Infinitivos de conducta sueltos (informes que redactan en infinitivo)
    r"derivar|realizar|efectuar|completar|repetir|solicitar|tomar\s+muestra|"
    # Deber + infinitivo (incluye forma reflexiva -se) y formas sueltas
    r"deber[ai]\s+\w+|debe\s+(?:tomar|realizar|realizarse|efectuar|efectuarse|"
    r"acudir|repetir|derivarse|completar|complementar|puncionar|biopsiar|"
    r"derivar|referir|controlar|correlacionar|comparar|caracterizar|confirmar)|"
    r"debe\s+realizarse|debera|deberia"
    r")\b",
    re.IGNORECASE,
)

# Directivas "peladas": conducta escrita como sustantivo de acción sin verbo
# gatillo (p. ej. "Control en 3 meses", "Biopsia", "Ecografía mamaria"). Se
# detectan cuando el segmento COMIENZA con uno de estos sustantivos.
_SUSTANTIVO_DIRECTIVA = re.compile(
    r"^\s*(?:[-•*]\s*)?("
    # sustantivos de acción
    r"control|biopsia|ecografia|ecotomografia|resonancia|derivacion|"
    r"correlacion|seguimiento|puncion|incidencias|magnificacion|marcacion|"
    r"comparacion|"
    # verbos en infinitivo/imperativo que inician una conducta
    r"complementar|realizar|referir|derivar|controlar|correlacionar|"
    r"comparar|cotejar|puncionar|biopsiar|caracterizar|confirmar|"
    r"mantener|continuar|proseguir|proponer|proponemos|reevaluar|reevaluacion"
    r")\b",
    re.IGNORECASE,
)


def _quita_acentos(texto: str) -> str:
    """Devuelve el texto en minúsculas y sin tildes (para matcheo robusto)."""
    t = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


# -----------------------------------------------------------------------------
# Tolerancia difusa (fuzzy) a errores ortográficos en los verbos gatillo.
# Cuando el match exacto falla, se acepta una palabra que esté a distancia de
# edición <= 1 de un verbo gatillo distintivo (p. ej. "suguiere" -> "sugiere").
# CRITERIO CONSERVADOR (alta precisión):
#   - Solo verbos LARGOS y distintivos (>= 6 letras); los cortos o ambiguos se
#     dejan solo con match exacto para no sobre-corregir.
#   - Distancia máxima 1 (un solo error de tipeo).
#   - NUNCA se aplica a números ni a la categoría BI-RADS: esos se leen literales.
# -----------------------------------------------------------------------------
_TRIGGERS_FUZZY = [
    "sugiere", "sugieren", "sugerimos", "recomienda", "recomendamos",
    "recomiendo", "aconseja", "aconsejo", "ameritar", "amerita", "solicita",
    "solicitamos", "derivar", "correlacion", "correlacionar",
]
_FUZZY_MIN_LEN = 6      # no aplicar fuzzy a palabras cortas
_FUZZY_MAX_DIST = 1     # umbral conservador (un solo error)


def _distancia_edicion(a: str, b: str) -> int:
    """Distancia de Damerau-Levenshtein (OSA) entre dos strings.

    Cuenta como 1 operación: inserción, borrado, sustitución Y transposición de
    letras adyacentes (p. ej. "recomendia" -> "recomienda"), que son de los
    typos más frecuentes.
    """
    if a == b:
        return 0
    m, n = len(a), len(b)
    if abs(m - n) > _FUZZY_MAX_DIST:
        return max(m, n)  # atajo: no pueden estar a distancia <= umbral
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            costo = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # borrado
                d[i][j - 1] + 1,      # inserción
                d[i - 1][j - 1] + costo,  # sustitución
            )
            # Transposición de letras adyacentes
            if (i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[m][n]


def _detectar_gatillo_difuso(segmento: str) -> Optional[Dict[str, str]]:
    """Busca un verbo gatillo con un typo (distancia de edición 1).

    Returns:
        Dict {palabra, gatillo, distancia} si encuentra un match difuso, o None.
    """
    for palabra in re.findall(r"[a-zñ]+", segmento.lower()):
        if len(palabra) < _FUZZY_MIN_LEN:
            continue
        for kw in _TRIGGERS_FUZZY:
            if abs(len(palabra) - len(kw)) > _FUZZY_MAX_DIST:
                continue
            d = _distancia_edicion(palabra, kw)
            if 1 <= d <= _FUZZY_MAX_DIST:  # d>=1: solo typos, no matches exactos
                return {"palabra": palabra, "gatillo": kw, "distancia": str(d)}
    return None


def _extraer_recomendacion_por_frases_gatillo(
    full_report: str,
) -> Optional[Dict[str, Any]]:
    """Vía 2.5: extrae la recomendación desde prosa sin encabezado explícito.

    Cuando el informe no tiene un bloque 'RECOMENDACIONES:' pero sí redacta la
    conducta en prosa (p. ej. 'Se sugiere estudio histológico... biopsia'),
    este fallback localiza las oraciones que contienen una frase gatillo de
    recomendación y las devuelve como texto de recomendación.

    Args:
        full_report: texto completo del informe.

    Returns:
        Dict con texto e info de trazabilidad, o None si no hay frases gatillo.
    """
    if not isinstance(full_report, str) or not full_report.strip():
        return None

    # Segmentar en oraciones aproximadas: por punto, salto de línea o guion
    # separador (" - ", " – ", " — "), común en informes tipo "Birads I - Control...".
    segmentos = re.split(r"(?<=[.\n])\s+|\n|\s+[-–—]\s+", full_report)

    oraciones_recomendacion: List[str] = []
    correcciones_difusas: List[Dict[str, str]] = []
    for seg in segmentos:
        seg_limpio = seg.strip()
        if not seg_limpio:
            continue
        # Versión sin tildes y en minúsculas para matcheo robusto
        seg_norm = _quita_acentos(seg_limpio)
        # 1) Match exacto de frase gatillo (verbo/perífrasis de recomendación)
        if _FRASES_GATILLO_RECOMENDACION.search(seg_norm):
            oraciones_recomendacion.append(seg_limpio)
            continue
        # 2) Directiva "pelada": el segmento comienza con un sustantivo de acción
        #    ("Control en 3 meses", "Biopsia", "Ecografía mamaria")
        if _SUSTANTIVO_DIRECTIVA.search(seg_norm):
            oraciones_recomendacion.append(seg_limpio)
            continue
        # 3) Fallback difuso: ¿algún verbo gatillo con un typo (distancia 1)?
        difuso = _detectar_gatillo_difuso(seg_norm)
        if difuso:
            oraciones_recomendacion.append(seg_limpio)
            correcciones_difusas.append(difuso)

    if not oraciones_recomendacion:
        return None

    texto_bloque = " ".join(oraciones_recomendacion).strip()

    return {
        "texto": texto_bloque,
        "inicio": None,
        "fin": None,
        "encabezado": None,
        "n_oraciones": len(oraciones_recomendacion),
        "correcciones_difusas": correcciones_difusas,
    }


# =============================================================================
# FUNCIONES INTERNAS (CLASIFICACIÓN)
# =============================================================================

def _detectar_por_regex(
    texto_normalizado: str,
) -> List[Dict[str, Any]]:
    """Detecta categorías por regex y devuelve trazabilidad de cada match.

    Args:
        texto_normalizado: texto ya pasado por _normalizar_texto.

    Returns:
        Lista de matches, cada uno con categoría, patrón, fragmento y posición.
    """
    matches: List[Dict[str, Any]] = []

    for categoria, patrones in PATRONES_POR_CATEGORIA.items():
        for patron in patrones:
            for m in re.finditer(patron, texto_normalizado):
                matches.append({
                    "categoria": categoria,
                    "patron": patron,
                    "fragmento_matcheado": m.group(0),
                    "posicion": [m.start(), m.end()],
                })

    return matches


def _detectar_por_tfidf(
    texto_normalizado: str,
    umbral: float = UMBRAL_SIMILITUD_TFIDF,
) -> Optional[Dict[str, Any]]:
    """Fallback semántico: similitud TF-IDF contra frases de referencia.

    Args:
        texto_normalizado: texto ya normalizado.
        umbral: umbral mínimo de similitud coseno (default 0.55).

    Returns:
        Dict con la categoría más similar y métricas, o None si nada
        supera el umbral.
    """
    if not texto_normalizado.strip():
        return None

    # Vectorizar el texto
    vector_texto = _VECTORIZADOR_TFIDF.transform([texto_normalizado])

    # Calcular similitud contra todas las frases de referencia
    similitudes = cosine_similarity(
        vector_texto,
        _DATOS_REFERENCIA["matriz"],
    ).flatten()

    # Identificar la frase más similar
    idx_mejor = int(similitudes.argmax())
    score_mejor = float(similitudes[idx_mejor])

    if score_mejor < umbral:
        return None

    return {
        "categoria": _DATOS_REFERENCIA["categorias"][idx_mejor],
        "frase_referencia": _DATOS_REFERENCIA["frases"][idx_mejor],
        "similitud_score": round(score_mejor, 4),
        "umbral_usado": umbral,
    }


def _aplicar_jerarquia(
    categorias_detectadas: List[str],
) -> Optional[str]:
    """Selecciona la categoría principal según la jerarquía clínica.

    Args:
        categorias_detectadas: lista (puede tener duplicados).

    Returns:
        Categoría principal, o None si la lista está vacía.
    """
    if not categorias_detectadas:
        return None

    unicas = set(categorias_detectadas)
    for cat in JERARQUIA_CLINICA:
        if cat in unicas:
            return cat

    return list(unicas)[0]


# =============================================================================
# FUNCIÓN PÚBLICA 1: extraer_texto_recomendacion
# =============================================================================

def extraer_texto_recomendacion(
    recommendations_col: Optional[str],
    full_report: Optional[str] = None,
    usar_ner: bool = False,
) -> Dict[str, Any]:
    """Obtiene el texto de la recomendación con trazabilidad de origen.

    Estrategia:
        1. Si la columna Recommendations no es 'sin_recomendacion' ni vacía,
           usar su contenido (vía rápida, 99.8% del corpus).
        2. Si la columna está vacía y se proporciona full_report, buscar
           el bloque RECOMENDACIONES con regex.
        3. Si nada funciona, devolver no_encontrado.

    Args:
        recommendations_col: contenido de la columna Recommendations del CSV.
        full_report: texto completo del informe (opcional, para fallback).

    Returns:
        Dict con:
            - texto (str): texto crudo de la recomendación
            - texto_normalizado (str): texto tras normalización
            - fuente (str): 'columna_recommendations' | 'regex_full_report' |
                            'no_encontrado'
            - encontrado (bool)
            - encabezado_detectado (str | None): solo si se usó regex
            - typos_corregidos (list): typos aplicados al normalizar
    """
    resultado: Dict[str, Any] = {
        "texto": "",
        "texto_normalizado": "",
        "fuente": "no_encontrado",
        "encontrado": False,
        "encabezado_detectado": None,
        "typos_corregidos": [],
    }

    # Vía 1: columna Recommendations
    if (
        isinstance(recommendations_col, str)
        and recommendations_col.strip()
        and recommendations_col != "sin_recomendacion"
    ):
        resultado["texto"] = recommendations_col
        resultado["fuente"] = "columna_recommendations"
        resultado["encontrado"] = True
        normalizado, typos = _normalizar_texto(recommendations_col)
        resultado["texto_normalizado"] = normalizado
        resultado["typos_corregidos"] = typos
        return resultado

    # Vía 2: fallback en Full_Report
    if isinstance(full_report, str) and full_report.strip():
        bloque = _extraer_bloque_recomendaciones_de_full_report(full_report)
        if bloque:
            resultado["texto"] = bloque["texto"]
            resultado["fuente"] = "regex_full_report"
            resultado["encontrado"] = True
            resultado["encabezado_detectado"] = bloque["encabezado"]
            normalizado, typos = _normalizar_texto(bloque["texto"])
            resultado["texto_normalizado"] = normalizado
            resultado["typos_corregidos"] = typos
            return resultado

        # Vía 2.5: sin encabezado, buscar recomendación en prosa (frases gatillo)
        bloque_prosa = _extraer_recomendacion_por_frases_gatillo(full_report)
        if bloque_prosa:
            resultado["texto"] = bloque_prosa["texto"]
            resultado["fuente"] = "regex_frases_gatillo"
            resultado["encontrado"] = True
            resultado["encabezado_detectado"] = None
            normalizado, typos = _normalizar_texto(bloque_prosa["texto"])
            resultado["texto_normalizado"] = normalizado
            resultado["typos_corregidos"] = typos
            # Trazabilidad: correcciones ortográficas difusas (fuzzy) aplicadas
            # para detectar la frase gatillo (p. ej. "suguiere" -> "sugiere").
            if bloque_prosa.get("correcciones_difusas"):
                resultado["correcciones_difusas"] = bloque_prosa["correcciones_difusas"]
            return resultado

    # Vía 3: no encontrado por reglas -> intentar con el modelo NER (si se pidió)
    if usar_ner and isinstance(full_report, str) and full_report.strip():
        try:
            from src.extractor_ner import extraer_recomendacion_ner
            ner = extraer_recomendacion_ner(full_report)
        except Exception:
            ner = {"encontrado": False}
        if ner.get("encontrado"):
            span = ner["texto"]
            resultado["texto"] = span
            resultado["fuente"] = "ner_distilbeto"
            resultado["encontrado"] = True
            resultado["encabezado_detectado"] = None
            normalizado, typos = _normalizar_texto(span)
            resultado["texto_normalizado"] = normalizado
            resultado["typos_corregidos"] = typos
            return resultado

    # Vía 4: no encontrado por ningún método
    return resultado


# =============================================================================
# FUNCIÓN PÚBLICA 2: clasificar_recomendacion
# =============================================================================

def clasificar_recomendacion(
    texto: str,
    es_ya_normalizado: bool = False,
) -> Dict[str, Any]:
    """Clasifica la recomendación en categorías clínicas.

    Aplica regex (capa 1) primero. Si no encuentra categorías, recurre
    a TF-IDF (capa 2). Siempre devuelve trazabilidad detallada (nivel 2).

    Args:
        texto: recomendación a clasificar.
        es_ya_normalizado: si True, asume que el texto ya está normalizado
            (saltea la capa de normalización). Útil cuando se llama después
            de extraer_texto_recomendacion.

    Returns:
        Dict con resultado de clasificación + trazabilidad completa.
    """
    resultado: Dict[str, Any] = {
        "categorias_detectadas": [],
        "categoria_principal": None,
        "confianza": "no_clasificada",
        "metodo": None,
        "trazabilidad": {
            "texto_original": texto if isinstance(texto, str) else "",
            "texto_normalizado": "",
            "typos_corregidos": [],
            "patrones_que_matchearon": [],
            "similitud_tfidf": None,
            "jerarquia_aplicada": None,
            "razon_seleccion_principal": None,
        },
    }

    if not isinstance(texto, str) or not texto.strip():
        resultado["trazabilidad"]["razon_seleccion_principal"] = (
            "Texto vacío o no es string"
        )
        return resultado

    # Normalizar (a menos que ya venga normalizado)
    if es_ya_normalizado:
        texto_norm = texto
        typos: List[Dict[str, str]] = []
    else:
        texto_norm, typos = _normalizar_texto(texto)

    resultado["trazabilidad"]["texto_normalizado"] = texto_norm
    resultado["trazabilidad"]["typos_corregidos"] = typos

    # CAPA 1: regex sobre patrones por categoría
    matches_regex = _detectar_por_regex(texto_norm)

    if matches_regex:
        resultado["trazabilidad"]["patrones_que_matchearon"] = matches_regex
        cats_unicas = list(dict.fromkeys(m["categoria"] for m in matches_regex))
        resultado["categorias_detectadas"] = cats_unicas
        resultado["categoria_principal"] = _aplicar_jerarquia(cats_unicas)
        resultado["confianza"] = "alta"
        resultado["metodo"] = "regex"
        resultado["trazabilidad"]["jerarquia_aplicada"] = JERARQUIA_CLINICA

        if len(cats_unicas) > 1:
            resultado["trazabilidad"]["razon_seleccion_principal"] = (
                f"Múltiples categorías detectadas ({cats_unicas}); se "
                f"aplicó jerarquía clínica (peor caso) → "
                f"{resultado['categoria_principal']}"
            )
        else:
            resultado["trazabilidad"]["razon_seleccion_principal"] = (
                f"Categoría única detectada por regex: "
                f"{resultado['categoria_principal']}"
            )

        return resultado

    # CAPA 2: fallback TF-IDF
    similitud = _detectar_por_tfidf(texto_norm)

    if similitud:
        resultado["categorias_detectadas"] = [similitud["categoria"]]
        resultado["categoria_principal"] = similitud["categoria"]
        resultado["confianza"] = "media"
        resultado["metodo"] = "tf_idf_similitud"
        resultado["trazabilidad"]["similitud_tfidf"] = similitud
        resultado["trazabilidad"]["razon_seleccion_principal"] = (
            f"Regex no encontró coincidencias. Asignado por similitud "
            f"TF-IDF ({similitud['similitud_score']:.4f}) contra frase "
            f"de referencia: '{similitud['frase_referencia']}'"
        )
        return resultado

    # Nada funcionó
    resultado["categorias_detectadas"] = ["ambigua"]
    resultado["categoria_principal"] = "ambigua"
    resultado["confianza"] = "no_clasificada"
    resultado["metodo"] = None
    resultado["trazabilidad"]["razon_seleccion_principal"] = (
        "Ningún patrón regex ni similitud TF-IDF (umbral "
        f"{UMBRAL_SIMILITUD_TFIDF}) alcanzó coincidencia. Marcado como "
        "'ambigua' para revisión humana."
    )

    return resultado


# =============================================================================
# FUNCIÓN PÚBLICA 3: generar_reporte_auditoria
# =============================================================================

def generar_reporte_auditoria(
    resultado: Dict[str, Any],
    informe_id: Optional[str] = None,
) -> str:
    """Convierte el resultado de clasificación en reporte legible.

    Args:
        resultado: dict devuelto por clasificar_recomendacion().
        informe_id: identificador opcional del informe.

    Returns:
        String con el reporte formateado para mostrar al usuario.
    """
    trz = resultado.get("trazabilidad", {})
    linea = "═" * 70

    reporte = [linea]
    if informe_id:
        reporte.append(f"REPORTE DE AUDITORÍA — Informe ID: {informe_id}")
    else:
        reporte.append("REPORTE DE AUDITORÍA")
    reporte.append(linea)

    # Sección: texto leído
    reporte.append("\nTEXTO LEÍDO POR EL SISTEMA:")
    texto_original = trz.get("texto_original", "")
    reporte.append(f"  Original:    {texto_original[:300]}")
    reporte.append(f"  Normalizado: {trz.get('texto_normalizado', '')[:300]}")

    typos = trz.get("typos_corregidos", [])
    if typos:
        reporte.append(f"\n  Typos corregidos durante normalización:")
        for t in typos:
            reporte.append(f"    • '{t['original']}' → '{t['corregido']}'")

    # Sección: clasificación
    reporte.append(f"\nCLASIFICACIÓN OBTENIDA:")
    cats = resultado.get("categorias_detectadas", [])
    if cats:
        reporte.append(f"  Categorías detectadas: {cats}")
    reporte.append(
        f"  Categoría principal:   {resultado.get('categoria_principal')}"
    )
    reporte.append(f"  Confianza:             {resultado.get('confianza')}")
    reporte.append(f"  Método utilizado:      {resultado.get('metodo')}")

    # Sección: por qué se eligió esa categoría
    razon = trz.get("razon_seleccion_principal")
    if razon:
        reporte.append(f"\nRAZÓN DE LA DECISIÓN:")
        reporte.append(f"  {razon}")

    # Sección: trazabilidad de patrones regex
    patrones = trz.get("patrones_que_matchearon", [])
    if patrones:
        reporte.append(f"\nPATRONES QUE COINCIDIERON ({len(patrones)}):")
        for p in patrones:
            reporte.append(
                f"  • [{p['categoria']}] '{p['fragmento_matcheado']}' "
                f"(pos {p['posicion'][0]}-{p['posicion'][1]})"
            )

    # Sección: trazabilidad de TF-IDF (si aplica)
    similitud = trz.get("similitud_tfidf")
    if similitud:
        reporte.append(f"\nSIMILITUD TF-IDF (FALLBACK):")
        reporte.append(f"  Frase de referencia: '{similitud['frase_referencia']}'")
        reporte.append(
            f"  Score de similitud:  {similitud['similitud_score']:.4f} "
            f"(umbral: {similitud['umbral_usado']})"
        )

    # Sección: descripción clínica de la categoría
    categoria_principal = resultado.get("categoria_principal")
    if categoria_principal and categoria_principal in CATEGORIAS_CLINICAS:
        reporte.append(f"\nSIGNIFICADO CLÍNICO DE LA CATEGORÍA:")
        reporte.append(f"  {CATEGORIAS_CLINICAS[categoria_principal]}")

    reporte.append("\n" + linea)
    return "\n".join(reporte)


# =============================================================================
# FUNCIÓN HELPER: guardar_auditoria
# =============================================================================

def guardar_auditoria(
    resultado: Dict[str, Any],
    informe_id: str,
    output_dir: str = "audit_logs",
) -> str:
    """Persiste el resultado de clasificación como JSON con timestamp.

    Args:
        resultado: dict devuelto por clasificar_recomendacion().
        informe_id: identificador del informe.
        output_dir: directorio donde guardar el JSON.

    Returns:
        Ruta del archivo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{informe_id}_{timestamp}.json"
    ruta = os.path.join(output_dir, nombre_archivo)

    payload = {
        "informe_id": informe_id,
        "timestamp": datetime.now().isoformat(),
        "resultado": resultado,
    }

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    return ruta


# =============================================================================
# TESTS INLINE
# =============================================================================

def _ejecutar_tests() -> None:
    """Suite de tests inline. Ejecutar con: python src/extractor_recomendacion.py"""

    casos = [
        {
            "nombre": "T1: Control anual estándar",
            "texto": "- Se sugiere control mamográfico anual.",
            "esperado_principal": "control_anual",
            "esperado_metodo": "regex",
        },
        {
            "nombre": "T2: Combinada (correlación + control anual)",
            "texto": "- Se sugiere correlación con ecografía mamaria y control anual.",
            "esperado_principal": "correlacion_ecografica",
            "esperado_metodo": "regex",
            "esperado_min_cats": 2,
        },
        {
            "nombre": "T3: Biopsia (debe ganar sobre todo lo demás por jerarquía)",
            "texto": "- Se sugiere ecografía mamaria y caracterización histológica.",
            "esperado_principal": "biopsia_histologia",
            "esperado_metodo": "regex",
            "esperado_min_cats": 2,
        },
        {
            "nombre": "T4: BI-RADS 4 con criterio médico (caso clínico ambiguo)",
            "texto": "- SE SUGIERE CORRELACIÓN CON ECOGRAFÍA MAMARIA ACTUALIZADA Y CONTROLES SEGÚN CRITERIO DEL MÉDICO TRATANTE.",
            "esperado_principal": "correlacion_ecografica",  # gana por jerarquía sobre criterio_medico
            "esperado_metodo": "regex",
        },
        {
            "nombre": "T5: Control corto plazo",
            "texto": "- Se sugiere control semestral.",
            "esperado_principal": "control_corto_plazo",
            "esperado_metodo": "regex",
        },
        {
            "nombre": "T6: Estudio complementario imagen",
            "texto": "- Se sugiere ecografía mamaria para posterior recategorización.",
            "esperado_principal": "estudio_complementario_imagen",
            "esperado_metodo": "regex",
        },
        {
            "nombre": "T7: Typo corregido (CONTRO ANUAL)",
            "texto": "- SE SUGIERE CONTRO ANUAL.",
            "esperado_principal": "control_anual",
            "esperado_metodo": "regex",
            "espera_typos": True,
        },
        {
            "nombre": "T8: Control bianual → criterio_medico",
            "texto": "- Se sugiere control mamográfico bianual.",
            "esperado_principal": "criterio_medico",
            "esperado_metodo": "regex",
        },
        {
            "nombre": "T9: Sinonimo puncion->biopsia captado por regla",
            "texto": "- favor coordinar punción con aguja gruesa de la lesión",
            "esperado_principal": "biopsia_histologia",
            "esperado_metodo": "regex",
        },
        {
            "nombre": "T10: Frase atípica que cae como ambigua",
            "texto": "- favor coordinar con secretaría del hospital",
            "esperado_principal": "ambigua",
            "esperado_metodo": None,
        },
        {
            "nombre": "T11: Texto vacío",
            "texto": "",
            "esperado_principal": None,
        },
        {
            "nombre": "T12: Derivación oncológica (capacidad latente)",
            "texto": "- Se sugiere derivación a oncología.",
            "esperado_principal": "derivacion_oncologica",
            "esperado_metodo": "regex",
        },
        {
            "nombre": "T13: Comparación con estudios previos",
            "texto": "- Se sugiere comparación con estudios anteriores.",
            "esperado_principal": "comparacion_estudios_previos",
            "esperado_metodo": "regex",
        },
    ]

    print("=" * 70)
    print("TESTS DE src/extractor_recomendacion.py")
    print("=" * 70)

    n_pasados = 0
    fallas: List[Tuple[str, str]] = []

    for caso in casos:
        resultado = clasificar_recomendacion(caso["texto"])

        checks: List[Tuple[bool, str]] = []

        checks.append((
            resultado["categoria_principal"] == caso["esperado_principal"],
            f"principal: esperado={caso['esperado_principal']}, "
            f"obtenido={resultado['categoria_principal']}",
        ))

        if "esperado_metodo" in caso:
            checks.append((
                resultado["metodo"] == caso["esperado_metodo"],
                f"metodo: esperado={caso['esperado_metodo']}, "
                f"obtenido={resultado['metodo']}",
            ))

        if "esperado_min_cats" in caso:
            checks.append((
                len(resultado["categorias_detectadas"]) >= caso["esperado_min_cats"],
                f"min_cats: esperado>={caso['esperado_min_cats']}, "
                f"obtenido={len(resultado['categorias_detectadas'])}",
            ))

        if caso.get("espera_typos"):
            checks.append((
                len(resultado["trazabilidad"]["typos_corregidos"]) > 0,
                "espera_typos: esperado≥1, obtenido=0",
            ))

        paso = all(ok for ok, _ in checks)
        estado = "PASA" if paso else "FALLA"

        if paso:
            n_pasados += 1
            print(f"  [{estado}] {caso['nombre']}")
        else:
            print(f"  [{estado}] {caso['nombre']}")
            for ok, msg in checks:
                if not ok:
                    print(f"         {msg}")

    # -------------------------------------------------------------------------
    # Tests de EXTRACCIÓN con tolerancia difusa (Vía 2.5 fuzzy)
    # -------------------------------------------------------------------------
    casos_fuzzy = [
        {
            "nombre": "F1: typo 'suguiere' se detecta como frase gatillo",
            "full_report": "Mamografia sin hallazgos.\nse suguiere biopsia Birads 4",
            "espera_encontrado": True,
            "espera_difuso": True,
        },
        {
            "nombre": "F2: transposición 'sugeire' se detecta",
            "full_report": "Nodulo sospechoso.\nse sugeire biopsia core.",
            "espera_encontrado": True,
            "espera_difuso": True,
        },
        {
            "nombre": "F3: texto sin gatillo NO se detecta (no sobre-dispara)",
            "full_report": "Mama densa heterogenea. Ganglio intramamario benigno.",
            "espera_encontrado": False,
            "espera_difuso": False,
        },
        {
            "nombre": "F4: ortografía correcta sigue funcionando (exacto)",
            "full_report": "Hallazgo sospechoso.\nse sugiere biopsia.",
            "espera_encontrado": True,
            "espera_difuso": False,  # match exacto, sin fuzzy
        },
    ]

    for caso in casos_fuzzy:
        res = extraer_texto_recomendacion(None, full_report=caso["full_report"])
        tiene_difuso = bool(res.get("correcciones_difusas"))
        ok = (
            res["encontrado"] == caso["espera_encontrado"]
            and tiene_difuso == caso["espera_difuso"]
        )
        estado = "PASA" if ok else "FALLA"
        print(f"  [{estado}] {caso['nombre']}")
        if not ok:
            print(f"         encontrado={res['encontrado']} (esp {caso['espera_encontrado']}), "
                  f"difuso={tiene_difuso} (esp {caso['espera_difuso']})")
        if ok:
            n_pasados += 1

    total_casos = len(casos) + len(casos_fuzzy)
    print(f"\nResumen: {n_pasados}/{total_casos} tests pasados")

    if n_pasados == total_casos:
        print("Estado: OK — el extractor está listo para uso en producción.")
    else:
        print("Estado: FALLA — revisar los casos que no pasaron.")


if __name__ == "__main__":
    _ejecutar_tests()
